import logging

from jax import device_get
import numpy as np

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        img_for = 0
        # embed images
        for name in obs.images:
            print(f"当前循环第 {img_for} 次")
            print(f"obs.images 是：", name)
            print(f"obs.images 是：", jnp.array(obs.images[name]))
            print(f"obs.images[name].shape 是：",obs.images[name].shape)

            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            print(f"image_tokens.shape 是：",image_tokens.shape)

            tokens.append(image_tokens)
            print(f"当前的 tokens 长度是:", len(tokens))

            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            print(f"obs.image_masks[name] 对应的 mask 是：", einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                ))
            print(f"当前的 input_mask 长度是:", len(input_mask))

            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
            print(f"image_tokens.shape[1] 是：", image_tokens.shape[1])
            print(f"ar_mask 的长度是", len(ar_mask))
            img_for += 1
            print("============================================================================================================")

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            # obs.tokenized_prompt, 这个地方已经是 tokens 的 ID
            # 这个地方给了 200 个 int32
            print(f"obs.tokenized_prompt 是：", obs.tokenized_prompt)
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            
            # 每个 token ID 都被映射成一个 64 维向量
            # tokenized_inputs.shape 是： (2, 200, 64)
            print(f"tokenized_inputs.shape 是：", tokenized_inputs.shape)

            # 
            tokens.append(tokenized_inputs)
            print(f"当前的 tokens 长度是:", len(tokens))

            input_mask.append(obs.tokenized_prompt_mask)
            print(f"obs.tokenized_prompt_mask 是：", obs.tokenized_prompt_mask)

            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            print(f"ar_mask 的长度是", len(ar_mask))

        print("============================================================================================================")

        # (2, 256, 64) + (2, 256, 64) + (2, 256, 64) + (2, 200, 64) = (2, 968, 64)
        tokens = jnp.concatenate(tokens, axis=1)
        print(f"tokens.shape", tokens.shape)
        
        print("============================================================================================================")

        # (2, 256) + (2, 256) + (2, 256) + (2, 200) = (2, 968)
        input_mask = jnp.concatenate(input_mask, axis=1)
        print(f"input_mask.shape", input_mask.shape)

        print("============================================================================================================")

        ar_mask = jnp.array(ar_mask)
        print(f"ar_mask.shape", ar_mask.shape)

        print("============================================================================================================")

        return tokens, input_mask, ar_mask

    ##################################################################################################################################
    #
    #
    #
    #
    ##################################################################################################################################
    
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        print()
        print("==================================================suffix====================================================")
        print("==================================================suffix====================================================")
        print("==================================================suffix====================================================")
        print()

        # TODO: 看看 pi0 的 ar_mask 是不是前两个 True, 后面的全是 False
        if not self.pi05:

            # add a single state token
            # 添加一个 state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]
        
        print(f"noisy_actions 是：", noisy_actions)
        print(f"noisy_actions.shape 是：", noisy_actions.shape)

        # action_expert_config = _gemma.get_config(config.action_expert_variant)
        # action_expert_config.width = 64
        action_tokens = self.action_in_proj(noisy_actions)
        print(f"action_tokens.shape 是：", action_tokens.shape)
        print("============================================================================================================")

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 嵌入时间步长，使用正弦-余弦位置编码，敏感度在[0, 1]范围内
        print(f"timestep 是：", timestep)
        print(f"timestep.shape 是：", timestep.shape)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        print(f"time_emb.shape 是：", time_emb.shape)

        print("============================================================================================================")

        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            print(f"time_emb.shape 是：", time_emb.shape)

            time_emb = nnx.swish(time_emb)
            print(f"time_emb.shape 是：", time_emb.shape)

            time_emb = self.time_mlp_out(time_emb)
            print(f"time_emb.shape 是：", time_emb.shape)

            time_emb = nnx.swish(time_emb)
            print(f"time_emb.shape 是：", time_emb.shape)

            #################################################################################################################

            action_expert_tokens = action_tokens
            print(f"action_expert_tokens 是 action_tokens")
            print(f"action_expert_tokens.shape 是：", action_expert_tokens.shape)

            # time_emb 作为 adarms_cond 使用
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        
        print("============================================================================================================")

        #################################################################################################################
        # 这里添加了一个 action_expert_tokens
        # 如果是 pi05, 则 action_expert_tokens 是 action_tokens, 否则 action_expert_tokens 是 action_time_tokens
        # 这里的 action_tokens 是动作投影后的结果
        # 这里的 action_time_tokens 是动作和时间混合后的结果

        tokens.append(action_expert_tokens)

        print(f"tokens 的长度是：", len(tokens))

        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # 生成一个 (2, 50) 形状的 mask
        print(f"action_expert_tokens.shape[:2] 是", action_expert_tokens.shape[:2])
        print(f"input_mask 的长度是：", len(input_mask))

        print("============================================================================================================")

        # image/language/state inputs do not attend to action tokens
        # 图像/语言/状态输入不关注动作标记
        # 如果是 pi05, [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        print(f"ar_mask 的长度是：", len(ar_mask))
        print(f"ar_mask 是：", ar_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        print(f"tokens.shape 是：", tokens.shape)

        input_mask = jnp.concatenate(input_mask, axis=1)
        print(f"input_mask.shape 是：", input_mask.shape)

        ar_mask = jnp.array(ar_mask)
        print(f"ar_mask.shape 是：", ar_mask.shape)

        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        
        # 生成随机数
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        # 预处理观测值
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 获取动作的形状, batch_shape 是： (2,)
        batch_shape = actions.shape[:-2]
        print(f"batch_shape 是：", batch_shape)

        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        exit()
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
