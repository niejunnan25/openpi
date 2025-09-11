import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    # 连续时间/位置，用于告诉模型每个时间点在序列里的顺序
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)

    # 生成一个从 0 到 1 的均匀分布向量，长度为 dimension // 2，表示频率的归一化比例
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)

    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """

    """复制自 big_vision。

    Token 可以关注（attend）那些累计的 mask_ar 值小于或等于它们自己的有效输入 token。
    这样，`mask_ar` int[B, N] 就可以被用来设置多种注意力机制，例如：

    [[1 1 1 1 1 1]]：纯粹的因果注意力（causal attention）。

    [[0 0 0 1 1 1]]：前缀语言模型（prefix-lm）的注意力。
        前 3 个 token 可以互相注意，而后 3 个 token 采用因果注意力。
        第一个条目也可以是 1，而不会改变行为。

    [[1 0 1 0 1 0 0 1 0 0]]：分为 4 个区块的因果注意力。
        同一区块的 token 可以彼此关注，同时还能关注所有先前的区块。

    参数说明：
    input_mask: bool[B, N]，为 True 表示是输入的一部分，为 False 表示是 padding。
    mask_ar: int32[B, N]，在前序 token **不能依赖它**时为 1，  
            在与前一个 token **共享相同注意力掩码**时为 0。
    """


    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # 对 att_masks 在序列维度 (dim=1) 上做累积和。
    # 这样，每个位置的值表示从序列开头到当前位置，mask 的累计值。
    # [0, 0, 0, 1, 1, 1] -> [0, 0, 0, 1, 2, 3]
    cumsum = torch.cumsum(att_masks, dim=1)

    # 构造二维注意力掩码：
    # 对于位置 i（query）和位置 j（key），如果 cumsum[j] <= cumsum[i]，
    # 表示 query 可以“看到” key。
    # 这里通过广播实现：cumsum[:, None, :] 是 (B, 1, N)，cumsum[:, :, None] 是 (B, N, 1)，
    # 得到结果形状为 (B, N, N)。
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

    # 构造 padding 掩码：
    # pad_masks 是 (B, N)，True 表示有效 token，False 表示 padding。
    # 通过扩展为 (B, 1, N) 和 (B, N, 1)，相乘得到 (B, N, N)，
    # 只有 query 和 key 都是有效 token 时才为 True。
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]

    # 最终的注意力掩码：
    # 既要满足因果/结构化的可见性 (att_2d_masks)，
    # 又要保证不访问 padding 位置 (pad_2d_masks)。
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        # 若是 pi05 的话，则直接用 time_mlp，否则用 action_time_mlp
        # time_mlp 应该是指时间 t 的嵌入，对 t 做一次非线性变换，增强表达能力
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # 如果是 pi0 的话，则把 state 映射到特征空间中
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        # 最大化性能优化
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

####################################################################################################################################

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

####################################################################################################################################

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """
        构造 4D 注意力掩码，从原始的 3D 注意力掩码扩展到 4D
        """
        """
        tensor([[[[0.0, 0.0, 0.0, -2.38e38, -2.38e38, -2.38e38],
          [0.0, 0.0, 0.0, -2.38e38, -2.38e38, -2.38e38],
          [0.0, 0.0, 0.0, -2.38e38, -2.38e38, -2.38e38],
          [0.0, 0.0, 0.0, 0.0, -2.38e38, -2.38e38],
          [0.0, 0.0, 0.0, 0.0, 0.0, -2.38e38],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]])
        """
        # torch.where(condition, x, y) 函数
        # 对 condition 中为 True 的位置，结果取 x
        # 对 condition 中为 False 的位置，结果取 y
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

####################################################################################################################################

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)

        # observation 是 SimpleProcessedObservation 类，需要转换为 dict
        # 这个 SimpleProcessedObservation 类是预处理后的 observation，包含 images, image_masks, state, tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask
        # 这里没有用到 token_ar_mask, token_loss_mask
        
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        """
        采样标准正态分布噪声。

        参数：
            shape: tuple，噪声张量的形状
            device: torch 设备，例如 'cpu' 或 'cuda'

        返回：
            一个 float32 类型的张量，服从均值 0、标准差 1 的正态分布
        """
        # TODO：shape 的形状是什么呢？
        return torch.normal(
            mean=0.0,         # 均值 μ = 0
            std=1.0,          # 标准差 σ = 1
            size=shape,       # 张量形状
            dtype=torch.float32,
            device=device,
        )


    def sample_time(self, bsize, device):
        """
        采样时间步（time step），通常用于扩散模型或噪声调度。

        参数：
            bsize: int，批大小（batch size）
            device: torch 设备，例如 'cpu' 或 'cuda'

        返回：
            一个 float32 类型的张量，表示每个样本的时间步，
            范围大约在 [0.001, 1.0)
        """
        # # TODO：bsize 的形状是什么呢？
        # 先从 Beta 分布采样时间因子
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        
        # 将时间因子缩放到 [0.001, 1.0) 区间
        time = time_beta * 0.999 + 0.001
        
        # 转为 float32 并移动到指定设备
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对图像和语言 token 进行 embedding，为 PaliGemma Transformer 做准备。
        
        参数：
            images: List[Tensor]，每个元素为一张图像的 tensor
            img_masks: List[Tensor]，每个图像对应的 padding mask（1=有效，0=padding）
            lang_tokens: Tensor，语言 token 序列
            lang_masks: Tensor，语言 token 的 padding mask（1=有效，0=padding）
        
        返回：
            tuple:
                embs: Tensor，图像和语言 token 的 embedding 拼接结果，形状 (B, N, D)
                pad_masks: Tensor，对应 token 的 padding mask，形状 (B, N)
                att_masks: Tensor，token 的 attention mask，形状 (B, N)
        """

        # 存储所有 token 的 embedding
        embs = []

        # 存储 padding mask
        pad_masks = []

        # 存储 attention mask
        att_masks = []

        # -----------------------
        # 处理图像
        # -----------------------
        for img, img_mask in zip(images, img_masks, strict=True):

            """
            
            """
            # TODO：原始输入的 img_mask 是 (B, 1) 的形状吗？
            # TODO：原始输入的 img 是什么形状，for 取出来的 img 又是什么形状
            # 定义图像 embedding 函数，支持梯度检查点
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            # 对图像进行 embedding，并应用梯度检查点（节省显存）
            img_emb = self._apply_checkpoint(image_embed_func, img)

            # 获取 batch size 和图像 token 数量
            # TODO：img_emb 的形状是什么呢？
            bsize, num_img_embs = img_emb.shape[:2]

            # 保存图像 embedding
            embs.append(img_emb)

            # 扩展图像 mask 到每个 token
            # TODO：原始的 img_mask 长什么样子呢？
            # TODO：扩展后又是什么样的呢？
            # .expand = 广播复制维度，不复制内存
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # 创建图像 token 的 attention mask
            # 图像 token 之间可以互相 attention
            att_masks += [0] * num_img_embs  # 0 表示可见

        # -----------------------
        # 处理语言 token
        # -----------------------
        def lang_embed_func(lang_tokens):
            # 语言 token embedding
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            # 对 embedding 进行缩放（乘以 sqrt(dim)）
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        # 对语言 token embedding 也应用梯度检查点
        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        # 保存语言 embedding 和对应的 padding mask
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像和语言 token 之间采用全 attention
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs  # 0 表示可见

        # -----------------------
        # 拼接所有 embedding 和 mask
        # -----------------------

        embs = torch.cat(embs, dim=1)       # 拼接图像和语言 embedding，维度 (B, N, D)
        pad_masks = torch.cat(pad_masks, dim=1)   # 拼接 padding mask，维度 (B, N)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)  # 转为布尔型

        # 获取 batch size
        bsize = pad_masks.shape[0]
        # 扩展 attention mask 到 batch 维度
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # 返回 embedding、padding mask 和 attention mask
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """将状态（state）、带噪动作（noisy_actions）和时间步（timestep）嵌入，用于 Expert Gemma 模型处理。"""
        
        # 用于存放 embedding 的列表
        embs = []
        # pad mask，用于指示哪些位置是真实 token（1）还是填充（0）
        pad_masks = []
        # attention mask，用于控制不同 token 之间是否可相互注意
        att_masks = []

        # 如果不是 pi05
        if not self.pi05:
            # 确保 state 的数据类型为 float32
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # 定义 state embedding 的前向函数
            def state_proj_func(state):
                return self.state_proj(state)

            # 使用 checkpoint 技术计算 state embedding（节省显存）
            state_emb = self._apply_checkpoint(state_proj_func, state)

            # 给 embedding 添加时间维度（1），并加入列表
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]  # batch size
            device = state_emb.device

            # 创建 state mask（全 1，表示这些位置有效）
            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # 设置 attention mask，使图像和语言输入不能注意到 state 或动作
            att_masks += [1]  # 这里 1 表示允许关注 state（具体语义依赖实现）

        # ----------------- 时间步 embedding -----------------
        # 使用正弦余弦位置编码生成 timestep embedding，数值在 [0, 1] 范围内敏感
        time_emb = create_sinusoidal_pos_embedding(
            timestep,                      # 时间步
            self.action_in_proj.out_features,  # embedding 维度
            min_period=4e-3,               # 最小周期
            max_period=4.0,                # 最大周期
            device=timestep.device         # device
        )
        # 保持 dtype 与 timestep 一致
        time_emb = time_emb.type(dtype=timestep.dtype)

        # ----------------- 动作 embedding -----------------
        # 定义动作 embedding 的前向函数
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        # 使用 checkpoint 技术计算动作 embedding
        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # 如果不是 pi05 模式
        if not self.pi05:
            # 将时间 embedding 扩展到动作 embedding 的 shape
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            # 将动作 embedding 和时间 embedding 拼接
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # 定义 MLP 前向函数，融合动作和时间信息
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)  # 输入 MLP
                x = F.silu(x)                                 # 激活函数 swish (SiLU)
                return self.action_time_mlp_out(x)           # 输出 MLP

            # 使用 checkpoint 技术计算最终动作-时间 embedding
            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None  # 不使用 adaRMS 条件
        else:
            # pi05 模式下，单独对时间 embedding 做 MLP（用于 adaRMS 条件）
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            # 计算时间 embedding
            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb  # 动作 embedding 不拼接时间
            adarms_cond = time_emb        # adaRMS 条件使用时间 embedding

        # 将动作-时间 embedding 添加到总 embedding 列表
        embs.append(action_time_emb)

        # ----------------- 创建动作 mask -----------------
        bsize, action_time_dim = action_time_emb.shape[:2]
        # 动作位置 mask，全 1 表示这些位置有效
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # 设置 attention mask，使图像、语言、state 输入不能注意到动作 token
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        # ----------------- 合并所有 embedding 和 mask -----------------
        embs = torch.cat(embs, dim=1)         # 合并 embedding
        pad_masks = torch.cat(pad_masks, dim=1)  # 合并 pad mask
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)  # 转成 tensor
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))  # 扩展到 batch size

        # 返回最终结果
        return embs, pad_masks, att_masks, adarms_cond


    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""

        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """对给定时间步的噪声 `x_t` 应用一次去噪（denoising）步骤。"""
        
        # ----------------- 1. 生成 suffix embedding -----------------
        # 将 state、当前噪声动作 x_t、时间步嵌入成 suffix embedding
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # 获取 suffix embedding 的长度、batch size 以及 prefix 的长度
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # ----------------- 2. 构造 2D attention mask -----------------
        # 将 prefix mask 扩展为 2D mask，用于与 suffix 做注意力连接
        # shape: [batch_size, suffix_len, prefix_len]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # 生成 suffix 的 2D 注意力 mask
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # 将 prefix 和 suffix 的注意力 mask 拼接成完整 mask
        # 最终 shape: [batch_size, suffix_len, prefix_len + suffix_len]
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # ----------------- 3. 构造位置 id -----------------
        # 计算 prefix 的偏移量（prefix 有效 token 的数量）
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        # suffix 的位置 id = prefix 偏移 + suffix 内部累积索引 - 1
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        # ----------------- 4. 准备 4D 注意力 mask -----------------
        # 将 full_att_2d_masks 转换为 4D mask 以供 Transformer 使用
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        # 强制使用 eager attention 实现
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # ----------------- 5. 调用模型前向传播 -----------------
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,  # 注意力 mask
            position_ids=position_ids,             # token 位置 id
            past_key_values=past_key_values,       # 可选的缓存 key/value，用于加速推理
            inputs_embeds=[None, suffix_embs],     # 输入 embedding，prefix 为空，suffix 有 embedding
            use_cache=False,                        # 不使用缓存
            adarms_cond=[None, adarms_cond],       # adaRMS 条件信息
        )

        # ----------------- 6. 提取 suffix 输出 -----------------
        suffix_out = outputs_embeds[1]  # suffix 对应的 embedding
        # 只保留最近 action_horizon 个 token 的输出
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # 转为 float32
        suffix_out = suffix_out.to(dtype=torch.float32)

        # ----------------- 7. 投影为动作输出 -----------------
        # 通过线性层投影为最终动作输出
        return self.action_out_proj(suffix_out)

