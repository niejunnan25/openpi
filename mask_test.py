import torch

att_masks = torch.tensor([[0, 0, 0, 1, 1, 1]])
cumsum = torch.cumsum(att_masks, dim=1)

c1 = cumsum[:, None, :]    # (B, 1, N)
c2 = cumsum[:, :, None]    # (B, N, 1)
att_2d_masks = c1 <= c2    # (B, N, N)

print("cumsum:", cumsum)
print("c1 shape, c1[0,0,:]:", c1.shape, c1[0,0,:])
print(c1[0])
print("c2 shape, c2[0,:,0]:", c2.shape, c2[0,:,0])
print(c2[0])
print("att_2d_masks (as int):\n", att_2d_masks.int()[0])
print(att_2d_masks[0])
"""
tensor([[ True,  True,  True, False, False, False],
        [ True,  True,  True, False, False, False],
        [ True,  True,  True, False, False, False],
        [ True,  True,  True,  True, False, False],
        [ True,  True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True,  True]])
"""

pad_masks = torch.tensor([[1, 1, 1, 1, 0, 0]])
p1 = pad_masks[:, None, :]
p2 = pad_masks[:, :, None]

pad_masks = p1 * p2

print("====================================================")

print(pad_masks)
print(p1)
print(p2)
print(pad_masks.int()[0])
print(pad_masks[0])

"""
tensor([[1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
"""

print(att_2d_masks & pad_masks)

"""
tensor([[[1, 1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]])
"""
# 若是因果注意力的话
"""
[[1, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0],
 [1, 1, 1, 1, 0, 0],
 [0, 0, 0, 0, 0, 0],  # padding
 [0, 0, 0, 0, 0, 0]]  # padding
"""
