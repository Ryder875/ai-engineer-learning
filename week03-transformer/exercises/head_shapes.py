import torch

B, T, D, H = 2, 5, 12, 3
d_head = D // H

# 用连续整数，方便检查元素是否被放到了正确的位置
x = torch.arange(B * T * D).reshape(B, T, D)

# [B, T, D] → [B, T, H, d_head] → [B, H, T, d_head]
heads = x.reshape(B, T, H, d_head).transpose(1, 2)

# [B, H, T, d_head] → [B, T, H, d_head] → [B, T, D]
restored = heads.transpose(1, 2).reshape(B, T, D)

assert heads.shape == (B, H, T, d_head)
assert torch.equal(x, restored)

# 检查同一个 token 的特征如何分配给三个头
print("原始 token：", x[0, 0])
print("三个头中的对应特征：", heads[0, :, 0, :])
