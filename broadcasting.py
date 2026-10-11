import torch


# 一、正确的行中心化
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

row_mean = x.mean(dim=1, keepdim=True)
centered = x - row_mean

print("原始矩阵:", x)
print("行均值:", row_mean)
print("行均值 shape:", row_mean.shape)
print("行中心化结果:", centered)
print("行中心化结果 shape:", centered.shape)

assert row_mean.shape == (2, 1)
assert centered.shape == x.shape

assert torch.allclose(
    centered,
    torch.tensor([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]),
)

# 行中心化后，每行的均值应接近零。
assert torch.allclose(
    centered.mean(dim=1),
    torch.zeros(2),
    atol=1e-6,
)


# 二、[2, 3] 与 [3] 广播
column_values = torch.tensor([10.0, 20.0, 30.0])
shifted = x - column_values

print("按列对应相减的结果:", shifted)

# 同一组三个值用于每一行。
assert torch.allclose(
    shifted,
    torch.tensor([
        [-9.0, -18.0, -27.0],
        [-6.0, -15.0, -24.0],
    ]),
)


# 三、代码能运行，但计算含义可能错误
square = torch.tensor([
    [1.0, 3.0],
    [5.0, 7.0],
])

# shape [2] 按 [1, 2] 对齐，均值被用于对应列。
wrong_mean = square.mean(dim=1)
wrong_centered = square - wrong_mean

# shape [2, 1] 让每行减去自己的均值。
correct_mean = square.mean(dim=1, keepdim=True)
correct_centered = square - correct_mean

print("未保留维度的均值:", wrong_mean)
print("错误的行中心化结果:", wrong_centered)
print("正确的行中心化结果:", correct_centered)

assert torch.allclose(
    wrong_centered,
    torch.tensor([
        [-1.0, -3.0],
        [3.0, 1.0],
    ]),
)

assert torch.allclose(
    correct_centered,
    torch.tensor([
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]),
)

# 验证业务含义，而不只是验证代码能运行。
assert not torch.allclose(
    wrong_centered.mean(dim=1),
    torch.zeros(2),
)

assert torch.allclose(
    correct_centered.mean(dim=1),
    torch.zeros(2),
    atol=1e-6,
)

print("Broadcasting 基本测试通过")