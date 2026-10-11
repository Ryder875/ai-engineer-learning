import torch


# 一、用 Tensor 计算 MSE
predictions = torch.tensor([1.0, 5.0, 4.0])
targets = torch.tensor([1.0, 3.0, 5.0])

errors = predictions - targets
squared_errors = errors ** 2
mse = squared_errors.mean()

print("预测值 shape:", predictions.shape)
print("误差:", errors)
print("误差平方:", squared_errors)
print("MSE:", mse)
print("MSE shape:", mse.shape)
print("MSE 数值:", mse.item())

assert torch.allclose(
    errors,
    torch.tensor([0.0, 2.0, -1.0]),
)
assert torch.allclose(mse, torch.tensor(5.0 / 3.0))
assert mse.ndim == 0


# 二、区分 shape、维度数和元素数
a = torch.tensor(5.0)
b = torch.tensor([5.0])
c = torch.tensor([[5.0]])

for value in [a, b, c]:
    print(
        "值:", value,
        "shape:", value.shape,
        "维度数:", value.ndim,
        "元素数:", value.numel(),
    )

assert a.shape == ()
assert b.shape == (1,)
assert c.shape == (1, 1)
assert a.numel() == b.numel() == c.numel() == 1


# 三、沿不同轴求平均
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

overall_mean = matrix.mean()
column_means = matrix.mean(dim=0)
row_means = matrix.mean(dim=1)

print("矩阵 shape:", matrix.shape)
print("所有元素的平均值:", overall_mean)
print("每列平均值:", column_means)
print("每行平均值:", row_means)

assert torch.allclose(overall_mean, torch.tensor(3.5))
assert torch.allclose(
    column_means,
    torch.tensor([2.5, 3.5, 4.5]),
)
assert torch.allclose(
    row_means,
    torch.tensor([2.0, 5.0]),
)

# keepdim=True 保留被聚合的轴，其长度变为 1。
print("保留维度的列均值 shape:",
      matrix.mean(dim=0, keepdim=True).shape)
print("保留维度的行均值 shape:",
      matrix.mean(dim=1, keepdim=True).shape)

assert matrix.mean(dim=0, keepdim=True).shape == (1, 3)
assert matrix.mean(dim=1, keepdim=True).shape == (2, 1)


# 四、每个 token 使用同一个权重矩阵进行线性变换
torch.manual_seed(42)

B, T, D, H = 2, 3, 4, 5

x = torch.randn(B, T, D)
w = torch.randn(D, H)
y = x @ w

print("输入 shape:", x.shape)
print("权重 shape:", w.shape)
print("输出 shape:", y.shape)

assert y.shape == (B, T, H)

# 单独计算第一个样本的第一个 token。
single_token_output = x[0, 0] @ w

assert torch.allclose(
    y[0, 0],
    single_token_output,
    atol=1e-6,
    rtol=1e-5,
)

print("Tensor 基本测试通过")