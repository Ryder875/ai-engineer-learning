"""复习：MSE、Tensor、广播、梯度、模型与二分类；内置基本检查。"""
import math
import sys

import torch
from torch import nn


def mean_squared_error(predictions, targets):
    if len(predictions) != len(targets) or len(targets) == 0:
        raise ValueError("输入必须等长且非空")
    total = 0.0
    for prediction, target in zip(predictions, targets):
        total += (prediction - target) ** 2
    return total / len(targets)


# 1. Python 循环与 Tensor 向量化计算。
predictions, targets = [1.0, 5.0, 4.0], [1.0, 3.0, 5.0]
assert math.isclose(mean_squared_error(predictions, targets), 5 / 3)
assert mean_squared_error(targets, targets) == 0.0
mse = ((torch.tensor(predictions) - torch.tensor(targets)) ** 2).mean()
assert mse.ndim == 0 and torch.allclose(mse, torch.tensor(5 / 3))
for value in [torch.tensor(5.0), torch.tensor([5.0]), torch.tensor([[5.0]])]:
    print(f"shape={value.shape}, ndim={value.ndim}, numel={value.numel()}")

# 2. 聚合与广播：保留行轴，避免均值错误地按列对齐。
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert torch.allclose(x.mean(dim=0), torch.tensor([2.5, 3.5, 4.5]))
assert torch.allclose(x.mean(dim=1), torch.tensor([2.0, 5.0]))
centered = x - x.mean(dim=1, keepdim=True)
assert centered.shape == x.shape
assert torch.allclose(centered.mean(dim=1), torch.zeros(2), atol=1e-6)
square = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
wrong = square - square.mean(dim=1)
assert not torch.allclose(wrong.mean(dim=1), torch.zeros(2))

# 3. [B,T,D] @ [D,H] -> [B,T,H]，各 token 共享权重。
torch.manual_seed(42)
x, w = torch.randn(2, 3, 4), torch.randn(4, 5)
y = x @ w
assert y.shape == (2, 3, 5)
assert torch.allclose(y[0, 0], x[0, 0] @ w, atol=1e-6)

# 4. backward 累积梯度，不更新参数；每次重新前向计算。
x = torch.tensor(3.0, requires_grad=True)
(2 * x ** 2 + 3 * x + 1).backward()
assert x.grad.item() == 15.0 and x.item() == 3.0
(x ** 2).backward()
assert x.grad.item() == 21.0
x.grad = None
(x ** 2).backward()
assert x.grad.item() == 6.0


# 5. Module 注册参数；链式法则与 SGD 更新。
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


model = LinearModel()
assert set(dict(model.named_parameters())) == {"w", "b"}
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
loss = (model(torch.tensor(2.0)) - 6.0) ** 2
loss.backward()
assert model.w.grad.item() == -16.0 and model.b.grad.item() == -8.0
optimizer.step()
assert torch.allclose(model.w.detach(), torch.tensor(2.6))
assert torch.allclose(model.b.detach(), torch.tensor(0.8))
with torch.no_grad():
    assert torch.allclose(model(torch.tensor(2.0)), torch.tensor(6.0))
layer = nn.Linear(3, 2)
assert layer.weight.shape == (2, 3)
assert sum(p.numel() for p in layer.parameters()) == 8
assert layer(torch.zeros(4, 3)).shape == (4, 2)
assert torch.equal(nn.ReLU()(torch.tensor([-2., 0., 3.])), torch.tensor([0., 0., 3.]))

# 6. BCE 接收原始 logits；浮点标签 shape 与输出一致。
logits = torch.tensor([[-2.0], [0.0], [2.0]], requires_grad=True)
labels = torch.tensor([False, True, True]).float().unsqueeze(1)
assert logits.shape == labels.shape == (3, 1)
assert torch.equal((logits >= 0).float(), labels)
loss = nn.BCEWithLogitsLoss()(logits, labels)
loss.backward()
# 默认取均值，因此各样本梯度除以样本数。
assert torch.allclose(logits.grad, (torch.sigmoid(logits.detach()) - labels) / 3)
print("CPU 基础检查全部通过")

# 7. 环境与真实 GPU 前向/反向检查。
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    x = torch.randn(32, 16, device="cuda", requires_grad=True)
    w = torch.randn(16, 8, device="cuda", requires_grad=True)
    y = x @ w
    y.square().mean().backward()
    torch.cuda.synchronize()
    assert y.shape == (32, 8)
    assert x.grad is not None and w.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    print("GPU 检查通过:", torch.cuda.get_device_name(0))
else:
    print("无可用 GPU，跳过 GPU 检查")
