import torch


# 一、只有权重的模型：prediction = w * x
print("\n一、只有权重的线性模型")

x = torch.tensor(2.0)
target = torch.tensor(6.0)
w = torch.tensor(1.0, requires_grad=True)

optimizer = torch.optim.SGD([w], lr=0.1)
optimizer.zero_grad()

prediction = w * x
loss = (prediction - target) ** 2
loss.backward()

print("初始预测:", prediction.item())
print("初始损失:", loss.item())
print("w 的梯度:", w.grad.item())

# dL/dprediction = 2 * (2 - 6) = -8
# dprediction/dw = x = 2
# dL/dw = -8 * 2 = -16
assert torch.allclose(w.grad, torch.tensor(-16.0))

optimizer.step()

with torch.no_grad():
    new_prediction = w * x
    new_loss = (new_prediction - target) ** 2

print("更新后 w:", w.item())
print("新预测:", new_prediction.item())
print("新损失:", new_loss.item())

assert torch.allclose(w.detach(), torch.tensor(2.6))
assert torch.allclose(new_prediction, torch.tensor(5.2))
assert torch.allclose(new_loss, torch.tensor(0.64))


# 二、加入偏置：prediction = w * x + b
print("\n二、包含权重和偏置的线性模型")

x = torch.tensor(2.0)
target = torch.tensor(6.0)

# 重新初始化参数，让本实验独立开始。
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.1)
optimizer.zero_grad()

prediction = w * x + b
loss = (prediction - target) ** 2

print("更新前预测:", prediction.item())
print("更新前损失:", loss.item())

loss.backward()

print("w 的梯度:", w.grad.item())
print("b 的梯度:", b.grad.item())

# dL/dw = 2 * (prediction - target) * x = -16
# dL/db = 2 * (prediction - target) * 1 = -8
assert torch.allclose(w.grad, torch.tensor(-16.0))
assert torch.allclose(b.grad, torch.tensor(-8.0))

optimizer.step()

with torch.no_grad():
    new_prediction = w * x + b
    new_loss = (new_prediction - target) ** 2

print("更新后 w:", w.item())
print("更新后 b:", b.item())
print("更新后预测:", new_prediction.item())
print("更新后损失:", new_loss.item())

assert torch.allclose(w.detach(), torch.tensor(2.6))
assert torch.allclose(b.detach(), torch.tensor(0.8))
assert torch.allclose(new_prediction, target, atol=1e-6)
assert torch.allclose(new_loss, torch.tensor(0.0), atol=1e-6)

print("\n线性模型全部基本测试通过")