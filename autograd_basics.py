import torch


# 一、基本求导：y = x²
print("\n一、基本求导")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

print("x:", x.item())
print("y:", y.item())
print("反向传播前 x.grad:", x.grad)

y.backward()

print("反向传播后 x.grad:", x.grad.item())

assert torch.allclose(y.detach(), torch.tensor(9.0))
assert torch.allclose(x.grad, torch.tensor(6.0))

# backward 计算梯度，没有更新 x。
assert torch.allclose(x.detach(), torch.tensor(3.0))


# 二、多项式求导：y = 2x² + 3x + 1
print("\n二、多项式求导")

# 重新创建 x，让这个实验从空梯度开始。
x = torch.tensor(3.0, requires_grad=True)
y = 2 * x ** 2 + 3 * x + 1

y.backward()

print("y:", y.item())
print("x.grad:", x.grad.item())

assert torch.allclose(y.detach(), torch.tensor(28.0))
assert torch.allclose(x.grad, torch.tensor(15.0))


# 三、梯度默认累积
print("\n三、梯度累积")

x = torch.tensor(3.0, requires_grad=True)

loss_1 = x ** 2
loss_1.backward()
print("第一次 backward 后:", x.grad.item())

assert torch.allclose(x.grad, torch.tensor(6.0))

# 重新前向计算，构建新的计算图。
loss_2 = x ** 2
loss_2.backward()
print("第二次 backward 后:", x.grad.item())

assert torch.allclose(x.grad, torch.tensor(12.0))

# 清空已有梯度，再进行一次求导。
x.grad = None

loss_3 = x ** 2
loss_3.backward()
print("清空后再次 backward:", x.grad.item())

assert torch.allclose(x.grad, torch.tensor(6.0))


# 四、SGD 单次更新
print("\n四、SGD 单次更新")

w = torch.tensor(3.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

optimizer.zero_grad()
loss = w ** 2
loss.backward()

print("更新前 w:", w.item())
print("更新前 loss:", loss.item())
print("梯度:", w.grad.item())

optimizer.step()

with torch.no_grad():
    new_loss = w ** 2

print("更新后 w:", w.item())
print("更新后 loss:", new_loss.item())
print("原来的 loss 仍然是:", loss.item())

assert torch.allclose(w.detach(), torch.tensor(2.4))
assert torch.allclose(new_loss, torch.tensor(5.76))
assert torch.allclose(loss.detach(), torch.tensor(9.0))


# 五、连续更新五次，每次清空梯度
print("\n五、连续更新五次")

w = torch.tensor(3.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

for i in range(5):
    optimizer.zero_grad()

    loss = w ** 2
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        new_loss = w ** 2

    print(
        f"第 {i + 1} 次更新："
        f"w={w.item():.6f}, loss={new_loss.item():.6f}"
    )

assert torch.allclose(w.detach(), torch.tensor(0.98304))


# 六、故意不清空梯度，观察更新结果
print("\n六、不清空梯度的实验")

w = torch.tensor(3.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

for i in range(2):
    # 本实验故意省略 optimizer.zero_grad()。
    loss = w ** 2
    loss.backward()

    print(
        f"第 {i + 1} 次更新前："
        f"loss={loss.item():.2f}, grad={w.grad.item():.2f}"
    )

    optimizer.step()
    print(f"更新后 w={w.item():.2f}")

# 第二次梯度为 6 + 4.8 = 10.8。
assert torch.allclose(w.grad, torch.tensor(10.8))
assert torch.allclose(w.detach(), torch.tensor(1.32))

print("\nAutograd 全部基本测试通过")