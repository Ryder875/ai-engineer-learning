import sys
import torch

print("Python 路径:", sys.executable)
print("PyTorch 版本:", torch.__version__)
print("PyTorch CUDA 版本:", torch.version.cuda)
print("GPU 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))

    x = torch.randn(32, 16, device="cuda", requires_grad=True)
    w = torch.randn(16, 8, device="cuda", requires_grad=True)

    y = x @ w
    loss = y.square().mean()
    loss.backward()
    torch.cuda.synchronize()

    print("输出 shape:", y.shape)
    print("x 的梯度 shape:", x.grad.shape)
    print("w 的梯度 shape:", w.grad.shape)
    print("GPU 运算和反向传播验证通过")
else:
    print("GPU 暂未可用，需要排查环境")