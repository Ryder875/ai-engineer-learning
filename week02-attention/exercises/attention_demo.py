"""从本周根目录运行：python -m exercises.attention_demo。"""
import torch
from model import SelfAttention


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 5, 4)
    model = SelfAttention(d_model=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    old_weight = model.q_proj.weight.detach().clone()

    optimizer.zero_grad()
    output, _ = model(x)
    # 只用于演示反向传播，不是语言建模任务。
    loss = output.square().mean()
    print(f"loss: {loss.item():.4f}")
    print("反向传播前 Q 参数的梯度:", model.q_proj.weight.grad)
    loss.backward()
    for name, parameter in model.named_parameters():
        print(f"{name} 梯度范数: {parameter.grad.norm().item():.4f}")
    optimizer.step()

    change = (model.q_proj.weight.detach() - old_weight).abs().max()
    print(f"Q 参数的最大变化量: {change.item():.4f}")


if __name__ == "__main__":
    main()
