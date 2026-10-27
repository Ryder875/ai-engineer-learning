"""运行 python check.py，检查 Attention 的关键行为。"""
import torch

from model import attention, SelfAttention


def main():
    torch.manual_seed(42)
    q, k, v = [torch.randn(2, 5, 4) for _ in range(3)]
    output, weights = attention(q, k, v)
    assert output.shape == (2, 5, 4)
    assert weights.shape == (2, 5, 5)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))

    # 矩阵乘法等价于逐个 value 加权后求和。
    manual = (weights[0, 0].unsqueeze(-1) * v[0]).sum(dim=0)
    torch.testing.assert_close(manual, output[0, 0])
    uniform_output, uniform_weights = attention(
        torch.zeros_like(q), torch.zeros_like(k), v
    )
    torch.testing.assert_close(uniform_weights, torch.full_like(weights, 0.2))
    expected = v.mean(dim=1, keepdim=True).expand_as(output)
    torch.testing.assert_close(uniform_output, expected)
    print("通过：shape、权重和、加权汇总、均匀权重时的均值输出。")

    x = torch.randn(2, 5, 4)
    model = SelfAttention(d_model=4)
    model.eval()
    with torch.no_grad():
        output, weights = model(x, causal=True)
        future = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
        assert torch.count_nonzero(weights[:, future]).item() == 0
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))

        # 只改变最后两个 token，前三个位置的输出不应改变。
        changed_x = x.clone()
        changed_x[:, 3:] += 10
        changed_output, _ = model(changed_x, causal=True)
        torch.testing.assert_close(output[:, :3], changed_output[:, :3])

        # 对照：没有 mask 时，此输入变化确实能影响前面的输出。
        full_output, _ = model(x)
        changed_full_output, _ = model(changed_x)
        assert not torch.allclose(full_output[:, :3], changed_full_output[:, :3])
    print("通过：未来权重为零、权重和、未来输入隔离及无 mask 对照。")

    model.train()
    model.zero_grad()
    output, _ = model(x, causal=True)
    output.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    print("通过：加入 mask 后，各投影参数的梯度存在且有限。")


if __name__ == "__main__":
    main()
