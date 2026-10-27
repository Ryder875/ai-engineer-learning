"""运行：python check.py；无需 pytest。"""

import torch

from model import MultiHeadAttention, TransformerBlock, MiniLanguageModel


def check_module(module, label):
    x = torch.randn(2, 5, 12)
    output, weights = module(x)
    assert output.shape == x.shape
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert torch.all(weights.triu(diagonal=1) == 0)

    changed = x.clone()
    changed[:, 3:, :] += 100
    with torch.no_grad():
        changed_output, _ = module(changed)
    torch.testing.assert_close(output[:, :3], changed_output[:, :3])

    module.zero_grad(set_to_none=True)
    output.square().mean().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        # key bias 的梯度理论上为零；其他参数应参与此次反向传播。
        if not name.endswith('k_proj.bias'):
            assert param.grad.norm() > 0, name

    # backward 后还必须 step 才会更新参数。
    before = {name: p.detach().clone() for name, p in module.named_parameters()}
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in module.named_parameters())
    print(f'{label}: shape / weights / causality / gradients / update PASS')


def check_reference():
    # 使用 PyTorch 官方模块作为数值参照，检查拆头、合头与输出投影。
    ours = MultiHeadAttention(12, 3)
    reference = torch.nn.MultiheadAttention(12, 3, dropout=0.0, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([
            ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight,
        ]))
        reference.in_proj_bias.copy_(torch.cat([
            ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias,
        ]))
        reference.out_proj.load_state_dict(ours.out_proj.state_dict())
    x = torch.randn(2, 5, 12)
    mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
    actual, weights = ours(x)
    expected, expected_weights = reference(
        x, x, x, attn_mask=mask, average_attn_weights=False,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights, expected_weights)
    print('PyTorch reference: output / weights PASS')


def residual_experiment():
    block = TransformerBlock(12, 3)
    x = torch.randn(2, 5, 12)
    with torch.no_grad():
        # 令两条分支的最终投影为零，整个 block 应成为恒等映射。
        block.attention.out_proj.weight.zero_()
        block.attention.out_proj.bias.zero_()
        block.ffn[-1].weight.zero_()
        block.ffn[-1].bias.zero_()
        output, _ = block(x)
    error = (output - x).abs().max().item()
    assert torch.equal(output, x)
    print(f'Zero branches: max |output - input| = {error:.1f} PASS')


def check_language_model():
    tokens = torch.randint(10, (2, 5))
    for self_only in (False, True):
        model = MiniLanguageModel(10, 12, 3, 5, self_only=self_only)
        logits = model(tokens)
        assert logits.shape == (2, 5, 10)
        changed = tokens.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 10
        torch.testing.assert_close(logits[:, :3], model(changed)[:, :3])
        if self_only:
            changed = tokens.clone()
            changed[:, 0] = (changed[:, 0] + 1) % 10
            torch.testing.assert_close(logits[:, 1:], model(changed)[:, 1:])
        logits.square().mean().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    print('Language model: logits / gradients / causal and self-only isolation PASS')


if __name__ == '__main__':
    torch.manual_seed(42)
    check_module(MultiHeadAttention(12, 3), 'Attention')
    check_module(TransformerBlock(12, 3), 'TransformerBlock')
    check_reference()
    residual_experiment()
    check_language_model()
