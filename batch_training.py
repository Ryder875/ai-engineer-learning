"""MLP 二分类：用验证集选学习率，--test 开启最终测试。"""
import argparse

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.activation = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))


def evaluate(model, features, targets):
    model.eval()  # eval 不会关闭梯度跟踪，仍需 no_grad。
    with torch.no_grad():
        logits = model(features)
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        predictions = (logits >= 0).float()
        accuracy = (predictions == targets).float().mean()
    return loss.item(), accuracy.item(), predictions


def main(learning_rate=0.1, run_test=False):
    # 每次运行从相同的数据、划分和模型初始化开始。
    torch.manual_seed(42)
    features = torch.rand(600, 2) * 2 - 1
    targets = (features[:, 0] * features[:, 1] > 0).float().unsqueeze(1)
    indices = torch.randperm(600)
    train_idx, val_idx, test_idx = indices[:360], indices[360:480], indices[480:]
    x_train, y_train = features[train_idx], targets[train_idx]
    x_val, y_val = features[val_idx], targets[val_idx]

    # 不重复且覆盖全部样本，确保三个集合无交叉。
    assert torch.unique(indices).numel() == 600
    assert y_train.shape == (360, 1) and y_val.shape == (120, 1)
    model = MLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    assert model(x_train).shape == y_train.shape
    assert sum(p.numel() for p in model.parameters()) == 33
    initial_weight = model.hidden.weight.detach().clone()

    print(f"seed=42, lr={learning_rate}, epochs=2000, batch_size=360, device=cpu")
    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        if epoch == 0:
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
        optimizer.step()
        if epoch == 0:
            assert not torch.equal(initial_weight, model.hidden.weight.detach())
        if (epoch + 1) % 200 == 0:
            val_loss, val_accuracy, _ = evaluate(model, x_val, y_val)
            print(f"epoch={epoch + 1:4d}, train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.2%}")

    # 训练损失日志来自本次更新前，验证指标来自更新后。
    if run_test:
        x_test, y_test = features[test_idx], targets[test_idx]
        test_loss, test_accuracy, predictions = evaluate(model, x_test, y_test)
        wrong = (predictions != y_test).squeeze(1)
        correct = int((predictions == y_test).sum().item())
        print(f"test_loss={test_loss:.4f}, test_acc={test_accuracy:.2%} "
              f"({correct}/{len(y_test)})")
        print("错误样本特征:", x_test[wrong])
        print("真实标签:", y_test[wrong].squeeze(1))
        print("预测标签:", predictions[wrong].squeeze(1))
    else:
        print("测试集未评估；配置选定后使用 --test。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=0.1, help="SGD 学习率，默认 0.1")
    parser.add_argument("--test", action="store_true", help="训练结束后评估测试集")
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error("学习率必须大于 0")
    main(args.lr, args.test)
