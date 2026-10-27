"""MLP batch size 实验：使用验证集观察训练表现。"""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


LEARNING_RATE = 0.1
BATCH_SIZE = 360
EPOCHS = 2400
RUN_TEST = False  # 用验证集选好配置后，改为 True 做最终测试。


from model import MLP


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        accuracy = ((logits >= 0) == y).float().mean()
    return loss.item(), accuracy.item()


def main():
    # 固定数据、划分和模型初始化，方便比较实验。
    torch.manual_seed(42)
    x = torch.rand(600, 2) * 2 - 1
    y = (x[:, 0] * x[:, 1] > 0).float().unsqueeze(1)
    indices = torch.randperm(600)
    train, val, test = indices[:360], indices[360:480], indices[480:]
    x_train, y_train = x[train], y[train]
    x_val, y_val = x[val], y[val]
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        train_loss = total_loss / len(train_dataset)

        if epoch % 200 == 0:
            val_loss, val_acc = evaluate(model, x_val, y_val)
            print(
                f"epoch={epoch}, train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
            )

    # train_loss 汇总本轮各批次更新前的损失；
    # 验证指标使用本轮所有更新完成后的模型计算。
    if RUN_TEST:
        test_loss, test_acc = evaluate(model, x[test], y[test])
        print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.2%}")


if __name__ == "__main__":
    main()

