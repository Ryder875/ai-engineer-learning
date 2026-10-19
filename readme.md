# Week 1：PyTorch 与第一个 MLP

已完成基础练习、MLP 二分类、学习率对比、最终测试和原理问答。
投入约 10 小时；[阶段报告](weekly_report.md)。下一步：从空文件独立重写训练流程。

## 运行

Windows · Python 3.13.16 · PyTorch 2.11.0+cu128 · RTX 5070 Ti。
项目保留两个脚本：`basics.py` 用于复习和检查，`batch_training.py` 用于训练与评估。
依赖记录见 `requirements.txt`；环境和缓存由 `.gitignore` 忽略。

首次配置：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

在项目根目录运行（训练使用 CPU，基础脚本额外检查 GPU）：

```powershell
.\.venv\Scripts\python.exe basics.py
.\.venv\Scripts\python.exe batch_training.py --lr 0.1
.\.venv\Scripts\python.exe batch_training.py --lr 0.01
.\.venv\Scripts\python.exe batch_training.py --lr 0.1 --test
```

每次从相同初始化重新训练。默认不评估测试集，`--test` 用于选定配置后的最终报告。
脚本内置 assert，覆盖已知计算结果、广播含义、梯度、参数注册、MLP shape 与参数更新；尚未整理为 pytest。

## 复习要点

- Shape 是各轴长度；广播从右对齐，长度相等或其一为 1。行中心化保留均值维度：`x-x.mean(dim=1, keepdim=True)`。
- `[B,T,D] @ [D,H] -> [B,T,H]`，共享权重，不混合 token。
- `zero_grad → forward → loss → backward → step`；梯度默认累积，参数更新后旧 loss 不变。
- 链式法则：`prediction=wx+b`、平方误差时，`dL/dw=2*(prediction-target)*x`。
- `nn.Parameter` 注册参数；`model(x)` 调用 forward；Linear(a,b) 有 `a*b+b` 个参数。
- ReLU 作用在线性层输出；没有非线性时，多层 Linear 可合并为一层。
- BCEWithLogitsLoss 接收原始 logits 和同 shape 的浮点标签；logit 阈值 0 等价于概率阈值 0.5。
- `eval()` 不关闭梯度，评估同时用 `no_grad()`；训练集更新参数，验证集选配置，测试集最终评估。

## MLP 实验

600 个二维样本，特征在 `[-1,1)`，乘积大于零为 1，否则为 0。
固定种子 42，训练/验证/测试为 360/120/120。
模型：Linear(2,8) → ReLU → Linear(8,1)，共 33 个参数。
CPU/float32，SGD，全批量，BCEWithLogitsLoss，2000 epochs；仅改变学习率。

| 学习率 | 训练损失 | 验证损失 | 验证准确率 |
|---|---:|---:|---:|
| 0.1 | 0.0719 | 0.1057 | 99.17%（119/120） |
| 0.01 | 0.6028 | 0.6051 | 85.00%（102/120） |

训练损失来自最后一次更新前，验证指标来自更新后。同一预算下按验证集选 0.1；较小学习率仍在收敛，不能据此判断最终性能上限。

最终测试：损失 **0.0798**，准确率 **98.33%（118/120）**。
错误样本 `[0.0235,-0.6831]`、`[-0.0260,0.7806]` 都将类别 0 预测为 1，接近真实边界 `x₁=0`。
仅一次固定划分和初始化，未评估跨种子稳定性，也未据测试结果继续调参。

历史练习：三个样本拟合 `y=2x+1`，SGD lr=0.01、1000 次，最终 `w≈2.0040, b≈0.9909`，训练 MSE≈`1.18128e-5`；原训练脚本已升级为 MLP。
