# Week 2：单头 Attention

日常先看 **model.py 学 Attention，train.py 复习 mini-batch 实验**。
本周 train.py 训练的是 MLP；Attention 的梯度更新演示位于 exercises。

```text
week02-attention/
├── README.md
├── model.py          # 缩放点积 Attention、SelfAttention、实验用 MLP
├── train.py          # MLP batch size 对照实验
├── check.py          # Attention 的统一检查
├── docs/
│   ├── notes.md
│   ├── experiment.md
│   └── weekly_report.md
└── exercises/
    └── attention_demo.py  # Attention 的一次梯度更新演示
```

## 运行（PowerShell）

复用第一周环境：

```powershell
cd D:\ai-learning\week02-attention
$taskPython = 'D:\ai-learning\week01-mlp\.venv\Scripts\python.exe'
& $taskPython -B check.py
& $taskPython -B -m exercises.attention_demo
& $taskPython -B train.py
```

默认 batch_size=360、epochs=2400、lr=0.1。重跑其他对照时修改 train.py 顶部配置。
RUN_TEST 默认 False，本周实验使用验证集；本次目录整理不重新使用测试集。
Attention 演示的输出平方均值仅用于展示反向传播，不是语言模型训练目标。

## 阅读顺序

docs/notes.md → model.py 中的 attention 与 SelfAttention → check.py → docs/experiment.md。
批训练复习看 train.py 与 model.py 末尾的 MLP；周报见 docs/weekly_report.md。

原 attention_basics.py 与 self_attention.py 的模型定义合并到 model.py。
原 test_attention.py 改为 check.py，train_minibatch.py 改为 train.py。
