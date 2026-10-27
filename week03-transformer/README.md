# Week 3：Transformer 与小型序列模型

日常只需要看两个文件：**model.py 学模型，train.py 跑实验**。

```text
week03-transformer/
├── README.md
├── model.py             # Attention → TransformerBlock → MiniLanguageModel
├── train.py             # 三种实验共用的训练入口
├── check.py             # 统一功能检查
├── docs/
│   ├── notes.md         # 知识笔记
│   ├── experiment.md    # 配置、结果、解释边界
│   └── weekly_report.md # 周报，实际学习时长待补充
└── exercises/
    └── head_shapes.py   # 已完成的拆头、合头练习
```

## 运行（PowerShell）

```powershell
cd D:\ai-learning\week03-transformer
$taskPython = 'D:\ai-learning\week01-mlp\.venv\Scripts\python.exe'
& $taskPython check.py
& $taskPython train.py --task cycle
& $taskPython train.py --task copy
& $taskPython train.py --task copy --self-only
```

可选复习练习：`& $taskPython exercises/head_shapes.py`。

默认 causal Attention；--self-only 只允许关注自己，用于消融，无需修改模型代码。
默认 seed=42；cycle 训练 300 步，copy 训练 1000 步；可用 --seed 和 --steps 调整。
实验使用 CPU，在 20 个新 batch 上评估。相同配置和种子的复制对照使用相同初始化、训练及评估数据流。

## 阅读顺序

1. docs/notes.md：复习概念。
2. model.py：三个类按依赖顺序放在同一文件，计算过程保持显式可读。
3. train.py：数据、训练、评估。
4. docs/experiment.md 与 docs/weekly_report.md：结果和总结。

当前是单 block、固定短序列的合成任务，尚未包含真实文本训练或自回归生成。
