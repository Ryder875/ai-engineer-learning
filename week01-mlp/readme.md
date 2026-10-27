# Week 1：PyTorch 与第一个 MLP

日常只需关注 **model.py 和 train.py**。原学习结果与周报已保留。

```text
week01-mlp/
├── README.md
├── model.py          # 两层 MLP：2 → 8 → 1
├── train.py          # 训练、验证、可选最终测试
├── check.py          # 基础练习、内置检查与环境检查
├── requirements.txt
└── docs/
    ├── notes.md
    ├── experiment.md
    └── weekly_report.md
```

## 运行（PowerShell）

复用当前目录已有的 .venv：

```powershell
cd D:\ai-learning\week01-mlp
$taskPython = '.\.venv\Scripts\python.exe'
& $taskPython -B check.py
& $taskPython -B train.py --lr 0.1
& $taskPython -B train.py --lr 0.01
```

选定配置后的最终测试：`& $taskPython -B train.py --lr 0.1 --test`。
训练使用 CPU；check.py 额外检查 GPU（可用时）。脚本内置 assert，无需 pytest。
每次运行使用相同种子、数据划分和初始化，默认不评估测试集。

## 阅读顺序

docs/notes.md → model.py → train.py → docs/experiment.md → docs/weekly_report.md。
check.py 用于复习 MSE、Tensor、广播、梯度与参数更新。

## 首次配置

已有 .venv 不必重复安装。需要新建环境时：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

原 basics.py 改为 check.py，batch_training.py 拆成 model.py 和 train.py。
.git、.venv、依赖记录均保留；学习记录没有因目录整理而改写。
