# Week 1：PyTorch MLP

目标：自己实现可训练、可测试的 MLP，并解释完整训练过程。

## 环境

- Windows
- Python 3.13.16
- NVIDIA GeForce RTX 5070 Ti
- PyTorch CUDA 12.8 安装源

## 安装依赖

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 环境检查

```powershell
.\.venv\Scripts\python.exe check_env.py
```

已验证 GPU 矩阵乘法和反向传播可以运行。