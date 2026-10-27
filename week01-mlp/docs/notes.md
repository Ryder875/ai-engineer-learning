# Week 1 核心笔记

- Shape 是各轴长度；广播从右对齐，长度相等或其一为 1。行中心化保留均值维度：`x-x.mean(dim=1, keepdim=True)`。
- `[B,T,D] @ [D,H] -> [B,T,H]`，共享权重，不混合 token。
- `zero_grad → forward → loss → backward → step`；梯度默认累积，参数更新后旧 loss 不变。
- 链式法则：`prediction=wx+b`、平方误差时，`dL/dw=2*(prediction-target)*x`。
- `nn.Parameter` 注册参数；`model(x)` 调用 forward；Linear(a,b) 有 `a*b+b` 个参数。
- ReLU 作用在线性层输出；没有非线性时，多层 Linear 可合并为一层。
- BCEWithLogitsLoss 接收原始 logits 和同 shape 的浮点标签；logit 阈值 0 等价于概率阈值 0.5。
- `eval()` 不关闭梯度，评估同时用 `no_grad()`；训练集更新参数，验证集选配置，测试集最终评估。
