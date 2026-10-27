# Week 2 笔记：训练与单头 Attention

## 1. Mini-batch 训练

- Epoch：遍历一次训练集。Step：本课指一次参数更新。
- 360 个样本、batch_size=32，每轮更新 12 次，最后一批有 8 个样本。
- 每批执行：zero_grad → forward → loss → backward → step。
- backward 累加梯度；step 更新参数但不清空梯度。
- 训练平均 loss = sum(每批平均 loss × 该批样本数) / 总样本数。
- 训练日志汇总不同参数状态下的批次 loss；验证使用本轮结束后的固定模型。
- eval() 切换模型行为；no_grad() 关闭梯度跟踪。
- Accuracy 相同，分配给真实类别的概率仍可不同，因此 loss 可不同。
- 相同 epoch 数、更新次数和实际耗时，是不同的比较条件。结果见 experiment.md。

## 2. Attention 的计算

```text
X → 三个不同的线性层 → Q、K、V
Q @ Kᵀ / √d_k → softmax → weights
weights @ V → output
```

- B：批次大小；T：位置数；D：特征维度。
- Linear 只变换最后一维，不混合不同位置；所有位置共享参数。
- Q/K 为 [B,T,Dk]，V 为 [B,T,Dv]：weights 为 [B,T,T]，output 为 [B,T,Dv]。
- scores[b,i,j] 是第 b 个样本的 query i 与 key j 的匹配分数。
- softmax(dim=-1) 沿 key 维度归一化，每行权重和为 1。
- 权重的一行，对 V 的各行加权求和；不同 batch 独立计算。
- 例如：[0.25,0.75] @ [[2,0],[0,4]] = [0.5,3]。
- 同一行分数相同时，权重均匀，输出为各 value 的均值。
- 简化独立、单位方差假设下，点积方差为 d_k；除以 √d_k 可稳定尺度，缓和 softmax 饱和。

## 3. 参数与梯度

- Q/K/V 是输入经过投影得到的中间结果；可训练参数是三个线性层的权重。
- d_model=4、无偏置时，共有 3×4×4=48 个参数。
- 输出产生 loss 后，梯度可传回三个投影层。
- output.square().mean() 仅用于梯度演示，不代表模型学会了语言任务。

## 4. Causal mask

- 每个位置只读取自己和之前的位置。
- triu(diagonal=1) 标记未来位置，True 表示禁止读取。
- softmax 前将这些分数设为 -inf，使对应权重为 0；设为 0 不能屏蔽。
- 修改未来 token，过去位置的输出保持不变；未来位置自己的输出仍可能改变。
- 检查覆盖：shape、权重和、加权汇总、均匀权重、未来隔离、梯度。

## 预习：本周不要求实现

- 下一个 token 预测：inputs=tokens[:-1]，targets=tokens[1:]。
- Token ID 是编号；Embedding 查表得到 [B,T,D] 向量。
- 可学习位置向量与 token 向量相加，shape 不变。
- 输出层 Linear(D,V) 产生 [B,T,V] logits，交叉熵比较下一个 token 的 ID。
- V 为固定词表规模，不等于句子数；增加训练句子不一定扩大词表。
