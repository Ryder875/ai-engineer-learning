# Week 3 核心笔记

## Attention

- 输入 [B,T,D]，D=d_model；每个头的维度 d_head=D/H。
- Q/K/V 分别投影；[B,T,D] → [B,T,H,d_head] → [B,H,T,d_head]。
- scores = Q @ K.transpose(-2,-1) / sqrt(d_head)，shape 为 [B,H,T,T]。
- causal mask 在 softmax 前屏蔽未来；沿最后的 key 轴归一化。
- weights @ V 后先交换头轴与序列轴，再 reshape 合头；输出投影混合头的特征。
- key bias 对某个 query 的可见 key 得分加入相同常数，softmax 不变，故其梯度理论上为零。

## Transformer block

- Pre-Norm：x = x + Attention(LN1(x))，然后 x = x + FFN(LN2(x))。
- 分支学习对输入的修正；残差也提供直接的梯度路径。
- LayerNorm(D) 对每个 token 的特征独立计算统计量；缩放、平移参数跨 token 共享。
- FFN：Linear(D,4D) → GELU → Linear(4D,D)，逐位置使用相同网络。
- Attention 汇总跨 token 信息；FFN 加工每个位置的特征，其中已可包含 Attention 带来的上下文。

## 小型语言模型与训练

- Token Embedding 按 ID 查可训练向量；位置 Embedding 提供位置信息，两者相加。
- block → 最终 LayerNorm → Linear(D,vocab_size)，输出 logits [B,T,V]。
- next-token 任务将序列错位一格；cross_entropy 使用原始 logits，无需先 softmax。
- logits reshape 为 [B*T,V]，targets 为 [B*T]；默认对有效位置取平均。
- 标签 -100 默认不计入交叉熵；准确率也需排除这些位置。
- 均匀猜测时准确率为 1/V，交叉熵为 ln(V)。本周 V=10，对应 10% 和约 2.303。
- train() 与 eval() 设置模块模式；no_grad() 关闭梯度记录，它们作用不同。

## 实验解释

- 循环数字只需当前 token，不能用它的高准确率证明模型利用了历史。
- 复制前一个独立随机 token 需要跨位置信息；这是诊断任务，不是 next-token 语言建模。
- Attention argmax 只说明最大权重位置；完整权重才能反映集中程度。
- 本次样本四个头均最关注前一位置，说明多个头可能学到相似模式。
- self-only 消融保留参数结构，只阻断跨位置 Attention；默认模型仍使用 causal mask。
- 相同生成分布下的高分不代表长序列、真实语言或跨分布泛化。
