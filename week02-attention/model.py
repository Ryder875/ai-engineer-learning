"""单头 Attention；末尾 MLP 用于 batch size 对照实验。"""
import torch
from torch import nn


def attention(q, k, v, causal=False):
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)

    if causal:
        if q.size(-2) != k.size(-2):
            raise ValueError("本课 causal mask 要求 Q、K 序列长度相同")
        length = q.size(-2)
        # True 为禁止读取的位置；[T,T] 广播到每个 batch。
        mask = torch.ones(
            length, length, dtype=torch.bool, device=q.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=False):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return attention(q, k, v, causal=causal)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.layers(x)
