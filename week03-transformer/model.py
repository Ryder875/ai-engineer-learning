"""按 Attention → Block → LanguageModel 顺序阅读。"""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, self_only=False):
        super().__init__()

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.self_only = self_only
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        # 1. 三次独立投影：[B, T, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [T, T]：True 表示需要屏蔽的未来位置
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        if self.self_only:
            mask = ~torch.eye(T, dtype=torch.bool, device=x.device)

        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        head_outputs = weights @ v
        merged = head_outputs.transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(merged)
        return output, weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, self_only=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, self_only=self_only)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        attn_output, weights = self.attention(self.norm1(x))
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x, weights


class MiniLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_length, self_only=False):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.block = TransformerBlock(d_model, num_heads, self_only=self_only)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        _, T = token_ids.shape
        if T > self.position_embedding.num_embeddings:
            raise ValueError("输入长度超过位置 Embedding 的最大长度")
        positions = torch.arange(T, device=token_ids.device)

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        x, _ = self.block(x)
        logits = self.lm_head(self.final_norm(x))

        return logits
