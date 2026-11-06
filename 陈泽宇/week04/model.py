"""Decoder-only Transformer with tied input/output embeddings."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 64
    width: int = 192
    heads: int = 3
    layers: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.heads, self.layers) < 1:
            raise ValueError("Model sizes must be positive.")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Width must be divisible by heads and dropout must be in [0, 1).")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x):
        batch, time, width = x.shape
        qkv = self.qkv(x).reshape(batch, time, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(z.transpose(1, 2).reshape(batch, time, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.width)
        self.attn = CausalAttention(config)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.width, config.width),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.attn_norm(x)))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.ffn[-1].weight, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, last_only=False):
        if not 0 < ids.size(1) <= self.config.context:
            raise ValueError("Input length must be within the model context.")
        pos = torch.arange(ids.size(1), device=ids.device)
        hidden = self.blocks(self.drop(self.token(ids) + self.position(pos)))
        if last_only:
            hidden = hidden[:, -1:, :]
        return F.linear(self.norm(hidden), self.token.weight)
