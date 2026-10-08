"""A tiny causal Transformer LM written on PyTorch primitives.

Architecture per prototype spec: sinusoidal PE (zero params), scaled token
embeddings, pre-LN residual blocks, hand-written causal multi-head attention
with optional dropout on the attention weights only, and a weight-tied output
head. Init is pinned: embeddings/head ~ N(0, 0.02); Linear layers keep the
PyTorch default (kaiming_uniform). GELU is exact (approximate='none').
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .utils import Config


def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding, Vaswani's formula. Returns (seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(two_i * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    # odd d_model: the cos half has one fewer column, so trim div to match
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])

    return pe


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the projected attention output and the attention weights (B, H, T, T)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)  # each (B, T, C)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        # Causal mask: a position may attend only to itself and earlier positions.
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~causal, float("-inf"))
        att = torch.softmax(att, dim=-1)

        if self.training and self.dropout_p > 0.0:
            att = nn.functional.dropout(att, p=self.dropout_p)  # attention weights only

        y = att @ v  # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y), att


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate="none"),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN residual block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, att = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))

        return x, att


class TinyLM(nn.Module):
    """A tiny causal Transformer LM with tied embeddings and a sinusoidal PE."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pe", sinusoidal_pe(cfg.seq_len, cfg.d_model), persistent=False
        )  # (seq_len, d_model), zero params
        self.blocks = nn.ModuleList(
            [
                Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tied

        nn.init.normal_(self.token_emb.weight, 0.0, 0.02)
        # All Linear layers keep the PyTorch default init (kaiming_uniform).

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return next-token logits, or (logits, attention) for the last layer."""
        x = self.token_emb(x) * math.sqrt(
            self.cfg.d_model
        )  # scale before adding PE (Vaswani)

        x = x + self.pe[: x.shape[1]]
        att = None

        for blk in self.blocks:
            x, att = blk(x)

        logits = self.head(self.ln_f(x))

        if return_attention:
            return logits, att

        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
