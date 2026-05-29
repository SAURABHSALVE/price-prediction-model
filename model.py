"""
PyTorch building blocks for compound-token market sequence modeling.

Provides a sliding-window dataset over pipeline token arrays and a fused,
causally masked multi-head self-attention module tuned for throughput.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class QuantMarketDataset(Dataset):
    """
    Overlapping (x, y) windows over a 1D token stream.

    x = tokens[i : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]  (next-token targets, same length)
    """

    def __init__(self, tokens: np.ndarray, block_size: int) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        arr = np.asarray(tokens, dtype=np.int64).ravel()
        if arr.ndim != 1:
            raise ValueError(f"tokens must be 1D, got shape {arr.shape}")
        if len(arr) <= block_size:
            raise ValueError(
                f"Need more than block_size tokens (len={len(arr)}, block_size={block_size})"
            )
        self.tokens = arr
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError(f"Index {i} out of range for dataset of length {len(self)}")
        start = i
        end = i + self.block_size
        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1 : end + 1], dtype=torch.long)
        return x, y


class CausalSelfAttention(nn.Module):
    """
    Fused QKV projection, multi-head causal scaled dot-product attention,
    and output projection with dropout.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        causal_mask = torch.tril(torch.ones(self.block_size, self.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, self.block_size, self.block_size))

        self._attn_weights_pre_softmax: torch.Tensor | None = None

    def _attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Scaled dot-product scores with strict causal masking (pre-softmax)."""
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        return att.masked_fill(mask == 0, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_embd)
            return_attn: If True, also return pre-softmax attention weights
                of shape (batch, n_head, seq_len, seq_len).
        """
        batch, seq_len, channels = x.shape
        if channels != self.n_embd:
            raise ValueError(f"Expected n_embd={self.n_embd}, got {channels}")
        if seq_len > self.block_size:
            raise ValueError(f"seq_len {seq_len} exceeds block_size {self.block_size}")

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = self._attention_scores(q, k, seq_len)
        self._attn_weights_pre_softmax = att.detach()

        att_probs = F.softmax(att, dim=-1)
        att_probs = self.attn_dropout(att_probs)

        out = att_probs @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        out = self.resid_dropout(self.c_proj(out))

        if return_attn:
            return out, att
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    mock_tokens = rng.integers(0, 16, size=2000, dtype=np.int64)

    block_size = 10
    dataset = QuantMarketDataset(mock_tokens, block_size=block_size)
    x, y = dataset[0]

    assert len(x) == len(y) == block_size
    assert torch.equal(y[:-1], x[1:]), "y must be x advanced by one token at each position"
    assert y[-1].item() == mock_tokens[block_size]
    print("QuantMarketDataset: (x, y) window alignment OK")

    class DummyConfig:
        block_size = 32
        n_embd = 384
        n_head = 6
        dropout = 0.1

    config = DummyConfig()
    attn_module = CausalSelfAttention(config)
    attn_module.eval()

    batch_size = 4
    seq_len = config.block_size
    x_in = torch.randn(batch_size, seq_len, config.n_embd)
    out, att_pre_softmax = attn_module(x_in, return_attn=True)

    assert out.shape == (batch_size, seq_len, config.n_embd), (
        f"Expected output shape {(batch_size, seq_len, config.n_embd)}, got {tuple(out.shape)}"
    )
    print(f"CausalSelfAttention output shape: {tuple(out.shape)}")

    sub = att_pre_softmax[0, 0, :5, :5].detach()
    print("\nPre-softmax attention (batch=0, head=0), 5x5 upper-left:")
    print(sub)

    ui, uj = torch.triu_indices(5, 5, offset=1)
    upper_vals = sub[ui, uj]
    assert torch.all(torch.isinf(upper_vals)), (
        "Strict upper triangle must be -inf before softmax"
    )
    print("\nUpper-triangular (diagonal=1) elements are all -inf: OK")
    print("\nAll model.py checks passed.")
