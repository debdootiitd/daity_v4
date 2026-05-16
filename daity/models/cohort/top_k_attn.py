"""Top-K multi-head attention building blocks.

Standard attention learns O(N²) attention weights per layer per head, fed by
only 2000 supervision signals per gradient step in our cohort setup —
empirically (v5 diagnostic) this collapses to uniform attention. Top-K
attention restricts each query to attend to the K highest-scoring keys
(masking the rest to -inf before softmax). Effects:

  - Sparsity reduces the attention surface from N to K params per query
    (parameter count of the projections is unchanged; the attention
    *weight matrix* becomes K-sparse).
  - Softmax over fewer items is easier to learn — the gradient signal
    per nonzero attention weight scales up by N/K.
  - Forces the model to commit to a small set of attending keys per
    query, instead of paying a "tiny weight" tax across all keys.

Two block flavors:
  - `TopKSelfAttnBlock`  — Q/K/V from the same input.
  - `TopKCrossAttnBlock` — Q from one input; K/V from another.

Both use pre-norm + SDPA-style (no RoPE — for cohort-axis attention,
positions are set-like / stock-indexed, not temporal).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _top_k_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    top_k: int | None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Sparse attention: each query keeps only its top-K key scores.

    Inputs:
      q: (B, H, T_q, d_head)
      k: (B, H, T_k, d_head)
      v: (B, H, T_k, d_v)
    Returns:
      (B, H, T_q, d_v)

    When `top_k` is None or >= T_k, this is plain softmax attention.
    """
    d_head = q.size(-1)
    scale = 1.0 / math.sqrt(d_head)
    scores = (q @ k.transpose(-2, -1)) * scale            # (B, H, T_q, T_k)
    T_k = scores.size(-1)
    if top_k is not None and 0 < top_k < T_k:
        # Mask all but top-K positions per query.
        topk_vals, topk_idx = scores.topk(top_k, dim=-1)  # (..., K)
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(-1, topk_idx, 0.0)
        scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v                                       # (B, H, T_q, d_v)


class _MHAProjection(nn.Module):
    """Linear projections for multi-head Q, K, V plus an output proj."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)

    def heads(self, t: torch.Tensor, T: int, B: int) -> torch.Tensor:
        """`(B, T, d_model)` → `(B, n_heads, T, d_head)`."""
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)


class TopKSelfAttnBlock(nn.Module):
    """Pre-norm top-K self-attention + FFN.

    `top_k=None` (or >= seq length) = plain dense attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: int = 2,
        *,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.norm_attn = nn.LayerNorm(d_model)
        self.proj = _MHAProjection(d_model, n_heads)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_n = self.norm_attn(x)
        q = self.proj.heads(self.proj.q_proj(x_n), T, B)
        k = self.proj.heads(self.proj.k_proj(x_n), T, B)
        v = self.proj.heads(self.proj.v_proj(x_n), T, B)
        out = _top_k_attention(q, k, v, top_k=self.top_k)              # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.proj.o_proj(out)
        x = x + out
        return x + self.ffn(self.norm_ffn(x))


class TopKCrossAttnBlock(nn.Module):
    """Pre-norm top-K cross-attention + FFN.

    Q from `q_in`; K, V from `kv_in`. `top_k` applies to the K/V axis.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: int = 2,
        *,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.proj = _MHAProjection(d_model, n_heads)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, T_q, _ = q_in.shape
        T_k = kv_in.size(1)
        q_n = self.norm_q(q_in)
        kv_n = self.norm_kv(kv_in)
        q = self.proj.heads(self.proj.q_proj(q_n), T_q, B)
        k = self.proj.heads(self.proj.k_proj(kv_n), T_k, B)
        v = self.proj.heads(self.proj.v_proj(kv_n), T_k, B)
        out = _top_k_attention(q, k, v, top_k=self.top_k)              # (B, H, T_q, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        out = self.proj.o_proj(out)
        q_in = q_in + out
        return q_in + self.ffn(self.norm_ffn(q_in))
