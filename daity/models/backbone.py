"""Patch-transformer backbone for daity_v3 pretraining (DESIGN §3.1).

PatchTST/Chronos-style: channel-independent in the lower layers,
channel-mixing in the upper 2. RoPE applied to the time axis. The backbone
takes the tokenizer's output sequence (FORECAST token at position 0,
followed by per-scale patches) and produces a same-shape hidden-state
sequence the SSL heads consume.

Default sizing (DESIGN §3.1):
  d_model      = 320
  n_heads      = 8
  n_layers     = 8
  ffn_ratio    = 4×
  channel_mix_layers = 2  (last 2 layers see other channels)

Param count at default size: roughly
    8 layers × (4 × d² + 8 × d²) ≈ 8 × 12 × 320² ≈ 9.8M for self-attn + FFN,
    plus norms / projection ≈ ~12-15M total. Below the DESIGN target of
    ~20M; the gap is filled when we add the cross-attention OB fusion in
    Phase 4.

Implementation notes:
  - Standard pre-norm transformer block (LayerNorm before attn / FFN).
  - SDPA (`F.scaled_dot_product_attention`) — picks up FlashAttention-2
    on CUDA / Memory-Efficient Attention on MPS / standard on CPU.
  - RoPE applied to query + key in every layer's attention. Pre-computed
    sin/cos cached as a buffer up to a large max sequence length.
  - The "channel-mixing only in upper layers" intent in DESIGN is
    operationalized by the tokenizer choice: lower layers see flattened
    (channel * patch_len) features as one token; upper layers can
    optionally do cross-channel attention via a separate token-by-channel
    representation. For Phase 2 SSL pretraining, all layers operate on
    the channel-mixed token sequence — channel-independent vs channel-
    mixing distinction is folded into Phase 4's OB fusion architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _rope_freqs(d_head: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    """Pre-compute RoPE sin/cos for `max_seq_len` positions × `d_head` dims.

    Returns a `(max_seq_len, d_head)` tensor where the first half is
    cosines and the second half is sines (interleaved so the standard RoPE
    rotation works via element-wise multiply + roll).
    """
    if d_head % 2 != 0:
        msg = f"RoPE requires an even d_head, got {d_head}"
        raise ValueError(msg)
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)            # (T, d_head/2)
    # Stack so that final dim alternates (cos, sin) per pair — enables the
    # standard RoPE multiply with a "rotate-pairs" shuffle.
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1)                       # (T, d_head)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to `x` of shape `(B, H, T, d_head)`.

    `freqs` is `(T, d_head)` from `_rope_freqs`. The rotation is:
        x_rot[:, :, :, ::2]  = x[..., ::2]  * cos - x[..., 1::2] * sin
        x_rot[:, :, :, 1::2] = x[..., 1::2] * cos + x[..., ::2]  * sin
    """
    T = x.size(-2)
    d_head = x.size(-1)
    cos = freqs[:T, : d_head // 2]
    sin = freqs[:T, d_head // 2 :]
    # Pair up (x_even, x_odd) and rotate.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_odd * cos + x_even * sin
    # Re-interleave.
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with RoPE on Q + K, SDPA backend.

    Uses `F.scaled_dot_product_attention` so the runtime picks the best
    available kernel: FlashAttention-2 on CUDA, Memory-Efficient Attention
    on MPS, standard on CPU.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        max_seq_len: int = 4096,
        rope_base: float = 10_000.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_dropout = attn_dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        self.register_buffer(
            "rope_freqs",
            _rope_freqs(self.d_head, max_seq_len=max_seq_len, base=rope_base),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                                 # each (B, T, H, d_head)
        # Move heads to dim 1 for the SDPA call: (B, H, T, d_head).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)
        # SDPA with optional causal mask.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )                                                            # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMS-LN → SDPA-RoPE → residual → LN → FFN → residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: int = 4,
        *,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(
            d_model, n_heads, max_seq_len=max_seq_len, attn_dropout=attn_dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        return x + self.ffn(self.norm2(x))


class Backbone(nn.Module):
    """Patch-transformer backbone (DESIGN §3.1).

    Forward input: `(B, T, d_model)` token sequence from `MultiResTokenizer`.
    Forward output: same shape; the downstream heads slice out:
      - `out[:, 0]`        — FORECAST token's hidden state (for quantile /
                              ranking / direction / vol heads in Phase 3).
      - `out[:, 1:]`        — per-patch hidden states (for masked
                              reconstruction + next-patch prediction
                              SSL heads in Phase 2c).
    """

    def __init__(
        self,
        d_model: int = 320,
        n_layers: int = 8,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        *,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            msg = f"n_layers must be positive, got {n_layers}"
            raise ValueError(msg)
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, ffn_ratio=ffn_ratio,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        """`(B, T, d_model)` → `(B, T, d_model)`. `causal=True` for the NTP head."""
        for layer in self.layers:
            x = layer(x, causal=causal)
        return self.final_norm(x)
