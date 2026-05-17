"""SSL heads for the Phase 2 pretraining backbone (DESIGN §3.2).

Four objectives, weighted sum (initial weights 1.0 / 0.5 / 0.3 / 0.5):

  1. **Masked patch reconstruction** (primary). A fraction of patches in
     each scale's stream is replaced with a `[MASK]` embedding before the
     backbone sees them. The reconstruction head projects the post-
     backbone hidden states back to `(C × patch_len)` and we minimize a
     volume-weighted MSE against the original RevIN-normalized values.

  2. **Next-patch prediction** (causal). A separate head reads the
     backbone's *causal* pass output and predicts the next patch from
     the prefix. Same `(C × patch_len)` output shape; loss is plain
     unweighted MSE on RevIN-normalized values.

  3. **Cross-sectional contrastive** (InfoNCE). The backbone produces a
     FORECAST token per sample (per-(symbol, end_ts) pair); the
     contrastive head projects it to a unit-norm embedding, and InfoNCE
     pulls together same-end_ts pairs from co-moving symbols, pushes
     apart unrelated pairs.

  4. **Multi-horizon forecast on FORECAST token**. Reads the *non-causal*
     pass's hidden at position 0 (which has attended to the full input
     window) and predicts the next H non-overlapping patches at a single
     forecast scale. Trains the FORECAST token to be a useful summary
     for downstream forecasting heads (Phase 3+).

This module ships:
  - `MaskedReconstructionHead` — `(B, P, d_model)` → `(B, P, C * patch_len)`.
  - `NextPatchHead` — same shape signature, used with the causal output.
  - `ContrastiveHead` — `(B, d_model)` → `(B, proj_dim)` ℓ2-normalized.
  - `MultiHorizonForecastHead` — `(B, d_model)` → `(B, H, C, patch_len)`.

The actual masking + loss code lives in `daity.training.losses` so the
heads stay focused on architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedReconstructionHead(nn.Module):
    """Predict the original `(C, patch_len)` patch values from a hidden state.

    Two-layer MLP with GELU. Input: `(B, P, d_model)`. Output: `(B, P, C * patch_len)`.
    Caller reshapes to `(B, P, C, patch_len)` if needed.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        patch_len: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if num_channels <= 0 or patch_len <= 0:
            msg = f"num_channels and patch_len must be positive ({num_channels}, {patch_len})"
            raise ValueError(msg)
        self.num_channels = num_channels
        self.patch_len = patch_len
        out_dim = num_channels * patch_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`(B, P, d_model)` → `(B, P, C * patch_len)`."""
        return self.net(hidden)


class NextPatchHead(nn.Module):
    """Predict the next patch from the causal-prefix hidden state.

    Architecturally identical to `MaskedReconstructionHead`; kept as a
    separate class so the two losses' weight init / fine-tuning don't
    interfere, and so DESIGN's "separate head" intent is explicit in code.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        patch_len: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_len = patch_len
        out_dim = num_channels * patch_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`(B, P, d_model)` → `(B, P, C * patch_len)`. Caller is responsible
        for slicing to `[:, :-1]` (the last position predicts past the end)
        before computing loss against `[:, 1:]` of the targets."""
        return self.net(hidden)


class ContrastiveHead(nn.Module):
    """Project the FORECAST hidden state to a contrastive embedding.

    Two-layer MLP + ℓ2 normalization on output. The InfoNCE loss expects
    unit-norm vectors so cosine similarity == dot product.
    """

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if proj_dim <= 0:
            msg = f"proj_dim must be positive, got {proj_dim}"
            raise ValueError(msg)
        self.proj_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, proj_dim),
        )

    def forward(self, forecast_hidden: torch.Tensor) -> torch.Tensor:
        """`(B, d_model)` → `(B, proj_dim)` with unit ℓ2 norm per row."""
        if forecast_hidden.dim() != 2:
            msg = f"Expected (B, d_model), got shape {tuple(forecast_hidden.shape)}"
            raise ValueError(msg)
        z = self.net(forecast_hidden)
        return F.normalize(z, p=2, dim=-1)


class MultiHorizonForecastHead(nn.Module):
    """Predict the next `n_patches` non-overlapping patches from the FORECAST token.

    Reads the non-causal-pass hidden at position 0 (which has attended to
    the full input window) and emits a bundle of `n_patches` future
    patches at a single forecast scale. Output shape is
    `(B, n_patches, num_channels, patch_len)`.

    Two-layer MLP with a single direct projection to the full bundle —
    deliberately not autoregressive. Multi-step direct heads outperform
    rollout-style autoregressive forecasts on patch-transformers per the
    PatchTST / Chronos / MOIRAI literature. The head is small enough
    that we can scale n_patches without bloating the param count
    (one Linear from d_model → n_patches·C·patch_len).
    """

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        num_channels: int,
        patch_len: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if n_patches <= 0:
            msg = f"n_patches must be positive, got {n_patches}"
            raise ValueError(msg)
        if num_channels <= 0 or patch_len <= 0:
            msg = (
                f"num_channels and patch_len must be positive "
                f"({num_channels}, {patch_len})"
            )
            raise ValueError(msg)
        self.n_patches = n_patches
        self.num_channels = num_channels
        self.patch_len = patch_len
        out_dim = n_patches * num_channels * patch_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, out_dim),
        )

    def forward(self, forecast_hidden: torch.Tensor) -> torch.Tensor:
        """`(B, d_model)` → `(B, n_patches, num_channels, patch_len)`."""
        if forecast_hidden.dim() != 2:
            msg = f"Expected (B, d_model), got shape {tuple(forecast_hidden.shape)}"
            raise ValueError(msg)
        B = forecast_hidden.size(0)
        flat = self.net(forecast_hidden)                 # (B, n_patches * C * patch_len)
        return flat.view(B, self.n_patches, self.num_channels, self.patch_len)


class MultiHorizonScalarForecastHead(nn.Module):
    """Phase 2.5 — predict log returns at SPECIFIC future horizons.

    Unlike `MultiHorizonForecastHead` (which predicts 6 consecutive 5m
    patches as dense look-ahead) and `CrossAttentionForecastHead` (same,
    cross-attention variant), this head predicts a single scalar per
    (horizon, channel) at the exact horizons used by the downstream
    supervised Phase 3 (15m, 30m, 45m, 60m, 90m, 120m, 180m forward).

    Architecture:
      - `n_horizons` learnable queries (one per horizon), each `d_model`-dim.
      - K cross-attention blocks (queries attend over encoder hidden states).
      - Final norm + Linear(d_model → num_channels) per horizon.

    Output shape: `(B, n_horizons, num_channels)` — log return at each
    horizon for each forecast channel.

    Compared to the consecutive-patch heads: 7×5=35 targets per sample
    (vs 96×5=480 for 6-patch dense). Less dense supervision but each
    target is structurally aligned with the supervised downstream
    (Phase 3 posttrain reads the same horizon labels).
    """

    def __init__(
        self,
        d_model: int,
        n_horizons: int,
        num_channels: int,
        *,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if n_horizons <= 0:
            msg = f"n_horizons must be positive, got {n_horizons}"
            raise ValueError(msg)
        if num_channels <= 0:
            msg = f"num_channels must be positive, got {num_channels}"
            raise ValueError(msg)
        if d_model % n_heads != 0:
            msg = (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                f"for MultiHorizonScalarForecastHead"
            )
            raise ValueError(msg)
        self.n_horizons = n_horizons
        self.num_channels = num_channels
        # Learnable per-horizon query — each carries "what does horizon h
        # look like" and is updated by cross-attention to encoder states.
        self.queries = nn.Parameter(
            torch.randn(n_horizons, d_model) * query_init_std,
        )
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        # Per-horizon → per-channel scalar. Shared projection across horizons.
        self.proj = nn.Linear(d_model, num_channels)

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """`(B, T, d_model)` → `(B, n_horizons, num_channels)`."""
        if encoder_hidden.dim() != 3:
            msg = (
                f"Expected (B, T, d_model), got shape {tuple(encoder_hidden.shape)}"
            )
            raise ValueError(msg)
        B = encoder_hidden.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, H, d_model)
        for block in self.blocks:
            q = block(q, encoder_hidden)
        q = self.final_norm(q)
        return self.proj(q)                                            # (B, H, num_channels)


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN block. Q comes from learned forecast
    queries; K, V come from the encoder hidden states.

    Standard transformer-decoder block but without self-attention on Q
    (the forecast queries don't attend to each other across positions —
    each future-position prediction is independent of the others).

    Adding self-attention on the queries would let predictions condition
    on each other (autoregressive-ish coupling); the published Time-FM
    forecast heads (PatchTST-2024, MOIRAI, TimesFM) split on this — some
    do, some don't. We default to NO self-attention for simplicity; can
    be added as a flag if multi-position coherence becomes a concern.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (B, H, d_model); kv: (B, T, d_model). Returns (B, H, d_model)."""
        # Cross-attention: q queries the encoder hidden states.
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + attn_out
        # FFN on the queries.
        q = q + self.ffn(self.norm2(q))
        return q


class CrossAttentionForecastHead(nn.Module):
    """Multi-horizon forecast head — cross-attention over encoder output.

    This is the architect-recommended replacement for `MultiHorizonForecastHead`:
    instead of compressing every relevant signal through a single
    `(B, d_model)` FORECAST-token bottleneck (480 dims for d_model=480),
    H learnable forecast-position queries cross-attend over the FULL
    encoder output (`(B, T, d_model)` = 81 patch tokens × 480 dims =
    ~39K conditioning dims per sample). This is the canonical Time-FM
    forecast-head architecture (PatchTST 2024, MOIRAI-Base, Chronos,
    TimesFM all use cross-attention).

    Shape contract:
      Input: encoder_hidden (B, T, d_model) — the patch-token hidden
        states from the non-causal backbone pass. Caller passes
        `hidden_nc[:, 1:]` to skip the FORECAST token (it's redundant
        as a key/value since the cross-attn head is its replacement).
      Output: (B, n_patches, num_channels, patch_len) — same as the
        MLP head; downstream loss code is unchanged.

    Head capacity at d_model=480, n_layers=2, n_heads=8, ffn_ratio=2,
    n_patches=6, num_channels=18, patch_len=16:
      - 6 learned queries × 480 dims = 2.9K params
      - Per block: 4 × 480² (MHA) + 2 × 2 × 480² (FFN) + 3 × 2 × 480 (LN)
        ≈ 1.85M
      - 2 blocks ≈ 3.7M
      - Final norm + projection: 480 → 18 × 16 = 288 → 138K
      - Total ≈ 3.85M (vs MLP head's ~2.0M at the same shape)

    Roughly 2× the parameter count of the MLP head, well worth it for
    the structural improvement: the bottleneck is removed, and each
    forecast-position query independently selects relevant encoder states.
    """

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        num_channels: int,
        patch_len: int,
        *,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if n_patches <= 0:
            msg = f"n_patches must be positive, got {n_patches}"
            raise ValueError(msg)
        if num_channels <= 0 or patch_len <= 0:
            msg = (
                f"num_channels and patch_len must be positive "
                f"({num_channels}, {patch_len})"
            )
            raise ValueError(msg)
        if d_model % n_heads != 0:
            msg = (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                f"for CrossAttentionForecastHead"
            )
            raise ValueError(msg)
        self.n_patches = n_patches
        self.num_channels = num_channels
        self.patch_len = patch_len
        # Learnable forecast-position queries — one per future patch.
        # Each query carries its own representation of "what does
        # 'patch k future-steps ahead' look like" and is updated by
        # cross-attention to the encoder hidden states.
        self.queries = nn.Parameter(
            torch.randn(n_patches, d_model) * query_init_std,
        )
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_channels * patch_len)

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """`(B, T, d_model)` → `(B, n_patches, num_channels, patch_len)`.

        Args:
          encoder_hidden: typically `hidden_nc[:, 1:]` — the patch-token
            slice of the non-causal backbone output. Including the
            FORECAST token in the keys/values is harmless but redundant.

        Returns:
          (B, n_patches, num_channels, patch_len) — same shape as the
          MLP head's output, no downstream loss-code changes needed.
        """
        if encoder_hidden.dim() != 3:
            msg = (
                f"Expected (B, T, d_model), got shape {tuple(encoder_hidden.shape)}"
            )
            raise ValueError(msg)
        B = encoder_hidden.size(0)
        # Broadcast queries across batch.
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, H, d_model)
        # Stack of cross-attn + FFN blocks.
        for block in self.blocks:
            q = block(q, encoder_hidden)
        q = self.final_norm(q)
        # Project each query position to its patch values.
        out = self.proj(q)                                             # (B, H, C * patch_len)
        return out.view(B, self.n_patches, self.num_channels, self.patch_len)
