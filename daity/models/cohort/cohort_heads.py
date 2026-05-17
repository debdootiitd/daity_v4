"""Calendar-aware multi-horizon prediction heads for the cohort architecture.

Per-horizon learned queries cross-attend over each stock's full conditioned
sequence (CLS + patch tokens). Replaces the prior CLS-only bottleneck head
that fed only the FORECAST token to a shared MLP — that design discarded the
81 patch tokens the cross-attention conditioner enriched, killing intraday
signal (where short-term patches carry the relevant context).

Architecture mirrors `daity.models.heads.MultiHorizonScalarForecastHead`:
  - n_horizons learnable queries, each d_model.
  - K cross-attention blocks where Q = horizon queries, K/V = stock's
    (S = 1 + total_patches) conditioned tokens.
  - Final LayerNorm + per-horizon Linear(d_model -> 1).

Input: `conditioned_state (B, N, S, d_model)` — output of CrossAttentionConditioner.
Output: `(B, N, n_horizons)` — one scalar per (cohort, stock, horizon).
"""

from __future__ import annotations

import torch
from torch import nn

from daity.data.intraday_horizons import N_HORIZONS
from daity.models.heads import _CrossAttnBlock


class CohortHeads(nn.Module):
    """Per-horizon cross-attention heads.

    H learned queries × cross-attend over each stock's S tokens × project per
    horizon. Each horizon sees the entire stock context, not just CLS.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_horizons: int = N_HORIZONS,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        query_init_std: float = 0.02,
        enable_classifier: bool = False,
    ) -> None:
        super().__init__()
        if n_horizons <= 0:
            msg = f"n_horizons must be positive, got {n_horizons}"
            raise ValueError(msg)
        if d_model % n_heads != 0:
            msg = (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
            raise ValueError(msg)
        self.d_model = d_model
        self.n_horizons = n_horizons
        self.enable_classifier = enable_classifier
        # Learnable per-horizon queries — one per horizon. Each carries the
        # "what does horizon h look like" prior, updated by cross-attention.
        self.queries = nn.Parameter(
            torch.randn(n_horizons, d_model) * query_init_std,
        )
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        # Per-horizon scalar projection — Linear(d_model -> 1) shared across
        # (cohort, stock) since each horizon's query already carries
        # horizon-specific structure.
        self.proj = nn.Linear(d_model, 1)
        # Optional per-horizon classifier head: P(real_lr > threshold).
        # Shares the cross-attention conditioned features with the regression
        # head, only the final linear is separate. Adds n_horizons × d_model
        # parameters (~3840 for d_model=480, H=8). Output: logits.
        if enable_classifier:
            self.proj_clf = nn.Linear(d_model, 1)
        else:
            self.proj_clf = None

    def forward(self, conditioned: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """`(B, N, S, d_model)` → regression `(B, N, H)` or dict with both.

        If `enable_classifier` is False (default), returns the regression
        tensor directly for backward compatibility. If True, returns:
            {"reg": (B, N, H) log-return predictions,
             "clf": (B, N, H) classifier logits (BCEWithLogits-ready)}
        """
        if conditioned.dim() != 4:
            msg = (
                f"conditioned must be (B, N, S, d_model), got {tuple(conditioned.shape)}"
            )
            raise ValueError(msg)
        B, N, S, D = conditioned.shape
        # Flatten (B, N) into batch axis so each stock is an independent KV.
        kv = conditioned.view(B * N, S, D)                        # (B*N, S, D)
        q = self.queries.unsqueeze(0).expand(B * N, -1, -1).contiguous()
        for block in self.blocks:
            q = block(q, kv)                                      # (B*N, H, D)
        q = self.final_norm(q)
        reg = self.proj(q).squeeze(-1).view(B, N, self.n_horizons)  # (B, N, H)
        if self.proj_clf is None:
            return reg
        clf = self.proj_clf(q).squeeze(-1).view(B, N, self.n_horizons)  # (B, N, H)
        return {"reg": reg, "clf": clf}
