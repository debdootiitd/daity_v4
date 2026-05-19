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
        enable_cohort_bias: bool = False,
        n_regime_feats: int = 0,
        cohort_bias_hidden: int = 32,
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
        self.enable_cohort_bias = enable_cohort_bias
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
        # Optional cohort-bias head: predicts the cohort-level mean per horizon.
        # ARCHITECTURE: takes regime_feats DIRECTLY (small MLP), bypassing the
        # per-stock pipeline. Predicting next-day cohort mean is a market-level
        # problem — mean-pooling 200 per-stock conditioned features dilutes the
        # cohort signal. Direct path keeps the signal clean and trains fast.
        #
        # Design details:
        #   - NO LayerNorm at input. LN across mixed-scale features is wrong:
        #     it normalizes across the 10 feature values per sample, so the
        #     dominant-magnitude feature (e.g. min_of_day_cos ~[-1, +1])
        #     hijacks the mean/std and washes out the small-magnitude signal
        #     features (cohort_mean_ret_30m ~3e-3). Instead, we apply a fixed
        #     per-feature pre-scale (so each feature has roughly unit std at
        #     init), then let Adagrad adapt per-weight LR thereafter.
        #   - Final Linear zero-initialized: bias head starts at 0 contribution.
        #     The alpha head sees clean targets at init; bias learns up from 0.
        # Input: regime_feats (B, n_regime_feats); Output: (B, H).
        if enable_cohort_bias:
            if n_regime_feats <= 0:
                msg = (
                    "enable_cohort_bias=True requires n_regime_feats > 0; the "
                    "bias head now reads regime_feats directly."
                )
                raise ValueError(msg)
            # Hardcoded per-feature inverse-std priors (computed from empirical
            # ranges). Order MUST match REGIME_FEAT_NAMES in cohort_dataset.py.
            # Provides ~unit-scale input to the MLP regardless of feature units.
            # We tolerate extra features by extending with 1.0 (no-op scale).
            DEFAULT_INV_STD = [
                67.0,   # univ_5d_return   (~0.015 std → ×67 → ~1)
                7.0,    # univ_breadth_1d  (~0.15 std → ×7)
                200.0,  # univ_xs_vol      (~0.005)
                125.0,  # univ_1d_return   (~0.008)
                333.0,  # sector_disp_1d   (~0.003)
                333.0,  # cohort_mean_ret_30m (~0.003)
                200.0,  # cohort_mean_ret_1h  (~0.005)
                100.0,  # cohort_mean_ret_today_so_far (~0.010)
                1.0,    # anchor_min_of_day_sin (already in [-1, +1])
                1.0,    # anchor_min_of_day_cos (already in [-1, +1])
            ]
            inv_std = DEFAULT_INV_STD[:n_regime_feats]
            if len(inv_std) < n_regime_feats:
                inv_std = inv_std + [1.0] * (n_regime_feats - len(inv_std))
            self.register_buffer(
                "regime_feat_inv_std",
                torch.tensor(inv_std, dtype=torch.float32),
            )
            # Per-horizon MLPs: each horizon gets its own small MLP from
            # regime_feats. Avoids the shared-trunk gradient-conflict problem
            # where different horizons want different feature combinations
            # (e.g. intraday_30m wants short-window cohort_mean; next_day_eod
            # wants longer 5d univ_return). Each MLP zero-inits its final
            # Linear so the head starts at 0 contribution.
            self.cohort_bias_mlps = nn.ModuleList([
                self._build_per_horizon_mlp(n_regime_feats, cohort_bias_hidden)
                for _ in range(n_horizons)
            ])
            # Legacy attribute kept for backward-compat in optimizer-group
            # detection (online_walk_forward groups params by name substr).
            self.cohort_bias_mlp = None
        else:
            self.cohort_bias_mlp = None
            self.cohort_bias_mlps = None
            self.regime_feat_inv_std = None

    @staticmethod
    def _build_per_horizon_mlp(in_dim: int, hidden: int) -> nn.Sequential:
        """Small MLP (in_dim -> hidden -> 1) with zero-init final layer."""
        final = nn.Linear(hidden, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            final,
        )

    def forward(
        self,
        conditioned: torch.Tensor,
        *,
        regime_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """`(B, N, S, d_model)` → regression `(B, N, H)` or dict with both.

        Args:
            conditioned:  per-stock conditioned tokens, shape (B, N, S, d_model)
            regime_feats: market-aggregate scalars, shape (B, n_regime_feats).
                Required when `enable_cohort_bias=True`. Feeds the cohort-bias
                MLP directly (separate branch — no per-stock dilution).

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
        out: dict[str, torch.Tensor] = {"reg": reg}
        if self.proj_clf is not None:
            clf = self.proj_clf(q).squeeze(-1).view(B, N, self.n_horizons)  # (B, N, H)
            out["clf"] = clf
        if self.cohort_bias_mlps is not None:
            if regime_feats is None:
                msg = (
                    "cohort_bias head requires regime_feats (B, n_regime_feats)"
                )
                raise ValueError(msg)
            # Direct MLP branch — keeps cohort_mean signal undiluted. Pre-scale
            # each feature to ~unit std so the MLP doesn't see input dominated
            # by the largest-magnitude feature.
            scaled = regime_feats * self.regime_feat_inv_std.to(regime_feats.dtype)
            # Per-horizon MLPs: each head produces (B, 1); stack to (B, H).
            per_h = [mlp(scaled) for mlp in self.cohort_bias_mlps]
            cohort_bias = torch.cat(per_h, dim=-1)  # (B, H)
            out["cohort_bias"] = cohort_bias
        # Backward compat: if no extras enabled, return just the regression tensor
        if self.proj_clf is None and self.cohort_bias_mlps is None:
            return reg
        return out
