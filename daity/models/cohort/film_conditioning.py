"""FiLM-style cohort conditioning — replaces the 17M-param MarketContextBuilder
+ CrossAttentionConditioner stack (which empirical analysis on v5 showed
collapses to uniform attention).

Why FiLM:
  ML architect review (2026-05-16) on v5/step=1750 found:
    - market_context self-attn entropy ratio = 0.999-1.000 (uniform) at layers 1-3
    - cross-attention CLS-to-regime weight = 0.013 (sub-uniform 1/M=0.015);
      per-stock std = 1e-6 across 198 stocks → zero per-stock conditioning
    - 17M of 52M total params are doing the work of a single mean-pool +
      uniform attention.
  FiLM (Perez et al. 2018) injects regime + cohort-summary via per-token
  affine (gain, shift) modulation. Forces per-stock variance (because
  shift/gain is broadcast over patch tokens but each stock starts from
  its own CLS) and uses ~1M params instead of 17M.

Design:
  Cohort summary  = mean(stock_CLS over N stocks) + regime_proj(regime_feats)
                  ∈ (B, d_model)
  FiLM params     = MLP(cohort_summary) → (B, 2*d_model)
                    split into (gamma, beta) each (B, d_model)
  conditioned[i, t, :] = (1 + gamma) * stock_state[i, t, :] + beta
  Applied to every (stock i, token t) — including patch tokens, so
  the heads see the modulated representation.

Output is same shape as input: (B, N, S, d_model).
"""

from __future__ import annotations

import torch
from torch import nn


class FiLMCohortConditioner(nn.Module):
    """Cohort-level FiLM conditioner.

    Args:
        d_model: hidden size of the stock state and the FiLM modulation.
        n_regime_feats: number of scalar regime features (3 in current pipeline).
        hidden_ratio: FFN expansion inside the FiLM-param MLP. Default 2 keeps
            params small (~2 * d_model^2 ≈ 0.5M at d=480).
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_regime_feats: int = 0,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)
        self.d_model = d_model
        self.n_regime_feats = n_regime_feats

        # Project regime scalars (B, R) → (B, d_model) so they live in the
        # cohort-summary space.
        if n_regime_feats > 0:
            self.regime_proj = nn.Sequential(
                nn.Linear(n_regime_feats, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.regime_proj = None

        # MLP that emits FiLM params from cohort summary.
        # Output dim = 2 * d_model (gamma, beta).
        hidden = hidden_ratio * d_model
        self.film_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        # Initialize the FiLM MLP's final layer near zero so conditioning
        # starts as identity (gamma=0, beta=0).
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    def forward(
        self,
        stock_state: torch.Tensor,
        regime_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(B, N, S, d_model)` + `(B, R)` → `(B, N, S, d_model)`.

        - Compute cohort summary = mean(stock CLS) + regime_proj(regime_feats).
        - Generate per-cohort FiLM (gamma, beta).
        - Apply per-stock-per-token modulation: out = (1+gamma) * x + beta.
        """
        if stock_state.dim() != 4:
            msg = f"stock_state must be (B, N, S, d_model), got {tuple(stock_state.shape)}"
            raise ValueError(msg)
        B, N, S, D = stock_state.shape

        # Cohort summary: mean of stock CLS tokens (position 0) across N stocks.
        cohort_summary = stock_state[:, :, 0, :].mean(dim=1)          # (B, d_model)
        if regime_feats is not None and self.regime_proj is not None:
            if regime_feats.shape != (B, self.n_regime_feats):
                msg = (
                    f"regime_feats must be (B={B}, n_regime_feats={self.n_regime_feats}), "
                    f"got {tuple(regime_feats.shape)}"
                )
                raise ValueError(msg)
            cohort_summary = cohort_summary + self.regime_proj(regime_feats)

        film = self.film_mlp(cohort_summary)                          # (B, 2*d_model)
        gamma, beta = film.chunk(2, dim=-1)                           # each (B, d_model)
        # Broadcast over N and S.
        gamma = gamma.view(B, 1, 1, D)
        beta  = beta.view(B, 1, 1, D)
        return (1.0 + gamma) * stock_state + beta
