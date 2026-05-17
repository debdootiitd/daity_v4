"""Market Context Builder for the cohort architecture.

Takes per-stock CLS tokens (one per stock in the cohort) + regime tokens
(NIFTY/breadth/sector means/vol) + calendar tokens, runs a small
self-attention transformer over them, and emits an enriched market
representation each stock can attend over.

Design rationale:
  - Stock CLS are *uninformed* about the universe: each was computed in
    isolation. Self-attention here lets every stock see every other stock,
    plus the regime context.
  - Regime tokens are produced from market-wide aggregate features the
    cohort dataset pre-computes. Each is a learned projection of one
    aggregate (NIFTY 5d return, breadth %, sector-mean returns, vol),
    plus a learned "regime type" embedding to tell them apart.
  - The market context is per-cohort, not per-stock. The downstream
    cross-attention conditioner uses it as keys/values for every stock.

Shapes:
  Input:
    stock_cls  : (n_cohorts, n_stocks, d_model)  — from StockContextEncoder
    regime_feats: (n_cohorts, n_regime_feats)    — market-wide scalars
  Output:
    market_ctx : (n_cohorts, n_stocks + n_regime_tokens, d_model)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from daity.models.cohort.top_k_attn import TopKSelfAttnBlock


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN block (no RoPE — market tokens are
    set-like, position has no temporal meaning across stocks).

    The market context is a SET of (stocks, regime, calendar) tokens.
    Permutation invariance would be technically wrong if we cared about
    a fixed stock-ordering — but with explicit stock_id embedding upstream,
    the model has identity info regardless of order. So a vanilla self-
    attention block with no positional encoding is correct here.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int = 2) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm_attn(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm_ffn(x))


class MarketContextBuilder(nn.Module):
    """Stack stock CLS + regime + calendar tokens, run self-attention.

    Construction:
        n_regime_feats: how many scalar regime features the dataset
            produces (e.g., NIFTY 5d return, breadth, sector means × K,
            vol). The builder projects each through a shared 1→d_model
            Linear and adds a learned "regime type" embedding so different
            regime features stay distinguishable.

    Forward:
        stock_cls: (B, N, d_model) — one CLS per stock.
        regime_feats: (B, n_regime_feats) — float32 scalars.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        n_regime_feats: int,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if n_regime_feats < 0:
            msg = f"n_regime_feats must be >= 0, got {n_regime_feats}"
            raise ValueError(msg)
        self.d_model = d_model
        self.n_regime_feats = n_regime_feats

        if n_regime_feats > 0:
            # Per-feature scalar projection (shared across features) +
            # learnable per-feature "type" embedding (so the model can
            # tell NIFTY return apart from breadth, etc.).
            self.regime_scalar_proj = nn.Linear(1, d_model)
            self.regime_type_embed = nn.Parameter(
                torch.randn(n_regime_feats, d_model) * 0.02,
            )
        else:
            self.regime_scalar_proj = None
            self.regime_type_embed = None

        if top_k is not None and top_k > 0:
            self.blocks = nn.ModuleList([
                TopKSelfAttnBlock(d_model, n_heads, ffn_ratio, top_k=top_k)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                _SelfAttnBlock(d_model, n_heads, ffn_ratio)
                for _ in range(n_layers)
            ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        stock_cls: torch.Tensor,
        regime_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(B, N, d_model)` + `(B, n_regime_feats)` → `(B, N + n_regime, d_model)`.

        If `regime_feats` is None (or n_regime_feats == 0), only stocks are
        used. Useful for ablation runs.
        """
        if stock_cls.dim() != 3:
            msg = f"stock_cls must be (B, N, d_model), got {tuple(stock_cls.shape)}"
            raise ValueError(msg)
        B = stock_cls.size(0)

        seq = [stock_cls]
        if regime_feats is not None and self.n_regime_feats > 0:
            if regime_feats.shape != (B, self.n_regime_feats):
                msg = (
                    f"regime_feats must be (B={B}, n_regime_feats={self.n_regime_feats}), "
                    f"got {tuple(regime_feats.shape)}"
                )
                raise ValueError(msg)
            # (B, n_regime) → (B, n_regime, 1) → project to (B, n_regime, d_model)
            r = self.regime_scalar_proj(regime_feats.unsqueeze(-1))
            r = r + self.regime_type_embed.unsqueeze(0)            # (B, n_regime, d_model)
            seq.append(r)

        x = torch.cat(seq, dim=1)                                  # (B, N + n_regime, d_model)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
