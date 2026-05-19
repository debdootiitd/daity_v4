"""Lightweight FT-Transformer + Set Transformer for cohort prediction.

Architecture:
  STAGE 1 (per-stock):
    FT-Transformer aggregates numerical + categorical features into a stock CLS embedding.
    Each numerical feature is a token (via projection); categorical → embedding.
    CLS token aggregates via multi-head self-attention.

  STAGE 2 (cross-stock):
    Set Transformer takes the N stock embeddings + market context tokens,
    self-attends across stocks (relative-value / portfolio interactions),
    emits conditioned per-stock representations.

  STAGE 3 (heads):
    - Regression: linear → (B, N, H) log-return predictions per horizon
    - Classification: linear → (B, N, H) win-probability logits
    - PortfolioHead (NEW): top-K weight vector with STE for Sharpe fine-tune

Param budget at default config: ~310K (vs 4.77M cohort transformer).
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al. 2021).

    Each numerical feature → linear projection to d_token → token.
    Each categorical feature → embedding to d_token → token.
    Prepend learnable CLS token.
    Multi-head self-attention layers process all tokens jointly.
    Final CLS embedding = stock representation.

    Input:
        num_feats: (B*N, F_num)  numerical features
        cat_feats: (B*N, F_cat)  long-int categorical features
    Output:
        cls_repr: (B*N, d_token)
    """

    def __init__(
        self,
        n_numerical: int,
        cat_class_sizes: list[int],
        d_token: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        token_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.n_numerical = n_numerical
        self.d_token = d_token
        self.n_categorical = len(cat_class_sizes)

        # Per-feature numerical tokenizer: each scalar → token via
        # x * w_i + b_i (where w_i, b_i are per-feature parameters)
        self.num_weight = nn.Parameter(torch.randn(n_numerical, d_token) * token_init_std)
        self.num_bias   = nn.Parameter(torch.randn(n_numerical, d_token) * token_init_std)

        # Categorical embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(n_classes, d_token) for n_classes in cat_class_sizes
        ])

        # Learnable CLS token (one for all stocks; broadcast at forward)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * token_init_std)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=d_token * ffn_ratio,
            dropout=dropout, batch_first=True,
            norm_first=True,  # pre-norm: more stable
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_token)

    def forward(self, num_feats: torch.Tensor, cat_feats: torch.Tensor) -> torch.Tensor:
        """
        num_feats: (B*N, F_num) float
        cat_feats: (B*N, F_cat) long
        Returns: (B*N, d_token) — CLS embedding per row
        """
        bn = num_feats.shape[0]

        # Numerical: (B*N, F_num) → (B*N, F_num, d_token)
        # x_i * w_i + b_i element-wise (broadcast over batch)
        num_tokens = num_feats.unsqueeze(-1) * self.num_weight + self.num_bias

        # Categorical: each cat_feats[:, i] is (B*N,) of int IDs → (B*N, d_token)
        if self.n_categorical > 0:
            cat_tokens = torch.stack(
                [emb(cat_feats[:, i]) for i, emb in enumerate(self.cat_embeds)],
                dim=1,
            )  # (B*N, F_cat, d_token)
        else:
            cat_tokens = num_feats.new_zeros((bn, 0, self.d_token))

        # Prepend CLS
        cls = self.cls_token.expand(bn, -1, -1)  # (B*N, 1, d_token)
        tokens = torch.cat([cls, num_tokens, cat_tokens], dim=1)
        # (B*N, 1 + F_num + F_cat, d_token)

        # Transformer
        tokens = self.encoder(tokens)
        tokens = self.final_norm(tokens)

        # Return CLS (position 0)
        return tokens[:, 0]  # (B*N, d_token)


class SetTransformer(nn.Module):
    """Cross-stock set transformer: N stock reprs + market context tokens.

    Each stock's representation is enriched by attending to other stocks
    in the cohort + regime feature tokens. Enables relative-value reasoning.

    Input:
        stock_reprs: (B, N, d_model)
        regime_tokens: (B, R, d_model) or None
    Output:
        conditioned: (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        stock_reprs: torch.Tensor,
        regime_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if regime_tokens is not None:
            tokens = torch.cat([stock_reprs, regime_tokens], dim=1)
        else:
            tokens = stock_reprs
        tokens = self.encoder(tokens)
        tokens = self.final_norm(tokens)
        # Drop regime tokens, return only stock representations
        N = stock_reprs.shape[1]
        return tokens[:, :N]


class PortfolioHead(nn.Module):
    """Top-K weight vector head with Straight-Through Estimator.

    Forward: hard top-K by score, equal-weighted (sum=1, K non-zero entries).
    Backward: gradient flows through softmax(scores / tau) for smooth training.
    """

    def __init__(self, d_model: int, K: int = 5, tau: float = 0.005) -> None:
        super().__init__()
        self.K = K
        self.tau = tau
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, conditioned: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        conditioned: (B, N, d_model)
        valid_mask:  (B, N) bool, True = tradeable
        Returns: w (B, N) with K non-zero entries summing to 1 (per b).
        """
        scores = self.score_proj(conditioned).squeeze(-1)  # (B, N)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -1e9)
        # Soft weights (for gradient)
        soft_w = F.softmax(scores / max(self.tau, 1e-6), dim=-1)
        # Hard top-K equal-weight (for forward / inference)
        _, top_k_idx = torch.topk(scores, self.K, dim=-1)  # (B, K)
        hard_w = torch.zeros_like(scores)
        hard_w.scatter_(-1, top_k_idx, 1.0 / self.K)
        # Straight-Through: forward=hard, backward=soft
        w = hard_w.detach() + soft_w - soft_w.detach()
        return w


class LightweightCohortModel(nn.Module):
    """Full model: FT-Transformer + Set Transformer + heads."""

    def __init__(
        self,
        n_numerical: int = 50,
        n_stocks: int = 205,
        n_sectors: int = 15,
        n_regime_feats: int = 10,
        n_horizons: int = 8,
        d_ft: int = 32,
        d_model: int = 128,
        ft_layers: int = 2,
        ft_heads: int = 4,
        set_layers: int = 2,
        set_heads: int = 8,
        enable_portfolio_head: bool = False,
        portfolio_K: int = 5,
        portfolio_tau: float = 0.005,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons

        # Per-stock FT-Transformer
        # Categorical features: stock_id + sector_id (2 categoricals)
        self.ft = FTTransformer(
            n_numerical=n_numerical,
            cat_class_sizes=[n_stocks, n_sectors],
            d_token=d_ft,
            n_layers=ft_layers,
            n_heads=ft_heads,
            dropout=dropout,
        )
        # Project per-stock FT-Transformer CLS → d_model
        self.stock_proj = nn.Linear(d_ft, d_model)

        # Regime feature tokenizer (each regime scalar → token in d_model)
        self.regime_proj = nn.Linear(1, d_model, bias=False)
        self.regime_bias = nn.Parameter(torch.randn(n_regime_feats, d_model) * 0.02)

        # Cross-stock Set Transformer
        self.set_transformer = SetTransformer(
            d_model=d_model,
            n_layers=set_layers,
            n_heads=set_heads,
            dropout=dropout,
        )

        # Multi-task heads
        self.regression_head = nn.Linear(d_model, n_horizons)
        self.classifier_head = nn.Linear(d_model, n_horizons)

        # Portfolio head (used in fine-tune phase)
        if enable_portfolio_head:
            self.portfolio_head = PortfolioHead(d_model, K=portfolio_K, tau=portfolio_tau)
        else:
            self.portfolio_head = None

    def forward(
        self,
        num_feats: torch.Tensor,        # (B, N, F_num)
        cat_feats: torch.Tensor,        # (B, N, F_cat)  long
        regime_feats: torch.Tensor | None = None,  # (B, R) or None
        valid_mask: torch.Tensor | None = None,    # (B, N) bool
    ) -> dict[str, torch.Tensor]:
        B, N, F_num = num_feats.shape

        # Per-stock FT-Transformer
        nf = num_feats.reshape(B * N, F_num)
        cf = cat_feats.reshape(B * N, -1)
        stock_emb = self.ft(nf, cf)                                       # (B*N, d_ft)
        stock_emb = self.stock_proj(stock_emb)                            # (B*N, d_model)
        stock_emb = stock_emb.reshape(B, N, -1)                           # (B, N, d_model)

        # Regime tokens (broadcast linear + per-feature bias)
        if regime_feats is not None:
            # regime_feats: (B, R) → (B, R, 1) → (B, R, d_model)
            regime_tokens = self.regime_proj(regime_feats.unsqueeze(-1)) + self.regime_bias.unsqueeze(0)
        else:
            regime_tokens = None

        # Cross-stock Set Transformer
        conditioned = self.set_transformer(stock_emb, regime_tokens)      # (B, N, d_model)

        # Heads
        out = {
            "conditioned": conditioned,
            "reg": self.regression_head(conditioned),                     # (B, N, H)
            "clf": self.classifier_head(conditioned),                     # (B, N, H)
        }
        if self.portfolio_head is not None:
            out["portfolio_w"] = self.portfolio_head(conditioned, valid_mask)
        return out
