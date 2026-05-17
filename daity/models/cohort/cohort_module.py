"""Top-level cohort model: stitches StockContextEncoder + MarketContextBuilder
+ CrossAttentionConditioner + CohortHeads together.

This is a pure `nn.Module` — the training loop (Lightning module / walk-
forward driver) is a separate file.

Input contract (one cohort = one anchor):
    batch = {
        'x_by_scale': {scale: (B, N, L_scale, C)},   # multi-scale OHLCV per stock
        'stock_ids':  (B, N) long,
        'sector_ids': (B, N) long,
        'anchor_ts':  (B,) int64 microseconds (one anchor per cohort),
        'regime_feats': (B, n_regime_feats) float32   # optional, may be None
    }
where B = number of cohorts in this minibatch (often 1), N = stocks per cohort.

Output:
    pred: (B, N, n_horizons) — predicted cumulative log returns.

Loss is computed elsewhere (training driver): cross-sectional regression +
listwise rank, masked by per-horizon validity.
"""

from __future__ import annotations

from __future__ import annotations

import torch
from torch import nn

from daity.data.intraday_horizons import N_HORIZONS
from daity.models.cohort.cohort_heads import CohortHeads
from daity.models.cohort.cross_attn import CrossAttentionConditioner
from daity.models.cohort.market_context import MarketContextBuilder
from daity.models.cohort.stock_encoder import StockContextEncoder


class CohortModel(nn.Module):
    """The full cohort transformer.

    Construction:
        scales, num_channels, d_model, ... : passed to StockContextEncoder
        n_stocks, n_sectors                : universe + sector counts
        n_market_layers, n_cross_layers    : depth of market self-attn and cross-attn
        n_regime_feats                     : number of scalar market-aggregate features

    Forward:
        batch: dict — see file docstring.
    """

    def __init__(
        self,
        *,
        # Stock encoder
        scales: tuple[str, ...],
        num_channels: int,
        d_model: int = 480,
        stock_enc_layers: int = 12,
        n_heads: int = 8,
        ffn_ratio: int = 4,
        patch_len: int = 16,
        patch_stride: int = 8,
        max_seq_len: int = 4096,
        n_stocks: int,
        n_sectors: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        revin_affine: bool = False,
        channel_pool: str = "mean",
        # Market + cross-attn
        n_market_layers: int = 4,
        n_cross_layers: int = 4,
        market_n_heads: int = 8,
        market_ffn_ratio: int = 2,
        cross_n_heads: int = 8,
        cross_ffn_ratio: int = 2,
        n_regime_feats: int = 0,
        market_top_k: int | None = None,
        cross_top_k: int | None = None,
        # Heads (per-horizon cross-attention)
        head_n_layers: int = 2,
        head_n_heads: int = 8,
        head_ffn_ratio: int = 2,
        n_horizons: int = N_HORIZONS,
        # Sector classification auxiliary head (set n_sector_classes > 0 to enable)
        n_sector_classes: int = 0,
        # Contrastive projection head: outputs (B, N, contrastive_dim) when > 0,
        # for use by InfoNCE in the loss. Operates on the encoder's raw CLS
        # (pre-market-context) so the encoder is directly forced to organize.
        contrastive_dim: int = 0,
        # Win-rate classifier head: per-horizon P(real_lr > threshold).
        # When True, the model emits both regression and classifier logits.
        # Threshold is set in the loss / strategy layer, not the model.
        enable_classifier_head: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_horizons = n_horizons

        self.stock_encoder = StockContextEncoder(
            scales=scales,
            num_channels=num_channels,
            d_model=d_model,
            n_layers=stock_enc_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            patch_len=patch_len,
            patch_stride=patch_stride,
            max_seq_len=max_seq_len,
            n_stocks=n_stocks,
            n_sectors=n_sectors,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            revin_affine=revin_affine,
            channel_pool=channel_pool,
        )
        self.market_context = MarketContextBuilder(
            d_model=d_model,
            n_layers=n_market_layers,
            n_heads=market_n_heads,
            ffn_ratio=market_ffn_ratio,
            n_regime_feats=n_regime_feats,
            top_k=market_top_k,
        )
        self.cross_attn = CrossAttentionConditioner(
            d_model=d_model,
            n_layers=n_cross_layers,
            n_heads=cross_n_heads,
            ffn_ratio=cross_ffn_ratio,
            top_k=cross_top_k,
        )
        self.heads = CohortHeads(
            d_model=d_model,
            n_horizons=n_horizons,
            n_layers=head_n_layers,
            n_heads=head_n_heads,
            ffn_ratio=head_ffn_ratio,
            enable_classifier=enable_classifier_head,
        )
        # Optional sector classification head (operates on the encoder's
        # stock CLS, BEFORE market context — so it directly supervises the
        # per-stock representation that the architecture relies on).
        if n_sector_classes > 0:
            self.sector_classifier = nn.Linear(d_model, n_sector_classes)
        else:
            self.sector_classifier = None

        # Optional contrastive projection head (SimCLR-style two-layer MLP).
        # Operates on encoder's stock CLS pre-market-context so the loss
        # directly diversifies the encoder representation. Output is L2-
        # normalized inside the loss.
        if contrastive_dim > 0:
            self.contrastive_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_dim),
            )
        else:
            self.contrastive_head = None

    def forward(self, batch: dict) -> torch.Tensor | tuple:
        """Run the full cohort pipeline.

        Returns:
          - `(B, N, n_horizons)` predicted log returns (default), OR
          - dict `{pred, sector_logits, contrastive_embeds}` when either of
            the auxiliary heads is enabled. Keys that are `None` are omitted.
        """
        x_by_scale: dict[str, torch.Tensor] = batch["x_by_scale"]
        stock_ids:  torch.Tensor = batch["stock_ids"]                  # (B, N)
        sector_ids: torch.Tensor = batch["sector_ids"]                 # (B, N)
        anchor_ts:  torch.Tensor = batch["anchor_ts"]                  # (B,)
        regime_feats: torch.Tensor | None = batch.get("regime_feats")  # (B, R) or None

        # Validate B & N consistency.
        if stock_ids.dim() != 2:
            msg = f"stock_ids must be (B, N), got {tuple(stock_ids.shape)}"
            raise ValueError(msg)
        B, N = stock_ids.shape
        if sector_ids.shape != (B, N):
            msg = (
                f"sector_ids must match stock_ids shape (B={B}, N={N}); "
                f"got {tuple(sector_ids.shape)}"
            )
            raise ValueError(msg)
        if anchor_ts.shape != (B,):
            msg = f"anchor_ts must be (B={B},), got {tuple(anchor_ts.shape)}"
            raise ValueError(msg)

        # --- (A) Stock encoding: flatten (B, N) into one batch axis ----
        # Each scale: (B, N, L, C) → (B*N, L, C).
        x_flat: dict[str, torch.Tensor] = {}
        for sc, v in x_by_scale.items():
            if v.dim() != 4:
                msg = (
                    f"x_by_scale[{sc!r}] must be (B, N, L, C), got {tuple(v.shape)}"
                )
                raise ValueError(msg)
            if v.shape[:2] != (B, N):
                msg = (
                    f"x_by_scale[{sc!r}] first two dims must match (B={B}, N={N}); "
                    f"got {tuple(v.shape)}"
                )
                raise ValueError(msg)
            x_flat[sc] = v.reshape(B * N, v.size(2), v.size(3))
        sids_flat   = stock_ids.reshape(B * N)
        secs_flat   = sector_ids.reshape(B * N)
        anchor_flat = anchor_ts.unsqueeze(1).expand(B, N).reshape(B * N)

        stock_state = self.stock_encoder(
            x_flat,
            stock_ids=sids_flat,
            sector_ids=secs_flat,
            anchor_ts=anchor_flat,
        )                                                              # (B*N, 1+T, d)
        S = stock_state.size(1)
        D = stock_state.size(2)
        stock_state = stock_state.view(B, N, S, D)                     # (B, N, 1+T, d)

        # --- (B) Market context: stack stock CLS + regime tokens, self-attn ----
        stock_cls = stock_state[:, :, 0, :]                            # (B, N, d)

        # Auxiliary outputs on the ENCODER's CLS (pre-market-context) so they
        # directly supervise the encoder's per-stock representation.
        sector_logits = (
            self.sector_classifier(stock_cls)
            if self.sector_classifier is not None else None
        )
        contrastive_embeds = (
            self.contrastive_head(stock_cls)                           # (B, N, c_dim)
            if self.contrastive_head is not None else None
        )

        market_ctx = self.market_context(stock_cls, regime_feats)      # (B, N+R, d)

        # --- (C) Cross-attention: each stock attends over market_ctx ----
        conditioned = self.cross_attn(stock_state, market_ctx)         # (B, N, 1+T, d)

        # --- (D) Heads: per-horizon queries cross-attend over each stock's
        # full conditioned sequence (CLS + patch tokens). ----
        head_out = self.heads(conditioned)
        # head_out is either a Tensor (B, N, H) when classifier is disabled,
        # or a dict {"reg": ..., "clf": ...} when enabled.
        if isinstance(head_out, dict):
            pred = head_out["reg"]
            clf_logits = head_out.get("clf")
        else:
            pred = head_out
            clf_logits = None
        if (sector_logits is not None or contrastive_embeds is not None
                or clf_logits is not None):
            return {
                "pred": pred,
                "clf_logits": clf_logits,
                "sector_logits": sector_logits,
                "contrastive_embeds": contrastive_embeds,
            }
        return pred
