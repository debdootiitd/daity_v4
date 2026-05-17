"""Stock Context Encoder for the cohort architecture.

Wraps `MultiResTokenizer + Backbone` with:
  - anchor-time embedding added to the FORECAST token (position 0)
  - sector embedding added to FORECAST token (semantic cluster info)
  - stock_id embedding added to patch tokens 1..T (NOT FORECAST)

This is the same encoder architecture that worked for the 50M champion,
but with the bugfixes already in place:
  - HIGH #3: stock_id only on patch tokens, never on the FORECAST/CLS token
    that downstream heads consume (avoids per-symbol bias dominating
    cross-sectional alpha signal).

The output is a per-stock hidden tensor `(B, 1+T, d_model)` where:
  - position 0 = "stock CLS" — anchor-time-aware, sector-aware, used to
    build the market context.
  - positions 1..T = patch hiddens — stock-aware, used for the cross-
    attention conditioner.

Reuses without modification:
  - daity.models.tokenizer.MultiResTokenizer (channel_independent=True)
  - daity.models.backbone.Backbone (causal=False)

So the existing 50M champion checkpoint state can be loaded for warm-start
later if we want curriculum: champion → cohort fine-tune.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import torch
from torch import nn

from daity.models.backbone import Backbone
from daity.models.tokenizer import MultiResTokenizer


def anchor_time_features(anchor_ts: torch.Tensor) -> torch.Tensor:
    """Compute calendar-feature vector for each anchor.

    Input `anchor_ts`: `(B,)` int64 tensor of UTC microseconds since epoch.
    Output: `(B, 6)` float32 tensor:
        [sin(2π·time_of_day/day), cos(...),
         sin(2π·day_of_week/7),  cos(...),
         sin(2π·day_of_year/365), cos(...)]

    Time-of-day uses IST minute (anchor is shifted +5h30 to IST). Stationary
    features the model can absorb without learning a calendar lookup.
    """
    if anchor_ts.dim() != 1:
        msg = f"anchor_ts must be 1-D (B,), got shape {tuple(anchor_ts.shape)}"
        raise ValueError(msg)
    device = anchor_ts.device
    # Convert microseconds → fractional days since epoch (1970-01-01 UTC = Thu).
    # IST is UTC+5:30.
    SECONDS_PER_DAY = 86_400.0
    us = anchor_ts.to(torch.float64)
    sec = us / 1_000_000.0
    ist_sec = sec + 5.5 * 3600.0
    day_frac = (ist_sec.fmod(SECONDS_PER_DAY)) / SECONDS_PER_DAY   # in [0,1)
    days = (ist_sec / SECONDS_PER_DAY).floor()
    # 1970-01-01 was a Thursday (weekday=3). Mon=0..Sun=6.
    dow = ((days + 3) % 7) / 7.0
    doy = (days % 365.25) / 365.25
    two_pi = 2.0 * math.pi
    feats = torch.stack([
        torch.sin(two_pi * day_frac),
        torch.cos(two_pi * day_frac),
        torch.sin(two_pi * dow),
        torch.cos(two_pi * dow),
        torch.sin(two_pi * doy),
        torch.cos(two_pi * doy),
    ], dim=-1).to(torch.float32).to(device)                         # (B, 6)
    return feats


class StockContextEncoder(nn.Module):
    """Per-stock encoder: OHLCV multi-scale → stock state tokens.

    Forward signature:
        forward(
            x_by_scale: dict[scale, (B, L_scale, C)],
            *,
            stock_ids: (B,) long,
            sector_ids: (B,) long,
            anchor_ts: (B,) int64 (UTC microseconds),
        ) -> (B, 1 + total_patches, d_model)

    Where:
        B = cohort size (number of stocks at one anchor)
        d_model = stock state dimensionality
        position 0 = "stock CLS" with anchor-time + sector embedding
        positions 1..T = patch tokens with stock_id embedding (broadcast)
    """

    def __init__(
        self,
        *,
        scales: tuple[str, ...],
        num_channels: int,
        d_model: int,
        n_layers: int = 12,
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
    ) -> None:
        super().__init__()
        if n_stocks <= 0 or n_sectors <= 0:
            msg = f"n_stocks and n_sectors must be positive ({n_stocks}, {n_sectors})"
            raise ValueError(msg)
        self.d_model = d_model
        self.tokenizer = MultiResTokenizer(
            scales=scales,
            num_channels=num_channels,
            d_model=d_model,
            patch_len=patch_len,
            patch_stride=patch_stride,
            revin_affine=revin_affine,
            channel_independent=True,            # Phase 2.5+ default
            channel_pool=channel_pool,
        )
        self.backbone = Backbone(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )

        # Stock ID embedding: added to ALL token positions.
        #
        # Original HIGH#3 fix (post-v78) restricted this to patch tokens only,
        # to prevent per-stock bias dominating the CLS that fed the regression
        # head. But repr-probe diagnostic on v5 showed CLS representations
        # collapsed to effective rank 1.03 — the model has no per-stock cue
        # at the CLS, so the entire (stock, anchor) population converges to
        # a single direction. With v5+'s cross-sectionally demeaned loss the
        # original v78 bias problem cannot recur (loss is shift-invariant), so
        # it is safe to broadcast stock_id everywhere including position 0.
        self.stock_embed = nn.Embedding(n_stocks, d_model)
        nn.init.normal_(self.stock_embed.weight, std=0.02)

        # Sector embedding: added to the CLS token (low-dim semantic
        # cluster info that helps cross-stock attention pool by industry).
        self.sector_embed = nn.Embedding(n_sectors, d_model)
        nn.init.normal_(self.sector_embed.weight, std=0.02)

        # Anchor-time MLP: 6-dim calendar features → d_model. Added to
        # the CLS token only (the per-bar time-of-day signal is already
        # baked into the input via min_of_day_sin/cos channels).
        self.anchor_time_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x_by_scale: dict[str, torch.Tensor],
        *,
        stock_ids: torch.Tensor,
        sector_ids: torch.Tensor,
        anchor_ts: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_by_scale: per-scale (B, L_scale, C) input bars.
            stock_ids: (B,) long, in [0, n_stocks).
            sector_ids: (B,) long, in [0, n_sectors).
            anchor_ts: (B,) int64 microseconds since UTC epoch.

        Returns:
            (B, 1 + total_patches, d_model) hidden states.
            Position 0 = stock CLS with anchor+sector embedding.
            Positions 1..T = patch hiddens with stock_id embedding.
        """
        tokens = self.tokenizer(x_by_scale)                       # (B, 1+T, d_model)
        B, S, _ = tokens.shape

        # Stock-id embedding broadcast to ALL S positions including CLS.
        stock_e = self.stock_embed(stock_ids).unsqueeze(1)        # (B, 1, d_model)
        stock_e = stock_e.expand(B, S, self.d_model)

        # CLS-only additions: anchor-time + sector.
        time_feats = anchor_time_features(anchor_ts)              # (B, 6)
        cls_only_add = self.anchor_time_mlp(time_feats) + self.sector_embed(sector_ids)
        cls_only_add = cls_only_add.unsqueeze(1)                  # (B, 1, d_model)

        add = stock_e.clone()
        add[:, :1, :] = add[:, :1, :] + cls_only_add
        tokens = tokens + add

        return self.backbone(tokens, causal=False)                # (B, S, d_model)
