"""Phase 3X — TradeableHead: classifier on top of the FORECAST hidden.

Two architectures:

* `TradeableHead` (legacy MLP): 2-layer per-symbol MLP. Each symbol scored
  in isolation. Worked but cap-limited — see ML-scientist diagnosis on
  cross-sectional ranking blind spot.

* `CrossSectionTradeableHead` (v2): cross-symbol Transformer head. Reads
  ALL symbols at one timestamp together with a learned CLS market token,
  applies FiLM modulation on (vol_bucket, time-of-day), 2 Pre-LN
  Transformer blocks, then per-symbol classification + CLS aux head for
  market-state prediction.

Emits logits for `{long_idx=0, neutral_idx=1, short_idx=2}` (matching the
{−1, 0, +1} tradeable label, indexed for cross-entropy convenience):

    label = +1 LONG    → class index 0
    label =  0 neutral → class index 1
    label = -1 SHORT   → class index 2

At inference, `P(long | x) = softmax(logits)[0]` is the natural ranking
score for "is this stock a good long candidate at time t". Symmetric for
short via `softmax(logits)[2]`.
"""

from __future__ import annotations

import math
from datetime import datetime

import torch
from torch import nn

# Label-index convention used throughout Phase 3X.
LABEL_TO_IDX: dict[int, int] = {+1: 0, 0: 1, -1: 2}
IDX_TO_LABEL: dict[int, int] = {v: k for k, v in LABEL_TO_IDX.items()}
LONG_IDX = LABEL_TO_IDX[+1]
NEUTRAL_IDX = LABEL_TO_IDX[0]
SHORT_IDX = LABEL_TO_IDX[-1]
N_CLASSES = 3


def label_tensor_to_index(label_int8: torch.Tensor) -> torch.Tensor:
    """Convert {-1, 0, +1} int tensor to class-index tensor for CE loss."""
    out = torch.empty_like(label_int8, dtype=torch.long)
    out[label_int8 == 1]  = LONG_IDX
    out[label_int8 == 0]  = NEUTRAL_IDX
    out[label_int8 == -1] = SHORT_IDX
    return out


# ---------------------------------------------------------------------------
# Legacy: per-symbol MLP head (kept for backward compat with head_only ckpts)
# ---------------------------------------------------------------------------

class TradeableHead(nn.Module):
    """2-layer MLP, FORECAST hidden → 3 class logits.

    Per-symbol independent classification. Zero cross-symbol context.
    """

    def __init__(self, d_in: int, d_hidden: int | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        h = d_hidden if d_hidden is not None else d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input `(B, d_in)`; output `(B, 3)` logits."""
        return self.net(x)

    def predict_long_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)[..., LONG_IDX]

    def predict_short_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)[..., SHORT_IDX]


# ---------------------------------------------------------------------------
# v2: cross-symbol attention head with CLS + FiLM
# ---------------------------------------------------------------------------

# NSE intraday 30-min bars: 09:15..15:15 = 13 bars per session
N_TOD_BARS = 13
NSE_OPEN_MINUTES = 9 * 60 + 15      # 9:15 IST


def _tod_features(t: datetime, device: torch.device) -> torch.Tensor:
    """16-dim sin/cos time-of-day encoding.

    Maps minutes-from-open to (sin/cos pairs at 4 harmonics) for the 13-bar
    intraday session. Returns `(16,)` tensor on the given device.
    """
    # Minutes from open in IST. Convert UTC → IST (+5:30).
    ist_minute_of_day = (t.hour * 60 + t.minute + 5 * 60 + 30) % (24 * 60)
    mins_from_open = ist_minute_of_day - NSE_OPEN_MINUTES
    # Normalize to [0, 1] across the 13-bar session (390 min = 6.5 hr).
    frac = float(mins_from_open) / (N_TOD_BARS * 30.0)
    pairs = []
    for k in range(1, 9):       # 8 harmonics → 16-dim sin/cos encoding
        pairs.append(math.sin(2 * math.pi * k * frac))
        pairs.append(math.cos(2 * math.pi * k * frac))
    return torch.tensor(pairs, dtype=torch.float32, device=device)


class _PreLNTransformerBlock(nn.Module):
    """Pre-LN multi-head self-attention + GLU MLP block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        # GLU MLP: project to 2*d_ff, split, gate.
        self.ffn_in = nn.Linear(d_model, 2 * d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (1, S, d_model)
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        a, b = self.ffn_in(h).chunk(2, dim=-1)
        h = a * torch.nn.functional.gelu(b)   # GLU-GELU gating
        x = x + self.dropout(self.ffn_out(h))
        return x


class CrossSectionTradeableHead(nn.Module):
    """Cross-symbol attention head.

    Pipeline (one cross-section at a time, N = #valid symbols ≤ 205):

      forecast_hidden (N, d_model)
        → LayerNorm(d_model)                          # feature standardize
        → cross-sectional LayerNorm over N axis       # market-level standardize
        → FiLM(vol_bucket, time-of-day)               # regime modulation
        → prepend learned CLS_market token (1, d_model)
        → 2 × Pre-LN Transformer (n_heads, d_ff, GLU)
        → per-symbol classifier → (N, 3) logits
        → CLS → market-state scalar prediction (aux target)

    Symbol identity is intentionally NOT injected — head must be
    permutation-equivariant over symbols.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_vol_buckets: int = 5,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.feature_norm = nn.LayerNorm(d_model)
        # Optional aux-feature projection: when aux_dim > 0, project per-symbol
        # aux features (e.g. Phase 3 XGB-importance features) into d_model and
        # ADD to encoder output before cross-symbol attention.
        if aux_dim > 0:
            self.aux_proj = nn.Sequential(
                nn.LayerNorm(aux_dim),
                nn.Linear(aux_dim, d_model),
            )
            # Init projection small — start with encoder output dominant.
            nn.init.zeros_(self.aux_proj[-1].weight)
            nn.init.zeros_(self.aux_proj[-1].bias)
        else:
            self.aux_proj = None

        # Cross-sectional LN: standardize per-feature across the N axis.
        # We compute mean/var across N manually (need a mask for invalid rows).
        # Then apply learnable affine.
        self.cs_norm_weight = nn.Parameter(torch.ones(d_model))
        self.cs_norm_bias   = nn.Parameter(torch.zeros(d_model))

        # FiLM: (vol_bucket, time-of-day) → (γ, β) ∈ R^{2 * d_model}
        self.vol_embed = nn.Embedding(n_vol_buckets, 32)
        self.film_mlp = nn.Sequential(
            nn.Linear(32 + 16, 128),    # 32 vol_embed + 16 tod_sincos
            nn.GELU(),
            nn.Linear(128, 2 * d_model),
        )
        # Init the FiLM output near zero so we start at "no modulation".
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # Learned CLS market token.
        self.cls_market = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.cls_market, std=0.02)

        self.layers = nn.ModuleList([
            _PreLNTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, N_CLASSES)
        # Init classifier small so we start near uniform predictions.
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        # CLS aux head: 1-dim regression to cross-section mean signed return.
        self.cls_aux = nn.Linear(d_model, 1)
        nn.init.zeros_(self.cls_aux.weight)
        nn.init.zeros_(self.cls_aux.bias)
        # Quantile aux head: predict 5 quantile levels of forward return per
        # symbol. Dense per-symbol regression gradient — counter-balances the
        # ListNet loss which concentrates gradient on tail symbols.
        self.quantile_head = nn.Linear(d_model, 5)
        nn.init.zeros_(self.quantile_head.weight)
        nn.init.zeros_(self.quantile_head.bias)

    def forward(
        self,
        forecast_hidden: torch.Tensor,    # (N, d_model)
        vol_bucket: torch.Tensor,          # (N,) int (clamped to n_vol_buckets-1)
        t: datetime,                       # cross-section timestamp
        valid_mask: torch.Tensor,          # (N,) bool — True = include
        aux: torch.Tensor | None = None,  # (N, aux_dim) — optional aux features
    ) -> dict[str, torch.Tensor]:
        n = forecast_hidden.shape[0]
        x = self.feature_norm(forecast_hidden)                          # (N, d)
        if self.aux_proj is not None and aux is not None:
            x = x + self.aux_proj(aux)                                  # add aux signal

        # Cross-sectional standardization across valid rows only.
        mask_f = valid_mask.to(x.dtype).unsqueeze(-1)                   # (N, 1)
        n_valid = mask_f.sum().clamp(min=1.0)
        mean = (x * mask_f).sum(0, keepdim=True) / n_valid              # (1, d)
        var  = ((x - mean) ** 2 * mask_f).sum(0, keepdim=True) / n_valid
        x_cs = (x - mean) / (var + 1e-5).sqrt()
        x = self.cs_norm_weight * x_cs + self.cs_norm_bias              # (N, d)

        # FiLM (regime modulation).
        vol_bucket = vol_bucket.clamp(min=0, max=self.vol_embed.num_embeddings - 1)
        vol_e = self.vol_embed(vol_bucket)                              # (N, 32)
        tod_e = _tod_features(t, x.device).unsqueeze(0).expand(n, -1)   # (N, 16)
        film_in = torch.cat([vol_e, tod_e], dim=-1)                     # (N, 48)
        gamma_beta = self.film_mlp(film_in)                             # (N, 2d)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        x = (1.0 + gamma) * x + beta                                    # FiLM

        # Prepend CLS market token.
        cls = self.cls_market.expand(1, -1)                             # (1, d)
        xs  = torch.cat([cls, x], dim=0)                                # (N+1, d)
        cls_valid = valid_mask.new_ones(1)
        seq_valid = torch.cat([cls_valid, valid_mask], dim=0)           # (N+1,)
        key_padding_mask = ~seq_valid                                   # True=IGNORE

        # Treat one cross-section as one sequence (batch_dim=1).
        xs  = xs.unsqueeze(0)                                            # (1, N+1, d)
        kpm = key_padding_mask.unsqueeze(0)                              # (1, N+1)
        for layer in self.layers:
            xs = layer(xs, key_padding_mask=kpm)
        xs = self.final_norm(xs)
        xs = xs.squeeze(0)                                               # (N+1, d)

        cls_out = xs[0]                                                  # (d,)
        sym_out = xs[1:]                                                 # (N, d)
        logits  = self.classifier(sym_out)                               # (N, 3)
        cls_pred = self.cls_aux(cls_out).squeeze(-1)                     # scalar
        quantiles = self.quantile_head(sym_out)                          # (N, 5)
        return {
            "logits":     logits,
            "cls_market": cls_pred,
            "quantiles":  quantiles,
        }

    def predict_long_probs(self, *args, **kwargs) -> torch.Tensor:
        out = self.forward(*args, **kwargs)
        return torch.softmax(out["logits"], dim=-1)[..., LONG_IDX]

    def predict_short_probs(self, *args, **kwargs) -> torch.Tensor:
        out = self.forward(*args, **kwargs)
        return torch.softmax(out["logits"], dim=-1)[..., SHORT_IDX]
