"""Stationary preprocessing for OHLCV bars (Phase 2.3).

Why this module exists: predicting raw OHLCV (or even RevIN-normalized
OHLCV) makes the SSL forecast head fight non-stationarity — prices are
random-walk-like, levels drift across regimes, and MSE on absolute values
is dominated by trend rather than the predictable signal. Log-returns are
the canonical stationary representation:

    log(close[t] / close[t-1])  ≈  the bar's "alpha"

This module decomposes raw OHLCV into 5 stationary channels:

    [0] open_ret  = log(open[t]  / close[t-1])     # overnight gap
    [1] high_pos  = log(high[t]  / open[t])        # intra-bar upper extreme  (≥ 0)
    [2] low_pos   = log(low[t]   / open[t])        # intra-bar lower extreme  (≤ 0)
    [3] close_ret = log(close[t] / close[t-1])     # canonical log-return
    [4] log_volume = log(1 + volume[t])            # stabilized log-volume

All five are stationary; (high_pos, low_pos) are bounded by intra-bar
range so they live in a small range; close_ret + open_ret are the
"signal" channels; log_volume is the "regime" channel. RevIN on top of
this gives per-(B, channel) standardization for free.

The transform takes (B, L, 5) raw OHLCV → (B, L, 5) log-return form.
Length is preserved by using the FIRST raw bar as anchor (its log-return
is by definition 0 for the close channel; the high/low/volume features
are still well-defined).

For forecast targets specifically: the FUTURE bars must be anchored to
the LAST INPUT BAR'S close, not to the first future bar. `to_log_returns`
takes an optional `anchor_close` to handle this case.
"""

from __future__ import annotations

import torch

# Channel indices in the canonical OHLCV order used by `OHLCVPretrainDataset`.
IDX_OPEN, IDX_HIGH, IDX_LOW, IDX_CLOSE, IDX_VOLUME = 0, 1, 2, 3, 4
N_CHANNELS_OHLCV = 5

# Output channel indices in the LOG-RETURN form (same count, 5 channels).
IDX_OPEN_RET, IDX_HIGH_POS, IDX_LOW_POS, IDX_CLOSE_RET, IDX_LOG_VOLUME = 0, 1, 2, 3, 4

# Numerical floor for log() to handle zero/near-zero prices or volumes.
# Log(<small>) is large-negative; clamping prevents NaN/Inf cascades into the
# loss. Real NSE prices are >= 1 INR; volumes are usually >= 1 share. The
# floor is tiny enough not to perturb any realistic value.
_LOG_FLOOR = 1e-8


def to_log_returns(
    x: torch.Tensor,
    *,
    anchor_close: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert raw OHLCV to a 5-channel stationary log-return form.

    Args:
      x: (B, L, 5) raw OHLCV bars. Channel order = (open, high, low, close,
         volume) per `OHLCVPretrainDataset.DEFAULT_CHANNELS`.
      anchor_close: optional (B,) tensor of "previous close" values used as
         the anchor for the FIRST output bar. If None, the first bar is
         self-anchored (open_ret[0] = 0, close_ret[0] = 0). Provide this
         when computing forecast targets — the anchor is the last input
         bar's close, NOT the first future bar's close (which would leak).

    Returns:
      (B, L, 5) tensor of log-return-form channels. See module docstring
      for channel definitions.

    Notes:
      - Length is preserved (L → L). The anchor handling means we don't
        drop a bar.
      - All 5 channels are dimensionless / stationary. Standardization
        on top (RevIN) handles the per-window scale variation.
    """
    if x.dim() != 3 or x.size(-1) != N_CHANNELS_OHLCV:
        msg = f"Expected (B, L, 5) raw OHLCV, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    open_  = x[..., IDX_OPEN].clamp(min=_LOG_FLOOR)
    high   = x[..., IDX_HIGH].clamp(min=_LOG_FLOOR)
    low    = x[..., IDX_LOW].clamp(min=_LOG_FLOOR)
    close  = x[..., IDX_CLOSE].clamp(min=_LOG_FLOOR)
    volume = x[..., IDX_VOLUME].clamp(min=0.0)

    # Build "previous close" series with the anchor.
    # prev_close[:, 0] = anchor_close (if given) or close[:, 0] (self-anchor)
    # prev_close[:, t] = close[:, t-1] for t >= 1
    if anchor_close is None:
        # Self-anchor: prev_close[0] = close[0] makes close_ret[0] = 0,
        # which is the only sensible "no information" value at the boundary.
        anchor = close[:, :1]                                # (B, 1)
    else:
        if anchor_close.shape != (x.size(0),):
            msg = (
                f"anchor_close must be shape (B,)={(x.size(0),)}, "
                f"got {tuple(anchor_close.shape)}"
            )
            raise ValueError(msg)
        anchor = anchor_close.clamp(min=_LOG_FLOOR).unsqueeze(-1)   # (B, 1)
    prev_close = torch.cat([anchor, close[:, :-1]], dim=1)          # (B, L)

    open_ret  = torch.log(open_  / prev_close)              # (B, L)
    high_pos  = torch.log(high   / open_)                   # (B, L), ≥ 0 typically
    low_pos   = torch.log(low    / open_)                   # (B, L), ≤ 0 typically
    close_ret = torch.log(close  / prev_close)              # (B, L)
    log_volume = torch.log1p(volume)                        # (B, L), log(1 + V)

    return torch.stack(
        [open_ret, high_pos, low_pos, close_ret, log_volume], dim=-1,
    )


def to_log_returns_partial(
    x: torch.Tensor,
    *,
    n_bar_channels: int = N_CHANNELS_OHLCV,
    anchor_close: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply `to_log_returns` to ONLY the first `n_bar_channels` of `x`.

    For v3_features (18-channel input) — channels 0:5 are raw OHLCV
    (`bar_channels` from `feature_precompute.CHANNEL_ORDER_18`), the
    remaining 13 channels are already-stationary derived features
    (log_return lags, wicks, vol_z, time covariates) that should pass
    through unchanged.

    Args:
      x: (B, L, C) where C ≥ n_bar_channels (≥ 5).
      n_bar_channels: how many leading channels to log-return-ize.
      anchor_close: optional (B,) anchor for the FIRST output bar's
         close-return; same semantics as `to_log_returns`.

    Returns:
      (B, L, C) with the first `n_bar_channels` replaced by their
      log-return form, the rest passed through.
    """
    if x.dim() != 3:
        msg = f"Expected (B, L, C), got shape {tuple(x.shape)}"
        raise ValueError(msg)
    if x.size(-1) < n_bar_channels:
        msg = (
            f"Expected at least {n_bar_channels} bar channels, "
            f"got input with {x.size(-1)} channels total"
        )
        raise ValueError(msg)
    if x.size(-1) == n_bar_channels:
        # Pure-OHLCV case: avoid the slice/concat overhead.
        return to_log_returns(x, anchor_close=anchor_close)
    bars = x[..., :n_bar_channels]                        # (B, L, n_bar_channels)
    rest = x[..., n_bar_channels:]                        # (B, L, C - n_bar_channels)
    bars_lr = to_log_returns(bars, anchor_close=anchor_close)
    return torch.cat([bars_lr, rest], dim=-1)


def reconstruct_close_from_returns(
    close_ret: torch.Tensor, anchor_close: torch.Tensor,
) -> torch.Tensor:
    """Inverse: rebuild close prices from log-returns + anchor.

    Args:
      close_ret: (B, L) log-returns of close.
      anchor_close: (B,) "previous close" before the first return.

    Returns:
      (B, L) reconstructed close prices.

    Used at decode time (e.g., when visualizing predictions in absolute
    INR space). The trip is:

        log_returns = to_log_returns(raw_ohlcv, anchor_close=anchor)
        ...
        close = reconstruct_close_from_returns(log_returns[..., IDX_CLOSE_RET], anchor)

    Verifies bit-equality (up to log/exp roundoff) when round-tripped.
    """
    if close_ret.dim() != 2 or anchor_close.dim() != 1:
        msg = (
            f"Expected close_ret (B, L) and anchor_close (B,), got "
            f"{tuple(close_ret.shape)} and {tuple(anchor_close.shape)}"
        )
        raise ValueError(msg)
    cum_log = close_ret.cumsum(dim=1)
    return anchor_close.unsqueeze(-1) * torch.exp(cum_log)
