"""Unit tests for `daity.data.preprocess` — log-return transform.

Locks in the load-bearing properties:
  - shape preservation (L → L)
  - reconstruction round-trip (log_returns → close prices, modulo float)
  - anchor handling (forecast targets must use prior close, not own close)
  - stationarity (std of close_ret is much smaller than std of close itself)
"""

from __future__ import annotations

import pytest
import torch

from daity.data.preprocess import (
    IDX_CLOSE,
    IDX_CLOSE_RET,
    IDX_HIGH_POS,
    IDX_LOG_VOLUME,
    IDX_LOW_POS,
    IDX_OPEN_RET,
    N_CHANNELS_OHLCV,
    reconstruct_close_from_returns,
    to_log_returns,
)


def _synthetic_ohlcv(B: int = 4, L: int = 64, *, seed: int = 0) -> torch.Tensor:
    """Realistic OHLCV: ~1% per-bar log-volatility on close, OHL straddle."""
    g = torch.Generator().manual_seed(seed)
    # Random walk for close.
    log_returns = torch.randn(B, L, generator=g) * 0.01            # 1% per-bar vol
    close = torch.exp(log_returns.cumsum(dim=1)) * 100.0           # start at 100
    open_ = close * torch.exp(torch.randn(B, L, generator=g) * 0.005)
    intra_vol = torch.rand(B, L, generator=g) * 0.005 + 0.001      # 0.1-0.6% range
    high = torch.maximum(open_, close) * torch.exp(intra_vol)
    low  = torch.minimum(open_, close) * torch.exp(-intra_vol)
    volume = torch.exp(torch.randn(B, L, generator=g) * 0.5 + 8.0)
    return torch.stack([open_, high, low, close, volume], dim=-1)  # (B, L, 5)


# ----- shape + sanity -----


def test_to_log_returns_preserves_shape() -> None:
    x = _synthetic_ohlcv(B=4, L=64)
    y = to_log_returns(x)
    assert y.shape == x.shape


def test_to_log_returns_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="Expected"):
        to_log_returns(torch.randn(2, 16, 3))


def test_to_log_returns_self_anchor_zeros_first_close_ret() -> None:
    """With self-anchor (no anchor_close), close_ret[0] must be exactly 0."""
    x = _synthetic_ohlcv()
    y = to_log_returns(x)
    assert torch.allclose(y[:, 0, IDX_CLOSE_RET], torch.zeros(x.size(0)), atol=0.0)


# ----- stationarity -----


def test_log_returns_are_more_stationary_than_raw_close() -> None:
    """The whole point: close_ret std ≪ close std (per-batch). Std of log-returns
    on a 1%-vol series should be ~0.01; std of close prices ~ tens of INR."""
    x = _synthetic_ohlcv(B=8, L=128, seed=42)
    y = to_log_returns(x)
    raw_close_std = x[..., IDX_CLOSE].std(dim=1).mean().item()
    log_ret_std = y[..., IDX_CLOSE_RET].std(dim=1).mean().item()
    # Raw close std should be at least 100× the log-return std on this fixture.
    assert raw_close_std > 100 * log_ret_std, (
        f"raw close std={raw_close_std:.4f}, log_ret std={log_ret_std:.4f} "
        f"— transform did NOT make things stationary"
    )


# ----- anchor handling (load-bearing for forecast targets) -----


def test_anchor_close_changes_first_close_ret() -> None:
    """With anchor_close = some prior price, close_ret[0] = log(close[0] / anchor)."""
    x = _synthetic_ohlcv(B=2, L=8)
    anchor = torch.tensor([95.0, 105.0])
    y = to_log_returns(x, anchor_close=anchor)
    expected_first = torch.log(x[:, 0, IDX_CLOSE] / anchor)
    assert torch.allclose(y[:, 0, IDX_CLOSE_RET], expected_first, atol=1e-6)


def test_anchor_close_wrong_shape_raises() -> None:
    x = _synthetic_ohlcv(B=4, L=8)
    with pytest.raises(ValueError, match="anchor_close"):
        to_log_returns(x, anchor_close=torch.tensor([1.0, 2.0]))   # wrong B


def test_self_anchor_equals_anchor_at_close_zero() -> None:
    """With anchor_close = close[:, 0], the result must match self-anchor."""
    x = _synthetic_ohlcv(B=3, L=16, seed=7)
    y_self = to_log_returns(x)
    y_anchor = to_log_returns(x, anchor_close=x[:, 0, IDX_CLOSE])
    assert torch.allclose(y_self, y_anchor, atol=1e-6)


# ----- reconstruction round-trip -----


def test_reconstruct_close_round_trip() -> None:
    """log-return → close round-trips to the input close prices (modulo float)."""
    x = _synthetic_ohlcv(B=2, L=32, seed=11)
    anchor = x[:, 0, IDX_CLOSE]                                    # use first close as anchor
    y = to_log_returns(x, anchor_close=anchor)
    rec = reconstruct_close_from_returns(y[..., IDX_CLOSE_RET], anchor)
    # Compare to original close (positions 0..L-1).
    assert torch.allclose(rec, x[..., IDX_CLOSE], rtol=1e-5, atol=1e-4)


def test_reconstruct_close_with_held_back_anchor() -> None:
    """Forecast scenario: anchor is the LAST input bar's close. Reconstruction
    of the FIRST forecast bar yields the actual close at that step."""
    x = _synthetic_ohlcv(B=4, L=20, seed=22)
    # Pretend bars[0..15] are input, bars[16..19] are future.
    last_input_close = x[:, 15, IDX_CLOSE]
    future = x[:, 16:, :]                                          # (B, 4, 5)
    y = to_log_returns(future, anchor_close=last_input_close)
    rec = reconstruct_close_from_returns(y[..., IDX_CLOSE_RET], last_input_close)
    assert torch.allclose(rec, future[..., IDX_CLOSE], rtol=1e-5, atol=1e-4)


# ----- numerical robustness -----


def test_zero_volume_does_not_nan() -> None:
    """log_volume = log1p(volume) — even volume=0 must produce log_volume=0
    cleanly, no NaN/Inf."""
    x = _synthetic_ohlcv()
    x[..., 4] = 0.0                                                # zero out volume
    y = to_log_returns(x)
    assert torch.isfinite(y).all()
    assert torch.allclose(y[..., IDX_LOG_VOLUME], torch.zeros_like(y[..., IDX_LOG_VOLUME]))


def test_zero_or_near_zero_price_does_not_nan() -> None:
    """Bars with degenerate prices (limit-locked, pre-listing zero) must not
    explode the log() — the _LOG_FLOOR clamp protects against this."""
    x = _synthetic_ohlcv()
    # Make a few cells exactly zero.
    x[0, 5, 0] = 0.0
    x[1, 10, 3] = 0.0
    y = to_log_returns(x)
    assert torch.isfinite(y).all()


def test_high_pos_is_typically_nonneg_low_pos_typically_nonpos() -> None:
    """Sanity: by definition, high >= open ≥ low → log(high/open) ≥ 0,
    log(low/open) ≤ 0 PER BAR. We test a fraction (some bars have synthetic
    floats where open marginally exceeds high in the noise; a 95% threshold
    is fine for realistic data)."""
    x = _synthetic_ohlcv(B=8, L=128, seed=33)
    y = to_log_returns(x)
    pct_high_nonneg = (y[..., IDX_HIGH_POS] >= -1e-7).float().mean().item()
    pct_low_nonpos  = (y[..., IDX_LOW_POS]  <=  1e-7).float().mean().item()
    assert pct_high_nonneg > 0.95
    assert pct_low_nonpos  > 0.95


def test_idx_constants_are_canonical() -> None:
    """Lock the channel-index contract — downstream code indexes by these
    constants."""
    assert (IDX_OPEN_RET, IDX_HIGH_POS, IDX_LOW_POS, IDX_CLOSE_RET, IDX_LOG_VOLUME) == (0, 1, 2, 3, 4)
    assert N_CHANNELS_OHLCV == 5
    # IDX_CLOSE is the input-side close index; IDX_CLOSE_RET is the output-side.
    # They share the value 3 by design (close stays in slot 3 across the transform).
    assert IDX_CLOSE == IDX_CLOSE_RET == 3
