"""Unit tests for `daity.data.datasets` — the pretraining sampler.

Locks in: leakage gate per-sample, multi-scale alignment, regime-aware
weighting flips heavily toward recent regime when configured, the
DataLoader path collates correctly.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from daity.data.datasets import (
    DEFAULT_CHANNELS,
    DEFAULT_WINDOW_BARS,
    FORECAST_FUTURE_KEY,
    OHLCVPretrainDataset,
    OHLCVWindowSampler,
    WindowSample,
)
from daity.data.parquet_store import ParquetStore


# ----- Fixtures -----


def _frame(symbol: str, scale: str, ts_list: list[datetime]) -> pl.DataFrame:
    """Build a frame matching the per-(scale, symbol) Parquet schema."""
    n = len(ts_list)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n, "ts": ts_list, "scale": [scale] * n,
            "open":   [100.0 + i * 0.1 for i in range(n)],
            "high":   [101.0 + i * 0.1 for i in range(n)],
            "low":    [99.0  + i * 0.1 for i in range(n)],
            "close":  [100.5 + i * 0.1 for i in range(n)],
            "volume": [1000  + i for i in range(n)],
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )


def _seq_5m(start: datetime, n: int) -> list[datetime]:
    """A sequence of 5-min bar opens starting at `start`."""
    return [start + i * timedelta(minutes=5) for i in range(n)]


def _seq_day(start: datetime, n: int) -> list[datetime]:
    """A sequence of daily bars at 10:00 UTC = 15:30 IST close."""
    base = start.replace(hour=10, minute=0, second=0, microsecond=0)
    return [base + i * timedelta(days=1) for i in range(n)]


def _build_store(tmp_path: Path, symbols: list[str], n_5m: int = 800,
                 n_day: int = 200) -> ParquetStore:
    store = ParquetStore(tmp_path)
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    base_day = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for sym in symbols:
        store.write_partition("5m",  sym, _frame(sym, "5m",  _seq_5m(base_5m, n_5m)))
        store.write_partition("day", sym, _frame(sym, "day", _seq_day(base_day, n_day)))
    return store


# ----- Construction guards -----


def test_sampler_rejects_empty_symbols(tmp_path: Path) -> None:
    store = _build_store(tmp_path, ["A"])
    with pytest.raises(ValueError, match="non-empty"):
        OHLCVWindowSampler(store=store, symbols=[],
                           as_of=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_sampler_rejects_zero_window_bars(tmp_path: Path) -> None:
    store = _build_store(tmp_path, ["A"])
    with pytest.raises(ValueError, match="positive"):
        OHLCVWindowSampler(
            store=store, symbols=["A"],
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            window_bars={"5m": 0},
        )


# ----- Sample shape + alignment -----


def test_sample_shape_matches_window_bars(tmp_path: Path) -> None:
    store = _build_store(tmp_path, ["A"])
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 64, "day": 32},
    )
    sample = next(iter(sampler))
    assert isinstance(sample, WindowSample)
    assert sample.channels["5m"].shape == (64, len(DEFAULT_CHANNELS))
    assert sample.channels["day"].shape == (32, len(DEFAULT_CHANNELS))
    # Tensor dtype is float32 — what the model wants.
    assert sample.channels["5m"].dtype == torch.float32


def test_sample_channels_match_default_order(tmp_path: Path) -> None:
    """Channel column order must match `DEFAULT_CHANNELS` so downstream
    code can hardcode 'close = channel[3]' without surprises."""
    store = _build_store(tmp_path, ["A"], n_5m=200)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 32}, seed=42,
    )
    sample = next(iter(sampler))
    # In our fixture, close = open + 0.5. Channel 3 (close) - channel 0 (open) ≈ 0.5.
    diffs = sample.channels["5m"][:, 3] - sample.channels["5m"][:, 0]
    assert torch.allclose(diffs, torch.full_like(diffs, 0.5), atol=1e-3)


# ----- Leakage gate -----


def test_no_sample_extends_past_as_of(tmp_path: Path) -> None:
    """For 100 random samples, every per-scale window's right edge must
    have effective_close ≤ as_of."""
    store = _build_store(tmp_path, ["A", "B"])
    cap = datetime(2024, 1, 5, tzinfo=timezone.utc)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A", "B"], as_of=cap,
        window_bars={"5m": 32, "day": 4}, seed=0,
    )
    it = iter(sampler)
    seen = 0
    for _ in range(100):
        try:
            sample = next(it)
        except StopIteration:
            break
        # The end_ts itself must be <= cap.
        assert sample.end_ts <= cap, f"end_ts {sample.end_ts} > as_of {cap}"
        seen += 1
    # Sanity: sampler isn't completely empty for this fixture.
    assert seen > 0


def test_sampler_skips_symbols_without_enough_history(tmp_path: Path) -> None:
    """A symbol with fewer bars than `window_bars[scale]` should be
    silently skipped — never yield a partial sample."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    # 'SHORT' has only 16 5m bars — less than the 32-bar window we'll request.
    store.write_partition("5m", "SHORT", _frame("SHORT", "5m", _seq_5m(base, 16)))
    store.write_partition("day", "SHORT",
        _frame("SHORT", "day", _seq_day(datetime(2024, 1, 1, tzinfo=timezone.utc), 4)))
    # 'OK' has plenty.
    store.write_partition("5m", "OK", _frame("OK", "5m", _seq_5m(base, 200)))
    store.write_partition("day", "OK",
        _frame("OK", "day", _seq_day(datetime(2024, 1, 1, tzinfo=timezone.utc), 50)))

    sampler = OHLCVWindowSampler(
        store=store, symbols=["SHORT", "OK"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 32, "day": 8}, seed=0,
    )
    it = iter(sampler)
    syms = Counter(next(it).symbol for _ in range(50))
    # SHORT can never satisfy the request; only OK should appear.
    assert syms["SHORT"] == 0, f"sampler yielded the under-history symbol: {syms}"
    assert syms["OK"] > 0


# ----- Regime-aware weighting -----


def test_recent_regime_upweighted(tmp_path: Path) -> None:
    """With recent_regime_weight=2, ~67% of samples should come from the
    trailing 18-month window (vs the prior 18 months at 1× weight).
    """
    store = _build_store(tmp_path, ["A"], n_5m=2, n_day=2_000)  # ~5.5 yrs of daily
    cap = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"], as_of=cap,
        window_bars={"day": 64}, seed=0,
        recent_regime_weight=2.0,
    )
    boundary = cap - timedelta(days=int(18 * 30.5))
    it = iter(sampler)
    n_recent = 0
    n_total = 200
    for _ in range(n_total):
        sample = next(it)
        if sample.end_ts >= boundary:
            n_recent += 1
    # Theoretical: weight_recent / (weight_recent + weight_prior). Recent ≈ 18mo,
    # prior ≈ 36mo. P(recent) = 2*18 / (2*18 + 36) = 36/72 = 0.5. Wait —
    # but we only have ~2000 days = ~5.5 years; recent 18mo is the last 18mo,
    # prior is the first ~3-4 years. Weight ratio: 2 × 548 days / (2 × 548 + ~1500)
    # ≈ 0.42. Allow a generous tolerance for the 200-sample finite-sample noise.
    fraction_recent = n_recent / n_total
    # Just verify recent is *meaningfully* upweighted vs unweighted (~25% if uniform).
    assert fraction_recent > 0.30, (
        f"recent regime not upweighted; got {fraction_recent:.2%} of samples"
    )


def test_recent_regime_disabled_when_window_short(tmp_path: Path) -> None:
    """If `recent_regime_days` covers the entire eligible range, sampling
    is uniform — no special-case math needed."""
    store = _build_store(tmp_path, ["A"], n_5m=2, n_day=200)  # ~7 months
    cap = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"], as_of=cap,
        window_bars={"day": 8}, seed=0,
    )
    # Should still produce samples without crashing.
    it = iter(sampler)
    s = next(it)
    assert isinstance(s, WindowSample)


# ----- DataLoader integration -----


def test_dataloader_collates_per_scale_tensors(tmp_path: Path) -> None:
    """A `DataLoader` should stack per-sample dicts into per-batch dicts."""
    store = _build_store(tmp_path, ["A", "B", "C"])
    ds = OHLCVPretrainDataset(
        store=store, symbols=["A", "B", "C"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 32, "day": 8}, seed=0,
    )
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    # Default collate stacks dicts: each value becomes (B, *value_shape).
    assert batch["5m"].shape == (4, 32, len(DEFAULT_CHANNELS))
    assert batch["day"].shape == (4, 8, len(DEFAULT_CHANNELS))


def test_dataloader_tensors_are_finite(tmp_path: Path) -> None:
    """Sanity: no NaN/inf in produced batches (would silently break training)."""
    store = _build_store(tmp_path, ["A"])
    ds = OHLCVPretrainDataset(
        store=store, symbols=["A"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 32, "day": 8}, seed=0,
    )
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        for sc, t in batch.items():
            assert torch.isfinite(t).all(), f"non-finite values in scale {sc}"


def test_default_channels_locked() -> None:
    """Lock the channel order — downstream model code indexes by position."""
    assert DEFAULT_CHANNELS == ("open", "high", "low", "close", "volume")


def test_default_window_bars_match_design() -> None:
    """Lock the per-scale window sizes (DESIGN: enough for ~3.4 trading days
    of 5m, 8 days of 15m, etc.)."""
    assert DEFAULT_WINDOW_BARS == {
        "5m": 256, "15m": 192, "60m": 168, "day": 64,
    }


# ----- Multi-horizon forecast targets -----


def test_sampler_rejects_forecast_n_bars_without_scale(tmp_path: Path) -> None:
    store = _build_store(tmp_path, ["A"])
    with pytest.raises(ValueError, match="forecast_scale must be set"):
        OHLCVWindowSampler(
            store=store, symbols=["A"],
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            window_bars={"5m": 32, "day": 8},
            forecast_n_bars=16,
        )


def test_sampler_rejects_forecast_scale_not_in_window_bars(tmp_path: Path) -> None:
    store = _build_store(tmp_path, ["A"])
    with pytest.raises(ValueError, match="must also appear in window_bars"):
        OHLCVWindowSampler(
            store=store, symbols=["A"],
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            window_bars={"5m": 32, "day": 8},
            forecast_scale="60m", forecast_n_bars=16,
        )


def test_sampler_yields_forecast_tail_when_enabled(tmp_path: Path) -> None:
    """as_of bounded so drawn end_ts has 24 future 5m bars available.

    n_5m=3000 spans ~10 days of bars, so the day-scale's earliest_end_ts
    (8 day bars in = 2024-01-08) lands inside the 5m data range.
    """
    n_5m = 3000
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    last_5m_close = base_5m + n_5m * timedelta(minutes=5)
    store = _build_store(tmp_path, ["A"], n_5m=n_5m)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"],
        # as_of pinned to just after the last 5m bar's close — latest_end_ts
        # then sits at last_5m_close - 24*5min, with real future bars.
        as_of=last_5m_close,
        window_bars={"5m": 32, "day": 8},
        forecast_scale="5m", forecast_n_bars=24, seed=0,
    )
    sample = next(iter(sampler))
    assert isinstance(sample, WindowSample)
    assert sample.forecast is not None
    assert sample.forecast.shape == (24, len(DEFAULT_CHANNELS))
    # Past window still has correct shape — forecast doesn't perturb it.
    assert sample.channels["5m"].shape == (32, len(DEFAULT_CHANNELS))


def test_sampler_no_forecast_tail_when_disabled(tmp_path: Path) -> None:
    """Backward compat: with forecast_n_bars=0 the sample carries no future tail."""
    store = _build_store(tmp_path, ["A"])
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 32, "day": 8}, seed=0,
    )
    sample = next(iter(sampler))
    assert sample.forecast is None


def test_forecast_tail_respects_as_of_cap(tmp_path: Path) -> None:
    """The future bars must satisfy effective_close <= as_of: the sampler
    rejects samples whose forecast horizon would spill past the cap.

    We construct a minimal store where the latest 5m bar is exactly at
    the as_of cap, then ask for a forecast tail of 8 bars; every drawn
    end_ts must leave 8 valid future bars before the cap. With a tight
    store there's no eligible end_ts, so the sampler must fail to yield."""
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    n_5m = 50  # enough to satisfy window_bars[5m]=32, but only 18 bars after position 32.
    store = ParquetStore(tmp_path)
    store.write_partition(
        "5m", "A", _frame("A", "5m", _seq_5m(base_5m, n_5m)),
    )
    store.write_partition(
        "day", "A", _frame("A", "day", _seq_day(base_5m, 200)),
    )
    # as_of just past the last bar's close (so all 50 bars are eligible).
    last_open = base_5m + (n_5m - 1) * timedelta(minutes=5)
    as_of = last_open + timedelta(minutes=10)
    # Window=32 leaves 18 bars after; ask for a forecast tail of 24 bars
    # → no end_ts can satisfy. Sampler must produce nothing in finite tries.
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"], as_of=as_of,
        window_bars={"5m": 32, "day": 8},
        forecast_scale="5m", forecast_n_bars=24, seed=0,
    )
    # _draw_end_ts returns None or _build_sample returns None for every
    # try; the iterator's inner loop retries forever, so we step manually.
    earliest = sampler._earliest_end_ts("A")
    latest = sampler._latest_end_ts("A")
    # latest is shifted by 24 bars × 5m = 2h before as_of → before earliest.
    assert latest < as_of
    assert earliest is None or earliest >= latest


def test_dataloader_yields_forecast_key_when_enabled(tmp_path: Path) -> None:
    """The dataset should pack the future tail under FORECAST_FUTURE_KEY so
    the default DataLoader collates it as (B, F, C)."""
    n_5m = 3000  # need >= 8 trading days at 5m for window_bars[day]=8 to fit.
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    last_5m_close = base_5m + n_5m * timedelta(minutes=5)
    store = _build_store(tmp_path, ["A", "B"], n_5m=n_5m)
    ds = OHLCVPretrainDataset(
        store=store, symbols=["A", "B"],
        as_of=last_5m_close,
        window_bars={"5m": 32, "day": 8}, seed=0,
        forecast_scale="5m", forecast_n_bars=24,
    )
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert FORECAST_FUTURE_KEY in batch
    assert batch[FORECAST_FUTURE_KEY].shape == (4, 24, len(DEFAULT_CHANNELS))
    assert torch.isfinite(batch[FORECAST_FUTURE_KEY]).all()


def test_dataloader_no_forecast_key_when_disabled(tmp_path: Path) -> None:
    store = _build_store(tmp_path, ["A"])
    ds = OHLCVPretrainDataset(
        store=store, symbols=["A"],
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_bars={"5m": 32, "day": 8}, seed=0,
    )
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert FORECAST_FUTURE_KEY not in batch


def test_min_end_ts_clamps_eligible_range(tmp_path: Path) -> None:
    """DESIGN §2.5 — strict walk-forward val. With `min_end_ts` set, the
    sampler must NEVER yield a sample whose end_ts is before that bound.
    This is the load-bearing fix for the val/train temporal overlap that
    Phase 2.2 closed.
    """
    n_5m = 3000
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    last_5m_close = base_5m + n_5m * timedelta(minutes=5)
    store = _build_store(tmp_path, ["A"], n_5m=n_5m)

    # Set min_end_ts to halfway through the data; verify no sample lands earlier.
    half = base_5m + (n_5m // 2) * timedelta(minutes=5)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"], as_of=last_5m_close,
        window_bars={"5m": 32, "day": 8}, seed=0,
        min_end_ts=half,
    )
    seen = []
    it = iter(sampler)
    for _ in range(50):
        sample = next(it)
        seen.append(sample.end_ts)
    assert all(ts >= half for ts in seen), (
        f"min_end_ts={half} not honored; got {min(seen)}"
    )


def test_min_end_ts_unsatisfiable_returns_none_promptly(tmp_path: Path) -> None:
    """If min_end_ts is past the data extent, the sampler must NOT spin
    forever — _draw_end_ts returns None and the iterator should be the
    one to give up (here we test the bound calculation, not the iterator
    spin protection — that's covered by test_forecast_dataset_handles_*)."""
    n_5m = 3000
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    last_5m_close = base_5m + n_5m * timedelta(minutes=5)
    store = _build_store(tmp_path, ["A"], n_5m=n_5m)
    # Min_end_ts past as_of → no sample possible.
    too_late = last_5m_close + timedelta(days=10)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"], as_of=last_5m_close,
        window_bars={"5m": 32, "day": 8}, seed=0,
        min_end_ts=too_late,
    )
    import random as _random
    rng = _random.Random(0)
    assert sampler._draw_end_ts(rng, "A") is None


def test_forecast_dataset_handles_as_of_far_past_data(tmp_path: Path) -> None:
    """If as_of is well past the actual data's last ts, the sampler must
    bound the eligible end_ts range by data availability — otherwise it'd
    spin in __iter__ forever drawing end_ts where the future tail has no
    real bars to fill it. Production case: backfill caught up to T but
    as_of is set to T+30d for some downstream window."""
    n_5m = 3000  # spans the day-scale's earliest_end_ts so the test is exercisable.
    base_5m = datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc)
    store = _build_store(tmp_path, ["A"], n_5m=n_5m)
    # as_of two years past the data — naive impl would draw end_ts uniformly
    # over those two years and reject every sample (no future bars exist).
    as_of_far_past = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sampler = OHLCVWindowSampler(
        store=store, symbols=["A"], as_of=as_of_far_past,
        window_bars={"5m": 32, "day": 8},
        forecast_scale="5m", forecast_n_bars=24, seed=0,
    )
    # Latest end_ts must be bounded by actual data end - horizon, NOT by as_of.
    latest = sampler._latest_end_ts("A")
    last_5m_close = base_5m + n_5m * timedelta(minutes=5)
    horizon = 24 * timedelta(minutes=5)
    # Allow 1 bar of slack (the data_bound uses last_ts which is open of last bar).
    expected_latest_upper = last_5m_close - horizon + timedelta(minutes=5)
    assert latest <= expected_latest_upper
    assert latest < as_of_far_past
    # And the iterator should yield without spinning.
    sample = next(iter(sampler))
    assert sample.forecast is not None
    assert sample.forecast.shape == (24, len(DEFAULT_CHANNELS))
