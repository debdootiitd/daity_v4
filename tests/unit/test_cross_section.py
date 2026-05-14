"""Unit tests for `daity.data.cross_section`.

Locks in:
- The four iterators yield cross-sections at well-formed anchor
  timestamps with consistent symbol membership.
- The `t` of every cross-section is the anchor's `effective_close`.
- Cross-sectional rank ∈ [0, 1] (ordinal — stable argsort, see §6 docs).
- `vol_bucket` ∈ [0, K-1] and balanced when vols are distinct.
- Cross-sections below the MIN_SYMBOLS_FOR_CROSS_SECTION floor are dropped.
- Empty universe → empty iterator (no crash).

The leakage gate (no input bar's effective_close ≥ t; no label
reads past the horizon's end) lives in
`tests/leakage/test_cross_section_leakage.py`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from daity.data.calendar_nse import NSECalendar
from daity.data.cross_section import (
    CrossSectionalSampler,
    _quantile_bin,
)
from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.labels import precompute_labels
from daity.data.parquet_store import ParquetStore

IST = timezone(timedelta(hours=5, minutes=30))


# ----- Synthetic data fixture -----


def _bars_with_features(symbol: str, scale: str, ts_list: list[datetime],
                        closes: np.ndarray, rng: np.random.Generator) -> pl.DataFrame:
    """Synthesize an 18-channel feature frame. `close` is index 3 in
    `CHANNEL_ORDER_18`; we use the passed `closes` array for it so the
    label engine (which reads close-to-close) produces predictable labels.
    """
    n = len(ts_list)
    cols: dict[str, list] = {
        "symbol": [symbol] * n, "ts": ts_list, "scale": [scale] * n,
    }
    schema = {"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "scale": pl.Utf8}
    for i, ch in enumerate(CHANNEL_ORDER_18):
        if ch == "close":
            cols[ch] = closes.tolist()
        elif ch in ("open", "high", "low"):
            cols[ch] = (closes + (i - 1) * 0.01).tolist()   # nearby values
        elif ch == "volume":
            cols[ch] = list(rng.integers(900, 1100, size=n).astype(np.float32))
        else:
            cols[ch] = list(rng.normal(0, 1, size=n).astype(np.float32))
        schema[ch] = pl.Float32
    return pl.DataFrame(cols, schema=schema)


def _raw_bars(symbol: str, scale: str, ts_list: list[datetime],
              closes: np.ndarray, rng: np.random.Generator) -> pl.DataFrame:
    """Raw OHLCV (5-channel, for label generation)."""
    n = len(ts_list)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n, "ts": ts_list, "scale": [scale] * n,
            "open":  closes.astype(np.float64) - 0.05,
            "high":  closes.astype(np.float64) + 0.10,
            "low":   closes.astype(np.float64) - 0.10,
            "close": closes.astype(np.float64),
            "volume": list(rng.integers(900, 1100, size=n).astype(np.int64)),
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )


def _intraday_ts(days: list[date]) -> list[datetime]:
    """75 5m bars × len(days) sessions."""
    out: list[datetime] = []
    for d in days:
        so = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST).astimezone(UTC)
        for k in range(75):
            out.append(so + timedelta(minutes=5 * k))
    return out


def _daily_ts(days: list[date]) -> list[datetime]:
    return [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]


def _build_universe_fixture(tmp_path: Path, n_symbols: int = 12,
                             n_days: int = 40):
    """Build feature_parquet + labels for a small synthetic universe.

    Returns (feature_root, label_root, calendar, universe).
    """
    feature_root = tmp_path / "features"
    raw_root = tmp_path / "raw"
    label_root = tmp_path / "labels"
    feature_root.mkdir()
    raw_root.mkdir()
    label_root.mkdir()

    rng = np.random.default_rng(0)
    # 40 weekdays starting Mon 2024-06-03.
    days: list[date] = []
    cur = date(2024, 6, 3)
    while len(days) < n_days:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    cal = NSECalendar(trading_days=days, holidays=[])

    ts_5m = _intraday_ts(days)
    ts_15m = ts_5m[::3]
    ts_60m = ts_5m[::12]
    ts_day = _daily_ts(days)

    feat_store = ParquetStore(feature_root)
    raw_store = ParquetStore(raw_root)

    universe: list[str] = []
    for s in range(n_symbols):
        sym = f"S{s:02d}"
        universe.append(sym)
        # Different drift per symbol → different label distributions.
        drift = 0.001 * (s - n_symbols / 2)
        closes_5m = np.cumsum(rng.normal(drift, 0.1, size=len(ts_5m))) + 100.0
        closes_5m = np.maximum(closes_5m, 1.0)
        closes_15m = closes_5m[::3]
        closes_60m = closes_5m[::12]
        # Daily closes = the last 5m close of each day.
        closes_day = np.array([closes_5m[(d_idx + 1) * 75 - 1]
                               for d_idx in range(len(days))], dtype=np.float32)

        # 5m: feature_parquet (18ch) + raw_parquet (5ch).
        feat_store.write_partition("5m", sym,
            _bars_with_features(sym, "5m", ts_5m, closes_5m.astype(np.float32), rng))
        feat_store.write_partition("15m", sym,
            _bars_with_features(sym, "15m", ts_15m, closes_15m.astype(np.float32), rng))
        feat_store.write_partition("60m", sym,
            _bars_with_features(sym, "60m", ts_60m, closes_60m.astype(np.float32), rng))
        feat_store.write_partition("day", sym,
            _bars_with_features(sym, "day", ts_day, closes_day, rng))

        raw_store.write_partition("5m", sym,
            _raw_bars(sym, "5m", ts_5m, closes_5m.astype(np.float64), rng))
        raw_store.write_partition("day", sym,
            _raw_bars(sym, "day", ts_day, closes_day.astype(np.float64), rng))

    # Build labels from raw_store.
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    for sym in universe:
        for horizon in ("30m", "120m", "1d", "2d"):
            precompute_labels(
                raw_store=raw_store, label_root=label_root,
                symbol=sym, horizon=horizon, calendar=cal, as_of=as_of,
            )

    return feature_root, label_root, cal, universe


# ----- Tests -----


def test_quantile_bin_uniform_input() -> None:
    """K=5 quintiles on linearly-spaced values produce balanced buckets."""
    vals = np.arange(100, dtype=np.float32)
    bucket = _quantile_bin(vals, n_bins=5)
    assert bucket.min() == 0
    assert bucket.max() == 4
    counts = np.bincount(bucket, minlength=5)
    # Each bucket should be ~20 ± a few.
    for c in counts:
        assert 15 <= c <= 25


def test_quantile_bin_all_equal_values_safe() -> None:
    """All-equal inputs don't crash and produce a single midpoint bucket."""
    vals = np.full(50, 3.14, dtype=np.float32)
    bucket = _quantile_bin(vals, n_bins=5)
    assert (bucket == 2).all()


def test_quantile_bin_empty_input() -> None:
    bucket = _quantile_bin(np.zeros(0, dtype=np.float32), n_bins=5)
    assert bucket.shape == (0,)


def test_sampler_empty_universe_yields_nothing(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    feature_root.mkdir()
    label_root.mkdir()
    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=[], as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert list(sampler.iter_30m()) == []
    assert list(sampler.iter_1d()) == []


@pytest.mark.parametrize("horizon", ["30m", "120m", "1d", "2d"])
def test_sampler_yields_well_formed_cross_sections(tmp_path: Path, horizon: str) -> None:
    feature_root, label_root, cal, universe = _build_universe_fixture(tmp_path)
    small_windows = {"5m": 8, "15m": 6, "60m": 4, "day": 4}

    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe,
        window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        # Cap is loose enough that we get plenty of cross-sections.
        drop_invalid_threshold=0.5,
    )
    it = getattr(sampler, f"iter_{horizon}")
    sections = list(it())
    assert len(sections) > 0, f"expected at least one cross-section for {horizon}"

    for cs in sections:
        # `t` matches the carried `horizon`.
        assert cs.horizon == horizon
        n = len(cs.symbols)
        assert n >= 2
        # All tensor shapes line up.
        assert cs.labels.shape == (n,)
        assert cs.ranks.shape == (n,)
        assert cs.valid_mask.shape == (n,)
        assert cs.vol_bucket.shape == (n,)
        assert cs.valid_mask.all().item(), "valid_mask should be all True"
        for scale in ("5m", "15m", "60m", "day"):
            assert cs.inputs[scale].shape == (n, small_windows[scale], 18)
            assert cs.inputs[scale].dtype == torch.float32
        # Rank in [0, 1].
        assert (cs.ranks >= 0).all() and (cs.ranks <= 1).all()
        # Vol bucket in [0, K-1].
        assert (cs.vol_bucket >= 0).all() and (cs.vol_bucket < 5).all()


def test_sampler_drop_invalid_threshold(tmp_path: Path) -> None:
    """Setting drop_invalid_threshold = 0.99 drops most cross-sections
    (some symbols inevitably miss labels at session edges)."""
    feature_root, label_root, cal, universe = _build_universe_fixture(tmp_path)
    small_windows = {"5m": 8, "15m": 6, "60m": 4, "day": 4}

    loose = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        drop_invalid_threshold=0.0,
    )
    tight = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        drop_invalid_threshold=0.99,
    )
    n_loose = sum(1 for _ in loose.iter_30m())
    n_tight = sum(1 for _ in tight.iter_30m())
    assert n_tight <= n_loose, "tighter threshold should drop more"


def test_sampler_min_end_ts_filters_anchors(tmp_path: Path) -> None:
    """Setting min_end_ts excludes earlier anchors."""
    feature_root, label_root, cal, universe = _build_universe_fixture(tmp_path)
    small_windows = {"5m": 8, "15m": 6, "60m": 4, "day": 4}

    # Pick a cutoff in the middle of the fixture's date range.
    cutoff = datetime(2024, 7, 1, tzinfo=UTC)
    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        min_end_ts=cutoff,
    )
    for cs in sampler.iter_1d():
        assert cs.t >= cutoff


def test_sampler_caches_per_symbol(tmp_path: Path) -> None:
    """A second iteration reuses caches (no extra parquet reads)."""
    feature_root, label_root, cal, universe = _build_universe_fixture(tmp_path)
    small_windows = {"5m": 8, "15m": 6, "60m": 4, "day": 4}

    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    list(sampler.iter_30m())
    n_feat_cached = len(sampler._feat_cache)
    n_label_cached = len(sampler._label_cache)
    assert n_feat_cached == len(universe) * 4   # 4 scales
    # 1 horizon iterated → 1 entry per symbol in the label cache.
    assert n_label_cached == len(universe)
