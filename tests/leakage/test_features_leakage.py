"""Leakage gate for the feature engine — DESIGN §2.5.

The pipeline-level guarantee: for any `(symbol, scale, as_of)` triple, no
output row's `ts` exceeds `as_of`, AND no input bar with `effective_close >
as_of` is consulted in computing the output.

This is the ParquetStore leakage gate's "one layer up" sibling: the latter
guards the data layer, this one guards the feature layer where rolling
windows have a real chance to peek into the future if the implementation
slips.

Marked `leakage` so CI can select it: `pytest -m leakage`.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from daity.data.candles import INTERVAL_DURATIONS
from daity.data.features import FeatureEngine
from daity.data.parquet_store import ParquetStore

pytestmark = pytest.mark.leakage


def _seeded_random() -> random.Random:
    return random.Random(20260510)


def _build_store_with_intraday_and_daily(tmp_path: Path) -> tuple[ParquetStore, dict]:
    """Build a fixture store: 5 symbols, both 5m and day partitions, ~3 years of bars.

    Realistic enough that rolling features have plenty of warm-up but the
    test still completes in <2s.
    """
    store = ParquetStore(tmp_path)
    rng = _seeded_random()
    by = {"day": {}, "5m": {}}

    base_day = datetime(2023, 1, 2, 10, 0, tzinfo=UTC)   # 15:30 IST close
    for sym in ("A", "B", "C", "D", "E"):
        # Daily: ~750 bars (3 yrs of trading days approx).
        ts_list = [base_day + timedelta(days=i) for i in range(750)]
        closes = [100.0 + rng.gauss(0, 1) for _ in range(750)]
        by["day"][sym] = ts_list
        store.write_partition("day", sym, pl.DataFrame(
            {
                "symbol": [sym] * 750, "ts": ts_list, "scale": ["day"] * 750,
                "open": closes, "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes], "close": closes,
                "volume": [1000 + rng.randint(-100, 100) for _ in range(750)],
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        ))

        # 5m: a smaller window (~10 trading days × 75 bars = 750 bars).
        base_5m = datetime(2024, 4, 1, 9, 15, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        base_5m_utc = base_5m.astimezone(UTC)
        ts_5m: list[datetime] = []
        for d in range(10):
            for k in range(75):
                ts_5m.append(base_5m_utc + timedelta(days=d, minutes=5 * k))
        closes_5m = [100.0 + rng.gauss(0, 0.5) for _ in ts_5m]
        by["5m"][sym] = ts_5m
        store.write_partition("5m", sym, pl.DataFrame(
            {
                "symbol": [sym] * len(ts_5m), "ts": ts_5m,
                "scale": ["5m"] * len(ts_5m),
                "open": closes_5m, "high": [c + 0.1 for c in closes_5m],
                "low": [c - 0.1 for c in closes_5m], "close": closes_5m,
                "volume": [100 + rng.randint(-10, 10) for _ in ts_5m],
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        ))
    return store, by


def test_features_never_emit_rows_past_as_of(tmp_path: Path) -> None:
    """100 random `(symbol, scale, as_of)` triples — no output row's ts > as_of."""
    store, by = _build_store_with_intraday_and_daily(tmp_path)
    engine = FeatureEngine(store)
    rng = _seeded_random()

    for trial in range(100):
        scale = rng.choice(["day", "5m"])
        sym = rng.choice(list(by[scale].keys()))
        ts_list = by[scale][sym]
        choice = rng.choices(
            ["deep_inside", "near_min", "near_max", "before_all", "after_all", "exact_ts"],
            weights=[40, 15, 15, 5, 10, 15],
        )[0]
        if choice == "deep_inside":
            as_of = ts_list[rng.randint(len(ts_list) // 4, 3 * len(ts_list) // 4)]
        elif choice == "near_min":
            as_of = ts_list[0] + timedelta(minutes=rng.randint(-60, 60))
        elif choice == "near_max":
            as_of = ts_list[-1] + timedelta(minutes=rng.randint(-60, 60))
        elif choice == "before_all":
            as_of = ts_list[0] - timedelta(days=rng.randint(1, 365))
        elif choice == "after_all":
            as_of = ts_list[-1] + timedelta(days=rng.randint(1, 365))
        else:  # exact_ts
            as_of = ts_list[rng.randint(0, len(ts_list) - 1)]

        out = engine.compute(symbol=sym, scale=scale, as_of=as_of)
        if out.height == 0:
            continue
        assert out["ts"].max() <= as_of, (
            f"trial {trial}: leak! sym={sym} scale={scale} "
            f"as_of={as_of} max_ts={out['ts'].max()}"
        )


def test_features_never_emit_in_progress_intraday_bars(tmp_path: Path) -> None:
    """An intraday bar whose effective_close > as_of must be excluded.

    A bar opening at 09:30 IST with scale='5m' has effective_close 09:35;
    with as_of = 09:32 it must NOT appear in the output (still in progress)."""
    store, by = _build_store_with_intraday_and_daily(tmp_path)
    engine = FeatureEngine(store)
    rng = _seeded_random()

    for _ in range(50):
        sym = rng.choice(list(by["5m"].keys()))
        ts_list = by["5m"][sym]
        # Pick an as_of that lands strictly in the *middle* of some bar.
        ts = rng.choice(ts_list)
        as_of = ts + timedelta(minutes=rng.choice([1, 2, 3, 4]))

        out = engine.compute(symbol=sym, scale="5m", as_of=as_of)
        if out.height == 0:
            continue
        # The strict invariant: every output row's effective_close <= as_of.
        # For intraday this means ts + duration <= as_of.
        max_ts = out["ts"].max()
        assert max_ts + INTERVAL_DURATIONS["5m"] <= as_of, (
            f"in-progress bar leaked: sym={sym} as_of={as_of} max_ts={max_ts}"
        )


def test_rolling_features_no_future_peek(tmp_path: Path) -> None:
    """A rolling-feature value at row R must depend only on rows <= R.

    Strategy: compute features twice — once with the full series, once
    truncated at row R. The feature values at R must be identical.
    """
    store, by = _build_store_with_intraday_and_daily(tmp_path)
    engine = FeatureEngine(store)
    rng = _seeded_random()

    for _ in range(30):
        sym = rng.choice(list(by["day"].keys()))
        ts_list = by["day"][sym]
        # Pick an "anchor" ts deep enough that all rolling windows are warm.
        anchor_idx = rng.randint(70, len(ts_list) - 50)
        anchor = ts_list[anchor_idx]

        # Full-series compute, restricted via window.
        full = engine.compute(
            symbol=sym, scale="day",
            as_of=ts_list[-1],   # all data eligible
            window_start=anchor, window_end=anchor,
        )
        # Truncated compute: as_of right at the anchor (so the engine only
        # ever sees rows up to the anchor in the underlying read).
        trunc = engine.compute(
            symbol=sym, scale="day",
            as_of=anchor,
            window_start=anchor, window_end=anchor,
        )
        if full.height == 0 or trunc.height == 0:
            continue
        # Compare the rolling-feature columns at the anchor row.
        for col in ("ret_1", "ret_5", "ret_15", "ret_60", "vol_z_20"):
            full_v = full[col][0]
            trunc_v = trunc[col][0]
            if full_v is None and trunc_v is None:
                continue
            assert full_v == pytest.approx(trunc_v, rel=1e-9, abs=1e-9), (
                f"future-peek detected: sym={sym} col={col} anchor={anchor} "
                f"full={full_v} truncated={trunc_v}"
            )
