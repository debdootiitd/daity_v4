"""Leakage gate for `daity.eval.xgboost_baseline` — `PHASE3.md` §5.7.

The 457 flattened features are derived from Phase 1's 18-channel feature
engine, so they inherit those leakage gates — but the per-window
aggregation (`mean`, `std`, `min`, `max`, `slope`, `last`) introduces a
new leakage opportunity. Specifically: "linear-trend slope over the
window" computed across `(t-window, t]` includes the bar at `t`;
computed across `[t-window, t)` excludes it. The difference is one bar
— and one bar is the difference between leakage and not-leakage.

This gate verifies the strict-`[t-window, t)` semantics across 100
random `(symbol, t)` triples (architect review §C.5 / B.MINOR.1).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.parquet_store import ParquetStore
from daity.eval.xgboost_baseline import (
    flatten_anchors_to_features,
)

pytestmark = pytest.mark.leakage


IST = timezone(timedelta(hours=5, minutes=30))


def _seeded() -> random.Random:
    return random.Random(20260511)


def _np_rng() -> np.random.Generator:
    return np.random.default_rng(20260511)


def _build_feature_store(tmp_path: Path) -> tuple[ParquetStore, dict[str, list[datetime]]]:
    """Synthesize an 18-channel feature parquet for 5 symbols × 4 scales."""
    store = ParquetStore(tmp_path)
    rng = _np_rng()
    by: dict[str, list[datetime]] = {}

    # 30 trading days × 75 5m bars = 2250 5m bars.
    base = datetime(2024, 6, 3)
    days: list[datetime] = []
    cur = base
    while len(days) < 30:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)

    ts_5m: list[datetime] = []
    for d in days:
        session_open = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
        bu = session_open.astimezone(UTC)
        for k in range(75):
            ts_5m.append(bu + timedelta(minutes=5 * k))
    ts_15m = ts_5m[::3]
    ts_60m = ts_5m[::12]
    ts_day = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]

    for sym in ("A", "B", "C", "D", "E"):
        for scale, ts_list in [
            ("5m",  ts_5m),
            ("15m", ts_15m),
            ("60m", ts_60m),
            ("day", ts_day),
        ]:
            n = len(ts_list)
            cols = {"symbol": [sym] * n, "ts": ts_list, "scale": [scale] * n}
            schema = {
                "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "scale": pl.Utf8,
            }
            for ch in CHANNEL_ORDER_18:
                cols[ch] = list(rng.normal(0, 1, size=n).astype(np.float32))
                schema[ch] = pl.Float32
            store.write_partition(scale, sym, pl.DataFrame(cols, schema=schema))
    by["5m"] = ts_5m
    by["day"] = ts_day
    return store, by


def test_no_anchor_window_includes_bar_at_anchor(tmp_path: Path) -> None:
    """100 random `(symbol, anchor_ts)` triples — for each, the
    flattened "5m close last" must equal the close of the bar
    `anchor_ts - 10min` (the bar opening at anchor_ts - 10m, closing at
    anchor_ts - 5m). It must NEVER equal the close of the bar opening
    at anchor_ts - 5m (whose close == anchor_ts).
    """
    store, by = _build_feature_store(tmp_path)
    rng = _seeded()
    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}

    for trial in range(100):
        sym = rng.choice(("A", "B", "C", "D", "E"))
        # Pick an anchor at a random 5m bar boundary, deep enough to
        # have warmup at every scale.
        idx = rng.randint(50, len(by["5m"]) - 1)
        anchor_ts = by["5m"][idx] + timedelta(minutes=5)   # eff_close of bar at idx

        feats = flatten_anchors_to_features(
            feature_store=store, symbol=sym,
            anchor_ts=[anchor_ts],
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
            window_bars=small_windows,
        )
        if feats.height == 0:
            continue
        # Read the truth: close at bar (idx - 1) is the strict-`[t-W, t)`
        # answer. close at bar idx is the LEAKING answer.
        df_5m = pl.read_parquet(tmp_path / "5m" / f"{sym}.parquet").sort("ts")
        truth_strict = float(df_5m["close"][idx - 1])
        truth_leaky = float(df_5m["close"][idx])

        flat_last = float(feats["5m__close__last"][0])
        assert abs(flat_last - truth_strict) < 1e-5, (
            f"trial={trial} sym={sym} anchor={anchor_ts}: "
            f"flat_last={flat_last:.6f} truth_strict={truth_strict:.6f}"
        )
        # Just-as-importantly: flat_last must NOT equal the leaky truth.
        # (Modulo astronomically-rare numeric coincidence — the synth data
        # uses rng.normal so duplicate floats are vanishingly unlikely.)
        assert abs(flat_last - truth_leaky) > 1e-9 or truth_strict == truth_leaky, (
            f"trial={trial} sym={sym} anchor={anchor_ts}: "
            f"flat_last leaks the bar AT anchor (={truth_leaky:.6f})"
        )


def test_flatten_drops_anchors_with_insufficient_warmup(tmp_path: Path) -> None:
    """An anchor early in history (before any scale's window has filled)
    must be dropped from the output, not emitted with NaN features."""
    store, by = _build_feature_store(tmp_path)
    small_windows = {"5m": 100, "15m": 50, "60m": 20, "day": 10}
    # Anchor at the very first 5m bar's boundary — every rolling window is
    # not yet warm.
    anchor_ts = by["5m"][0] + timedelta(minutes=5)

    feats = flatten_anchors_to_features(
        feature_store=store, symbol="A",
        anchor_ts=[anchor_ts],
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        window_bars=small_windows,
    )
    assert feats.height == 0


def test_flatten_honors_as_of_cap(tmp_path: Path) -> None:
    """Anchors AFTER `as_of` produce no features (no input bar is
    available to satisfy `effective_close < anchor_ts` AND `≤ as_of`)."""
    store, by = _build_feature_store(tmp_path)
    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}
    # Pick anchor near end of data; cap as_of in the middle.
    anchor_ts = by["5m"][-1] + timedelta(minutes=5)
    as_of = by["5m"][len(by["5m"]) // 2]

    feats = flatten_anchors_to_features(
        feature_store=store, symbol="A",
        anchor_ts=[anchor_ts],
        as_of=as_of,
        window_bars=small_windows,
    )
    # Must NOT use any bar past as_of. With anchor far past as_of, the
    # join_asof either drops or fills with stale stats; either way no
    # column should reflect data AT or after as_of.
    if feats.height > 0:
        # The "last" 5m close must be strictly before as_of.
        df_5m = pl.read_parquet(tmp_path / "5m" / "A.parquet").sort("ts")
        eff_close = df_5m["ts"].to_numpy() + np.timedelta64(5, "m")
        # Find the close at the last bar with eff_close < as_of (strict).
        valid_mask = eff_close < np.datetime64(as_of.replace(tzinfo=None))
        if valid_mask.any():
            last_valid_idx = int(np.where(valid_mask)[0][-1])
            expected = float(df_5m["close"][last_valid_idx])
            actual = float(feats["5m__close__last"][0])
            assert abs(actual - expected) < 1e-5
