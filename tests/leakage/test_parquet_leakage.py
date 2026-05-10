"""Leakage gate: `ParquetStore.read` must never return a row with `ts > as_of`.

This is the test DESIGN.md §2.5 calls out: "spot-check 100 random `(symbol,
as_of)` triples". If it ever fails, an upstream bug has caused the data layer
to leak future information into a window that should be point-in-time clean.

Marked `leakage` so CI can select it explicitly: `pytest -m leakage`.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from daity.data.parquet_store import ParquetStore

pytestmark = pytest.mark.leakage


def _seeded_random() -> random.Random:
    """Deterministic randomness so failures are reproducible."""
    return random.Random(20260510)


def _multi_symbol_store(tmp_path: Path) -> tuple[ParquetStore, dict[str, list[datetime]]]:
    """Build a store with several symbols, each spanning multiple years of daily bars.

    Returns the store + a dict mapping symbol → its full ts list (so the
    test can pick random as_of values that cover edge cases: well before
    first ts, well after last ts, exactly on a ts, between two ts values).
    """
    store = ParquetStore(tmp_path)
    by_sym: dict[str, list[datetime]] = {}
    rng = _seeded_random()
    base = datetime(2020, 1, 1, 10, 0, tzinfo=UTC)
    for sym in ("A", "B", "C", "D", "E"):
        # ~5 years of daily bars (NSE-ish, 250 trading days/year). Slightly
        # different start/end per symbol so the as_of-vs-data alignment varies.
        offset = rng.randint(0, 30)
        n_days = rng.randint(900, 1300)
        ts_list = [base + timedelta(days=offset + i) for i in range(n_days)]
        by_sym[sym] = ts_list
        n = len(ts_list)
        frame = pl.DataFrame(
            {
                "symbol": [sym] * n,
                "ts": ts_list,
                "scale": ["day"] * n,
                "open":   [100.0 + i * 0.1 for i in range(n)],
                "high":   [101.0] * n, "low": [99.0] * n,
                "close":  [100.5] * n, "volume": [1000] * n,
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        )
        store.write_partition("day", sym, frame)
    return store, by_sym


def test_per_symbol_read_never_returns_future_rows(tmp_path: Path) -> None:
    """The headline guarantee: 100 random `(symbol, as_of)` triples, no ts > as_of escapes."""
    store, by_sym = _multi_symbol_store(tmp_path)
    rng = _seeded_random()
    symbols = list(by_sym.keys())

    for trial in range(100):
        sym = rng.choice(symbols)
        ts_list = by_sym[sym]
        # Mix of as_of cases: deep in window, near edges, outside window.
        choice = rng.choices(
            ["deep_inside", "near_min", "near_max", "before_all", "after_all", "exact_ts"],
            weights=[40, 15, 15, 5, 10, 15],
        )[0]
        if choice == "deep_inside":
            as_of = ts_list[rng.randint(len(ts_list) // 4, 3 * len(ts_list) // 4)]
        elif choice == "near_min":
            as_of = ts_list[0] + timedelta(hours=rng.randint(-12, 12))
        elif choice == "near_max":
            as_of = ts_list[-1] + timedelta(hours=rng.randint(-12, 12))
        elif choice == "before_all":
            as_of = ts_list[0] - timedelta(days=rng.randint(1, 365))
        elif choice == "after_all":
            as_of = ts_list[-1] + timedelta(days=rng.randint(1, 365))
        else:  # exact_ts
            as_of = ts_list[rng.randint(0, len(ts_list) - 1)]

        out = store.read(symbol=sym, scale="day", as_of=as_of)
        if out.height == 0:
            continue
        max_ts = out["ts"].max()
        assert max_ts <= as_of, (
            f"trial {trial}: leaked! sym={sym} as_of={as_of} max_ts={max_ts} "
            f"(diff = {max_ts - as_of})"
        )


def test_per_symbol_read_with_start_end_bounds_never_leaks(tmp_path: Path) -> None:
    """The combination of start/end + as_of must still respect the leakage gate."""
    store, by_sym = _multi_symbol_store(tmp_path)
    rng = _seeded_random()
    symbols = list(by_sym.keys())

    for _ in range(100):
        sym = rng.choice(symbols)
        ts_list = by_sym[sym]
        i, j = sorted(rng.sample(range(len(ts_list)), 2))
        start = ts_list[i]
        end = ts_list[j]
        # Pick an as_of that may be before, inside, or after [start, end].
        as_of = ts_list[rng.randint(0, len(ts_list) - 1)]
        out = store.read(symbol=sym, scale="day", start=start, end=end, as_of=as_of)
        if out.height == 0:
            continue
        assert out["ts"].max() <= as_of
        assert out["ts"].min() >= start
        assert out["ts"].max() <= end


def test_cross_section_read_never_returns_future_rows(tmp_path: Path) -> None:
    """The cross-section reader must also honor `as_of` (daily case)."""
    store, by_sym = _multi_symbol_store(tmp_path)
    rng = _seeded_random()

    for _ in range(50):
        # Pick a random ts that exists for at least one symbol.
        sym = rng.choice(list(by_sym.keys()))
        ts = rng.choice(by_sym[sym])
        # Pick an as_of: sometimes < ts (should return empty), sometimes >= ts.
        offset = rng.choice([
            -timedelta(days=rng.randint(1, 100)),
            timedelta(0),
            timedelta(days=rng.randint(1, 100)),
        ])
        as_of = ts + offset
        out = store.read_cross_section(scale="day", ts=ts, as_of=as_of)
        if as_of < ts:
            assert out.height == 0, (
                f"cross-section leaked: ts={ts} as_of={as_of} returned {out.height} rows"
            )
        if out.height > 0:
            # Every returned row's ts must equal the requested ts AND be <= as_of.
            assert (out["ts"] == ts).all()
            assert out["ts"].max() <= as_of


def test_cross_section_intraday_respects_effective_close(tmp_path: Path) -> None:
    """Regression for Phase-1 review Critical #2: an intraday cross-section
    asked at as_of=09:30 UTC for the bar opening at 09:30 UTC must return
    EMPTY (the bar's effective close is 09:35 UTC > as_of).
    """
    import polars as pl
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 9, 15, tzinfo=UTC)
    for sym in ("A", "B", "C"):
        ts_list = [base + timedelta(minutes=5 * i) for i in range(10)]
        n = len(ts_list)
        store.write_partition("5m", sym, pl.DataFrame(
            {
                "symbol": [sym] * n, "ts": ts_list, "scale": ["5m"] * n,
                "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
                "close": [100.5] * n, "volume": [1000] * n,
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        ))

    bar_open = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    # as_of exactly at the bar's open: bar is in progress (closes at 09:35).
    out = store.read_cross_section(scale="5m", ts=bar_open, as_of=bar_open)
    assert out.height == 0, (
        f"in-progress 5m bar leaked through cross-section: got {out.height} rows"
    )
    # as_of right before the bar's effective close: still in progress.
    out2 = store.read_cross_section(scale="5m", ts=bar_open,
                                    as_of=bar_open + timedelta(minutes=4))
    assert out2.height == 0
    # as_of exactly AT the effective close (09:35): bar is now finalized.
    out3 = store.read_cross_section(scale="5m", ts=bar_open,
                                    as_of=bar_open + timedelta(minutes=5))
    assert out3.height == 3, "all 3 symbols must appear at finalized bar boundary"


def test_cross_section_random_intraday_triples_no_leak(tmp_path: Path) -> None:
    """Same 100-triple spirit as the per-symbol test, but for the cross-section
    reader on intraday scales. Catches the case where an intraday `as_of`
    happens to land mid-bar."""
    import polars as pl
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 9, 15, tzinfo=UTC)
    rng = _seeded_random()
    syms = ("A", "B", "C", "D", "E")
    ts_list = [base + timedelta(minutes=5 * i) for i in range(800)]
    for sym in syms:
        n = len(ts_list)
        store.write_partition("5m", sym, pl.DataFrame(
            {
                "symbol": [sym] * n, "ts": ts_list, "scale": ["5m"] * n,
                "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
                "close": [100.5] * n, "volume": [1000] * n,
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        ))

    interval = timedelta(minutes=5)
    for _ in range(100):
        ts = rng.choice(ts_list)
        as_of = ts + timedelta(seconds=rng.randint(-60, 600))
        out = store.read_cross_section(scale="5m", ts=ts, as_of=as_of)
        eff_close = ts + interval
        if as_of < eff_close:
            assert out.height == 0, (
                f"intraday cross-section leaked: ts={ts} as_of={as_of} "
                f"effective_close={eff_close} got {out.height} rows"
            )
        else:
            assert out.height == len(syms)
