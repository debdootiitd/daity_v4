"""Leakage gate for `daity.data.labels` — `PHASE3.md` §3.4.

Three guarantees, each spot-checked across 100 random `(symbol, horizon, t)`
triples (the same shape DESIGN §2.5 mandates for the feature engine):

1. **No row past `as_of` is emitted.** For any `(sym, horizon, as_of)`, every
   output row's `t_horizon` ≤ as_of. (We allow strict-equality at the boundary;
   the labels module's own test on this is an inequality.)
2. **No bar with effective_close > as_of is consulted.** This is the data-
   layer guarantee `ParquetStore.read(as_of=…)` already enforces; the labels
   module inherits it. Test: build labels at `as_of = T`, then assert no
   t_horizon > T appears.
3. **Bit-equality under truncated history.** The label at `(sym, t)`
   produced under `as_of = T1` is bit-equal to the same `(sym, t)` produced
   under `as_of = T2` whenever both queries include the same forward bar.
   This is the no-future-peek invariant — rolling/forward computations must
   not depend on data past the forward window.

Marked `leakage` so CI can select it: `pytest -m leakage`.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from daity.data.calendar_nse import NSECalendar
from daity.data.labels import (
    ALL_HORIZONS,
    HORIZON_DAY_COUNT,
    HORIZONS_DAILY,
    HORIZONS_INTRADAY,
    build_labels,
)
from daity.data.parquet_store import ParquetStore

pytestmark = pytest.mark.leakage


IST = timezone(timedelta(hours=5, minutes=30))


def _seeded() -> random.Random:
    return random.Random(20260510)


def _build_store(tmp_path: Path) -> tuple[ParquetStore, NSECalendar, dict]:
    """Realistic fixture: 5 symbols × (5m + day) over ~3 trading months.

    Enough rows for forward windows on every horizon to either be valid
    or get cleanly dropped at the tail.
    """
    store = ParquetStore(tmp_path)
    rng = _seeded()
    by_5m: dict[str, list[datetime]] = {}
    by_day: dict[str, list[datetime]] = {}

    # ~63 trading days from 2024-06-03 (Mon) onward, skip weekends.
    start = date(2024, 6, 3)
    days: list[date] = []
    cur = start
    while len(days) < 63:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    cal = NSECalendar(trading_days=days, holidays=[])

    for sym in ("A", "B", "C", "D", "E"):
        # 5m: 75 bars per session × 63 sessions = 4725 bars.
        ts_5m: list[datetime] = []
        closes_5m: list[float] = []
        price = 100.0 + 5 * rng.random()
        for d in days:
            session_open = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
            base = session_open.astimezone(UTC)
            for k in range(75):
                ts_5m.append(base + timedelta(minutes=5 * k))
                price = max(1.0, price + rng.gauss(0.0, 0.2))
                closes_5m.append(price)
        store.write_partition("5m", sym, pl.DataFrame(
            {
                "symbol": [sym] * len(ts_5m), "ts": ts_5m,
                "scale": ["5m"] * len(ts_5m),
                "open":  [c - 0.05 for c in closes_5m],
                "high":  [c + 0.10 for c in closes_5m],
                "low":   [c - 0.10 for c in closes_5m],
                "close": closes_5m,
                "volume": [1000 + rng.randint(-100, 100) for _ in ts_5m],
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        ))
        by_5m[sym] = ts_5m

        # day: one bar per trading day.
        ts_day = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]
        closes_day = [100.0 + i * (1.0 + rng.gauss(0, 0.5)) for i in range(len(ts_day))]
        store.write_partition("day", sym, pl.DataFrame(
            {
                "symbol": [sym] * len(ts_day), "ts": ts_day,
                "scale": ["day"] * len(ts_day),
                "open": closes_day,
                "high": [c + 1.0 for c in closes_day],
                "low":  [c - 1.0 for c in closes_day],
                "close": closes_day,
                "volume": [10000 + rng.randint(-1000, 1000) for _ in ts_day],
            },
            schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                    "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
        ))
        by_day[sym] = ts_day

    return store, cal, {"5m": by_5m, "day": by_day}


def test_labels_never_emit_t_horizon_past_as_of(tmp_path: Path) -> None:
    """100 random (sym, horizon, as_of) triples — every t_horizon ≤ as_of."""
    store, cal, by = _build_store(tmp_path)
    rng = _seeded()

    for trial in range(100):
        horizon = rng.choice(ALL_HORIZONS)
        sym = rng.choice(list(by["5m"].keys()))
        # Pick an as_of from the union of 5m + day timestamps, with some
        # outside-range cases to test edge handling.
        ts_pool = by["5m" if horizon in HORIZONS_INTRADAY else "day"][sym]
        choice = rng.choices(
            ["mid", "near_min", "near_max", "before", "after"],
            weights=[55, 10, 15, 10, 10],
        )[0]
        if choice == "mid":
            as_of = ts_pool[rng.randint(len(ts_pool) // 4, 3 * len(ts_pool) // 4)]
        elif choice == "near_min":
            as_of = ts_pool[0] + timedelta(minutes=rng.randint(-30, 60))
        elif choice == "near_max":
            as_of = ts_pool[-1] - timedelta(minutes=rng.randint(0, 60 * 24))
        elif choice == "before":
            as_of = ts_pool[0] - timedelta(days=rng.randint(1, 60))
        else:  # after
            as_of = ts_pool[-1] + timedelta(days=rng.randint(1, 60))

        df = build_labels(symbol=sym, horizon=horizon,
                          raw_store=store, calendar=cal, as_of=as_of)

        if df.height == 0:
            # Either insufficient data or as_of before any valid forward window.
            continue
        max_t_h = df["t_horizon"].max()
        assert max_t_h <= as_of, (
            f"trial={trial} sym={sym} horizon={horizon} as_of={as_of}: "
            f"emitted t_horizon={max_t_h} > as_of"
        )
        # Also: no t > as_of (sanity — t ≤ t_horizon by construction).
        assert df["t"].max() <= as_of


def test_labels_bit_equal_under_truncated_history(tmp_path: Path) -> None:
    """For 100 random (sym, horizon, t) triples, the label at t is bit-equal
    whether `as_of = T_full` or `as_of = T_trunc` provided both include the
    full forward window for t.

    Truncated history must NEVER change a label whose forward bars are
    entirely in `[t, T_trunc]`. This catches forward-peek bugs (e.g., if
    halt_count accidentally read bars past forward_idx).
    """
    store, cal, by = _build_store(tmp_path)
    rng = _seeded()

    full_as_of = datetime(2030, 1, 1, tzinfo=UTC)   # well past data extent

    for trial in range(100):
        horizon = rng.choice(ALL_HORIZONS)
        sym = rng.choice(list(by["5m"].keys()))

        # Build the full label set first.
        df_full = build_labels(symbol=sym, horizon=horizon,
                               raw_store=store, calendar=cal, as_of=full_as_of)
        if df_full.height < 5:
            continue  # too small to truncate meaningfully

        # Pick a target row near the middle, then truncate as_of to ≥ that
        # row's t_horizon (so the row remains valid) but before any later
        # rows could possibly contribute.
        target_idx = rng.randint(0, df_full.height - 1)
        t_target = df_full["t"][target_idx]
        t_h_target = df_full["t_horizon"][target_idx]
        # as_of_trunc EXACTLY at t_h_target — the row remains valid (its
        # forward bar's effective_close == as_of). Anything strictly later
        # should be excluded.
        as_of_trunc = t_h_target

        df_trunc = build_labels(symbol=sym, horizon=horizon,
                                raw_store=store, calendar=cal, as_of=as_of_trunc)

        # The target row must appear in df_trunc with bit-equal values.
        match = df_trunc.filter(pl.col("t") == t_target)
        assert match.height == 1, (
            f"trial={trial} sym={sym} horizon={horizon}: "
            f"target row at t={t_target} dropped under as_of_trunc={as_of_trunc}"
        )
        full_row = df_full.filter(pl.col("t") == t_target).row(0, named=True)
        trunc_row = match.row(0, named=True)
        for col in ("t_horizon", "y_log_return", "valid_label", "halt_count"):
            assert full_row[col] == trunc_row[col], (
                f"trial={trial} sym={sym} horizon={horizon} col={col}: "
                f"full={full_row[col]} trunc={trunc_row[col]}"
            )


def test_labels_strict_intraday_never_cross_session_boundary(tmp_path: Path) -> None:
    """Intraday horizons (30m, 120m) emit labels strictly within a session.
    For every output row, t and t_horizon share the same UTC date.
    """
    store, cal, by = _build_store(tmp_path)
    full_as_of = datetime(2030, 1, 1, tzinfo=UTC)

    for horizon in HORIZONS_INTRADAY:
        for sym in list(by["5m"].keys()):
            df = build_labels(symbol=sym, horizon=horizon,
                              raw_store=store, calendar=cal, as_of=full_as_of)
            if df.height == 0:
                continue
            t_dates = df["t"].dt.date()
            t_h_dates = df["t_horizon"].dt.date()
            assert (t_dates == t_h_dates).all(), (
                f"{horizon}/{sym}: an intraday label spans the day boundary"
            )


def test_labels_daily_t_horizon_is_n_trading_days_forward(tmp_path: Path) -> None:
    """For 1d / 2d, t_horizon is exactly N trading days after t (per NSE calendar)."""
    store, cal, by = _build_store(tmp_path)
    full_as_of = datetime(2030, 1, 1, tzinfo=UTC)

    for horizon in HORIZONS_DAILY:
        n = HORIZON_DAY_COUNT[horizon]
        for sym in list(by["day"].keys()):
            df = build_labels(symbol=sym, horizon=horizon,
                              raw_store=store, calendar=cal, as_of=full_as_of)
            if df.height == 0:
                continue
            for i in range(min(df.height, 30)):  # sample first 30 rows
                t_d = df["t"][i].date()
                t_h_d = df["t_horizon"][i].date()
                expected = cal.shift(t_d, n)
                assert expected == t_h_d, (
                    f"{horizon}/{sym} row {i}: t={t_d}, t_horizon={t_h_d}, "
                    f"expected calendar.shift({t_d}, {n}) = {expected}"
                )
