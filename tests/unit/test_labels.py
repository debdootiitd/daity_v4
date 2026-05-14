"""Unit tests for `daity.data.labels`.

Locks in:
- 5m anchor + forward-bar close arithmetic for 30m and 120m horizons
- 1d / 2d horizons follow the NSE trading calendar (skip weekends/holidays)
- Halt detection: limit-locked + zero-volume bars increment `halt_count`
- Out-of-session anchors (forward bar past day boundary) are dropped
- `y_rank` column is reserved (null) at per-symbol-file write time
- Output schema matches `OUTPUT_SCHEMA`
- Idempotency: precompute_labels skips when output exists with right schema

The leakage gate (no row past `as_of`, bit-equality under truncated history)
lives in `tests/leakage/test_label_leakage.py`.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from daity.data.calendar_nse import Holiday, NSECalendar
from daity.data.labels import (
    ALL_HORIZONS,
    HORIZON_BAR_COUNT,
    HORIZON_DAY_COUNT,
    HORIZONS_DAILY,
    HORIZONS_INTRADAY,
    OUTPUT_SCHEMA,
    build_labels,
    precompute_labels,
)
from daity.data.parquet_store import ParquetStore

IST = timezone(timedelta(hours=5, minutes=30))


# ----- Fixtures -----


def _bars_5m(symbol: str, ts_list: list[datetime],
             closes: list[float],
             volumes: list[int] | None = None,
             highs: list[float] | None = None,
             lows: list[float] | None = None) -> pl.DataFrame:
    n = len(ts_list)
    if volumes is None:
        volumes = [1000] * n
    if highs is None:
        highs = [c + 0.1 for c in closes]
    if lows is None:
        lows = [c - 0.1 for c in closes]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts": ts_list,
            "scale": ["5m"] * n,
            "open": [c - 0.05 for c in closes],
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        schema={
            "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
            "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
        },
    )


def _bars_day(symbol: str, ts_list: list[datetime],
              closes: list[float],
              volumes: list[int] | None = None) -> pl.DataFrame:
    n = len(ts_list)
    if volumes is None:
        volumes = [10000] * n
    return pl.DataFrame(
        {
            "symbol": [symbol] * n, "ts": ts_list, "scale": ["day"] * n,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": volumes,
        },
        schema={
            "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
            "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
        },
    )


def _one_session_ts(d: date) -> list[datetime]:
    """Return the 75 5m bar opens of a single regular session on date `d`.

    NSE regular session: 09:15 IST → 15:30 IST (75 5m bars).
    Stored as UTC, which is `IST_open - 5h30m`.
    """
    open_ist = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
    base_utc = open_ist.astimezone(UTC)
    return [base_utc + timedelta(minutes=5 * k) for k in range(75)]


def _flat_calendar(start: date, end: date) -> NSECalendar:
    """A 5-day-week calendar (no holidays) spanning `[start, end]`."""
    days: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon..Fri
            days.append(cur)
        cur += timedelta(days=1)
    return NSECalendar(trading_days=days, holidays=[])


# ----- Schema / API tests -----


def test_horizon_registries_consistent() -> None:
    """The four registry constants don't disagree."""
    assert set(HORIZONS_INTRADAY + HORIZONS_DAILY) == set(ALL_HORIZONS)
    assert set(HORIZON_BAR_COUNT) == set(HORIZONS_INTRADAY)
    assert set(HORIZON_DAY_COUNT) == set(HORIZONS_DAILY)
    assert HORIZON_BAR_COUNT["30m"] == 6
    assert HORIZON_BAR_COUNT["120m"] == 24
    assert HORIZON_DAY_COUNT["1d"] == 1
    assert HORIZON_DAY_COUNT["2d"] == 2


@pytest.mark.parametrize("horizon", ALL_HORIZONS)
def test_output_schema_present(tmp_path: Path, horizon: str) -> None:
    """Each horizon produces a frame with exactly the documented schema."""
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))

    if horizon in HORIZONS_INTRADAY:
        ts = _one_session_ts(date(2024, 6, 3))   # a Monday
        store.write_partition("5m", "X", _bars_5m("X", ts, [100.0 + i * 0.1 for i in range(75)]))
    else:
        days = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC)
                for d in cal.trading_days(date(2024, 6, 3), date(2024, 7, 3))]
        store.write_partition("day", "X", _bars_day("X", days, [100.0 + i for i in range(len(days))]))

    df = build_labels(symbol="X", horizon=horizon,
                      raw_store=store, calendar=cal,
                      as_of=datetime(2025, 1, 1, tzinfo=UTC))
    assert df.columns == list(OUTPUT_SCHEMA)
    if df.height > 0:
        assert df["y_rank"].null_count() == df.height, "y_rank must be null at per-symbol write time"


# ----- 30m intraday arithmetic -----


def test_30m_label_at_first_anchor(tmp_path: Path) -> None:
    """First valid 30m anchor: bar 0 (effective_close = first_bar_open + 5m).
    Forward bar = bar 6, close at first_bar_open + 30m.
    y = log(close[6] / close[0]).
    """
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    d = date(2024, 6, 3)  # Mon
    ts = _one_session_ts(d)
    closes = [100.0 + i for i in range(75)]   # close[0]=100, close[6]=106
    store.write_partition("5m", "X", _bars_5m("X", ts, closes))

    df = build_labels(symbol="X", horizon="30m", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # Anchor at bar 0: t = bar0.ts + 5m = 09:20 IST = 03:50 UTC.
    # The first row should have t = ts[0] + 5m, t_horizon = ts[6] + 5m.
    five_min = timedelta(minutes=5)
    expected_t = ts[0] + five_min
    expected_t_h = ts[6] + five_min
    assert df.height >= 1
    assert df["t"][0] == expected_t
    assert df["t_horizon"][0] == expected_t_h
    expected_y = math.log(closes[6] / closes[0])
    assert abs(df["y_log_return"][0] - expected_y) < 1e-5


def test_30m_drops_anchors_crossing_session_boundary(tmp_path: Path) -> None:
    """An anchor near end-of-session whose forward bar would cross to next session
    must be dropped — `t + 30m` must lie within the same trading day's regular session.
    """
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    # Two consecutive trading days, each with 75 5m bars.
    ts = _one_session_ts(date(2024, 6, 3)) + _one_session_ts(date(2024, 6, 4))
    closes = [100.0 + i * 0.1 for i in range(150)]
    store.write_partition("5m", "X", _bars_5m("X", ts, closes))

    df = build_labels(symbol="X", horizon="30m", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # Each session has 75 5m bars at indices 0..74. For 30m horizon
    # (n_forward=6), the latest valid anchor is bar 68 (forward bar 74,
    # which closes at session close 15:30). So 69 valid anchors per session.
    # 69 × 2 sessions = 138 labels. (PHASE3.md §3.1's "~70 anchors/day" is
    # rounded; the exact count is 69.)
    assert df.height == 69 * 2

    # Verify no label spans across the day boundary: t and t_horizon must
    # share the same UTC date. (NSE session 09:15-15:30 IST = 03:45-10:00 UTC,
    # entirely within one UTC date.)
    for i in range(df.height):
        t_date = df["t"][i].date()
        t_h_date = df["t_horizon"][i].date()
        assert t_date == t_h_date, f"row {i} crosses day: {t_date} → {t_h_date}"


def test_120m_anchor_count(tmp_path: Path) -> None:
    """120m horizon has ~52 anchors per session: bars whose +24-bar forward
    fits within the same 75-bar session, i.e., bars 0..50 → 51 anchors.
    """
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    ts = _one_session_ts(date(2024, 6, 3))
    closes = [100.0 + i * 0.1 for i in range(75)]
    store.write_partition("5m", "X", _bars_5m("X", ts, closes))

    df = build_labels(symbol="X", horizon="120m", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # Bars 0..50 are valid anchors (bar 50 + 24 = bar 74, the last bar). 51 anchors.
    assert df.height == 51


# ----- 1d / 2d daily arithmetic -----


def test_1d_label_value_matches_close_to_close(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 6, 3), date(2024, 6, 30))
    days = cal.trading_days(date(2024, 6, 3), date(2024, 6, 14))   # 10 trading days
    ts = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]
    closes = [100.0 + i for i in range(len(days))]
    store.write_partition("day", "X", _bars_day("X", ts, closes))

    df = build_labels(symbol="X", horizon="1d", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # 9 valid labels (each anchor 0..8 has a forward day).
    assert df.height == 9
    expected_y0 = math.log(closes[1] / closes[0])
    assert abs(df["y_log_return"][0] - expected_y0) < 1e-6


def test_1d_label_skips_weekends_via_calendar(tmp_path: Path) -> None:
    """Friday's 1d label should target Monday's close, not Saturday."""
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 5, 27), date(2024, 6, 14))   # Mon..Fri spans
    # Days: Mon..Fri × ~3 weeks = 15 trading days. Friday is index 4, Monday is index 5.
    days = cal.trading_days(date(2024, 5, 27), date(2024, 6, 14))
    ts = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]
    closes = [100.0 + i * 1.5 for i in range(len(days))]
    store.write_partition("day", "X", _bars_day("X", ts, closes))

    df = build_labels(symbol="X", horizon="1d", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # The Friday-anchor (index 4) should target the Monday close (index 5).
    fri = days[4]
    fri_row = df.filter(pl.col("t") == datetime(fri.year, fri.month, fri.day, 10, 0, tzinfo=UTC))
    assert fri_row.height == 1
    expected_y = math.log(closes[5] / closes[4])
    assert abs(fri_row["y_log_return"][0] - expected_y) < 1e-6


def test_2d_label_skips_holiday_via_calendar(tmp_path: Path) -> None:
    """A 2d label should skip a known exchange holiday in between.

    Universe of trading days: Mon..Fri × 3 weeks, MINUS Wed 2024-06-12.
    Anchor on Mon 2024-06-10. Forward 2 trading days = (Tue 06-11, Thu 06-13).
    The 2d label at Mon should target Thu's close (skipping the holiday).
    """
    days: list[date] = []
    cur = date(2024, 6, 3)
    while cur <= date(2024, 6, 21):
        if cur.weekday() < 5 and cur != date(2024, 6, 12):
            days.append(cur)
        cur += timedelta(days=1)
    cal = NSECalendar(
        trading_days=days,
        holidays=[Holiday(trade_date=date(2024, 6, 12),
                          description="test", source_year=2024)],
    )

    store = ParquetStore(tmp_path)
    ts = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]
    closes = [100.0 + i * 1.5 for i in range(len(days))]
    store.write_partition("day", "X", _bars_day("X", ts, closes))

    df = build_labels(symbol="X", horizon="2d", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))

    mon_anchor_ts = datetime(2024, 6, 10, 10, 0, tzinfo=UTC)
    mon_row = df.filter(pl.col("t") == mon_anchor_ts)
    assert mon_row.height == 1
    expected_t_horizon = datetime(2024, 6, 13, 10, 0, tzinfo=UTC)  # Thu, not Wed
    assert mon_row["t_horizon"][0] == expected_t_horizon
    mon_idx = days.index(date(2024, 6, 10))
    thu_idx = days.index(date(2024, 6, 13))
    expected_y = math.log(closes[thu_idx] / closes[mon_idx])
    assert abs(mon_row["y_log_return"][0] - expected_y) < 1e-6


# ---- Halt detection tests ----


def test_halt_count_includes_zero_volume_bars(tmp_path: Path) -> None:
    """Forward window containing a volume==0 bar yields halt_count > 0
    and valid_label=False.
    """
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    ts = _one_session_ts(date(2024, 6, 3))
    closes = [100.0 + i * 0.1 for i in range(75)]
    vols = [1000] * 75
    vols[3] = 0   # a halted bar in the middle of the first 30m forward window
    store.write_partition("5m", "X", _bars_5m("X", ts, closes, volumes=vols))

    df = build_labels(symbol="X", horizon="30m", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # Anchor 0's forward window covers bars 1..6, which includes bar 3.
    # halt_count for row 0 should be 1; valid_label False.
    assert df["halt_count"][0] == 1
    assert df["valid_label"][0] is False or df["valid_label"][0] == False  # noqa: E712


def test_halt_count_includes_limit_locked_bars(tmp_path: Path) -> None:
    """Limit-locked bar (high == low, volume > 0) counts as untradeable."""
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    ts = _one_session_ts(date(2024, 6, 3))
    closes = [100.0 + i * 0.1 for i in range(75)]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    # Bar 4 limit-locked: high == low.
    highs[4] = closes[4]
    lows[4] = closes[4]
    store.write_partition("5m", "X", _bars_5m("X", ts, closes, highs=highs, lows=lows))

    df = build_labels(symbol="X", horizon="30m", raw_store=store,
                      calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC))
    # Anchor 0's forward bars 1..6 include bar 4 → halt_count=1.
    assert df["halt_count"][0] == 1


# ---- Idempotency / write tests ----


def test_precompute_labels_writes_and_skips(tmp_path: Path) -> None:
    """precompute_labels writes a parquet on first call, skips on second."""
    raw_root = tmp_path / "parquet"
    label_root = tmp_path / "labels"
    raw_root.mkdir()
    label_root.mkdir()

    store = ParquetStore(raw_root)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    ts = _one_session_ts(date(2024, 6, 3))
    closes = [100.0 + i * 0.1 for i in range(75)]
    store.write_partition("5m", "X", _bars_5m("X", ts, closes))

    n1 = precompute_labels(
        raw_store=store, label_root=label_root, symbol="X", horizon="30m",
        calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    out = label_root / "30m" / "X.parquet"
    assert out.exists()
    assert n1 == 69

    # Second call: idempotent skip — returns same row count, doesn't recompute.
    n2 = precompute_labels(
        raw_store=store, label_root=label_root, symbol="X", horizon="30m",
        calendar=cal, as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert n2 == 69


def test_build_labels_rejects_unknown_horizon(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    cal = _flat_calendar(date(2024, 1, 1), date(2024, 12, 31))
    with pytest.raises(ValueError, match="unknown horizon"):
        build_labels(symbol="X", horizon="7d", raw_store=store, calendar=cal,
                     as_of=datetime(2025, 1, 1, tzinfo=UTC))
