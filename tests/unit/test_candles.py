"""Unit tests for daity.data.candles — the canonical timestamp conventions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from daity.data.candles import (
    DAILY_KITE_TO_CLOSE_OFFSET,
    INTERVAL_DURATIONS,
    effective_close,
    kite_daily_ts_to_close,
)

IST = timezone(timedelta(hours=5, minutes=30))


def test_effective_close_5m_open_plus_5min() -> None:
    open_ts = datetime(2026, 4, 29, 9, 15, tzinfo=IST)
    close = effective_close("5m", open_ts)
    assert close == datetime(2026, 4, 29, 9, 20, tzinfo=IST)


def test_effective_close_15m_open_plus_15min() -> None:
    open_ts = datetime(2026, 4, 29, 9, 30, tzinfo=IST)
    assert effective_close("15m", open_ts) == datetime(2026, 4, 29, 9, 45, tzinfo=IST)


def test_effective_close_60m_open_plus_1h() -> None:
    open_ts = datetime(2026, 4, 29, 14, 30, tzinfo=IST)
    assert effective_close("60m", open_ts) == datetime(2026, 4, 29, 15, 30, tzinfo=IST)


def test_effective_close_day_passthrough() -> None:
    """Daily ts already represents the close; helper must NOT shift it."""
    close_ts = datetime(2026, 4, 29, 15, 30, tzinfo=IST)
    assert effective_close("day", close_ts) == close_ts


def test_effective_close_rejects_unknown_interval() -> None:
    with pytest.raises(ValueError, match="Unknown interval"):
        effective_close("2m", datetime(2026, 4, 29, tzinfo=IST))


def test_kite_daily_shift_open_ist_to_close_ist() -> None:
    """Kite returns daily at 00:00 IST = 18:30 UTC prev day; we want 15:30 IST = 10:00 UTC same day."""
    kite_raw = datetime(2026, 4, 29, 0, 0, tzinfo=IST)  # 18:30 UTC 2026-04-28
    shifted = kite_daily_ts_to_close(kite_raw)
    assert shifted == datetime(2026, 4, 29, 15, 30, tzinfo=IST)
    # And in UTC terms:
    assert shifted.astimezone(UTC) == datetime(2026, 4, 29, 10, 0, tzinfo=UTC)


def test_kite_daily_shift_works_in_utc_too() -> None:
    """Operates correctly on tz-aware UTC inputs (the form our pipeline uses)."""
    kite_raw_utc = datetime(2026, 4, 28, 18, 30, tzinfo=UTC)
    shifted = kite_daily_ts_to_close(kite_raw_utc)
    assert shifted == datetime(2026, 4, 29, 10, 0, tzinfo=UTC)


def test_intraday_5m_close_does_not_collide_with_next_15m_close() -> None:
    """Regression: ensures a 5m candle ending at 09:30 has effective_close before
    the 15m candle ending at 09:30 — they're equal but never AFTER, so as-of joins
    of `5m` with `15m` context never accidentally include the still-open 15m bar."""
    # 5m bar starting at 09:25 → closes 09:30
    five_min = effective_close("5m", datetime(2026, 4, 29, 9, 25, tzinfo=IST))
    # 15m bar starting at 09:15 → closes 09:30
    fifteen = effective_close("15m", datetime(2026, 4, 29, 9, 15, tzinfo=IST))
    assert five_min == fifteen
    # And the next 15m bar (starts 09:30, closes 09:45) is strictly later.
    next_fifteen = effective_close("15m", datetime(2026, 4, 29, 9, 30, tzinfo=IST))
    assert next_fifteen > five_min


def test_interval_durations_cover_all_design_resolutions() -> None:
    for label in ("5m", "15m", "60m", "day"):
        assert label in INTERVAL_DURATIONS


def test_daily_kite_to_close_offset_is_15h30m() -> None:
    assert timedelta(hours=15, minutes=30) == DAILY_KITE_TO_CLOSE_OFFSET
