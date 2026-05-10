"""Unit tests for the NSE trading-day calendar.

Uses a small synthetic calendar (3 weeks, 1 mid-period holiday) to exercise the
membership / step / `as_of` behaviors. The BQ pull path is NOT exercised here —
that is covered by an integration test in tests/leakage/.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from daity.data.calendar_nse import Holiday, NSECalendar, _to_date


def _calendar() -> NSECalendar:
    """Three Mon-Fri weeks (2026-01-05..2026-01-23), with 2026-01-14 missing."""
    weekdays: list[date] = []
    d = date(2026, 1, 5)
    while d <= date(2026, 1, 23):
        if d.weekday() < 5:  # Mon-Fri
            weekdays.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    # Drop one trading day to simulate a holiday.
    weekdays = [d for d in weekdays if d != date(2026, 1, 14)]
    holidays = [
        Holiday(trade_date=date(2026, 1, 14), description="Republic Day adjacent", source_year=2026)
    ]
    return NSECalendar(trading_days=weekdays, holidays=holidays)


def test_to_date_handles_all_inputs() -> None:
    assert _to_date(date(2026, 1, 1)) == date(2026, 1, 1)
    assert _to_date(datetime(2026, 1, 1, 9, 15)) == date(2026, 1, 1)
    assert _to_date("2026-01-01") == date(2026, 1, 1)
    assert _to_date("2026-01-01T09:15:00+00:00") == date(2026, 1, 1)


def test_to_date_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        _to_date(123)  # type: ignore[arg-type]


def test_calendar_rejects_empty_trading_days() -> None:
    with pytest.raises(ValueError):
        NSECalendar(trading_days=[], holidays=[])


def test_first_last_and_membership() -> None:
    cal = _calendar()
    assert cal.first_day == date(2026, 1, 5)
    assert cal.last_day == date(2026, 1, 23)
    assert cal.is_trading_day(date(2026, 1, 5))  # Monday
    assert not cal.is_trading_day(date(2026, 1, 10))  # Saturday
    assert not cal.is_trading_day(date(2026, 1, 14))  # holiday-removed


def test_trading_days_inclusive_range() -> None:
    cal = _calendar()
    days = cal.trading_days(date(2026, 1, 12), date(2026, 1, 16))
    # Mon 12, Tue 13, [Wed 14 = holiday], Thu 15, Fri 16
    assert days == [date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 15), date(2026, 1, 16)]


def test_trading_days_as_of_caps_upper_bound() -> None:
    cal = _calendar()
    days = cal.trading_days(date(2026, 1, 12), date(2026, 1, 23), as_of=date(2026, 1, 15))
    assert days == [date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 15)]


def test_trading_days_empty_when_lo_greater_than_hi() -> None:
    cal = _calendar()
    assert cal.trading_days(date(2026, 1, 20), date(2026, 1, 10)) == []


def test_next_trading_day_skips_weekends_and_holidays() -> None:
    cal = _calendar()
    # Friday 2026-01-09 → next is Monday 2026-01-12
    assert cal.next_trading_day(date(2026, 1, 9)) == date(2026, 1, 12)
    # Tuesday 2026-01-13 → next is Thursday 2026-01-15 (Wed is holiday)
    assert cal.next_trading_day(date(2026, 1, 13)) == date(2026, 1, 15)


def test_next_trading_day_returns_none_past_end() -> None:
    cal = _calendar()
    assert cal.next_trading_day(date(2026, 1, 23)) is None


def test_next_trading_day_honors_as_of() -> None:
    cal = _calendar()
    # next of Tue 13 would be Thu 15; with as_of=Wed 14 there is no eligible day
    assert cal.next_trading_day(date(2026, 1, 13), as_of=date(2026, 1, 14)) is None
    # but with as_of=Thu 15 it should return Thu 15
    assert cal.next_trading_day(date(2026, 1, 13), as_of=date(2026, 1, 15)) == date(2026, 1, 15)


def test_prev_trading_day() -> None:
    cal = _calendar()
    assert cal.prev_trading_day(date(2026, 1, 12)) == date(2026, 1, 9)
    assert cal.prev_trading_day(date(2026, 1, 15)) == date(2026, 1, 13)
    assert cal.prev_trading_day(date(2026, 1, 5)) is None


def test_shift_zero_returns_self_only_if_trading_day() -> None:
    cal = _calendar()
    assert cal.shift(date(2026, 1, 5), 0) == date(2026, 1, 5)
    assert cal.shift(date(2026, 1, 10), 0) is None  # Saturday


def test_shift_forward_from_non_trading_day() -> None:
    cal = _calendar()
    # Saturday 2026-01-10 + 1 step → Monday 2026-01-12
    assert cal.shift(date(2026, 1, 10), 1) == date(2026, 1, 12)


def test_shift_forward_skips_holiday() -> None:
    cal = _calendar()
    # Tue 2026-01-13 +1 step → Thu 2026-01-15 (Wed is holiday)
    assert cal.shift(date(2026, 1, 13), 1) == date(2026, 1, 15)
    # Tue 2026-01-13 +2 steps → Fri 2026-01-16
    assert cal.shift(date(2026, 1, 13), 2) == date(2026, 1, 16)


def test_shift_backward() -> None:
    cal = _calendar()
    # Thu 2026-01-15 -1 step → Tue 2026-01-13 (Wed is holiday)
    assert cal.shift(date(2026, 1, 15), -1) == date(2026, 1, 13)
    # Mon 2026-01-12 -3 steps → trading days: Fri 9 (-1), Thu 8 (-2), Wed 7 (-3)
    assert cal.shift(date(2026, 1, 12), -3) == date(2026, 1, 7)
    # From a non-trading day: Sunday 11 -1 → snap-back to Friday 9 = step 1
    assert cal.shift(date(2026, 1, 11), -1) == date(2026, 1, 9)


def test_shift_returns_none_past_bounds() -> None:
    cal = _calendar()
    assert cal.shift(date(2026, 1, 5), -10) is None
    assert cal.shift(date(2026, 1, 23), 10) is None


def test_shift_forward_capped_by_as_of() -> None:
    cal = _calendar()
    # Tue 13 +1 step would normally be Thu 15; with as_of=Wed 14 it returns None
    assert cal.shift(date(2026, 1, 13), 1, as_of=date(2026, 1, 14)) is None
    # With as_of=Thu 15 the +1 step is allowed
    assert cal.shift(date(2026, 1, 13), 1, as_of=date(2026, 1, 15)) == date(2026, 1, 15)


def test_n_trading_days_between() -> None:
    cal = _calendar()
    # Mon 5 .. Fri 9 inclusive = 5 trading days
    assert cal.n_trading_days_between(date(2026, 1, 5), date(2026, 1, 9)) == 5
    # Mon 12 .. Fri 16 inclusive, minus holiday Wed 14 = 4
    assert cal.n_trading_days_between(date(2026, 1, 12), date(2026, 1, 16)) == 4


def test_holidays_filter_by_year() -> None:
    cal = _calendar()
    assert len(cal.holidays()) == 1
    assert len(cal.holidays(year=2026)) == 1
    assert cal.holidays(year=2025) == []


def test_cache_round_trip(tmp_path: Path) -> None:
    cal = _calendar()
    cal.write_cache(tmp_path)
    loaded = NSECalendar.from_cache(tmp_path)
    assert list(loaded.trading_days()) == list(cal.trading_days())
    assert [h.trade_date for h in loaded.holidays()] == [h.trade_date for h in cal.holidays()]
    assert [h.description for h in loaded.holidays()] == [h.description for h in cal.holidays()]


def test_from_cache_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NSECalendar.from_cache(tmp_path)
