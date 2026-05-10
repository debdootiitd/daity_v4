"""Leakage gate: walk-forward splits must never overlap or violate embargoes.

This is the cheapest-and-loudest defense against the worst class of bug we
can introduce. If any of these tests fails on a PR, the merge is blocked.

Marked `leakage` so CI can select it explicitly: `pytest -m leakage`.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from daity.data.calendar_nse import Holiday, NSECalendar
from daity.data.splits import Split, WalkForwardSplitter

pytestmark = pytest.mark.leakage


def _calendar() -> NSECalendar:
    """7 years of Mon-Fri trading days, 2018-01-01 .. 2024-12-31."""
    days: list[date] = []
    d = date(2018, 1, 1)
    while d <= date(2024, 12, 31):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    # A handful of mid-year holidays to make sure embargoes track trading days,
    # not calendar days.
    holidays = [
        Holiday(trade_date=date(y, 8, 15), description="Independence Day", source_year=y)
        for y in range(2018, 2025)
        if date(y, 8, 15).weekday() < 5
    ]
    days = [x for x in days if not any(x == h.trade_date for h in holidays)]
    return NSECalendar(trading_days=days, holidays=holidays)


def _trading_days_inclusive(cal: NSECalendar, lo: date, hi: date) -> int:
    return cal.n_trading_days_between(lo, hi)


def test_split_construction_invariant() -> None:
    """The dataclass should reject any malformed ordering at construction."""
    with pytest.raises(ValueError):
        Split(
            name="bad",
            train_start=date(2024, 1, 1), train_end=date(2024, 6, 30),
            val_start=date(2024, 6, 1),  # overlaps train
            val_end=date(2024, 9, 30),
            test_start=date(2024, 10, 1), test_end=date(2024, 12, 31),
            embargo_days=10,
        )


def test_negative_embargo_rejected() -> None:
    with pytest.raises(ValueError):
        Split(
            name="bad",
            train_start=date(2024, 1, 1), train_end=date(2024, 3, 31),
            val_start=date(2024, 4, 1), val_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1), test_end=date(2024, 9, 30),
            embargo_days=-1,
        )


def test_no_overlap_across_windows() -> None:
    """train < val < test, with no overlap, on every fold."""
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    folds = splitter.splits(first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30))
    assert folds, "expected at least one fold"
    for f in folds:
        assert f.train_end < f.val_start, f"{f.name} train/val overlap"
        assert f.val_end < f.test_start, f"{f.name} val/test overlap"
        assert f.train_start <= f.train_end
        assert f.val_start <= f.val_end
        assert f.test_start <= f.test_end


def test_embargo_is_in_trading_days_not_calendar_days() -> None:
    """The gap between train_end and val_start must be exactly `embargo_days+1`
    trading days inclusive — calendar-day gaps will vary across folds because
    of weekends and holidays, but the trading-day count must be exact."""
    cal = _calendar()
    embargo = 10
    splitter = WalkForwardSplitter(
        cal, embargo_days=embargo, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    folds = splitter.splits(first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30))
    for f in folds:
        # train_end -> val_start spans exactly embargo+2 trading days inclusive
        # (one for train_end itself, embargo gap days, one for val_start).
        gap_train_val = _trading_days_inclusive(cal, f.train_end, f.val_start)
        assert gap_train_val == embargo + 2, (
            f"{f.name}: train_end→val_start spans {gap_train_val} trading days, "
            f"expected {embargo + 2} (embargo+2 inclusive)"
        )
        gap_val_test = _trading_days_inclusive(cal, f.val_end, f.test_start)
        assert gap_val_test == embargo + 2, (
            f"{f.name}: val_end→test_start spans {gap_val_test} trading days, "
            f"expected {embargo + 2}"
        )


def test_val_and_test_window_lengths_are_exact() -> None:
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    for f in splitter.splits(first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30)):
        assert _trading_days_inclusive(cal, f.val_start, f.val_end) == 126
        assert _trading_days_inclusive(cal, f.test_start, f.test_end) == 126


def test_train_window_is_expanding() -> None:
    """Train start is fixed at `history_start`; train_end grows fold-by-fold."""
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    folds = splitter.splits(first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30))
    starts = {f.train_start for f in folds}
    assert len(starts) == 1, f"train_start not fixed across folds: {starts}"
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends), "train_end should grow monotonically"
    # Strictly grows fold-to-fold.
    for a, b in zip(train_ends, train_ends[1:], strict=False):
        assert b > a, f"train_end stagnant: {a} -> {b}"


def test_as_of_caps_all_folds() -> None:
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    cap = date(2023, 6, 30)
    folds = splitter.splits(
        first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30), as_of=cap,
    )
    for f in folds:
        for d in (f.train_start, f.train_end, f.val_start, f.val_end,
                  f.test_start, f.test_end):
            assert d <= cap, f"{f.name} day {d} exceeds as_of {cap}"


def test_zero_folds_when_history_too_short() -> None:
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    # Anchor the first test before there is any room for train+val+embargoes.
    folds = splitter.splits(first_test_end=date(2018, 6, 1), last_test_end=date(2018, 12, 31))
    assert folds == []


def test_invalid_construction_args_rejected() -> None:
    cal = _calendar()
    with pytest.raises(ValueError):
        WalkForwardSplitter(cal, embargo_days=-1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(cal, embargo_days=10, val_days=0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(cal, embargo_days=10, test_days=-5)
    with pytest.raises(ValueError):
        WalkForwardSplitter(cal, embargo_days=10, roll_days=0)


def test_first_test_end_snapped_to_trading_day() -> None:
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    # 2022-12-31 is a Saturday. The first fold's test_end should snap forward
    # to the next trading day (Monday 2023-01-02).
    folds = splitter.splits(first_test_end=date(2022, 12, 31), last_test_end=date(2023, 1, 31))
    assert folds
    assert cal.is_trading_day(folds[0].test_end)
    assert folds[0].test_end == date(2023, 1, 2)


def test_fold_names_are_unique_and_stable() -> None:
    cal = _calendar()
    splitter = WalkForwardSplitter(
        cal, embargo_days=10, val_days=126, test_days=126, roll_days=63,
        history_start=date(2018, 1, 1),
    )
    folds = splitter.splits(first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30))
    names = [f.name for f in folds]
    assert len(set(names)) == len(names), f"duplicate fold names: {names}"
    # Re-running the same splitter must produce identical names (no run-stamping).
    folds2 = splitter.splits(first_test_end=date(2022, 12, 30), last_test_end=date(2024, 12, 30))
    assert [f.name for f in folds2] == names
