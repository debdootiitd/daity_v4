"""Unit tests for the diff helper in `daity.scripts.refresh_calendar`.

The script-level orchestration (BQ writes, MERGE) is covered by the live
integration smoke; here we exercise the pure-function diff logic so the
no-mutation semantics + discrepancy detection are guaranteed.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from daity.scripts.refresh_calendar import _diff


def _frame(*days: date) -> pl.DataFrame:
    return pl.DataFrame({"trade_date": list(days)}).with_columns(
        pl.col("trade_date").cast(pl.Date)
    )


def test_diff_pure_extension() -> None:
    existing = _frame(date(2026, 1, 5), date(2026, 1, 6))
    refresh = _frame(date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    d = _diff(existing, refresh)
    assert d["n_new"] == 1
    assert d["first_new"] == date(2026, 1, 7)
    assert d["last_new"] == date(2026, 1, 7)
    assert d["discrepancies"] == []


def test_diff_no_change() -> None:
    f = _frame(date(2026, 1, 5), date(2026, 1, 6))
    d = _diff(f, f)
    assert d["n_new"] == 0
    assert d["discrepancies"] == []


def test_diff_flags_overlap_discrepancy() -> None:
    """A day in prod but absent in refresh on the overlap window is suspicious."""
    existing = _frame(date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    refresh = _frame(date(2026, 1, 5), date(2026, 1, 7))  # Jan 6 missing
    d = _diff(existing, refresh)
    assert d["discrepancies"] == [date(2026, 1, 6)]
    assert d["n_new"] == 0


def test_diff_ignores_dates_outside_overlap_window() -> None:
    """Prod dates after the refresh's max are NOT counted as discrepancies —
    they're just future days the refresh didn't cover."""
    existing = _frame(date(2026, 1, 5), date(2026, 1, 6), date(2027, 1, 5))
    refresh = _frame(date(2026, 1, 5), date(2026, 1, 6))
    d = _diff(existing, refresh)
    assert d["discrepancies"] == []
    assert d["n_new"] == 0


def test_diff_handles_empty_existing() -> None:
    existing = pl.DataFrame({"trade_date": []}, schema={"trade_date": pl.Date})
    refresh = _frame(date(2026, 1, 5), date(2026, 1, 6))
    d = _diff(existing, refresh)
    assert d["n_new"] == 2
    assert d["discrepancies"] == []


def test_diff_overlap_window_is_intersection_not_union() -> None:
    """Regression: prod dates BEFORE the refresh's earliest day are NOT
    discrepancies. The overlap window must be the intersection of the two
    date ranges, not the union.
    """
    # Prod has 7 years of history (2018..2025). Refresh only covers a recent
    # 30-day overlap + forward extension.
    existing = _frame(
        date(2018, 1, 2), date(2020, 6, 30), date(2025, 12, 1), date(2025, 12, 12)
    )
    refresh = _frame(
        date(2025, 11, 12), date(2025, 12, 1), date(2025, 12, 12),
        date(2026, 1, 5), date(2027, 5, 7),
    )
    d = _diff(existing, refresh)
    # Overlap window: [max(2018-01-02, 2025-11-12), min(2025-12-12, 2027-05-07)]
    #                = [2025-11-12, 2025-12-12]
    assert d["overlap_window"] == (date(2025, 11, 12), date(2025, 12, 12))
    # Existing dates inside that overlap: 2025-12-01, 2025-12-12 — both present in refresh.
    # 2018 + 2020 dates are OUTSIDE the overlap and must NOT be discrepancies.
    assert d["discrepancies"] == []
    assert d["n_new"] == 3  # 2025-11-12, 2026-01-05, 2027-05-07
