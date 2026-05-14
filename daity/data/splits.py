"""Walk-forward splits with trading-day-aware embargoes.

Implements DESIGN §2.5. For each anchor `T` (end of test window) we produce a
single fold:

    train = [history_start,    train_end ]
    val   = [val_start,        val_end   ]   val_start  = train_end + 1 + embargo (trading days)
    test  = [test_start,       test_end  ]   test_start = val_end   + 1 + embargo (trading days)

`val_end` and `test_start/end` are derived backwards from `T` so the test
window has the requested length in trading days. Anchors march forward by a
configurable trading-day stride (default = one quarter ≈ 63 trading days).

Every operation is funneled through the `NSECalendar`, so the embargo is
expressed in *trading* days rather than calendar days — which is what matters
for a forecast-horizon-based gap.

The splitter never returns days strictly after `as_of` (point-in-time
correctness, per CLAUDE.md). A fold whose test window cannot fit before `as_of`
is silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime

from daity.data.calendar_nse import NSECalendar, _to_date
from daity.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Split:
    """A single walk-forward fold.

    All bounds are inclusive trading days. `name` is a stable, fold-identifying
    string (e.g., `fold_2024-03-29`) suitable for run tagging and cache keys.
    """

    name: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    embargo_days: int

    def __post_init__(self) -> None:
        # The leakage invariant is non-negotiable; assert it at construction.
        if not (
            self.train_start <= self.train_end < self.val_start
            <= self.val_end < self.test_start <= self.test_end
        ):
            msg = (
                f"Split {self.name} violates ordering: train [{self.train_start}, "
                f"{self.train_end}] val [{self.val_start}, {self.val_end}] test "
                f"[{self.test_start}, {self.test_end}]"
            )
            raise ValueError(msg)
        if self.embargo_days < 0:
            msg = f"Split {self.name} has negative embargo {self.embargo_days}"
            raise ValueError(msg)


class WalkForwardSplitter:
    """Generates expanding-window walk-forward splits with embargoes.

    Args:
        calendar: NSECalendar source of trading days.
        embargo_days: trading-day gap between train/val and val/test. Set to
            the longest forecast horizon you intend to use (10 for the swing
            10-day head per DESIGN §3.3).
        val_days, test_days: lengths of val/test windows in trading days.
            Defaults give a ~1-year val and ~1-year test.
        roll_days: trading-day stride between consecutive fold anchors. ~63
            ≈ one quarter, matching DESIGN §2.5.
        history_start: earliest trading day eligible to be in training. If
            `None`, uses the first day in the calendar (subject to `as_of`).
    """

    def __init__(
        self,
        calendar: NSECalendar,
        *,
        embargo_days: int,
        val_days: int = 252,
        test_days: int = 252,
        roll_days: int = 63,
        history_start: date | datetime | None = None,
    ) -> None:
        if embargo_days < 0 or val_days <= 0 or test_days <= 0 or roll_days <= 0:
            msg = (
                f"Invalid splitter params: embargo={embargo_days}, val={val_days}, "
                f"test={test_days}, roll={roll_days}"
            )
            raise ValueError(msg)
        self.calendar = calendar
        self.embargo_days = embargo_days
        self.val_days = val_days
        self.test_days = test_days
        self.roll_days = roll_days
        self.history_start = _to_date(history_start) if history_start is not None else None

    # ----- Generation -----

    def splits(
        self,
        *,
        first_test_end: date | datetime,
        last_test_end: date | datetime | None = None,
        as_of: date | datetime | None = None,
    ) -> list[Split]:
        """Materialize all folds with `test_end` in `[first_test_end, last_test_end]`.

        - `first_test_end` is snapped forward to a trading day if needed.
        - `last_test_end` defaults to the calendar's last day (or `as_of`).
        - Folds whose train window would be empty (e.g., not enough history)
          are skipped with a warning.
        """
        return list(self.iter_splits(first_test_end=first_test_end, last_test_end=last_test_end, as_of=as_of))

    def iter_splits(
        self,
        *,
        first_test_end: date | datetime,
        last_test_end: date | datetime | None = None,
        as_of: date | datetime | None = None,
    ) -> Iterator[Split]:
        """Streaming variant of `splits` — useful for very long histories."""
        cap = _to_date(as_of) if as_of is not None else self.calendar.last_day
        last = _to_date(last_test_end) if last_test_end is not None else cap
        last = min(last, cap)

        # Snap the first test-end forward to a trading day.
        anchor = _to_date(first_test_end)
        if not self.calendar.is_trading_day(anchor):
            nxt = self.calendar.next_trading_day(anchor, as_of=cap)
            if nxt is None:
                log.warning("No trading days at or after first_test_end=%s", first_test_end)
                return
            anchor = nxt

        history_start = self.history_start or self.calendar.first_day

        idx = 0
        while anchor is not None and anchor <= last:
            split = self._build_split(test_end=anchor, history_start=history_start, as_of=cap)
            if split is not None:
                yield split
            else:
                log.warning("Skipping fold at test_end=%s (insufficient history)", anchor)
            idx += 1
            # Stride forward by `roll_days` trading days.
            nxt = self.calendar.shift(anchor, self.roll_days, as_of=cap)
            if nxt is None or nxt <= anchor:
                break
            anchor = nxt

        if idx == 0:
            log.warning(
                "WalkForwardSplitter produced 0 folds for first_test_end=%s last=%s as_of=%s",
                first_test_end, last, as_of,
            )

    # ----- Single-fold construction -----

    def _build_split(  # noqa: PLR0911 — sequential null-checks read clearer as early returns
        self,
        *,
        test_end: date,
        history_start: date,
        as_of: date,
    ) -> Split | None:
        """Build one fold with `test_end` as the anchor.

        Returns `None` if the geometry doesn't fit (e.g., not enough history
        to populate train + val + embargoes before `test_start`).
        """
        cal = self.calendar

        # test_start = test_end stepped back (test_days - 1) trading days.
        test_start = cal.shift(test_end, -(self.test_days - 1))
        if test_start is None:
            return None

        # val_end is one embargo before test_start.
        val_end = cal.shift(test_start, -(self.embargo_days + 1))
        if val_end is None:
            return None

        # val_start = val_end stepped back (val_days - 1) trading days.
        val_start = cal.shift(val_end, -(self.val_days - 1))
        if val_start is None:
            return None

        # train_end is one embargo before val_start.
        train_end = cal.shift(val_start, -(self.embargo_days + 1))
        if train_end is None:
            return None

        # train_start is the first trading day at or after history_start.
        if cal.is_trading_day(history_start):
            train_start = history_start
        else:
            ts = cal.next_trading_day(history_start, as_of=as_of)
            if ts is None:
                return None
            train_start = ts

        if train_start > train_end:
            return None  # not enough history

        try:
            return Split(
                name=f"fold_{test_end.isoformat()}",
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=self.embargo_days,
            )
        except ValueError as exc:
            log.warning("Discarding malformed split at test_end=%s: %s", test_end, exc)
            return None


# ---- Phase 4 OB-window walk-forward (DESIGN amendment 19 — pending) ----
#
# OB sample spans 2026-03-08 → 2026-05-04 (~40 trading days). Phase 4
# acceptance gate operates on its own walk-forward windows (NOT Phase 3's
# F-9..F-12) because Phase 3 folds pre-date the OB window. Four mini-folds
# with expanding-window training give us cross-fold t-stat statistical
# power even at small per-fold sample size.
#
# Fold layout: train_start fixed at OB-window start, train_end grows by 5
# trading days per fold, val is the next 5 trading days. Embargo = 0 (the
# horizon is 30m intraday, no carry-over).

@dataclass(frozen=True, slots=True)
class OBFold:
    """Phase 4 OB fine-tune fold spec. Simpler than `Split` — train + val,
    no test (val IS the held-out evaluation per Phase 4 §1)."""
    name: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date


OB_WALK_FORWARD_FOLDS: tuple[OBFold, ...] = (
    OBFold(
        name="OB-F1",
        train_start=date(2026, 3, 9),  train_end=date(2026, 4, 2),
        val_start=date(2026, 4, 6),    val_end=date(2026, 4, 10),
    ),
    OBFold(
        name="OB-F2",
        train_start=date(2026, 3, 9),  train_end=date(2026, 4, 10),
        val_start=date(2026, 4, 13),   val_end=date(2026, 4, 17),
    ),
    OBFold(
        name="OB-F3",
        train_start=date(2026, 3, 9),  train_end=date(2026, 4, 17),
        val_start=date(2026, 4, 20),   val_end=date(2026, 4, 24),
    ),
    OBFold(
        name="OB-F4",
        train_start=date(2026, 3, 9),  train_end=date(2026, 4, 24),
        val_start=date(2026, 4, 27),   val_end=date(2026, 5, 1),
    ),
)
