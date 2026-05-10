"""NSE trading-day calendar (point-in-time safe).

Wraps the pre-existing BigQuery tables `nse_trading_calendar` and `nse_holidays`
(captured in Phase 0 audit) so the rest of the codebase never touches BQ for
calendar lookups. The tables are tiny (~1.7K + ~172 rows), so we pull them once
into a local Parquet cache and answer every subsequent question in-memory.

Every public method that returns trading days accepts `as_of: date | None`. When
`as_of` is set, no trading day strictly after `as_of` is ever returned. This is
the single mechanism that prevents lookahead leakage at split / label time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from daity.utils.logging import get_logger

if TYPE_CHECKING:
    from daity.data.bq import BQClient

log = get_logger(__name__)

CALENDAR_TABLE = "nse_trading_calendar"
HOLIDAYS_TABLE = "nse_holidays"

# Default cache layout under DAITY_CACHE_ROOT (resolved by the caller).
CACHE_SUBDIR = "calendar"
CALENDAR_PARQUET = "trading_days.parquet"
HOLIDAYS_PARQUET = "holidays.parquet"


@dataclass(frozen=True, slots=True)
class Holiday:
    """A single NSE non-trading day with its annotation."""

    trade_date: date
    description: str
    source_year: int | None


class NSECalendar:
    """Point-in-time NSE trading-day calendar.

    Construct via `NSECalendar.from_bq(...)` (pulls and caches) or
    `NSECalendar.from_cache(...)` (reads the pre-pulled Parquet files).
    """

    def __init__(self, trading_days: list[date], holidays: list[Holiday]) -> None:
        if not trading_days:
            msg = "Empty trading-day list — refusing to construct calendar."
            raise ValueError(msg)
        # Sorted, deduped, immutable view.
        self._trading_days: tuple[date, ...] = tuple(sorted(set(trading_days)))
        self._holidays: tuple[Holiday, ...] = tuple(
            sorted(holidays, key=lambda h: h.trade_date)
        )
        self._trading_set: frozenset[date] = frozenset(self._trading_days)

    # ----- Construction -----

    @classmethod
    def from_bq(
        cls,
        bq: BQClient,
        *,
        cache_root: Path | None = None,
        force_refresh: bool = False,
    ) -> NSECalendar:
        """Pull both tables from BQ, write Parquet caches, return a calendar.

        If `cache_root` is provided and a cache exists, reuse it unless
        `force_refresh=True`.
        """
        if cache_root is not None and not force_refresh:
            cached = cls._try_from_cache(cache_root)
            if cached is not None:
                log.info("NSECalendar: reusing cache at %s", cache_root)
                return cached

        log.info("NSECalendar: fetching %s + %s from BigQuery", CALENDAR_TABLE, HOLIDAYS_TABLE)
        cal_rows = bq.query_rows(
            f"SELECT trade_date FROM `{bq.cfg.fq_table(CALENDAR_TABLE)}` "
            f"ORDER BY trade_date"
        )
        hol_rows = bq.query_rows(
            f"SELECT trade_date, description, source_year "
            f"FROM `{bq.cfg.fq_table(HOLIDAYS_TABLE)}` ORDER BY trade_date"
        )

        trading_days = [_to_date(r["trade_date"]) for r in cal_rows]
        holidays = [
            Holiday(
                trade_date=_to_date(r["trade_date"]),
                description=str(r.get("description") or ""),
                source_year=(int(r["source_year"]) if r.get("source_year") is not None else None),
            )
            for r in hol_rows
        ]

        cal = cls(trading_days=trading_days, holidays=holidays)
        if cache_root is not None:
            cal.write_cache(cache_root)
        return cal

    @classmethod
    def from_cache(cls, cache_root: Path) -> NSECalendar:
        """Load from a previously-written Parquet cache. Raises if absent."""
        cached = cls._try_from_cache(cache_root)
        if cached is None:
            msg = f"No calendar cache under {cache_root / CACHE_SUBDIR}"
            raise FileNotFoundError(msg)
        return cached

    @classmethod
    def _try_from_cache(cls, cache_root: Path) -> NSECalendar | None:
        d = Path(cache_root) / CACHE_SUBDIR
        cal_p = d / CALENDAR_PARQUET
        hol_p = d / HOLIDAYS_PARQUET
        if not cal_p.exists() or not hol_p.exists():
            return None
        cal_df = pl.read_parquet(cal_p)
        hol_df = pl.read_parquet(hol_p)
        trading_days = [r for r in cal_df["trade_date"].to_list()]
        holidays = [
            Holiday(
                trade_date=row["trade_date"],
                description=row.get("description") or "",
                source_year=row.get("source_year"),
            )
            for row in hol_df.iter_rows(named=True)
        ]
        return cls(trading_days=trading_days, holidays=holidays)

    def write_cache(self, cache_root: Path) -> None:
        """Write the calendar + holidays as Parquet under `cache_root/calendar/`."""
        d = Path(cache_root) / CACHE_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"trade_date": list(self._trading_days)}).write_parquet(
            d / CALENDAR_PARQUET
        )
        pl.DataFrame(
            {
                "trade_date": [h.trade_date for h in self._holidays],
                "description": [h.description for h in self._holidays],
                "source_year": [h.source_year for h in self._holidays],
            }
        ).write_parquet(d / HOLIDAYS_PARQUET)
        log.info("NSECalendar: cached %d trading days + %d holidays at %s",
                 len(self._trading_days), len(self._holidays), d)

    # ----- Read API -----

    @property
    def first_day(self) -> date:
        return self._trading_days[0]

    @property
    def last_day(self) -> date:
        return self._trading_days[-1]

    def __len__(self) -> int:
        return len(self._trading_days)

    def is_trading_day(self, d: date | datetime) -> bool:
        """True iff `d` is a known NSE trading day. Pure membership; no `as_of`."""
        return _to_date(d) in self._trading_set

    def trading_days(
        self,
        start: date | datetime | None = None,
        end: date | datetime | None = None,
        *,
        as_of: date | datetime | None = None,
    ) -> list[date]:
        """Return trading days in `[start, end]`, masked by `as_of` if provided.

        Bounds are inclusive. `as_of` enforces point-in-time correctness — if
        set, no day strictly after `as_of` is returned even if `end` is later.
        """
        lo = _to_date(start) if start is not None else self.first_day
        hi = _to_date(end) if end is not None else self.last_day
        if as_of is not None:
            cap = _to_date(as_of)
            hi = min(hi, cap)
        if lo > hi:
            return []
        # Trading days are sorted; binary search would be tighter but this list
        # is tiny (~1.7K entries) — linear is fine and avoids a bisect import.
        return [d for d in self._trading_days if lo <= d <= hi]

    def next_trading_day(
        self,
        d: date | datetime,
        *,
        as_of: date | datetime | None = None,
    ) -> date | None:
        """Smallest trading day strictly greater than `d`, or `None` if none.

        Honors `as_of`: if the next trading day would exceed `as_of`, returns `None`.
        """
        target = _to_date(d)
        cap = _to_date(as_of) if as_of is not None else None
        for td in self._trading_days:
            if td <= target:
                continue
            if cap is not None and td > cap:
                return None
            return td
        return None

    def prev_trading_day(self, d: date | datetime) -> date | None:
        """Largest trading day strictly less than `d`, or `None`."""
        target = _to_date(d)
        last: date | None = None
        for td in self._trading_days:
            if td >= target:
                break
            last = td
        return last

    def shift(
        self,
        d: date | datetime,
        n: int,
        *,
        as_of: date | datetime | None = None,
    ) -> date | None:
        """Return the trading day `n` steps from `d` (positive = forward).

        - If `n == 0`: returns `d` itself iff it is a trading day, else `None`.
        - If `n > 0`: the *n-th* trading day strictly after `d` when `d` is a
          trading day; when `d` is not, the snap to the next trading day counts
          as step 1 (so `Sat + 1 == Mon`, not Tue).
        - If `n < 0`: symmetric backwards.
        - With `as_of` set, returns `None` if the resulting day would exceed it.
        """
        target = _to_date(d)
        cap = _to_date(as_of) if as_of is not None else None

        if n == 0:
            return target if target in self._trading_set else None

        if n > 0:
            cur: date | None = target
            steps_remaining = n
            if cur not in self._trading_set:
                cur = self.next_trading_day(cur, as_of=cap)
                steps_remaining -= 1  # snap itself counts as one forward step
            for _ in range(steps_remaining):
                if cur is None:
                    return None
                cur = self.next_trading_day(cur, as_of=cap)
            return cur

        # n < 0 — `prev_trading_day` cannot exceed `as_of`, so no cap check needed.
        cur2: date | None = target
        steps_remaining = -n
        if cur2 not in self._trading_set:
            cur2 = self.prev_trading_day(cur2)
            steps_remaining -= 1
        for _ in range(steps_remaining):
            if cur2 is None:
                return None
            cur2 = self.prev_trading_day(cur2)
        return cur2

    def n_trading_days_between(
        self,
        start: date | datetime,
        end: date | datetime,
        *,
        as_of: date | datetime | None = None,
    ) -> int:
        """Count trading days in the inclusive interval `[start, end]`."""
        return len(self.trading_days(start, end, as_of=as_of))

    def holidays(self, year: int | None = None) -> list[Holiday]:
        """All known holidays, optionally filtered by `source_year`."""
        if year is None:
            return list(self._holidays)
        return [h for h in self._holidays if h.source_year == year]


def _to_date(v: date | datetime | str) -> date:
    """Coerce a `date`, `datetime`, or ISO string to `date`."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    msg = f"Cannot coerce {type(v).__name__} to date: {v!r}"
    raise TypeError(msg)
