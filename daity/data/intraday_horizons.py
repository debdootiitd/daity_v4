"""Forecast horizons for the cohort architecture (DESIGN amendment, 2026-05-16).

The cohort model emits 10 calendar-aware horizons per anchor. Each horizon
either fits within today's session, requires next-day data, or spans
multiple trading days. We encode the rule for each horizon and provide a
validity mask so the loss only fires on (anchor, horizon) pairs where the
target is well-defined and lives entirely within the dataset.

Convention:
    - Anchor time is stored in UTC; NSE trading day = 09:15 IST (03:45 UTC)
      to 15:30 IST (10:00 UTC).
    - Intraday horizons (1-4) are valid iff `anchor + horizon_minutes <= close`.
    - to_close (5) is valid iff anchor strictly before close (else 0-length).
    - Overnight + next-day horizons (6-8) need the calendar's next trading
      day to exist within the data window.
    - Multi-day horizons (9-10) need n trading days after anchor to exist.

The forecast target itself is a cumulative log return:
    log(close_{anchor + horizon} / close_{anchor})
where `close_{anchor}` is the price at the anchor bar's close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time as dtime, timedelta

# NSE session boundaries in UTC.
NSE_OPEN_UTC: dtime = dtime(3, 45)
NSE_CLOSE_UTC: dtime = dtime(10, 0)
NSE_OPEN_IST_MINUTES: int = 9 * 60 + 15   # 09:15 IST
NSE_CLOSE_IST_MINUTES: int = 15 * 60 + 30  # 15:30 IST


@dataclass(frozen=True, slots=True)
class Horizon:
    """One forecast horizon's identity + validity rule.

    Fields:
        name: stable string key (matches head module name).
        kind: 'intraday' | 'to_close' | 'overnight' | 'next_day_open' |
              'next_day_eod' | 'multi_day'.
        minutes_intraday: for 'intraday' kind only — minutes after anchor.
        trading_days_offset: for 'multi_day' / 'next_day_*' — count of trading
          days after the anchor's trading day. 0 = same-day, 1 = next.
        target_minutes_into_day: when this horizon lives on a future trading
          day, what minute of that day's session is the target close at?
          - next_day_open: market open + 5min (the +1 5m bar ≈ 09:20 IST)
          - next_day_1h: open + 60min ≈ 10:15 IST
          - next_day_eod: close ≈ 15:30 IST
          - multi_day: close
    """

    name: str
    kind: str  # 'intraday'|'to_close'|'overnight'|'next_day_open'|'next_day_eod'|'multi_day'
    minutes_intraday: int = 0
    trading_days_offset: int = 0
    target_minutes_into_day: int = 0


# The 10 heads, in their canonical order. Index in this tuple == head ID.
HORIZONS: tuple[Horizon, ...] = (
    Horizon(name="intraday_30m",  kind="intraday", minutes_intraday=30),
    Horizon(name="intraday_60m",  kind="intraday", minutes_intraday=60),
    Horizon(name="intraday_120m", kind="intraday", minutes_intraday=120),
    Horizon(name="intraday_180m", kind="intraday", minutes_intraday=180),
    Horizon(name="to_close",      kind="to_close"),
    Horizon(name="overnight",     kind="overnight",
            trading_days_offset=1, target_minutes_into_day=0),
    Horizon(name="next_day_1h",   kind="next_day_open",
            trading_days_offset=1, target_minutes_into_day=60),
    Horizon(name="next_day_eod",  kind="next_day_eod",
            trading_days_offset=1,
            target_minutes_into_day=NSE_CLOSE_IST_MINUTES - NSE_OPEN_IST_MINUTES),
    Horizon(name="day_plus_3",    kind="multi_day",
            trading_days_offset=3,
            target_minutes_into_day=NSE_CLOSE_IST_MINUTES - NSE_OPEN_IST_MINUTES),
    Horizon(name="day_plus_5",    kind="multi_day",
            trading_days_offset=5,
            target_minutes_into_day=NSE_CLOSE_IST_MINUTES - NSE_OPEN_IST_MINUTES),
)

N_HORIZONS: int = len(HORIZONS)
HORIZON_NAME_TO_IDX: dict[str, int] = {h.name: i for i, h in enumerate(HORIZONS)}


def anchor_minute_of_day(anchor_ts: datetime) -> int:
    """Minutes-from-midnight-IST for the anchor.

    Returns an int in `[NSE_OPEN_IST_MINUTES, NSE_CLOSE_IST_MINUTES]` for a
    typical 09:55 UTC = 15:25 IST anchor.

    The anchor is expected to mark the *close* of the last input bar
    (e.g., a 09:55 UTC anchor means the model has seen the bar that closed
    at 09:55 UTC = 15:25 IST). Returns 15*60 + 25 = 925 for that case.
    """
    if anchor_ts.tzinfo is None:
        anchor_ts = anchor_ts.replace(tzinfo=UTC)
    ist = anchor_ts.astimezone(UTC) + timedelta(hours=5, minutes=30)
    return ist.hour * 60 + ist.minute


def head_validity_mask(anchor_ts: datetime) -> list[bool]:
    """Return a list of `N_HORIZONS` booleans: True if head h is valid at anchor.

    Validity rule depends only on the anchor's time-of-day; calendar-level
    validity (does the next trading day exist?) is enforced separately by
    the cohort sampler at label-fetch time.
    """
    a = anchor_minute_of_day(anchor_ts)
    out: list[bool] = []
    for h in HORIZONS:
        if h.kind == "intraday":
            out.append(a + h.minutes_intraday <= NSE_CLOSE_IST_MINUTES)
        elif h.kind == "to_close":
            out.append(a < NSE_CLOSE_IST_MINUTES)
        else:
            # overnight / next_day / multi_day — always valid w.r.t. anchor
            # time-of-day. The sampler enforces calendar existence.
            out.append(True)
    return out


def target_datetime(
    anchor_ts: datetime,
    horizon_idx: int,
    *,
    trading_days_after_fn,
) -> datetime | None:
    """Compute the UTC datetime of the bar at which horizon `horizon_idx`
    is realized (i.e., the close at the horizon's endpoint).

    `trading_days_after_fn(d, n) -> date | None` is injected so this module
    has no dependency on the NSE calendar — pass `NSECalendar.add_trading_days`
    (or an equivalent helper). Returns None if the target trading day is
    outside the calendar's range.
    """
    h = HORIZONS[horizon_idx]
    if anchor_ts.tzinfo is None:
        anchor_ts = anchor_ts.replace(tzinfo=UTC)
    if h.kind == "intraday":
        return anchor_ts + timedelta(minutes=h.minutes_intraday)
    if h.kind == "to_close":
        # 15:30 IST = 10:00 UTC, same trading day as anchor.
        ist_anchor = anchor_ts + timedelta(hours=5, minutes=30)
        close_ist = ist_anchor.replace(hour=15, minute=30, second=0, microsecond=0)
        return (close_ist - timedelta(hours=5, minutes=30)).astimezone(UTC)
    # overnight / next_day / multi_day
    ist_anchor = anchor_ts + timedelta(hours=5, minutes=30)
    target_day: date | None = trading_days_after_fn(
        ist_anchor.date(), h.trading_days_offset,
    )
    if target_day is None:
        return None
    # Minute-of-day on the target trading day.
    ist_midnight = datetime.combine(target_day, dtime(0, 0))
    minutes_into = h.target_minutes_into_day or NSE_OPEN_IST_MINUTES
    ist_target = ist_midnight + timedelta(minutes=NSE_OPEN_IST_MINUTES + minutes_into)
    return (ist_target - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def intraday_anchor_grid(
    *,
    start_minute_ist: int = NSE_OPEN_IST_MINUTES + 60,  # 10:15 IST
    end_minute_ist:   int = NSE_CLOSE_IST_MINUTES - 5,  # 15:25 IST
    step_minutes:     int = 30,
) -> list[dtime]:
    """Generate per-day anchor times (IST time-of-day) for cohort training.

    Default: 10:15, 10:45, 11:15, ..., 15:25 — 11 anchors/day. Caller pairs
    each with every trading date in the window.
    """
    out: list[dtime] = []
    m = start_minute_ist
    while m <= end_minute_ist:
        h, mm = divmod(m, 60)
        out.append(dtime(hour=h, minute=mm))
        m += step_minutes
    return out
