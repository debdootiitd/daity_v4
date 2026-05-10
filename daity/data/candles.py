"""Candle timestamp conventions for daity_v3.

There is a deliberate asymmetry between intraday and daily candles in how
their `ts` column is interpreted, driven entirely by point-in-time
correctness:

- **Intraday (`5m`, `15m`, `60m`)**: `ts` is the candle's *open* time. The
  candle's data (O, H, L, C, V) only becomes available `interval_duration`
  later, at the close of the bar. So an as-of join at `as_of` may safely use
  the candle iff `ts + interval_duration <= as_of`.
- **Daily (`day`)**: `ts` is the candle's *close* time, **15:30 IST = 10:00
  UTC** of the trading day. This matches when the daily O/H/L/C/V actually
  becomes known (NSE regular session ends at 15:30 IST). An as-of join at
  `as_of` may safely use the candle iff `ts <= as_of`.

Both rules collapse into a single helper: callers should always join on
`effective_close(interval, ts) <= as_of`. The intraday-vs-daily semantic
mismatch is hidden behind that helper so as-of-join code stays uniform.

Why store daily at the close instead of the open?
  Storing daily at the open (09:15 IST) would mean a naive `ts <= as_of`
  join at, say, 09:30 IST would happily return today's daily close — pure
  lookahead. Forcing daily ts to be the close removes the foot-gun.

If you ever add a new interval, register it in `INTERVAL_DURATIONS` and
decide explicitly whether ts represents open or close (default: open).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Open-to-close duration per interval label. Used both for as-of join logic
# and for shifting daily candles from Kite's open-time convention to our
# close-time convention.
INTERVAL_DURATIONS: dict[str, timedelta] = {
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "60m": timedelta(hours=1),
    # NSE regular session: 09:15 IST → 15:30 IST = 6h15m.
    "day": timedelta(hours=6, minutes=15),
}

# Intervals whose stored `ts` represents the *close* of the bar. Anything
# not listed here defaults to open-time semantics.
CLOSE_TIME_INTERVALS: frozenset[str] = frozenset({"day"})

# IST is the trading-day boundary; daily ts lives at 15:30 IST (10:00 UTC).
IST_TZ = timezone(timedelta(hours=5, minutes=30))

# Constant offset to shift Kite's raw daily timestamps (00:00 IST = 18:30 UTC
# previous day) to our convention (15:30 IST = 10:00 UTC same day).
DAILY_KITE_TO_CLOSE_OFFSET = timedelta(hours=15, minutes=30)


def effective_close(interval: str, ts: datetime) -> datetime:
    """Return the moment a candle's data becomes available for as-of joins.

    For intervals where `ts` is open-time, this is `ts + interval_duration`.
    For intervals in `CLOSE_TIME_INTERVALS` (currently just `day`), `ts`
    already *is* the close time, so the helper is a passthrough.

    Use this anywhere you join multiple intervals on `as_of`:

        cond = effective_close(iv, ts) <= as_of

    rather than reasoning about open-vs-close per call site.
    """
    if interval not in INTERVAL_DURATIONS:
        msg = f"Unknown interval {interval!r}; expected one of {list(INTERVAL_DURATIONS)}"
        raise ValueError(msg)
    if interval in CLOSE_TIME_INTERVALS:
        return ts
    return ts + INTERVAL_DURATIONS[interval]


def kite_daily_ts_to_close(ts: datetime) -> datetime:
    """Shift a Kite-raw daily timestamp (00:00 IST same day = 18:30 UTC prev day)
    to our close-time convention (15:30 IST same day = 10:00 UTC same day).

    This is a fixed `+15h30m` shift applied at ingest time. Other intervals
    keep their open-time semantics; only daily is rewritten because Kite
    returns it at the wrong end of the bar for our purposes.
    """
    return ts + DAILY_KITE_TO_CLOSE_OFFSET
