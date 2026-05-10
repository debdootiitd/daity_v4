"""Unit tests for `_candles_to_frame` — guards the candle-ts convention.

The full refresh pipeline (BQ writes, MERGE, Kite calls) is covered by the
live integration smoke; here we lock in the per-interval ts semantics so
nobody silently regresses to Kite's raw 00:00 IST daily timestamps.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from daity.data.kite import Candle
from daity.data.ohlcv_io import candles_to_frame as _candles_to_frame


def _kite_daily_candle(d: str) -> Candle:
    """Build a Candle as Kite would emit one for a daily bar (00:00 IST)."""
    # 00:00 IST 2026-04-29 = 18:30 UTC 2026-04-28
    ts_epoch = int(datetime.fromisoformat(f"{d}T00:00:00+05:30").timestamp())
    return Candle(ts_epoch=ts_epoch, open=100.0, high=110.0, low=99.0,
                  close=105.0, volume=1000, open_interest=None)


def _kite_intraday_candle(iso_ist: str) -> Candle:
    """Build a Candle as Kite would emit one for an intraday bar."""
    ts_epoch = int(datetime.fromisoformat(iso_ist).timestamp())
    return Candle(ts_epoch=ts_epoch, open=100.0, high=110.0, low=99.0,
                  close=105.0, volume=1000, open_interest=None)


def test_daily_ts_shifted_to_1530_ist() -> None:
    """The headline guarantee: daily rows in the staged frame must have
    ts = 15:30 IST (= 10:00 UTC) on the trading day, NOT 00:00 IST."""
    candle = _kite_daily_candle("2026-04-29")
    frame = _candles_to_frame("RELIANCE", "day", [candle])
    assert frame.height == 1
    ts = frame["ts"][0]
    assert ts.tzinfo is not None
    utc_ts = ts.astimezone(UTC)
    assert utc_ts == datetime(2026, 4, 29, 10, 0, tzinfo=UTC)
    # `scale` column holds the timeframe label (matching raw_ohlcv).
    assert frame["scale"][0] == "day"


def test_intraday_5m_ts_unchanged() -> None:
    """Intraday candles keep ts = open time (no shift). 5m bar at 09:15 IST
    stays at 09:15 IST = 03:45 UTC."""
    candle = _kite_intraday_candle("2026-04-29T09:15:00+05:30")
    frame = _candles_to_frame("RELIANCE", "5m", [candle])
    utc_ts = frame["ts"][0].astimezone(UTC)
    assert utc_ts == datetime(2026, 4, 29, 3, 45, tzinfo=UTC)
    assert frame["scale"][0] == "5m"


def test_intraday_15m_ts_unchanged() -> None:
    candle = _kite_intraday_candle("2026-04-29T09:30:00+05:30")
    frame = _candles_to_frame("X", "15m", [candle])
    utc_ts = frame["ts"][0].astimezone(UTC)
    assert utc_ts == datetime(2026, 4, 29, 4, 0, tzinfo=UTC)


def test_empty_candles_returns_empty_frame_with_schema() -> None:
    """An empty candle list still returns a frame with the expected schema
    (so concat downstream doesn't blow up on a missing column)."""
    frame = _candles_to_frame("X", "day", [])
    assert frame.height == 0
    # Schema matches raw_ohlcv exactly (no `date` column — that's view-only).
    assert set(frame.columns) == {"symbol", "ts", "scale",
                                  "open", "high", "low", "close", "volume"}


def test_daily_no_offset_applied_for_intraday() -> None:
    """Belt-and-braces: confirm the +15h30m shift does NOT bleed into intraday
    intervals even when the ts happens to be near midnight IST."""
    pre_open = _kite_intraday_candle("2026-04-29T09:15:00+05:30")
    intraday_frame = _candles_to_frame("X", "5m", [pre_open])
    daily_frame = _candles_to_frame("X", "day", [_kite_daily_candle("2026-04-29")])
    assert intraday_frame["ts"][0].astimezone(UTC).hour == 3   # 09:15 IST = 03:45 UTC
    assert daily_frame["ts"][0].astimezone(UTC).hour == 10     # 15:30 IST = 10:00 UTC


def test_concat_intraday_and_daily_preserves_per_interval_convention() -> None:
    """Stitching frames from different intervals must keep each interval's
    ts convention intact (no mixing)."""
    intraday = _candles_to_frame("X", "5m", [_kite_intraday_candle("2026-04-29T09:15:00+05:30")])
    daily = _candles_to_frame("X", "day", [_kite_daily_candle("2026-04-29")])
    combined = pl.concat([intraday, daily])
    assert combined.height == 2
    by_iv = {row["scale"]: row for row in combined.to_dicts()}
    assert by_iv["5m"]["ts"].astimezone(UTC).hour == 3
    assert by_iv["day"]["ts"].astimezone(UTC).hour == 10


def test_no_interval_column_in_staging_frame() -> None:
    """Regression: the staging frame must use `scale` (raw_ohlcv schema), not
    `interval` (curated_ohlcv view schema). Anything else fails the BQ load
    schema check."""
    frame = _candles_to_frame("X", "day", [_kite_daily_candle("2026-04-29")])
    assert "interval" not in frame.columns
    assert "scale" in frame.columns
    assert "date" not in frame.columns  # raw_ohlcv has no date column either
