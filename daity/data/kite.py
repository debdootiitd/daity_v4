"""Zerodha Kite Connect client (Phase 1 ingest, post-Groww-pivot).

Wraps `kiteconnect.KiteConnect` with the same idioms as `daity.data.groww`:
- Construct from our `KiteConfig` (which reads `.kite_cred.env` via the env
  loader's sidecar mechanism).
- Cached instruments dump (Polars + optional Parquet file under the cache
  root).
- Bulk symbol → instrument_token lookups.
- `candles(...)` with chunked time-window pulls that respect Kite's
  per-interval window caps.

Design choices worth flagging:
- Kite's `historical_data` returns a list of dicts with `date` as a Python
  datetime. We normalize to `Candle.ts_epoch` (UTC seconds) so the staging
  / merge path is identical to whatever was set up for Groww.
- Kite identifiers are numeric `instrument_token`, not string `groww_symbol`.
  All our lookups go through the instruments cache; callers should never
  hard-code tokens.
- Auth is just `set_access_token(...)`. The token is daily-rotating, but
  refreshing it is the responsibility of `daity.scripts.kite_login`, not
  this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests.exceptions
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from daity.utils.env import KiteConfig
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Map our internal interval labels to (kite_interval, max_chunk_days).
# Window caps come from Kite's docs:
#   minute / 3minute               → 60 days/req
#   5minute                        → 100 days/req
#   10minute / 15minute / 30minute → 200 days/req
#   60minute / day                 → 2000 days/req
INTERVAL_MAP: dict[str, tuple[str, int]] = {
    "5m":  ("5minute",  100),
    "15m": ("15minute", 200),
    "60m": ("60minute", 2000),
    "day": ("day",      2000),
}

DEFAULT_EXCHANGE = "NSE"

INSTRUMENTS_CACHE_SUBDIR = "kite"
INSTRUMENTS_PARQUET = "instruments.parquet"

# Kite SDK defaults to a 7s read timeout, which we hit during peak load on the
# historical endpoint. 30s is generous but well under the script's "stuck for
# minutes" pain threshold.
DEFAULT_HTTP_TIMEOUT_SECONDS = 30

# Retry config for the historical endpoint. Exponential backoff (1s, 2s, 4s)
# on transient network errors only — KiteException (which signals an explicit
# server-side reject like bad token / no permission) is NOT retried.
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt

# Kite's historical-data endpoint interprets the `from_date`/`to_date` strings
# as IST (no timezone info is sent — the SDK strftimes without tzinfo). We
# convert any tz-aware input to IST-naive before calling. This is the *only*
# place where IST-vs-UTC matters in the client; everywhere else we use UTC.
IST_TZ = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True, slots=True)
class Candle:
    """One Kite candle, normalized to the same shape as Groww candles.

    `ts_epoch` is UTC seconds (Kite returns IST-aware datetimes; we convert).
    `open_interest` is None for cash-segment instruments.
    """

    ts_epoch: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int | None


class KiteClient:
    """High-level wrapper. Construct from `KiteConfig`."""

    def __init__(self, cfg: KiteConfig, *, api: KiteConnect | None = None) -> None:
        self.cfg = cfg
        self._api = api if api is not None else self._build_api(cfg)
        self._instruments: pl.DataFrame | None = None

    @staticmethod
    def _build_api(cfg: KiteConfig) -> KiteConnect:
        if not cfg.api_key:
            msg = "KITE_API_KEY is empty; cannot construct KiteConnect."
            raise ValueError(msg)
        api = KiteConnect(api_key=cfg.api_key, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS)
        if cfg.access_token:
            api.set_access_token(cfg.access_token)
        return api

    # ----- Instruments -----

    @property
    def instruments(self) -> pl.DataFrame:
        if self._instruments is None:
            self._instruments = self._load_instruments()
        return self._instruments

    def load_instruments_cache(
        self,
        cache_root: Path,
        *,
        exchange: str = DEFAULT_EXCHANGE,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Load instruments from a Parquet cache, refreshing from Kite if missing."""
        cache_dir = Path(cache_root) / INSTRUMENTS_CACHE_SUBDIR
        cache_path = cache_dir / INSTRUMENTS_PARQUET
        if cache_path.exists() and not force_refresh:
            log.info("Kite instruments: reusing cache at %s", cache_path)
            self._instruments = pl.read_parquet(cache_path)
            return self._instruments
        df = self._load_instruments(exchange=exchange)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        log.info("Kite instruments: cached %d rows at %s", df.height, cache_path)
        self._instruments = df
        return df

    def _load_instruments(self, *, exchange: str = DEFAULT_EXCHANGE) -> pl.DataFrame:
        log.info("Kite instruments: fetching dump (exchange=%s)", exchange)
        rows = self._api.instruments(exchange=exchange) if exchange else self._api.instruments()
        # Kite returns list[dict]; normalize via pandas to handle mixed-type columns,
        # then to polars for project consistency.
        import pandas as pd
        return pl.from_pandas(pd.DataFrame(rows))

    # ----- Identifier lookup -----

    def lookup(self, trading_symbol: str, exchange: str = DEFAULT_EXCHANGE) -> dict | None:
        """Return the instruments-dump row for `(exchange, trading_symbol)`, or None."""
        df = self.instruments.filter(
            (pl.col("exchange") == exchange) & (pl.col("tradingsymbol") == trading_symbol)
        )
        if df.height == 0:
            return None
        return df.head(1).to_dicts()[0]

    def bulk_lookup(
        self,
        trading_symbols: list[str],
        *,
        exchange: str = DEFAULT_EXCHANGE,
        instrument_type: str | None = "EQ",
    ) -> pl.DataFrame:
        """Return `trading_symbol → (instrument_token, name, instrument_type, segment)` for many tickers.

        Defaults to `instrument_type='EQ'` so cash-equity rows win over any
        FNO derivatives that happen to share the same trading_symbol root.
        Missing tickers come back with null `instrument_token` so the caller
        can detect gaps.
        """
        df = self.instruments.filter(pl.col("exchange") == exchange)
        if instrument_type is not None and "instrument_type" in df.columns:
            df = df.filter(pl.col("instrument_type") == instrument_type)
        keep = ["tradingsymbol", "instrument_token", "name", "instrument_type", "segment"]
        keep = [c for c in keep if c in df.columns]
        df = df.select(keep).unique(subset=["tradingsymbol"], keep="first")

        wanted = pl.DataFrame({"tradingsymbol": trading_symbols})
        joined = wanted.join(df, on="tradingsymbol", how="left")
        n_missing = joined.filter(pl.col("instrument_token").is_null()).height
        if n_missing:
            log.warning(
                "bulk_lookup: %d / %d tickers missing in Kite instruments dump",
                n_missing,
                len(trading_symbols),
            )
        return joined

    # ----- Candles -----

    def candles(
        self,
        instrument_token: int,
        *,
        interval: str,
        start: datetime | date,
        end: datetime | date,
        oi: bool = False,
    ) -> list[Candle]:
        """Fetch candles for `instrument_token` in `[start, end]`.

        `start` and `end` may be tz-aware (any zone) or naive. Naive datetimes
        are treated as IST (Kite's native zone). The values passed to the SDK
        are converted to IST-naive — Kite's `historical_data` strftimes the
        input without tzinfo and the server interprets the result as IST.
        """
        kite_iv, max_days = self._resolve_interval(interval)
        start_dt = _to_ist_naive(_as_datetime(start))
        end_dt = _to_ist_naive(_as_datetime(end))
        chunks = list(_iter_time_chunks(start_dt, end_dt, max_days=max_days))
        seen: dict[int, Candle] = {}
        for chunk_start, chunk_end in chunks:
            log.debug(
                "Kite candles: token=%d %s [%s, %s] IST",
                instrument_token, kite_iv, chunk_start, chunk_end,
            )
            rows = _historical_with_retry(
                self._api, instrument_token, chunk_start, chunk_end, kite_iv, oi=oi,
            )
            for c in _to_candles(rows):
                seen[c.ts_epoch] = c
        return sorted(seen.values(), key=lambda c: c.ts_epoch)

    @staticmethod
    def _resolve_interval(interval: str) -> tuple[str, int]:
        if interval not in INTERVAL_MAP:
            msg = f"Unknown interval {interval!r}; expected one of {list(INTERVAL_MAP)}"
            raise ValueError(msg)
        return INTERVAL_MAP[interval]


# ----- Module helpers -----


def _as_datetime(d: datetime | date) -> datetime:
    """Coerce a `date` to a `datetime` at midnight; pass through datetimes."""
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day)
    msg = f"Cannot coerce {type(d).__name__} to datetime: {d!r}"
    raise TypeError(msg)


def _to_ist_naive(dt: datetime) -> datetime:
    """Convert a tz-aware datetime to IST and strip tzinfo. Naive input is
    assumed to already be IST and is returned unchanged."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(IST_TZ).replace(tzinfo=None)


def _historical_with_retry(
    api: KiteConnect,
    instrument_token: int,
    chunk_start: datetime,
    chunk_end: datetime,
    kite_iv: str,
    *,
    oi: bool = False,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> Any:
    """Call `historical_data` with retry on transient network errors.

    Retried: `requests.exceptions.Timeout`, `ConnectionError`, generic
    `RequestException` (covers SSL hiccups, DNS, etc.).
    NOT retried: `KiteException` (explicit server reject — wrong token,
    permission denied, malformed request — those won't get better with
    a retry and we want to surface them fast).

    Backoff is exponential: `base_delay * 2**attempt` (1s, 2s, 4s by default).
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return api.historical_data(
                instrument_token, chunk_start, chunk_end, kite_iv, oi=oi,
            )
        except KiteException as exc:
            log.warning(
                "Kite candles failed (no retry — server reject) "
                "for token=%d %s [%s, %s]: %s",
                instrument_token, kite_iv, chunk_start, chunk_end, exc,
            )
            raise
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                delay = base_delay * (2 ** attempt)
                log.warning(
                    "Kite candles transient error (%s) for token=%d %s "
                    "[%s, %s] — retry %d/%d in %.1fs",
                    type(exc).__name__, instrument_token, kite_iv,
                    chunk_start, chunk_end, attempt + 1, attempts, delay,
                )
                time.sleep(delay)
            else:
                log.warning(
                    "Kite candles exhausted retries for token=%d %s [%s, %s]",
                    instrument_token, kite_iv, chunk_start, chunk_end,
                )
    assert last_exc is not None  # exhausted all attempts → must have a last error
    raise last_exc


def _iter_time_chunks(
    start: datetime, end: datetime, *, max_days: int
) -> list[tuple[datetime, datetime]]:
    """Split `[start, end]` into chunks of at most `max_days` calendar days each."""
    if max_days <= 0:
        msg = f"max_days must be positive, got {max_days}"
        raise ValueError(msg)
    if start > end:
        return []
    chunks: list[tuple[datetime, datetime]] = []
    cur = start
    step = timedelta(days=max_days)
    while cur < end:
        nxt = min(cur + step, end)
        chunks.append((cur, nxt))
        cur = nxt
    if not chunks:
        chunks.append((start, end))
    return chunks


def _to_candles(rows: Any) -> list[Candle]:
    """Convert kiteconnect's `historical_data` rows (list[dict]) to Candles.

    Each row is `{date: tz-aware datetime, open, high, low, close, volume, oi?}`.
    We normalize `date` to UTC epoch seconds so downstream BQ writes never
    have to think about timezones again.

    **Naive-datetime contract (Major #8 fix):** the SDK *typically* returns
    tz-aware IST datetimes, but if a future SDK version (or a fixture / mock)
    hands us a naive value, we treat it as IST — Kite's native zone — not
    UTC. Defaulting to UTC would silently shift every bar by 5h30m. Tests
    (`test_kite_client.test_to_candles_handles_naive_datetime_as_*`) lock
    the convention.
    """
    out: list[Candle] = []
    if not rows:
        return out
    for r in rows:
        if not isinstance(r, dict) or "date" not in r:
            continue
        d = r["date"]
        if isinstance(d, datetime):
            ts_dt = d if d.tzinfo is not None else d.replace(tzinfo=IST_TZ)
            ts_epoch = int(ts_dt.timestamp())
        elif isinstance(d, date):
            # Kite-style daily-candle date: the bar represents 09:15-15:30
            # IST of that calendar day. Anchor at midnight IST to match
            # the SDK's behavior, then `kite_daily_ts_to_close` shifts to
            # the project's 15:30-IST close convention at write time.
            ts_epoch = int(datetime(d.year, d.month, d.day, tzinfo=IST_TZ).timestamp())
        else:
            log.warning("Skipping Kite candle with unparseable date: %r", d)
            continue
        try:
            out.append(
                Candle(
                    ts_epoch=ts_epoch,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=int(r["volume"]),
                    open_interest=int(r["oi"]) if r.get("oi") is not None else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping malformed Kite candle %r: %s", r, exc)
    return out
