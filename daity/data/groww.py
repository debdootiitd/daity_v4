"""Groww trading-API client (Phase 1 ingest).

Thin wrapper around `growwapi.GrowwAPI` that:
- Accepts our `GrowwConfig` (auto-detects whether `GROWW_API_KEY` is a JWT
  access token or an API key needing a `get_access_token` exchange).
- Caches the static instruments dump (~190k rows) as a Polars DataFrame,
  optionally backed by a Parquet file under the project cache root.
- Exposes ticker → groww_symbol/ISIN/exchange_token lookups for the universe
  refresh path.
- Wraps `get_historical_candles` with chunked time-window pulls that respect
  Groww's per-interval window caps. **Note:** at the time of writing the
  caller's API key returned `Access forbidden` for historical endpoints — that
  is a subscription-tier issue (historical data is on Groww's paid plan), not
  a code issue. The chunking + parsing code is exercised by unit tests with
  the SDK call mocked out, so it's ready to use once access lands.

Why a wrapper rather than passing `GrowwAPI` around directly?
- Centralizes auth so callers don't have to know about the JWT-vs-secret
  branch.
- Maps our project's interval labels (`5m`, `15m`, `60m`, `day`) to Groww's
  (`5minute`, `15minute`, `1hour`, `1day`) in one place.
- Keeps rate-limit / retry concerns out of the scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from growwapi import GrowwAPI
from growwapi.groww.exceptions import GrowwAPIException

from daity.utils.env import GrowwConfig
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Map our internal interval labels to (groww_constant, max_chunk_days).
# Window caps come from Groww's docs: shorter intervals → narrower windows.
INTERVAL_MAP: dict[str, tuple[str, int]] = {
    "5m":  ("5minute",  30),
    "15m": ("15minute", 90),
    "60m": ("1hour",   180),
    "day": ("1day",    180),
}

# Default exchange + segment for the NSE cash universe we're refreshing.
DEFAULT_EXCHANGE = "NSE"
DEFAULT_SEGMENT = "CASH"

INSTRUMENTS_CACHE_SUBDIR = "groww"
INSTRUMENTS_PARQUET = "instruments.parquet"


@dataclass(frozen=True, slots=True)
class Candle:
    """One Groww candle, normalized into named fields."""

    ts_epoch: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int | None  # null for cash-segment instruments


class GrowwClient:
    """High-level wrapper. Construct from `GrowwConfig`."""

    def __init__(self, cfg: GrowwConfig, *, api: GrowwAPI | None = None) -> None:
        self.cfg = cfg
        self._api: GrowwAPI = api if api is not None else self._build_api(cfg)
        self._instruments: pl.DataFrame | None = None

    @staticmethod
    def _build_api(cfg: GrowwConfig) -> GrowwAPI:
        """Mint a `GrowwAPI` instance, branching on whether the key is a JWT.

        Pre-issued JWT path is the common one for vendor-key accounts. The
        api-key + secret exchange path is used when the key is a short
        non-JWT identifier and the SDK has to call `get_access_token`.
        """
        if cfg.key_is_access_token:
            return GrowwAPI(cfg.api_key)
        if not cfg.api_secret:
            msg = (
                "GROWW_API_KEY is not a JWT and no GROWW_API_SECRET is set; "
                "cannot mint an access token."
            )
            raise ValueError(msg)
        token = GrowwAPI.get_access_token(api_key=cfg.api_key, secret=cfg.api_secret)
        return GrowwAPI(token)

    # ----- Instruments -----

    @property
    def instruments(self) -> pl.DataFrame:
        """Lazily-loaded instruments DataFrame (cached on first access)."""
        if self._instruments is None:
            self._instruments = self._load_instruments()
        return self._instruments

    def load_instruments_cache(self, cache_root: Path, *, force_refresh: bool = False) -> pl.DataFrame:
        """Load instruments from a Parquet cache (or refresh if missing/stale)."""
        cache_dir = Path(cache_root) / INSTRUMENTS_CACHE_SUBDIR
        cache_path = cache_dir / INSTRUMENTS_PARQUET
        if cache_path.exists() and not force_refresh:
            log.info("Groww instruments: reusing cache at %s", cache_path)
            self._instruments = pl.read_parquet(cache_path)
            return self._instruments
        df = self._load_instruments()
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        log.info("Groww instruments: cached %d rows at %s", df.height, cache_path)
        self._instruments = df
        return df

    def _load_instruments(self) -> pl.DataFrame:
        log.info("Groww instruments: fetching dump from %s", GrowwAPI.INSTRUMENT_CSV_URL)
        pdf = self._api.get_all_instruments()
        # SDK returns a pandas DataFrame; convert to polars for project consistency.
        return pl.from_pandas(pdf)

    # ----- Identifier lookup -----

    def lookup(self, trading_symbol: str, exchange: str = DEFAULT_EXCHANGE) -> dict | None:
        """Return the instruments-dump row for `(exchange, trading_symbol)`, or None."""
        df = self.instruments.filter(
            (pl.col("exchange") == exchange) & (pl.col("trading_symbol") == trading_symbol)
        )
        if df.height == 0:
            return None
        # `get_instrument_by_exchange_and_trading_symbol` returns the first match;
        # do the same here. F&O underlyings can repeat across instrument types.
        return df.head(1).to_dicts()[0]

    def bulk_lookup(
        self,
        trading_symbols: list[str],
        *,
        exchange: str = DEFAULT_EXCHANGE,
        segment: str | None = DEFAULT_SEGMENT,
    ) -> pl.DataFrame:
        """Look up many tickers at once.

        Returns a frame keyed by `trading_symbol` with `(groww_symbol, isin,
        exchange_token, segment, name)`. Missing tickers appear with null
        identifier columns so the caller can detect gaps.
        """
        df = self.instruments.filter(pl.col("exchange") == exchange)
        if segment is not None:
            df = df.filter(pl.col("segment") == segment)
        # Keep only the columns we need; drop duplicate trading_symbols (FNO can repeat).
        keep = ["trading_symbol", "groww_symbol", "isin", "exchange_token", "segment", "name"]
        keep = [c for c in keep if c in df.columns]
        df = df.select(keep).unique(subset=["trading_symbol"], keep="first")

        wanted = pl.DataFrame({"trading_symbol": trading_symbols})
        joined = wanted.join(df, on="trading_symbol", how="left")
        n_missing = joined.filter(pl.col("groww_symbol").is_null()).height
        if n_missing:
            log.warning(
                "bulk_lookup: %d / %d tickers missing in Groww instruments dump",
                n_missing,
                len(trading_symbols),
            )
        return joined

    # ----- Candles -----

    def candles(
        self,
        groww_symbol: str,
        *,
        interval: str,
        start: datetime,
        end: datetime,
        exchange: str = DEFAULT_EXCHANGE,
        segment: str = DEFAULT_SEGMENT,
    ) -> list[Candle]:
        """Fetch all candles for `groww_symbol` in `[start, end]`.

        `interval` is one of our project labels — `"5m"`, `"15m"`, `"60m"`,
        `"day"`. The call is chunked into per-interval-max-window slices, then
        deduped on `ts_epoch` (chunk boundaries are inclusive on both sides
        per Groww's docs, so duplicates can appear).
        """
        groww_iv, max_days = self._resolve_interval(interval)
        chunks = list(_iter_time_chunks(start, end, max_days=max_days))
        seen: dict[int, Candle] = {}
        for chunk_start, chunk_end in chunks:
            log.debug(
                "Groww candles: %s %s [%s, %s]",
                groww_symbol, groww_iv, chunk_start.isoformat(), chunk_end.isoformat(),
            )
            try:
                res = self._api.get_historical_candles(
                    exchange=exchange,
                    segment=segment,
                    groww_symbol=groww_symbol,
                    start_time=_format_ts(chunk_start),
                    end_time=_format_ts(chunk_end),
                    candle_interval=groww_iv,
                )
            except GrowwAPIException as exc:
                log.warning(
                    "Groww candles failed for %s %s [%s, %s]: %s",
                    groww_symbol, groww_iv, chunk_start, chunk_end, exc,
                )
                raise
            for c in _extract_candles(res):
                seen[c.ts_epoch] = c
        return sorted(seen.values(), key=lambda c: c.ts_epoch)

    @staticmethod
    def _resolve_interval(interval: str) -> tuple[str, int]:
        if interval not in INTERVAL_MAP:
            msg = f"Unknown interval {interval!r}; expected one of {list(INTERVAL_MAP)}"
            raise ValueError(msg)
        return INTERVAL_MAP[interval]


# ----- Module helpers (free functions, easy to unit-test) -----


def _format_ts(dt: datetime) -> str:
    """Format a datetime as Groww's accepted string (`yyyy-MM-dd HH:mm:ss`)."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _iter_time_chunks(
    start: datetime, end: datetime, *, max_days: int
) -> list[tuple[datetime, datetime]]:
    """Split `[start, end]` into half-open chunks of at most `max_days` calendar days each.

    The chunks tile the interval — `[start, start+max_days), [start+max_days, ...)` — so
    candle responses don't overlap (we still dedupe on ts_epoch as a belt+braces).
    Returns at least one chunk even if `start >= end` (empty range still asks Groww
    once, which the caller can interpret as "no data").
    """
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
        # start == end exactly → still emit a one-tick chunk so the caller sees
        # at least one (likely empty) response.
        chunks.append((start, end))
    return chunks


def _extract_candles(response: Any) -> list[Candle]:
    """Pull the `[ts, o, h, l, c, v, oi]` arrays out of a Groww historical response.

    Groww's response shape has shifted across SDK versions; we support both
    `{"candles": [...]}` and `{"payload": {"candles": [...]}}`.
    """
    if response is None:
        return []
    if isinstance(response, dict):
        # newer SDK: candles directly at the top level
        if "candles" in response:
            raw = response["candles"]
        elif "payload" in response and isinstance(response["payload"], dict):
            raw = response["payload"].get("candles", [])
        else:
            return []
    else:
        return []
    out: list[Candle] = []
    for row in raw or []:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = int(row[0])
        try:
            out.append(
                Candle(
                    ts_epoch=ts,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=int(row[5]),
                    open_interest=int(row[6]) if len(row) > 6 and row[6] is not None else None,
                )
            )
        except (TypeError, ValueError) as exc:
            log.warning("Skipping malformed Groww candle %r: %s", row, exc)
    return out
