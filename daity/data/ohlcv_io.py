"""Reusable building blocks for OHLCV ingest / refresh / pointwise-fix scripts.

The CLI scripts in `daity/scripts/` are intentionally thin — `<200 LOC` per
CLAUDE.md — and delegate to this module for the actual data-layer work:

  - `candles_to_frame(symbol, scale, candles)` — convert Kite candles into
    a Polars frame matching the `raw_ohlcv` schema, applying the project's
    candle-ts convention (daily ts shifted to 15:30 IST close).
  - `stage_to_bq(bq, frame, table_name)` — write a Polars frame to a
    BQ staging table with the canonical schema.
  - `drop_staging(bq, table_name)` — best-effort cleanup.
  - `read_max_ts_per_symbol_scale(bq, scales)` — index of where each
    `(symbol, scale)` partition currently ends in `raw_ohlcv`.
  - `fetch_one(kc, key, ...)` — pull candles for one (symbol, scale) with
    overlap lookback.

All BQ writes target `raw_ohlcv` (the physical table); `curated_ohlcv` is
a view over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl
from google.cloud import bigquery

from daity.data.bq import BQClient
from daity.data.candles import CLOSE_TIME_INTERVALS, kite_daily_ts_to_close
from daity.data.kite import Candle, KiteClient
from daity.utils.logging import get_logger

log = get_logger(__name__)

OHLCV_TABLE = "raw_ohlcv"

# Schema we write to staging tables — mirrors raw_ohlcv exactly.
STAGING_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ts", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("scale", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("open", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("high", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("low", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("close", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("volume", "INTEGER", mode="NULLABLE"),
]


@dataclass(frozen=True, slots=True)
class GapKey:
    """One (symbol, scale) pair to refresh."""

    symbol: str
    scale: str
    instrument_token: int
    last_ts_utc: datetime | None  # max ts in prod for this (sym, scale), None if absent


def candles_to_frame(symbol: str, scale: str, candles: list[Candle]) -> pl.DataFrame:
    """Convert Kite candles to a Polars frame in `raw_ohlcv` schema.

    Applies the candle-ts convention: intraday `ts` stays at bar open,
    daily `ts` is shifted from Kite's 00:00 IST to 15:30 IST = 10:00 UTC.
    """
    if not candles:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
            }
        )
    ts_utc = [datetime.fromtimestamp(c.ts_epoch, tz=UTC) for c in candles]
    if scale in CLOSE_TIME_INTERVALS:
        ts_utc = [kite_daily_ts_to_close(t) for t in ts_utc]
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(candles),
            "ts": ts_utc,
            "scale": [scale] * len(candles),
            "open":   [c.open  for c in candles],
            "high":   [c.high  for c in candles],
            "low":    [c.low   for c in candles],
            "close":  [c.close for c in candles],
            "volume": [c.volume for c in candles],
        },
        schema={
            "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
            "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
        },
    )


def stage_to_bq(bq: BQClient, frame: pl.DataFrame, table_name: str) -> str:
    """Write `frame` to `<dataset>.<table_name>` as a BQ staging table.

    Uses WRITE_TRUNCATE so re-running with the same table name overwrites.
    Returns the unqualified table name for caller convenience.
    """
    fq = bq.cfg.fq_table(table_name)
    log.info("Staging %d rows to %s", frame.height, fq)
    job = bq._client.load_table_from_dataframe(
        frame.to_pandas(), fq,
        job_config=bigquery.LoadJobConfig(
            schema=STAGING_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    return table_name


def drop_staging(bq: BQClient, table_name: str) -> None:
    """Best-effort drop of a staging table; silent if it doesn't exist."""
    fq = bq.cfg.fq_table(table_name)
    bq._client.delete_table(fq, not_found_ok=True)
    log.info("Dropped %s", fq)


def read_max_ts_per_symbol_scale(
    bq: BQClient, scales: tuple[str, ...],
) -> dict[tuple[str, str], datetime]:
    """For each `(symbol, scale)`, return the most recent ts in `raw_ohlcv`.

    Skips combinations that don't exist (no key in the returned dict).
    """
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    quoted = ", ".join(f"'{iv}'" for iv in scales)
    sql = (
        f"SELECT symbol, scale, MAX(ts) AS last_ts "
        f"FROM `{fq}` WHERE scale IN ({quoted}) "
        f"GROUP BY symbol, scale"
    )
    rows = bq.query_rows(sql)
    out: dict[tuple[str, str], datetime] = {}
    for r in rows:
        ts = r["last_ts"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            out[(r["symbol"], r["scale"])] = ts
    return out


def fetch_one_window(
    kc: KiteClient, key: GapKey,
    *, floor_start: datetime, end: datetime,
    overlap_lookback: timedelta = timedelta(days=5),
) -> list[Candle]:
    """Pull Kite candles for `key` over `[start, end]`.

    `start` = `key.last_ts_utc - overlap_lookback` if known (overlap powers
    the spot-check); otherwise `floor_start`. Returns [] if `start >= end`.
    """
    if key.last_ts_utc is not None:
        start = key.last_ts_utc - overlap_lookback
    else:
        start = floor_start
    if start >= end:
        return []
    return kc.candles(key.instrument_token, interval=key.scale,
                      start=start, end=end)


def merge_into_raw_ohlcv(bq: BQClient, staging: str) -> int:
    """MERGE staging rows into `raw_ohlcv` on `(symbol, ts, scale)`,
    `WHEN NOT MATCHED THEN INSERT`. Returns the number of rows inserted."""
    prod_fq = bq.cfg.fq_table(OHLCV_TABLE)
    stg_fq = bq.cfg.fq_table(staging)
    sql = f"""
        MERGE `{prod_fq}` T
        USING `{stg_fq}` S
          ON T.symbol = S.symbol AND T.ts = S.ts AND T.scale = S.scale
        WHEN NOT MATCHED THEN INSERT (symbol, ts, scale, open, high, low, close, volume)
          VALUES (S.symbol, S.ts, S.scale, S.open, S.high, S.low, S.close, S.volume)
    """
    job = bq._client.query(sql, job_config=bq._job_config())
    job.result()
    return int(job.num_dml_affected_rows or 0)


def merge_with_update(bq: BQClient, staging: str, *, threshold: float) -> int:
    """MERGE: UPDATE diverging rows (close differs > threshold), INSERT missing.

    Used by pointwise-fix flows. Matching rows where the diff is below the
    threshold are left untouched.
    """
    prod_fq = bq.cfg.fq_table(OHLCV_TABLE)
    stg_fq = bq.cfg.fq_table(staging)
    sql = f"""
        MERGE `{prod_fq}` T
        USING `{stg_fq}` S
          ON T.symbol = S.symbol AND T.ts = S.ts AND T.scale = S.scale
        WHEN MATCHED AND
             SAFE_DIVIDE(ABS(T.close - S.close), NULLIF(T.close, 0)) > {threshold}
          THEN UPDATE SET
            open = S.open, high = S.high, low = S.low,
            close = S.close, volume = S.volume
        WHEN NOT MATCHED THEN
          INSERT (symbol, ts, scale, open, high, low, close, volume)
          VALUES (S.symbol, S.ts, S.scale, S.open, S.high, S.low, S.close, S.volume)
    """
    job = bq._client.query(sql, job_config=bq._job_config())
    job.result()
    return int(job.num_dml_affected_rows or 0)


def spot_check_overlap(
    bq: BQClient, staging: str, *, tolerance: float,
) -> dict[str, int | float | None]:
    """Compare close prices on overlapping (symbol, ts, scale) rows."""
    prod_fq = bq.cfg.fq_table(OHLCV_TABLE)
    stg_fq = bq.cfg.fq_table(staging)
    sql = f"""
        WITH overlap_rows AS (
          SELECT
            S.symbol AS symbol, S.scale AS iv, S.ts AS ts,
            S.close AS staged_close, P.close AS prod_close,
            SAFE_DIVIDE(ABS(S.close - P.close), NULLIF(P.close, 0)) AS rel_diff
          FROM `{stg_fq}` S
          INNER JOIN `{prod_fq}` P
            ON P.symbol = S.symbol AND P.ts = S.ts AND P.scale = S.scale
        )
        SELECT
          COUNT(*) AS n_overlap,
          COUNTIF(rel_diff > {tolerance}) AS n_diff,
          MAX(rel_diff) AS max_rel_diff,
          APPROX_QUANTILES(rel_diff, 100)[OFFSET(95)] AS p95_rel_diff
        FROM overlap_rows
    """
    rows = bq.query_rows(sql)
    return rows[0] if rows else {}
