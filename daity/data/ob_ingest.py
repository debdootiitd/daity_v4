"""Phase 4 step 0 — ingest BQ `order_book_depth` → per-symbol Parquet.

Mirrors `parquet_ingest.py`'s BQ-EXPORT → GCS shards → local-Parquet pattern
but reads from the wide-format L2 snapshot table audited in Phase 0:

    schema: (symbol, date, ts, ltp, volume, total_buy_qty, total_sell_qty,
             bid1..bid5_price/qty, ask1..ask5_price/qty, spread_bps, fetched_at)
    cadence: 60s snapshots
    coverage: 2026-03-08 → 2026-05-04, ~6,510 symbol-days (98.9% fill)

Output layout: `data/ob_parquet/{symbol}.parquet` (one parquet per symbol,
all snapshots concatenated and sorted by `ts`). The 2-month window per
symbol is small enough that per-symbol partitioning works without the
year/month subtree we use for OHLCV.

Per DESIGN §3.4 / amendment 19 (pending), Phase 4 only needs these top-5
columns per side. We keep `fetched_at` only for leakage assertions; the
event time is `ts`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import polars as pl
from google.cloud import storage

from daity.data.bq import BQClient
from daity.utils.logging import get_logger

log = get_logger(__name__)

OB_TABLE = "order_book_depth"

# Columns we materialize from the BQ wide-format L2 (5 levels per side).
OB_COLUMNS: tuple[str, ...] = (
    "symbol", "ts", "ltp", "volume",
    "total_buy_qty", "total_sell_qty",
    "bid1_price", "bid1_qty", "bid2_price", "bid2_qty",
    "bid3_price", "bid3_qty", "bid4_price", "bid4_qty",
    "bid5_price", "bid5_qty",
    "ask1_price", "ask1_qty", "ask2_price", "ask2_qty",
    "ask3_price", "ask3_qty", "ask4_price", "ask4_qty",
    "ask5_price", "ask5_qty",
    "spread_bps",
)


@dataclass(slots=True)
class OBIngestResult:
    """Per-symbol OB ingest outcome."""
    symbol: str
    n_rows_written: int = 0
    n_partitions: int = 0
    elapsed_sec: float = 0.0
    error: str = ""


def ob_shards_uri(bucket: str, prefix: str, run_id: str) -> str:
    return f"gs://{bucket}/{prefix}/ob-{run_id}/shard-*.parquet"


def ob_shards_blob_prefix(prefix: str, run_id: str) -> str:
    return f"{prefix}/ob-{run_id}/"


def export_ob_to_gcs(
    bq: BQClient, *, symbols: list[str], uri: str,
    date_start: str | None = None, date_end: str | None = None,
) -> None:
    """Server-side EXPORT DATA for the OB table → GCS Parquet shards.

    `date_start` / `date_end` (ISO yyyy-mm-dd) optionally cap to a sub-
    window. Default: full table.
    """
    fq = bq.cfg.fq_table(OB_TABLE)
    esc = lambda s: s.replace("'", "''")
    in_list = ", ".join(f"'{esc(s)}'" for s in symbols)
    cols = ", ".join(OB_COLUMNS)

    where = [f"symbol IN ({in_list})"]
    if date_start is not None:
        where.append(f"date >= DATE '{date_start}'")
    if date_end is not None:
        where.append(f"date <= DATE '{date_end}'")
    where_sql = " AND ".join(where)

    sql = f"""
    EXPORT DATA OPTIONS(
      uri='{uri}',
      format='PARQUET',
      overwrite=true,
      compression='ZSTD'
    ) AS
    SELECT {cols}
    FROM `{fq}`
    WHERE {where_sql}
    ORDER BY symbol, ts
    """
    log.info(
        "EXPORT DATA OB: symbols=%d window=[%s, %s] uri=%s",
        len(symbols), date_start or "min", date_end or "max", uri,
    )
    bq._client.query(sql, job_config=bq._job_config()).result()


def download_ob_shards(
    storage_client: storage.Client, bucket_name: str, blob_prefix: str,
) -> tuple[pl.DataFrame, int]:
    """Pull every OB shard under `blob_prefix` into one frame."""
    blobs = list(storage_client.list_blobs(bucket_name, prefix=blob_prefix))
    if not blobs:
        return pl.DataFrame(), 0
    frames: list[pl.DataFrame] = []
    for b in blobs:
        if not b.name.endswith(".parquet"):
            continue
        frames.append(pl.read_parquet(io.BytesIO(b.download_as_bytes())))
    if not frames:
        return pl.DataFrame(), len(blobs)
    return pl.concat(frames, how="vertical"), len(blobs)


def cleanup_ob_shards(
    storage_client: storage.Client, bucket_name: str, blob_prefix: str,
) -> int:
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=blob_prefix))
    for b in blobs:
        bucket.blob(b.name).delete()
    return len(blobs)


def write_ob_per_symbol(
    out_root, frame: pl.DataFrame,
) -> list[OBIngestResult]:
    """Group `frame` by symbol, write each slice as
    `{out_root}/{symbol}.parquet` sorted by `ts`.

    Returns one `OBIngestResult` per symbol seen.
    """
    if frame.height == 0:
        return []
    from pathlib import Path
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out: list[OBIngestResult] = []
    for sym in frame["symbol"].unique().sort().to_list():
        sym_frame = frame.filter(pl.col("symbol") == sym).sort("ts")
        if sym_frame.height == 0:
            continue
        path = out_root / f"{sym}.parquet"
        sym_frame.write_parquet(path)
        out.append(OBIngestResult(
            symbol=sym, n_rows_written=sym_frame.height, n_partitions=1,
        ))
    return out


def list_ob_symbols(out_root) -> list[str]:
    """List symbols already ingested locally."""
    from pathlib import Path
    out_root = Path(out_root)
    if not out_root.exists():
        return []
    return sorted(p.stem for p in out_root.glob("*.parquet"))
