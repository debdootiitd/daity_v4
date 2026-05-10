"""Helpers for the BQ-EXPORT → GCS → local-Parquet ingest path.

The CLI (`daity-ingest-parquet`) orchestrates these in a `for scale in scales`
loop. See the CLI module's docstring for the why; this module is the how.

The ingest:
  1. `export_scale_to_gcs(bq, scale, symbols, gcs_uri)` — server-side EXPORT
     DATA to a GCS staging prefix.
  2. `download_shards_as_frame(gcs_client, bucket, blob_prefix)` — pull every
     shard back, concat into one Polars frame.
  3. `write_per_symbol(store, scale, frame)` — group by symbol, write each
     slice to the per-(scale, symbol) Parquet partition.
  4. `cleanup_gcs_shards(gcs_client, bucket, blob_prefix)` — delete shards.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import polars as pl
from google.cloud import storage

from daity.data.bq import BQClient
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)

OHLCV_TABLE = "raw_ohlcv"


@dataclass(slots=True)
class ScaleResult:
    """Per-scale ingest outcome reported back to the CLI."""

    scale: str
    n_symbols_requested: int
    n_symbols_skipped: int = 0
    n_symbols_written: int = 0
    n_rows_written: int = 0
    n_shards: int = 0
    elapsed_sec: float = 0.0
    error: str = ""


def shards_uri(bucket: str, prefix: str, scale: str, run_id: str) -> str:
    """EXPORT DATA URI pattern. The trailing `*` is BQ's shard placeholder."""
    return f"gs://{bucket}/{prefix}/run-{run_id}/scale-{scale}/shard-*.parquet"


def shards_blob_prefix(prefix: str, scale: str, run_id: str) -> str:
    """GCS prefix matching the shards from `shards_uri` — for list/cleanup."""
    return f"{prefix}/run-{run_id}/scale-{scale}/"


def resolve_symbols_to_ingest(
    store: ParquetStore, scale: str, all_symbols: list[str], *, force: bool,
) -> list[str]:
    """Pick symbols whose `(scale, symbol)` partition is missing locally."""
    if force:
        return list(all_symbols)
    return [
        s for s in all_symbols
        if not store.partition_path(scale, s).exists()
    ]


def export_scale_to_gcs(
    bq: BQClient, *, scale: str, symbols: list[str], uri: str,
) -> None:
    """Server-side EXPORT DATA for one scale × symbol-list to a GCS URI."""
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    esc = lambda s: s.replace("'", "''")
    in_list = ", ".join(f"'{esc(s)}'" for s in symbols)
    sql = f"""
    EXPORT DATA OPTIONS(
      uri='{uri}',
      format='PARQUET',
      overwrite=true,
      compression='ZSTD'
    ) AS
    SELECT symbol, ts, scale, open, high, low, close, volume
    FROM `{fq}`
    WHERE scale = '{scale}' AND symbol IN ({in_list})
    ORDER BY symbol, ts
    """
    log.info("EXPORT DATA: scale=%s symbols=%d uri=%s", scale, len(symbols), uri)
    bq._client.query(sql, job_config=bq._job_config()).result()


def download_shards_as_frame(
    storage_client: storage.Client, bucket_name: str, blob_prefix: str,
) -> tuple[pl.DataFrame, int]:
    """Download every shard under `blob_prefix` into one Polars frame."""
    blobs = list(storage_client.list_blobs(bucket_name, prefix=blob_prefix))
    if not blobs:
        return pl.DataFrame(), 0
    frames: list[pl.DataFrame] = []
    for b in blobs:
        if not b.name.endswith(".parquet"):
            continue
        log.debug("downloading gs://%s/%s (%s bytes)", bucket_name, b.name, b.size)
        frames.append(pl.read_parquet(io.BytesIO(b.download_as_bytes())))
    if not frames:
        return pl.DataFrame(), len(blobs)
    return pl.concat(frames, how="vertical"), len(blobs)


def cleanup_gcs_shards(
    storage_client: storage.Client, bucket_name: str, blob_prefix: str,
) -> int:
    """Delete every blob under `blob_prefix`; returns count deleted."""
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=blob_prefix))
    for b in blobs:
        bucket.blob(b.name).delete()
    return len(blobs)


def write_per_symbol(
    store: ParquetStore, scale: str, frame: pl.DataFrame,
) -> tuple[int, int]:
    """Group `frame` by symbol and write each slice as a partition.

    Returns `(n_partitions_written, total_rows_written)`.
    """
    if frame.height == 0:
        return 0, 0
    n_parts = n_rows = 0
    for sym in frame["symbol"].unique().sort().to_list():
        sym_frame = frame.filter(pl.col("symbol") == sym)
        if sym_frame.height == 0:
            continue
        store.write_partition(scale, sym, sym_frame)
        n_parts += 1
        n_rows += sym_frame.height
    return n_parts, n_rows
