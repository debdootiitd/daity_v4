"""BQ → GCS export → local Parquet ingest for `raw_ohlcv` (Phase 1.5).

Per scale:
  1. Run a single `EXPORT DATA` → BigQuery writes Parquet shards to GCS
     (no client-side memory pressure, no readsessions IAM grant required).
  2. Download all shards, concat into one Polars frame.
  3. Group by symbol, write each per-symbol slice to the Parquet store.
  4. Cleanup the GCS staging shards.

Idempotent: partitions already on disk are skipped, so the EXPORT WHERE
clause filters them out before they leave BQ. `--force` rewrites all.

See `daity.data.parquet_ingest` for the building blocks.

Run:
    daity-ingest-parquet                              # all scales, all symbols
    daity-ingest-parquet --scale day                  # one scale only
    daity-ingest-parquet --symbols RELIANCE,TCS       # one or two symbols
    daity-ingest-parquet --force                      # rewrite all
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import click
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.data.parquet_ingest import (
    ScaleResult,
    cleanup_gcs_shards,
    download_shards_as_frame,
    export_scale_to_gcs,
    resolve_symbols_to_ingest,
    shards_blob_prefix,
    shards_uri,
    write_per_symbol,
)
from daity.data.parquet_store import VALID_SCALES, ParquetStore
from daity.data.symbols import SymbolMaster
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

DEFAULT_CACHE_ROOT = Path("./data/parquet")
SYMBOL_CACHE_ROOT = Path("./data/cache")
DEFAULT_GCS_BUCKET = "daity-raw-data-476315"
DEFAULT_GCS_PREFIX = "parquet-ingest-staging"


def _ingest_one_scale(
    bq: BQClient, storage_client: storage.Client, store: ParquetStore,
    *, scale: str, symbols: list[str], force: bool,
    bucket: str, prefix: str, run_id: str, keep_gcs: bool,
) -> ScaleResult:
    """End-to-end ingest of one scale × the provided symbol list."""
    res = ScaleResult(scale=scale, n_symbols_requested=len(symbols))
    todo = resolve_symbols_to_ingest(store, scale, symbols, force=force)
    res.n_symbols_skipped = len(symbols) - len(todo)
    if not todo:
        return res

    t0 = time.time()
    uri = shards_uri(bucket, prefix, scale, run_id)
    blob_prefix = shards_blob_prefix(prefix, scale, run_id)
    try:
        export_scale_to_gcs(bq, scale=scale, symbols=todo, uri=uri)
    except gcp_exceptions.GoogleAPIError as exc:
        res.error = f"EXPORT failed: {type(exc).__name__}: {exc}"
        return res

    try:
        frame, n_shards = download_shards_as_frame(storage_client, bucket, blob_prefix)
        res.n_shards = n_shards
    except gcp_exceptions.GoogleAPIError as exc:
        res.error = f"download failed: {type(exc).__name__}: {exc}"
        return res

    if frame.height > 0:
        n_parts, n_rows = write_per_symbol(store, scale, frame)
        res.n_symbols_written = n_parts
        res.n_rows_written = n_rows

    if not keep_gcs:
        deleted = cleanup_gcs_shards(storage_client, bucket, blob_prefix)
        log.info("Cleaned up %d GCS shard(s) under gs://%s/%s",
                 deleted, bucket, blob_prefix)

    res.elapsed_sec = time.time() - t0
    return res


@click.command()
@click.option("--scale", default=None,
              help=f"Restrict to one scale (default: all of {list(VALID_SCALES)}).")
@click.option("--symbols", default=None,
              help="Comma-separated subset (default: all from symbol_list).")
@click.option("--force", is_flag=True, help="Rewrite even if partition exists.")
@click.option("--cache-root", type=click.Path(path_type=Path),
              default=DEFAULT_CACHE_ROOT, show_default=True,
              help="Local Parquet cache root.")
@click.option("--gcs-bucket", default=DEFAULT_GCS_BUCKET, show_default=True,
              help="GCS bucket for EXPORT DATA staging.")
@click.option("--gcs-prefix", default=DEFAULT_GCS_PREFIX, show_default=True,
              help="GCS prefix; this run gets a unique sub-prefix.")
@click.option("--keep-gcs", is_flag=True, help="Don't delete GCS staging shards.")
def main(scale: str | None, symbols: str | None, force: bool,
         cache_root: Path, gcs_bucket: str, gcs_prefix: str, keep_gcs: bool) -> None:
    """Ingest `raw_ohlcv` into per-(scale, symbol) Parquet partitions via GCS."""
    try:
        cfg = load_env()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)
    storage_client = storage.Client(project=cfg.project)

    scales = (scale,) if scale else VALID_SCALES
    bad = [s for s in scales if s not in VALID_SCALES]
    if bad:
        console.print(f"[red]Unknown scales: {bad}; expected subset of {list(VALID_SCALES)}[/red]")
        sys.exit(2)

    if symbols:
        wanted = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        wanted = SymbolMaster.from_bq(bq, cache_root=SYMBOL_CACHE_ROOT).symbols

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    store = ParquetStore(cache_root)
    console.print(
        f"[bold]Plan:[/bold] {len(wanted)} symbols × {len(scales)} scales = "
        f"{len(wanted) * len(scales)} target partitions  "
        f"(GCS staging: gs://{gcs_bucket}/{gcs_prefix}/run-{run_id}/)"
    )

    results: list[ScaleResult] = []
    for sc in scales:
        console.print(f"[bold]→ ingesting scale={sc}...[/bold]")
        res = _ingest_one_scale(
            bq, storage_client, store,
            scale=sc, symbols=wanted, force=force,
            bucket=gcs_bucket, prefix=gcs_prefix, run_id=run_id,
            keep_gcs=keep_gcs,
        )
        results.append(res)
        if res.error:
            console.print(f"  [red]{res.error}[/red]")
        else:
            console.print(
                f"  shards={res.n_shards}  written={res.n_symbols_written}  "
                f"skipped={res.n_symbols_skipped}  rows={res.n_rows_written:,}  "
                f"elapsed={res.elapsed_sec:.1f}s"
            )

    summary = Table(title="Ingest summary")
    summary.add_column("scale"); summary.add_column("written"); summary.add_column("skipped")
    summary.add_column("rows"); summary.add_column("elapsed"); summary.add_column("status")
    total_w = total_s = total_r = 0
    total_t = 0.0
    for r in results:
        status = r.error or "OK"
        summary.add_row(r.scale, f"{r.n_symbols_written:,}", f"{r.n_symbols_skipped:,}",
                        f"{r.n_rows_written:,}", f"{r.elapsed_sec:.1f}s", status[:40])
        total_w += r.n_symbols_written; total_s += r.n_symbols_skipped
        total_r += r.n_rows_written; total_t += r.elapsed_sec
    summary.add_row("[bold]total[/bold]", f"{total_w:,}", f"{total_s:,}",
                    f"{total_r:,}", f"{total_t:.1f}s", "")
    console.print(summary)

    if any(r.error for r in results):
        sys.exit(3)


if __name__ == "__main__":
    main()
