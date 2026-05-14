"""`daity-ingest-ob` — Phase 4 step 0 OB ingest CLI.

Streams the BQ `order_book_depth` table through GCS Parquet shards into
`data/ob_parquet/{symbol}.parquet`. Idempotent: re-running with the
same `--symbols` list skips symbols already present (unless `--force`).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import click
from google.cloud import storage
from rich.console import Console

from daity.data.bq import BQClient
from daity.data.ob_ingest import (
    cleanup_ob_shards,
    download_ob_shards,
    export_ob_to_gcs,
    list_ob_symbols,
    ob_shards_blob_prefix,
    ob_shards_uri,
    write_ob_per_symbol,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


@click.command()
@click.option("--out-root", type=click.Path(path_type=Path),
              default=Path("data/ob_parquet"), show_default=True)
@click.option("--symbols", default=None,
              help="Comma-separated symbols. Default: read from "
                   "`data/cache/symbols/universe.txt` if present.")
@click.option("--max-symbols", type=int, default=None)
@click.option("--date-start", default="2026-03-08", show_default=True,
              help="OB-window start (inclusive).")
@click.option("--date-end", default="2026-05-04", show_default=True,
              help="OB-window end (inclusive).")
@click.option("--gcs-bucket", default="daity-raw-data-476315", show_default=True,
              help="GCS bucket for EXPORT DATA shards. Default matches OHLCV ingest.")
@click.option("--gcs-prefix", default="ob_staging", show_default=True,
              help="GCS object prefix under bucket for shards.")
@click.option("--force", is_flag=True, help="Re-ingest symbols already present.")
def main(
    out_root: Path, symbols: str | None, max_symbols: int | None,
    date_start: str, date_end: str,
    gcs_bucket: str, gcs_prefix: str, force: bool,
) -> None:
    """Ingest BQ order_book_depth → per-symbol Parquet under out_root."""
    from daity.utils.env import load_env
    out_root.mkdir(parents=True, exist_ok=True)
    bq = BQClient(load_env())
    bucket = gcs_bucket

    # Resolve symbol list.
    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        universe_path = Path("data/cache/symbols/universe.txt")
        if universe_path.exists():
            sym_list = [s.strip() for s in universe_path.read_text().splitlines() if s.strip()]
        else:
            raise click.UsageError(
                "no --symbols and data/cache/symbols/universe.txt missing. "
                "Pass --symbols explicitly or write the universe file."
            )
    if max_symbols is not None:
        sym_list = sym_list[:max_symbols]

    if not force:
        already = set(list_ob_symbols(out_root))
        new = [s for s in sym_list if s not in already]
        skipped = len(sym_list) - len(new)
        sym_list = new
        if skipped:
            console.print(f"[dim]skipping {skipped} symbols already on disk[/dim]")
    if not sym_list:
        console.print("[bold]nothing to ingest.[/bold]")
        return

    run_id = uuid.uuid4().hex[:8]
    uri = ob_shards_uri(bucket, gcs_prefix, run_id)
    blob_prefix = ob_shards_blob_prefix(gcs_prefix, run_id)

    console.print(
        f"[bold]OB ingest:[/bold] {len(sym_list)} symbols, "
        f"window=[{date_start}, {date_end}], run_id={run_id}",
    )

    storage_client = storage.Client()
    t0 = time.time()

    # Step 1: BQ → GCS export.
    export_ob_to_gcs(
        bq, symbols=sym_list, uri=uri,
        date_start=date_start, date_end=date_end,
    )
    log.info("export done in %.1fs", time.time() - t0)

    # Step 2: GCS → local frame.
    t1 = time.time()
    frame, n_shards = download_ob_shards(storage_client, bucket, blob_prefix)
    log.info("downloaded %d shards (%d rows) in %.1fs",
             n_shards, frame.height, time.time() - t1)

    # Step 3: per-symbol Parquet write.
    t2 = time.time()
    results = write_ob_per_symbol(out_root, frame)
    log.info("wrote %d per-symbol partitions in %.1fs",
             len(results), time.time() - t2)

    # Step 4: cleanup GCS shards.
    n_deleted = cleanup_ob_shards(storage_client, bucket, blob_prefix)
    log.info("cleaned %d GCS shards", n_deleted)

    total_rows = sum(r.n_rows_written for r in results)
    console.print(
        f"[bold]done.[/bold] symbols={len(results)} rows={total_rows} "
        f"shards={n_shards} elapsed={time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
