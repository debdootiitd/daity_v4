"""`daity-precompute-features` — one-shot Phase-1 feature engine over
the full local Parquet store.

Reads from `data/parquet/{scale}/{symbol}.parquet`, writes to
`data/features_parquet/{scale}/{symbol}.parquet` with 18 channels (per
`daity.data.feature_precompute.CHANNEL_ORDER_18`).

This is the precompute step for the v3_features experiment — the
on-the-fly alternative would slow the dataloader by 100-400 ms/batch
(rolling vol-z over 20 bars × 4 scales × batch=512), regressing the
99% GPU util we got from the spawn-worker fix. Precomputing once into
a separate Parquet store is the right trade.

Run:
    daity-precompute-features                 # all symbols × all scales
    daity-precompute-features --symbols RELIANCE,TCS --scales 5m,day  # subset
    daity-precompute-features --force         # recompute even if outputs exist
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, TaskID

from daity.data.feature_precompute import precompute_features
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _precompute_one(
    parquet_root: Path, feature_root: Path, symbol: str, scale: str,
    as_of: datetime, force: bool,
) -> tuple[str, str, int, str]:
    """Worker entrypoint: precompute one (symbol, scale).

    Returns (symbol, scale, n_rows, status). `status` is "ok" / "skip" /
    "error: <msg>". The row count distinguishes a real precompute from
    a skip: a skip returns the existing file's row count and status="skip".
    """
    raw_store = ParquetStore(parquet_root)
    out_path = feature_root / scale / f"{symbol}.parquet"
    pre_existed = out_path.exists() and not force
    try:
        n_rows = precompute_features(
            raw_store=raw_store, feature_root=feature_root,
            symbol=symbol, scale=scale, as_of=as_of, force=force,
        )
        status = "skip" if pre_existed else "ok"
        return (symbol, scale, n_rows, status)
    except Exception as exc:  # noqa: BLE001
        return (symbol, scale, 0, f"error: {exc}")


@click.command()
@click.option("--parquet-root", type=click.Path(path_type=Path),
              default=Path("data/parquet"), show_default=True,
              help="Root of the source Parquet store.")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True,
              help="Root of the output feature store.")
@click.option("--scales", default="5m,15m,60m,day",
              help="Comma-separated scales to precompute.")
@click.option("--symbols", default=None,
              help="Comma-separated symbols (default: all on disk).")
@click.option("--as-of", default=None,
              help="Hard upper bound on input bars. Default = today + 1d (UTC).")
@click.option("--max-workers", type=int, default=8, show_default=True,
              help="Process pool size for parallel precompute.")
@click.option("--force", is_flag=True,
              help="Recompute even if output files exist.")
def main(
    parquet_root: Path, feature_root: Path,
    scales: str, symbols: str | None, as_of: str | None,
    max_workers: int, force: bool,
) -> None:
    """Precompute the 18-channel feature table for the full universe."""
    scales_list = [s.strip() for s in scales.split(",") if s.strip()]
    raw_store = ParquetStore(parquet_root)

    # Determine the symbol universe — either user-supplied or all on disk.
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        # Use the day partitions as the canonical roster (every symbol with
        # a day partition has all four scales by ingestion contract).
        symbol_list = sorted(p.stem for p in raw_store.list_partitions("day"))
    if not symbol_list:
        raise click.UsageError(
            f"No symbols found at {parquet_root}/day/. Run `daity-ingest-parquet` first."
        )

    # Default as_of: today + 1d in UTC. Generous upper bound that includes
    # whatever's currently ingested.
    as_of_dt = (
        datetime.fromisoformat(as_of).replace(tzinfo=UTC)
        if as_of is not None
        else datetime.now(UTC) + timedelta(days=1)
    )

    feature_root.mkdir(parents=True, exist_ok=True)

    pairs = [(sym, sc) for sc in scales_list for sym in symbol_list]
    console.print(
        f"[bold]Precompute:[/bold] {len(symbol_list)} symbols × "
        f"{len(scales_list)} scales = {len(pairs)} (symbol, scale) pairs\n"
        f"  source:  {parquet_root}\n"
        f"  output:  {feature_root}\n"
        f"  as_of:   {as_of_dt.isoformat()}\n"
        f"  workers: {max_workers}  force={force}"
    )

    t0 = time.time()
    results: list[tuple[str, str, int, str]] = []
    with Progress(console=console) as progress:
        task: TaskID = progress.add_task("[cyan]precomputing...", total=len(pairs))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_precompute_one, parquet_root, feature_root,
                          sym, sc, as_of_dt, force)
                for sym, sc in pairs
            ]
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                progress.update(task, advance=1)

    elapsed = time.time() - t0

    # Summarize.
    n_ok = sum(1 for r in results if r[3] == "ok")
    n_skip = sum(1 for r in results if r[3] == "skip")
    n_err = sum(1 for r in results if r[3].startswith("error"))
    total_rows = sum(r[2] for r in results)
    console.print(
        f"\n[bold]Done in {elapsed:.1f}s.[/bold]  "
        f"{n_ok} computed, {n_skip} skipped, {n_err} errored. "
        f"Total {total_rows:,} feature rows."
    )
    if n_err:
        console.print("\n[red]Errors:[/red]")
        for sym, sc, _, status in results:
            if status.startswith("error"):
                console.print(f"  {scale}/{sym}: {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
