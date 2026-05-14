"""`daity-build-labels` — Phase 3.1 forward-return labels for the full universe.

Reads from `data/parquet/{scale}/{symbol}.parquet` (raw 5m + day bars),
writes to `data/labels/{horizon}/{symbol}.parquet` for each requested
horizon. Schema and semantics live in `daity.data.labels`.

Run:
    daity-build-labels                                     # all horizons × all symbols
    daity-build-labels --horizons 30m,1d                   # subset of horizons
    daity-build-labels --symbols RELIANCE,TCS              # subset of symbols
    daity-build-labels --as-of 2026-04-30                  # leakage cap
    daity-build-labels --force                             # recompute even if outputs exist
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

from daity.data.calendar_nse import NSECalendar
from daity.data.labels import ALL_HORIZONS, precompute_labels
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _build_one(
    parquet_root: Path, label_root: Path,
    cache_dir: Path,
    symbol: str, horizon: str,
    as_of: datetime, force: bool,
) -> tuple[str, str, int, str]:
    """Worker: build labels for one (symbol, horizon).

    Returns (symbol, horizon, n_rows, status). Status: "ok" | "skip" |
    "error: <msg>".
    """
    raw_store = ParquetStore(parquet_root)
    calendar = NSECalendar.from_cache(cache_dir)
    out_path = label_root / horizon / f"{symbol}.parquet"
    pre_existed = out_path.exists() and not force
    try:
        n_rows = precompute_labels(
            raw_store=raw_store, label_root=label_root,
            symbol=symbol, horizon=horizon,
            calendar=calendar, as_of=as_of, force=force,
        )
        status = "skip" if pre_existed else "ok"
        return (symbol, horizon, n_rows, status)
    except Exception as exc:  # noqa: BLE001
        return (symbol, horizon, 0, f"error: {exc}")


@click.command()
@click.option("--parquet-root", type=click.Path(path_type=Path),
              default=Path("data/parquet"), show_default=True,
              help="Root of the source raw OHLCV parquet store.")
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True,
              help="Root of the output label store.")
@click.option("--cache-dir", type=click.Path(path_type=Path),
              default=Path("data/cache"), show_default=True,
              help="Calendar cache root (used by NSECalendar.from_cache).")
@click.option("--horizons", default=",".join(ALL_HORIZONS),
              show_default=True,
              help="Comma-separated horizons (any of: 30m, 120m, 1d, 2d).")
@click.option("--symbols", default=None,
              help="Comma-separated symbols (default: all symbols on disk).")
@click.option("--as-of", default=None,
              help="ISO 8601 leakage cap. Default = today + 1d (UTC).")
@click.option("--max-workers", type=int, default=8, show_default=True,
              help="Process pool size for parallel build.")
@click.option("--force", is_flag=True,
              help="Recompute even if output files exist.")
def main(
    parquet_root: Path, label_root: Path, cache_dir: Path,
    horizons: str, symbols: str | None, as_of: str | None,
    max_workers: int, force: bool,
) -> None:
    """Build forward-return labels for the universe at the requested horizons."""
    horizons_list = [h.strip() for h in horizons.split(",") if h.strip()]
    unknown = [h for h in horizons_list if h not in ALL_HORIZONS]
    if unknown:
        raise click.UsageError(
            f"Unknown horizon(s): {unknown}. Allowed: {list(ALL_HORIZONS)}"
        )

    raw_store = ParquetStore(parquet_root)
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        # Day partitions are the canonical roster — every fully-ingested
        # symbol has both 5m and day partitions (per ingest contract).
        symbol_list = sorted(p.stem for p in raw_store.list_partitions("day"))
    if not symbol_list:
        raise click.UsageError(
            f"No symbols found at {parquet_root}/day/. Run `daity-ingest-parquet` first."
        )

    as_of_dt = (
        datetime.fromisoformat(as_of).replace(tzinfo=UTC)
        if as_of is not None
        else datetime.now(UTC) + timedelta(days=1)
    )

    label_root.mkdir(parents=True, exist_ok=True)

    pairs = [(sym, h) for h in horizons_list for sym in symbol_list]
    console.print(
        f"[bold]Build labels:[/bold] {len(symbol_list)} symbols × "
        f"{len(horizons_list)} horizons = {len(pairs)} (symbol, horizon) pairs\n"
        f"  source:  {parquet_root}\n"
        f"  output:  {label_root}\n"
        f"  as_of:   {as_of_dt.isoformat()}\n"
        f"  workers: {max_workers}  force={force}"
    )

    t0 = time.time()
    results: list[tuple[str, str, int, str]] = []
    with Progress(console=console) as progress:
        task: TaskID = progress.add_task("[cyan]building labels...", total=len(pairs))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_build_one, parquet_root, label_root, cache_dir,
                          sym, h, as_of_dt, force)
                for sym, h in pairs
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
                progress.update(task, advance=1)

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r[3] == "ok")
    n_skip = sum(1 for r in results if r[3] == "skip")
    n_err = sum(1 for r in results if r[3].startswith("error"))
    total_rows = sum(r[2] for r in results)
    console.print(
        f"\n[bold]Done in {elapsed:.1f}s.[/bold]  "
        f"{n_ok} computed, {n_skip} skipped, {n_err} errored. "
        f"Total {total_rows:,} label rows across all horizons."
    )

    # Per-horizon row counts (sanity check for §3.1 claims).
    by_h: dict[str, int] = {h: 0 for h in horizons_list}
    for _sym, h, n, status in results:
        if status in ("ok", "skip"):
            by_h[h] += n
    console.print("\n[bold]Per-horizon totals:[/bold]")
    for h, n in by_h.items():
        console.print(f"  {h:>5s}: {n:>10,} rows")

    if n_err:
        console.print("\n[red]Errors:[/red]")
        for sym, h, _, status in results:
            if status.startswith("error"):
                console.print(f"  {h}/{sym}: {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
