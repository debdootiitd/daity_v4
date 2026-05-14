"""`daity-precompute-aux-features` — Phase 3.4b aux-features build CLI.

Writes per-symbol parquet of the 50 top XGB features at every 5m anchor.
Inputs are `data/features_parquet/` (the 18-channel store); outputs to
`data/aux_features/{symbol}.parquet`. Run once per `as_of` cap.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
from rich.console import Console

from daity.data.aux_features import write_aux_features
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _worker(symbol: str, feature_root: Path, out_root: Path,
            as_of: datetime, overwrite: bool) -> tuple[str, int, str | None]:
    try:
        from daity.data.aux_features import build_aux_features_for_symbol
        feature_store = ParquetStore(feature_root)
        out_path = out_root / f"{symbol}.parquet"
        if out_path.exists() and not overwrite:
            return symbol, 0, None
        df = build_aux_features_for_symbol(
            feature_store=feature_store, symbol=symbol, as_of=as_of,
        )
        df.write_parquet(out_path)
        return symbol, df.height, None
    except Exception as e:
        return symbol, -1, str(e)


@click.command()
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path),
              default=Path("data/aux_features"), show_default=True)
@click.option("--symbols", default=None,
              help="Comma-separated symbols. Default: all in feature store.")
@click.option("--max-symbols", type=int, default=None)
@click.option("--as-of", default=None,
              help="Leakage cap (ISO). Default: today + 1d.")
@click.option("--max-workers", type=int, default=64, show_default=True)
@click.option("--force", is_flag=True, help="Overwrite existing parquet.")
def main(feature_root: Path, out_root: Path, symbols: str | None,
         max_symbols: int | None, as_of: str | None,
         max_workers: int, force: bool) -> None:
    """Precompute aux features for all symbols."""
    feature_store = ParquetStore(feature_root)
    if symbols:
        universe = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if max_symbols is not None:
        universe = universe[:max_symbols]
    if not universe:
        raise click.UsageError(f"no symbols at {feature_root}/day/")
    as_of_dt = (
        datetime.fromisoformat(as_of.replace("Z", "+00:00")).astimezone(UTC)
        if as_of else datetime.now(UTC) + timedelta(days=1)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold]aux-features:[/bold] {len(universe)} symbols, as_of={as_of_dt.isoformat()}"
    )

    total_rows = 0; errors = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_worker, sym, feature_root, out_root, as_of_dt, force)
            for sym in universe
        ]
        for fut in as_completed(futures):
            sym, n, err = fut.result()
            if err:
                console.print(f"[red]ERROR[/red] {sym}: {err}")
                errors += 1
                continue
            total_rows += n

    console.print(
        f"[bold]done.[/bold] symbols={len(universe)} errored={errors} "
        f"rows_written={total_rows}"
    )


if __name__ == "__main__":
    main()
