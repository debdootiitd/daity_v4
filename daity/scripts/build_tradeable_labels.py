"""`daity-build-tradeable-labels` — Phase 3X recsys-style ternary label CLI.

Builds per-symbol parquets of `(ts, t, tradeable_label ∈ {-1, 0, +1}, …aux)`
under `data/tradeable_labels/{config_tag}/{symbol}.parquet`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
from rich.console import Console

from daity.data.parquet_store import ParquetStore
from daity.data.tradeable_labels import (
    TradeableLabelConfig,
    build_tradeable_labels_for_symbol,
    class_distribution,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _worker(symbol: str, parquet_root: Path, out_dir: Path,
            as_of: datetime, cfg_dict: dict, overwrite: bool) -> tuple[str, int, str | None]:
    try:
        cfg = TradeableLabelConfig(**cfg_dict)
        out_path = out_dir / f"{symbol}.parquet"
        if out_path.exists() and not overwrite:
            return symbol, 0, None
        df = build_tradeable_labels_for_symbol(
            parquet_root=parquet_root, symbol=symbol, as_of=as_of, cfg=cfg,
        )
        if df.height == 0:
            return symbol, 0, None
        df.write_parquet(out_path)
        return symbol, df.height, None
    except Exception as e:
        return symbol, -1, str(e)


@click.command()
@click.option("--parquet-root", type=click.Path(path_type=Path),
              default=Path("data/parquet"), show_default=True,
              help="Raw OHLCV parquet root (NOT data/features_parquet).")
@click.option("--out-root", type=click.Path(path_type=Path),
              default=Path("data/tradeable_labels"), show_default=True)
@click.option("--symbols", default=None,
              help="Comma-separated symbols. Default: all in parquet store's 5m partitioned dir.")
@click.option("--max-symbols", type=int, default=None)
@click.option("--as-of", default=None,
              help="Leakage cap (ISO). Default: today + 1d.")
@click.option("--trigger-threshold", type=float, default=0.005, show_default=True)
@click.option("--floor-threshold", type=float, default=0.003, show_default=True)
@click.option("--d-min-minutes", type=int, default=5, show_default=True)
@click.option("--d-max-minutes", type=int, default=30, show_default=True)
@click.option("--k-minutes", type=int, default=60, show_default=True)
@click.option("--interval", default="5m", show_default=True)
@click.option("--max-workers", type=int, default=64, show_default=True)
@click.option("--force", is_flag=True, help="Overwrite existing parquet.")
def main(
    parquet_root: Path, out_root: Path,
    symbols: str | None, max_symbols: int | None,
    as_of: str | None,
    trigger_threshold: float, floor_threshold: float,
    d_min_minutes: int, d_max_minutes: int, k_minutes: int,
    interval: str, max_workers: int, force: bool,
) -> None:
    """Build tradeable-candidate labels."""
    cfg = TradeableLabelConfig(
        trigger_threshold=trigger_threshold, floor_threshold=floor_threshold,
        d_min_minutes=d_min_minutes, d_max_minutes=d_max_minutes,
        k_minutes=k_minutes, interval=interval,
    )
    as_of_dt = (
        datetime.fromisoformat(as_of.replace("Z", "+00:00")).astimezone(UTC)
        if as_of else datetime.now(UTC) + timedelta(days=1)
    )
    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        sym_list = sorted(p.stem for p in (parquet_root / interval).glob("*.parquet"))
    if max_symbols is not None:
        sym_list = sym_list[:max_symbols]
    if not sym_list:
        raise click.UsageError(f"no symbols at {parquet_root}/{interval}/")

    from dataclasses import asdict
    import json
    out_dir = out_root / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    console.print(
        f"[bold]Tradeable labels:[/bold] {len(sym_list)} symbols, "
        f"as_of={as_of_dt.isoformat()}\n"
        f"  config: trigger={cfg.trigger_threshold:.3f} floor={cfg.floor_threshold:.3f} "
        f"d_min={cfg.d_min_minutes}m d_max={cfg.d_max_minutes}m k={cfg.k_minutes}m\n"
        f"  out: {out_dir}",
    )

    total_rows = 0; errors = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        from dataclasses import asdict
        cfg_dict = asdict(cfg)
        futs = [
            ex.submit(_worker, s, parquet_root, out_dir, as_of_dt, cfg_dict, force)
            for s in sym_list
        ]
        for fut in as_completed(futs):
            sym, n, err = fut.result()
            if err:
                console.print(f"[red]ERROR[/red] {sym}: {err}")
                errors += 1
                continue
            total_rows += n

    # Class distribution summary.
    dist = class_distribution(out_root, cfg)
    console.print(f"\n[bold]Done.[/bold] symbols={len(sym_list)} errored={errors} rows={total_rows}")
    console.print(f"[bold]Class distribution:[/bold]")
    console.print(f"  +1 LONG:     {dist['n_long']:>10,d}  ({dist['pct_long']:5.2f}%)")
    console.print(f"   0 neutral:  {dist['n_neutral']:>10,d}  ({dist['pct_neutral']:5.2f}%)")
    console.print(f"  -1 SHORT:    {dist['n_short']:>10,d}  ({dist['pct_short']:5.2f}%)")
    console.print(f"  invalid:    {dist['n_invalid']:>10,d}")


if __name__ == "__main__":
    main()
