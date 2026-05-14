"""`daity-build-residual-labels` — Phase 3.4c residualized labels CLI.

Per-horizon, per-symbol parquet at `data/labels_resid/{horizon}/{symbol}.parquet`
with `y_resid` column (sector-cross-sectionally demeaned `y_log_return`).

Default: residualize 1d + 2d only (intraday horizons stay raw — sector
β is too noisy at 30m / 120m).
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from daity.data.factor_residual import (
    compute_residual_labels,
    load_sector_mapping,
    sector_mapping_from_default_csv,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


@click.command()
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path),
              default=Path("data/labels_resid"), show_default=True)
@click.option("--horizons", default="1d,2d", show_default=True,
              help="Comma-separated horizons to residualize.")
@click.option("--symbol-master", type=click.Path(path_type=Path),
              default=Path("data/cache/symbols/symbol_master.parquet"),
              show_default=True,
              help="SymbolMaster parquet (provides symbol→sector).")
@click.option("--sector-csv", type=click.Path(path_type=Path),
              default=None,
              help="Fallback symbol,sector CSV if --symbol-master missing.")
@click.option("--symbols", default=None,
              help="Comma-separated subset of symbols. Default: all.")
@click.option("--force", is_flag=True, help="Overwrite existing parquet.")
def main(label_root: Path, out_root: Path, horizons: str,
         symbol_master: Path, sector_csv: Path | None,
         symbols: str | None, force: bool) -> None:
    """Build sector-relative residualized labels."""
    hz = tuple(h.strip() for h in horizons.split(",") if h.strip())

    sec_map = load_sector_mapping(symbol_master)
    if not sec_map and sector_csv is not None:
        sec_map = sector_mapping_from_default_csv(sector_csv)
    if not sec_map:
        console.print(
            "[yellow]WARNING:[/yellow] no sector mapping loaded. "
            "Residualization will reduce to universe demeaning (no per-sector lift)."
        )

    universe = (
        [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    )

    counts = compute_residual_labels(
        label_root=label_root, out_root=out_root,
        symbol_to_sector=sec_map, horizons=hz,
        universe=universe, overwrite=force,
    )
    console.print(f"[bold]done.[/bold] residualized:")
    for h, n in counts.items():
        console.print(f"  {h:>5s}: {n} rows -> {out_root / h}")


if __name__ == "__main__":
    main()
