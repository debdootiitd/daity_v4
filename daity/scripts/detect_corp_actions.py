"""Auto-detect missed corp-action adjustments by comparing prod vs a Kite-fetched
staging table. Outputs proposals to YAML for review.

The CLI is a thin orchestrator: pull (symbol, ts, ratio) triples from a BQ
staging table → segment per-symbol → emit YAML in the schema
`daity-readjust-symbols` consumes. See `daity.data.corp_action_detect` for
the segmentation logic.

Workflow:
    daity-refresh-ohlcv --intervals day --overlap-days 365   # populate staging
    daity-detect-corp-actions --staging raw_ohlcv__staging_<id>
    # → review configs/data/corp_actions.detected.yaml
    # → merge entries into corp_actions.yaml
    # → daity-readjust-symbols --apply
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.data.corp_action_detect import (
    Segment,
    proposed_readjustments,
    segments_for_symbol,
)
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

OHLCV_TABLE = "raw_ohlcv"
DEFAULT_RATIO_NOISE = 0.005   # within-segment tolerance
DEFAULT_DIVIDEND_BAND = 0.05  # ratios in [1±this] of 1.0 skipped
DEFAULT_MIN_SEGMENT_DAYS = 3  # drop transient blips (1-2 day vol spikes)


def _detect(bq: BQClient, staging: str, *, ratio_noise: float, dividend_band: float,
            min_segment_days: int,
            ) -> tuple[dict[str, list[Segment]], list[dict]]:
    """Pull staged-vs-prod ratios from BQ and run segmentation per symbol."""
    fq_stg = bq.cfg.fq_table(staging)
    fq_prod = bq.cfg.fq_table(OHLCV_TABLE)
    sql = f"""
        SELECT
          S.symbol AS symbol,
          DATE(S.ts, 'Asia/Kolkata') AS dt,
          ROUND(SAFE_DIVIDE(S.close, NULLIF(P.close, 0)), 6) AS ratio
        FROM `{fq_stg}` S
        INNER JOIN `{fq_prod}` P
          ON P.symbol = S.symbol AND P.ts = S.ts AND P.scale = S.scale
        WHERE S.scale = 'day' AND P.close IS NOT NULL AND P.close > 0
        ORDER BY symbol, dt
    """
    rows = bq.query_rows(sql)
    by_sym: dict[str, list[dict]] = {}
    for r in rows:
        if r["ratio"] is None:
            continue
        dt = r["dt"]
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        by_sym.setdefault(r["symbol"], []).append({
            "symbol": r["symbol"], "dt": dt, "ratio": float(r["ratio"]),
        })

    all_segments: dict[str, list[Segment]] = {}
    proposals: list[dict] = []
    for sym, sym_rows in by_sym.items():
        segs = segments_for_symbol(sym_rows, ratio_noise=ratio_noise)
        all_segments[sym] = segs
        proposals.extend(proposed_readjustments(
            segs, dividend_band=dividend_band, min_segment_days=min_segment_days,
        ))
    return all_segments, proposals


def _print_summary(all_segments: dict[str, list[Segment]], proposals: list[dict]) -> None:
    by_sym: dict[str, list[dict]] = {}
    for p in proposals:
        by_sym.setdefault(p["symbol"], []).append(p)
    console.print(
        f"[bold]Symbols scanned:[/bold] {len(all_segments)}  "
        f"[bold]With proposed adjustments:[/bold] {len(by_sym)}  "
        f"[bold]Total proposals:[/bold] {len(proposals)}"
    )
    if not proposals:
        return
    tbl = Table(title="Proposed re-adjustments (sorted by magnitude)")
    tbl.add_column("symbol"); tbl.add_column("ratio")
    tbl.add_column("cutoff (IST)"); tbl.add_column("event/segment-boundary")
    for p in sorted(proposals, key=lambda x: abs(x["ratio"] - 1.0), reverse=True):
        tbl.add_row(p["symbol"], f"{p['ratio']:g}",
                    str(p["cutoff_date_ist"]), p["event"])
    console.print(tbl)


@click.command()
@click.option("--staging", required=True,
              help="Name of the staging table from `daity-refresh-ohlcv --overlap-days <N>`.")
@click.option("--output", type=click.Path(path_type=Path),
              default=Path("configs/data/corp_actions.detected.yaml"), show_default=True,
              help="Where to write the proposed readjust YAML.")
@click.option("--ratio-noise", default=DEFAULT_RATIO_NOISE, show_default=True,
              help="Within-segment tolerance for grouping consecutive ratios.")
@click.option("--dividend-band", default=DEFAULT_DIVIDEND_BAND, show_default=True,
              help="Ratios within [1±this] of 1.0 are skipped as likely dividends.")
@click.option("--min-segment-days", default=DEFAULT_MIN_SEGMENT_DAYS, show_default=True,
              help="Reject transitions where either adjacent segment is shorter "
                   "than N days — suppresses single-day vol-spike false positives.")
def main(staging: str, output: Path, ratio_noise: float, dividend_band: float,
         min_segment_days: int) -> None:
    """Detect missed corp-action adjustments in raw_ohlcv vs a staged Kite snapshot."""
    try:
        cfg = load_env()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)

    log.info("Detecting corp actions: staging=%s, ratio_noise=%g, dividend_band=%g, "
             "min_segment_days=%d",
             staging, ratio_noise, dividend_band, min_segment_days)
    all_segments, proposals = _detect(
        bq, staging, ratio_noise=ratio_noise, dividend_band=dividend_band,
        min_segment_days=min_segment_days,
    )
    _print_summary(all_segments, proposals)

    if proposals:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump({"readjustments": proposals},
                                          sort_keys=False, default_flow_style=False))
        console.print(
            f"\n[green]Wrote {len(proposals)} proposals to {output}.[/green]\n"
            f"Review, prune, then merge into configs/data/corp_actions.yaml and "
            f"run [bold]daity-readjust-symbols --apply[/bold]."
        )
    else:
        console.print("[green]No proposed readjustments — prod looks aligned with Kite.[/green]")


if __name__ == "__main__":
    main()
