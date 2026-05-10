"""Apply per-symbol corp-action adjustments to historical `raw_ohlcv` rows.

Reads a YAML config of `(symbol, ratio, cutoff_date_ist)` entries and runs:

    UPDATE raw_ohlcv
    SET open=open*ratio, high=high*ratio, low=low*ratio, close=close*ratio,
        volume=CAST(volume / ratio AS INT64)
    WHERE symbol = <symbol>
      AND DATE(ts, 'Asia/Kolkata') < <cutoff_date_ist>     -- or always-true if null

Applies *across all `scale` values* (5m / 15m / 60m / day) — splits and
bonuses affect every bar timeframe equally.

**Idempotency.** Every applied `(symbol, cutoff_date_ist, ratio)` triple
is recorded in `data/applied_readjustments.json`. Re-runs are no-ops. Use
`--allow-rerun` to deliberately re-apply (e.g., after a manual UNDO). The
marker is the load-bearing safety net — the WHERE clause alone is *not*
idempotent.

Run:
    daity-readjust-symbols                            # dry-run
    daity-readjust-symbols --apply                    # actual UPDATE
    daity-readjust-symbols --apply --allow-rerun      # bypass guard
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.data.readjust import (
    Readjustment,
    filter_already_applied,
    load_config,
    record_applied,
    where_clause,
)
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

OHLCV_TABLE = "raw_ohlcv"
DEFAULT_CONFIG = Path("configs/data/corp_actions.yaml")
DEFAULT_MARKER = Path("data/applied_readjustments.json")


def _preview(bq: BQClient, r: Readjustment) -> dict[str, Any]:
    """Count rows that would be affected by one readjustment."""
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    where = where_clause(r)
    sql = f"""
        WITH affected AS (
          SELECT * FROM `{fq}` WHERE {where}
        )
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT scale) AS n_scales,
          COUNT(DISTINCT DATE(ts, 'Asia/Kolkata')) AS n_dates,
          MIN(DATE(ts, 'Asia/Kolkata')) AS first_date,
          MAX(DATE(ts, 'Asia/Kolkata')) AS last_date
        FROM affected
    """
    rows = bq.query_rows(sql)
    return rows[0] if rows else {}


def _apply(bq: BQClient, plan: list[Readjustment]) -> dict[str, int]:
    """Run each UPDATE separately; return {symbol: rows_affected}."""
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    counts: dict[str, int] = {}
    for r in plan:
        where = where_clause(r)
        sql = (
            f"UPDATE `{fq}` "
            f"SET open = open * {r.ratio}, high = high * {r.ratio}, "
            f"low = low * {r.ratio}, close = close * {r.ratio}, "
            f"volume = CAST(volume / {r.ratio} AS INT64) "
            f"WHERE {where}"
        )
        log.info("Applying readjustment for %s (ratio=%g): %s", r.symbol, r.ratio, where)
        job = bq._client.query(sql, job_config=bq._job_config())
        job.result()
        counts[r.symbol] = int(job.num_dml_affected_rows or 0)
        log.info("  → %s: %d rows updated", r.symbol, counts[r.symbol])
    return counts


@click.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path),
              default=DEFAULT_CONFIG, show_default=True,
              help="YAML config of readjustment entries.")
@click.option("--symbols", default=None,
              help="Comma-separated subset of symbols (default: all entries).")
@click.option("--apply", "do_apply", is_flag=True, help="Actually run the UPDATEs.")
@click.option("--marker-path", type=click.Path(path_type=Path),
              default=DEFAULT_MARKER, show_default=True,
              help="Marker JSON recording every applied (symbol, cutoff, ratio).")
@click.option("--allow-rerun", is_flag=True,
              help="Bypass the marker idempotency guard.")
def main(config_path: Path, symbols: str | None, do_apply: bool,
         marker_path: Path, allow_rerun: bool) -> None:
    """Re-apply per-symbol corp-action adjustments to raw_ohlcv historical rows."""
    try:
        cfg = load_env()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)

    plan = load_config(config_path)
    if symbols:
        wanted = {s.strip() for s in symbols.split(",") if s.strip()}
        plan = [r for r in plan if r.symbol in wanted]
    if not plan:
        console.print("[yellow]Nothing to do (empty config or filtered out).[/yellow]")
        return

    todo, already_applied = filter_already_applied(plan, marker_path)
    if already_applied and not allow_rerun:
        console.print(
            f"[yellow]Skipping {len(already_applied)} entries already applied "
            f"per {marker_path}:[/yellow]"
        )
        for r in already_applied:
            console.print(f"  ALREADY APPLIED  {r.symbol}  ratio={r.ratio:g}  "
                          f"cutoff={r.cutoff_date_ist or '(all)'}")
        if not todo:
            console.print(
                "[yellow]Nothing new to apply.[/yellow] Use --allow-rerun if "
                "you've manually undone an earlier readjust and need to redo it."
            )
            return
        plan = todo
    elif allow_rerun and already_applied:
        console.print(
            f"[red]--allow-rerun:[/red] re-applying {len(already_applied)} "
            f"previously applied entries. This will multiply the ratio AGAIN."
        )

    bq = BQClient(cfg)

    tbl = Table(title=f"Re-adjustment plan ({config_path})")
    tbl.add_column("symbol"); tbl.add_column("ratio"); tbl.add_column("cutoff (IST)")
    tbl.add_column("rows"); tbl.add_column("scales"); tbl.add_column("dates")
    tbl.add_column("date range"); tbl.add_column("event")
    total = 0
    for r in plan:
        p = _preview(bq, r)
        n = int(p.get("n_rows", 0) or 0)
        total += n
        tbl.add_row(
            r.symbol, f"{r.ratio:g}",
            r.cutoff_date_ist.isoformat() if r.cutoff_date_ist else "(all)",
            f"{n:,}", str(p.get("n_scales", 0)), str(p.get("n_dates", 0)),
            f"{p.get('first_date')} .. {p.get('last_date')}", r.event,
        )
    console.print(tbl)
    console.print(f"[bold]Total rows touched:[/bold] {total:,}")

    if not do_apply:
        console.print(
            "[yellow]DRY RUN.[/yellow] Re-run with [bold]--apply[/bold] to execute."
        )
        return

    counts = _apply(bq, plan)
    res = Table(title="Applied")
    res.add_column("symbol"); res.add_column("rows updated")
    grand = 0
    for sym, n in counts.items():
        res.add_row(sym, f"{n:,}")
        grand += n
    console.print(res)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    record_applied(marker_path, plan, run_id=run_id, n_rows_per_sym=counts)
    console.print(f"[green]Done:[/green] {grand:,} rows updated across "
                  f"{len(counts)} symbols. Marker updated (run_id={run_id}).")


if __name__ == "__main__":
    main()
