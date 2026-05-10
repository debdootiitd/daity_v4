"""One-off fix: shift rogue daily rows in `raw_ohlcv` from 18:30 UTC
(Kite's raw 00:00 IST convention) to 10:00 UTC (our 15:30 IST close-time
convention). See `daity.data.candles` for why this matters.

NB: `curated_ohlcv` is a VIEW over `raw_ohlcv` (just renames `scale` →
`interval` and adds a derived `date` column). All physical writes go to
`raw_ohlcv` and the view picks them up automatically.

Background: a partial backfill that ran on 2025-12-15 onwards used Kite's
raw daily timestamps (00:00 IST = 18:30 UTC prev day) instead of applying
our close-time convention (15:30 IST = 10:00 UTC same day). 10,971 rogue
rows across 207 symbols sit at the wrong time-of-day and would leak future
data through any `ts <= as_of` join (since 18:30 UTC of the *previous* day
is well before market open of the affected day).

Defaults to dry-run. `--apply` runs the actual UPDATE. Belt-and-braces
checks: the WHERE clause restricts to `scale='day'` AND the exact 18:30
UTC time-of-day, so it can't accidentally touch correctly-timestamped rows
or any intraday rows.

Run:
    daity-fix-daily-ts                     # dry-run: count + sample only
    daity-fix-daily-ts --apply             # run the UPDATE
"""

from __future__ import annotations

import sys
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

OHLCV_TABLE = "raw_ohlcv"  # the underlying physical table; curated_ohlcv is a view over it

# Rogue rows: scale='day' AND time-of-day = 18:30 UTC.
# (Kite emits daily at 00:00 IST = 18:30 UTC the prior day. Our convention
# is 15:30 IST = 10:00 UTC the same day.)
ROGUE_HOUR_UTC = 18
ROGUE_MINUTE_UTC = 30

# Shift to apply: +15h30m. Brings 18:30 UTC prev day → 10:00 UTC same day in IST.
SHIFT_MINUTES = 15 * 60 + 30  # 930

WHERE_ROGUE = (
    f"scale = 'day' "
    f"AND EXTRACT(HOUR FROM ts) = {ROGUE_HOUR_UTC} "
    f"AND EXTRACT(MINUTE FROM ts) = {ROGUE_MINUTE_UTC}"
)


def _summary(bq: BQClient) -> dict[str, Any]:
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    # raw_ohlcv has no `date` column — derive the IST trading day from ts.
    sql = f"""
        SELECT
          COUNT(*) AS n_rogue,
          COUNT(DISTINCT symbol) AS n_symbols,
          MIN(DATE(ts, 'Asia/Kolkata')) AS first_date,
          MAX(DATE(ts, 'Asia/Kolkata')) AS last_date
        FROM `{fq}`
        WHERE {WHERE_ROGUE}
    """
    rows = bq.query_rows(sql)
    return rows[0] if rows else {}


def _sample(bq: BQClient, n: int = 5) -> list[dict[str, Any]]:
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    sql = f"""
        SELECT symbol, DATE(ts, 'Asia/Kolkata') AS date, ts, close
        FROM `{fq}`
        WHERE {WHERE_ROGUE}
        ORDER BY symbol, ts
        LIMIT {int(n)}
    """
    return bq.query_rows(sql)


def _verify_no_target_collision(bq: BQClient) -> int:
    """Check whether the SHIFTED ts would collide with any existing row.

    If `(symbol, ts+shift, scale)` already exists, the UPDATE would create
    duplicates. Returns the collision count; safe-to-proceed is 0.
    """
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    sql = f"""
        SELECT COUNT(*) AS n_collisions
        FROM `{fq}` rogue
        INNER JOIN `{fq}` target
          ON target.symbol = rogue.symbol
         AND target.scale = 'day'
         AND target.ts = TIMESTAMP_ADD(rogue.ts, INTERVAL {SHIFT_MINUTES} MINUTE)
        WHERE rogue.scale = 'day'
          AND EXTRACT(HOUR FROM rogue.ts) = {ROGUE_HOUR_UTC}
          AND EXTRACT(MINUTE FROM rogue.ts) = {ROGUE_MINUTE_UTC}
    """
    rows = bq.query_rows(sql)
    return int(rows[0]["n_collisions"]) if rows else 0


def _apply_update(bq: BQClient) -> int:
    fq = bq.cfg.fq_table(OHLCV_TABLE)
    sql = f"""
        UPDATE `{fq}`
        SET ts = TIMESTAMP_ADD(ts, INTERVAL {SHIFT_MINUTES} MINUTE)
        WHERE {WHERE_ROGUE}
    """
    job = bq._client.query(sql, job_config=bq._job_config())
    job.result()
    return int(job.num_dml_affected_rows or 0)


@click.command()
@click.option("--apply", "do_apply", is_flag=True,
              help="Actually run the UPDATE (default is dry-run).")
def main(do_apply: bool) -> None:
    """Migrate rogue daily ts (18:30 UTC) to the close-time convention (10:00 UTC)."""
    try:
        cfg = load_env()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)

    summary = _summary(bq)
    n_rogue = summary.get("n_rogue", 0)
    if not n_rogue:
        console.print("[green]Nothing to fix:[/green] no rogue daily rows found.")
        sys.exit(0)

    tbl = Table(title="Rogue daily rows (interval='day' AND ts time-of-day = 18:30 UTC)")
    tbl.add_column("metric"); tbl.add_column("value")
    tbl.add_row("rogue rows", f"{n_rogue:,}")
    tbl.add_row("symbols affected", f"{summary.get('n_symbols', 0)}")
    tbl.add_row("date range", f"{summary.get('first_date')} .. {summary.get('last_date')}")
    console.print(tbl)

    console.print("[bold]Sample rows (before shift):[/bold]")
    for r in _sample(bq):
        console.print(f"  {r['symbol']:<14} date={r['date']}  ts={r['ts']}  close={r['close']}")

    n_collisions = _verify_no_target_collision(bq)
    if n_collisions > 0:
        console.print(
            f"[red]ABORT:[/red] shifting would collide with {n_collisions} existing row(s) "
            f"at the post-shift ts. Investigate before proceeding (this should never "
            f"happen if prod is consistent)."
        )
        sys.exit(3)
    console.print("[green]No collisions:[/green] post-shift ts values are all unique.")

    if not do_apply:
        console.print(
            f"[yellow]DRY RUN.[/yellow] Re-run with [bold]--apply[/bold] to UPDATE "
            f"{n_rogue:,} rows: ts := ts + {SHIFT_MINUTES} minutes."
        )
        return

    n_affected = _apply_update(bq)
    console.print(f"[green]UPDATE complete:[/green] shifted {n_affected:,} rows in {OHLCV_TABLE}.")

    # Post-condition check.
    after = _summary(bq)
    if after.get("n_rogue", 0) > 0:
        console.print(
            f"[red]warning:[/red] {after['n_rogue']} rogue rows still present after UPDATE — "
            f"investigate."
        )
    else:
        console.print("[green]Verified:[/green] 0 rogue rows remaining.")


if __name__ == "__main__":
    main()
