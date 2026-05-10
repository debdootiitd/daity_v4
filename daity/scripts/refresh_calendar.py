"""Refresh `nse_trading_calendar` from `pandas_market_calendars` (XNSE).

Pulls the canonical NSE trading-day list from `pandas_market_calendars`,
stages it in BigQuery as `nse_trading_calendar__staging_<run_id>`, diffs
against the production table, and MERGEs only when `--apply` is set.

Two-step flow is deliberate:
    daity-refresh-calendar                 # stage + show diff, no mutation
    daity-refresh-calendar --apply         # also MERGE staging into prod

Notes
-----
- Source is `pandas_market_calendars.get_calendar('XNSE')`. Its `.name`
  attribute prints "BSE" for legacy reasons but the holiday set is the same
  as NSE (the two exchanges share the calendar).
- `nse_holidays` is intentionally NOT touched here. That table carries
  human-readable holiday names; XNSE's rule-based output doesn't preserve
  those, so backfilling it is a separate, smaller concern.
- The refresh range backs up `--lookback-days` from the prod last-day for a
  sanity overlap (default 30) and extends `--forward-days` past today
  (default 365). Any existing prod row that XNSE *omits* in the overlap is
  flagged as a discrepancy and aborts the run unless `--allow-discrepancy`
  is set.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from typing import Any

import click
import pandas as pd
import pandas_market_calendars as mcal
import polars as pl
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

CALENDAR_TABLE = "nse_trading_calendar"
STAGING_PREFIX = "nse_trading_calendar__staging_"
XNSE_CALENDAR = "XNSE"


def _run_id() -> str:
    """Stable per-invocation tag for the staging table name."""
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _read_existing(bq: BQClient) -> pl.DataFrame:
    """Pull the current `nse_trading_calendar` into a Polars frame."""
    fq = bq.cfg.fq_table(CALENDAR_TABLE)
    rows = bq.query_rows(f"SELECT trade_date FROM `{fq}` ORDER BY trade_date")
    if not rows:
        return pl.DataFrame({"trade_date": []}, schema={"trade_date": pl.Date})
    return pl.DataFrame(rows).with_columns(pl.col("trade_date").cast(pl.Date))


def _xnse_trading_days(start: date, end: date) -> list[date]:
    """Return XNSE trading days in `[start, end]` as Python `date`s (date-only)."""
    cal = mcal.get_calendar(XNSE_CALENDAR)
    schedule = cal.valid_days(start_date=start.isoformat(), end_date=end.isoformat())
    return [pd.Timestamp(d).date() for d in schedule]


def _stage(bq: BQClient, frame: pl.DataFrame, *, run_id: str) -> str:
    """Write `frame` (single column `trade_date`) to a staging table; return its name."""
    from google.cloud import bigquery

    staging = f"{STAGING_PREFIX}{run_id}"
    fq = bq.cfg.fq_table(staging)
    log.info("Staging %d trading days to %s", frame.height, fq)
    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField("trade_date", "DATE", mode="REQUIRED")],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    pdf = frame.to_pandas()
    job = bq._client.load_table_from_dataframe(pdf, fq, job_config=job_config)
    job.result()
    return staging


def _drop_staging(bq: BQClient, staging: str) -> None:
    fq = bq.cfg.fq_table(staging)
    bq._client.delete_table(fq, not_found_ok=True)
    log.info("Dropped %s", fq)


def _diff(existing: pl.DataFrame, refresh: pl.DataFrame) -> dict[str, Any]:
    """Compare existing prod rows vs. the refresh-source rows.

    Discrepancies are dates that fall inside the *intersection* of the two
    date ranges and are present in prod but absent in the refresh source —
    i.e., something the existing pipeline thought was a trading day but
    XNSE doesn't. (Dates in prod outside the refresh window are not
    discrepancies — they're just history XNSE wasn't asked about.)
    """
    e = set(existing["trade_date"].to_list())
    r = set(refresh["trade_date"].to_list())
    new = sorted(r - e)
    overlap_window_lo = max(min(e), min(r)) if e and r else None
    overlap_window_hi = min(max(e), max(r)) if e and r else None
    discrepancies: list[date] = []
    if overlap_window_lo is not None and overlap_window_hi is not None and overlap_window_lo <= overlap_window_hi:
        for d in sorted(e):
            if d < overlap_window_lo or d > overlap_window_hi:
                continue
            if d not in r:
                discrepancies.append(d)
    return {
        "n_existing": len(e),
        "n_refresh": len(r),
        "n_new": len(new),
        "first_new": new[0] if new else None,
        "last_new": new[-1] if new else None,
        "discrepancies": discrepancies,
        "overlap_window": (overlap_window_lo, overlap_window_hi),
    }


def _print_diff(d: dict[str, Any]) -> None:
    tbl = Table(title="nse_trading_calendar diff")
    tbl.add_column("metric")
    tbl.add_column("value")
    tbl.add_row("rows in prod", str(d["n_existing"]))
    tbl.add_row("rows in refresh source", str(d["n_refresh"]))
    tbl.add_row("new days to insert", str(d["n_new"]))
    tbl.add_row("first new day", str(d["first_new"] or "—"))
    tbl.add_row("last new day", str(d["last_new"] or "—"))
    tbl.add_row("overlap window", " .. ".join(str(x) for x in d["overlap_window"] if x))
    tbl.add_row(
        "discrepancies (in prod but absent in XNSE on overlap)",
        f"{len(d['discrepancies'])} → {d['discrepancies'][:5]}",
    )
    console.print(tbl)


def _apply_merge(bq: BQClient, staging: str) -> int:
    """MERGE staging into prod; return the number of inserted rows."""
    prod_fq = bq.cfg.fq_table(CALENDAR_TABLE)
    stg_fq = bq.cfg.fq_table(staging)
    sql = f"""
        MERGE `{prod_fq}` T
        USING `{stg_fq}` S
        ON T.trade_date = S.trade_date
        WHEN NOT MATCHED THEN INSERT (trade_date) VALUES (S.trade_date)
    """
    job = bq._client.query(sql, job_config=bq._job_config())
    job.result()
    n = int(job.num_dml_affected_rows or 0)
    log.info("MERGE inserted %d new rows into %s", n, prod_fq)
    return n


@click.command()
@click.option("--lookback-days", default=30, show_default=True,
              help="Days BEFORE the prod last-day to also pull (sanity overlap).")
@click.option("--forward-days", default=365, show_default=True,
              help="Days AFTER today to extend the calendar.")
@click.option("--apply", "do_apply", is_flag=True,
              help="Actually MERGE staging into nse_trading_calendar (otherwise dry-run).")
@click.option("--allow-discrepancy", is_flag=True,
              help="Proceed even if prod has trading days that XNSE marks non-trading.")
@click.option("--keep-staging", is_flag=True,
              help="Don't drop the staging table after the run.")
def main(lookback_days: int, forward_days: int, do_apply: bool,
         allow_discrepancy: bool, keep_staging: bool) -> None:
    """Refresh `nse_trading_calendar` from XNSE via pandas_market_calendars."""
    try:
        cfg = load_env()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)

    existing = _read_existing(bq)
    last_known = (existing["trade_date"].max() if existing.height else None)
    today = date.today()
    pull_start = (
        (last_known - timedelta(days=lookback_days)) if last_known else (today - timedelta(days=365))
    )
    pull_end = today + timedelta(days=forward_days)
    console.print(
        f"[bold]Refresh window:[/bold] {pull_start} → {pull_end}  "
        f"(prod last-known: {last_known}, today: {today})"
    )

    days = _xnse_trading_days(pull_start, pull_end)
    refresh = pl.DataFrame({"trade_date": days}).with_columns(
        pl.col("trade_date").cast(pl.Date)
    )

    diff = _diff(existing, refresh)
    _print_diff(diff)

    if diff["discrepancies"] and not allow_discrepancy:
        console.print(
            "[red]ABORT:[/red] prod has trading days that XNSE marks non-trading. "
            "Inspect them and re-run with --allow-discrepancy if you intend to keep them."
        )
        sys.exit(3)

    if diff["n_new"] == 0:
        console.print("[green]Nothing to insert.[/green]")
        sys.exit(0)

    run_id = _run_id()
    staging = _stage(bq, refresh, run_id=run_id)
    try:
        if do_apply:
            n = _apply_merge(bq, staging)
            console.print(f"[green]MERGE complete:[/green] {n} new rows inserted into {CALENDAR_TABLE}")
        else:
            console.print(
                f"[yellow]DRY RUN:[/yellow] staging table left at "
                f"{bq.cfg.fq_table(staging)} for inspection. Re-run with --apply to MERGE."
            )
    finally:
        if not keep_staging and do_apply:
            _drop_staging(bq, staging)


if __name__ == "__main__":
    main()
