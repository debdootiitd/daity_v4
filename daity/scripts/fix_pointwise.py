"""Replace specific OHLCV rows in `raw_ohlcv` with values re-fetched from Kite.

Used to repair pointwise data-entry errors that aren't multiplicative (so
`daity-readjust-symbols` can't fix them) — e.g. PIIND in early Jan 2019,
where some bars show a ~9× decimal-misread vs the true close.

Flow (per --symbol × --from..--to):
  1. Pull all Kite candles for the window across the configured intervals.
  2. Stage in BQ as `raw_ohlcv__pointwise_<run_id>`.
  3. Diff vs prod: count rows where |staged - prod| / prod > --threshold.
  4. On --apply, MERGE into raw_ohlcv with WHEN MATCHED THEN UPDATE for
     diverging rows + WHEN NOT MATCHED THEN INSERT for prod-missing rows.
     Rows already matching are not touched.

Always restricted to one symbol at a time so the blast radius is small.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import polars as pl
from kiteconnect.exceptions import KiteException
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.data.kite import INTERVAL_MAP, KiteClient
from daity.data.ohlcv_io import (
    OHLCV_TABLE,
    candles_to_frame,
    drop_staging,
    merge_with_update,
    stage_to_bq,
)
from daity.utils.env import EnvError, load_env, load_kite
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

STAGING_PREFIX = "raw_ohlcv__pointwise_"
DEFAULT_INTERVALS: tuple[str, ...] = ("5m", "15m", "60m", "day")


def _diff_summary(bq: BQClient, staging: str, *, threshold: float) -> list[dict]:
    """Per-scale: count rows that diverge / match / are missing in prod."""
    prod = bq.cfg.fq_table(OHLCV_TABLE)
    stg = bq.cfg.fq_table(staging)
    sql = f"""
        WITH ov AS (
          SELECT S.symbol, S.scale, S.ts, S.close AS s_close, P.close AS p_close,
                 SAFE_DIVIDE(ABS(S.close - P.close), NULLIF(P.close, 0)) AS rel_diff
          FROM `{stg}` S LEFT JOIN `{prod}` P
            ON P.symbol = S.symbol AND P.ts = S.ts AND P.scale = S.scale
        )
        SELECT scale,
               COUNT(*) AS n_staged,
               COUNTIF(p_close IS NULL) AS n_missing_in_prod,
               COUNTIF(p_close IS NOT NULL AND rel_diff > {threshold}) AS n_diverge,
               COUNTIF(p_close IS NOT NULL AND rel_diff <= {threshold}) AS n_match
        FROM ov GROUP BY scale ORDER BY scale
    """
    return bq.query_rows(sql)


@click.command()
@click.option("--symbol", required=True, help="The single symbol to fix.")
@click.option("--from", "ts_from", required=True, help="ISO date, IST. Inclusive lower.")
@click.option("--to", "ts_to", required=True, help="ISO date, IST. Inclusive upper.")
@click.option("--intervals", default=",".join(DEFAULT_INTERVALS), show_default=True,
              help="Comma-separated subset of {5m,15m,60m,day}.")
@click.option("--threshold", default=0.005, show_default=True,
              help="Min |staged-prod|/prod for a row to count as diverging.")
@click.option("--throttle-ms", default=350, show_default=True, help="Sleep between Kite calls.")
@click.option("--apply", "do_apply", is_flag=True, help="Actually run the MERGE.")
@click.option("--keep-staging", is_flag=True, help="Don't drop the staging table.")
def main(symbol: str, ts_from: str, ts_to: str, intervals: str,
         threshold: float, throttle_ms: int, do_apply: bool,
         keep_staging: bool) -> None:
    """Pointwise repair: replace specific raw_ohlcv rows from Kite for one symbol."""
    iv_set = tuple(s.strip() for s in intervals.split(",") if s.strip())
    bad = [iv for iv in iv_set if iv not in INTERVAL_MAP]
    if bad:
        console.print(f"[red]Unknown intervals: {bad}; expected subset of {list(INTERVAL_MAP)}[/red]")
        sys.exit(2)

    try:
        cfg = load_env(); kcfg = load_kite()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)
    kc = KiteClient(kcfg)
    kc.load_instruments_cache(Path("./data/cache"))

    inst = kc.lookup(symbol)
    if not inst or inst.get("instrument_token") is None:
        raise click.UsageError(f"Could not resolve {symbol!r} in Kite instruments dump.")
    token = int(inst["instrument_token"])

    start = datetime.fromisoformat(ts_from).replace(tzinfo=UTC)
    end = (datetime.fromisoformat(ts_to) + timedelta(days=1)).replace(tzinfo=UTC)
    console.print(
        f"[bold]Fix plan:[/bold] symbol={symbol} (token={token}) "
        f"window=[{ts_from}, {ts_to}] IST  intervals={list(iv_set)}  "
        f"threshold={threshold:.1%}"
    )

    frames: list[pl.DataFrame] = []
    throttle = throttle_ms / 1000.0
    for iv in iv_set:
        try:
            cands = kc.candles(token, interval=iv, start=start, end=end)
        except KiteException as exc:
            console.print(f"[red]Kite failed for {iv}:[/red] {exc}")
            sys.exit(3)
        if cands:
            frames.append(candles_to_frame(symbol, iv, cands))
        if throttle > 0:
            time.sleep(throttle)
    if not frames:
        console.print("[yellow]Kite returned no candles. Nothing to do.[/yellow]")
        return

    staged = pl.concat(frames, how="vertical")
    console.print(f"[bold]Fetched:[/bold] {staged.height:,} rows from Kite.")

    staging = f"{STAGING_PREFIX}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    stage_to_bq(bq, staged, staging)

    summary = _diff_summary(bq, staging, threshold=threshold)
    tbl = Table(title=f"{symbol}: pointwise diff vs prod (threshold={threshold:.1%})")
    tbl.add_column("scale"); tbl.add_column("staged"); tbl.add_column("missing in prod")
    tbl.add_column("diverging > thr"); tbl.add_column("already matching")
    total_diverge = total_missing = total_match = 0
    for r in summary:
        tbl.add_row(r["scale"], f"{r['n_staged']:,}", f"{r['n_missing_in_prod']:,}",
                    f"{r['n_diverge']:,}", f"{r['n_match']:,}")
        total_diverge += r["n_diverge"]; total_missing += r["n_missing_in_prod"]
        total_match += r["n_match"]
    console.print(tbl)
    console.print(
        f"Will [bold]UPDATE {total_diverge:,}[/bold] diverging + "
        f"[bold]INSERT {total_missing:,}[/bold] missing; "
        f"{total_match:,} rows already match (untouched)."
    )

    if not do_apply:
        console.print(
            f"[yellow]DRY RUN.[/yellow] Staging at "
            f"[bold]{bq.cfg.fq_table(staging)}[/bold]. Re-run with --apply."
        )
        return

    n = merge_with_update(bq, staging, threshold=threshold)
    console.print(f"[green]MERGE complete:[/green] {n:,} rows affected in {OHLCV_TABLE}.")
    if not keep_staging:
        drop_staging(bq, staging)


if __name__ == "__main__":
    main()
