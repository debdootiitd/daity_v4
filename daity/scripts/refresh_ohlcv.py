"""Refresh `raw_ohlcv` from Kite Connect for the post-existing-max-ts gap.

For each (symbol, scale) in the universe × {5m, 15m, 60m, day}, fetch from
Kite starting `--overlap-days` before the existing max-ts, stage in BQ,
spot-check vs prod on the overlap, and MERGE on `--apply`.

`curated_ohlcv` is a VIEW over `raw_ohlcv`; all writes go to `raw_ohlcv`
and the view picks them up. See `daity.data.ohlcv_io` for the building
blocks.

Run:
    daity-refresh-ohlcv                        # all scales, all symbols, dry-run
    daity-refresh-ohlcv --apply                # MERGE into raw_ohlcv
    daity-refresh-ohlcv --intervals day        # one scale only
    daity-refresh-ohlcv --symbols RELIANCE     # one symbol
    daity-refresh-ohlcv --overlap-days 365     # audit window for adjustment drift
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
    GapKey,
    candles_to_frame,
    drop_staging,
    fetch_one_window,
    merge_into_raw_ohlcv,
    read_max_ts_per_symbol_scale,
    spot_check_overlap,
    stage_to_bq,
)
from daity.data.symbols import SymbolMaster
from daity.utils.env import EnvError, load_env, load_kite
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

STAGING_PREFIX = "raw_ohlcv__staging_"
DEFAULT_INTERVALS: tuple[str, ...] = ("5m", "15m", "60m", "day")


def _build_gap_keys(
    sm: SymbolMaster, kc: KiteClient,
    last_ts: dict[tuple[str, str], datetime], scales: tuple[str, ...],
) -> list[GapKey]:
    """Cross symbols × scales, attach Kite instrument_token + last_ts from prod."""
    lookup = kc.bulk_lookup(sm.symbols)
    by_sym = {r["tradingsymbol"]: r for r in lookup.to_dicts()}
    keys: list[GapKey] = []
    missing: list[str] = []
    for sym in sm.symbols:
        row = by_sym.get(sym)
        if not row or row.get("instrument_token") is None:
            missing.append(sym)
            continue
        for sc in scales:
            keys.append(GapKey(
                symbol=sym, scale=sc,
                instrument_token=int(row["instrument_token"]),
                last_ts_utc=last_ts.get((sym, sc)),
            ))
    if missing:
        log.warning("Symbols missing in Kite instruments dump (skipped): %s", missing)
    return keys


@click.command()
@click.option("--intervals", default=",".join(DEFAULT_INTERVALS), show_default=True,
              help="Comma-separated subset of {5m,15m,60m,day}.")
@click.option("--symbols", default=None,
              help="Comma-separated subset (default: all from symbol_list).")
@click.option("--floor-start", default="2026-02-28",
              help="Start date when no prior data exists for a (symbol, scale).")
@click.option("--overlap-days", default=5, show_default=True,
              help="Re-fetch this many days BEFORE each existing max-ts. Use "
                   "365 to audit historical adjustment drift.")
@click.option("--throttle-ms", default=350, show_default=True,
              help="Sleep between Kite calls (cap is ~3/s).")
@click.option("--apply", "do_apply", is_flag=True, help="MERGE into raw_ohlcv.")
@click.option("--keep-staging", is_flag=True, help="Don't drop staging on MERGE.")
@click.option("--tolerance", default=0.005, show_default=True,
              help="Max |close diff| / prod_close on overlap rows.")
def main(intervals: str, symbols: str | None, floor_start: str, overlap_days: int,
         throttle_ms: int, do_apply: bool, keep_staging: bool, tolerance: float) -> None:
    """Refresh `raw_ohlcv` for the gap between max(ts) and now."""
    iv_set = tuple(s.strip() for s in intervals.split(",") if s.strip())
    bad = [iv for iv in iv_set if iv not in INTERVAL_MAP]
    if bad:
        console.print(f"[red]Unknown intervals: {bad}; expected subset of {list(INTERVAL_MAP)}[/red]")
        sys.exit(2)

    try:
        cfg = load_env()
        kcfg = load_kite()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)

    sm = SymbolMaster.from_bq(bq, cache_root=Path("./data/cache"))
    if symbols:
        wanted = {s.strip() for s in symbols.split(",") if s.strip()}
        sm = SymbolMaster(frame=sm.frame.filter(pl.col("symbol").is_in(list(wanted))))
    console.print(f"[bold]Universe:[/bold] {len(sm)} symbols × intervals {list(iv_set)}")

    kc = KiteClient(kcfg)
    kc.load_instruments_cache(Path("./data/cache"))

    keys = _build_gap_keys(sm, kc, read_max_ts_per_symbol_scale(bq, iv_set), iv_set)
    console.print(f"[bold]Gap keys:[/bold] {len(keys)}  (max possible = {len(sm)} × {len(iv_set)})")

    floor = datetime.fromisoformat(floor_start).replace(tzinfo=UTC)
    end = datetime.now(tz=UTC)
    overlap = timedelta(days=overlap_days)
    throttle = throttle_ms / 1000.0

    frames: list[pl.DataFrame] = []
    n_calls = n_candles = n_failures = 0
    t0 = time.time()
    for i, key in enumerate(keys, start=1):
        try:
            cands = fetch_one_window(kc, key, floor_start=floor, end=end,
                                     overlap_lookback=overlap)
            n_candles += len(cands)
            if cands:
                frames.append(candles_to_frame(key.symbol, key.scale, cands))
        except KiteException as exc:
            n_failures += 1
            log.warning("Skipping %s %s: %s", key.symbol, key.scale, exc)
        n_calls += 1
        if throttle > 0:
            time.sleep(throttle)
        if i % 50 == 0 or i == len(keys):
            elapsed = time.time() - t0
            console.print(
                f"  [{i}/{len(keys)}]  candles={n_candles:,}  failures={n_failures}  "
                f"elapsed={elapsed:.1f}s"
            )

    if not frames:
        console.print("[yellow]No new candles fetched. Nothing to stage.[/yellow]")
        sys.exit(0)

    refresh = pl.concat(frames, how="vertical")
    console.print(
        f"[bold]Fetched:[/bold] {refresh.height:,} rows × "
        f"{refresh['symbol'].n_unique()} symbols × "
        f"{refresh['scale'].n_unique()} scales  "
        f"(API calls: {n_calls}, failures: {n_failures})"
    )

    staging = f"{STAGING_PREFIX}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    stage_to_bq(bq, refresh, staging)
    diff = spot_check_overlap(bq, staging, tolerance=tolerance)

    tbl = Table(title="curated_ohlcv refresh diff")
    tbl.add_column("metric"); tbl.add_column("value")
    tbl.add_row("staged rows", f"{refresh.height:,}")
    tbl.add_row("overlap rows", f"{diff.get('n_overlap', 0):,}")
    tbl.add_row(f"close diffs > {tolerance:.1%}", f"{diff.get('n_diff', 0):,}")
    tbl.add_row("max |rel diff|", f"{diff.get('max_rel_diff') or 0:.4%}")
    tbl.add_row("p95 |rel diff|", f"{diff.get('p95_rel_diff') or 0:.4%}")
    console.print(tbl)
    if diff.get("n_diff", 0) > 0:
        console.print("[yellow]warning:[/yellow] some overlap rows differ — "
                      "likely an adjustment-treatment mismatch. Inspect before merging.")

    if not do_apply:
        console.print(
            f"[yellow]DRY RUN:[/yellow] staging at {bq.cfg.fq_table(staging)}. "
            "Re-run with --apply to MERGE."
        )
        return

    n = merge_into_raw_ohlcv(bq, staging)
    console.print(f"[green]MERGE complete:[/green] inserted {n:,} new rows into raw_ohlcv")
    if not keep_staging:
        drop_staging(bq, staging)


if __name__ == "__main__":
    main()
