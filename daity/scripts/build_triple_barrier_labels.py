"""Build triple-barrier labels for all symbols.

For each anchor ts (5m bar boundary), look at the next h_max minutes of 5m
bars and classify by which barrier the price hits FIRST:
  +1 LONG  : high[k] >= entry * (1 + target_pct)  before low[k] <= entry * (1 - stop_pct)
  -1 SHORT : low[k]  <= entry * (1 - target_pct)  before high[k] >= entry * (1 + stop_pct)
   0       : neither barrier hit within h_max minutes (timeout)

Writes parquet per symbol with the same schema as v1 tradeable labels so it
plugs into `CrossSectionalSampler` directly.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time, timezone
from pathlib import Path

import click
import numpy as np
import polars as pl
from rich.console import Console

UTC = timezone.utc
console = Console()

# NSE intraday hours (UTC = IST - 5:30): 03:45 UTC = 09:15 IST  → 10:00 UTC = 15:30 IST
NSE_OPEN_UTC = time(3, 45)
NSE_CLOSE_UTC = time(10, 0)


def build_for_symbol(
    feature_path: Path, out_path: Path,
    target_long_pct: float, stop_long_pct: float,
    target_short_pct: float, stop_short_pct: float,
    h_max_minutes: int,
) -> tuple[str, int, dict[int, int]]:
    """Build triple-barrier labels for one symbol. Returns (sym, n, dist)."""
    sym = feature_path.stem
    df = pl.read_parquet(feature_path).sort("ts").select(["ts", "high", "low", "close"])
    if len(df) == 0:
        return sym, 0, {}

    ts = df["ts"].to_numpy()
    highs = df["high"].to_numpy().astype(np.float64)
    lows  = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)

    # 5m bars → 60min = 12 future bars to scan.
    n_future_bars = h_max_minutes // 5

    labels: list[int] = []
    valids: list[bool] = []
    # Diagnostic fields
    first_long_target_min: list[float] = []
    first_short_target_min: list[float] = []
    max_high_pct: list[float] = []
    min_low_pct: list[float] = []

    # Vectorize: for each i, look at i+1 .. i+n_future_bars
    # Skip the last n_future_bars rows (no future window).
    n = len(df)
    for i in range(n):
        ts_i = ts[i]
        # Extract intraday hour to enforce NSE session
        ts_dt = ts_i.astype("datetime64[us]").astype(object)
        hh_mm = time(ts_dt.hour, ts_dt.minute)
        if not (NSE_OPEN_UTC <= hh_mm <= NSE_CLOSE_UTC):
            labels.append(0); valids.append(False)
            first_long_target_min.append(np.nan); first_short_target_min.append(np.nan)
            max_high_pct.append(np.nan); min_low_pct.append(np.nan)
            continue
        # Entry at close[i]
        entry = closes[i]
        if entry <= 0 or i + 1 >= n:
            labels.append(0); valids.append(False)
            first_long_target_min.append(np.nan); first_short_target_min.append(np.nan)
            max_high_pct.append(np.nan); min_low_pct.append(np.nan)
            continue
        # Window: bars i+1 .. i+n_future_bars (inclusive), but must be same trading day.
        end = min(i + n_future_bars + 1, n)
        # Same-day check: drop bars whose hour exceeds NSE_CLOSE_UTC OR that
        # belong to a different date.
        i_day = ts_dt.date()
        future_hi: list[float] = []
        future_lo: list[float] = []
        future_ts: list[datetime] = []
        for k in range(i + 1, end):
            ts_k_dt = ts[k].astype("datetime64[us]").astype(object)
            if ts_k_dt.date() != i_day:
                break
            hh_mm_k = time(ts_k_dt.hour, ts_k_dt.minute)
            if hh_mm_k > NSE_CLOSE_UTC:
                break
            future_hi.append(highs[k])
            future_lo.append(lows[k])
            future_ts.append(ts_k_dt)
        if len(future_hi) < 4:
            # Need at least 20-min lookahead to be valid.
            labels.append(0); valids.append(False)
            first_long_target_min.append(np.nan); first_short_target_min.append(np.nan)
            max_high_pct.append(np.nan); min_low_pct.append(np.nan)
            continue

        long_target = entry * (1 + target_long_pct)
        long_stop   = entry * (1 - stop_long_pct)
        short_target = entry * (1 - target_short_pct)
        short_stop  = entry * (1 + stop_short_pct)

        label = 0
        long_target_at = np.nan
        short_target_at = np.nan
        for j, (hi, lo, t_k) in enumerate(zip(future_hi, future_lo, future_ts)):
            mins_in = (j + 1) * 5  # 5-min bars
            long_hit_target = hi >= long_target
            long_hit_stop   = lo <= long_stop
            short_hit_target = lo <= short_target
            short_hit_stop  = hi >= short_stop

            if long_hit_target and not long_hit_stop:
                label = +1; long_target_at = mins_in; break
            if short_hit_target and not short_hit_stop:
                label = -1; short_target_at = mins_in; break
            # If both barriers hit in the same bar: ambiguous. Conservative:
            # assume the adverse side hit first (no label assignment).
            if long_hit_target and long_hit_stop:
                label = 0; break
            if short_hit_target and short_hit_stop:
                label = 0; break
            # Pure stop hit means neither LONG-target nor SHORT-target → continue
            # (don't label as -1 just because LONG-stop hit — that's a non-trade)
            if long_hit_stop and not short_hit_target:
                # LONG would have stopped; SHORT didn't trigger → null label.
                label = 0; break
            if short_hit_stop and not long_hit_target:
                label = 0; break
        # If loop ends with no barrier hit, label = 0 (timeout — neither side)

        labels.append(label); valids.append(True)
        first_long_target_min.append(long_target_at)
        first_short_target_min.append(short_target_at)
        max_high_pct.append((max(future_hi) - entry) / entry)
        min_low_pct.append((min(future_lo) - entry) / entry)

    # Compose output dataframe with same schema as v1 labels.
    out_df = pl.DataFrame({
        "ts": df["ts"],
        "t": df["ts"],                      # label-time same as anchor for first-touch
        "tradeable_label": np.array(labels, dtype=np.int8),
        "valid_label": np.array(valids, dtype=bool),
        "first_long_target_min": np.array(first_long_target_min, dtype=np.float32),
        "first_short_target_min": np.array(first_short_target_min, dtype=np.float32),
        "max_high_pct": np.array(max_high_pct, dtype=np.float32),
        "min_low_pct": np.array(min_low_pct, dtype=np.float32),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)

    arr = np.array(labels)
    dist = {-1: int((arr == -1).sum()), 0: int((arr == 0).sum()), +1: int((arr == 1).sum())}
    return sym, len(df), dist


@click.command()
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet/5m"), show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path), required=True,
              help="Where to write per-symbol triple-barrier parquets.")
@click.option("--target-long-pct",  type=float, default=0.0075, show_default=True)
@click.option("--stop-long-pct",    type=float, default=0.004,  show_default=True)
@click.option("--target-short-pct", type=float, default=0.0075, show_default=True)
@click.option("--stop-short-pct",   type=float, default=0.004,  show_default=True)
@click.option("--h-max-minutes",    type=int,   default=60,     show_default=True)
@click.option("--max-workers",      type=int,   default=8,      show_default=True)
def main(
    feature_root: Path, out_root: Path,
    target_long_pct: float, stop_long_pct: float,
    target_short_pct: float, stop_short_pct: float,
    h_max_minutes: int, max_workers: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(feature_root.glob("*.parquet"))
    console.print(
        f"[bold]Triple-barrier labels:[/bold]\n"
        f"  target_long={target_long_pct*100:.2f}%  stop_long={stop_long_pct*100:.2f}%\n"
        f"  target_short={target_short_pct*100:.2f}%  stop_short={stop_short_pct*100:.2f}%\n"
        f"  h_max={h_max_minutes}min\n"
        f"  symbols: {len(files)}",
    )
    grand_dist = {-1: 0, 0: 0, +1: 0}
    grand_n = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                build_for_symbol,
                f, out_root / f.name,
                target_long_pct, stop_long_pct,
                target_short_pct, stop_short_pct,
                h_max_minutes,
            )
            for f in files
        ]
        for fut in as_completed(futures):
            sym, n, dist = fut.result()
            grand_n += n
            for k in (-1, 0, +1):
                grand_dist[k] += dist.get(k, 0)
    total = max(1, sum(grand_dist.values()))
    console.print(
        f"\n[bold]Done.[/bold]  total_rows={grand_n}  written to {out_root}\n"
        f"  LONG (+1):  {grand_dist[+1]:>9,d}  ({grand_dist[+1]/total:.2%})\n"
        f"  NEUTRAL(0): {grand_dist[ 0]:>9,d}  ({grand_dist[ 0]/total:.2%})\n"
        f"  SHORT(-1):  {grand_dist[-1]:>9,d}  ({grand_dist[-1]/total:.2%})",
    )


if __name__ == "__main__":
    main()
