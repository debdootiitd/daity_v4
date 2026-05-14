"""Build path-conditioned 'smooth uptrend' labels.

For each anchor ts (5m bar boundary), look at the next h_max minutes of 5m
bars and classify by PATH (max + min within the window), NOT just first-touch.

LONG (+1):  max(highs in window) / entry  >= 1 + target_pct
        AND min(lows  in window) / entry  >= 1 - floor_pct
        (price reaches +target% AND never drops below -floor% from entry)
SHORT (-1): mirror

NEUTRAL (0): neither path condition satisfied.

Compared to triple-barrier first-touch labels, this is STRICTER:
- triple-barrier counts spike-then-stop trades; we filter them out
- This selects smooth-uptrend / smooth-downtrend setups only
- Base rate will be much lower (~1-2%) but the trades that DO trigger
  should have far better PnL stability.

Goal: reduce the "stop-first" rate observed in v1 backtest (62%).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time
from pathlib import Path

import click
import numpy as np
import polars as pl
from rich.console import Console

console = Console()

NSE_OPEN_UTC = time(3, 45)
NSE_CLOSE_UTC = time(10, 0)


def build_for_symbol(
    feature_path: Path, out_path: Path,
    target_pct: float, floor_pct: float, h_max_minutes: int,
) -> tuple[str, int, dict[int, int]]:
    sym = feature_path.stem
    df = pl.read_parquet(feature_path).sort("ts").select(["ts", "high", "low", "close"])
    if len(df) == 0:
        return sym, 0, {}

    ts = df["ts"].to_numpy()
    highs = df["high"].to_numpy().astype(np.float64)
    lows  = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)

    n_future_bars = h_max_minutes // 5
    n = len(df)

    labels = np.zeros(n, dtype=np.int8)
    valids = np.zeros(n, dtype=bool)
    max_hi_pct = np.full(n, np.nan, dtype=np.float32)
    min_lo_pct = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        ts_i = ts[i]
        ts_dt = ts_i.astype("datetime64[us]").astype(object)
        hh_mm = time(ts_dt.hour, ts_dt.minute)
        if not (NSE_OPEN_UTC <= hh_mm <= NSE_CLOSE_UTC):
            continue
        entry = closes[i]
        if entry <= 0 or i + 1 >= n:
            continue
        end = min(i + n_future_bars + 1, n)
        i_day = ts_dt.date()
        win_hi: list[float] = []
        win_lo: list[float] = []
        for k in range(i + 1, end):
            ts_k_dt = ts[k].astype("datetime64[us]").astype(object)
            if ts_k_dt.date() != i_day:
                break
            hh_mm_k = time(ts_k_dt.hour, ts_k_dt.minute)
            if hh_mm_k > NSE_CLOSE_UTC:
                break
            win_hi.append(highs[k])
            win_lo.append(lows[k])
        if len(win_hi) < 4:
            continue

        win_hi_arr = np.array(win_hi)
        win_lo_arr = np.array(win_lo)
        max_h = win_hi_arr.max() / entry - 1
        min_l = win_lo_arr.min() / entry - 1
        max_hi_pct[i] = max_h
        min_lo_pct[i] = min_l
        valids[i] = True

        long_ok = (max_h >= target_pct) and (min_l >= -floor_pct)
        short_ok = (-min_l >= target_pct) and (max_h <= floor_pct)
        if long_ok and short_ok:
            labels[i] = 0   # can't be both — ambiguous
        elif long_ok:
            labels[i] = +1
        elif short_ok:
            labels[i] = -1
        else:
            labels[i] = 0

    out_df = pl.DataFrame({
        "ts": df["ts"],
        "t":  df["ts"],
        "tradeable_label": labels,
        "valid_label": valids,
        "max_high_pct": max_hi_pct,
        "min_low_pct": min_lo_pct,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)

    dist = {-1: int((labels == -1).sum()),
            0:  int((labels ==  0).sum()),
            +1: int((labels ==  1).sum())}
    return sym, n, dist


@click.command()
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet/5m"), show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path), required=True)
@click.option("--target-pct", type=float, default=0.01,  show_default=True,
              help="Min favorable excursion required (e.g. 0.01 = 1%).")
@click.option("--floor-pct", type=float, default=0.002, show_default=True,
              help="Max ADVERSE excursion tolerated (e.g. 0.002 = -0.2%).")
@click.option("--h-max-minutes", type=int, default=60, show_default=True)
@click.option("--max-workers",   type=int, default=8,  show_default=True)
def main(
    feature_root: Path, out_root: Path,
    target_pct: float, floor_pct: float,
    h_max_minutes: int, max_workers: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(feature_root.glob("*.parquet"))
    console.print(
        f"[bold]Smooth-uptrend labels:[/bold]\n"
        f"  target=+{target_pct*100:.2f}%  floor=-{floor_pct*100:.2f}%  h_max={h_max_minutes}min\n"
        f"  symbols: {len(files)}",
    )
    grand_dist = {-1: 0, 0: 0, +1: 0}
    grand_n = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                build_for_symbol,
                f, out_root / f.name,
                target_pct, floor_pct, h_max_minutes,
            )
            for f in files
        ]
        for fut in as_completed(futures):
            _sym, n, dist = fut.result()
            grand_n += n
            for k in (-1, 0, +1):
                grand_dist[k] += dist.get(k, 0)
    total = max(1, sum(grand_dist.values()))
    console.print(
        f"\n[bold]Done.[/bold]  rows={grand_n}\n"
        f"  LONG (+1):  {grand_dist[+1]:>9,d}  ({grand_dist[+1]/total:.2%})\n"
        f"  NEUTRAL(0): {grand_dist[ 0]:>9,d}  ({grand_dist[ 0]/total:.2%})\n"
        f"  SHORT(-1):  {grand_dist[-1]:>9,d}  ({grand_dist[-1]/total:.2%})",
    )


if __name__ == "__main__":
    main()
