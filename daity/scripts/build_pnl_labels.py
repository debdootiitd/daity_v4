"""Build realistic PnL labels for direct-PnL training.

For each anchor ts, simulate a LONG trade with realistic slippage and
triple-barrier exit:
  Entry price:  high(5m bar at t+entry_offset_min)   ← worst-case slippage
  Exit rule (in order of priority during t+entry_offset_min .. t+h_max_min):
    - high[k] >= entry * (1 + target_pct)  → +target_pct  (target first)
    - low[k]  <= entry * (1 - stop_pct)    → -stop_pct    (stop first)
    - else timeout: close at t+h_max_min   → (close - entry) / entry
  Realized PnL: above − cost_bps/10000

SHORT side is mirrored (entry = low(5m bar at t+entry_offset_min), target
hit = low <= entry × (1 - target_pct)).

Output: parquet per symbol with columns
  ts, t, tradeable_label (int8: +1 if LONG net pnl > 0 else 0 if SHORT net
    pnl > 0 ELSE 0 = NEUTRAL), realized_pnl_long (f32), realized_pnl_short
    (f32), valid_label (bool).
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


def _simulate_long(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   entry_px: float, target_pct: float, stop_pct: float,
                   first_idx: int, last_idx: int) -> float:
    """Returns realized return (LONG) before cost."""
    target_px = entry_px * (1 + target_pct)
    stop_px = entry_px * (1 - stop_pct)
    for k in range(first_idx, last_idx + 1):
        if highs[k] >= target_px and lows[k] <= stop_px:
            # Ambiguous within bar — conservative: stop first
            return -stop_pct
        if highs[k] >= target_px:
            return target_pct
        if lows[k] <= stop_px:
            return -stop_pct
    return (closes[last_idx] - entry_px) / entry_px


def _simulate_short(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    entry_px: float, target_pct: float, stop_pct: float,
                    first_idx: int, last_idx: int) -> float:
    """Returns realized return (SHORT) before cost. POSITIVE = win."""
    # SHORT: profit when price falls. target = entry × (1 - target_pct)
    target_px = entry_px * (1 - target_pct)
    stop_px = entry_px * (1 + stop_pct)
    for k in range(first_idx, last_idx + 1):
        if lows[k] <= target_px and highs[k] >= stop_px:
            return -stop_pct
        if lows[k] <= target_px:
            return target_pct
        if highs[k] >= stop_px:
            return -stop_pct
    return (entry_px - closes[last_idx]) / entry_px


def build_for_symbol(
    feature_path: Path, out_path: Path,
    target_pct: float, stop_pct: float,
    entry_offset_min: int, h_max_min: int,
    cost_bps: float,
) -> tuple[str, int, dict[int, int]]:
    sym = feature_path.stem
    df = pl.read_parquet(feature_path).sort("ts").select(["ts", "high", "low", "close"])
    if len(df) == 0:
        return sym, 0, {}

    ts = df["ts"].to_numpy()
    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)

    entry_offset_bars = entry_offset_min // 5
    h_max_bars = h_max_min // 5
    n = len(df)
    cost = cost_bps / 10000.0

    labels = np.zeros(n, dtype=np.int8)
    valids = np.zeros(n, dtype=bool)
    pnl_long = np.zeros(n, dtype=np.float32)
    pnl_short = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ts_i = ts[i]
        ts_dt = ts_i.astype("datetime64[us]").astype(object)
        hh_mm = time(ts_dt.hour, ts_dt.minute)
        if not (NSE_OPEN_UTC <= hh_mm <= NSE_CLOSE_UTC):
            continue
        # Entry bar at i + entry_offset_bars
        i_entry = i + entry_offset_bars
        i_end = i + entry_offset_bars + h_max_bars
        if i_end >= n:
            continue
        # Same trading day check
        i_day = ts_dt.date()
        # Find last valid bar within the window on the same day
        last_valid = i_entry
        for k in range(i_entry, min(i_end + 1, n)):
            ts_k_dt = ts[k].astype("datetime64[us]").astype(object)
            if ts_k_dt.date() != i_day:
                break
            hh_mm_k = time(ts_k_dt.hour, ts_k_dt.minute)
            if hh_mm_k > NSE_CLOSE_UTC:
                break
            last_valid = k
        if last_valid - i_entry < 3:
            continue   # need ≥15 min lookahead

        # LONG: entry = high of t+entry_offset_min bar (worst-case slippage)
        long_entry = highs[i_entry]
        if long_entry <= 0:
            continue
        long_raw = _simulate_long(
            highs, lows, closes, long_entry,
            target_pct, stop_pct, i_entry + 1, last_valid,
        )
        long_net = long_raw - cost

        # SHORT: entry = low of t+entry_offset_min bar (worst-case slippage)
        short_entry = lows[i_entry]
        if short_entry <= 0:
            continue
        short_raw = _simulate_short(
            highs, lows, closes, short_entry,
            target_pct, stop_pct, i_entry + 1, last_valid,
        )
        short_net = short_raw - cost

        pnl_long[i] = long_net
        pnl_short[i] = short_net
        valids[i] = True

        # Categorical label for backward compat: +1 if LONG hit target,
        # -1 if SHORT hit target, 0 otherwise.
        if abs(long_raw - target_pct) < 1e-9:
            labels[i] = +1
        elif abs(short_raw - target_pct) < 1e-9:
            labels[i] = -1
        else:
            labels[i] = 0

    out_df = pl.DataFrame({
        "ts": df["ts"],
        "t":  df["ts"],
        "tradeable_label":  labels,
        "valid_label":      valids,
        "realized_pnl_long":  pnl_long,
        "realized_pnl_short": pnl_short,
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
@click.option("--target-pct", type=float, default=0.01, show_default=True)
@click.option("--stop-pct", type=float, default=0.002, show_default=True)
@click.option("--entry-offset-min", type=int, default=5, show_default=True,
              help="Bars after anchor for entry — uses HIGH for LONG, LOW for SHORT.")
@click.option("--h-max-min", type=int, default=60, show_default=True)
@click.option("--cost-bps", type=float, default=19, show_default=True,
              help="Round-trip cost subtracted from realized return.")
@click.option("--max-workers", type=int, default=8, show_default=True)
def main(
    feature_root: Path, out_root: Path,
    target_pct: float, stop_pct: float,
    entry_offset_min: int, h_max_min: int,
    cost_bps: float, max_workers: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(feature_root.glob("*.parquet"))
    console.print(
        f"[bold]PnL labels (realistic slippage):[/bold]\n"
        f"  entry: high(+{entry_offset_min}m bar) for LONG, low(+{entry_offset_min}m) for SHORT\n"
        f"  exit:  triple-barrier target=+{target_pct*100:.2f}%/-{stop_pct*100:.2f}%, h={h_max_min}m\n"
        f"  cost:  {cost_bps:.0f} bps round-trip\n"
        f"  symbols: {len(files)}",
    )
    grand_dist = {-1: 0, 0: 0, +1: 0}
    grand_n = 0
    long_pos_count = 0
    short_pos_count = 0
    total_valid = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                build_for_symbol,
                f, out_root / f.name,
                target_pct, stop_pct,
                entry_offset_min, h_max_min, cost_bps,
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
        f"  LONG target-hit (+1):  {grand_dist[+1]:>9,d}  ({grand_dist[+1]/total:.2%})\n"
        f"  NEUTRAL/stop/timeout:  {grand_dist[ 0]:>9,d}  ({grand_dist[ 0]/total:.2%})\n"
        f"  SHORT target-hit (-1): {grand_dist[-1]:>9,d}  ({grand_dist[-1]/total:.2%})",
    )


if __name__ == "__main__":
    main()
