"""Stop-loss-adjusted P&L simulator.

Reads a per-day strategy output (from `adaptive_calibrated_strategy.py`)
that contains a `chosen_stocks` column. For each pick, walks the 5m bar
path from entry anchor to horizon endpoint, exiting early if the bar low
hits a stop-loss level. Re-computes daily P&L and cumulative sleeve.

Output: a new parquet with the same schema plus `today_pnl_bps_stop` and
`exit_reason` (target/stop/forced) per pick row.

Usage:
  python -m daity.scripts.stop_loss_simulator \
      --strategy-parquet reports/v12_strategy/gbm_k1_armB_adagrad_N5.parquet \
      --feature-root data/features_parquet \
      --anchor-ist 10:15 \
      --horizon next_day_eod \
      --stop-bps 200 \
      --cost-bps 15 \
      --out reports/v12_strategy/gbm_k1_armB_adagrad_N5_stop200.parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


# Horizon → (exit_ist_time, trading_days_forward).
# E.g., next_day_eod: exit at 15:30 IST of next trading day.
HORIZON_EXIT = {
    "intraday_30m":  ("anchor+30m", 0),
    "intraday_60m":  ("anchor+60m", 0),
    "intraday_120m": ("anchor+120m", 0),
    "intraday_180m": ("anchor+180m", 0),
    "to_close":      ("15:30", 0),
    "overnight":     ("09:15", 1),
    "next_day_1h":   ("10:15", 1),
    "next_day_eod":  ("15:30", 1),
    "day_plus_3":    ("15:30", 3),
    "day_plus_5":    ("15:30", 5),
}


def _ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-parquet", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--cache-root",   type=Path, default=Path("data/cache"))
    ap.add_argument("--anchor-ist", type=str, default="10:15")
    ap.add_argument("--horizon", type=str, default="next_day_eod")
    ap.add_argument("--stop-bps", type=float, default=200.0,
                    help="Stop-loss level in bps from entry; 0 disables (hold to expiry).")
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def _next_trading_day(cal, d: date, n_forward: int) -> date | None:
    """Return the n_forward-th trading day strictly after d (1-indexed)."""
    if n_forward == 0:
        return d
    cur = d
    for _ in range(n_forward):
        cur = cur + timedelta(days=1)
        while not cal.is_trading_day(cur):
            cur = cur + timedelta(days=1)
            if (cur - d).days > 30:
                return None
    return cur


def _load_5m_bars(feature_root: Path, sym: str,
                  ts_lo: datetime, ts_hi: datetime) -> pl.DataFrame | None:
    """Load 5m bars for sym in the window [ts_lo, ts_hi]."""
    path = feature_root / "5m" / f"{sym}.parquet"
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path).filter(
            (pl.col("ts") >= ts_lo) & (pl.col("ts") <= ts_hi)
        ).sort("ts")
        return df
    except Exception:
        return None


def _simulate_one(bars: pl.DataFrame, entry_price: float,
                  stop_level_pct: float) -> tuple[float, str]:
    """Walk forward through bars. Returns (exit_price, reason).

    stop_level_pct: e.g. 0.98 means exit if low <= entry * 0.98.
    Reason: 'target' (held to last bar), 'stop' (stop hit), 'no_bars'.
    """
    if bars.height == 0:
        return entry_price, "no_bars"
    stop_px = entry_price * stop_level_pct
    lows = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    # Find first bar where low <= stop
    if stop_level_pct < 1.0:
        hit = np.where(lows <= stop_px)[0]
        if hit.size > 0:
            return float(stop_px), "stop"
    return float(closes[-1]), "target"


def main() -> int:
    args = parse_args()

    from daity.data.calendar_nse import NSECalendar
    cal = NSECalendar.from_cache(args.cache_root)

    df = pl.read_parquet(args.strategy_parquet).sort("test_date")
    if "chosen_stocks" not in df.columns:
        print(f"ERROR: {args.strategy_parquet} has no 'chosen_stocks' column. "
              f"Re-run adaptive_calibrated_strategy.py after the patch.",
              file=sys.stderr)
        return 1

    h, m = (int(x) for x in args.anchor_ist.split(":"))
    anchor_t = dtime(h, m)
    horizon = args.horizon
    if horizon not in HORIZON_EXIT:
        print(f"ERROR: unknown horizon '{horizon}'. Known: {list(HORIZON_EXIT)}",
              file=sys.stderr)
        return 1
    exit_t_str, n_fwd = HORIZON_EXIT[horizon]

    stop_pct = 1.0 - args.stop_bps / 10000.0
    print(f"stop_bps={args.stop_bps}  stop_pct={stop_pct:.4f}  horizon={horizon}  "
          f"exit_t={exit_t_str}  n_fwd_td={n_fwd}", flush=True)

    rows_out = []
    n_traded = 0
    n_stop = 0
    n_target = 0
    n_nobars = 0

    for r in df.iter_rows(named=True):
        d = r["test_date"]
        bps_orig = r["today_pnl_bps"]
        nt = r["n_today_trades"]
        syms = r.get("chosen_stocks") or ""
        if nt == 0 or not syms:
            rows_out.append({**r, "today_pnl_bps_stop": 0.0, "exit_reason": "sitout"})
            continue
        syms_list = [s for s in syms.split(",") if s]
        anchor_utc = _ist_to_utc(d, anchor_t)
        # Compute exit timestamp
        if exit_t_str.startswith("anchor+"):
            minutes = int(exit_t_str.removeprefix("anchor+").removesuffix("m"))
            exit_utc = anchor_utc + timedelta(minutes=minutes)
        else:
            exit_d = _next_trading_day(cal, d, n_fwd)
            if exit_d is None:
                rows_out.append({**r, "today_pnl_bps_stop": bps_orig, "exit_reason": "no_exit_day"})
                continue
            eh, em = (int(x) for x in exit_t_str.split(":"))
            exit_utc = _ist_to_utc(exit_d, dtime(eh, em))

        # For each chosen stock, walk 5m bars from anchor to exit
        sym_returns = []
        sym_reasons = []
        for sym in syms_list:
            bars = _load_5m_bars(args.feature_root, sym, anchor_utc, exit_utc)
            if bars is None or bars.height == 0:
                # Fall back to the original log return for this stock; we don't have it
                # broken out, so use the average for the day (conservative).
                sym_returns.append((np.exp(np.log(1 + bps_orig/10000.0 + args.cost_bps/10000.0)) - 1.0))
                sym_reasons.append("no_bars")
                n_nobars += 1
                continue
            # Entry price: open of first bar
            entry_px = float(bars["open"][0])
            exit_px, reason = _simulate_one(bars, entry_px, stop_pct)
            sym_returns.append((exit_px - entry_px) / entry_px)
            sym_reasons.append(reason)

        # Average return across K picks (equal-weighted)
        avg_ret = float(np.mean(sym_returns))
        bps_stop = avg_ret * 10000.0 - args.cost_bps
        # Tag exit reason as 'stop' if any pick stopped out
        any_stop = any(r == "stop" for r in sym_reasons)
        all_nobars = all(r == "no_bars" for r in sym_reasons)
        reason = "stop" if any_stop else ("no_bars" if all_nobars else "target")
        if any_stop:
            n_stop += 1
        elif all_nobars:
            pass  # already counted
        else:
            n_target += 1
        n_traded += 1
        rows_out.append({**r, "today_pnl_bps_stop": float(bps_stop), "exit_reason": reason})

    out_df = pl.DataFrame(rows_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(args.out)

    # Summary
    bps_orig = out_df["today_pnl_bps"].to_numpy()
    bps_stop = out_df["today_pnl_bps_stop"].to_numpy()
    realized_offset = {"to_close": 0, "next_day_eod": 1, "next_day_1h": 1,
                       "day_plus_3": 3, "day_plus_5": 5}.get(horizon, 1)
    hold = max(1, realized_offset)
    ann = float(np.sqrt(252.0 / hold))

    def stats(bps, label):
        n = len(bps); m = float(bps.mean()); s = float(bps.std()) or 1e-9
        sh = m / s * ann
        sleeve = float(np.prod(1 + bps / 10000.0 / hold) - 1) * 100
        nt = int((bps != 0).sum())
        hit = float((bps[bps != 0] > 0).sum() / max(nt, 1) * 100)
        print(f"  {label}: n={n} traded={nt}({100*nt/n:.0f}%) "
              f"mean_all={m:+7.2f} Sh={sh:+5.2f} sleeve={sleeve:+.2f}% hit={hit:.0f}%",
              flush=True)
    print(f"\n=== Summary ({horizon}, stop_bps={args.stop_bps}) ===", flush=True)
    stats(bps_orig, "orig (hold-to-expiry)")
    stats(bps_stop, "stop-loss adjusted   ")
    print(f"  exits: stop={n_stop}, target={n_target}, no_bars={n_nobars}, traded_days={n_traded}",
          flush=True)
    print(f"  → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
