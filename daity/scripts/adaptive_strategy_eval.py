"""Adaptive strategy selection on top of pre-computed daily predictions.

For each test date D:
  1. Look at last `lookback_days` (D-N .. D-1) where realized labels exist.
  2. Sweep over (top_K, horizon, threshold) combos — compute Sharpe of each
     configuration on the lookback window.
  3. Pick the configuration with highest lookback Sharpe.
  4. Apply that configuration to day D's predictions → record D's actual P&L.

The model's predictions are FIXED (loaded from a predictions.parquet produced
by `online_walk_forward.py`). Only the STRATEGY hyperparameters adapt daily.

This is the cleanest form of online adaptation: no encoder drift risk, no
optimizer state to manage, no GPU needed. Pure pandas/polars.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class StrategyConfig:
    top_k: int
    horizon: str
    threshold: float    # only trade if top-K mean predicted log return > threshold

    def key(self) -> str:
        return f"K{self.top_k}_{self.horizon}_t{self.threshold:.4f}"


def compute_day_pnl(
    day_preds: pl.DataFrame,
    cfg: StrategyConfig,
    cost_bps: float,
) -> tuple[float | None, int]:
    """Run the strategy on one (date, horizon)-filtered slice.

    Returns (net_bps, n_alive). net_bps is None if not enough stocks or
    top-K mean prediction is below the threshold (no trade).
    """
    sub = day_preds.filter(pl.col("horizon") == cfg.horizon)
    n = sub.shape[0]
    if n < cfg.top_k:
        return None, n
    top = sub.sort("pred_lr", descending=True).head(cfg.top_k)
    pred_mean = float(top["pred_lr"].mean())
    if pred_mean < cfg.threshold:
        return None, n
    real_mean_lr = float(top["real_lr"].mean())
    net_bps = (np.exp(real_mean_lr) - 1.0) * 10_000.0 - cost_bps
    return net_bps, n


def evaluate_config_on_window(
    preds: pl.DataFrame,
    dates: list[date],
    cfg: StrategyConfig,
    cost_bps: float,
) -> tuple[float, int]:
    """Return (sharpe, n_traded) over the given dates with config `cfg`.

    Days where the strategy doesn't trade (threshold not met / not enough
    stocks) are EXCLUDED from the Sharpe computation (NOT counted as zero —
    a no-trade day is no exposure).
    """
    bps_list = []
    for d in dates:
        day_df = preds.filter(pl.col("date") == d)
        if day_df.shape[0] == 0:
            continue
        bps, _ = compute_day_pnl(day_df, cfg, cost_bps)
        if bps is None:
            continue
        bps_list.append(bps)
    if len(bps_list) < 3:
        return -1e9, 0
    arr = np.asarray(bps_list)
    mean = arr.mean()
    sd = arr.std() or 1e-9
    sharpe = mean / sd * np.sqrt(252)
    return sharpe, len(arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end",   type=str, required=True)
    ap.add_argument("--lookback-days", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--ks",        type=str, default="5,10,20")
    ap.add_argument("--horizons",  type=str,
                    default="intraday_60m,intraday_120m,intraday_180m,"
                            "to_close,overnight,next_day_1h,next_day_eod")
    ap.add_argument("--thresholds", type=str, default="0.0,0.001,0.003,0.005")
    ap.add_argument("--metric", type=str, default="sharpe", choices=["sharpe"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    preds = pl.read_parquet(args.predictions_path)
    print(f"loaded {preds.shape[0]} rows; date range "
          f"{preds['date'].min()} .. {preds['date'].max()}", flush=True)

    test_start = date.fromisoformat(args.test_start)
    test_end   = date.fromisoformat(args.test_end)
    ks         = [int(x) for x in args.ks.split(",")]
    horizons   = [x.strip() for x in args.horizons.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    configs = [
        StrategyConfig(top_k=k, horizon=h, threshold=t)
        for k in ks for h in horizons for t in thresholds
    ]
    print(f"sweep {len(configs)} configs × {(test_end-test_start).days} test days", flush=True)

    # All unique test dates available
    all_dates = sorted(
        preds.filter(
            (pl.col("date") >= test_start) & (pl.col("date") <= test_end),
        )["date"].unique().to_list(),
    )

    # For lookback we also need dates BEFORE test_start.
    lookback_start = test_start - timedelta(days=args.lookback_days * 3)
    candidate_dates = sorted(
        preds.filter(
            (pl.col("date") >= lookback_start) & (pl.col("date") <= test_end),
        )["date"].unique().to_list(),
    )

    rows = []
    for i, d in enumerate(all_dates):
        # Find the lookback window: last `lookback_days` dates strictly < d.
        prior = [pd for pd in candidate_dates if pd < d]
        if len(prior) < args.lookback_days:
            continue
        lookback = prior[-args.lookback_days:]

        # Tune: pick best config by lookback Sharpe.
        best_cfg = None
        best_sharpe = -1e9
        best_n = 0
        for cfg in configs:
            sharpe, n = evaluate_config_on_window(preds, lookback, cfg, args.cost_bps)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_cfg = cfg
                best_n = n
        if best_cfg is None:
            continue

        # Apply tuned config to day D.
        day_df = preds.filter(pl.col("date") == d)
        bps, n_alive = compute_day_pnl(day_df, best_cfg, args.cost_bps)
        rows.append({
            "date": d,
            "tuned_K":       best_cfg.top_k,
            "tuned_horizon": best_cfg.horizon,
            "tuned_threshold": best_cfg.threshold,
            "lookback_sharpe": best_sharpe,
            "lookback_n_trades": best_n,
            "traded":   bps is not None,
            "net_bps":  bps if bps is not None else 0.0,
            "n_alive":  n_alive,
        })
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(all_dates)} test days", flush=True)

    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    if df.shape[0] == 0:
        print("NO DAYS TRADED", flush=True)
        return 2

    print(f"\nsaved -> {args.out}", flush=True)
    n_total = df.shape[0]
    n_traded = int(df["traded"].sum())
    print(f"test days: {n_total}, traded: {n_traded} ({n_traded/n_total*100:.1f}%)", flush=True)

    if n_traded > 0:
        traded_df = df.filter(pl.col("traded"))
        nb = traded_df["net_bps"]
        mean = nb.mean()
        sd = nb.std() or 1e-9
        sharpe = mean / sd * np.sqrt(252)
        hit = (nb > 0).sum() / n_traded * 100
        # Also report assuming non-traded days = 0 bps (full-period view):
        all_bps = df["net_bps"]
        sharpe_all = all_bps.mean() / (all_bps.std() or 1e-9) * np.sqrt(252)
        print(f"\n=== Traded days ({n_traded}) ===")
        print(f"  mean bps/day: {mean:+.2f}")
        print(f"  hit rate:     {hit:.1f}%")
        print(f"  sharpe:       {sharpe:+.2f}")
        print(f"  total bps:    {nb.sum():+.1f}")
        print(f"\n=== Full window ({n_total} days, no-trade = 0 bps) ===")
        print(f"  mean bps/day: {all_bps.mean():+.2f}")
        print(f"  sharpe:       {sharpe_all:+.2f}")

        # Tuned config distribution
        print(f"\n=== Tuned config distribution ===")
        tc = df.group_by(["tuned_K", "tuned_horizon", "tuned_threshold"]).len().sort("len", descending=True)
        for row in tc.head(10).to_dicts():
            print(f"  K={row['tuned_K']:<3} {row['tuned_horizon']:<15} thresh={row['tuned_threshold']:.4f} : {row['len']} days")

    return 0


if __name__ == "__main__":
    sys.exit(main())
