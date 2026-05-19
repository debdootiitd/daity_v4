"""Rolling-anchor backtest scorer.

For hold=K days, instead of just running one start-offset (D, D+K, D+2K, ...),
run ALL K offsets:
  offset 0: D, D+K, D+2K, ...
  offset 1: D+1, D+K+1, D+2K+1, ...
  ...
  offset K-1: D+K-1, D+2K-1, ...

For each offset, compute the standard top-K backtest metrics (Sharpe, sleeve, hit).
Aggregate across offsets to produce robust mean/median/CI.

This multiplies the effective sample size from N/K to N rebalances (still
non-overlapping per offset, but pooled across offsets — closer to a true
out-of-sample estimate).

Usage:
  python score_rolling_anchor.py <predictions.parquet> <label> [--K 5,10,20] [--hold 5,10,20]
"""
import sys
import argparse
import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("predictions")
ap.add_argument("label")
ap.add_argument("--ks", default="5,10,20")
ap.add_argument("--holds", default="5,10,20")
ap.add_argument("--cost-bps", type=float, default=30.0)
args = ap.parse_args()

KS = [int(x) for x in args.ks.split(",")]
HOLDS = [int(x) for x in args.holds.split(",")]

df = pl.read_parquet(args.predictions).sort(["date", "stock"])
all_dates = sorted(df["date"].unique().to_list())
n = len(all_dates)
print(f"=== {args.label} ===")
print(f"rows: {df.shape[0]:,} days: {n} date range: {all_dates[0]} → {all_dates[-1]}")

# Pre-cache per-stock real_lr series
psd = {}
for s in df["stock"].unique():
    sub = df.filter(pl.col("stock") == s).sort("date")
    psd[s] = {"dates": sub["date"].to_list(), "real_lr": sub["real_lr"].to_numpy()}


def one_offset_backtest(K: int, hold: int, offset: int):
    """Pick top-K every `hold` days starting at all_dates[offset]."""
    daily_lr = []
    di = offset
    while di + hold <= n - 1:
        d_rebal = all_dates[di]
        sub = df.filter(pl.col("date") == d_rebal).sort("pred_lr", descending=True).head(K)
        if sub.shape[0] < K:
            di += hold
            continue
        picked = sub["stock"].to_list()
        stock_rets = []
        for s in picked:
            p = psd[s]
            if d_rebal not in p["dates"]:
                stock_rets.append(0.0)
                continue
            s_idx = p["dates"].index(d_rebal)
            end = min(s_idx + hold, len(p["dates"]))
            stock_rets.append(float(p["real_lr"][s_idx:end].sum()))
        port = float(np.log(np.exp(np.array(stock_rets)).mean()))
        daily_lr.append(port - args.cost_bps / 10000.0)
        di += hold
    return np.array(daily_lr)


print(f"\n{'hold':>5} {'K':>3} {'offset':>7} {'n':>4} {'sharpe':>7} {'sleeve%':>8} {'hit':>5}")
print("-" * 50)
agg = {}
for hold in HOLDS:
    for K in KS:
        sharpes, sleeves, hits = [], [], []
        for offset in range(hold):
            arr = one_offset_backtest(K, hold, offset)
            if len(arr) < 2:
                continue
            mean = arr.mean()
            std = arr.std(ddof=1)
            sh = (mean / (std + 1e-12)) * np.sqrt(250 / hold) if std > 0 else 0
            sl = (np.exp(arr.sum()) - 1) * 100
            ht = (arr > 0).mean()
            sharpes.append(sh)
            sleeves.append(sl)
            hits.append(ht)
        if not sharpes:
            continue
        sharpes = np.array(sharpes)
        sleeves = np.array(sleeves)
        hits = np.array(hits)
        agg[(hold, K)] = {
            "n_offsets": len(sharpes),
            "sharpe_mean": sharpes.mean(),
            "sharpe_median": np.median(sharpes),
            "sharpe_std": sharpes.std(ddof=1),
            "sharpe_min": sharpes.min(),
            "sharpe_max": sharpes.max(),
            "sleeve_mean": sleeves.mean(),
            "sleeve_std": sleeves.std(ddof=1),
            "hit_mean": hits.mean(),
        }
        # Print every offset for hold=20 K=5/10 to inspect
        if hold in (10, 20) and K in (5, 10):
            for i, (sh, sl, ht) in enumerate(zip(sharpes, sleeves, hits)):
                print(f"{hold:>5} {K:>3} {i:>7} - {sh:+7.2f} {sl:+8.2f} {ht:.2f}")

print("\n=== Aggregate (mean across offsets) ===")
print(f"{'hold':>5} {'K':>3} {'offs':>5} {'sharpe_mean':>11} {'std':>5} {'min':>6} {'max':>6} {'sleeve_mean':>11} {'hit':>5}")
for (hold, K), v in sorted(agg.items(), key=lambda x: -x[1]["sharpe_mean"]):
    print(f"{hold:>5} {K:>3} {v['n_offsets']:>5} {v['sharpe_mean']:>+11.2f} {v['sharpe_std']:>5.2f} "
          f"{v['sharpe_min']:>+6.2f} {v['sharpe_max']:>+6.2f} {v['sleeve_mean']:>+11.2f} {v['hit_mean']:.2f}")

# Best config by mean sharpe
best_key = max(agg.keys(), key=lambda k: agg[k]["sharpe_mean"])
v = agg[best_key]
print(f"\nBest (mean Sharpe): hold={best_key[0]} K={best_key[1]} →")
print(f"  Sharpe across offsets: mean={v['sharpe_mean']:+.2f} std={v['sharpe_std']:.2f} "
      f"min={v['sharpe_min']:+.2f} max={v['sharpe_max']:+.2f}")
print(f"  Sleeve mean: {v['sleeve_mean']:+.2f}% (std {v['sleeve_std']:.2f}%)")
print(f"  Hit mean: {v['hit_mean']:.2f}")

# Compare to naive single-offset estimate
print("\nSingle-offset (offset=0) vs rolling-anchor mean:")
for (hold, K), v in sorted(agg.items(), key=lambda x: -x[1]["sharpe_mean"])[:5]:
    arr = one_offset_backtest(K, hold, 0)
    sh0 = (arr.mean() / (arr.std(ddof=1) + 1e-12)) * np.sqrt(250 / hold) if arr.std() > 0 else 0
    print(f"  hold={hold} K={K}: offset0 Sharpe={sh0:+.2f}  vs rolling mean={v['sharpe_mean']:+.2f} "
          f"(spread {sh0 - v['sharpe_mean']:+.2f})")
