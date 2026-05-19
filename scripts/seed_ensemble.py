"""Average pred_lr across multiple seed runs of the same config.

Takes a list of predictions.parquet files, computes the average pred_lr
per (date, stock), and writes a new predictions parquet with the ensemble.

Then scores via the standard scorer and the rolling-anchor scorer.
"""
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True,
                help="List of predictions.parquet files (same OOS dates, different seeds)")
ap.add_argument("--output", required=True)
ap.add_argument("--label", default="ensemble")
ap.add_argument("--cost-bps", type=float, default=30.0)
args = ap.parse_args()

dfs = []
for p in args.inputs:
    d = pl.read_parquet(p).select(["date", "stock", "pred_lr", "real_lr"])
    dfs.append(d)
    print(f"  loaded {p}: {d.shape[0]} rows")

# Join all on (date, stock); compute mean of pred_lr
base = dfs[0].select(["date", "stock", "real_lr"])
for i, d in enumerate(dfs):
    base = base.join(d.select(["date", "stock", pl.col("pred_lr").alias(f"pred_lr_{i}")]),
                     on=["date", "stock"], how="inner")
pred_cols = [f"pred_lr_{i}" for i in range(len(dfs))]
base = base.with_columns(
    pl.mean_horizontal(pred_cols).alias("pred_lr")
).select(["date", "stock", "pred_lr", "real_lr"])

# Per-day IC
ics = []
all_dates = sorted(base["date"].unique().to_list())
for d in all_dates:
    sub = base.filter(pl.col("date") == d)
    if sub.shape[0] < 20:
        continue
    rho, _ = spearmanr(sub["pred_lr"].to_numpy(), sub["real_lr"].to_numpy())
    if not np.isnan(rho):
        ics.append(rho)
ic_mean = float(np.mean(ics)) if ics else 0
ic_ir = float(ic_mean / (np.std(ics) + 1e-9) * np.sqrt(len(ics))) if ics else 0
print(f"\n=== {args.label} ensemble of {len(dfs)} seeds ===")
print(f"  rows: {base.shape[0]:,}  days: {len(all_dates)}")
print(f"  IC mean: {ic_mean:+.4f}  IR: {ic_ir:+.2f}")

# Backtest sweep K∈{5,10,20} hold∈{5,10,20}
psd = {}
for s in base["stock"].unique():
    sub = base.filter(pl.col("stock") == s).sort("date")
    psd[s] = {"dates": sub["date"].to_list(), "real_lr": sub["real_lr"].to_numpy()}

n_dates = len(all_dates)
best = {"sharpe": -1e9}
for hold in [5, 10, 20]:
    for K in [5, 10, 20]:
        daily_lr = []
        di = 0
        while di + hold <= n_dates - 1:
            d_rebal = all_dates[di]
            sub = base.filter(pl.col("date") == d_rebal).sort("pred_lr", descending=True).head(K)
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
        if not daily_lr:
            continue
        arr = np.array(daily_lr)
        std = arr.std(ddof=1)
        sh = (arr.mean() / (std + 1e-12)) * np.sqrt(250 / hold) if std > 0 else 0
        sl = (np.exp(arr.sum()) - 1) * 100
        ht = (arr > 0).mean()
        if sh > best["sharpe"]:
            best = {"sharpe": sh, "sleeve": sl, "hit": ht, "hold": hold, "K": K}
print(f"  Best (single-offset): hold={best['hold']} K={best['K']} → Sharpe={best['sharpe']:+.2f} sleeve={best['sleeve']:+.2f}% hit={best['hit']:.2f}")

# Write ensemble parquet
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
base.write_parquet(out_path)
print(f"  wrote {out_path}")
