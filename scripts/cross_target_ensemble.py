"""Combine predictions from models trained on different label horizons (1d, 10d).

Different horizons produce pred_lr on different scales. To combine fairly, we
z-score each model's pred_lr per day (rank-normalised within the day's universe),
then average. The result is a unit-variance per-day cross-sectional signal that
can be used directly in rolling-entry threshold and chunky-rebal ranking.

Threshold interpretation: z-score units. e.g., threshold=0.5 means "average
z-score across constituent models must be > 0.5 standard deviations above the
cross-sectional mean today."
"""
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--label", default="z-score ensemble")
args = ap.parse_args()

dfs = []
for p in args.inputs:
    d = pl.read_parquet(p).select(["date", "stock", "pred_lr", "real_lr"])
    dfs.append(d)
    print(f"  loaded {p}: {d.shape[0]} rows")

# Z-score pred_lr per (date) within each input
def zscore_per_day(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
    return df.with_columns([
        ((pl.col("pred_lr") - pl.col("pred_lr").mean().over("date"))
         / (pl.col("pred_lr").std(ddof=0).over("date") + 1e-9))
        .alias(f"pred_z_{suffix}")
    ]).select(["date", "stock", f"pred_z_{suffix}"])

zs = []
for i, d in enumerate(dfs):
    zs.append(zscore_per_day(d, str(i)))

# Inner-join on (date, stock) — take only rows present in all
base = dfs[0].select(["date", "stock", "real_lr"])
for i, z in enumerate(zs):
    base = base.join(z, on=["date", "stock"], how="inner")
pred_cols = [f"pred_z_{i}" for i in range(len(zs))]
base = base.with_columns(pl.mean_horizontal(pred_cols).alias("pred_lr")).select(["date", "stock", "pred_lr", "real_lr"])

# Per-day IC
ics = []
all_dates = sorted(base["date"].unique().to_list())
for d in all_dates:
    sub = base.filter(pl.col("date") == d)
    if sub.shape[0] < 20: continue
    rho, _ = spearmanr(sub["pred_lr"].to_numpy(), sub["real_lr"].to_numpy())
    if not np.isnan(rho): ics.append(rho)
ic_mean = float(np.mean(ics)) if ics else 0
ic_ir = float(ic_mean / (np.std(ics) + 1e-9) * np.sqrt(len(ics))) if ics else 0
print(f"\n=== {args.label} (n={len(dfs)} models) ===")
print(f"  rows: {base.shape[0]:,}  days: {len(all_dates)}")
print(f"  IC mean: {ic_mean:+.4f}  IR: {ic_ir:+.2f}")
print(f"  pred_lr is z-score-mean (unit-variance per day)")

base.write_parquet(args.output)
print(f"  wrote {args.output}")
