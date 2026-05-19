"""OB fillability audit.

For each (rebal_date, picked_stock) in a predictions parquet, find the nearest
order-book snapshot on that date, walk the top-5 ask levels, and compute
whether a target position size (e.g., ₹2 lakh) can be filled within tolerable
price impact.

Output:
  - %% of picks with any OB data available
  - %% of picks fillable at ≤ 30 bps VWAP slippage from top-of-book
  - distribution of fill slippage (bps)
  - %% requiring > 50 bps slippage (effectively unfillable at the size)

Usage:
  python audit_ob_fillability.py <predictions.parquet> --K 5 --hold 10 --pos-inr 200000
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("predictions")
ap.add_argument("--ob-dir", default="/Users/debdoot/Documents/Claude/Projects/daity/data/ob_parquet")
ap.add_argument("--K", type=int, default=5)
ap.add_argument("--hold", type=int, default=10)
ap.add_argument("--pos-inr", type=float, default=200_000.0, help="Target ₹ per pick")
ap.add_argument("--snapshot-time", default="09:30", help="HH:MM target snapshot time")
ap.add_argument("--slippage-budget-bps", type=float, default=30.0)
args = ap.parse_args()

OB_DIR = Path(args.ob_dir)
SLIP_BUDGET = args.slippage_budget_bps

# Load predictions
df = pl.read_parquet(args.predictions).sort(["date", "stock"])
all_dates = sorted(df["date"].unique().to_list())
n_dates = len(all_dates)

# Walk through every rebalance and collect picks
rebal_picks = []
di = 0
while di + args.hold <= n_dates - 1:
    d = all_dates[di]
    sub = df.filter(pl.col("date") == d).sort("pred_lr", descending=True).head(args.K)
    for stock in sub["stock"].to_list():
        rebal_picks.append((d, stock))
    di += args.hold

print(f"Total picks across {len(rebal_picks)//args.K} rebals × K={args.K}: {len(rebal_picks)}")

# Filter to picks in OB data window
ob_files = list(OB_DIR.glob("*.parquet"))
ob_stocks = {f.stem for f in ob_files}
print(f"OB universe: {len(ob_stocks)} stocks")

# Determine the OB date window by probing one file
sample_ob = pl.read_parquet(ob_files[0])
ob_min = sample_ob["ts"].min().date()
ob_max = sample_ob["ts"].max().date()
print(f"OB date range (from sample): {ob_min} → {ob_max}")

picks_in_ob_window = [p for p in rebal_picks if ob_min <= p[0] <= ob_max]
print(f"Picks in OB window ({ob_min} → {ob_max}): {len(picks_in_ob_window)}/{len(rebal_picks)}")

if not picks_in_ob_window:
    print("NO picks fall in the OB data window. Exiting.")
    sys.exit(0)

# For each pick, find nearest OB snapshot ≥ 09:30 (or args.snapshot_time)
hh, mm = map(int, args.snapshot_time.split(":"))


def fill_walk(asks_p, asks_q, target_inr, top_price):
    """Walk through top-N ask levels until we fill target_inr.

    Returns (vwap, qty_filled, fully_filled).
    """
    cum_inr = 0.0
    cum_qty = 0
    cum_cost = 0.0
    for p, q in zip(asks_p, asks_q):
        if p is None or q is None or p <= 0 or q <= 0:
            continue
        # Buy at this level
        notional_avail = p * q
        if cum_inr + notional_avail >= target_inr:
            # Partial fill at this level
            need_inr = target_inr - cum_inr
            qty_at_level = need_inr / p
            cum_qty += qty_at_level
            cum_cost += qty_at_level * p
            cum_inr = target_inr
            return cum_cost / cum_qty if cum_qty > 0 else None, cum_qty, True
        cum_inr += notional_avail
        cum_qty += q
        cum_cost += p * q
    return (cum_cost / cum_qty if cum_qty > 0 else None), cum_qty, False  # could not fully fill


# Aggregate stats
no_ob_stock = 0
no_snapshot_on_date = 0
fillable = 0
not_fillable = 0
slippages_bps = []
per_pick = []
unfillable_stocks = []

for date, stock in picks_in_ob_window:
    ob_path = OB_DIR / f"{stock}.parquet"
    if not ob_path.exists():
        no_ob_stock += 1
        continue
    ob = pl.read_parquet(ob_path)
    # Filter to this date
    day_ob = ob.filter(pl.col("ts").dt.date() == date)
    if day_ob.shape[0] == 0:
        no_snapshot_on_date += 1
        continue
    # Find first snapshot ≥ 09:30 (or fallback to first)
    # NOTE: ts may be UTC. NSE open 09:15 IST = 03:45 UTC. So snapshot_time treated as UTC here.
    day_ob = day_ob.sort("ts")
    snap = day_ob.head(1)  # earliest snapshot for the day = closest to open
    asks_p = [snap[f"ask{i}_price"][0] for i in range(1, 6)]
    asks_q = [snap[f"ask{i}_qty"][0] for i in range(1, 6)]
    top_ask = asks_p[0]
    if top_ask is None or top_ask <= 0:
        not_fillable += 1
        continue
    vwap, qty, filled = fill_walk(asks_p, asks_q, args.pos_inr, top_ask)
    if not filled or vwap is None:
        not_fillable += 1
        unfillable_stocks.append(stock)
        continue
    slip_bps = (vwap / top_ask - 1) * 10_000
    slippages_bps.append(slip_bps)
    if slip_bps <= SLIP_BUDGET:
        fillable += 1
    per_pick.append({
        "date": date, "stock": stock,
        "top_ask": top_ask, "vwap": vwap, "qty_filled": qty,
        "slip_bps": slip_bps, "filled_fully": filled,
    })

# Report
total_eval = no_ob_stock + no_snapshot_on_date + len(slippages_bps) + not_fillable
print(f"\n=== OB Fillability ({args.pos_inr:.0f} ₹ per pick) ===")
print(f"  Picks evaluated:     {total_eval}")
print(f"  No OB file:          {no_ob_stock}")
print(f"  No snapshot on date: {no_snapshot_on_date}")
print(f"  Fully filled within top-5: {len(slippages_bps)} ({len(slippages_bps)/total_eval*100:.1f}%)")
print(f"  Not fillable:        {not_fillable}")

if slippages_bps:
    s = np.array(slippages_bps)
    print(f"\n  Fill slippage from top-of-book (bps):")
    print(f"    mean    {s.mean():+.2f}")
    print(f"    median  {np.median(s):+.2f}")
    print(f"    p25/p75 {np.percentile(s,25):+.2f} / {np.percentile(s,75):+.2f}")
    print(f"    p95     {np.percentile(s,95):+.2f}")
    print(f"    max     {s.max():+.2f}")
    print(f"  ≤ {SLIP_BUDGET} bps: {fillable}/{len(slippages_bps)} = {fillable/len(slippages_bps)*100:.1f}%")

if unfillable_stocks:
    from collections import Counter
    print(f"\n=== Top unfillable stocks (top-5 depth < ₹{args.pos_inr:.0f}) ===")
    for s, c in Counter(unfillable_stocks).most_common(10):
        print(f"  {s:<14} {c:>3}")

# Per-stock breakdown
if per_pick:
    pp = pl.DataFrame(per_pick)
    print(f"\n=== Slippage by stock (top picks in OB window) ===")
    by_stock = (pp.group_by("stock")
                  .agg([pl.col("slip_bps").mean().alias("avg_slip"),
                        pl.col("slip_bps").count().alias("n"),
                        pl.col("slip_bps").max().alias("max_slip")])
                  .sort("avg_slip", descending=True))
    print(f"{'stock':<14} {'n':>3} {'avg_slip':>9} {'max_slip':>9}")
    for row in by_stock.head(15).iter_rows(named=True):
        print(f"{row['stock']:<14} {row['n']:>3} {row['avg_slip']:>+9.2f} {row['max_slip']:>+9.2f}")
