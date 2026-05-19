"""OB-based paper P&L for an ensemble predictions parquet.

For each rebalance day in the OB-window:
  - Pick top-K stocks by ensemble pred_lr.
  - Entry: walk top-5 ask depth at first OB snapshot of the rebal day -> entry VWAP.
  - Exit (rebal + hold trading days): walk top-5 bid depth at first OB snapshot of the exit day -> exit VWAP.
  - Realized log-return per pick = log(exit_vwap / entry_vwap).
  - Drop picks where:
      - no OB snapshot on entry or exit day
      - fully filling target ₹ requires > top-5 (i.e. not fully fillable)
  - Per-rebal P&L = log(mean exp(returns)) (1/K equal-weight).
  - Total cost paid is captured implicitly by entry/exit VWAP vs LTP.

Also produces:
  - Per-day pnl chart over the OB window (cumulative, entry/exit dates marked).
  - Comparison row: idealized close-to-close P&L for the same picks/holds (uses
    real_lr from the predictions parquet) at 30 bps assumed cost.

Usage:
  python ob_paper_pnl.py <predictions.parquet> --K 10 --hold 10 --pos-inr 200000 \
    --out reports/ob_paper_pnl.png
"""
import argparse
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("predictions")
ap.add_argument("--ob-dir", default="/Users/debdoot/Documents/Claude/Projects/daity/data/ob_parquet")
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--hold", type=int, default=10)
ap.add_argument("--pos-inr", type=float, default=200_000.0)
ap.add_argument("--out", default="/Users/debdoot/Documents/Claude/Projects/daity/reports/ob_paper_pnl.png")
ap.add_argument("--all-offsets", action="store_true", help="Use all hold offsets (rolling anchor)")
args = ap.parse_args()

OB_DIR = Path(args.ob_dir)

# Load predictions and identify OB window
df = pl.read_parquet(args.predictions).sort(["date", "stock"])
all_dates = sorted(df["date"].unique().to_list())
n = len(all_dates)
print(f"predictions: {df.shape[0]:,} rows, {n} OOS days")

# Discover OB window
sample_ob = pl.read_parquet(next(OB_DIR.glob("*.parquet")))
ob_min = sample_ob["ts"].min().date()
ob_max = sample_ob["ts"].max().date()
print(f"OB window: {ob_min} → {ob_max}")

# Filter predictions to dates inside OB window (those where entry AND exit both have OB data)
def candidate_rebal_dates(offset: int):
    dates = []
    di = offset
    while di + args.hold <= n - 1:
        entry = all_dates[di]
        exit_ = all_dates[di + args.hold]
        if ob_min <= entry <= ob_max and ob_min <= exit_ <= ob_max:
            dates.append((entry, exit_))
        di += args.hold
    return dates


def walk_asks(snap, target_inr):
    """VWAP for buying target_inr from top-5 ask. Returns (vwap, fully_filled)."""
    cum_inr = 0.0
    cum_qty = 0.0
    cum_cost = 0.0
    for i in range(1, 6):
        p = snap[f"ask{i}_price"][0]
        q = snap[f"ask{i}_qty"][0]
        if p is None or q is None or p <= 0 or q <= 0:
            continue
        notional = p * q
        if cum_inr + notional >= target_inr:
            need = target_inr - cum_inr
            qty_here = need / p
            cum_qty += qty_here
            cum_cost += qty_here * p
            return cum_cost / cum_qty, True
        cum_inr += notional
        cum_qty += q
        cum_cost += p * q
    return (cum_cost / cum_qty if cum_qty > 0 else None), False


def walk_bids(snap, target_qty):
    """VWAP for selling target_qty from top-5 bids. Returns (vwap, fully_filled)."""
    cum_qty = 0.0
    cum_proceeds = 0.0
    remaining = target_qty
    for i in range(1, 6):
        p = snap[f"bid{i}_price"][0]
        q = snap[f"bid{i}_qty"][0]
        if p is None or q is None or p <= 0 or q <= 0:
            continue
        take = min(q, remaining)
        cum_qty += take
        cum_proceeds += take * p
        remaining -= take
        if remaining <= 0:
            return cum_proceeds / cum_qty, True
    return (cum_proceeds / cum_qty if cum_qty > 0 else None), False


def first_snap_for(stock: str, date) -> pl.DataFrame | None:
    ob_path = OB_DIR / f"{stock}.parquet"
    if not ob_path.exists():
        return None
    ob = pl.read_parquet(ob_path)
    day_ob = ob.filter(pl.col("ts").dt.date() == date).sort("ts")
    return day_ob.head(1) if day_ob.shape[0] > 0 else None


# Gather rebals
offsets_to_try = list(range(args.hold)) if args.all_offsets else [0]
all_rebals = []
for off in offsets_to_try:
    for entry, exit_ in candidate_rebal_dates(off):
        all_rebals.append((off, entry, exit_))
print(f"Total (offset, entry, exit) triples in OB window: {len(all_rebals)}")

# Compute per-rebal P&L (OB execution + ideal close-to-close)
rebal_results = []
for off, entry, exit_ in all_rebals:
    picks_df = (df.filter(pl.col("date") == entry)
                  .sort("pred_lr", descending=True)
                  .head(args.K))
    if picks_df.shape[0] < args.K:
        continue
    picks = picks_df["stock"].to_list()

    # Per-pick OB and ideal returns
    ob_returns = []
    ideal_returns = []
    n_dropped = 0
    for stock in picks:
        entry_snap = first_snap_for(stock, entry)
        exit_snap = first_snap_for(stock, exit_)
        if entry_snap is None or exit_snap is None or entry_snap.shape[0] == 0 or exit_snap.shape[0] == 0:
            n_dropped += 1
            continue
        # Entry: buy ₹pos_inr worth at top-5 ask
        entry_vwap, fully_buy = walk_asks(entry_snap, args.pos_inr)
        if not fully_buy or entry_vwap is None:
            n_dropped += 1
            continue
        qty = args.pos_inr / entry_vwap
        # Exit: sell qty at top-5 bid
        exit_vwap, fully_sell = walk_bids(exit_snap, qty)
        if not fully_sell or exit_vwap is None:
            n_dropped += 1
            continue
        ob_ret = float(np.log(exit_vwap / entry_vwap))
        ob_returns.append(ob_ret)

        # Ideal close-to-close from predictions parquet
        stock_lrs = df.filter((pl.col("stock") == stock) & (pl.col("date") >= entry) & (pl.col("date") < exit_))["real_lr"].to_numpy()
        if len(stock_lrs) > 0:
            ideal_returns.append(float(stock_lrs.sum()))

    if not ob_returns:
        continue
    ob_port = float(np.log(np.exp(np.array(ob_returns)).mean()))
    ideal_port = float(np.log(np.exp(np.array(ideal_returns)).mean())) if ideal_returns else 0.0
    ideal_port_net = ideal_port - 30.0 / 10_000  # 30 bps cost assumed
    rebal_results.append({
        "offset": off, "entry": entry, "exit": exit_,
        "n_filled": len(ob_returns), "n_dropped": n_dropped,
        "ob_port_lr": ob_port, "ob_port_pct": (np.exp(ob_port) - 1) * 100,
        "ideal_lr": ideal_port, "ideal_net": ideal_port_net,
        "ideal_pct": (np.exp(ideal_port_net) - 1) * 100,
        "ob_vs_ideal_bps": (ob_port - ideal_port) * 10_000,
        "picks": picks,
    })

if not rebal_results:
    print("No filled rebalances. Try smaller pos-inr or different K/hold.")
    raise SystemExit(1)

rr = pl.DataFrame([{k: v for k, v in r.items() if k != "picks"} for r in rebal_results])
print(f"\nFilled rebalances: {rr.shape[0]}")
print(f"  Mean OB-executed return: {rr['ob_port_pct'].mean():+.3f}%")
print(f"  Mean ideal-net return:   {rr['ideal_pct'].mean():+.3f}%")
print(f"  Mean OB-vs-ideal slippage (bps): {rr['ob_vs_ideal_bps'].mean():+.1f}")
print(f"  Median OB drop count per rebal: {rr['n_dropped'].median()}")

# Per-rebal table
print(f"\n{'off':>3} {'entry':>11} {'exit':>11} {'filled':>6} {'drop':>4} {'OB%':>7} {'ideal%':>7} {'OB-id bps':>10}")
for r in sorted(rebal_results, key=lambda x: (x["entry"], x["offset"])):
    print(f"{r['offset']:>3} {str(r['entry']):>11} {str(r['exit']):>11} {r['n_filled']:>6} {r['n_dropped']:>4} "
          f"{r['ob_port_pct']:>+7.3f} {r['ideal_pct']:>+7.3f} {r['ob_vs_ideal_bps']:>+10.1f}")

# Aggregate: Sharpe of OB vs Ideal
def ann_sharpe(rets, hold):
    rets = np.asarray(rets)
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return 0.0
    return rets.mean() / rets.std(ddof=1) * np.sqrt(250 / hold)
ob_sh = ann_sharpe(rr["ob_port_lr"].to_numpy(), args.hold)
id_sh = ann_sharpe(rr["ideal_net"].to_numpy(), args.hold)
print(f"\nAnnualised Sharpe (n={rr.shape[0]} rebalances, hold={args.hold}):")
print(f"  OB-executed: {ob_sh:+.2f}")
print(f"  Ideal-net:   {id_sh:+.2f}")

# Plot daily P&L
import datetime as dt
fig, axes = plt.subplots(3, 1, figsize=(13, 10))

# Panel 1: scatter of per-rebal returns (each dot = one offset's rebal)
ax = axes[0]
for r in rebal_results:
    x = dt.datetime.combine(r["entry"], dt.time())
    ax.scatter(x, r["ob_port_pct"], s=40, color="#1f77b4", alpha=0.85)
    ax.scatter(x, r["ideal_pct"], s=40, color="#ff7f0e", alpha=0.6, marker="x")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel(f"Per-rebal return % (K={args.K}, hold={args.hold})")
ax.set_title(f"OB-executed (•, blue) vs ideal (×, orange) P&L across all {len(rebal_results)} rebals · "
             f"{Path(args.predictions).stem} · pos=₹{args.pos_inr:,.0f}/pick")
ax.grid(alpha=0.3)

# Panel 2: daily P&L aggregated by entry date (mean across offsets that started that day)
from collections import defaultdict
by_entry_ob = defaultdict(list)
by_entry_id = defaultdict(list)
for r in rebal_results:
    by_entry_ob[r["entry"]].append(r["ob_port_pct"])
    by_entry_id[r["entry"]].append(r["ideal_pct"])
dates_sorted = sorted(by_entry_ob.keys())
ob_means = [np.mean(by_entry_ob[d]) for d in dates_sorted]
id_means = [np.mean(by_entry_id[d]) for d in dates_sorted]
xs = [dt.datetime.combine(d, dt.time()) for d in dates_sorted]
ax = axes[1]
w = dt.timedelta(hours=8)
ax.bar([x - w/2 for x in xs], ob_means, width=w, label="OB-executed", color="#1f77b4")
ax.bar([x + w/2 for x in xs], id_means, width=w, label="Ideal (30bps assumed)", color="#ff7f0e", alpha=0.7)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Daily P&L % (mean across offsets entering that day)")
ax.set_title(f"Daily P&L per entry date · OB Sharpe ann={ob_sh:+.2f}, Ideal Sharpe ann={id_sh:+.2f}")
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: equity curve for offset=0 only (clean cumulative)
off0 = sorted([r for r in rebal_results if r["offset"] == 0], key=lambda x: x["entry"])
if off0:
    cum_ob = np.exp(np.cumsum([r["ob_port_lr"] for r in off0])) - 1
    cum_id = np.exp(np.cumsum([r["ideal_net"] for r in off0])) - 1
    xs0 = [dt.datetime.combine(r["entry"], dt.time()) for r in off0]
    ax = axes[2]
    ax.plot(xs0, cum_ob * 100, marker="o", label="OB cumulative (offset=0)", color="#1f77b4")
    ax.plot(xs0, cum_id * 100, marker="s", label="Ideal cumulative (offset=0)", color="#ff7f0e", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cumulative return %")
    ax.set_xlabel("Rebalance entry date (offset=0 only)")
    ax.legend()
    ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"\nWrote chart: {args.out}")
