"""OB paper P&L with realistic partial-fill execution.

For each pick at ₹pos_inr target:
  - Walk top-5 ask depth. Buy as much as available within target.
  - actual_qty = min(target_qty, top5_total_qty_available)
  - VWAP = cum_cost / actual_qty
  - Capital deployed = actual_qty * VWAP_entry  (may be < target)
  - Track unused cash per pick
At exit (rebal + hold trading days):
  - Walk top-5 bid depth. Sell actual_qty.
  - If even bids can't absorb the sell, sell what's possible. (Residual qty marked at last bid available — penalty for illiquidity.)
  - exit_VWAP = cum_proceeds / qty_sold

Per-rebalance metrics:
  - actual capital deployed (sum of qty * entry_vwap)
  - realized capital at exit (sum of qty_sold * exit_vwap + qty_unsold * last_bid)
  - realized log-return per pick = log(realized_pick / deployed_pick)
  - portfolio return weighted by deployed capital
  - "Slippage vs ideal" = realized log-return vs ideal close-to-close

Also compares idealized close-to-close P&L (using real_lr from predictions parquet),
applies same 30 bps assumed cost.

Usage:
  python ob_paper_pnl_partial.py runs/autoloop_predictions/ensemble_9_models.parquet \
    --K 10 --hold 10 --pos-inr 5000000 --all-offsets --out reports/ob_50L.png
"""
import argparse
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime as dt
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("predictions")
ap.add_argument("--ob-dir", default="/Users/debdoot/Documents/Claude/Projects/daity/data/ob_parquet")
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--hold", type=int, default=10)
ap.add_argument("--pos-inr", type=float, default=5_000_000.0, help="Target ₹ per pick")
ap.add_argument("--out", default="/Users/debdoot/Documents/Claude/Projects/daity/reports/ob_paper_partial.png")
ap.add_argument("--all-offsets", action="store_true")
args = ap.parse_args()

OB_DIR = Path(args.ob_dir)

df = pl.read_parquet(args.predictions).sort(["date", "stock"])
all_dates = sorted(df["date"].unique().to_list())
n = len(all_dates)
print(f"predictions: {df.shape[0]:,} rows, {n} OOS days")

# OB window
sample_ob = pl.read_parquet(next(OB_DIR.glob("*.parquet")))
ob_min = sample_ob["ts"].min().date()
ob_max = sample_ob["ts"].max().date()
print(f"OB window: {ob_min} → {ob_max}")
print(f"Target position: ₹{args.pos_inr:,.0f} per pick")


def candidate_rebal_dates(offset: int):
    out = []
    di = offset
    while di + args.hold <= n - 1:
        e = all_dates[di]
        x = all_dates[di + args.hold]
        if ob_min <= e <= ob_max and ob_min <= x <= ob_max:
            out.append((e, x))
        di += args.hold
    return out


def walk_asks_partial(snap, target_inr):
    """Buy up to target_inr worth from top-5 ask. Returns (cum_qty, vwap)."""
    cum_inr, cum_qty, cum_cost = 0.0, 0.0, 0.0
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
            return cum_qty, cum_cost / cum_qty
        cum_inr += notional
        cum_qty += q
        cum_cost += p * q
    # Used full top-5
    return cum_qty, (cum_cost / cum_qty if cum_qty > 0 else None)


def walk_bids_partial(snap, target_qty):
    """Sell up to target_qty from top-5 bid. Returns (qty_sold, vwap, last_bid_used)."""
    cum_qty, cum_proceeds = 0.0, 0.0
    remaining = target_qty
    last_bid_used = None
    for i in range(1, 6):
        p = snap[f"bid{i}_price"][0]
        q = snap[f"bid{i}_qty"][0]
        if p is None or q is None or p <= 0 or q <= 0:
            continue
        take = min(q, remaining)
        cum_qty += take
        cum_proceeds += take * p
        last_bid_used = p
        remaining -= take
        if remaining <= 0:
            return cum_qty, cum_proceeds / cum_qty, last_bid_used
    return cum_qty, (cum_proceeds / cum_qty if cum_qty > 0 else None), last_bid_used


def first_snap(stock, date):
    p = OB_DIR / f"{stock}.parquet"
    if not p.exists(): return None
    ob = pl.read_parquet(p)
    day_ob = ob.filter(pl.col("ts").dt.date() == date).sort("ts")
    return day_ob.head(1) if day_ob.shape[0] > 0 else None


offsets = list(range(args.hold)) if args.all_offsets else [0]
triples = []
for off in offsets:
    for e, x in candidate_rebal_dates(off):
        triples.append((off, e, x))
print(f"Rebals in OB window: {len(triples)}")

rebals = []
for off, entry, exit_ in triples:
    picks_df = (df.filter(pl.col("date") == entry)
                  .sort("pred_lr", descending=True)
                  .head(args.K))
    if picks_df.shape[0] < args.K:
        continue
    picks = picks_df["stock"].to_list()
    target_per_pick = args.pos_inr  # all picks share total ₹K*pos
    pick_results = []
    total_deployed = 0.0
    total_terminal = 0.0
    ideal_returns = []
    for stock in picks:
        e_snap = first_snap(stock, entry)
        x_snap = first_snap(stock, exit_)
        if e_snap is None or x_snap is None or e_snap.shape[0] == 0 or x_snap.shape[0] == 0:
            continue
        qty_bought, entry_vwap = walk_asks_partial(e_snap, target_per_pick)
        if qty_bought <= 0 or entry_vwap is None:
            continue
        deployed = qty_bought * entry_vwap
        qty_sold, exit_vwap, last_bid = walk_bids_partial(x_snap, qty_bought)
        if exit_vwap is None or qty_sold <= 0:
            continue
        # Terminal value of position: sold portion + unsold @ last bid available (penalty)
        unsold = qty_bought - qty_sold
        if unsold > 1e-9 and last_bid is not None:
            # apply ~50bps penalty for unsellable residual (deep market impact)
            terminal = qty_sold * exit_vwap + unsold * last_bid * 0.995
        else:
            terminal = qty_sold * exit_vwap
        pick_ret_lr = float(np.log(terminal / deployed))

        # Ideal close-to-close (uses real_lr from preds)
        stock_lrs = (df.filter((pl.col("stock") == stock) & (pl.col("date") >= entry) & (pl.col("date") < exit_))
                       ["real_lr"].to_numpy())
        ideal_lr = float(stock_lrs.sum()) if len(stock_lrs) > 0 else 0.0

        pick_results.append({
            "stock": stock, "deployed": deployed, "terminal": terminal,
            "qty_bought": qty_bought, "qty_sold": qty_sold,
            "entry_vwap": entry_vwap, "exit_vwap": exit_vwap,
            "pick_lr_ob": pick_ret_lr, "pick_lr_ideal": ideal_lr,
            "fill_pct_entry": deployed / target_per_pick * 100,
        })
        total_deployed += deployed
        total_terminal += terminal
        ideal_returns.append(ideal_lr)
    if not pick_results:
        continue
    # Equal-deployed weight (each pick gets its actual deployed share)
    ob_port_lr = float(np.log(total_terminal / total_deployed))
    # Ideal portfolio: 1/K equal-weighted
    ideal_port_lr = float(np.log(np.exp(np.array(ideal_returns)).mean())) if ideal_returns else 0.0
    ideal_port_net = ideal_port_lr - 30.0 / 10_000
    rebals.append({
        "offset": off, "entry": entry, "exit": exit_,
        "n_picks": len(pick_results),
        "total_target": args.K * target_per_pick,
        "total_deployed": total_deployed,
        "total_terminal": total_terminal,
        "fill_pct": total_deployed / (args.K * target_per_pick) * 100,
        "ob_port_lr": ob_port_lr,
        "ob_port_pct": (np.exp(ob_port_lr) - 1) * 100,
        "ideal_lr": ideal_port_lr,
        "ideal_net": ideal_port_net,
        "ideal_pct": (np.exp(ideal_port_net) - 1) * 100,
        "ob_vs_ideal_bps": (ob_port_lr - ideal_port_lr) * 10_000,
        "picks": pick_results,
    })

if not rebals:
    print("No rebals filled. Try smaller pos-inr or different window.")
    raise SystemExit(1)

rr = pl.DataFrame([{k: v for k, v in r.items() if k != "picks"} for r in rebals])

print(f"\n=== OB execution summary (₹{args.pos_inr:,.0f}/pick, K={args.K}, hold={args.hold}) ===")
print(f"  Rebals with at least 1 fill: {rr.shape[0]}")
print(f"  Mean fill %% of total target portfolio: {rr['fill_pct'].mean():.1f}%")
print(f"  Mean picks per rebal: {rr['n_picks'].mean():.1f} / {args.K}")
print(f"  Mean OB-executed return per rebal: {rr['ob_port_pct'].mean():+.3f}%")
print(f"  Mean ideal-net return per rebal:   {rr['ideal_pct'].mean():+.3f}%")
print(f"  Mean OB-vs-ideal slippage (bps):   {rr['ob_vs_ideal_bps'].mean():+.1f}")
print(f"  P&L IRR per rebal std: OB {rr['ob_port_pct'].std(ddof=1):.2f}%, ideal {rr['ideal_pct'].std(ddof=1):.2f}%")

# Sharpe (in-window)
def ann_sharpe(x, hold):
    x = np.asarray(x)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / x.std(ddof=1) * np.sqrt(250 / hold)
ob_sh = ann_sharpe(rr["ob_port_lr"].to_numpy(), args.hold)
id_sh = ann_sharpe(rr["ideal_net"].to_numpy(), args.hold)
print(f"  In-window annualised Sharpe: OB {ob_sh:+.2f}, Ideal {id_sh:+.2f}")

# Fill stats per pick across all rebals
all_picks = []
for r in rebals:
    for p in r["picks"]:
        all_picks.append(p)
fill_pcts = np.array([p["fill_pct_entry"] for p in all_picks])
print(f"\n  Per-pick fill stats (n={len(all_picks)}):")
print(f"    Mean fill %% of target ₹/pick: {fill_pcts.mean():.1f}%")
print(f"    Median fill %%:                {np.median(fill_pcts):.1f}%")
print(f"    p25/p75 fill %%:               {np.percentile(fill_pcts,25):.1f}% / {np.percentile(fill_pcts,75):.1f}%")
print(f"    Picks fully filled (>=99%):    {sum(1 for x in fill_pcts if x>=99)} / {len(all_picks)} = {sum(1 for x in fill_pcts if x>=99)/len(all_picks)*100:.1f}%")

# Per-rebal table
print(f"\n{'off':>3} {'entry':>11} {'fill%':>6} {'picks':>5} {'deploy(L)':>10} {'OB%':>7} {'ideal%':>7} {'OB-id bps':>10}")
for r in sorted(rebals, key=lambda x: (x["entry"], x["offset"])):
    print(f"{r['offset']:>3} {str(r['entry']):>11} {r['fill_pct']:>6.1f} {r['n_picks']:>5} {r['total_deployed']/100_000:>10.1f} "
          f"{r['ob_port_pct']:>+7.3f} {r['ideal_pct']:>+7.3f} {r['ob_vs_ideal_bps']:>+10.1f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
entries = [r["entry"] for r in rebals]
ob_pct = [r["ob_port_pct"] for r in rebals]
id_pct = [r["ideal_pct"] for r in rebals]
xs = [dt.datetime.combine(e, dt.time()) for e in entries]

ax = axes[0]
for r in rebals:
    x = dt.datetime.combine(r["entry"], dt.time())
    ax.scatter(x, r["ob_port_pct"], s=40, color="#1f77b4", alpha=0.85)
    ax.scatter(x, r["ideal_pct"], s=40, color="#ff7f0e", marker="x", alpha=0.6)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel(f"Per-rebal return % (K={args.K} h={args.hold})")
ax.set_title(f"OB execution at ₹{args.pos_inr:,.0f}/pick · {Path(args.predictions).stem} · "
             f"avg deploy={rr['fill_pct'].mean():.0f}% of target")
ax.grid(alpha=0.3)

ax = axes[1]
deploy_pcts = [r["fill_pct"] for r in rebals]
ax.bar(xs, deploy_pcts, width=dt.timedelta(hours=12), color="#2ca02c", alpha=0.6, label="Capital deployed %")
ax.axhline(100, color="k", lw=0.5, linestyle="--", alpha=0.5)
ax.set_ylabel("% of target portfolio deployed")
ax.set_ylim(0, 110)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
off0 = sorted([r for r in rebals if r["offset"] == 0], key=lambda x: x["entry"])
if off0:
    cum_ob = np.exp(np.cumsum([r["ob_port_lr"] for r in off0])) - 1
    cum_id = np.exp(np.cumsum([r["ideal_net"] for r in off0])) - 1
    xs0 = [dt.datetime.combine(r["entry"], dt.time()) for r in off0]
    ax.plot(xs0, cum_ob * 100, marker="o", label=f"OB cumulative (Sharpe ann {ob_sh:+.2f})", color="#1f77b4")
    ax.plot(xs0, cum_id * 100, marker="s", label=f"Ideal cumulative (Sharpe ann {id_sh:+.2f})", color="#ff7f0e", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cumulative return %")
    ax.set_xlabel("Rebalance entry date")
    ax.legend()
    ax.grid(alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"\nWrote chart: {args.out}")
