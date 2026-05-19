"""Rolling-entry / rolling-exit strategy with portfolio cap.

Each trading day:
  1. EXIT any position whose age = hold_days (close it today).
  2. SCREEN candidates: top-N by pred_lr today AND pred_lr > threshold.
  3. ENTER as many candidates as fit in remaining slots (max_positions − held).
  4. NO same-stock duplicate: skip a candidate already in portfolio.

Daily P&L:
  port_lr(d) = (1/max_positions) * sum(daily log-return of each held stock)
  Cost: 15 bps deducted from day d for each entry; 15 bps deducted from day d for each exit.

Output:
  - Per-day P&L series + portfolio fill (# active positions).
  - Annualised Sharpe, sleeve, max DD.
  - Position-level table, turnover.
  - Plot comparing to chunky rebal baseline.

Usage:
  python rolling_entry_strategy.py runs/autoloop_predictions/ensemble_9_models.parquet \
    --max-pos 10 --hold 10 --threshold 0.0 --top-n 20 --cost-bps 30 \
    --out reports/rolling_entry_pnl.png
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
ap.add_argument("--max-pos", type=int, default=10, help="Max concurrent positions")
ap.add_argument("--hold", type=int, default=10, help="Hold days per position")
ap.add_argument("--threshold", type=float, default=0.0, help="Min pred_lr to consider entering")
ap.add_argument("--top-n", type=int, default=20, help="Daily candidate pool (top-N by pred_lr)")
ap.add_argument("--cost-bps", type=float, default=30.0, help="Round-trip cost in bps per stock")
ap.add_argument("--out", default="/Users/debdoot/Documents/Claude/Projects/daity/reports/rolling_entry_pnl.png")
ap.add_argument("--label", default=None)
args = ap.parse_args()

if args.label is None:
    args.label = Path(args.predictions).stem

# Load
df = pl.read_parquet(args.predictions).sort(["date", "stock"])
all_dates = sorted(df["date"].unique().to_list())
print(f"Predictions: {df.shape[0]:,} rows · {len(all_dates)} OOS days · {all_dates[0]} → {all_dates[-1]}")

# Build per-stock daily real_lr lookup
psd = {}
for s in df["stock"].unique():
    sub = df.filter(pl.col("stock") == s).sort("date")
    psd[s] = {d: float(r) for d, r in zip(sub["date"].to_list(), sub["real_lr"].to_numpy())}

# Pre-rank candidates per day
top_picks_per_day = {}
for d in all_dates:
    sub = (df.filter(pl.col("date") == d)
             .sort("pred_lr", descending=True)
             .head(args.top_n))
    top_picks_per_day[d] = sub.select(["stock", "pred_lr"]).rows()

# Simulate day-by-day
portfolio = []          # list of (stock, entry_date, entry_idx_in_all_dates)
positions_log = []      # full closed positions
daily_pnl_lr = []       # net daily log-return
fill_count = []         # active positions count per day
entries_per_day = []
exits_per_day = []

half_cost_lr = args.cost_bps / 2 / 10_000  # per-side cost in log-return units
w = 1.0 / args.max_pos                       # weight per position

for di, d in enumerate(all_dates):
    # 1. EXIT positions whose age == hold (trading days)
    new_portfolio = []
    exits_today = 0
    for p in portfolio:
        age = di - p["entry_idx"]
        if age >= args.hold:
            # Close out, log realized
            cum_lr = 0.0
            for j in range(p["entry_idx"], di):
                cum_lr += psd.get(p["stock"], {}).get(all_dates[j], 0.0)
            positions_log.append({
                **p, "exit_date": d, "hold_actual": age, "cum_lr": cum_lr,
                "cum_pct": (np.exp(cum_lr) - 1) * 100,
            })
            exits_today += 1
        else:
            new_portfolio.append(p)
    portfolio = new_portfolio

    # 2. Daily P&L from positions held over (d-1, d)  — use today's real_lr
    todays_lrs = [psd.get(p["stock"], {}).get(d, 0.0) for p in portfolio]
    port_simple = w * sum(np.exp(r) - 1 for r in todays_lrs)
    # Convert to log-return
    port_lr_day = float(np.log(1 + port_simple))

    # 3. ENTER new positions if slots available
    entries_today = 0
    slots = args.max_pos - len(portfolio)
    held_stocks = {p["stock"] for p in portfolio}
    if slots > 0 and di < len(all_dates) - 1:  # don't enter on last day
        for stock, pred in top_picks_per_day[d]:
            if slots == 0:
                break
            if pred < args.threshold:
                break  # sorted descending
            if stock in held_stocks:
                continue
            portfolio.append({"stock": stock, "entry_date": d, "entry_idx": di, "pred_lr": pred})
            held_stocks.add(stock)
            entries_today += 1
            slots -= 1

    # 4. Cost: half-RT per entry today + half-RT per exit today, weighted
    cost_today = (entries_today + exits_today) * half_cost_lr * w

    daily_pnl_lr.append(port_lr_day - cost_today)
    fill_count.append(len(portfolio))
    entries_per_day.append(entries_today)
    exits_per_day.append(exits_today)

# Close out any remaining positions at end (forced exit on last day, with cost)
for p in portfolio:
    cum_lr = sum(psd.get(p["stock"], {}).get(all_dates[j], 0.0)
                 for j in range(p["entry_idx"], len(all_dates)))
    positions_log.append({**p, "exit_date": all_dates[-1], "hold_actual": len(all_dates) - 1 - p["entry_idx"],
                          "cum_lr": cum_lr, "cum_pct": (np.exp(cum_lr) - 1) * 100})
    daily_pnl_lr[-1] -= half_cost_lr * w  # exit cost on last day

daily_pnl = np.array(daily_pnl_lr)
cum_lr = np.cumsum(daily_pnl)
sleeve_pct = (np.exp(cum_lr[-1]) - 1) * 100

# Stats
ann_factor = np.sqrt(250)
sharpe = daily_pnl.mean() / (daily_pnl.std(ddof=1) + 1e-12) * ann_factor if daily_pnl.std() > 0 else 0
n_pos_closed = len(positions_log)
total_entries = sum(entries_per_day)
total_exits = sum(exits_per_day) + (args.max_pos)  # plus final close
hit = sum(1 for p in positions_log if p["cum_lr"] > 0) / max(n_pos_closed, 1)
mean_pos_pct = np.mean([p["cum_pct"] for p in positions_log]) if positions_log else 0
median_pos_pct = np.median([p["cum_pct"] for p in positions_log]) if positions_log else 0
mean_hold = np.mean([p["hold_actual"] for p in positions_log]) if positions_log else 0
# Max drawdown on equity curve
eq = np.exp(cum_lr)
running_max = np.maximum.accumulate(eq)
dd = eq / running_max - 1
mdd = dd.min() * 100

print(f"\n=== Rolling-entry strategy: {args.label} ===")
print(f"  max_pos={args.max_pos} hold={args.hold} threshold={args.threshold:.4f} top_n={args.top_n} cost={args.cost_bps:.0f} bps RT")
print(f"  OOS span: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} days)")
print()
print(f"  Sleeve (net):        {sleeve_pct:+.2f}%")
print(f"  Annualised Sharpe:   {sharpe:+.2f}")
print(f"  Max drawdown:        {mdd:+.2f}%")
print(f"  Daily mean / std:    {daily_pnl.mean() * 10_000:+.2f} bps / {daily_pnl.std(ddof=1) * 10_000:.2f} bps")
print()
print(f"  Positions opened:    {total_entries}")
print(f"  Positions closed:    {n_pos_closed}")
print(f"  Position hit-rate:   {hit:.1%}")
print(f"  Mean position return: {mean_pos_pct:+.2f}%  median: {median_pos_pct:+.2f}%")
print(f"  Mean hold actual:    {mean_hold:.1f} days (target {args.hold})")
print()
print(f"  Annualised turnover: {total_entries + n_pos_closed} trades/yr "
      f"vs chunky-rebal-baseline at K={args.max_pos} hold={args.hold} = ~{int(250/args.hold) * args.max_pos * 2} trades/yr")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
xs = all_dates

# Panel 1: Equity curve
ax = axes[0]
ax.plot(xs, (eq - 1) * 100, color="#1f77b4", lw=1.5, label=f"Net equity (sleeve {sleeve_pct:+.1f}%, Sharpe {sharpe:+.2f})")
ax.fill_between(xs, (eq - 1) * 100, 0, where=eq < 1, color="red", alpha=0.15)
ax.fill_between(xs, (eq - 1) * 100, 0, where=eq >= 1, color="green", alpha=0.1)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Cumulative return %")
ax.set_title(f"Rolling-entry strategy · {args.label} · max_pos={args.max_pos} hold={args.hold} "
             f"thresh={args.threshold:.4f} cost={args.cost_bps:.0f} bps RT")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Panel 2: Daily P&L
ax = axes[1]
ax.bar(xs, daily_pnl * 100, width=1.0, color="#1f77b4")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Daily P&L %")
ax.grid(alpha=0.3)

# Panel 3: Portfolio fill + entries/exits
ax = axes[2]
ax.plot(xs, fill_count, color="#2ca02c", lw=1.5, label=f"Active positions (mean {np.mean(fill_count):.1f}/{args.max_pos})")
ax2 = ax.twinx()
ax2.bar(xs, entries_per_day, width=1.0, color="#1f77b4", alpha=0.5, label="entries")
ax2.bar(xs, [-x for x in exits_per_day], width=1.0, color="#d62728", alpha=0.5, label="exits")
ax.set_ylabel("Active positions")
ax2.set_ylabel("Entries (+) / Exits (−) per day")
ax.set_xlabel("Date")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"\nWrote chart: {args.out}")

# Top/bottom positions table
print(f"\n=== Top 10 winning positions ===")
sorted_pos = sorted(positions_log, key=lambda x: -x["cum_pct"])
print(f"{'stock':<14} {'entry':>11} {'exit':>11} {'hold':>4} {'cum%':>7}")
for p in sorted_pos[:10]:
    print(f"{p['stock']:<14} {str(p['entry_date']):>11} {str(p['exit_date']):>11} {p['hold_actual']:>4} {p['cum_pct']:>+7.2f}")
print(f"\n=== Top 10 losing positions ===")
for p in sorted_pos[-10:]:
    print(f"{p['stock']:<14} {str(p['entry_date']):>11} {str(p['exit_date']):>11} {p['hold_actual']:>4} {p['cum_pct']:>+7.2f}")
