"""Multi-horizon strategy v2: NO-LEVERAGE accounting.

Each position takes 1/max_positions of capital. P&L is normalized so total
capital is always 100%. No double-counting from overlapping holds.

Strategy:
  - At each first-anchor: pick top-K (no-dup-stock) for each horizon in priority
  - Each new entry takes one free slot of capital
  - If no free slot, skip
  - Position exits at horizon target date; slot frees up; P&L credited
"""
import argparse
import numpy as np
import polars as pl
from datetime import date

ap = argparse.ArgumentParser()
ap.add_argument("--predictions-path", default="runs/cohort_modeB_v6_clf30_multianchor/predictions.parquet")
ap.add_argument("--test-start", default="2025-02-01")
ap.add_argument("--test-end", default="2026-04-30")
ap.add_argument("--cost-bps", type=float, default=15.0)
ap.add_argument("--max-positions", type=int, default=2)
ap.add_argument("--config", default="D")
args = ap.parse_args()

HOLD_DAYS = {
    "intraday_60m": 0, "intraday_120m": 0, "intraday_180m": 0,
    "to_close": 0, "overnight": 1, "next_day_1h": 1, "next_day_eod": 1,
}

CONFIGS = {
    "J":    [{"horizon": "next_day_eod", "thr": 0.0, "topk": 1}],
    "D":    [{"horizon": "next_day_eod", "thr": 0.0, "topk": 1},
             {"horizon": "next_day_1h",  "thr": 0.1, "topk": 1}],
    "M":    [{"horizon": "next_day_eod", "thr": 0.0, "topk": 1},
             {"horizon": "next_day_1h",  "thr": 0.5, "topk": 1},
             {"horizon": "to_close",     "thr": 0.7, "topk": 1}],
    "N":    [{"horizon": "next_day_eod", "thr": 0.2, "topk": 2},
             {"horizon": "next_day_1h",  "thr": 0.3, "topk": 1}],
    "O":    [{"horizon": "next_day_eod", "thr": 0.5, "topk": 2},
             {"horizon": "next_day_1h",  "thr": 0.5, "topk": 1}],
}

cfgs = CONFIGS[args.config]
print(f"=== Config {args.config}, max_positions={args.max_positions}: {[(c['horizon'],c['thr'],c['topk']) for c in cfgs]} ===")

df = pl.scan_parquet(args.predictions_path).filter(
    (pl.col("date") >= date.fromisoformat(args.test_start)) &
    (pl.col("date") <= date.fromisoformat(args.test_end))
).select(["date","stock","horizon","anchor_us","pred_win_prob","real_lr"]).collect()

df = df.with_columns(pl.col("anchor_us").cast(pl.Int64))
first = df.group_by("date").agg(pl.col("anchor_us").min().alias("first_us"))
df = df.join(first, on="date").filter(pl.col("anchor_us") == pl.col("first_us")).drop("first_us")
all_dates = sorted(df["date"].unique().to_list())

# Portfolio: list of {stock, exit_idx, real_lr, horizon, capital_frac, entry_date}
# Each slot is 1/max_positions of capital.
SLOT_CAPITAL = 1.0 / args.max_positions
portfolio = []  # list of positions

daily_port_bps = []  # daily portfolio-level bps (normalized to total capital)
daily_open = []
daily_entries = []
realized_by_h = {c["horizon"]: [] for c in cfgs}
realized_all = []
unused_capital_fraction = []  # how much cash sits idle each day

for di, d in enumerate(all_dates):
    # 1) Exits: positions whose exit_idx <= di are realized today
    exits = [p for p in portfolio if p["exit_idx"] <= di]
    portfolio = [p for p in portfolio if p["exit_idx"] > di]
    pf_bps_today = 0.0
    for p in exits:
        # Cost was subtracted from gross at entry
        net_bps = (np.exp(p["real_lr"]) - 1) * 10000 - args.cost_bps
        # Contribution to portfolio bps = slot_capital * net_bps
        pf_bps_today += p["capital_frac"] * net_bps
        realized_by_h[p["horizon"]].append(net_bps)
        realized_all.append(net_bps)
    daily_port_bps.append(pf_bps_today)

    # 2) Entries: for each horizon, pick top-K of un-held stocks; allocate slot if free
    n_entries = 0
    free_slots = args.max_positions - len(portfolio)
    if free_slots > 0:
        held = set(p["stock"] for p in portfolio)
        day_rows = df.filter(pl.col("date") == d)
        for cfg in cfgs:
            if free_slots <= 0:
                break
            hor = cfg["horizon"]; thr = cfg["thr"]; topk = cfg["topk"]
            cands = day_rows.filter(
                (pl.col("horizon") == hor) & (pl.col("pred_win_prob") >= thr)
            ).sort("pred_win_prob", descending=True)
            if held:
                cands = cands.filter(~pl.col("stock").is_in(list(held)))
            if cands.shape[0] == 0:
                continue
            pick = cands.head(min(topk, free_slots))
            for row in pick.iter_rows(named=True):
                hd = HOLD_DAYS[hor]
                exit_idx = di + 2 if hd >= 1 else di  # daily horizon: hold thru di+1, exit at di+2
                portfolio.append({
                    "stock": row["stock"],
                    "exit_idx": exit_idx,
                    "real_lr": row["real_lr"],
                    "horizon": hor,
                    "capital_frac": SLOT_CAPITAL,
                    "entry_date": d,
                })
                held.add(row["stock"])
                free_slots -= 1
                n_entries += 1
                if free_slots <= 0:
                    break

    daily_open.append(len(portfolio))
    daily_entries.append(n_entries)
    unused_capital_fraction.append((args.max_positions - len(portfolio)) / args.max_positions)

# Close remaining
for p in portfolio:
    net_bps = (np.exp(p["real_lr"]) - 1) * 10000 - args.cost_bps
    realized_by_h[p["horizon"]].append(net_bps)
    realized_all.append(net_bps)

arr = np.array(daily_port_bps)
sharpe = float(arr.mean()) / (float(arr.std()) or 1e-9) * np.sqrt(252.0)
sleeve = float(np.prod(1 + arr/10000.0) - 1) * 100
n_trades = sum(daily_entries)
total = sum(realized_all)
avg_per_trade = total / max(n_trades, 1)
hit = sum(1 for p in realized_all if p > 0) / max(n_trades, 1) * 100
avg_idle_cash = float(np.mean(unused_capital_fraction)) * 100

print(f"\n=== UNLEVERAGED RESULT (1/{args.max_positions} per slot) ===")
print(f"  Test days: {len(all_dates)}")
print(f"  Total trades: {n_trades}, avg per day: {n_trades/len(all_dates):.2f}")
print(f"  Avg concurrent positions: {np.mean(daily_open):.2f} / {args.max_positions}, idle cash avg: {avg_idle_cash:.1f}%")
print(f"  Mean daily portfolio P&L: {arr.mean():+.2f} bps")
print(f"  Sharpe (annualized sqrt252): {sharpe:+.2f}")
print(f"  Sleeve return: {sleeve:+.2f}%")
print(f"  Avg per-trade gross: {avg_per_trade:+.2f} bps, hit: {hit:.1f}%")
print(f"\n=== BY HORIZON ===")
for c in cfgs:
    pnls = realized_by_h[c["horizon"]]
    if pnls:
        m = float(np.mean(pnls)); h_hit = sum(1 for p in pnls if p > 0)/len(pnls)*100
        print(f"  {c['horizon']:18s} thr={c['thr']:.2f} K={c['topk']}: n={len(pnls):4d}, mean={m:+.2f} bps, hit={h_hit:.1f}%")
