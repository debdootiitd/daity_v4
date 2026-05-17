"""Log daily P&L and cumulative sleeve return from a strategy parquet to wandb.

Reads a `<prefix>_N{N}.parquet` produced by `adaptive_calibrated_strategy.py` and
logs per-day metrics + summary scalars to a wandb run. Designed to run after the
backtest, on a CPU box (no GPU needed).

Usage:
  python -m daity.scripts.log_pnl_to_wandb \
      --pnl-path reports/v11_rerun/gbm_k1_n12_h3_N12.parquet \
      --project daity_phase25_rerun \
      --run-name "gbm_k1_n12_h3_d5_cost15" \
      --cost-bps 15 \
      --hold-trading-days 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnl-path", type=Path, required=True,
                    help="Per-day P&L parquet from adaptive_calibrated_strategy.py")
    ap.add_argument("--pnl-column", type=str, default="today_pnl_bps",
                    help="Column name to read as daily P&L (e.g. today_pnl_bps_stop for stop-loss adjusted)")
    ap.add_argument("--project", type=str, default="daity_phase25_rerun")
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--hold-trading-days", type=int, default=5,
                    help="For sleeve compounding (1/HOLD per sleeve)")
    ap.add_argument("--tags", type=str, default="", help="comma-separated wandb tags")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    import wandb

    df = pl.read_parquet(args.pnl_path).sort("test_date")
    n = df.shape[0]
    if n == 0:
        print(f"ERROR: empty P&L parquet at {args.pnl_path}", file=sys.stderr)
        return 1

    print(f"loaded {n} rows from {args.pnl_path}", flush=True)
    print(f"columns: {df.columns}", flush=True)

    if args.pnl_column not in df.columns:
        print(f"ERROR: column '{args.pnl_column}' not in parquet. "
              f"Available: {df.columns}", file=sys.stderr)
        return 1
    bps = df[args.pnl_column].to_numpy()
    nt = df["n_today_trades"].to_numpy()
    dates = df["test_date"].to_list()
    thresh = df["chosen_threshold"].to_list() if "chosen_threshold" in df.columns else [None] * n
    lookback_pnl = df["lookback_pnl_bps"].to_numpy() if "lookback_pnl_bps" in df.columns else np.zeros(n)

    hold = max(1, args.hold_trading_days)
    ann_factor = float(np.sqrt(252.0 / hold))

    # Cumulative sleeve return: prod(1 + bps/10000/hold) - 1 per day
    daily_factor = 1.0 + bps / 10000.0 / hold
    cumprod = np.cumprod(daily_factor)
    cum_sleeve_pct = (cumprod - 1.0) * 100.0

    # Summary scalars
    mean_all = float(bps.mean())
    sd_all = float(bps.std()) or 1e-9
    sharpe = mean_all / sd_all * ann_factor
    sleeve_final = float(cum_sleeve_pct[-1])
    traded = int((nt > 0).sum())
    trade_frac = traded / n
    traded_bps = bps[nt > 0] if traded > 0 else np.array([0.0])
    mean_traded = float(traded_bps.mean()) if traded > 0 else 0.0
    sd_traded = float(traded_bps.std()) or 1e-9
    sharpe_traded = mean_traded / sd_traded * ann_factor if traded > 0 else 0.0
    hit = float((traded_bps > 0).sum() / max(traded, 1) * 100)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        tags=tags,
        config={
            "cost_bps": args.cost_bps,
            "hold_trading_days": hold,
            "ann_factor": ann_factor,
            "n_test_days": n,
            "pnl_path": str(args.pnl_path),
        },
    )

    print(f"summary: traded={traded}/{n} ({100*trade_frac:.1f}%)  "
          f"mean_all={mean_all:+.2f} bps/day  Sharpe={sharpe:+.2f}  "
          f"sleeve_final={sleeve_final:+.2f}%  hit={hit:.0f}%",
          flush=True)

    # Step-level metrics (one per test day)
    for i, d in enumerate(dates):
        wandb.log({
            "test_day_index": i,
            "daily_pnl_bps": float(bps[i]),
            "cum_sleeve_return_pct": float(cum_sleeve_pct[i]),
            "gate_fired": int(nt[i] > 0),
            "n_today_trades": int(nt[i]),
            "lookback_pnl_bps": float(lookback_pnl[i]),
            "chosen_threshold": float(thresh[i]) if thresh[i] is not None else None,
        }, step=i)

    # Summary metrics (final scalars)
    wandb.summary["test_days"] = n
    wandb.summary["traded_days"] = traded
    wandb.summary["trade_frac"] = trade_frac
    wandb.summary["mean_bps_all"] = mean_all
    wandb.summary["sharpe_all"] = sharpe
    wandb.summary["mean_bps_traded"] = mean_traded
    wandb.summary["sharpe_traded"] = sharpe_traded
    wandb.summary["hit_pct"] = hit
    wandb.summary["sleeve_return_pct_final"] = sleeve_final
    wandb.summary["sleeve_return_annualized_pct"] = sleeve_final * (252 / hold) / n * 100

    # Custom matplotlib figure: daily P&L + cumulative sleeve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.bar(range(n), bps, color=["green" if b > 0 else ("red" if b < 0 else "gray") for b in bps],
            width=0.8, edgecolor="none")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Daily P&L (bps)")
    ax1.set_title(f"{args.run_name}: daily P&L (cost={args.cost_bps}bps, hold={hold}td)")
    ax1.grid(alpha=0.3)

    ax2.plot(range(n), cum_sleeve_pct, color="C0", lw=2)
    ax2.fill_between(range(n), 0, cum_sleeve_pct,
                     where=(cum_sleeve_pct > 0), color="C0", alpha=0.2)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Cumulative sleeve return (%)")
    ax2.set_xlabel("Test day index")
    ax2.set_title(f"Cumulative: final={sleeve_final:+.2f}%  Sharpe={sharpe:+.2f}  hit={hit:.0f}%")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    wandb.log({"daily_pnl_chart": wandb.Image(fig)})
    plt.close(fig)

    print(f"logged {n} steps and summary scalars to wandb run {run.url}", flush=True)
    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
