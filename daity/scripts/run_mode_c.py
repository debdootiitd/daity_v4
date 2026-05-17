"""Mode C walk-forward driver: 5-day rolling fine-tune + eval through 2025.

For each 5-day window in the configured period:
  1. train_end   = window_start - 1 trading day  (we never see window data in training)
  2. val_start   = window_start
  3. val_end     = window_start + 4 trading days
  4. Load the prior window's checkpoint (best.pt from the previous window,
     or `--seed-ckpt` for the very first window).
  5. Fine-tune the model from that checkpoint for `--ft-steps` steps using
     train_start..train_end as the training period (recent regime upweighted
     via time-ordered iteration).
  6. Evaluate on the 5-day val window: per-day P&L (top-K), per-horizon IC.
  7. Save the updated checkpoint into the window's log_dir/checkpoints/best.pt,
     which becomes the seed for the next window.

Outputs:
  - One run dir per window: `runs/cohort_modeC_v1/window_YYYYMMDD/`
  - Combined results parquet: `reports/cohort_modeC_v1_pnl.parquet`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-ckpt", type=Path, required=True,
                    help="Starting checkpoint (e.g. Mode A v5 best.pt).")
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    # Walk-forward
    ap.add_argument("--start", type=str, required=True,
                    help="First window start date (YYYY-MM-DD).")
    ap.add_argument("--end",   type=str, required=True,
                    help="Last window end date (YYYY-MM-DD).")
    ap.add_argument("--window-days", type=int, default=5)
    ap.add_argument("--train-start", type=str, default="2019-01-01",
                    help="Initial training window start (held across all windows).")
    # Universe
    ap.add_argument("--universe-end", type=str, default=None,
                    help="Upper bound for universe filter (matches seed ckpt).")
    # Fine-tune
    ap.add_argument("--ft-steps", type=int, default=400,
                    help="Fine-tune steps per window. Set 0 to disable (static-per-window).")
    ap.add_argument("--ft-lr",    type=float, default=5e-5)
    ap.add_argument("--ft-warmup", type=float, default=0.05)
    ap.add_argument("--wd",       type=float, default=5e-2)
    ap.add_argument("--rank-top-k", type=int, default=20)
    ap.add_argument("--smooth-l1-beta", type=float, default=0.005)
    # Eval / strategy
    ap.add_argument("--horizon", type=str, default="day_plus_3")
    ap.add_argument("--top-k",   type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--anchor-ist", type=str, default="15:25")
    # IO
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--wandb-project", type=str, default=None)
    return ap.parse_args()


def trading_days_in(start: date, end: date, calendar) -> list[date]:
    out = []
    d = start
    while d <= end:
        if calendar.is_trading_day(d):
            out.append(d)
        d = d + timedelta(days=1)
    return out


def main() -> int:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    from daity.data.calendar_nse import NSECalendar
    calendar = NSECalendar.from_cache(args.cache_root)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    all_trading = trading_days_in(start, end, calendar)
    if not all_trading:
        print(f"No trading days in [{start}, {end}]", file=sys.stderr)
        return 1
    # Chunk into windows of `window_days` trading days each.
    windows: list[tuple[date, date]] = []
    i = 0
    while i < len(all_trading):
        j = min(i + args.window_days, len(all_trading))
        windows.append((all_trading[i], all_trading[j - 1]))
        i = j
    print(f"Mode C: {len(windows)} windows × {args.window_days} trading days", flush=True)

    cur_ckpt = args.seed_ckpt
    all_pnl_rows: list[dict] = []
    for w_idx, (w_start, w_end) in enumerate(windows):
        prev_td = calendar.prev_trading_day(w_start)
        if prev_td is None:
            print(f"[{w_idx}] no prev trading day before {w_start}; skipping", flush=True)
            continue
        win_tag = w_start.strftime("%Y%m%d")
        win_dir = args.log_dir / f"window_{win_tag}"
        win_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Window {w_idx+1}/{len(windows)}: train [{args.train_start}..{prev_td}] → val [{w_start}..{w_end}] ===", flush=True)

        # 1. Fine-tune (skip if ft_steps == 0 — static-per-window baseline)
        if args.ft_steps <= 0:
            print(f"[{w_idx}] ft_steps=0, skipping fine-tune; using seed ckpt for eval", flush=True)
            new_ckpt = cur_ckpt
            # Fall through to eval below using new_ckpt = cur_ckpt.
            ft_cmd = None  # sentinel
        else:
            ft_cmd = [
            sys.executable, "-m", "daity.scripts.train_cohort",
            "--mode", "C",
            "--universe-source", "alive_in_range",
            "--train-start", args.train_start,
            "--train-end", prev_td.isoformat(),
            "--val-start", w_start.isoformat(),
            "--val-end", w_end.isoformat(),
            "--d-model", "480", "--stock-enc-layers", "12",
            "--n-heads", "8", "--ffn-ratio", "4",
            "--n-market-layers", "4", "--n-cross-layers", "4",
            "--max-steps", str(args.ft_steps),
            "--lr", str(args.ft_lr), "--warmup-pct", str(args.ft_warmup),
            "--weight-decay", str(args.wd),
            "--val-every", str(max(50, args.ft_steps // 4)),
            "--ckpt-every", str(max(100, args.ft_steps // 2)),
            "--w-reg", "1.0", "--w-rank", "0.5", "--w-bias", "0.01",
            "--rank-top-k", str(args.rank_top_k),
            "--smooth-l1-beta", str(args.smooth_l1_beta),
            "--val-max-cohorts", "100",
            "--time-ordered", "--single-epoch",
            "--init-ckpt", str(cur_ckpt),
            "--log-dir", str(win_dir),
        ]
        if ft_cmd is not None and args.universe_end is not None:
            ft_cmd.extend(["--universe-end", args.universe_end])
        if args.wandb_project:
            ft_cmd.extend([
                "--wandb-project", args.wandb_project,
                "--wandb-run-name", f"modeC_window_{win_tag}",
                "--wandb-tags", "cohort,modeC,walk_forward",
            ])
        if ft_cmd is not None:
            print(" ".join(ft_cmd), flush=True)
            ret = subprocess.run(ft_cmd, check=False)
            if ret.returncode != 0:
                print(f"[{w_idx}] fine-tune FAILED rc={ret.returncode}", flush=True)
                continue

            # 2. Eval — find the best.pt of this window's training
            new_ckpt = win_dir / "checkpoints" / "best.pt"
            if not new_ckpt.exists():
                new_ckpt = win_dir / "checkpoints" / "last.pt"
            if not new_ckpt.exists():
                print(f"[{w_idx}] no checkpoint saved by trainer; falling back to prior ckpt", flush=True)
                new_ckpt = cur_ckpt

        eval_out = win_dir / "pnl.parquet"
        eval_cmd = [
            sys.executable, "-m", "daity.scripts.eval_cohort_pnl",
            "--ckpt", str(new_ckpt),
            "--cache-root", str(args.cache_root),
            "--feature-root", str(args.feature_root),
            "--val-start", w_start.isoformat(),
            "--val-end", w_end.isoformat(),
            "--anchor-ist", args.anchor_ist,
            "--horizon", args.horizon,
            "--top-k", str(args.top_k),
            "--cost-bps", str(args.cost_bps),
            "--out", str(eval_out),
        ]
        if args.universe_end is not None:
            eval_cmd.extend(["--universe-end", args.universe_end])
        ret = subprocess.run(eval_cmd, check=False)
        if ret.returncode != 0:
            print(f"[{w_idx}] eval FAILED rc={ret.returncode}", flush=True)
            cur_ckpt = new_ckpt
            continue

        # 3. Read this window's P&L and accumulate.
        if eval_out.exists():
            df = pl.read_parquet(eval_out)
            df = df.with_columns(pl.lit(w_idx).alias("window"))
            all_pnl_rows.extend(df.to_dicts())
            mean_bps = df["net_bps"].mean() if df.shape[0] > 0 else None
            print(f"[{w_idx}] window mean_bps/day = {mean_bps}", flush=True)

        cur_ckpt = new_ckpt

    # Aggregate
    if all_pnl_rows:
        out = pl.DataFrame(all_pnl_rows)
        out = out.sort("date").with_columns(
            pl.col("net_bps").cum_sum().alias("cum_bps"),
        )
        out_path = args.log_dir / "all_pnl.parquet"
        out.write_parquet(out_path)
        print(f"\n=== MODE C SUMMARY ===", flush=True)
        print(f"days = {out.shape[0]}", flush=True)
        print(f"total bps = {out['net_bps'].sum():.1f}", flush=True)
        print(f"mean bps/day = {out['net_bps'].mean():.2f}", flush=True)
        print(f"hit rate = {(out['win'].sum() / out.shape[0]) * 100:.1f}%", flush=True)
        import numpy as np
        sd = out["net_bps"].std() or 1e-9
        print(f"sharpe = {out['net_bps'].mean() / sd * np.sqrt(252):.2f}", flush=True)
        print(f"saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
