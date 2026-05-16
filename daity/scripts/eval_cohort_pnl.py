"""Evaluate a cohort checkpoint on a val period: per-day top-K P&L + IC.

For each (date, anchor=15:25 IST) in the val period:
  1. Assemble cohort, predict 10 horizons.
  2. Rank stocks cross-sectionally by the chosen horizon's prediction.
  3. Long the top-K stocks (equal weight).
  4. Realize the (horizon-specific) cumulative log return for those K stocks.
  5. Aggregate to per-day net P&L (bps) after transaction costs.

The output is:
  - A parquet of per-(date, symbol) records.
  - A summary printout: total bps, hit rate, win rate, mean per-day P&L,
    per-horizon spearman IC.
  - An optional PNG of cumulative P&L over the val period.

Usage:
    python -m daity.scripts.eval_cohort_pnl \
        --ckpt runs/cohort_modeA_v2/checkpoints/best.pt \
        --val-start 2024-12-01 --val-end 2024-12-31 \
        --horizon next_day_eod \
        --top-k 10 \
        --out reports/cohort_modeA_v2_pnl.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from daity.data.calendar_nse import NSECalendar
from daity.data.cohort_dataset import (
    CohortAssembler,
    DEFAULT_COHORT_CHANNELS,
    N_REGIME_FEATS,
)
from daity.data.intraday_horizons import (
    HORIZONS,
    HORIZON_NAME_TO_IDX,
    N_HORIZONS,
)
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster
from daity.models.cohort.cohort_module import CohortModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--val-start", type=str, required=True)
    ap.add_argument("--val-end",   type=str, required=True)
    ap.add_argument("--universe-end", type=str, default=None,
                    help="Use this date as the upper bound for the alive_in_range "
                         "filter (default: val-end). Set to the training run's "
                         "val_end to match the ckpt's stock_embed dimensions.")
    # Eval at one anchor per day (15:25 IST default = end-of-day) unless
    # overridden. For intraday horizons this should be early in the day.
    ap.add_argument("--anchor-ist", type=str, default="15:25")
    ap.add_argument("--horizon", type=str, default="next_day_eod",
                    choices=[h.name for h in HORIZONS],
                    help="Which horizon to backtest on.")
    ap.add_argument("--top-k",   type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=15.0,
                    help="Round-trip cost per name in bps (brokerage + slippage + stt).")
    # Model architecture must match the ckpt's training run.
    ap.add_argument("--d-model",          type=int, default=480)
    ap.add_argument("--stock-enc-layers", type=int, default=12)
    ap.add_argument("--n-heads",          type=int, default=8)
    ap.add_argument("--ffn-ratio",        type=int, default=4)
    ap.add_argument("--n-market-layers",  type=int, default=4)
    ap.add_argument("--n-cross-layers",   type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    master = SymbolMaster.from_cache(args.cache_root)
    all_syms = sorted(master.frame["symbol"].to_list())
    val_end_dt = datetime.fromisoformat(args.val_end).replace(tzinfo=UTC)
    as_of = val_end_dt + timedelta(days=14)
    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in all_syms
    })
    sector_to_id = {s: i for i, s in enumerate(all_sectors)}
    # Stock IDs must match training: alive in the train+val range.
    # For simplicity we use the same alive-in-range filter as the trainer.
    t_start = datetime(2019, 1, 1, tzinfo=UTC)
    if args.universe_end is not None:
        univ_end_dt = datetime.fromisoformat(args.universe_end).replace(tzinfo=UTC)
    else:
        univ_end_dt = val_end_dt
    day_root = args.feature_root / "day"
    alive = []
    for s in all_syms:
        try:
            df = pl.read_parquet(day_root / f"{s}.parquet", columns=["ts"])
        except Exception:
            continue
        if df.height == 0:
            continue
        ts = df["ts"]
        if (ts >= t_start).any() and (ts <= univ_end_dt).any():
            alive.append(s)
    universe = tuple(alive)
    sector_id_by_symbol = {
        s: sector_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }
    print(f"universe: {len(universe)} stocks, {len(all_sectors)} sectors", flush=True)

    calendar = NSECalendar.from_cache(args.cache_root)
    store = ParquetStore(args.feature_root)
    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe,
        sector_id_by_symbol=sector_id_by_symbol,
        as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )

    # Build model + load ckpt.
    model = CohortModel(
        scales=("5m", "15m", "60m", "day"),
        num_channels=len(DEFAULT_COHORT_CHANNELS),
        d_model=args.d_model,
        stock_enc_layers=args.stock_enc_layers,
        n_heads=args.n_heads,
        ffn_ratio=args.ffn_ratio,
        n_stocks=len(universe),
        n_sectors=len(all_sectors),
        n_market_layers=args.n_market_layers,
        n_cross_layers=args.n_cross_layers,
        market_n_heads=args.n_heads,
        cross_n_heads=args.n_heads,
        n_regime_feats=N_REGIME_FEATS,
    ).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    m_sd = sd.get("model_state_dict", sd)
    missing, unexpected = model.load_state_dict(m_sd, strict=False)
    print(f"loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # Anchor time
    h_str, m_str = args.anchor_ist.split(":")
    anchor_ist = dtime(int(h_str), int(m_str))
    h_idx = HORIZON_NAME_TO_IDX[args.horizon]

    # Iterate val period.
    val_start = date.fromisoformat(args.val_start)
    val_end   = date.fromisoformat(args.val_end)
    rows: list[dict] = []
    per_horizon_p: list[list[float]] = [[] for _ in range(N_HORIZONS)]
    per_horizon_t: list[list[float]] = [[] for _ in range(N_HORIZONS)]
    d = val_start
    while d <= val_end:
        if not calendar.is_trading_day(d):
            d = d + timedelta(days=1); continue
        anchor_utc = (datetime.combine(d, anchor_ist)
                      - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)
        sample = assembler.assemble(anchor_utc)
        if sample is None:
            d = d + timedelta(days=1); continue
        batch = {
            "x_by_scale": {sc: v.unsqueeze(0).to(device)
                           for sc, v in sample.x_by_scale.items()},
            "stock_ids":  sample.stock_ids.unsqueeze(0).to(device),
            "sector_ids": sample.sector_ids.unsqueeze(0).to(device),
            "anchor_ts":  torch.tensor(
                [int(anchor_utc.timestamp() * 1_000_000)],
                dtype=torch.int64, device=device,
            ),
            "regime_feats": sample.regime_feats.unsqueeze(0).to(device),
        }
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            pred = model(batch)                              # (1, N, H)
        pred = pred[0].float().cpu().numpy()                 # (N, H)
        target = sample.labels.cpu().numpy()                 # (N, H)
        label_v = sample.label_validity_per_stock.cpu().numpy()
        # IC accumulation per horizon
        for h in range(N_HORIZONS):
            mask = label_v[:, h]
            if mask.sum() >= 5:
                per_horizon_p[h].extend(pred[mask, h].tolist())
                per_horizon_t[h].extend(target[mask, h].tolist())
        # Strategy: top-K by predicted alpha on the chosen horizon, only
        # among stocks with valid labels for that horizon.
        h_mask = label_v[:, h_idx]
        if h_mask.sum() < args.top_k:
            d = d + timedelta(days=1); continue
        p = pred[h_mask, h_idx]
        t_arr = target[h_mask, h_idx]
        syms = [sample.symbols[i] for i, ok in enumerate(h_mask) if ok]
        # Top-K by pred (highest predicted log return = strongest long signal).
        topk = np.argsort(-p)[: args.top_k]
        chosen = [syms[i] for i in topk]
        chosen_lr = float(np.mean(t_arr[topk]))                        # log return
        bps = (np.exp(chosen_lr) - 1.0) * 10000.0 - args.cost_bps      # net bps
        rows.append({
            "date": d,
            "anchor_ist": args.anchor_ist,
            "horizon": args.horizon,
            "n_alive": int(h_mask.sum()),
            "top_k": args.top_k,
            "chosen": ",".join(chosen),
            "mean_logret_top_k": chosen_lr,
            "net_bps": float(bps),
            "win": int(bps > 0),
        })
        d = d + timedelta(days=1)

    if not rows:
        print("NO RESULTS — likely no val anchors / labels", flush=True)
        return 2

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("net_bps").cum_sum().alias("cum_bps"))
    df.write_parquet(args.out)
    print(f"saved -> {args.out} rows={df.shape[0]}", flush=True)
    print("\n=== Summary ===", flush=True)
    print(f"days:           {df.shape[0]}", flush=True)
    print(f"horizon:        {args.horizon}", flush=True)
    print(f"top_k:          {args.top_k}", flush=True)
    print(f"cost_bps:       {args.cost_bps}", flush=True)
    print(f"total bps:      {df['net_bps'].sum():.2f}", flush=True)
    print(f"mean bps/day:   {df['net_bps'].mean():.2f}", flush=True)
    print(f"median bps/day: {df['net_bps'].median():.2f}", flush=True)
    print(f"hit rate:       {(df['win'].sum() / df.shape[0]) * 100:.1f}%", flush=True)
    print(f"sharpe (daily): {df['net_bps'].mean() / max(1e-9, df['net_bps'].std()) * np.sqrt(252):.2f}", flush=True)

    # IC table
    print("\n=== Per-horizon Spearman IC ===", flush=True)
    for h_idx2, h in enumerate(HORIZONS):
        p = np.asarray(per_horizon_p[h_idx2])
        t_arr = np.asarray(per_horizon_t[h_idx2])
        if len(p) < 50:
            print(f"  {h.name:>15}: n={len(p):>6}  (insufficient)", flush=True)
            continue
        ic, _ = spearmanr(p, t_arr)
        print(f"  {h.name:>15}: n={len(p):>6}  IC={ic:+.4f}", flush=True)

    # Optional plot
    if args.out_png is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            dates = df["date"].to_list()
            axes[0].bar(dates, df["net_bps"].to_list(), color=[
                "tab:green" if b > 0 else "tab:red" for b in df["net_bps"].to_list()
            ])
            axes[0].axhline(0, color="black", lw=0.5)
            axes[0].set_ylabel("net bps / day")
            axes[0].set_title(
                f"Cohort {args.horizon} top-{args.top_k} | "
                f"cost={args.cost_bps}bps | total={df['net_bps'].sum():.1f}bps | "
                f"hit={(df['win'].sum() / df.shape[0]) * 100:.1f}%"
            )
            axes[1].plot(dates, df["cum_bps"].to_list(), lw=2)
            axes[1].axhline(0, color="black", lw=0.5)
            axes[1].set_ylabel("cumulative bps")
            axes[1].set_xlabel("date")
            plt.tight_layout()
            plt.savefig(args.out_png, dpi=120)
            print(f"saved plot -> {args.out_png}", flush=True)
        except Exception as e:
            print(f"plot skipped: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
