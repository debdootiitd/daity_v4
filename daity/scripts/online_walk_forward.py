"""Online single-process daily walk-forward training + prediction.

Designed to avoid the cumulative-drift problem of `run_mode_c.py` (which
respawned a fresh subprocess + fresh optimizer for every window). Here the
model + optimizer state live in one process across all days.

Key design choices for the new (2026-05-16) experiment:

  * Anchor: 10:15 IST per trading day so intraday horizons stay valid.
  * Active horizons: the first 8 of N_HORIZONS=10 — drop day_plus_3 and
    day_plus_5 (model stale by ≥3 days makes those targets misleading).
  * Label lag: train on day D-2's anchor when evaluating day D. By D-2 EOD
    + 1 trading day = D-1 EOD, all of D-2's labels (intraday → next_day_eod)
    are fully realized.
  * Optimizer: Adagrad by default — per-parameter accumulator auto-decays
    LR over many online steps, which directly addresses Mode B's drift.
  * State persistence: optimizer state lives in-process across every day so
    Adagrad's accumulator carries over.
  * Optional batch warmup: run a standard Mode-A-style train on
    `[warmup_start, warmup_end]` before flipping to online mode. Use the
    same optimizer instance (state persists into the online phase).

Each day:
  1. assemble cohort at the day's 10:15-IST anchor.
  2. forward → predictions (no grad). store per-(day, stock, horizon).
  3. assemble cohort at (D − label_lag_trading_days)'s anchor.
  4. forward + backward + optimizer step on that cohort (1+ steps).
  5. occasional checkpoint.

Output:
  - `metrics.jsonl`: per-day record (anchor_ts, pred mean/std, train loss).
  - `predictions.parquet`: per-(day, stock, horizon) predicted log return.
  - `checkpoints/`: periodic snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from daity.data.calendar_nse import NSECalendar
from daity.data.cohort_dataset import (
    CohortAssembler,
    DEFAULT_COHORT_CHANNELS,
    N_REGIME_FEATS,
)
from daity.data.intraday_horizons import HORIZONS, N_HORIZONS
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster
from daity.models.cohort.cohort_loss import CohortLoss
from daity.models.cohort.cohort_module import CohortModel

# Horizons we ACT on (drop day_plus_3 and day_plus_5 — too stale for daily
# fine-tune). Indices into HORIZONS / model output.
ACTIVE_HORIZON_NAMES: tuple[str, ...] = (
    "intraday_30m", "intraday_60m", "intraday_120m", "intraday_180m",
    "to_close", "overnight", "next_day_1h", "next_day_eod",
)
ACTIVE_HORIZON_INDICES: tuple[int, ...] = tuple(
    i for i, h in enumerate(HORIZONS) if h.name in ACTIVE_HORIZON_NAMES
)

# Default anchor — 10:15 IST keeps intraday horizons valid.
DEFAULT_ANCHOR_IST = dtime(10, 15)


def _ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-ckpt", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    # Universe (must match seed ckpt's stock_embed)
    ap.add_argument("--universe-end", type=str, default="2024-12-31")
    # Online phase window
    ap.add_argument("--online-start", type=str, required=True)
    ap.add_argument("--online-end",   type=str, required=True)
    # Optional batch warmup before online phase
    ap.add_argument("--warmup-start", type=str, default=None)
    ap.add_argument("--warmup-end",   type=str, default=None)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    # Online schedule
    ap.add_argument("--label-lag-trading-days", type=int, default=2)
    ap.add_argument("--steps-per-day", type=int, default=1)
    ap.add_argument("--anchor-ist", type=str, default="10:15")
    # Optimizer
    ap.add_argument("--optimizer", choices=["adamw", "adagrad"], default="adagrad")
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight-decay",  type=float, default=0.0)
    ap.add_argument("--grad-clip",     type=float, default=1.0)
    # Loss
    ap.add_argument("--w-reg",          type=float, default=1.0)
    ap.add_argument("--w-rank",         type=float, default=0.5)
    ap.add_argument("--w-bias",         type=float, default=0.01)
    ap.add_argument("--smooth-l1-beta", type=float, default=0.005)
    ap.add_argument("--rank-top-k",     type=int,   default=20)
    # Model arch (must match seed ckpt)
    ap.add_argument("--d-model",          type=int, default=480)
    ap.add_argument("--stock-enc-layers", type=int, default=1)
    ap.add_argument("--n-heads",          type=int, default=8)
    ap.add_argument("--ffn-ratio",        type=int, default=4)
    ap.add_argument("--n-market-layers",  type=int, default=4)
    ap.add_argument("--n-cross-layers",   type=int, default=4)
    # IO
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--ckpt-every", type=int, default=50,
                    help="Save full ckpt every N online days.")
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    return ap.parse_args()


def _trading_days(start: date, end: date, cal: NSECalendar) -> list[date]:
    out = []
    d = start
    while d <= end:
        if cal.is_trading_day(d):
            out.append(d)
        d = d + timedelta(days=1)
    return out


def _build_universe(args, master: SymbolMaster) -> tuple[tuple[str, ...], dict[str, int], list[str]]:
    all_syms = sorted(master.frame["symbol"].to_list())
    univ_end = datetime.fromisoformat(args.universe_end).replace(tzinfo=UTC)
    t_start = datetime(2019, 1, 1, tzinfo=UTC)
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
        if (ts >= t_start).any() and (ts <= univ_end).any():
            alive.append(s)
    universe = tuple(alive)
    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in all_syms
    })
    sec_to_id = {s: i for i, s in enumerate(all_sectors)}
    sec_by = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }
    return universe, sec_by, all_sectors


def _build_model(args, n_stocks: int, n_sectors: int, device: str) -> CohortModel:
    return CohortModel(
        scales=("5m", "15m", "60m", "day"),
        num_channels=len(DEFAULT_COHORT_CHANNELS),
        d_model=args.d_model,
        stock_enc_layers=args.stock_enc_layers,
        n_heads=args.n_heads,
        ffn_ratio=args.ffn_ratio,
        n_stocks=n_stocks,
        n_sectors=n_sectors,
        n_market_layers=args.n_market_layers,
        n_cross_layers=args.n_cross_layers,
        market_n_heads=args.n_heads,
        cross_n_heads=args.n_heads,
        n_regime_feats=N_REGIME_FEATS,
    ).to(device)


def _sample_to_batch(sample, anchor_utc: datetime, device: str) -> dict:
    return {
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


def _active_validity(sample, device: str) -> torch.Tensor:
    """Validity mask restricted to ACTIVE horizons (drop d3, d5)."""
    validity = sample.validity.clone()
    for i in range(N_HORIZONS):
        if i not in ACTIVE_HORIZON_INDICES:
            validity[i] = False
    return validity.unsqueeze(0).to(device)


def _unpack_pred(model_out) -> torch.Tensor:
    if isinstance(model_out, dict):
        return model_out["pred"]
    if isinstance(model_out, tuple):
        return model_out[0]
    return model_out


def _shift_trading_days(d: date, n: int, cal: NSECalendar) -> date | None:
    """Return the trading day n trading days BEFORE d (n>0 → past)."""
    cur = d
    for _ in range(n):
        prv = cal.prev_trading_day(cur)
        if prv is None:
            return None
        cur = prv
    return cur


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.log_dir / "metrics.jsonl"
    preds_path = args.log_dir / "predictions.parquet"
    h_ist, m_ist = (int(x) for x in args.anchor_ist.split(":"))
    anchor_ist = dtime(h_ist, m_ist)

    # Universe + sectors
    master = SymbolMaster.from_cache(args.cache_root)
    universe, sec_by, all_sectors = _build_universe(args, master)
    print(f"universe: {len(universe)} | sectors: {len(all_sectors)} | "
          f"anchor: {anchor_ist} IST", flush=True)

    calendar = NSECalendar.from_cache(args.cache_root)
    store = ParquetStore(args.feature_root)
    # as_of needs to extend far enough past online_end for next_day_eod labels.
    online_end_dt = datetime.fromisoformat(args.online_end).replace(tzinfo=UTC)
    as_of = online_end_dt + timedelta(days=10)
    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe, sector_id_by_symbol=sec_by, as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )

    model = _build_model(args, len(universe), len(all_sectors), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params", flush=True)

    # Load seed ckpt
    print(f"loading seed: {args.seed_ckpt}", flush=True)
    sd = torch.load(args.seed_ckpt, map_location=device, weights_only=False)
    m_sd = sd.get("cohort_init_state_dict", sd.get("model_state_dict", sd))
    missing, unexpected = model.load_state_dict(m_sd, strict=False)
    print(f"  loaded | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # Optimizer (persistent across days)
    if args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

    loss_fn = CohortLoss(
        w_reg=args.w_reg, w_rank=args.w_rank, w_bias=args.w_bias,
        smooth_l1_beta=args.smooth_l1_beta, rank_top_k=args.rank_top_k,
    )

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb
            wandb_run.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.log_dir.name,
                dir=str(args.log_dir),
                config=vars(args),
            )
        except Exception as e:
            print(f"[wandb disabled: {e}]", flush=True)
            wandb_run = None

    def log_event(ev: dict) -> None:
        ev = {"ts": datetime.now(UTC).isoformat(), **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        if wandb_run is not None:
            payload = {k: v for k, v in ev.items() if isinstance(v, int | float)}
            wandb_run.log(payload)

    log_event({"event": "fit_start", "n_params": n_params,
               "optimizer": args.optimizer, "lr": args.lr,
               "online_start": args.online_start, "online_end": args.online_end,
               "warmup_start": args.warmup_start, "warmup_end": args.warmup_end,
               "active_horizons": list(ACTIVE_HORIZON_NAMES)})

    # ----- (Optional) Batch warmup phase -----
    if args.warmup_start and args.warmup_end:
        from daity.data.intraday_horizons import intraday_anchor_grid
        warm_days = _trading_days(
            date.fromisoformat(args.warmup_start),
            date.fromisoformat(args.warmup_end),
            calendar,
        )
        warm_anchors_ist = intraday_anchor_grid()  # 11 anchors/day
        anchor_pairs = [(d, t) for d in warm_days for t in warm_anchors_ist]
        # Time-ordered iteration.
        anchor_pairs.sort()
        print(f"warmup: {len(warm_days)} days × {len(warm_anchors_ist)} anchors = "
              f"{len(anchor_pairs)} cohorts. max_steps={args.warmup_steps}", flush=True)
        rng_idx = 0
        warmup_steps_done = 0
        t0 = time.time()
        model.train()
        while warmup_steps_done < args.warmup_steps and rng_idx < len(anchor_pairs):
            d, t = anchor_pairs[rng_idx]
            rng_idx += 1
            anchor_utc = _ist_to_utc(d, t)
            sample = assembler.assemble(anchor_utc)
            if sample is None:
                continue
            batch = _sample_to_batch(sample, anchor_utc, device)
            labels = sample.labels.unsqueeze(0).to(device)
            label_v = sample.label_validity_per_stock.unsqueeze(0).to(device)
            labels_clean = torch.where(label_v, labels, torch.zeros_like(labels))
            validity = _active_validity(sample, device)
            validity = validity & (label_v.float().mean(dim=1) > 0.8)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                model_out = model(batch)
                pred = _unpack_pred(model_out)
                out = loss_fn(pred, labels_clean, validity)
            out["total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            warmup_steps_done += 1
            if warmup_steps_done % 200 == 0:
                log_event({"event": "warmup_step", "step": warmup_steps_done,
                           "train_loss": float(out["total"])})
                print(f"  warmup step={warmup_steps_done} loss={float(out['total']):.4f}",
                      flush=True)
        log_event({"event": "warmup_end", "steps": warmup_steps_done,
                   "elapsed_sec": time.time() - t0})

    # ----- Online phase -----
    online_days = _trading_days(
        date.fromisoformat(args.online_start),
        date.fromisoformat(args.online_end),
        calendar,
    )
    print(f"online phase: {len(online_days)} trading days from "
          f"{args.online_start} to {args.online_end}", flush=True)

    pred_rows: list[dict] = []
    t0 = time.time()
    n_predicted = 0
    n_trained = 0
    for di, d in enumerate(online_days):
        anchor_utc = _ist_to_utc(d, anchor_ist)
        # 1. Predict for day d (no grad)
        model.eval()
        sample = assembler.assemble(anchor_utc)
        if sample is not None:
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                pred = _unpack_pred(model(_sample_to_batch(sample, anchor_utc, device)))
            pred_np = pred[0].float().cpu().numpy()                       # (N, H)
            labels_np = sample.labels.cpu().numpy()
            label_v_np = sample.label_validity_per_stock.cpu().numpy()
            for i, sym in enumerate(sample.symbols):
                for h_idx in ACTIVE_HORIZON_INDICES:
                    if not label_v_np[i, h_idx]:
                        continue
                    pred_rows.append({
                        "date": d,
                        "anchor_us": int(anchor_utc.timestamp() * 1_000_000),
                        "stock": sym,
                        "horizon": HORIZONS[h_idx].name,
                        "pred_lr": float(pred_np[i, h_idx]),
                        "real_lr": float(labels_np[i, h_idx]),
                    })
            n_predicted += 1

        # 2. Train on D - label_lag (whose labels are realized by now)
        train_day = _shift_trading_days(d, args.label_lag_trading_days, calendar)
        if train_day is not None and train_day >= date.fromisoformat(args.online_start):
            train_anchor_utc = _ist_to_utc(train_day, anchor_ist)
            train_sample = assembler.assemble(train_anchor_utc)
            if train_sample is not None:
                t_batch = _sample_to_batch(train_sample, train_anchor_utc, device)
                labels = train_sample.labels.unsqueeze(0).to(device)
                label_v = train_sample.label_validity_per_stock.unsqueeze(0).to(device)
                labels_clean = torch.where(label_v, labels, torch.zeros_like(labels))
                validity = _active_validity(train_sample, device)
                validity = validity & (label_v.float().mean(dim=1) > 0.8)
                model.train()
                for _step in range(args.steps_per_day):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        model_out = model(t_batch)
                        pred_t = _unpack_pred(model_out)
                        out = loss_fn(pred_t, labels_clean, validity)
                    out["total"].backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip,
                        )
                    optimizer.step()
                n_trained += 1
                if di % 25 == 0:
                    log_event({
                        "event": "online_day",
                        "day": str(d),
                        "train_day": str(train_day),
                        "train_loss": float(out["total"]),
                        "n_predicted": n_predicted,
                        "n_trained": n_trained,
                    })

        # 3. Periodic checkpoint
        if (di + 1) % args.ckpt_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "day": str(d), "day_idx": di + 1,
            }, ckpt_dir / f"day_{d.isoformat()}.pt")

    # Final ckpt + predictions parquet
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "day": str(online_days[-1]) if online_days else None,
        "day_idx": len(online_days),
    }, ckpt_dir / "last.pt")

    if pred_rows:
        pl.DataFrame(pred_rows).write_parquet(preds_path)

    log_event({"event": "fit_end", "elapsed_sec": time.time() - t0,
               "n_predicted_days": n_predicted, "n_trained_days": n_trained,
               "n_pred_rows": len(pred_rows)})
    print(f"DONE | predicted_days={n_predicted} trained_days={n_trained} "
          f"pred_rows={len(pred_rows)} elapsed={time.time()-t0:.1f}s", flush=True)

    if wandb_run is not None:
        try: wandb_run.finish()
        except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
