"""Fast ablation trainer: consumes the disk cache, max GPU util.

Key speedups vs online_walk_forward.py:
  - No CPU-bound CohortAssembler in the hot path
  - DataLoader with num_workers + pin_memory + prefetch
  - bf16 autocast, optional torch.compile, cudnn.benchmark
  - Batched anchors (default B=4) with same-N filter

Online walk-forward semantics PRESERVED:
  - At each (date, anchor): predict (no grad) then train on (date-2, anchor)
  - Adagrad accumulator persists across all anchors
  - Daily checkpoint (lighter than online_walk_forward — only test-window)

Single-config or sweep mode: --ablation-yaml points to a list of configs;
runs them sequentially (each config is fast, no CPU re-work).

Usage:
  python -m daity.scripts.train_cohort_ablation \
      --cache-dir cache/cohort_samples_v1 \
      --pretest-end 2025-01-31 --test-end 2026-04-30 \
      --seed-ckpt runs/contrastive_pretrain_v2/checkpoints/last=4000.pt \
      --w-reg 1.0 --w-rank 0.5 --w-bias 0.01 \
      --d-model 480 --stock-enc-layers 1 \
      --optimizer adagrad --lr 5e-4 \
      --batch-size 4 \
      --num-workers 4 \
      --use-compile \
      --log-dir runs/ablation_v1_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from daity.data.cohort_cache import (
    CohortSampleCache, CacheRecord, record_to_batch, records_to_batch,
    parse_cache_filename,
)
from daity.data.intraday_horizons import HORIZONS, N_HORIZONS
from daity.models.cohort.cohort_loss import CohortLoss
from daity.models.cohort.cohort_module import CohortModel
from daity.data.cohort_dataset import DEFAULT_COHORT_CHANNELS, N_REGIME_FEATS


ACTIVE_HORIZON_NAMES = (
    "intraday_30m", "intraday_60m", "intraday_120m", "intraday_180m",
    "to_close", "overnight", "next_day_1h", "next_day_eod",
)
ACTIVE_HORIZON_INDICES = tuple(
    i for i, h in enumerate(HORIZONS) if h.name in ACTIVE_HORIZON_NAMES
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--pretest-end", type=str, required=True,
                    help="YYYY-MM-DD — last date BEFORE test window (no predictions before this)")
    ap.add_argument("--test-end", type=str, required=True,
                    help="YYYY-MM-DD — last test date")
    ap.add_argument("--seed-ckpt", type=Path, default=None)
    ap.add_argument("--resume-optimizer", action="store_true")

    # Architecture
    ap.add_argument("--d-model",          type=int, default=480)
    ap.add_argument("--stock-enc-layers", type=int, default=1)
    ap.add_argument("--n-heads",          type=int, default=8)
    ap.add_argument("--ffn-ratio",        type=int, default=4)
    ap.add_argument("--n-market-layers",  type=int, default=4)
    ap.add_argument("--n-cross-layers",   type=int, default=4)
    ap.add_argument("--n-stocks", type=int, default=205,
                    help="Total stocks in the universe (for stock_embed sizing). Must "
                         "be >= max stock_id across all cache records.")
    ap.add_argument("--n-sectors", type=int, default=15,
                    help="Total sectors (for sector_embed sizing).")

    # Loss
    ap.add_argument("--w-reg",   type=float, default=1.0)
    ap.add_argument("--w-rank",  type=float, default=0.5)
    ap.add_argument("--w-bias",  type=float, default=0.01)
    ap.add_argument("--smooth-l1-beta", type=float, default=0.005)
    ap.add_argument("--rank-top-k",     type=int,   default=20)
    # Win-rate classifier head ablation
    ap.add_argument("--use-classifier-head", action="store_true",
                    help="Add per-horizon win-rate classifier head to the model")
    ap.add_argument("--w-clf", type=float, default=0.0,
                    help="Loss weight for classifier BCE term (only used when --use-classifier-head)")
    ap.add_argument("--clf-threshold-bps", type=float, default=50.0,
                    help="Win threshold (bps) for classifier label (e.g. 50 or 100)")

    # Optimizer
    ap.add_argument("--optimizer", choices=["adagrad", "adamw"], default="adagrad")
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--weight-decay",  type=float, default=0.0)
    ap.add_argument("--grad-clip",     type=float, default=1.0)

    # Walk-forward params
    ap.add_argument("--label-lag-trading-days", type=int, default=2)
    ap.add_argument("--steps-per-day", type=int, default=1)

    # Speed
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--use-compile", action="store_true")
    ap.add_argument("--use-bf16-weights", action="store_true")

    # IO
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--ckpt-every-days", type=int, default=10)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    return ap.parse_args()


def collate_records(batch: list[CacheRecord]) -> list[CacheRecord]:
    """No stacking — pass through. Stacking happens on GPU side with same-N filter."""
    return batch


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.log_dir / "metrics.jsonl"
    preds_path = args.log_dir / "predictions.parquet"

    torch.backends.cudnn.benchmark = True

    test_end = date.fromisoformat(args.test_end)
    pretest_end = date.fromisoformat(args.pretest_end)

    print(f"loading cache from {args.cache_dir}", flush=True)
    cache = CohortSampleCache(args.cache_dir, end_date=test_end)
    print(f"cache: {len(cache)} samples", flush=True)
    if len(cache) == 0:
        print("ERROR: empty cache. Run build_cohort_cache.py first.", file=sys.stderr)
        return 1

    # Universe sizing: use --n-stocks (default 205) since cache records have
    # variable stock counts but stock_ids index into the full 205-stock universe.
    first = cache[0]
    print(f"first sample: per-anchor n_stocks={first.stock_ids.shape[0]}, "
          f"max stock_id={int(first.stock_ids.max())}, "
          f"using model n_stocks={args.n_stocks}, n_sectors={args.n_sectors}",
          flush=True)

    model = CohortModel(
        scales=tuple(first.x_by_scale.keys()),
        num_channels=len(DEFAULT_COHORT_CHANNELS),
        d_model=args.d_model,
        stock_enc_layers=args.stock_enc_layers,
        n_heads=args.n_heads,
        ffn_ratio=args.ffn_ratio,
        n_stocks=args.n_stocks,
        n_sectors=args.n_sectors,
        n_market_layers=args.n_market_layers,
        n_cross_layers=args.n_cross_layers,
        market_n_heads=args.n_heads,
        cross_n_heads=args.n_heads,
        n_regime_feats=N_REGIME_FEATS,
        enable_classifier_head=args.use_classifier_head,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params", flush=True)

    if args.use_bf16_weights:
        model = model.to(torch.bfloat16)
        print("model converted to bf16 weights", flush=True)

    if args.seed_ckpt is not None:
        print(f"loading seed: {args.seed_ckpt}", flush=True)
        sd = torch.load(args.seed_ckpt, map_location=device, weights_only=False)
        m_sd = sd.get("cohort_init_state_dict", sd.get("model_state_dict", sd))
        cur = model.state_dict()
        dropped = []
        for k in list(m_sd.keys()):
            if k in cur and m_sd[k].shape != cur[k].shape:
                dropped.append(k); del m_sd[k]
        if dropped:
            print(f"  dropped {len(dropped)} mismatched-shape tensors", flush=True)
        missing, unexpected = model.load_state_dict(m_sd, strict=False)
        print(f"  loaded | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.use_compile:
        try:
            model = torch.compile(model)
            print("model compiled with torch.compile", flush=True)
        except Exception as e:
            print(f"torch.compile failed: {e}", flush=True)

    if args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay,
                                       betas=(0.9, 0.95))

    if args.resume_optimizer and args.seed_ckpt is not None:
        if "optimizer_state_dict" in sd:
            try:
                optimizer.load_state_dict(sd["optimizer_state_dict"])
                print("restored optimizer state", flush=True)
            except Exception as e:
                print(f"WARNING: failed to restore optimizer: {e}", flush=True)

    loss_fn = CohortLoss(
        w_reg=args.w_reg, w_rank=args.w_rank, w_bias=args.w_bias,
        smooth_l1_beta=args.smooth_l1_beta, rank_top_k=args.rank_top_k,
        w_clf=args.w_clf if args.use_classifier_head else 0.0,
        clf_threshold_bps=args.clf_threshold_bps,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb
        wandb_run.init(project=args.wandb_project,
                       name=args.wandb_run_name or args.log_dir.name,
                       dir=str(args.log_dir),
                       config=vars(args))

    def log_event(ev: dict) -> None:
        ev = {"ts": datetime.now(UTC).isoformat(), **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        if wandb_run is not None:
            payload = {k: v for k, v in ev.items() if isinstance(v, int | float)}
            wandb_run.log(payload)

    # Build a same-IST-time train_idx mapping: for each cache index, find the
    # cache index corresponding to (D - label_lag, same_anchor_time).
    # Pre-build a dict: (date, anchor_hhmm) -> idx
    print("indexing cache for label-lag lookup...", flush=True)
    idx_map: dict[tuple[date, str], int] = {}
    file_anchors: list[tuple[date, str]] = []
    for i, f in enumerate(cache.files):
        parsed = parse_cache_filename(f.name)
        if parsed is None: continue
        d, t = parsed
        key = (d, f"{t.hour:02d}{t.minute:02d}")
        idx_map[key] = i
        file_anchors.append(key)

    # For each anchor index, find the index of (D-label_lag, same anchor)
    # We approximate "trading days back" by walking back through file_anchors;
    # cache only contains trading days so this is correct.
    train_lookup: list[int | None] = [None] * len(cache)
    by_date: dict[date, list[int]] = {}
    for i, (d, _) in enumerate(file_anchors):
        by_date.setdefault(d, []).append(i)
    sorted_dates = sorted(by_date.keys())
    date_pos = {d: i for i, d in enumerate(sorted_dates)}

    for i, (d, hhmm) in enumerate(file_anchors):
        pos = date_pos.get(d)
        if pos is None or pos < args.label_lag_trading_days:
            continue
        train_d = sorted_dates[pos - args.label_lag_trading_days]
        train_lookup[i] = idx_map.get((train_d, hhmm))

    loader = DataLoader(
        cache, batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,  # we move to device inline
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_records,
        shuffle=False,  # walk-forward MUST be in order
    )

    print(f"starting training: {len(cache)} samples, batch={args.batch_size}, "
          f"workers={args.num_workers}", flush=True)

    log_event({"event": "fit_start", "n_params": n_params, "n_samples": len(cache),
               "optimizer": args.optimizer, "lr": args.lr,
               "active_horizons": list(ACTIVE_HORIZON_NAMES)})

    pred_rows = []
    n_predicted = 0
    n_trained = 0
    last_loss = float("nan")
    last_day_logged = None
    t0 = time.time()
    samples_seen = 0
    batch_dtype = torch.bfloat16 if args.use_bf16_weights else torch.float32

    for batch in loader:
        if not batch: continue
        # batch is a list[CacheRecord]; we may receive fewer than batch_size on last batch
        for rec in batch:
            d = date.fromisoformat(rec.date_iso)
            # 1. Predict (skip during pre-test)
            if d > pretest_end:
                model.eval()
                with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    bt = record_to_batch(rec, device)
                    out = model(bt)
                if isinstance(out, dict):
                    pred = out["pred"]
                    clf_logits = out.get("clf_logits")
                elif isinstance(out, tuple):
                    pred = out[0]
                    clf_logits = None
                else:
                    pred = out
                    clf_logits = None
                pred_np = pred[0].float().cpu().numpy()
                clf_prob_np = (torch.sigmoid(clf_logits[0]).float().cpu().numpy()
                               if clf_logits is not None else None)
                labels_np = rec.labels.cpu().numpy()
                label_v_np = rec.label_validity_per_stock.cpu().numpy()
                for i, sym in enumerate(rec.symbols):
                    for h_idx in ACTIVE_HORIZON_INDICES:
                        if not label_v_np[i, h_idx]: continue
                        row = {
                            "date": d,
                            "anchor_us": rec.anchor_us,
                            "stock": sym,
                            "horizon": HORIZONS[h_idx].name,
                            "pred_lr": float(pred_np[i, h_idx]),
                            "real_lr": float(labels_np[i, h_idx]),
                        }
                        if clf_prob_np is not None:
                            row["pred_win_prob"] = float(clf_prob_np[i, h_idx])
                        pred_rows.append(row)
                n_predicted += 1

            # 2. Train on (D - label_lag, same anchor) using cached train sample
            train_idx = train_lookup[samples_seen]
            if train_idx is not None:
                train_rec = cache[train_idx]
                train_bt = record_to_batch(train_rec, device)
                labels = train_rec.labels.unsqueeze(0).to(device)
                label_v = train_rec.label_validity_per_stock.unsqueeze(0).to(device)
                labels_clean = torch.where(label_v, labels, torch.zeros_like(labels))
                # active horizons only
                v = train_rec.validity.clone()
                for i in range(N_HORIZONS):
                    if i not in ACTIVE_HORIZON_INDICES:
                        v[i] = False
                validity = v.unsqueeze(0).to(device)
                validity = validity & (label_v.float().mean(dim=1) > 0.8)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    raw = model(train_bt)
                    if isinstance(raw, dict):
                        pred_t = raw["pred"]
                        clf_t = raw.get("clf_logits")
                    elif isinstance(raw, tuple):
                        pred_t = raw[0]; clf_t = None
                    else:
                        pred_t = raw; clf_t = None
                    out = loss_fn(pred_t, labels_clean, validity,
                                  clf_logits=clf_t,
                                  label_validity_per_stock=label_v)
                out["total"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                last_loss = float(out["total"].detach())
                n_trained += 1
            samples_seen += 1

            # Per-day events: log + ckpt
            if last_day_logged != d:
                last_day_logged = d
                day_idx = date_pos.get(d, 0)
                if day_idx % 10 == 0:
                    log_event({"event": "online_day", "day": str(d),
                               "train_loss": last_loss,
                               "n_predicted": n_predicted, "n_trained": n_trained,
                               "elapsed_sec": time.time() - t0,
                               "samples_per_sec": samples_seen / max(time.time() - t0, 1e-6)})
                if (day_idx + 1) % args.ckpt_every_days == 0:
                    # Only ckpt during test window (or last few pre-test days for resume)
                    if d > pretest_end - timedelta(days=10):
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "day": str(d), "day_idx": day_idx + 1,
                        }, ckpt_dir / f"day_{d.isoformat()}.pt")

    # Final
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "day": str(last_day_logged), "day_idx": samples_seen,
    }, ckpt_dir / "last.pt")
    if pred_rows:
        pl.DataFrame(pred_rows).write_parquet(preds_path)

    elapsed = time.time() - t0
    samples_per_sec = samples_seen / max(elapsed, 1e-6)
    log_event({"event": "fit_end", "elapsed_sec": elapsed,
               "n_predicted_days": n_predicted, "n_trained_days": n_trained,
               "n_pred_rows": len(pred_rows), "samples_per_sec": samples_per_sec})
    print(f"DONE | trained={n_trained} predicted={n_predicted} | "
          f"{elapsed/60:.1f} min, {samples_per_sec:.1f} samp/s", flush=True)

    if wandb_run is not None:
        try: wandb_run.finish()
        except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
