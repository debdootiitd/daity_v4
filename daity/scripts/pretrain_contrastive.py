"""Two-tower contrastive pretrain: shared StockContextEncoder + projection head.

Architecture (training-time):
  shared encoder = StockContextEncoder + projection (Linear → GELU → Linear → norm).
  Two towers (A, B) both consume single-stock single-anchor inputs and produce
  L2-normalized projection vectors. Loss is symmetric InfoNCE:
    sim(z_a, z_b) / tau  (paired = positive; all others in batch = negatives).

The encoder this trains becomes the load-checkpoint for downstream prediction
fine-tuning (the regression + rank phase) via the existing `--init-ckpt` flag
on `daity.scripts.train_cohort`.

Usage:
  python -m daity.scripts.pretrain_contrastive \
      --pairs-path runs/pair_dataset_v1/pairs.parquet \
      --train-start 2019-01-01 --train-end 2024-11-30 \
      --universe-end 2024-12-31 \
      --batch-pairs 64 --max-steps 4000 --lr 3e-4 \
      --tau 0.1 --proj-dim 128 \
      --log-dir runs/contrastive_pretrain_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from daity.data.calendar_nse import NSECalendar
from daity.data.cohort_dataset import (
    CohortAssembler, DEFAULT_COHORT_CHANNELS,
)
from daity.data.cohort_pair_dataset import CohortPairDataset
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster
from daity.models.cohort.stock_encoder import StockContextEncoder


class ContrastiveModel(nn.Module):
    """StockContextEncoder + projection head. CLS-only output projected to a
    unit-norm contrastive embedding."""

    def __init__(
        self,
        *,
        encoder_kwargs: dict,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = StockContextEncoder(**encoder_kwargs)
        d = encoder_kwargs["d_model"]
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, proj_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """`batch` has 4D x_by_scale (B, N=1, L, C) like the cohort model.

        Flatten (B, 1) → B for the encoder, return projected CLS.
        """
        x_flat = {sc: v.squeeze(1) for sc, v in batch["x_by_scale"].items()}
        out = self.encoder(
            x_flat,
            stock_ids=batch["stock_ids"].squeeze(1),
            sector_ids=batch["sector_ids"].squeeze(1),
            anchor_ts=batch["anchor_ts"],
        )                                                         # (B, 1+T, d)
        cls = out[:, 0, :]                                         # (B, d)
        z = self.proj(cls)
        return F.normalize(z, dim=-1)


def info_nce(za: torch.Tensor, zb: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE on paired projections.

    za, zb: (B, d) — each row is one tower's projection.
    Positive for row i in `za` is row i in `zb` (and vice-versa).
    Negatives: all other rows in `zb` (and `za`).
    """
    if za.shape != zb.shape:
        msg = f"za {tuple(za.shape)} vs zb {tuple(zb.shape)} mismatch"
        raise ValueError(msg)
    B = za.size(0)
    logits = (za @ zb.T) / tau                                    # (B, B)
    labels = torch.arange(B, device=za.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_ab + loss_ba)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-path", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--train-start", type=str, default="2019-01-01")
    ap.add_argument("--train-end",   type=str, default="2024-11-30")
    ap.add_argument("--universe-end", type=str, default=None)
    # Model
    ap.add_argument("--d-model", type=int, default=480)
    ap.add_argument("--stock-enc-layers", type=int, default=1)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ffn-ratio", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=128)
    # Train
    ap.add_argument("--batch-pairs", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--warmup-pct", type=float, default=0.05)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    # IO
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    return ap.parse_args()


def _collate(batch_pairs: list[dict]) -> tuple[dict, dict]:
    """Collate B paired samples → (batch_a, batch_b) with batch dim B."""
    a_x = {sc: [] for sc in batch_pairs[0]["a"]["x_by_scale"]}
    b_x = {sc: [] for sc in batch_pairs[0]["b"]["x_by_scale"]}
    a_sid, a_sec, a_ts = [], [], []
    b_sid, b_sec, b_ts = [], [], []
    a_rf, b_rf = [], []
    for p in batch_pairs:
        for sc in a_x:
            a_x[sc].append(p["a"]["x_by_scale"][sc])
            b_x[sc].append(p["b"]["x_by_scale"][sc])
        a_sid.append(p["a"]["stock_ids"])
        b_sid.append(p["b"]["stock_ids"])
        a_sec.append(p["a"]["sector_ids"])
        b_sec.append(p["b"]["sector_ids"])
        a_ts.append(p["a"]["anchor_ts"])
        b_ts.append(p["b"]["anchor_ts"])
        a_rf.append(p["a"]["regime_feats"])
        b_rf.append(p["b"]["regime_feats"])
    def _stack_x(d):
        return {sc: torch.cat(v, dim=0) for sc, v in d.items()}
    return (
        {
            "x_by_scale": _stack_x(a_x),
            "stock_ids":  torch.cat(a_sid, dim=0),
            "sector_ids": torch.cat(a_sec, dim=0),
            "anchor_ts":  torch.cat(a_ts, dim=0),
            "regime_feats": torch.cat(a_rf, dim=0),
        },
        {
            "x_by_scale": _stack_x(b_x),
            "stock_ids":  torch.cat(b_sid, dim=0),
            "sector_ids": torch.cat(b_sec, dim=0),
            "anchor_ts":  torch.cat(b_ts, dim=0),
            "regime_feats": torch.cat(b_rf, dim=0),
        },
    )


def _cosine_lr(step: int, *, max_steps: int, warmup_pct: float, lr_min_ratio: float) -> float:
    warmup_steps = max(1, int(warmup_pct * max_steps))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min_ratio + (1.0 - lr_min_ratio) * cos


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "metrics.jsonl"
    ckpt_dir = args.log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Universe + sectors
    master = SymbolMaster.from_cache(args.cache_root)
    all_syms = sorted(master.frame["symbol"].to_list())
    univ_end = datetime.fromisoformat(
        args.universe_end or args.train_end,
    ).replace(tzinfo=UTC)
    t_start = datetime.fromisoformat(args.train_start).replace(tzinfo=UTC)
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
    sec_by_sym = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }
    print(f"universe: {len(universe)} stocks, sectors: {len(all_sectors)}", flush=True)

    # Calendar + store + assembler
    calendar = NSECalendar.from_cache(args.cache_root)
    store = ParquetStore(args.feature_root)
    as_of = (datetime.fromisoformat(args.train_end).replace(tzinfo=UTC)
             + timedelta(days=14))
    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe, sector_id_by_symbol=sec_by_sym, as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )

    # Dataset
    pairs_ds = CohortPairDataset(
        assembler=assembler, pairs_path=args.pairs_path,
        shuffle=True, seed=0,
    )

    # Model
    encoder_kwargs = dict(
        scales=("5m", "15m", "60m", "day"),
        num_channels=len(DEFAULT_COHORT_CHANNELS),
        d_model=args.d_model,
        n_layers=args.stock_enc_layers,
        n_heads=args.n_heads,
        ffn_ratio=args.ffn_ratio,
        n_stocks=len(universe),
        n_sectors=len(all_sectors),
    )
    model = ContrastiveModel(encoder_kwargs=encoder_kwargs, proj_dim=args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # wandb (optional)
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

    def log_event(ev):
        with log_path.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(UTC).isoformat(), **ev}, default=str) + "\n")
        if wandb_run is not None:
            payload = {k: v for k, v in ev.items() if isinstance(v, int | float)}
            wandb_run.log(payload)

    log_event({"event": "fit_start", "n_params": n_params,
               "batch_pairs": args.batch_pairs, "max_steps": args.max_steps})

    # Training loop: streaming iterable + manual batching by `batch_pairs`.
    step = 0
    t0 = time.time()
    batch_buf: list[dict] = []
    iter_done = False
    n_epoch_passes = 0

    def consume_batch(buf):
        nonlocal step
        a, b = _collate(buf)
        # Move tensors to device.
        def _to_dev(d):
            return {
                "x_by_scale": {sc: v.to(device) for sc, v in d["x_by_scale"].items()},
                "stock_ids":  d["stock_ids"].to(device),
                "sector_ids": d["sector_ids"].to(device),
                "anchor_ts":  d["anchor_ts"].to(device),
                "regime_feats": d["regime_feats"].to(device),
            }
        a_d = _to_dev(a); b_d = _to_dev(b)
        lr_mult = _cosine_lr(step, max_steps=args.max_steps,
                             warmup_pct=args.warmup_pct,
                             lr_min_ratio=args.lr_min_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_mult
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            za = model(a_d)
            zb = model(b_d)
            loss = info_nce(za, zb, tau=args.tau)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step % args.log_every == 0:
            # Alignment: cos(za, zb) for paired rows.
            with torch.no_grad():
                paired_cos = (za * zb).sum(dim=-1).mean().item()
                # Uniformity proxy: mean off-diagonal cos.
                sim = za @ za.T
                offdiag = sim.masked_select(
                    ~torch.eye(za.size(0), device=device, dtype=torch.bool),
                ).mean().item()
            log_event({
                "event": "step", "step": step, "lr": args.lr * lr_mult,
                "loss": float(loss),
                "paired_cos": paired_cos,
                "self_offdiag_cos": offdiag,
                "elapsed_sec": time.time() - t0,
            })
        if step > 0 and step % args.ckpt_every == 0:
            _save(model, step, ckpt_dir, "step")
        step += 1

    def _save(m, s, d, tag):
        # Save TWO views of the state_dict:
        #   - full ContrastiveModel state (encoder + proj) under "model_state_dict"
        #   - encoder-only state with "stock_encoder.*" prefix so it loads
        #     directly into a CohortModel via the existing --init-ckpt path.
        full_sd = m.state_dict()
        cohort_sd = {}
        for k, v in full_sd.items():
            if k.startswith("encoder."):
                cohort_sd[f"stock_encoder.{k[len('encoder.'):]}"] = v
        torch.save({
            "model_state_dict": full_sd,
            "cohort_init_state_dict": cohort_sd,
            "step": s, "tag": tag,
        }, d / f"{tag}={s}.pt")

    while step < args.max_steps:
        for pair in pairs_ds:
            batch_buf.append(pair)
            if len(batch_buf) >= args.batch_pairs:
                consume_batch(batch_buf)
                batch_buf = []
                if step >= args.max_steps:
                    break
        # Reached end of pairs.
        n_epoch_passes += 1
        log_event({"event": "epoch_pass", "step": step,
                   "epoch_passes": n_epoch_passes})
        if step >= args.max_steps:
            break

    # Final ckpt.
    _save(model, step, ckpt_dir, "last")
    log_event({"event": "fit_end", "step": step,
               "elapsed_sec": time.time() - t0,
               "epoch_passes": n_epoch_passes})
    if wandb_run is not None:
        try: wandb_run.finish()
        except Exception: pass
    print(f"DONE | step={step} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
