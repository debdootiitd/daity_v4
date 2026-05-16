"""CLI entry point for cohort training.

Usage examples:
    # Mode A: initial fit 2019-2024
    python -m daity.scripts.train_cohort \
        --mode A \
        --train-start 2019-01-01 --train-end 2024-11-30 \
        --val-start 2024-12-01 --val-end 2024-12-31 \
        --log-dir runs/cohort_modeA_v1

    # Mode C: 5-day rolling, with init ckpt
    python -m daity.scripts.train_cohort \
        --mode C --init-ckpt runs/cohort_modeA_v1/checkpoints/best.pt \
        --train-start 2024-12-01 --train-end 2025-01-15 \
        --val-start 2025-01-16 --val-end 2025-01-20 \
        --max-steps 1500 --log-dir runs/cohort_modeC_w01
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import torch

from daity.data.calendar_nse import NSECalendar
from daity.data.cohort_dataset import (
    CohortAssembler,
    DEFAULT_COHORT_CHANNELS,
    N_REGIME_FEATS,
)
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster
from daity.models.cohort.cohort_module import CohortModel
from daity.training.walk_forward import (
    CohortTrainer,
    TrainerConfig,
    build_anchor_pairs,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "C"], default="A")
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--universe-source", choices=["all", "alive_in_range"], default="all",
                    help="all = use all 205 symbols in SymbolMaster; alive_in_range filters.")
    ap.add_argument("--universe-end", type=str, default=None,
                    help="Override the alive_in_range upper bound (default: val-end). "
                         "Pin to a fixed date when continuing training from a ckpt "
                         "whose stock_embed has fixed n_stocks.")
    ap.add_argument("--train-start", type=str, required=True)  # YYYY-MM-DD
    ap.add_argument("--train-end",   type=str, required=True)
    ap.add_argument("--val-start",   type=str, required=True)
    ap.add_argument("--val-end",     type=str, required=True)
    # Model size
    ap.add_argument("--d-model",          type=int, default=480)
    ap.add_argument("--stock-enc-layers", type=int, default=12)
    ap.add_argument("--n-heads",          type=int, default=8)
    ap.add_argument("--ffn-ratio",        type=int, default=4)
    ap.add_argument("--n-market-layers",  type=int, default=4)
    ap.add_argument("--n-cross-layers",   type=int, default=4)
    # Optimizer / schedule
    ap.add_argument("--lr",                type=float, default=3e-4)
    ap.add_argument("--max-steps",         type=int, default=6000)
    ap.add_argument("--warmup-pct",        type=float, default=0.025)
    ap.add_argument("--val-every",         type=int, default=500)
    ap.add_argument("--ckpt-every",        type=int, default=1000)
    ap.add_argument("--weight-decay",      type=float, default=1e-2)
    ap.add_argument("--grad-clip",         type=float, default=1.0)
    # Loss
    ap.add_argument("--w-reg",   type=float, default=1.0)
    ap.add_argument("--w-rank",  type=float, default=0.5)
    ap.add_argument("--w-bias",  type=float, default=0.01)
    ap.add_argument("--w-sector", type=float, default=0.0,
                    help="Sector classification auxiliary loss weight. 0 = off.")
    ap.add_argument("--w-contrastive", type=float, default=0.0,
                    help="Supervised InfoNCE intra-cohort contrastive loss weight. "
                         "0 = off. Positives: same-sector + cos(realized returns) > thresh.")
    ap.add_argument("--contrastive-dim", type=int, default=128)
    ap.add_argument("--contrastive-tau", type=float, default=0.1)
    ap.add_argument("--contrastive-ret-sim-thresh", type=float, default=0.5)
    ap.add_argument("--rank-top-k", type=int, default=20)
    ap.add_argument("--smooth-l1-beta", type=float, default=0.005)
    ap.add_argument("--market-top-k", type=int, default=0,
                    help="Top-K sparsity for MarketContext self-attn (0 = dense).")
    ap.add_argument("--cross-top-k", type=int, default=0,
                    help="Top-K sparsity for CrossAttn (0 = dense).")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Cohorts to accumulate before each optimizer step.")
    ap.add_argument("--val-max-cohorts", type=int, default=100)
    ap.add_argument("--time-ordered", action="store_true",
                    help="Iterate anchors chronologically (no random shuffle).")
    ap.add_argument("--single-epoch", action="store_true",
                    help="Stop after one pass through the anchor list (sets max_steps cap).")
    ap.add_argument("--wandb-project",  type=str, default=None,
                    help="If set, log to this wandb project.")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-tags",     type=str, default="",
                    help="Comma-separated wandb tags.")
    # IO
    ap.add_argument("--log-dir",   type=Path, required=True)
    ap.add_argument("--init-ckpt", type=Path, default=None)
    ap.add_argument("--seed",      type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    # Universe + sectors
    master = SymbolMaster.from_cache(args.cache_root)
    all_syms = sorted(master.frame["symbol"].to_list())
    if args.universe_source == "alive_in_range":
        # Filter to stocks alive throughout train_start..val_end.
        import polars as pl
        t_start = datetime.fromisoformat(args.train_start).replace(tzinfo=UTC)
        v_end = datetime.fromisoformat(
            args.universe_end or args.val_end,
        ).replace(tzinfo=UTC)
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
            if (ts >= t_start).any() and (ts <= v_end).any():
                alive.append(s)
        universe = tuple(alive)
    else:
        universe = tuple(all_syms)
    print(f"universe size: {len(universe)}", flush=True)

    # Sector mapping: stable string→int over the full 15 sectors in SymbolMaster.
    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in master.frame["symbol"].to_list()
    })
    sector_to_id = {s: i for i, s in enumerate(all_sectors)}
    sector_id_by_symbol = {
        s: sector_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }
    print(f"sectors: {len(all_sectors)} ({all_sectors})", flush=True)

    # Calendar + assembler
    calendar = NSECalendar.from_cache(args.cache_root)
    store = ParquetStore(args.feature_root)
    # as_of capped at val_end + safety buffer (10 days for label endpoints).
    val_end_dt = datetime.fromisoformat(args.val_end).replace(tzinfo=UTC)
    as_of = val_end_dt.replace(hour=10, minute=0) + (val_end_dt - val_end_dt) + \
        (datetime.fromisoformat(args.val_end).replace(tzinfo=UTC) - datetime.fromisoformat(args.val_end).replace(tzinfo=UTC))
    # Simpler: as_of = val_end + 14 days (gives multi-day labels room).
    from datetime import timedelta
    as_of = val_end_dt + timedelta(days=14)
    print(f"as_of cap = {as_of}", flush=True)

    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe,
        sector_id_by_symbol=sector_id_by_symbol,
        as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )

    train_start = date.fromisoformat(args.train_start)
    train_end   = date.fromisoformat(args.train_end)
    val_start   = date.fromisoformat(args.val_start)
    val_end     = date.fromisoformat(args.val_end)
    train_anchors = build_anchor_pairs(train_start, train_end, calendar)
    val_anchors   = build_anchor_pairs(val_start, val_end, calendar)
    print(f"train anchors: {len(train_anchors)}; val anchors: {len(val_anchors)}", flush=True)

    # Model
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
        market_top_k=(args.market_top_k if args.market_top_k > 0 else None),
        cross_top_k=(args.cross_top_k if args.cross_top_k > 0 else None),
        n_sector_classes=(len(all_sectors) if args.w_sector > 0 else 0),
        contrastive_dim=(args.contrastive_dim if args.w_contrastive > 0 else 0),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params", flush=True)

    if args.init_ckpt is not None:
        print(f"loading init ckpt: {args.init_ckpt}", flush=True)
        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        # Prefer cohort-shaped state if the ckpt was written by
        # pretrain_contrastive.py (which provides an encoder-only view with
        # the "stock_encoder.*" prefix already applied).
        m_sd = sd.get("cohort_init_state_dict", sd.get("model_state_dict", sd))
        missing, unexpected = model.load_state_dict(m_sd, strict=False)
        print(f"  loaded | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    cfg = TrainerConfig(
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_pct=args.warmup_pct, grad_clip=args.grad_clip,
        max_steps=args.max_steps,
        val_every_n_steps=args.val_every,
        ckpt_every_n_steps=args.ckpt_every,
        w_reg=args.w_reg, w_rank=args.w_rank, w_bias=args.w_bias,
        w_sector=args.w_sector,
        w_contrastive=args.w_contrastive,
        contrastive_tau=args.contrastive_tau,
        contrastive_ret_sim_thresh=args.contrastive_ret_sim_thresh,
        rank_top_k=args.rank_top_k,
        smooth_l1_beta=args.smooth_l1_beta,
        grad_accum_steps=args.grad_accum_steps,
        val_max_cohorts=args.val_max_cohorts,
        log_dir=args.log_dir,
        mode=args.mode,
        time_ordered=args.time_ordered,
        single_epoch=args.single_epoch,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tuple(t.strip() for t in args.wandb_tags.split(",") if t.strip()),
    )

    trainer = CohortTrainer(
        model=model,
        assembler=assembler,
        train_anchors=train_anchors,
        val_anchors=val_anchors,
        config=cfg,
        device=device,
        seed=args.seed,
    )
    state = trainer.fit()
    print(
        f"FIT DONE | step={state.step} best_val_step={state.best_val_step} "
        f"best_val_ic={state.best_val_ic_mean:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
