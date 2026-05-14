"""Phase 4 step 6 — fine-tune loop driver.

A minimal training loop adapted from `posttrain_loop.py`. Differences:
- Two horizons only (30m, 120m); 1d/2d hard-frozen.
- Per DESIGN §3.4: `gate_zero_steps` curriculum forces fusion gate to
  zero for the first ~1500 opt steps.
- Twin eval: at end of training, evaluate with `_use_ob=True` AND
  `_use_ob=False`, report Δ IC per horizon — the §3.4 acceptance signal.

Designed to be SHORT (≤300 LOC) — Phase 4 is a fine-tune, not a sweep.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from daity.data.cross_section import CrossSection, CrossSectionalSampler
from daity.data.parquet_store import ParquetStore
from daity.data.splits import OBFold
from daity.eval.metrics import cross_sectional_rank_ic
from daity.training.finetune import (
    PHASE4_TRAINABLE_HORIZONS,
    FineTuneConfig,
    FineTuneModule,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Phase 4 cadence: intraday only.
DEFAULT_OB_CADENCE: dict[str, float] = {"30m": 70.0, "120m": 52.0}


@dataclass
class FineTuneRunConfig:
    """Top-level Phase 4 fine-tune knobs."""
    fold: OBFold
    feature_root: Path
    label_root: Path
    ob_root: Path
    universe: list[str]
    parent_checkpoint: Path        # Phase-3 *.state.pt

    module_cfg: FineTuneConfig | None = None

    # Optim.
    max_steps: int = 8000
    lr_lora: float = 1e-4          # smaller than Phase 3 — fine-tune
    lr_head: float = 3e-4
    lr_ob: float = 3e-4            # OB encoder + fusion learn together
    weight_decay: float = 1e-2     # tighter than Phase 3 (DESIGN §3.4)
    warmup_steps_pct: float = 0.05
    lr_min_ratio: float = 0.05
    gradient_clip_val: float = 1.0
    seed: int = 0

    # Curriculum.
    gate_zero_steps: int = 1500    # steps with fusion gate forced to zero

    # Sampling.
    window_bars: dict[str, int] | None = None
    cadence_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OB_CADENCE),
    )
    drop_invalid_threshold: float = 0.3   # lower — small-N OB-covered subset

    # Eval.
    limit_val_cross_sections: int | None = 500
    log_every_n_steps: int = 100

    # Output.
    save_model_path: Path | None = None


@dataclass
class FineTuneResult:
    fold_id: str
    final_step: int
    per_horizon_rank_ic_with_ob: dict[str, float]
    per_horizon_rank_ic_without_ob: dict[str, float]
    per_horizon_n_cross_sections: dict[str, int]
    per_horizon_delta_ic: dict[str, float]  # with - without per horizon
    final_loss_by_horizon: dict[str, float]


def _build_samplers(cfg: FineTuneRunConfig) -> tuple[CrossSectionalSampler, CrossSectionalSampler]:
    """Build train + val samplers from the OB-fold time bounds."""
    feature_store = ParquetStore(cfg.feature_root)
    window_bars = cfg.window_bars or {"5m": 256, "15m": 192, "60m": 168, "day": 64}
    train_start = datetime.combine(cfg.fold.train_start, datetime.min.time()).replace(tzinfo=__import__("datetime").timezone.utc)
    train_end   = datetime.combine(cfg.fold.train_end,   datetime.min.time()).replace(tzinfo=__import__("datetime").timezone.utc)
    val_start   = datetime.combine(cfg.fold.val_start,   datetime.min.time()).replace(tzinfo=__import__("datetime").timezone.utc)
    val_end     = datetime.combine(cfg.fold.val_end,     datetime.min.time()).replace(tzinfo=__import__("datetime").timezone.utc)
    train = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe, as_of=train_end, window_bars=window_bars,
        min_end_ts=train_start, drop_invalid_threshold=cfg.drop_invalid_threshold,
        ob_root=cfg.ob_root,
    )
    val = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe, as_of=val_end, window_bars=window_bars,
        min_end_ts=val_start, drop_invalid_threshold=cfg.drop_invalid_threshold,
        ob_root=cfg.ob_root,
    )
    return train, val


def _cosine_lr(step: int, max_steps: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_optimizer(module: FineTuneModule, cfg: FineTuneRunConfig) -> torch.optim.Optimizer:
    """Three param groups: LoRA (intraday), heads (intraday), OB+fusion."""
    lora_params, head_params, ob_params = [], [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if "ob_encoder" in n or "fused_backbone.fusion_layers" in n:
            ob_params.append(p)
        elif "lora_" in n:
            lora_params.append(p)
        elif "_heads" in n:
            head_params.append(p)
        else:
            head_params.append(p)  # fallback bucket
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lr_lora,
                       "weight_decay": 0.0, "name": "lora"})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head,
                       "weight_decay": cfg.weight_decay, "name": "heads"})
    if ob_params:
        groups.append({"params": ob_params, "lr": cfg.lr_ob,
                       "weight_decay": cfg.weight_decay, "name": "ob"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=True)


def _evaluate(
    module: FineTuneModule, val: CrossSectionalSampler,
    *, use_ob: bool, limit: int | None,
) -> dict[str, tuple[float, int]]:
    """Per-horizon (mean_ic, n_cs) for the chosen `use_ob`."""
    module.eval()
    module.use_ob = use_ob
    device = next(module.parameters()).device
    out_dict: dict[str, tuple[float, int]] = {}
    for h in PHASE4_TRAINABLE_HORIZONS:
        preds_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        ts_list: list[np.datetime64] = []
        n_used = 0
        with torch.no_grad():
            for cs in getattr(val, f"iter_{h}")():
                if limit is not None and n_used >= limit:
                    break
                cs_dev = _to_device(cs, device)
                out = module(cs_dev, horizon=h)
                preds_list.append(out["rank"].detach().cpu().numpy())
                labels_list.append(cs.labels.numpy())
                ts_list.append(
                    np.full(len(cs.symbols), np.datetime64(cs.t.replace(tzinfo=None))),
                )
                n_used += 1
        if not preds_list:
            out_dict[h] = (0.0, 0)
            continue
        preds = np.concatenate(preds_list)
        labels = np.concatenate(labels_list)
        ts = np.concatenate(ts_list)
        rank_ic, _ = cross_sectional_rank_ic(
            predictions=preds, realizations=labels, timestamps=ts,
            return_per_cs=True,
        )
        out_dict[h] = (rank_ic.mean_ic, n_used)
    module.train()
    return out_dict


def _to_device(cs: CrossSection, device: torch.device) -> CrossSection:
    """Move CrossSection tensors to `device` (small copy of helper from posttrain_loop)."""
    return CrossSection(
        t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
        inputs={s: t.to(device) for s, t in cs.inputs.items()},
        labels=cs.labels.to(device), ranks=cs.ranks.to(device),
        valid_mask=cs.valid_mask.to(device), vol_bucket=cs.vol_bucket.to(device),
        aux_features=(cs.aux_features.to(device) if cs.aux_features is not None else None),
        ob_snapshot_window=(cs.ob_snapshot_window.to(device) if cs.ob_snapshot_window is not None else None),
        ob_mask=(cs.ob_mask.to(device) if cs.ob_mask is not None else None),
    )


class _HorizonStreams:
    """Maintains a persistent per-horizon generator and recycles on
    exhaustion. Replaces the buggy `next(iter(sampler.iter_X()))` pattern
    which always returned the FIRST cross-section."""

    def __init__(self, train: CrossSectionalSampler, horizons: list[str]) -> None:
        self.train = train
        self.horizons = list(horizons)
        self._iters: dict = {h: iter(getattr(train, f"iter_{h}")()) for h in horizons}

    def draw(self, h: str) -> CrossSection | None:
        try:
            return next(self._iters[h])
        except StopIteration:
            # Recycle.
            self._iters[h] = iter(getattr(self.train, f"iter_{h}")())
            try:
                return next(self._iters[h])
            except StopIteration:
                return None


def _draw_one(streams: _HorizonStreams, cadence: dict[str, float],
              rng: random.Random) -> CrossSection | None:
    """Draw one cross-section from a weighted-random horizon."""
    horizons = list(cadence.keys())
    weights = [cadence[h] for h in horizons]
    h = rng.choices(horizons, weights=weights, k=1)[0]
    return streams.draw(h)


def run_finetune_fold(cfg: FineTuneRunConfig) -> tuple[FineTuneModule, FineTuneResult]:
    """Train one Phase 4 OB fold; return module + twin-eval result."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    train_sampler, val_sampler = _build_samplers(cfg)
    module = FineTuneModule(cfg.module_cfg or FineTuneConfig(parent_checkpoint=cfg.parent_checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).train()
    optimizer = _build_optimizer(module, cfg)

    rng = random.Random(cfg.seed)
    warmup = int(cfg.max_steps * cfg.warmup_steps_pct)
    last_loss_by_h: dict[str, float] = {h: float("nan") for h in PHASE4_TRAINABLE_HORIZONS}

    # Persistent per-horizon streams (Phase 4 bug fix: prior version
    # `next(iter(sampler.iter_X()))` always returned the SAME first cs).
    streams = _HorizonStreams(train_sampler, list(cfg.cadence_weights.keys()))

    # Curriculum: gate force zero for the first gate_zero_steps.
    module.set_gate_force_zero(True)

    t0 = time.time()
    n_skipped = 0
    for step in range(cfg.max_steps):
        # Curriculum transition.
        if step == cfg.gate_zero_steps:
            log.info("gate_force_zero=False at step=%d (OB tower online)", step)
            module.set_gate_force_zero(False)

        # LR schedule.
        lr_scale = _cosine_lr(step, cfg.max_steps, warmup, cfg.lr_min_ratio)
        for g, base in zip(optimizer.param_groups, [cfg.lr_lora, cfg.lr_head, cfg.lr_ob], strict=False):
            g["lr"] = base * lr_scale

        cs = _draw_one(streams, cfg.cadence_weights, rng)
        if cs is None:
            n_skipped += 1
            continue
        cs_dev = _to_device(cs, device)
        h = cs.horizon
        out = module(cs_dev, horizon=h)
        loss = module.compute_loss(out, cs_dev, horizon=h)
        loss.backward()
        if cfg.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in module.trainable_parameters()],
                cfg.gradient_clip_val,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss_by_h[h] = float(loss.detach())

        if (step + 1) % cfg.log_every_n_steps == 0:
            log.info(
                "step=%d h=%s loss=%.4f lr_lora=%.2e lr_head=%.2e lr_ob=%.2e",
                step + 1, h, last_loss_by_h[h],
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else 0.0,
            )

    elapsed = time.time() - t0
    log.info("training done. steps=%d elapsed=%.1fs skipped=%d", cfg.max_steps, elapsed, n_skipped)

    # Twin eval.
    log.info("evaluating with use_ob=True")
    with_ob = _evaluate(module, val_sampler, use_ob=True, limit=cfg.limit_val_cross_sections)
    log.info("evaluating with use_ob=False (baseline)")
    without_ob = _evaluate(module, val_sampler, use_ob=False, limit=cfg.limit_val_cross_sections)

    deltas = {
        h: with_ob.get(h, (0.0, 0))[0] - without_ob.get(h, (0.0, 0))[0]
        for h in PHASE4_TRAINABLE_HORIZONS
    }
    for h in PHASE4_TRAINABLE_HORIZONS:
        log.info(
            "EVAL %s: with_ob=%+.4f  without_ob=%+.4f  delta=%+.4f",
            h, with_ob.get(h, (0.0, 0))[0],
            without_ob.get(h, (0.0, 0))[0], deltas[h],
        )

    if cfg.save_model_path is not None:
        cfg.save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": module.state_dict(),
            "module_cfg": (cfg.module_cfg or FineTuneConfig(parent_checkpoint=cfg.parent_checkpoint)).__dict__,
            "run_config": {
                "fold_id": cfg.fold.name,
                "max_steps": cfg.max_steps,
                "seed": cfg.seed,
                "gate_zero_steps": cfg.gate_zero_steps,
            },
            "with_ob_ic": with_ob,
            "without_ob_ic": without_ob,
        }, cfg.save_model_path)
        log.info("saved fine-tune state to %s", cfg.save_model_path)

    result = FineTuneResult(
        fold_id=cfg.fold.name,
        final_step=cfg.max_steps,
        per_horizon_rank_ic_with_ob={h: with_ob.get(h, (0.0, 0))[0] for h in PHASE4_TRAINABLE_HORIZONS},
        per_horizon_rank_ic_without_ob={h: without_ob.get(h, (0.0, 0))[0] for h in PHASE4_TRAINABLE_HORIZONS},
        per_horizon_n_cross_sections={h: with_ob.get(h, (0.0, 0))[1] for h in PHASE4_TRAINABLE_HORIZONS},
        per_horizon_delta_ic=deltas,
        final_loss_by_horizon=last_loss_by_h,
    )
    return module, result
