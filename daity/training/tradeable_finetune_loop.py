"""Phase 3X — TradeableHead training loop driver.

Minimal: train ONLY the new 3-way head on focal loss; encoder is frozen.
Eval metric: precision@k + recall@k on tradeable labels (LONG and SHORT).
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from daity.data.cross_section import CrossSection, CrossSectionalSampler
from daity.data.parquet_store import ParquetStore
from daity.eval.classifier_metrics import (
    aggregate_mean_pnl_at_k,
    aggregate_precision_recall_at_k,
)
from daity.training.tradeable_finetune import (
    TradeableFineTuneConfig, TradeableFineTuneModule,
    TradeableOBFineTuneConfig, TradeableOBFineTuneModule,
)
from daity.utils.logging import get_logger

UTC = timezone.utc
log = get_logger(__name__)


DEFAULT_CADENCE: dict[str, float] = {"30m": 70.0, "120m": 0.0}  # 30m only (matches label window)


@dataclass
class TradeableRunConfig:
    feature_root: Path
    label_root: Path
    tradeable_root: Path
    universe: list[str]
    parent_checkpoint: Path

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    as_of: datetime

    module_cfg: TradeableFineTuneConfig | None = None
    # Stage 2: when set, instantiate TradeableOBFineTuneModule (loads Phase-4
    # OB-fused encoder) instead of TradeableFineTuneModule. `module_cfg`
    # is interpreted as `TradeableOBFineTuneConfig` when this is set.
    parent_ob_checkpoint: Path | None = None
    ob_root: Path | None = None
    # Optional: when set, sampler attaches `aux_features: (N, 50)` to each
    # cross-section and the xsattn head (if head_aux_dim>0) consumes them.
    aux_root: Path | None = None

    max_steps: int = 4000
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps_pct: float = 0.05
    lr_min_ratio: float = 0.05
    gradient_clip_val: float = 1.0
    # LR ratios per param group. Defaults: lora=1/3 of head, backbone=1/10 of head.
    # For unfrozen-encoder training where the encoder needs material updates,
    # bump backbone ratio toward 1/3 (matches lora).
    lr_ratio_lora:     float = 1.0 / 3
    lr_ratio_backbone: float = 1.0 / 10
    # Number of cross-sections sampled per optimizer step. Each accum-step
    # runs forward+backward (no clip/step), then ONE clip + step + zero_grad.
    # Variance ↓ √N; smoother loss curves; same wall-clock per encoder forward.
    gradient_accumulation_steps: int = 1
    seed: int = 0

    window_bars: dict[str, int] | None = None
    drop_invalid_threshold: float = 0.3
    cadence_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CADENCE),
    )

    limit_val_cross_sections: int | None = 500
    log_every_n_steps: int = 100
    eval_ks: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 50, 100)

    # Intermediate val: run _evaluate every N steps to track convergence
    # and pick the best checkpoint by `best_metric` (e.g. "long_precision_at_10").
    # None disables intermediate eval — only the final eval runs.
    eval_every_n_steps: int | None = None
    best_metric: str = "long_precision_at_10"  # one of {long|short}_precision_at_{k}

    save_model_path: Path | None = None
    # When intermediate eval is on, save best-by-`best_metric` here. If None,
    # uses save_model_path.with_suffix(".best.state.pt").
    save_best_path: Path | None = None

    # W&B logging. Set `wandb_project` to enable; everything else optional.
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()


@dataclass
class TradeableRunResult:
    final_step: int
    long_precision_at_k: dict[int, float]
    long_recall_at_k:    dict[int, float]
    short_precision_at_k: dict[int, float]
    short_recall_at_k:    dict[int, float]
    base_rate_long:  float
    base_rate_short: float
    n_val_cross_sections: int
    final_loss: float


class _HorizonStreams:
    """Persistent generators per horizon. Recycles on exhaustion."""
    def __init__(self, sampler: CrossSectionalSampler, horizons: list[str]) -> None:
        self.sampler = sampler
        self.horizons = horizons
        self._iters = {h: iter(getattr(sampler, f"iter_{h}")()) for h in horizons}

    def draw(self, h: str) -> CrossSection | None:
        try:
            return next(self._iters[h])
        except StopIteration:
            self._iters[h] = iter(getattr(self.sampler, f"iter_{h}")())
            try:
                return next(self._iters[h])
            except StopIteration:
                return None


def _cosine_lr(step: int, max_steps: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _to_device(cs: CrossSection, device: torch.device) -> CrossSection:
    aux = cs.aux_features.to(device) if cs.aux_features is not None else None
    return CrossSection(
        t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
        inputs={s: t.to(device) for s, t in cs.inputs.items()},
        labels=cs.labels.to(device), ranks=cs.ranks.to(device),
        valid_mask=cs.valid_mask.to(device), vol_bucket=cs.vol_bucket.to(device),
        aux_features=aux,
        ob_snapshot_window=(cs.ob_snapshot_window.to(device) if cs.ob_snapshot_window is not None else None),
        ob_mask=(cs.ob_mask.to(device) if cs.ob_mask is not None else None),
        tradeable_label=(cs.tradeable_label.to(device) if cs.tradeable_label is not None else None),
        tradeable_valid=(cs.tradeable_valid.to(device) if cs.tradeable_valid is not None else None),
        realized_pnl_long=(cs.realized_pnl_long.to(device) if cs.realized_pnl_long is not None else None),
        realized_pnl_short=(cs.realized_pnl_short.to(device) if cs.realized_pnl_short is not None else None),
    )


def _build_samplers(cfg: TradeableRunConfig) -> tuple[CrossSectionalSampler, CrossSectionalSampler]:
    feature_store = ParquetStore(cfg.feature_root)
    window_bars = cfg.window_bars or {"5m": 256, "15m": 192, "60m": 168, "day": 64}
    train = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe, as_of=cfg.train_end, window_bars=window_bars,
        min_end_ts=cfg.train_start,
        drop_invalid_threshold=cfg.drop_invalid_threshold,
        tradeable_root=cfg.tradeable_root,
        aux_root=cfg.aux_root,
        ob_root=cfg.ob_root,
    )
    val = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe, as_of=cfg.val_end, window_bars=window_bars,
        min_end_ts=cfg.val_start,
        drop_invalid_threshold=cfg.drop_invalid_threshold,
        tradeable_root=cfg.tradeable_root,
        aux_root=cfg.aux_root,
        ob_root=cfg.ob_root,
    )
    return train, val


def _evaluate(
    module: TradeableFineTuneModule, val: CrossSectionalSampler,
    *, limit: int | None, eval_ks: tuple[int, ...],
) -> dict:
    """Forward through val sampler; aggregate precision/recall@k for LONG + SHORT.

    Also computes mean realized-PnL @ top-K (where realized_pnl_long is
    available in the dataset) — this is the metric aligned with the bandit
    objective and with backtest results.
    """
    module.eval()
    device = next(module.parameters()).device
    long_scores: list[np.ndarray] = []
    long_labels: list[np.ndarray] = []
    short_scores: list[np.ndarray] = []
    short_labels: list[np.ndarray] = []
    long_pnl:    list[np.ndarray | None] = []
    n_used = 0
    total_target_long = total_target_short = total_rows = 0
    with torch.no_grad():
        for cs in val.iter_30m():
            if limit is not None and n_used >= limit:
                break
            cs_dev = _to_device(cs, device)
            out = module(cs_dev, horizon="30m")
            valid = cs.tradeable_valid.numpy().astype(bool)
            if valid.sum() < 10:
                continue
            labs = cs.tradeable_label.numpy().astype(np.int8)[valid]
            lp = out["long_probs"].detach().cpu().numpy()[valid]
            sp = out["short_probs"].detach().cpu().numpy()[valid]
            long_scores.append(lp); long_labels.append(labs)
            short_scores.append(sp); short_labels.append(labs)
            # Realized PnL per symbol (if the dataset carries it).
            if cs.realized_pnl_long is not None:
                long_pnl.append(cs.realized_pnl_long.cpu().numpy()[valid])
            else:
                long_pnl.append(None)
            total_target_long  += int((labs == 1).sum())
            total_target_short += int((labs == -1).sum())
            total_rows += len(labs)
            n_used += 1
    long_metrics = aggregate_precision_recall_at_k(
        scores_per_cs=long_scores, labels_per_cs=long_labels,
        ks=list(eval_ks), target=+1,
    )
    short_metrics = aggregate_precision_recall_at_k(
        scores_per_cs=short_scores, labels_per_cs=short_labels,
        ks=list(eval_ks), target=-1,
    )
    long_pnl_metrics = aggregate_mean_pnl_at_k(
        scores_per_cs=long_scores, pnl_per_cs=long_pnl, ks=list(eval_ks),
    )
    module.train()
    return {
        "long":  long_metrics, "short": short_metrics,
        "long_pnl": long_pnl_metrics,
        "n_cs": n_used,
        "base_rate_long":  total_target_long  / max(1, total_rows),
        "base_rate_short": total_target_short / max(1, total_rows),
    }


def run_tradeable_finetune(
    cfg: TradeableRunConfig,
) -> tuple[TradeableFineTuneModule, TradeableRunResult]:
    """Train TradeableHead on one fold. Returns module + per-k metrics."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    train_sampler, val_sampler = _build_samplers(cfg)
    # Dispatch: Stage 1 (Phase-3 encoder) vs Stage 2 (Phase-4 OB-fused encoder).
    if cfg.parent_ob_checkpoint is not None:
        module: nn.Module = TradeableOBFineTuneModule(cfg.module_cfg)  # type: ignore[arg-type]
    else:
        module = TradeableFineTuneModule(
            cfg.module_cfg or TradeableFineTuneConfig(parent_checkpoint=cfg.parent_checkpoint)
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).train()
    # Param groups: head + (LoRA) + (backbone) at decreasing LRs.
    head_params, lora_params, backbone_params = [], [], []
    for n, p in module.trainable_parameters():
        if "tradeable_head" in n:
            head_params.append(p)
        elif "lora_" in n:
            lora_params.append(p)
        else:
            backbone_params.append(p)
    n_head = sum(p.numel() for p in head_params)
    n_lora = sum(p.numel() for p in lora_params)
    n_back = sum(p.numel() for p in backbone_params)
    log.info("trainable params: head=%d lora=%d backbone=%d (total=%d)",
             n_head, n_lora, n_back, n_head + n_lora + n_back)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr,
                       "weight_decay": cfg.weight_decay, "name": "head",
                       "_lr_base": cfg.lr})
    if lora_params:
        # LoRA gets 1/3 of head LR — standard PEFT recipe.
        groups.append({"params": lora_params, "lr": cfg.lr * cfg.lr_ratio_lora,
                       "weight_decay": 0.0, "name": "lora",
                       "_lr_base": cfg.lr * cfg.lr_ratio_lora})
    if backbone_params:
        # Backbone gets 1/10 of head LR — avoid catastrophic forgetting.
        groups.append({"params": backbone_params, "lr": cfg.lr * cfg.lr_ratio_backbone,
                       "weight_decay": cfg.weight_decay, "name": "backbone",
                       "_lr_base": cfg.lr * cfg.lr_ratio_backbone})
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.95), fused=True,
    )
    trainable = head_params + lora_params + backbone_params
    warmup = int(cfg.max_steps * cfg.warmup_steps_pct)
    streams = _HorizonStreams(train_sampler, list(cfg.cadence_weights.keys()))
    rng = random.Random(cfg.seed)
    last_loss = float("nan")

    # ---- W&B init ----
    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                entity=cfg.wandb_entity,
                tags=list(cfg.wandb_tags),
                config={
                    "max_steps": cfg.max_steps, "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay, "seed": cfg.seed,
                    "n_universe": len(cfg.universe),
                    "unfreeze_mode": cfg.module_cfg.unfreeze_mode if cfg.module_cfg else "none",
                    "focal_gamma":   cfg.module_cfg.focal_gamma if cfg.module_cfg else None,
                    "class_alpha":   list(cfg.module_cfg.class_alpha) if cfg.module_cfg else None,
                    "n_head_params":     n_head,
                    "n_lora_params":     n_lora,
                    "n_backbone_params": n_back,
                },
            )
        except Exception as e:
            log.warning("W&B init failed (%s); continuing without W&B", e)
            wandb_run = None

    horizons = [h for h, w in cfg.cadence_weights.items() if w > 0]
    weights  = [cfg.cadence_weights[h] for h in horizons]

    # Best-checkpoint tracking for intermediate eval.
    best_score: float = -1.0
    best_step:  int   = -1
    if cfg.eval_every_n_steps is not None and cfg.save_best_path is None and cfg.save_model_path is not None:
        cfg.save_best_path = cfg.save_model_path.with_suffix(".best.state.pt")

    t0 = time.time()
    # Smoothed loss tracker: training is single-cross-section per step + γ=2
    # focal loss → raw batch loss has large variance. Running mean over last
    # 100 batches is what we actually want to monitor for convergence.
    loss_window: deque[float] = deque(maxlen=100)
    for step in range(cfg.max_steps):
        lr_scale = _cosine_lr(step, cfg.max_steps, warmup, cfg.lr_min_ratio)
        # Each group's base LR was set at construction (stored in _lr_base);
        # apply cosine scale uniformly.
        for g in optimizer.param_groups:
            base = g.get("_lr_base", cfg.lr)
            g["lr"] = base * lr_scale

        # Gradient accumulation: average gradients across N cross-sections per
        # optimizer step. Same wall-clock per CS, variance ↓ √N on the
        # effective batch. Loss reported is the MEAN across the accum batch.
        accum_n = max(1, cfg.gradient_accumulation_steps)
        h = rng.choices(horizons, weights=weights, k=1)[0]  # one horizon per opt step
        accum_loss_sum = 0.0
        accum_n_used = 0
        for _ in range(accum_n):
            cs = streams.draw(h)
            if cs is None:
                continue
            cs_dev = _to_device(cs, device)
            out = module(cs_dev, horizon=h)
            cs_loss = module.compute_loss(out, cs_dev) / accum_n
            cs_loss.backward()
            accum_loss_sum += float(cs_loss.detach()) * accum_n
            accum_n_used += 1
        if accum_n_used == 0:
            continue
        # `loss` is the unscaled mean-loss-per-cross-section (for logging).
        loss = torch.tensor(accum_loss_sum / accum_n_used, device=device)
        # Compute pre-clip per-group norms ONLY on log steps (cheap on log step,
        # skip otherwise). Use max_norm=inf so clip_grad_norm_ returns the norm
        # without modifying gradients. THEN apply the actual clip on all params.
        if (step + 1) % cfg.log_every_n_steps == 0:
            grad_norm_head = float(torch.nn.utils.clip_grad_norm_(
                head_params, float("inf"),
            )) if head_params else 0.0
            grad_norm_lora = float(torch.nn.utils.clip_grad_norm_(
                lora_params, float("inf"),
            )) if lora_params else 0.0
            grad_norm_back = float(torch.nn.utils.clip_grad_norm_(
                backbone_params, float("inf"),
            )) if backbone_params else 0.0
        else:
            grad_norm_head = grad_norm_lora = grad_norm_back = 0.0
        if cfg.gradient_clip_val > 0:
            total_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainable, cfg.gradient_clip_val,
            ))
        else:
            total_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainable, float("inf"),
            ))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.detach())
        loss_window.append(last_loss)

        if (step + 1) % cfg.log_every_n_steps == 0:
            smoothed = sum(loss_window) / len(loss_window)
            log.info(
                "step=%d h=%s loss=%.4f smoothed=%.4f lr=%.2e "
                "grad_total=%.3f head=%.3f lora=%.3f back=%.3f",
                step + 1, h, last_loss, smoothed, optimizer.param_groups[0]["lr"],
                total_grad_norm, grad_norm_head, grad_norm_lora, grad_norm_back,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss":          last_loss,
                    "train/loss_smoothed": smoothed,
                    "train/grad_norm_total": total_grad_norm,
                    "train/grad_norm_head":  grad_norm_head,
                    "train/grad_norm_lora":  grad_norm_lora,
                    "train/grad_norm_backbone": grad_norm_back,
                    "train/lr_head":     next((g["lr"] for g in optimizer.param_groups if g.get("name") == "head"), 0.0),
                    "train/lr_lora":     next((g["lr"] for g in optimizer.param_groups if g.get("name") == "lora"), 0.0),
                    "train/lr_backbone": next((g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone"), 0.0),
                    "train/horizon":  h,
                }, step=step + 1)

        # ---- Intermediate eval + best-checkpoint tracking ----
        if (
            cfg.eval_every_n_steps is not None
            and (step + 1) % cfg.eval_every_n_steps == 0
            and step + 1 < cfg.max_steps
        ):
            eval_res_i = _evaluate(
                module, val_sampler, limit=cfg.limit_val_cross_sections,
                eval_ks=cfg.eval_ks,
            )
            # Extract `best_metric` value. Supported forms:
            #   long_precision_at_K / short_precision_at_K   (default)
            #   long_recall_at_K / short_recall_at_K
            #   long_mean_pnl_at_K  (realized PnL @ top-K, aligned with bandit)
            #   long_sharpe_at_K    (mean/std)
            bm = cfg.best_metric
            metric_k = int(bm.rsplit("_", 1)[-1])
            if "mean_pnl_at" in bm:
                current = next(m.mean_pnl for m in eval_res_i["long_pnl"] if m.k == metric_k)
            elif "sharpe_at" in bm:
                current = next(m.sharpe for m in eval_res_i["long_pnl"] if m.k == metric_k)
            else:
                direction, _, what = bm.partition("_")  # "long" | "short"
                if "recall" in bm:
                    current = next(m.recall for m in eval_res_i[direction] if m.k == metric_k)
                else:
                    current = next(m.precision for m in eval_res_i[direction] if m.k == metric_k)
            log.info(
                "INTERMEDIATE step=%d %s=%.4f (best=%.4f)",
                step + 1, cfg.best_metric, current, best_score,
            )
            if wandb_run is not None:
                d_long  = {m.k: (m.precision, m.recall) for m in eval_res_i["long"]}
                d_short = {m.k: (m.precision, m.recall) for m in eval_res_i["short"]}
                d_pnl   = {m.k: (m.mean_pnl, m.pnl_std, m.sharpe) for m in eval_res_i["long_pnl"]}
                wandb_run.log({
                    **{f"val/long_precision_at_{k}":  v[0] for k, v in d_long.items()},
                    **{f"val/long_recall_at_{k}":     v[1] for k, v in d_long.items()},
                    **{f"val/short_precision_at_{k}": v[0] for k, v in d_short.items()},
                    **{f"val/short_recall_at_{k}":    v[1] for k, v in d_short.items()},
                    **{f"val/long_mean_pnl_at_{k}":   v[0] for k, v in d_pnl.items()},
                    **{f"val/long_pnl_std_at_{k}":    v[1] for k, v in d_pnl.items()},
                    **{f"val/long_sharpe_at_{k}":     v[2] for k, v in d_pnl.items()},
                    "val/base_rate_long":  eval_res_i["base_rate_long"],
                    "val/base_rate_short": eval_res_i["base_rate_short"],
                    "val/n_cross_sections": eval_res_i["n_cs"],
                    f"val/best/{cfg.best_metric}": max(current, best_score),
                }, step=step + 1)
            if current > best_score:
                best_score = current
                best_step  = step + 1
                if cfg.save_best_path is not None:
                    cfg.save_best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "state_dict": module.state_dict(),
                        "module_cfg": (cfg.module_cfg or TradeableFineTuneConfig(
                            parent_checkpoint=cfg.parent_checkpoint
                        )).__dict__,
                        "best_step":   best_step,
                        "best_metric": cfg.best_metric,
                        "best_score":  best_score,
                    }, cfg.save_best_path)
                    log.info(
                        "NEW BEST step=%d %s=%.4f → saved %s",
                        best_step, cfg.best_metric, best_score, cfg.save_best_path,
                    )

    log.info("training done. elapsed=%.1fs", time.time() - t0)
    if cfg.eval_every_n_steps is not None:
        log.info(
            "best intermediate: step=%d %s=%.4f",
            best_step, cfg.best_metric, best_score,
        )
    eval_res = _evaluate(module, val_sampler, limit=cfg.limit_val_cross_sections,
                          eval_ks=cfg.eval_ks)

    long_p = {m.k: m.precision for m in eval_res["long"]}
    long_r = {m.k: m.recall    for m in eval_res["long"]}
    short_p = {m.k: m.precision for m in eval_res["short"]}
    short_r = {m.k: m.recall    for m in eval_res["short"]}

    for m in eval_res["long"]:
        lift = m.precision / max(eval_res["base_rate_long"], 1e-9)
        log.info("EVAL LONG  k=%d precision=%.4f recall=%.4f lift=%.2f×",
                 m.k, m.precision, m.recall, lift)
        if wandb_run is not None:
            wandb_run.log({
                f"val/long_precision_at_{m.k}": m.precision,
                f"val/long_recall_at_{m.k}":    m.recall,
                f"val/long_lift_at_{m.k}":      lift,
            }, step=cfg.max_steps)
    for m in eval_res["short"]:
        lift = m.precision / max(eval_res["base_rate_short"], 1e-9)
        log.info("EVAL SHORT k=%d precision=%.4f recall=%.4f lift=%.2f×",
                 m.k, m.precision, m.recall, lift)
        if wandb_run is not None:
            wandb_run.log({
                f"val/short_precision_at_{m.k}": m.precision,
                f"val/short_recall_at_{m.k}":    m.recall,
                f"val/short_lift_at_{m.k}":      lift,
            }, step=cfg.max_steps)
    for m in eval_res.get("long_pnl", []):
        log.info("EVAL LONG-PNL k=%d  mean_pnl=%+.5f  std=%.5f  sharpe=%+.2f",
                 m.k, m.mean_pnl, m.pnl_std, m.sharpe)
        if wandb_run is not None:
            wandb_run.log({
                f"val/long_mean_pnl_at_{m.k}": m.mean_pnl,
                f"val/long_pnl_std_at_{m.k}":  m.pnl_std,
                f"val/long_sharpe_at_{m.k}":   m.sharpe,
            }, step=cfg.max_steps)
    if wandb_run is not None:
        wandb_run.log({
            "val/base_rate_long":  eval_res["base_rate_long"],
            "val/base_rate_short": eval_res["base_rate_short"],
            "val/n_cross_sections": eval_res["n_cs"],
        }, step=cfg.max_steps)
        wandb_run.finish()

    if cfg.save_model_path is not None:
        cfg.save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": module.state_dict(),
            "module_cfg": (cfg.module_cfg or TradeableFineTuneConfig(
                parent_checkpoint=cfg.parent_checkpoint
            )).__dict__,
            "long_precision_at_k": long_p, "long_recall_at_k": long_r,
            "short_precision_at_k": short_p, "short_recall_at_k": short_r,
            "base_rate_long": eval_res["base_rate_long"],
            "base_rate_short": eval_res["base_rate_short"],
        }, cfg.save_model_path)
        log.info("saved tradeable-head state to %s", cfg.save_model_path)

    result = TradeableRunResult(
        final_step=cfg.max_steps,
        long_precision_at_k=long_p, long_recall_at_k=long_r,
        short_precision_at_k=short_p, short_recall_at_k=short_r,
        base_rate_long=eval_res["base_rate_long"],
        base_rate_short=eval_res["base_rate_short"],
        n_val_cross_sections=eval_res["n_cs"],
        final_loss=last_loss,
    )
    return module, result
