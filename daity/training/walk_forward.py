"""Walk-forward trainer for the cohort architecture.

Three modes (all share the same driver, parameterized by cadence):
  - Mode A: initial fit on a long history window (2019-2024)
  - Mode C: 5-day rolling retrain through 2025
  - Mode B: 1-day rolling retrain from 2026 onward

Driver flow:
  1. Build (date, anchor_time) iteration list for train + val periods.
  2. Per epoch, iterate cohorts; for each cohort:
       a. Assemble via CohortAssembler.
       b. Forward → CohortModel → CohortLoss.
       c. Backward + optimizer step.
  3. Periodically run validation; track per-horizon spearman IC.
  4. Save best checkpoint by mean val IC (top horizons).
  5. Log per-day val IC + cumulative top-K P&L to a JSONL file the
     dashboard can tail.

Design choices:
  - Vanilla PyTorch (no Lightning). One cohort per step. Optimizer:
    AdamW with cosine LR decay + warmup. bf16 autocast.
  - Gradient accumulation isn't needed at our cohort sizes (one cohort
    saturates one GPU step nicely).
  - Validation: subsample val anchors to keep eval cheap (default: all val
    anchors but evaluate every K train steps; per-day IC requires the
    full val sweep, run at epoch end).
"""

from __future__ import annotations

import gc
import json
import math
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from daity.data.cohort_dataset import CohortAssembler
from daity.data.intraday_horizons import (
    HORIZONS,
    N_HORIZONS,
    intraday_anchor_grid,
)
from daity.models.cohort.cohort_loss import CohortLoss
from daity.models.cohort.cohort_module import CohortModel
from daity.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Hyperparameters for the cohort trainer."""

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_pct: float = 0.025
    lr_min_ratio: float = 0.05
    grad_clip: float = 1.0
    # Schedule
    max_steps: int = 6000          # Mode A default
    val_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    # Loss
    w_reg: float = 1.0
    w_rank: float = 0.5
    w_bias: float = 0.01           # ML-scientist review: was 0.1; PL+reg are
                                    # both demean-invariant so bias regularizer
                                    # mostly just shrinks pred to 0.
    rank_top_k: int = 20
    smooth_l1_beta: float = 0.005   # ML-scientist review: log returns are O(0.005)
                                    # so beta=1.0 puts everything in the L2 region;
                                    # 0.005 puts the elbow inside the actual
                                    # return distribution.
    w_sector: float = 0.0           # sector classification auxiliary loss weight
    w_contrastive: float = 0.0      # supervised intra-cohort InfoNCE weight
    contrastive_tau: float = 0.1
    contrastive_ret_sim_thresh: float = 0.5
    # Gradient accumulation: cohorts to average before each optimizer step.
    # 1 = single-cohort step (original behavior); higher = denser supervision
    # per attention parameter without increasing memory beyond one cohort.
    grad_accum_steps: int = 1
    # Val subsample (caps memory + time per val sweep)
    val_max_cohorts: int = 100
    # Precision
    autocast_dtype: str = "bf16"   # "bf16" | "fp16" | "fp32"
    # IO
    log_dir: Path = field(default_factory=lambda: Path("runs/cohort_modeA"))
    ckpt_every_n_steps: int = 1000
    # Mode (informational, used in log filename)
    mode: str = "A"
    # Training order:
    #   time_ordered=False, single_epoch=False (default): random shuffle, cycle forever
    #   time_ordered=True,  single_epoch=True:            chronological, stop after one pass
    #   time_ordered=True,  single_epoch=False:           chronological per epoch, cycle
    time_ordered: bool = False
    single_epoch: bool = False
    # Wandb (off by default; set wandb_project to enable)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    # Optimizer choice + state persistence across windows.
    optimizer_name: str = "adamw"   # "adamw" | "adagrad"
    resume_optimizer: bool = False  # if True and init_ckpt provides optimizer
                                     # state, load it instead of starting fresh.


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val_ic_mean: float = -1e9
    best_val_step: int = 0
    last_train_loss: float = float("nan")


def build_anchor_pairs(
    train_start: date,
    train_end: date,
    calendar,
    *,
    anchor_grid: list[dtime] | None = None,
) -> list[tuple[date, dtime]]:
    """Enumerate (trading_date, anchor_time_ist) for all weekdays in [start, end]."""
    if anchor_grid is None:
        anchor_grid = intraday_anchor_grid()
    pairs: list[tuple[date, dtime]] = []
    cur = train_start
    while cur <= train_end:
        if calendar.is_trading_day(cur):
            for a in anchor_grid:
                pairs.append((cur, a))
        cur = cur + timedelta(days=1)
    return pairs


def _cosine_lr(step: int, *, max_steps: int, warmup_pct: float, lr_min_ratio: float) -> float:
    """Cosine decay with linear warmup. Returns multiplier in (0, 1]."""
    warmup_steps = max(1, int(warmup_pct * max_steps))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min_ratio + (1.0 - lr_min_ratio) * cos


def _autocast_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


class CohortTrainer:
    """Walk-forward cohort trainer.

    Construct with model + assembler + train/val anchor lists + config. Call
    `.fit()` to run the training loop end-to-end.
    """

    def __init__(
        self,
        *,
        model: CohortModel,
        assembler: CohortAssembler,
        train_anchors: list[tuple[date, dtime]],
        val_anchors: list[tuple[date, dtime]],
        config: TrainerConfig,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.assembler = assembler
        self.train_anchors = list(train_anchors)
        self.val_anchors = list(val_anchors)
        self.cfg = config
        self.device = device
        self.seed = seed
        self.state = TrainState()
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.cfg.log_dir / "metrics.jsonl"
        self._ckpt_dir = self.cfg.log_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        # wandb setup (optional)
        self._wandb = None
        if self.cfg.wandb_project:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(
                    project=self.cfg.wandb_project,
                    name=self.cfg.wandb_run_name or self.cfg.log_dir.name,
                    tags=list(self.cfg.wandb_tags),
                    dir=str(self.cfg.log_dir),
                    config={
                        "mode": self.cfg.mode,
                        "lr": self.cfg.lr,
                        "weight_decay": self.cfg.weight_decay,
                        "warmup_pct": self.cfg.warmup_pct,
                        "grad_clip": self.cfg.grad_clip,
                        "max_steps": self.cfg.max_steps,
                        "w_reg": self.cfg.w_reg,
                        "w_rank": self.cfg.w_rank,
                        "w_bias": self.cfg.w_bias,
                        "rank_top_k": self.cfg.rank_top_k,
                        "val_max_cohorts": self.cfg.val_max_cohorts,
                        "time_ordered": self.cfg.time_ordered,
                        "single_epoch": self.cfg.single_epoch,
                    },
                )
            except Exception as e:  # noqa: BLE001
                print(f"[wandb disabled — init failed: {e}]", flush=True)
                self._wandb = None
        # Optimizer + scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.loss_fn = CohortLoss(
            w_reg=self.cfg.w_reg,
            w_rank=self.cfg.w_rank,
            w_bias=self.cfg.w_bias,
            w_sector=self.cfg.w_sector,
            w_contrastive=self.cfg.w_contrastive,
            contrastive_tau=self.cfg.contrastive_tau,
            contrastive_ret_sim_thresh=self.cfg.contrastive_ret_sim_thresh,
            rank_top_k=self.cfg.rank_top_k,
            smooth_l1_beta=self.cfg.smooth_l1_beta,
        )
        self._autocast_dtype = _autocast_dtype(self.cfg.autocast_dtype)
        self._rng = np.random.default_rng(seed)

    # ------------- Public API -------------

    def fit(self) -> TrainState:
        """Run the training loop until `max_steps`."""
        self._log_event({"event": "fit_start",
                         "mode": self.cfg.mode,
                         "n_train_anchors": len(self.train_anchors),
                         "n_val_anchors": len(self.val_anchors)})
        t0 = time.time()
        epoch_order = self._shuffle()
        step_in_epoch = 0
        while self.state.step < self.cfg.max_steps:
            try:
                if step_in_epoch >= len(epoch_order):
                    if self.cfg.single_epoch:
                        self._log_event({"event": "single_epoch_complete",
                                         "step": self.state.step,
                                         "anchors_consumed": len(epoch_order)})
                        break
                    self.state.epoch += 1
                    epoch_order = self._shuffle()
                    step_in_epoch = 0
                anchor_pair = epoch_order[step_in_epoch]
                step_in_epoch += 1
                # Pre-fetch additional anchors for gradient-accumulation.
                K = max(1, self.cfg.grad_accum_steps)
                self._accum_extras = []
                while len(self._accum_extras) < K - 1 and step_in_epoch < len(epoch_order):
                    self._accum_extras.append(epoch_order[step_in_epoch])
                    step_in_epoch += 1
                ok = self._train_step(anchor_pair)
                if not ok:
                    continue
                self.state.step += 1
                # Validation + checkpointing
                if self.state.step % self.cfg.val_every_n_steps == 0:
                    self._run_val_and_log()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if self.state.step % self.cfg.ckpt_every_n_steps == 0:
                    self._save_ckpt(tag=f"step={self.state.step}")
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                self._log_event({"event": "exception", "step": self.state.step,
                                 "error": str(e), "traceback": tb[:5000]})
                print(f"[EXCEPTION step={self.state.step}] {e}\n{tb}", flush=True)
                # Try to recover: clear CUDA cache and continue. If 5 consecutive
                # exceptions, abort.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not hasattr(self, "_consec_errs"):
                    self._consec_errs = 0
                self._consec_errs += 1
                if self._consec_errs >= 5:
                    self._log_event({"event": "aborted_too_many_errors",
                                     "step": self.state.step})
                    break
            else:
                self._consec_errs = 0
        # Final val + ckpt
        self._run_val_and_log()
        self._save_ckpt(tag="last")
        self._log_event({"event": "fit_end",
                         "step": self.state.step,
                         "best_val_step": self.state.best_val_step,
                         "best_val_ic_mean": self.state.best_val_ic_mean,
                         "elapsed_sec": time.time() - t0})
        return self.state

    def predict_per_anchor(
        self,
        anchors: Iterable[tuple[date, dtime]],
    ) -> list[dict]:
        """Run model in eval mode over anchors; return per-cohort prediction
        dicts (anchor_ts, symbols, pred, target, validity)."""
        self.model.eval()
        out: list[dict] = []
        with torch.no_grad():
            for d, t in anchors:
                anchor_utc = self._ist_to_utc(d, t)
                sample = self.assembler.assemble(anchor_utc)
                if sample is None:
                    continue
                batch = self._sample_to_batch(sample, anchor_utc)
                model_out = self.model(batch)
                pred, _, _ = self._unpack_model_out(model_out)     # (1, N, H)
                out.append({
                    "anchor_ts": anchor_utc,
                    "symbols": list(sample.symbols),
                    "pred": pred.detach().cpu().float().numpy()[0],
                    "target": sample.labels.cpu().numpy(),
                    "validity": sample.validity.cpu().numpy(),
                    "label_validity": sample.label_validity_per_stock.cpu().numpy(),
                })
        return out

    # ------------- Internals -------------

    def _shuffle(self) -> list[tuple[date, dtime]]:
        if self.cfg.time_ordered:
            # Chronological order: sort by (date, anchor_time).
            return sorted(self.train_anchors, key=lambda dt: (dt[0], dt[1]))
        order = list(range(len(self.train_anchors)))
        self._rng.shuffle(order)
        return [self.train_anchors[i] for i in order]

    def _ist_to_utc(self, d: date, t: dtime) -> datetime:
        return (
            datetime.combine(d, t) - timedelta(hours=5, minutes=30)
        ).replace(tzinfo=UTC)

    def _sample_to_batch(self, sample, anchor_utc: datetime) -> dict:
        return {
            "x_by_scale": {sc: v.unsqueeze(0).to(self.device)
                           for sc, v in sample.x_by_scale.items()},
            "stock_ids":  sample.stock_ids.unsqueeze(0).to(self.device),
            "sector_ids": sample.sector_ids.unsqueeze(0).to(self.device),
            "anchor_ts":  torch.tensor(
                [int(anchor_utc.timestamp() * 1_000_000)],
                dtype=torch.int64, device=self.device,
            ),
            "regime_feats": sample.regime_feats.unsqueeze(0).to(self.device),
        }

    def _train_step(self, anchor_pair: tuple[date, dtime]) -> bool:
        """One MACRO step = `grad_accum_steps` cohort micro-steps + optimizer.step.

        Returns False if no usable cohort was assembled in this macro step.
        """
        K = max(1, self.cfg.grad_accum_steps)
        # LR update — apply once per macro step.
        lr_mult = _cosine_lr(
            self.state.step,
            max_steps=self.cfg.max_steps,
            warmup_pct=self.cfg.warmup_pct,
            lr_min_ratio=self.cfg.lr_min_ratio,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.cfg.lr * lr_mult

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        consumed = 0
        agg_loss = 0.0
        agg = {"reg": 0.0, "rank": 0.0, "bias": 0.0, "sector": 0.0, "contrastive": 0.0}
        # First cohort: the caller-supplied anchor_pair.
        anchor_pairs: list[tuple[date, dtime]] = [anchor_pair]
        # Additional cohorts (if K > 1): pull from the rotating epoch order.
        # Stash them via the caller below — handled here by attribute hook.
        extras: list[tuple[date, dtime]] = self._accum_extras  # set by caller
        anchor_pairs.extend(extras[: K - 1])

        for ap in anchor_pairs:
            d, t = ap
            anchor_utc = self._ist_to_utc(d, t)
            sample = self.assembler.assemble(anchor_utc)
            if sample is None:
                continue
            batch = self._sample_to_batch(sample, anchor_utc)
            labels = sample.labels.unsqueeze(0).to(self.device)
            validity = sample.validity.unsqueeze(0).to(self.device)
            label_v = sample.label_validity_per_stock.unsqueeze(0).to(self.device)
            labels_clean = torch.where(label_v, labels, torch.zeros_like(labels))
            combined_validity = validity & (label_v.float().mean(dim=1) > 0.8)

            def _compute():
                model_out = self.model(batch)
                pred, sector_logits, contrast_e = self._unpack_model_out(model_out)
                sector_labels = batch["sector_ids"] if sector_logits is not None else None
                contrast_sec = batch["sector_ids"] if contrast_e is not None else None
                return self.loss_fn(
                    pred, labels_clean, combined_validity,
                    sector_logits=sector_logits,
                    sector_labels=sector_labels,
                    contrastive_embeds=contrast_e,
                    contrastive_sector_ids=contrast_sec,
                    label_validity_per_stock=label_v,
                )

            if self._autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
                    out = _compute()
            else:
                out = _compute()
            # Scale by 1/K so the gradient is the MEAN of micro-step losses.
            (out["total"] / K).backward()
            agg_loss += float(out["total"])
            for k in agg:
                if k in out:
                    agg[k] += float(out[k])
            consumed += 1

        if consumed == 0:
            return False

        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip,
            )
        self.optimizer.step()
        # Report MEAN over consumed cohorts.
        self.state.last_train_loss = agg_loss / consumed
        if self.state.step % 50 == 0:
            self._log_event({
                "event": "step",
                "step": self.state.step,
                "epoch": self.state.epoch,
                "lr": self.cfg.lr * lr_mult,
                "train_loss": self.state.last_train_loss,
                "reg":          agg["reg"] / consumed,
                "rank":         agg["rank"] / consumed,
                "bias":         agg["bias"] / consumed,
                "sector":       agg["sector"] / consumed,
                "contrastive":  agg["contrastive"] / consumed,
                "cohorts_consumed": consumed,
            })
        return True

    @staticmethod
    def _unpack_model_out(model_out):
        """Handle (pred) or (pred, sector_logits) or dict returns.

        Returns: (pred, sector_logits, contrastive_embeds).
        """
        if isinstance(model_out, dict):
            return (
                model_out["pred"],
                model_out.get("sector_logits"),
                model_out.get("contrastive_embeds"),
            )
        if isinstance(model_out, tuple) and len(model_out) == 2:
            return model_out[0], model_out[1], None
        return model_out, None, None

    def _run_val_and_log(self) -> None:
        """Compute mean per-horizon spearman IC over val anchors.

        Subsamples val_anchors deterministically (per step) when there are
        more than `cfg.val_max_cohorts` to cap memory + wall time. Per-step
        seed makes the eval reproducible at each step.
        """
        self.model.eval()
        # Pick a deterministic-per-step subsample if val_anchors is large.
        if len(self.val_anchors) > self.cfg.val_max_cohorts:
            val_rng = np.random.default_rng(self.seed * 10_000 + self.state.step)
            idxs = val_rng.choice(
                len(self.val_anchors), size=self.cfg.val_max_cohorts, replace=False,
            )
            val_iter = [self.val_anchors[i] for i in idxs]
        else:
            val_iter = self.val_anchors
        per_horizon_preds: list[list[float]] = [[] for _ in range(N_HORIZONS)]
        per_horizon_targets: list[list[float]] = [[] for _ in range(N_HORIZONS)]
        n_cohorts = 0
        with torch.no_grad():
            for d, t in val_iter:
                anchor_utc = self._ist_to_utc(d, t)
                sample = self.assembler.assemble(anchor_utc)
                if sample is None:
                    continue
                n_cohorts += 1
                batch = self._sample_to_batch(sample, anchor_utc)
                if self._autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
                        model_out = self.model(batch)
                else:
                    model_out = self.model(batch)
                pred, _, _ = self._unpack_model_out(model_out)
                pred = pred[0].float().cpu().numpy()           # (N, H)
                target = sample.labels.cpu().numpy()           # (N, H)
                label_v = sample.label_validity_per_stock.cpu().numpy()
                validity = sample.validity.cpu().numpy()       # (H,)
                for h_idx in range(N_HORIZONS):
                    if not validity[h_idx]:
                        continue
                    mask = label_v[:, h_idx]
                    if mask.sum() < 5:
                        continue
                    p = pred[mask, h_idx]
                    tgt = target[mask, h_idx]
                    per_horizon_preds[h_idx].extend(p.tolist())
                    per_horizon_targets[h_idx].extend(tgt.tolist())

        ic_per_horizon: list[float] = []
        for h_idx in range(N_HORIZONS):
            p = np.asarray(per_horizon_preds[h_idx])
            t_arr = np.asarray(per_horizon_targets[h_idx])
            if len(p) < 50:
                ic_per_horizon.append(float("nan"))
                continue
            ic, _ = spearmanr(p, t_arr)
            ic_per_horizon.append(float(ic) if ic == ic else 0.0)

        # Score: mean IC of {next_day_eod, day_plus_3} — the two horizons
        # the strategy actually trades. ML-scientist review: prior score
        # included intraday horizons which produce ic ≈ 0 throughout training,
        # masking signal in the multi-day horizons. Restricting to working
        # horizons makes best-ckpt selection track P&L.
        score_horizons = ["next_day_eod", "day_plus_3"]
        score_indices = [i for i, h in enumerate(HORIZONS) if h.name in score_horizons]
        score_values = [ic_per_horizon[i] for i in score_indices
                        if not math.isnan(ic_per_horizon[i])]
        score = float(np.mean(score_values)) if score_values else float("nan")

        if not math.isnan(score) and score > self.state.best_val_ic_mean:
            self.state.best_val_ic_mean = score
            self.state.best_val_step = self.state.step
            self._save_ckpt(tag="best")

        self._log_event({
            "event": "val",
            "step": self.state.step,
            "epoch": self.state.epoch,
            "val_n_cohorts": n_cohorts,
            "val_ic_per_horizon": {
                HORIZONS[i].name: float(ic_per_horizon[i])
                for i in range(N_HORIZONS)
            },
            "val_ic_score_mean": score,
            "best_val_ic_score": self.state.best_val_ic_mean,
            "best_val_step": self.state.best_val_step,
        })
        self.model.train()

    def _save_ckpt(self, *, tag: str) -> None:
        path = self._ckpt_dir / f"{tag}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": {
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_val_ic_mean": self.state.best_val_ic_mean,
                "best_val_step": self.state.best_val_step,
            },
        }, path)

    def _log_event(self, ev: dict) -> None:
        ev = {"ts": datetime.now(UTC).isoformat(), **ev}
        with self._log_path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        if self._wandb is None:
            return
        # Mirror step+val events to wandb. Skip housekeeping events.
        et = ev.get("event")
        if et == "step":
            self._wandb.log(
                {
                    "train/loss":  ev.get("train_loss"),
                    "train/reg":   ev.get("reg"),
                    "train/rank":  ev.get("rank"),
                    "train/bias":  ev.get("bias"),
                    "train/lr":    ev.get("lr"),
                    "epoch":       ev.get("epoch"),
                },
                step=ev.get("step"),
            )
        elif et == "val":
            payload = {"val/n_cohorts": ev.get("val_n_cohorts"),
                       "val/score":     ev.get("val_ic_score_mean"),
                       "val/best_score": ev.get("best_val_ic_score"),
                       "val/best_step":  ev.get("best_val_step")}
            for hname, ic in (ev.get("val_ic_per_horizon") or {}).items():
                # Filter NaN — wandb refuses NaN by default.
                if ic is None or (isinstance(ic, float) and ic != ic):
                    continue
                payload[f"val/ic_{hname}"] = ic
            self._wandb.log(payload, step=ev.get("step"))
        elif et == "fit_end":
            self._wandb.log(
                {"summary/best_val_score": ev.get("best_val_ic_mean"),
                 "summary/best_val_step":  ev.get("best_val_step"),
                 "summary/total_steps":    ev.get("step"),
                 "summary/elapsed_sec":    ev.get("elapsed_sec")},
                step=ev.get("step"),
            )
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
        elif et in {"exception", "aborted_too_many_errors"}:
            self._wandb.log({"event/exception_step": ev.get("step")},
                            step=ev.get("step"))
