"""Phase 3.5 — posttrain training loop (single-fold) + random-init ablation.

A plain-PyTorch training loop wrapped around `PostTrainModule`. No Lightning
— at this scale the savings from Lightning (gradient clipping, mixed-
precision, EMA) don't outweigh the cost of fighting it for variable-N
cross-section batches.

Public entry points:

- `run_single_fold(cfg)` — train one fold under one mode, return per-horizon
  rank-IC + the trained module. Used by both the CLI and Phase 3.7's
  walk-forward sweep.

- `RunConfig` — top-level run knobs (steps, lr, fold boundaries, mode).
  Includes the `mode: "lora" | "random_init"` switch per §1.4 +
  amendment 3.

Acceptance criteria for §3.5 (per `PHASE3.md` §11):
- LoRA + heads converge for all 4 horizons.
- Random-init full-fine-tune baseline runs to completion (3 seeds).
- Basic rank-IC table printed.

The smoke test in `tests/unit/test_posttrain_loop.py` exercises this
end-to-end on a tiny synthetic universe; the real H200 run is gated on
having `data/labels` + `data/features_parquet` populated for the full
universe.

**Known carryover (review-finding Major-1 from §3.4 + §3.5):**
The vol-head target currently uses the §3.4-flagged `|y|.log()` proxy
(`PostTrainConfig.vol_target_mode="abs_y_log"`). The proxy is biased
but bounded; the magnitude-balance heuristic down-weights its
contribution so it doesn't drown out pinball + listmle. The §1.1 /
§1.2 acceptance gates don't depend on the vol head — they're rank-IC
on the rank-head's output. The vol head's output IS consumed by Phase 5
sizing, so §3.7's walk-forward sweep must plumb `vol_target_mode=
"forward_realized"` through the CrossSection carrier before shipping.
**Tracked as a §3.7-blocker.**
"""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from daity.data.cross_section import CrossSection, CrossSectionalSampler
from daity.data.parquet_store import ParquetStore
from daity.eval.metrics import RankICResult, cross_sectional_rank_ic
from daity.training.posttrain import (
    HORIZONS,
    PostTrainConfig,
    PostTrainModule,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Natural-cadence weights for the weighted-random horizon sampler
# (`PHASE3.md` §4.3). Roughly proportional to anchors-per-trading-day.
DEFAULT_CADENCE_WEIGHTS: dict[str, float] = {
    "30m":  70.0,
    "120m": 52.0,
    "1d":   1.0,
    "2d":   1.0,
}


@dataclass
class RunConfig:
    """Top-level Phase 3.5 run knobs."""

    # Required.
    feature_root: Path
    label_root: Path
    universe: list[str]

    # Time-window boundaries (one fold). Train: [train_start, train_end].
    # Val: [val_start, val_end] (for rank-IC eval; conformal owns its own
    # split in §3.6).
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    as_of: datetime

    # `lora`: load SSL checkpoint, freeze backbone, train 4 LoRA banks + heads.
    # `random_init`: ignore checkpoint, train ALL backbone weights + heads
    # at full fine-tune (per §1.4 random-init ablation; LoRA on random
    # init is degenerate).
    mode: Literal["lora", "random_init"] = "lora"

    # Underlying module config.
    module_cfg: PostTrainConfig | None = None

    # Optim.
    max_steps: int = 30_000
    lr_lora: float = 3e-4
    lr_head: float = 1e-3
    lr_backbone: float = 3e-4    # only used in random_init mode
    weight_decay: float = 1e-3
    warmup_steps_pct: float = 0.05
    lr_min_ratio: float = 0.05
    gradient_clip_val: float = 1.0
    seed: int = 0

    # Eval.
    val_every_n_steps: int = 2000
    # Cap on val cross-sections used for end-of-training rank-IC eval.
    # `None` = no cap (use all available val cross-sections).
    # Default 2000 covers the full intraday val window (~4300 30m anchors
    # would be too many, so 2000 is a representative chronological sample);
    # for daily horizons (~62 anchors) it's a no-op.
    # Original default of 50 evaluated only the first ~1% of intraday val,
    # which is statistically meaningful but NOT representative of the full
    # val window — F-9 diagnostic surfaced this. See PHASE3.md decision
    # log for the discussion.
    limit_val_cross_sections: int | None = 2000

    # Sampler.
    window_bars: dict[str, int] | None = None
    drop_invalid_threshold: float = 0.5
    cadence_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CADENCE_WEIGHTS),
    )
    calibrate_loss_weights_at_init: bool = True
    # Phase 3.4b iter-4: when set, the train + val samplers load top-50
    # XGB-importance aux features from this root and pack them on each
    # emitted CrossSection. PostTrainModule must also have
    # `cfg.use_aux_features=True` for the residual to actually apply.
    aux_root: Path | None = None
    # Phase 3.4c iter-5: label residualization. When `label_mode='resid'`,
    # the samplers read `y_resid` from `label_root_resid` for horizons in
    # `resid_horizons` (default 1d/2d). Other horizons stay on raw.
    label_mode: str = "raw"
    label_root_resid: Path | None = None
    resid_horizons: tuple[str, ...] = ("1d", "2d")

    # Logging.
    log_every_n_steps: int = 100

    # Gradient accumulation (Phase-3 H200 follow-up: amortize per-opt-step
    # Python overhead). With `grad_accum_steps=K`, the training loop draws
    # K cross-sections, does K forward+backward passes (gradients accumulate
    # in .grad buffers), then ONE optimizer.step(). Effective batch grows
    # K×; wall-clock per equivalent sample drops because the AdamW step
    # (the ~50ms Python-overhead cost at N=205) runs K× less often.
    #
    # Tradeoffs:
    # - Effective batch is K× larger → gradient noise lower → LR may need
    #   scaling. We do NOT auto-scale; the default LR was tuned at K=1.
    # - LR schedule (cosine + warmup) advances per opt step, so `max_steps`
    #   now counts opt steps, NOT samples seen. To match a K=1 run's sample
    #   count, scale max_steps by 1/K.
    grad_accum_steps: int = 1

    # GPU throughput optimizations (Phase-3 H200 follow-up).
    # See `_AsyncHorizonStream`, `_MPHorizonStream`, `torch.autocast`,
    # `torch.compile` below.
    #
    # Two modes for data prefetch:
    # - `_MPHorizonStream` (default): per-horizon `torch.utils.data.DataLoader`
    #   with multiprocess workers (no GIL contention). Each worker process
    #   reconstructs its own `CrossSectionalSampler`. Best throughput; uses
    #   ~1.2 GB RAM per worker for feature caches.
    # - `_AsyncHorizonStream` (fallback): single-process threading. Use when
    #   `num_data_workers=0` or when fork/spawn is unavailable.
    use_async_data_prefetch: bool = True   # legacy threading mode flag
    async_prefetch_queue_size: int = 4
    # Total DataLoader worker processes across all 4 horizons.
    # Allocated proportionally to `cadence_weights` (≥1 per horizon).
    # 0 = use single-process threading (`_AsyncHorizonStream`) instead.
    # Empirical Phase-3 H200 finding: MP workers (16) gave the same ~10
    # sps as threading. The bottleneck is the main loop (forward + opt
    # step + H2D), not the data path. Reverted default to 0 (threading);
    # MP startup overhead is ~60-90 sec which isn't worth paying.
    num_data_workers: int = 0
    data_prefetch_factor: int = 2
    use_bf16_autocast: bool = True
    # Fix-2 attempt (H200 diagnosis): `torch.compile(mode="reduce-overhead",
    # dynamic=False, fullgraph=False)` with padded shapes was benchmarked
    # and gave only ~5-10% steady-state gain + 6 min compile cold start
    # per run. At 30K-step training duration that's net-wash to slightly-
    # worse. Disabled by default. The code path stays; turn on with
    # `--pad-n-for-compile=205 --use-torch-compile` for the experiment.
    # Probable reason it didn't deliver the projected 2-3×: `fullgraph=False`
    # means CUDA Graphs gets broken at each peft branch — partial graphs
    # only. Re-enable when fullgraph=True path is clean (would require
    # rewriting peft adapter routing to be graph-friendly).
    use_torch_compile: bool = False

    # Static-shape padding for `torch.compile(dynamic=False)`. When > 0,
    # cross-sections are padded to this N before forward; outputs are
    # sliced to the original N before loss. Recommended: set to
    # `len(universe)` (typically 205).
    # `0` disables padding (use eager forward with dynamic shapes).
    pad_n_for_compile: int = 0

    # Save the final trained state_dict + LoRA banks so a future
    # `daity-posttrain-eval-only` can re-evaluate without re-training.
    # `None` = don't save. Default: write next to the report JSON with
    # extension `.state.pt`.
    save_model_path: Path | None = None


# ----- LR schedule -----


def _cosine_with_warmup_lr(
    step: int, *, max_steps: int, warmup_steps: int,
    base_lr: float, lr_min_ratio: float,
) -> float:
    """Cosine decay with linear warmup, floored at `base_lr * lr_min_ratio`."""
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * lr_min_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cos_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    return base_lr * (lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor)


# ----- Builders -----


def _build_module(cfg: RunConfig) -> PostTrainModule:
    """Build the `PostTrainModule` according to `cfg.mode`.

    `lora` mode loads the SSL checkpoint; `random_init` constructs the
    module then re-initializes the backbone with random weights and
    UNFREEZES it (per §6.7 amendment 12 / amendment 3).
    """
    if cfg.module_cfg is None:
        msg = "RunConfig.module_cfg is required (carries the SSL checkpoint path)"
        raise ValueError(msg)

    module = PostTrainModule(cfg.module_cfg)

    if cfg.mode == "random_init":
        # Reset BOTH backbone AND tokenizer to fresh random init, then
        # unfreeze. Review-finding Major-2: leaving the tokenizer at
        # SSL-init biases §1.4 in SSL's favor — the tokenizer's patch
        # projection (~556K params at d_model=480) is a meaningful share
        # of the SSL-trained surface and should be reset for a true
        # "useless representation" baseline.
        torch.manual_seed(cfg.seed)
        for sub_module in list(module.backbone.modules()) + list(module.tokenizer.modules()):
            if isinstance(sub_module, nn.Linear):
                nn.init.kaiming_normal_(sub_module.weight, nonlinearity="relu")
                if sub_module.bias is not None:
                    nn.init.zeros_(sub_module.bias)
            elif isinstance(sub_module, nn.LayerNorm):
                nn.init.ones_(sub_module.weight)
                nn.init.zeros_(sub_module.bias)
        # Reset the tokenizer's learnable parameters (forecast token,
        # resolution embed) to N(0, 0.02) like the SSL init.
        with torch.no_grad():
            if hasattr(module.tokenizer, "forecast_token"):
                nn.init.normal_(module.tokenizer.forecast_token, std=0.02)
            if hasattr(module.tokenizer, "resolution_embed"):
                nn.init.normal_(module.tokenizer.resolution_embed, std=0.02)
        # Unfreeze backbone + tokenizer for full fine-tune.
        for p in module.backbone.parameters():
            p.requires_grad_(True)
        for p in module.tokenizer.parameters():
            p.requires_grad_(True)
        # Disable LoRA adapter ROUTING (skip them at forward time) and
        # freeze their params so the optimizer never touches them. The
        # `_use_lora` flag is the only thing forward() checks; with it
        # False, forward never calls `set_adapter` or `_enable_all_lora_grads`.
        module.peft_backbone.disable_adapter_layers()
        for name, param in module.peft_backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad_(False)
        module._use_lora = False
        log.info(
            "random_init mode: backbone re-initialized + unfrozen; "
            "LoRA disabled (routing skipped + params frozen)",
        )

    return module


def _build_optimizer(module: PostTrainModule, cfg: RunConfig) -> torch.optim.Optimizer:
    """AdamW with parameter groups (lora vs heads vs backbone)."""
    lora_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif "_heads" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups: list[dict[str, object]] = []
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lr_lora, "weight_decay": 0.0,
                       "_name": "lora"})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head,
                       "weight_decay": cfg.weight_decay, "_name": "heads"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone,
                       "weight_decay": cfg.weight_decay, "_name": "backbone"})
    if not groups:
        msg = "No trainable parameters found — check freeze logic + mode."
        raise RuntimeError(msg)
    # NB: `fused=True` AdamW was tested on H200 and observed to slow
    # throughput at batch N=205 (8 sps vs 14 sps unfused). Likely the
    # fused-kernel overhead doesn't amortize at our trainable-param scale
    # (~3M LoRA). Keeping unfused; revisit if a future redesign batches
    # multiple cross-sections per step.
    return torch.optim.AdamW(groups)


# ----- Weighted-random horizon stream -----


class _HorizonStream:
    """Wraps the four sampler iterators and yields `(horizon, cs)` tuples
    via weighted-random sampling proportional to `cadence_weights`.

    Per-horizon iterators are restarted automatically when exhausted —
    we treat training as "unbounded over the train window" so step
    counts come from `cfg.max_steps`, not from the sampler.
    """

    def __init__(
        self,
        sampler: CrossSectionalSampler,
        cadence_weights: dict[str, float],
        seed: int,
    ) -> None:
        self.sampler = sampler
        self.cadence_weights = cadence_weights
        self.rng = random.Random(seed)
        self.horizons = list(cadence_weights.keys())
        self.weights = [cadence_weights[h] for h in self.horizons]
        self._iters: dict[str, Iterable[CrossSection]] = {}
        self._restart_iters()

    def _restart_iters(self) -> None:
        self._iters = {
            h: iter(getattr(self.sampler, f"iter_{h}")())
            for h in self.horizons
        }

    def next(self) -> tuple[str, CrossSection] | None:
        """Pick a horizon by weight, return its next cross-section.

        Returns None if ALL iterators are exhausted (i.e., the train
        window doesn't have a single emittable cross-section at any
        horizon — usually means the universe is too small or the
        window doesn't contain enough trading days).
        """
        # Try up to len(horizons) horizons before giving up.
        tried: set[str] = set()
        while len(tried) < len(self.horizons):
            h = self.rng.choices(self.horizons, weights=self.weights, k=1)[0]
            if h in tried:
                continue
            try:
                cs = next(self._iters[h])
                return h, cs
            except StopIteration:
                tried.add(h)
                # Restart this horizon's iterator; if it still raises on
                # the next call, it's truly empty and we'll skip it.
                self._iters[h] = iter(getattr(self.sampler, f"iter_{h}")())
                try:
                    cs = next(self._iters[h])
                    return h, cs
                except StopIteration:
                    continue
        return None


def _cs_to_device(
    cs: CrossSection, device: torch.device, *, non_blocking: bool = False,
) -> CrossSection:
    """Move a CrossSection's tensors to `device`. Safe on CPU device (no-op)."""
    if device.type == "cpu":
        return cs
    return CrossSection(
        t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
        inputs={
            sc: t.to(device, non_blocking=non_blocking)
            for sc, t in cs.inputs.items()
        },
        labels=cs.labels.to(device, non_blocking=non_blocking),
        ranks=cs.ranks.to(device, non_blocking=non_blocking),
        valid_mask=cs.valid_mask.to(device, non_blocking=non_blocking),
        vol_bucket=cs.vol_bucket.to(device, non_blocking=non_blocking),
    )


def _pad_cross_section(cs: CrossSection, pad_n: int) -> CrossSection:
    """Pad a CrossSection's tensors along dim 0 to `pad_n` rows.

    Fix-2 (H200): static shapes for `torch.compile(dynamic=False)`. Pads
    with zeros; the training loop slices outputs back to original N before
    loss so padding rows never contribute to gradients.
    """
    n = cs.labels.shape[0]
    if n >= pad_n:
        return cs
    n_pad = pad_n - n

    def _pad(t: torch.Tensor, n_pad: int) -> torch.Tensor:
        pad_shape = (n_pad, *t.shape[1:])
        pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=0)

    return CrossSection(
        t=cs.t, horizon=cs.horizon,
        symbols=list(cs.symbols) + [""] * n_pad,
        inputs={sc: _pad(t, n_pad) for sc, t in cs.inputs.items()},
        labels=_pad(cs.labels, n_pad),
        ranks=_pad(cs.ranks, n_pad),
        valid_mask=_pad(cs.valid_mask, n_pad),
        vol_bucket=_pad(cs.vol_bucket, n_pad),
    )


def _truncate_cross_section(cs: CrossSection, max_n: int) -> CrossSection:
    """Truncate a CrossSection to at most `max_n` rows. Defensive; the
    sampler should usually emit N ≤ universe size."""
    return CrossSection(
        t=cs.t, horizon=cs.horizon,
        symbols=cs.symbols[:max_n],
        inputs={sc: t[:max_n] for sc, t in cs.inputs.items()},
        labels=cs.labels[:max_n],
        ranks=cs.ranks[:max_n],
        valid_mask=cs.valid_mask[:max_n],
        vol_bucket=cs.vol_bucket[:max_n],
    )


def _cs_identity_collate(x: CrossSection) -> CrossSection:
    """Identity collate — with `batch_size=None`, the DataLoader yields each
    item directly and the default-collate tries to merge dataclass fields,
    which fails. We just pass through the single CrossSection."""
    return x


class _PerHorizonDataset(torch.utils.data.IterableDataset):
    """IterableDataset wrapping `CrossSectionalSampler.iter_<horizon>`.

    Each DataLoader worker process gets a copy of the dataset config and
    instantiates its own `CrossSectionalSampler` instance (re-reads parquet
    files into a per-worker cache). Workers shard the anchor space so they
    emit DISJOINT cross-sections — worker `i` emits anchor index `i,
    i+n_workers, i+2*n_workers, ...`.

    Caches per worker: ~1.2 GB (205 syms × 4 scales × float32 features).
    With 16 workers that's ~20 GB RAM — fine on the 2 TB H200 box.
    """

    def __init__(
        self,
        *,
        feature_root: Path,
        label_root: Path,
        universe: list[str],
        as_of: datetime,
        window_bars: dict[str, int] | None,
        min_end_ts: datetime | None,
        drop_invalid_threshold: float,
        horizon: str,
    ) -> None:
        super().__init__()
        self.feature_root = feature_root
        self.label_root = label_root
        self.universe = universe
        self.as_of = as_of
        self.window_bars = window_bars
        self.min_end_ts = min_end_ts
        self.drop_invalid_threshold = drop_invalid_threshold
        self.horizon = horizon

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0

        sampler = CrossSectionalSampler(
            feature_store=ParquetStore(self.feature_root),
            label_root=self.label_root,
            universe=self.universe,
            as_of=self.as_of,
            window_bars=self.window_bars
                if self.window_bars is not None
                else {"5m": 256, "15m": 192, "60m": 168, "day": 64},
            min_end_ts=self.min_end_ts,
            drop_invalid_threshold=self.drop_invalid_threshold,
        )
        it = getattr(sampler, f"iter_{self.horizon}")()
        for idx, cs in enumerate(it):
            if idx % n_workers != worker_id:
                continue
            yield cs


class _MPHorizonStream:
    """Multi-process data prefetch via per-horizon DataLoaders.

    Drop-in replacement for `_HorizonStream` / `_AsyncHorizonStream`. Each
    horizon has its own DataLoader with `num_workers` proportional to its
    cadence weight. The main thread pulls one cross-section per training
    step from a weighted-random-picked horizon's iterator.

    Why this beats `_AsyncHorizonStream`: worker PROCESSES don't share the
    GIL, so `_build_cross_section`'s per-symbol Python loops actually run
    in parallel across cores. On H200 (224 vCPU) with 16 workers, the
    data path goes from ~10 cross-sections/sec (single-thread bottleneck)
    to ~50+ cross-sections/sec (aggregate worker throughput).

    Each worker reconstructs its own `CrossSectionalSampler` and reads
    parquet files into its own cache (~1.2 GB per worker for feature data).
    Total RAM: ~20 GB for 16 workers — fine on the 2 TB box.
    """

    def __init__(
        self,
        *,
        cfg: RunConfig,
        sampler_as_of: datetime,
        sampler_min_end_ts: datetime | None,
        cadence_weights: dict[str, float],
        seed: int,
        device: torch.device,
        num_workers: int,
        prefetch_factor: int = 2,
    ) -> None:
        self.cadence_weights = cadence_weights
        self.device = device
        self._rng = random.Random(seed)
        self._horizons = list(cadence_weights.keys())
        self._weights = [cadence_weights[h] for h in self._horizons]

        total_w = float(sum(cadence_weights.values()))
        # Allocate workers proportionally to cadence weights; ≥1 per horizon.
        # `min_workers_total` is the integer floor; we may end up with
        # slightly fewer total workers than requested (e.g., 17 instead of
        # 16) because we floor() per-horizon then enforce ≥1. Acceptable.
        per_horizon_workers: dict[str, int] = {
            h: max(1, int(num_workers * cadence_weights[h] / total_w))
            for h in self._horizons
        }
        log.info("MPHorizonStream worker allocation: %s", per_horizon_workers)

        self.loaders: dict[str, torch.utils.data.DataLoader] = {}
        for h in self._horizons:
            ds = _PerHorizonDataset(
                feature_root=cfg.feature_root,
                label_root=cfg.label_root,
                universe=cfg.universe,
                as_of=sampler_as_of,
                window_bars=cfg.window_bars,
                min_end_ts=sampler_min_end_ts,
                drop_invalid_threshold=cfg.drop_invalid_threshold,
                horizon=h,
            )
            self.loaders[h] = torch.utils.data.DataLoader(
                ds,
                batch_size=None,
                num_workers=per_horizon_workers[h],
                prefetch_factor=prefetch_factor,
                pin_memory=False,  # we'll do non_blocking H2D in main thread
                collate_fn=_cs_identity_collate,
                persistent_workers=True,
                # `spawn` is safer than `fork` for CUDA + complex state,
                # but pays ~5 sec per worker on startup. For 16 workers
                # one-time cost is ~80 sec, amortized over a 30K-step run.
                multiprocessing_context="spawn",
            )

        self._iters: dict[str, Iterable[CrossSection]] = {
            h: iter(self.loaders[h]) for h in self._horizons
        }
        self._closed = False

    def next(self) -> tuple[str, CrossSection] | None:
        """Weighted-random horizon pick, return one device-resident CrossSection."""
        tried: set[str] = set()
        while len(tried) < len(self._horizons):
            h = self._rng.choices(self._horizons, weights=self._weights, k=1)[0]
            if h in tried:
                continue
            try:
                cs = next(self._iters[h])
            except StopIteration:
                # Restart this horizon's iterator (unbounded training).
                self._iters[h] = iter(self.loaders[h])
                try:
                    cs = next(self._iters[h])
                except StopIteration:
                    tried.add(h)
                    continue
            return h, _cs_to_device(cs, self.device, non_blocking=True)
        return None

    def close(self) -> None:
        """Tear down DataLoader workers cleanly."""
        if self._closed:
            return
        self._closed = True
        for loader in self.loaders.values():
            # Releasing the iterator triggers DataLoader's worker shutdown.
            del loader
        self._iters.clear()
        self.loaders.clear()

    def __del__(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):
            self.close()


class _AsyncHorizonStream:
    """Drop-in replacement for `_HorizonStream` that pre-builds + moves-to-device
    cross-sections in background producer threads (one per horizon).

    Why threads, not multiprocessing:
    - The per-anchor build is mostly numpy + `torch.from_numpy` work, which
      releases the GIL during the heavy parts.
    - Multiprocessing would duplicate the per-symbol caches (~3 GB each)
      across workers — wasteful when threads can share the sampler's cache.
    - The actual benefit is pipelining: while the GPU is doing forward+
      backward, the producer threads prep the next cross-section and queue
      it (already moved to device via `.to(device, non_blocking=True)`).

    With `queue_size=4` per horizon, the main loop almost never blocks
    waiting for data — taking the GPU from ~38% util to ~70-90% util on
    a single H200 process (empirical Phase-3 H200 follow-up).
    """

    def __init__(
        self,
        sampler: CrossSectionalSampler,
        cadence_weights: dict[str, float],
        *,
        seed: int,
        device: torch.device,
        queue_size: int = 4,
    ) -> None:
        self.sampler = sampler
        self.cadence_weights = cadence_weights
        self.device = device
        self._rng = random.Random(seed)
        self._horizons = list(cadence_weights.keys())
        self._weights = [cadence_weights[h] for h in self._horizons]
        self._queues: dict[str, queue.Queue] = {
            h: queue.Queue(maxsize=queue_size) for h in self._horizons
        }
        self._stop = threading.Event()
        self._exhausted: dict[str, bool] = {h: False for h in self._horizons}
        self._threads: dict[str, threading.Thread] = {}
        for h in self._horizons:
            t = threading.Thread(target=self._producer, args=(h,), daemon=True)
            t.start()
            self._threads[h] = t

    def _producer(self, horizon: str) -> None:
        """Endlessly yield cross-sections for `horizon` into the queue.

        Restarts the underlying iterator when exhausted (like `_HorizonStream`).
        Stops cleanly on `self._stop.set()`.
        """
        while not self._stop.is_set():
            try:
                it = iter(getattr(self.sampler, f"iter_{horizon}")())
                emitted = False
                for cs in it:
                    if self._stop.is_set():
                        return
                    cs_dev = self._to_device(cs)
                    # `queue.put` blocks if full — that's the desired
                    # back-pressure (don't overflow if main is slow).
                    while not self._stop.is_set():
                        try:
                            self._queues[horizon].put(cs_dev, timeout=1.0)
                            emitted = True
                            break
                        except queue.Full:
                            continue
                if not emitted:
                    # Iterator yielded zero CrossSections — horizon truly empty.
                    self._exhausted[horizon] = True
                    return
                # else: outer loop restarts iter for endless training.
            except Exception as exc:  # noqa: BLE001
                log.exception("async producer for %s crashed: %s", horizon, exc)
                self._exhausted[horizon] = True
                return

    def _to_device(self, cs: CrossSection) -> CrossSection:
        """Move a CrossSection to GPU on the producer thread.

        NB: `pin_memory()` before `.to(non_blocking=True)` was tested on
        H200 — observed to slow throughput at N=205 (8 sps vs 14 sps
        unpinned). The pin allocation cost dominates the async-overlap win
        at our batch size. Keeping unpinned synchronous copies.
        """
        if self.device.type == "cpu":
            return cs
        return CrossSection(
            t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
            inputs={
                sc: t.to(self.device, non_blocking=True)
                for sc, t in cs.inputs.items()
            },
            labels=cs.labels.to(self.device, non_blocking=True),
            ranks=cs.ranks.to(self.device, non_blocking=True),
            valid_mask=cs.valid_mask.to(self.device, non_blocking=True),
            vol_bucket=cs.vol_bucket.to(self.device, non_blocking=True),
        )

    def next(self) -> tuple[str, CrossSection] | None:
        """Weighted-random horizon pick, then pull from that horizon's queue.

        Returns None if every horizon is exhausted AND its queue is empty
        (i.e., training data is fully consumed and producers won't refill).
        Otherwise blocks until the chosen horizon has a cross-section
        ready (typically <1ms once the pipeline is warm).
        """
        # Try up to len(horizons) different horizons before giving up.
        tried: set[str] = set()
        while len(tried) < len(self._horizons):
            h = self._rng.choices(self._horizons, weights=self._weights, k=1)[0]
            if h in tried:
                continue
            try:
                cs = self._queues[h].get(timeout=30.0)
                return h, cs
            except queue.Empty:
                # Producer is dead AND queue is empty → this horizon's done.
                if self._exhausted[h]:
                    tried.add(h)
                    continue
                # Producer is alive but slow — keep waiting.
                tried.add(h)
                continue
        return None

    def close(self) -> None:
        """Stop all producer threads cleanly. Idempotent."""
        self._stop.set()
        # Drain queues so put-blocked producers can return.
        for q in self._queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        for t in self._threads.values():
            t.join(timeout=5.0)

    def __del__(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):
            self.close()


# ----- Eval -----


@dataclass
class HorizonEval:
    horizon: str
    rank_ic: RankICResult
    n_cross_sections_used: int


def evaluate(
    module: PostTrainModule,
    val_sampler: CrossSectionalSampler,
    *,
    limit_cross_sections: int | None = None,
) -> dict[str, HorizonEval]:
    """Compute per-horizon Spearman rank-IC on the val sampler.

    Predictions come from the rank head (`out["rank"]`), aligned to the
    cross-section's labels. `rank_ic` is computed per cross-section, then
    averaged across the val window.
    """
    module.eval()
    results: dict[str, HorizonEval] = {}
    device = next(module.parameters()).device

    for h in HORIZONS:
        preds_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        ts_list: list[np.datetime64] = []
        n_used = 0
        with torch.no_grad():
            for cs in getattr(val_sampler, f"iter_{h}")():
                if limit_cross_sections is not None and n_used >= limit_cross_sections:
                    break
                inputs_on_device = {sc: t.to(device) for sc, t in cs.inputs.items()}
                cs_dev = CrossSection(
                    t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
                    inputs=inputs_on_device,
                    labels=cs.labels.to(device),
                    ranks=cs.ranks.to(device),
                    valid_mask=cs.valid_mask.to(device),
                    vol_bucket=cs.vol_bucket.to(device),
                )
                out = module(cs_dev, horizon=h)
                preds_list.append(out["rank"].detach().cpu().numpy())
                labels_list.append(cs.labels.numpy())
                # Repeat the cross-section's t for each symbol's row.
                ts_list.append(
                    np.full(len(cs.symbols), np.datetime64(cs.t.replace(tzinfo=None))),
                )
                n_used += 1
        if not preds_list:
            results[h] = HorizonEval(
                horizon=h,
                rank_ic=RankICResult(0.0, 0.0, 0.0, 0, 0),
                n_cross_sections_used=0,
            )
            continue
        preds = np.concatenate(preds_list)
        labels = np.concatenate(labels_list)
        ts = np.concatenate(ts_list)
        rank_ic, per_cs = cross_sectional_rank_ic(
            predictions=preds, realizations=labels, timestamps=ts,
            return_per_cs=True,
        )
        # Distribution stats: surface so we can see WHICH cross-sections
        # the model is anti-predictive on (vs. uniformly bad).
        per_cs_arr = np.asarray(per_cs, dtype=np.float64) if per_cs else np.zeros(0)
        n_pos = int((per_cs_arr > 0).sum()) if per_cs_arr.size else 0
        n_neg = int((per_cs_arr < 0).sum()) if per_cs_arr.size else 0
        log.info(
            "eval h=%s n_cs=%d mean_ic=%+.4f std=%.4f pos=%d neg=%d "
            "p10=%+.4f p50=%+.4f p90=%+.4f",
            h, rank_ic.n_cross_sections, rank_ic.mean_ic, rank_ic.std_ic,
            n_pos, n_neg,
            float(np.quantile(per_cs_arr, 0.10)) if per_cs_arr.size else 0.0,
            float(np.quantile(per_cs_arr, 0.50)) if per_cs_arr.size else 0.0,
            float(np.quantile(per_cs_arr, 0.90)) if per_cs_arr.size else 0.0,
        )
        results[h] = HorizonEval(
            horizon=h, rank_ic=rank_ic, n_cross_sections_used=n_used,
        )
    module.train()
    return results


# ----- Main loop -----


@dataclass
class RunResult:
    """Summary of one Phase-3.5 single-fold run."""
    mode: str
    seed: int
    final_step: int
    per_horizon_rank_ic: dict[str, float]
    per_horizon_n_cross_sections: dict[str, int]
    final_loss_by_horizon: dict[str, float]


def run_single_fold(cfg: RunConfig) -> tuple[PostTrainModule, RunResult]:
    """Train one fold; return the trained module + per-horizon rank-IC."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Build module.
    module = _build_module(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    # GPU runtime knobs experimented with on H200 (Phase-3 follow-up):
    # `cudnn.benchmark`, `allow_tf32`, `pin_memory`, `fused AdamW`,
    # `torch.compile`, MP DataLoader workers — none gave net throughput
    # gains at our batch scale (N=205, ~3M trainable LoRA params). The
    # bottleneck is the main-loop Python overhead + small-batch GPU
    # underutilization; the data path is not the constraint. Steady state
    # is ~14 sps (LoRA) / ~10 sps (random_init). Revisit when batching
    # multiple cross-sections per step lands.

    # Build train + val samplers — review Critical-1: must cap each
    # sampler's anchors to its window. Without `as_of=train_end` the train
    # stream emits anchors from the val/test window, which would leak the
    # acceptance gate's IC numbers. Without `min_end_ts=val_start` the val
    # sampler emits anchors from the train window, double-counting.
    feature_store = ParquetStore(cfg.feature_root)
    window_bars = cfg.window_bars or {"5m": 256, "15m": 192, "60m": 168, "day": 64}
    train_sampler = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe,
        as_of=cfg.train_end,           # cap inputs + labels at end of train
        window_bars=window_bars,
        min_end_ts=cfg.train_start,    # anchors ≥ train_start
        drop_invalid_threshold=cfg.drop_invalid_threshold,
        aux_root=cfg.aux_root,
        label_mode=cfg.label_mode,
        label_root_resid=cfg.label_root_resid,
        resid_horizons=cfg.resid_horizons,
    )
    val_sampler = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe,
        as_of=cfg.val_end,             # cap inputs + labels at end of val
        window_bars=window_bars,
        min_end_ts=cfg.val_start,      # anchors ≥ val_start
        drop_invalid_threshold=cfg.drop_invalid_threshold,
        aux_root=cfg.aux_root,
        label_mode=cfg.label_mode,
        label_root_resid=cfg.label_root_resid,
        resid_horizons=cfg.resid_horizons,
    )

    # Magnitude-balance calibration (PHASE3.md §6.2 + amendment 11).
    # Pull one cross-section per horizon DIRECTLY (don't go through the
    # weighted-random stream — at the natural 70:52:1:1 cadence we'd burn
    # ~300 draws before seeing a 1d/2d sample).
    if cfg.calibrate_loss_weights_at_init:
        sample: dict[str, CrossSection] = {}
        for h in HORIZONS:
            try:
                cs = next(iter(getattr(train_sampler, f"iter_{h}")()))
            except StopIteration:
                continue
            sample[h] = CrossSection(
                t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
                inputs={sc: t.to(device) for sc, t in cs.inputs.items()},
                labels=cs.labels.to(device),
                ranks=cs.ranks.to(device),
                valid_mask=cs.valid_mask.to(device),
                vol_bucket=cs.vol_bucket.to(device),
            )
        if len(sample) == len(HORIZONS):
            weights = module.calibrate_loss_weights(sample)
            log.info("calibrate_loss_weights: %s", weights)
        else:
            log.warning(
                "calibrate_loss_weights: only got %d / %d horizons; skipping",
                len(sample), len(HORIZONS),
            )

    # Optim.
    optimizer = _build_optimizer(module, cfg)
    warmup_steps = int(cfg.warmup_steps_pct * cfg.max_steps)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # ----- Performance optimizations -----
    # bf16 autocast: ~1.4-1.7× speedup on H200 (no GradScaler needed for bf16).
    # Disabled on CPU (no benefit there).
    use_autocast = cfg.use_bf16_autocast and device.type == "cuda"
    autocast_ctx = lambda: (  # noqa: E731
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast else nullcontext()
    )
    if use_autocast:
        log.info("autocast: bf16 enabled on CUDA forward")

    # Fix-2 (H200 diagnosis): `torch.compile(mode="reduce-overhead",
    # dynamic=False, fullgraph=False)` lets PT capture CUDA Graphs over
    # the static-shape forward, amortizing the per-step kernel-launch
    # overhead (which is RC-1: ~30-50 ms/step floor in eager mode).
    # Requires `pad_n_for_compile > 0` so cross-section shapes are static.
    #
    # IMPORTANT: only compile the heavy `peft_backbone`. Compiling the
    # tokenizer too caused CUDAGraphs "output tensor overwritten" errors
    # because the tokenizer's output is held by autograd between steps.
    # The tokenizer is light enough that eager-mode is fine.
    if (
        cfg.use_torch_compile
        and cfg.pad_n_for_compile > 0
        and device.type == "cuda"
    ):
        try:
            module.peft_backbone = torch.compile(  # type: ignore[assignment]
                module.peft_backbone,
                mode="reduce-overhead",
                dynamic=False,
                fullgraph=False,
            )
            log.info(
                "torch.compile (reduce-overhead, dynamic=False): "
                "applied to peft_backbone (padding N to %d for static shapes)",
                cfg.pad_n_for_compile,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("torch.compile failed (%s); falling back to eager", exc)
    elif cfg.use_torch_compile and cfg.pad_n_for_compile == 0:
        log.warning(
            "torch.compile requested but pad_n_for_compile=0; skipping. "
            "Set pad_n_for_compile = len(universe) to enable.",
        )

    # Train loop with async data prefetching. Three modes:
    #   - MP DataLoader workers (default; best on H200): one DataLoader per
    #     horizon, workers are separate PROCESSES so no GIL contention.
    #   - Single-process threads (legacy): one thread per horizon. Bounded
    #     by GIL during per-symbol Python loops in `_build_cross_section`.
    #   - Synchronous (debug): no prefetch.
    stream: _HorizonStream | _AsyncHorizonStream | _MPHorizonStream
    if cfg.num_data_workers > 0:
        stream = _MPHorizonStream(
            cfg=cfg,
            sampler_as_of=cfg.train_end,
            sampler_min_end_ts=cfg.train_start,
            cadence_weights=cfg.cadence_weights,
            seed=cfg.seed, device=device,
            num_workers=cfg.num_data_workers,
            prefetch_factor=cfg.data_prefetch_factor,
        )
        log.info(
            "MP data workers: %d total across %d horizons, prefetch_factor=%d",
            cfg.num_data_workers, len(cfg.cadence_weights),
            cfg.data_prefetch_factor,
        )
    elif cfg.use_async_data_prefetch:
        stream = _AsyncHorizonStream(
            train_sampler, cfg.cadence_weights,
            seed=cfg.seed, device=device,
            queue_size=cfg.async_prefetch_queue_size,
        )
        log.info(
            "async data prefetch: %d threads, queue_size=%d",
            len(cfg.cadence_weights), cfg.async_prefetch_queue_size,
        )
    else:
        stream = _HorizonStream(
            train_sampler, cfg.cadence_weights, seed=cfg.seed,
        )

    module.train()
    last_loss_by_horizon: dict[str, float] = {h: float("inf") for h in HORIZONS}

    try:
        accum = max(1, int(cfg.grad_accum_steps))
        if accum > 1:
            log.info("gradient accumulation: %d cross-sections per opt step", accum)

        train_done = False
        for step in range(cfg.max_steps):
            if train_done:
                break

            # LR schedule (advances per OPT step, not per CS draw).
            for g, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                g["lr"] = _cosine_with_warmup_lr(
                    step, max_steps=cfg.max_steps, warmup_steps=warmup_steps,
                    base_lr=base_lr, lr_min_ratio=cfg.lr_min_ratio,
                )

            # When peft_backbone is compiled with `mode="reduce-overhead"`,
            # CUDA Graphs reuses output buffers across invocations. Signal
            # "new training step" so the reused-buffer detector knows the
            # previous step's tensors are now safe to overwrite. No-op if
            # the model isn't compiled with reduce-overhead.
            if cfg.use_torch_compile and cfg.pad_n_for_compile > 0 and device.type == "cuda":
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:  # noqa: BLE001
                    pass

            optimizer.zero_grad(set_to_none=True)

            # Accumulate gradients across `accum` cross-sections.
            last_h = None
            last_loss = float("inf")
            n_accumulated = 0
            for _accum_i in range(accum):
                res = stream.next()
                if res is None:
                    log.error(
                        "train stream exhausted at step %d (accum %d/%d); stopping early",
                        step, _accum_i, accum,
                    )
                    train_done = True
                    break
                h, cs = res

                if isinstance(stream, (_AsyncHorizonStream, _MPHorizonStream)):
                    cs_dev = cs   # already on device
                else:
                    cs_dev = _cs_to_device(cs, device, non_blocking=False)

                # Fix-2 (H200): pad cross-section to a fixed N so
                # torch.compile + CUDA Graphs can capture static-shape
                # forward. Slice outputs back to original N before loss
                # so we never train on padding rows.
                n_valid = cs_dev.labels.shape[0]
                pad_n = cfg.pad_n_for_compile if cfg.pad_n_for_compile > 0 else 0
                if pad_n > 0 and n_valid < pad_n:
                    cs_padded = _pad_cross_section(cs_dev, pad_n)
                elif pad_n > 0 and n_valid > pad_n:
                    log.warning(
                        "cross-section N=%d exceeds pad_n_for_compile=%d; "
                        "truncating (consider raising pad_n_for_compile)",
                        n_valid, pad_n,
                    )
                    cs_padded = _truncate_cross_section(cs_dev, pad_n)
                    n_valid = pad_n
                else:
                    cs_padded = cs_dev

                # Forward + loss + backward. Scale loss by 1/accum so the
                # gradient magnitude matches a single-CS step (standard
                # gradient-accumulation pattern).
                with autocast_ctx():
                    out = module(cs_padded, horizon=h)
                    if pad_n > 0 and n_valid < pad_n:
                        # Slice outputs to valid rows before loss (so the
                        # padding zeros don't contribute to gradients).
                        out = {
                            "quantile": out["quantile"][:n_valid],
                            "rank":     out["rank"][:n_valid],
                            "vol":      out["vol"][:n_valid],
                            "forecast_hidden": out["forecast_hidden"][:n_valid],
                        }
                    loss = module.compute_loss(out, cs_dev, horizon=h)
                (loss / accum).backward()
                last_h = h
                last_loss = float(loss.detach())
                last_loss_by_horizon[h] = last_loss
                n_accumulated += 1

            if n_accumulated == 0:
                # All `accum` draws returned None — nothing to step on.
                break

            if cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in module.trainable_parameters()],
                    cfg.gradient_clip_val,
                )
            optimizer.step()

            if (step + 1) % cfg.log_every_n_steps == 0:
                log.info(
                    "step=%d h=%s loss=%.4f lr_lora=%.2e lr_head=%.2e",
                    step + 1, last_h, last_loss,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                )
    finally:
        # Stop producer threads / DataLoader workers cleanly even on
        # training crash / early break.
        if isinstance(stream, (_AsyncHorizonStream, _MPHorizonStream)):
            stream.close()

    # Final eval.
    eval_results = evaluate(
        module, val_sampler, limit_cross_sections=cfg.limit_val_cross_sections,
    )
    per_h_ic = {h: ev.rank_ic.mean_ic for h, ev in eval_results.items()}
    per_h_n = {h: ev.n_cross_sections_used for h, ev in eval_results.items()}

    result = RunResult(
        mode=cfg.mode, seed=cfg.seed, final_step=cfg.max_steps,
        per_horizon_rank_ic=per_h_ic,
        per_horizon_n_cross_sections=per_h_n,
        final_loss_by_horizon=last_loss_by_horizon,
    )
    log.info("run_single_fold complete: %s", result)

    # Save trained state so future eval-only runs don't need to retrain.
    # Use the underlying nn.Module's state_dict (torch.compile-wrapped
    # modules accept the original module's state_dict cleanly on reload).
    if cfg.save_model_path is not None:
        save_path = Path(cfg.save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": module.state_dict(),
            "module_cfg": cfg.module_cfg.__dict__,
            "run_config": {
                "mode": cfg.mode,
                "seed": cfg.seed,
                "max_steps": cfg.max_steps,
                "train_start": cfg.train_start.isoformat(),
                "train_end": cfg.train_end.isoformat(),
                "val_start": cfg.val_start.isoformat(),
                "val_end": cfg.val_end.isoformat(),
                "as_of": cfg.as_of.isoformat(),
                "n_symbols": len(cfg.universe),
                "window_bars": cfg.window_bars,
                "cadence_weights": cfg.cadence_weights,
            },
            "loss_weights": (
                module.loss_weights if hasattr(module, "loss_weights") else None
            ),
            "per_horizon_rank_ic": per_h_ic,
            "per_horizon_n_cross_sections": per_h_n,
        }
        torch.save(payload, save_path)
        log.info("saved trained state to %s", save_path)

    return module, result
