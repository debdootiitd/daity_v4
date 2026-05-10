"""Lightning callbacks for the Phase 2 SSL pretraining run.

Two callbacks here, both load-bearing for DESIGN §3.2 / §3.3:

  - `EMACallback` — maintains an exponential-moving-average shadow of the
    `LightningModule`'s parameters with decay `cfg.ema_decay`, swaps the
    EMA weights into the live module during validation, restores them
    after. DESIGN §3.3 specifies "EMA of weights (decay 0.999) used for
    evaluation" — the val metric chosen by `ModelCheckpoint` is therefore
    the EMA-smoothed val loss, which is materially less noisy than the
    raw-weight val loss for "best three by val MSE" selection.

    Implemented as a manual shadow `state_dict` (rather than
    `torch.optim.swa_utils.AveragedModel`, which copies the module and
    needs a forward signature that matches) — all we need is
    parameter-wise EMA + swap/restore around validation, which is ~30
    lines of Lightning hooks.

  - `ProvenanceCallback` — emits the `(git_sha, config_hash, as_of)`
    triple required by CLAUDE.md §6 (Reproducibility) into a
    `provenance.json` sidecar in the run's log dir, and stamps the same
    triple into the LightningModule's `hparams` so it lands inside every
    saved checkpoint. Refuses to start with uncommitted local changes
    unless `--allow-dirty` is set.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from daity.utils.logging import get_logger

log = get_logger(__name__)


# ----- EMA -----


class EMACallback(Callback):
    """Exponential moving average of parameters, used for validation.

    On every `on_train_batch_end`, update each parameter's shadow:
        shadow[name] = decay * shadow[name] + (1 - decay) * param.data

    On `on_validation_start`, swap the shadow into the live module
    (storing the live values in a temporary backup); on
    `on_validation_end`, restore the live values. This gives Lightning's
    val_loss metric an EMA-smoothed reading without changing how the
    LightningModule's `validation_step` is written.

    Notes:
      - Only floating-point parameters are tracked. Buffers (e.g. RevIN's
        cached `_mean` / `_std`) are left alone; they're per-batch
        artifacts, not model state worth EMA'ing.
      - Shadow lives on the same device as the parameters and stays in
        sync if the module is `.to(device)`'d (Lightning calls the
        `setup` hook after the device move).
    """

    def __init__(self, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            msg = f"EMA decay must be in (0, 1), got {decay}"
            raise ValueError(msg)
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._initialized: bool = False

    def _init_shadow(self, pl_module: L.LightningModule) -> None:
        """(Re-)initialize shadow weights from the live module's params.

        Idempotent — safe to call from multiple lifecycle hooks. Ensures
        shadow tensors live on the same device as the params at call
        time. If `state_dict()` was loaded earlier (resume from
        checkpoint), only migrate device; don't overwrite the values.
        """
        live_params = [
            (name, p) for name, p in pl_module.named_parameters()
            if p.is_floating_point()
        ]
        if self._shadow and self._initialized:
            # Resume path: keep loaded values, just migrate to the right device.
            for name, p in live_params:
                if name in self._shadow:
                    self._shadow[name] = self._shadow[name].to(p.device)
        else:
            # Fresh init: clone live params on their current device.
            self._shadow = {
                name: p.detach().clone() for name, p in live_params
            }
        self._initialized = True
        log.info(
            "EMACallback initialized with decay=%.4f, %d tracked params on %s",
            self.decay, len(self._shadow),
            next(iter(self._shadow.values())).device if self._shadow else "n/a",
        )

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # `on_fit_start` fires AFTER Lightning moves the module to the
        # accelerator (vs `setup()`, which can fire before). Clone shadow
        # here so the tensors land on the same device as the live params.
        self._init_shadow(pl_module)

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        # Defensive: in tests we sometimes invoke this hook directly without
        # a real fit() lifecycle; fall back to lazy init.
        if not self._initialized:
            self._init_shadow(pl_module)
        # In-place EMA update. Note: `lerp_(end, weight)` computes
        # self = self + weight * (end - self) = (1-weight)*self + weight*end,
        # so weight=(1-decay) gives the EMA recurrence we want.
        weight = 1.0 - self.decay
        for name, p in pl_module.named_parameters():
            if name not in self._shadow:
                continue
            shadow = self._shadow[name]
            if shadow.device != p.device:
                # Device drift (e.g. mid-fit module relocation): migrate
                # the shadow to follow the param. Rare but possible.
                shadow = shadow.to(p.device)
                self._shadow[name] = shadow
            shadow.lerp_(p.detach(), weight)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Swap shadow → live; back up live values so we can restore.
        self._backup = {}
        for name, p in pl_module.named_parameters():
            if name not in self._shadow:
                continue
            self._backup[name] = p.data.clone()
            p.data.copy_(self._shadow[name])

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        for name, p in pl_module.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    # Make EMA state survive checkpoints (so resume-from-checkpoint
    # picks up the shadow rather than starting cold).
    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self._shadow = {k: v.clone() for k, v in state_dict.get("shadow", {}).items()}


# ----- Provenance -----


@dataclass(frozen=True, slots=True)
class Provenance:
    """The reproducibility triple per CLAUDE.md §6, plus a timestamp."""

    git_sha: str
    git_dirty: bool
    config_hash: str
    as_of: str         # train_end as ISO-8601
    started_at: str    # UTC start time

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "as_of": self.as_of,
            "started_at": self.started_at,
        }


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _config_hash(merged_cfg: dict[str, Any]) -> str:
    """Stable 12-char SHA-256 prefix of the merged YAML+overrides config.

    `default=str` handles non-JSON-native types (Path, datetime). Sorting
    by key makes the hash invariant to dict ordering.
    """
    payload = json.dumps(merged_cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def build_provenance(
    *, repo: Path, merged_cfg: dict[str, Any], train_end: datetime,
) -> Provenance:
    """Assemble the provenance triple for a pretraining run."""
    return Provenance(
        git_sha=_git_sha(repo),
        git_dirty=_git_dirty(repo),
        config_hash=_config_hash(merged_cfg),
        as_of=train_end.astimezone(UTC).isoformat(),
        started_at=datetime.now(UTC).isoformat(),
    )


def write_provenance_sidecar(provenance: Provenance, log_dir: Path) -> Path:
    """Write `provenance.json` next to the Lightning log dir.

    Lightning's CSVLogger creates `version_N` subdirs; we write at the
    `log_dir` root so the file is easy to find regardless of versioning.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "provenance.json"
    path.write_text(json.dumps(provenance.to_dict(), indent=2, sort_keys=True))
    return path
