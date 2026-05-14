"""Phase 4 step 5 — FineTuneModule (OB fine-tune of Phase 3 LoRA model).

Loads a Phase 3 `*.state.pt` (post-iter-3 cadence + per-horizon LoRA
banks + heads) and wraps it with an OB encoder + gated cross-attention
fusion. Per DESIGN §3.4:

  - Only intraday heads (30m, 120m) trainable; swing heads (1d, 2d)
    hard-frozen.
  - OB encoder + fusion blocks: fully trainable.
  - LoRA banks on backbone: loaded from Phase 3, optionally trainable
    for the 30m/120m banks (gated by `cfg.train_lora_intraday`).
  - `_use_ob` buffer flips the OB path off at inference for the
    with-/without-OB ablation gate (DESIGN §3.4 + acceptance §6.1.4).

When `_use_ob=False`, forward is bit-identical to the Phase 3 baseline
(the OB encoder + fusion blocks contribute no signal — `ob_kv` not
passed through fusion). This is exactly what the §3.4 acceptance gate
compares against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from daity.data.cross_section import CrossSection
from daity.models.fusion import OBFusedBackbone
from daity.models.orderbook import OBEncoder
from daity.training.posttrain import (
    HORIZONS,
    PostTrainConfig,
    PostTrainModule,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Phase 4 trains only intraday horizons (30m, 120m) per DESIGN §3.4.
PHASE4_TRAINABLE_HORIZONS: tuple[str, ...] = ("30m", "120m")


@dataclass
class FineTuneConfig:
    """Knobs for the Phase 4 OB fine-tune.

    `parent_checkpoint` is a Phase 3 `*.state.pt` produced by
    `daity-posttrain` (NOT the SSL checkpoint — Phase 4 starts from
    the post-trained model). We load tokenizer + backbone + LoRA banks
    + heads from there; OB encoder + fusion + gate are fresh.
    """

    parent_checkpoint: Path

    # OB encoder.
    ob_n_features: int = 8
    ob_k_snapshots: int = 64
    ob_d_model: int = 192
    ob_n_layers: int = 4
    ob_n_heads: int = 4
    ob_ffn_ratio: int = 4
    ob_dropout: float = 0.1

    # Fusion.
    fusion_n_heads: int = 8
    fusion_dropout: float = 0.3
    inject_at_layers: tuple[int, ...] = (10, 11)

    # Trainable surfaces.
    train_lora_intraday: bool = True   # 30m + 120m LoRA banks
    train_heads_intraday: bool = True  # 30m + 120m heads
    # All other things (1d/2d LoRA banks, 1d/2d heads, tokenizer, backbone)
    # are frozen.


class FineTuneModule(nn.Module):
    """Phase 3 LoRA model + OB encoder + gated cross-attention fusion.

    Forward path:
      1. Tokenizer (frozen) → backbone-token stream.
      2. OBFusedBackbone routes through Phase-3 LoRA banks AND injects
         gated cross-attention from OBEncoder output (when `_use_ob=True`
         AND `cs.ob_mask` has any True entries).
      3. Heads (30m, 120m intraday-trainable; 1d, 2d frozen) on the
         FORECAST hidden.

    `_use_ob` buffer: 1 = include OB path; 0 = skip OB path (output
    bit-equals Phase-3 baseline). Used for the §3.4 acceptance ablation.
    """

    def __init__(self, cfg: FineTuneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Load Phase-3 state.pt (which includes module_cfg + state_dict
        # of the entire PostTrainModule — tokenizer + backbone + LoRA + heads).
        if not cfg.parent_checkpoint.exists():
            msg = f"FineTuneModule: parent_checkpoint not found at {cfg.parent_checkpoint}"
            raise FileNotFoundError(msg)
        payload = torch.load(cfg.parent_checkpoint, map_location="cpu",
                             weights_only=False)
        module_cfg_dict: dict[str, Any] = dict(payload["module_cfg"])
        if isinstance(module_cfg_dict.get("parent_checkpoint"), str):
            module_cfg_dict["parent_checkpoint"] = Path(
                module_cfg_dict["parent_checkpoint"]
            )
        ssl_ckpt = module_cfg_dict["parent_checkpoint"]
        if not ssl_ckpt.exists():
            log.warning(
                "FineTuneModule: SSL checkpoint at %s missing — "
                "PostTrainModule init will skip SSL load (relies on state_dict).",
                ssl_ckpt,
            )

        # Bypass the SSL load: PostTrainModule normally re-reads the SSL
        # checkpoint at init. For Phase 4 we'll overwrite all those weights
        # with the Phase-3 state_dict immediately after — so just point
        # `parent_checkpoint` at the SSL file (same as Phase 3 did), then
        # load our state_dict over the top.
        post_train_cfg = PostTrainConfig(**module_cfg_dict)
        self.posttrain = PostTrainModule(post_train_cfg)
        # Replace SSL-init weights with the Phase-3-trained ones.
        missing, unexpected = self.posttrain.load_state_dict(
            payload["state_dict"], strict=False,
        )
        # `unexpected` may contain new buffers we don't care about; assert
        # missing is empty for the trained surfaces.
        if missing:
            log.warning(
                "FineTuneModule: %d missing keys when loading Phase-3 state: %s",
                len(missing), missing[:5],
            )

        # 2. Wrap backbone with OB fusion.
        # PostTrainModule.peft_backbone is the PEFT-wrapped backbone with
        # 4 LoRA banks. OBFusedBackbone expects a vanilla `Backbone` —
        # we wrap the PEFT one, since `peft_backbone(x, causal=...)` is
        # the same callable interface as `Backbone(x, causal=...)` plus
        # adapter routing.
        # NB: OBFusedBackbone iterates `self.backbone.layers`. The PEFT
        # wrapper exposes the same `layers` attribute (pass-through to
        # the wrapped Backbone), so iteration sees LoRA-adapted layers.
        self.ob_encoder = OBEncoder(
            n_features=cfg.ob_n_features, k_snapshots=cfg.ob_k_snapshots,
            d_model=cfg.ob_d_model, n_layers=cfg.ob_n_layers,
            n_heads=cfg.ob_n_heads, ffn_ratio=cfg.ob_ffn_ratio,
            dropout=cfg.ob_dropout,
        )
        # The actual transformer body — peft_backbone's `.base_model.model`
        # is the underlying Backbone. We pass that for OBFusedBackbone to
        # iterate layers (LoRA-modified attentions still work since the
        # peft routing reads `layer._active_adapter`).
        underlying_backbone = self.posttrain.peft_backbone.base_model.model
        self.fused_backbone = OBFusedBackbone(
            backbone=underlying_backbone,
            d_model_q=post_train_cfg.d_model,
            d_model_kv=cfg.ob_d_model,
            n_heads=cfg.fusion_n_heads,
            inject_at_layers=cfg.inject_at_layers,
            dropout=cfg.fusion_dropout,
        )

        # 3. Freeze the right things.
        # Tokenizer: already frozen by PostTrainModule. Stays frozen.
        # Backbone (vanilla layers): already frozen by PostTrainModule. Stays.
        # LoRA banks: PostTrainModule made all 4 banks trainable; we restrict
        # to 30m/120m (the intraday ones) per DESIGN §3.4.
        if cfg.train_lora_intraday:
            for name, param in self.posttrain.peft_backbone.named_parameters():
                if "lora_" in name:
                    # peft names banks via `.{name}.` segments.
                    is_intraday = any(
                        f".{h}." in name for h in PHASE4_TRAINABLE_HORIZONS
                    )
                    param.requires_grad_(is_intraday)
        else:
            for param in self.posttrain.peft_backbone.parameters():
                # Already covers LoRA params via PEFT wrapping.
                if param.requires_grad:
                    param.requires_grad_(False)

        # Heads: freeze 1d/2d, leave 30m/120m trainable (per DESIGN §3.4).
        for h in HORIZONS:
            trainable = h in PHASE4_TRAINABLE_HORIZONS and cfg.train_heads_intraday
            for module in (
                self.posttrain.quantile_heads[h],
                self.posttrain.rank_heads[h],
                self.posttrain.vol_heads[h],
            ):
                for p in module.parameters():
                    p.requires_grad_(trainable)

        # `_use_ob` switch — survives state_dict round-trip.
        self.register_buffer(
            "_use_ob_buf", torch.tensor(1, dtype=torch.int8),
            persistent=True,
        )

    # ----- Ablation toggle -----

    @property
    def use_ob(self) -> bool:
        return bool(self._use_ob_buf.item())

    @use_ob.setter
    def use_ob(self, value: bool) -> None:
        self._use_ob_buf.fill_(int(bool(value)))

    def set_gate_force_zero(self, force: bool) -> None:
        """Curriculum: epoch-1 forces OB fusion gate to zero so the
        intraday heads re-stabilize on the new training distribution
        before the OB tower comes online."""
        self.fused_backbone.set_gate_force_zero(force)

    # ----- Forward -----

    def forward(
        self,
        cs: CrossSection,
        horizon: str,
    ) -> dict[str, torch.Tensor]:
        """Forward through tokenizer → fused backbone (with OB cross-attn)
        → heads for `horizon`. Returns same dict as PostTrainModule:
        `{quantile, rank, vol, forecast_hidden}`.

        OB path: only active when `self.use_ob` is True AND
        `cs.ob_snapshot_window` is not None AND at least one row in
        `cs.ob_mask` is True. Otherwise the fusion residual is gate-zeroed
        (or `ob_kv=None` is passed through, also a no-op).
        """
        # Route through this horizon's LoRA bank (same Fix-1 pattern as
        # PostTrainModule.forward — direct `_active_adapter` assignment).
        if self.posttrain._use_lora and self.posttrain._lora_layers:
            for layer in self.posttrain._lora_layers:
                layer._active_adapter = [horizon]

        tokens = self.posttrain.tokenizer(cs.inputs)  # (B, 1+P, d_model)

        # Build OB key/value bank if applicable.
        ob_kv: torch.Tensor | None = None
        if (
            self.use_ob
            and cs.ob_snapshot_window is not None
            and cs.ob_mask is not None
            and cs.ob_mask.any()
        ):
            ob_in = cs.ob_snapshot_window.to(tokens.device, non_blocking=True)
            ob_kv = self.ob_encoder(ob_in)            # (B, K, d_ob)
            # Zero out kv for symbols whose ob_mask is False — they'll see
            # gate * cross_attn(q, [zeros]) ≈ ~0 (and the gate's per-token
            # output already small at init). This is the cleanest way to
            # mix OB-covered and OB-uncovered symbols in one cross-section.
            mask = cs.ob_mask.to(tokens.device, non_blocking=True)
            ob_kv = ob_kv * mask.view(-1, 1, 1).float()

        hidden = self.fused_backbone(tokens, ob_kv=ob_kv, causal=False)
        forecast_hidden = hidden[:, 0]                # (B, d_model)

        # Aux-feature late-fusion (same as PostTrainModule).
        if (
            self.posttrain.aux_proj is not None
            and cs.aux_features is not None
        ):
            aux = cs.aux_features.to(forecast_hidden.device, non_blocking=True)
            forecast_hidden = forecast_hidden + self.posttrain.aux_proj(
                self.posttrain.aux_norm(aux)
            )

        return {
            "quantile": self.posttrain.quantile_heads[horizon](forecast_hidden),
            "rank":     self.posttrain.rank_heads[horizon](forecast_hidden).squeeze(-1),
            "vol":      self.posttrain.vol_heads[horizon](forecast_hidden).squeeze(-1),
            "forecast_hidden": forecast_hidden,
        }

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        cs: CrossSection,
        horizon: str,
    ) -> torch.Tensor:
        """Reuse PostTrainModule's loss function."""
        return self.posttrain.compute_loss(out, cs, horizon)

    def trainable_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        """List `(name, param)` for parameters with `requires_grad=True`."""
        return [
            (n, p) for n, p in self.named_parameters() if p.requires_grad
        ]
