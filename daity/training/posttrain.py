"""Phase 3.4 — supervised post-training module.

`PostTrainModule` wraps:

- The Phase-2 SSL backbone (`tokenizer` + `Backbone`), loaded from the
  v3_xattn checkpoint with **hparam compatibility check** (`PHASE3.md`
  §6.10 + amendment 7).
- **Four LoRA banks** (`lora_30m`, `lora_120m`, `lora_1d`, `lora_2d`)
  applied to every layer's fused `qkv` projection. Backbone frozen;
  LoRA + heads trainable. (`PHASE3.md` §6.3 + amendment 5.)
- **Three heads × four horizons = 12 head instances**: quantile (5-quantile
  pinball), rank (ListMLE), realized-vol (MSE). Direction head dropped
  per amendment 4.

Training step (`train_step`):

1. Take a `CrossSection` (one horizon's worth of cross-symbol inputs).
2. Activate the horizon-specific LoRA bank via `peft.set_adapter()`.
3. Forward the full tokenizer + backbone on `(N, L_scale, 18)` per scale.
4. Extract `(N, d_model)` = the FORECAST-token's hidden state.
5. Run the three horizon-specific heads.
6. Compute losses: pinball + listmle + vol_mse.
7. Apply magnitude-balanced weights (set once at step 0 — `PHASE3.md` §6.2
   + amendment 11).
8. Backprop; one optimizer step.

Weighted-random alternation across horizons happens in the training-loop
driver (CLI), not inside the module — the module is horizon-aware but
horizon-agnostic per step.

Phase 3.4 acceptance criteria (`PHASE3.md` §11):
- Smoke run on CPU completes.
- Train_loss drops monotonically across 100 steps for all 4 horizons.
- Hparam-compat refuses on a synthetic mismatched checkpoint.

Phase 3.5 (separate sub-phase) covers the full single-fold + random-init
ablation run on H200.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from daity.data.cross_section import CrossSection
from daity.models.backbone import Backbone
from daity.models.tokenizer import MultiResTokenizer
from daity.training.losses import PHASE3_QUANTILES, listmle_loss, pinball_loss
from daity.utils.logging import get_logger

log = get_logger(__name__)


# The four supervised horizons (post-amendment 16).
HORIZONS: tuple[str, ...] = ("30m", "120m", "1d", "2d")

# Canonical input scales. Must match the SSL backbone's tokenizer scales.
SCALES: tuple[str, ...] = ("5m", "15m", "60m", "day")


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a posttrain run's hparams don't match the SSL checkpoint."""


@dataclass
class PostTrainConfig:
    """Knobs for the Phase 3 posttrain module (`PHASE3.md` §6).

    Hparams that come from the SSL checkpoint (d_model, n_layers, etc.) are
    NOT duplicated here — they're read off the checkpoint and asserted
    against the live tokenizer/backbone via `_check_hparam_compat`.
    """

    # Path to the v3_xattn (or compatible) SSL checkpoint.
    parent_checkpoint: Path

    # Backbone hparams — must MATCH the parent checkpoint exactly.
    # Defaults match `configs/train/pretrain_v3_xattn.yaml`.
    d_model: int = 480
    n_layers: int = 12
    n_heads: int = 8
    ffn_ratio: int = 4
    patch_len: int = 16
    patch_stride: int = 8
    num_channels: int = 18
    revin_affine: bool = False     # DESIGN amendment 12 — must be False for SSL
    scales: tuple[str, ...] = SCALES

    # LoRA (PHASE3.md §6.3 + amendment 5).
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Target the backbone's fused `qkv` linear. On d_model=480 with r=8 this
    # adds 4·r·d = 15,360 params per layer per bank → 184K / bank × 4 banks ≈
    # 737K LoRA total. Note this LoRA-on-`qkv` adapts K too (the K-projection
    # share of the fused linear), whereas §6.3 spec'd LoRA on `(q_proj, v_proj)`
    # only. The fused-architecture trade is: same param count, slightly broader
    # adapt surface (K included). Document this as a deliberate divergence
    # from §6.3 — review-finding Minor-3.
    lora_target_modules: tuple[str, ...] = ("qkv",)

    # Head dimensions (PHASE3.md §6.1).
    head_hidden: int | None = None      # defaults to d_model
    head_dropout: float = 0.1

    # Vol-head target mode (review-finding Major-2). The "real" target is
    # forward-realized vol over the horizon's window, which requires the
    # CrossSection to carry a `vol_target` tensor. The §3.4 smoke uses the
    # `abs_y_log` proxy (biased — collapses zero-return bars to log(eps)).
    # §3.5 must plumb `forward_realized` through `cross_section.py`.
    vol_target_mode: str = "abs_y_log"        # "abs_y_log" | "forward_realized"

    # Per-horizon × per-head loss weights. Filled at step 0 via
    # magnitude-balance heuristic (amendment 11); the values here are
    # initial 1.0s.
    initial_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            h: {"quantile": 1.0, "rank": 1.0, "vol": 1.0} for h in HORIZONS
        },
    )

    # Phase 3.4b (iter-4) — late-fusion of top-50 XGB-importance features
    # as a static-covariate residual added to the FORECAST hidden before
    # the three heads. Compensates for the backbone's tokenizer not
    # capturing the explicit aggregations that XGB uses (which beat LoRA
    # on 120m by Δ −0.0129 mean). When False (default), the aux path is
    # inactive and the model is bit-identical to the no-aux baseline.
    use_aux_features: bool = False
    aux_features_dim: int = 50


# ----- Heads (PHASE3.md §6.1) -----


class _MLPHead(nn.Module):
    """Shared 2-layer GELU-MLP pattern for all three Phase-3 heads."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        h = d_hidden if d_hidden is not None else d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----- The module -----


@dataclass
class _HparamDelta:
    """Single hparam mismatch between checkpoint and posttrain config."""
    key: str
    checkpoint_value: Any
    posttrain_value: Any


def _check_hparam_compat(
    checkpoint_hparams: dict[str, Any],
    cfg: PostTrainConfig,
) -> list[_HparamDelta]:
    """Return any backbone-hparam deltas. Empty list = compatible."""
    expected = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "ffn_ratio": cfg.ffn_ratio,
        "patch_len": cfg.patch_len,
        "patch_stride": cfg.patch_stride,
        "num_channels": cfg.num_channels,
        "revin_affine": cfg.revin_affine,
        # Scales must match exactly (order matters for tokenizer's
        # resolution embedding).
        "scales": list(cfg.scales),
    }
    deltas: list[_HparamDelta] = []
    for key, want in expected.items():
        got = checkpoint_hparams.get(key)
        # Lightning checkpoints may store scales as list — normalize.
        got_norm = list(got) if isinstance(got, tuple) else got
        want_norm = list(want) if isinstance(want, tuple) else want
        if got_norm != want_norm:
            deltas.append(
                _HparamDelta(key=key, checkpoint_value=got_norm, posttrain_value=want_norm),
            )
    return deltas


def _load_ssl_checkpoint(
    cfg: PostTrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read the SSL checkpoint's `state_dict` + `hyper_parameters`.

    Lightning saves `(state_dict, hyper_parameters, ...)` in a single
    `.ckpt` file under `pl_load`. We accept that format AND a plain
    `state_dict` format (no Lightning hparams). For the plain format,
    hparam compat is skipped — the caller is on their own.

    Returns `(state_dict_for_backbone_and_tokenizer, hparams_or_empty)`.
    """
    ckpt = torch.load(cfg.parent_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        hp = ckpt.get("hyper_parameters", {})
        # Lightning may wrap hparams as a Namespace / dict; normalize.
        if hasattr(hp, "__dict__"):
            hp = dict(vars(hp))
        # Inner key for the PretrainConfig.
        if isinstance(hp, dict) and "cfg" in hp:
            inner = hp["cfg"]
            if hasattr(inner, "__dict__"):
                hp = dict(vars(inner))
            elif isinstance(inner, dict):
                hp = inner
    else:
        sd = ckpt
        hp = {}
    return sd, hp


def _strip_prefix(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Extract `{key_without_prefix: tensor}` for keys starting with `prefix.`.

    Tries the bare prefix first; if no keys match, also tries common
    wrappers (`_orig_mod.{prefix}` from `torch.compile`, `module.{prefix}`
    from DDP). Returns an empty dict if no candidate matches — the caller
    MUST check for empty and fail loud (review-finding Major-1).
    """
    candidates = (prefix, f"_orig_mod.{prefix}", f"module.{prefix}")
    for pref in candidates:
        full = pref + "."
        n = len(full)
        stripped = {k[n:]: v for k, v in sd.items() if k.startswith(full)}
        if stripped:
            return stripped
    return {}


class PostTrainModule(nn.Module):
    """Posttrain wrapper: tokenizer + (frozen backbone + 4 LoRA banks) + 12 heads.

    Usage:
        cfg = PostTrainConfig(parent_checkpoint=..., ...)
        module = PostTrainModule(cfg)
        # Per training step:
        out = module(cross_section, horizon="30m")
        loss = module.compute_loss(out, cross_section, horizon="30m")
        loss.backward()
    """

    def __init__(self, cfg: PostTrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Build tokenizer + backbone in the SSL-compatible shape. ----
        self.tokenizer = MultiResTokenizer(
            scales=tuple(cfg.scales),
            num_channels=cfg.num_channels,
            d_model=cfg.d_model,
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            revin_affine=cfg.revin_affine,
        )
        self.backbone = Backbone(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_ratio=cfg.ffn_ratio,
        )

        # ---- Load SSL weights + hparam-compat check. ----
        state, ckpt_hp = _load_ssl_checkpoint(cfg)
        if ckpt_hp:
            deltas = _check_hparam_compat(ckpt_hp, cfg)
            if deltas:
                msg_lines = [
                    "PostTrainModule: hparam mismatch with SSL checkpoint "
                    f"{cfg.parent_checkpoint}:",
                ]
                for d in deltas:
                    msg_lines.append(
                        f"  - {d.key}: checkpoint={d.checkpoint_value!r}, "
                        f"posttrain={d.posttrain_value!r}",
                    )
                raise IncompatibleCheckpointError("\n".join(msg_lines))

        # Lightning save format: keys like `tokenizer.X`, `backbone.X`, plus
        # head keys we don't need. Load tokenizer + backbone with `strict=False`
        # (because the SSL checkpoint also carries head-state-dicts we don't
        # need); BUT we still require zero `missing` keys for our two
        # modules — silent random-init is the §3.4 review's Major-1 finding.
        # `unexpected` is allowed and ignored (the SSL forecast head etc).
        tok_sd = _strip_prefix(state, "tokenizer")
        bb_sd = _strip_prefix(state, "backbone")
        if not tok_sd or not bb_sd:
            msg = (
                f"PostTrainModule: SSL checkpoint {cfg.parent_checkpoint} "
                f"has no recognizable tokenizer/backbone state. "
                f"Tried prefixes: tokenizer., _orig_mod.tokenizer., module.tokenizer. "
                f"(and same for backbone). "
                f"Available top-level keys (sample): {sorted(state.keys())[:8]!r}"
            )
            raise IncompatibleCheckpointError(msg)
        tok_missing, _ = self.tokenizer.load_state_dict(tok_sd, strict=False)
        if tok_missing:
            msg = (
                f"PostTrainModule: tokenizer load_state_dict missing keys "
                f"{tok_missing!r}. Refusing to train with a partially-random "
                f"tokenizer (review-finding Major-1)."
            )
            raise IncompatibleCheckpointError(msg)
        bb_missing, _ = self.backbone.load_state_dict(bb_sd, strict=False)
        if bb_missing:
            msg = (
                f"PostTrainModule: backbone load_state_dict missing keys "
                f"{bb_missing!r}. Refusing to train with a partially-random "
                f"backbone (review-finding Major-1)."
            )
            raise IncompatibleCheckpointError(msg)

        # ---- Freeze backbone + tokenizer; LoRA + heads only trainable. ----
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # Tokenizer trainable params: per-scale projections, revin gamma/beta
        # (revin_affine=False so no gamma/beta), resolution embed, forecast
        # token. Freeze all of these — Phase 3 LoRA-only fine-tunes the
        # attention path only.
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        # ---- Wrap backbone with 4 named LoRA banks. ----
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules), bias="none",
        )
        # First adapter creates the PeftModel wrapper; subsequent ones are
        # added as named alternatives.
        self.peft_backbone: PeftModel = get_peft_model(
            self.backbone, lora_cfg, adapter_name=HORIZONS[0],
        )
        for h in HORIZONS[1:]:
            self.peft_backbone.add_adapter(h, lora_cfg)
        self.peft_backbone.set_adapter(HORIZONS[0])
        # PEFT freezes non-active adapters' `requires_grad`; for our 4-bank
        # training (one optimizer covering all banks) we re-enable them all.
        # Forward through `set_adapter(h)` still routes only through `h`, so
        # inactive banks contribute no gradient.
        self._enable_all_lora_grads()

        # Fix-1 (H200 perf): cache references to the 12 LoraLayer instances
        # so forward() can flip `_active_adapter` directly per step instead
        # of calling `peft_backbone.set_adapter(h)` which walks the full
        # module tree + mutates `requires_grad` on 384 tensors per step. The
        # direct assignment is mathematically identical (LoRA forward reads
        # `layer.active_adapters` which unwraps `[layer._active_adapter]`)
        # and ~5 µs vs ~565 µs per step.
        try:
            from peft.tuners.lora.layer import LoraLayer as _LoraLayer
            self._lora_layers: list[_LoraLayer] = [
                m for m in self.peft_backbone.modules()
                if isinstance(m, _LoraLayer)
            ]
        except ImportError:
            self._lora_layers = []

        # `_use_lora` gates whether forward() activates LoRA adapter
        # routing + re-enables grads. The random-init ablation
        # (`RunConfig.mode="random_init"`) flips this to False so the
        # backbone runs in plain (no-LoRA) mode. Registered as a buffer
        # (review Major-3) so it survives state_dict save/load — a
        # reloaded random-init checkpoint must NOT silently route through
        # LoRA at inference (Phase 5 / conformal).
        self.register_buffer(
            "_use_lora_buf", torch.tensor(1, dtype=torch.int8),
            persistent=True,
        )

        # ---- Heads: 3 types × 4 horizons = 12 instances. ----
        head_h = cfg.head_hidden if cfg.head_hidden is not None else cfg.d_model
        self.quantile_heads = nn.ModuleDict({
            h: _MLPHead(cfg.d_model, len(PHASE3_QUANTILES), head_h, cfg.head_dropout)
            for h in HORIZONS
        })
        self.rank_heads = nn.ModuleDict({
            h: _MLPHead(cfg.d_model, 1, head_h, cfg.head_dropout)
            for h in HORIZONS
        })
        self.vol_heads = nn.ModuleDict({
            h: _MLPHead(cfg.d_model, 1, head_h, cfg.head_dropout)
            for h in HORIZONS
        })

        # ---- Per-horizon × per-head loss weights, mutable for magnitude-balance. ----
        self.loss_weights: dict[str, dict[str, float]] = {
            h: dict(cfg.initial_weights[h]) for h in HORIZONS
        }

        # Quantile values registered as a buffer for device tracking.
        self.register_buffer(
            "_phase3_quantiles",
            torch.tensor(PHASE3_QUANTILES, dtype=torch.float32),
        )

        # ---- Aux-feature projection (Phase 3.4b iter-4). ----
        # LayerNorm input first (50-dim features have wildly different
        # scales — volumes, returns, sins/coses), then Linear → d_model.
        # Initialized to zero-output (Linear bias 0, weight near-zero) so
        # the first training step is bit-identical to no-aux baseline and
        # the optimizer learns the residual contribution from there.
        if cfg.use_aux_features:
            self.aux_norm = nn.LayerNorm(cfg.aux_features_dim)
            self.aux_proj = nn.Linear(cfg.aux_features_dim, cfg.d_model)
            nn.init.zeros_(self.aux_proj.weight)
            nn.init.zeros_(self.aux_proj.bias)
        else:
            self.aux_norm = None
            self.aux_proj = None

    # ----- LoRA-vs-no-LoRA flag (Major-3 — survives state_dict round-trips) -----

    @property
    def _use_lora(self) -> bool:
        return bool(self._use_lora_buf.item())

    @_use_lora.setter
    def _use_lora(self, value: bool) -> None:
        self._use_lora_buf.fill_(int(bool(value)))

    # ----- Forward -----

    def _enable_all_lora_grads(self) -> None:
        """Re-enable `requires_grad=True` on every LoRA bank's params.

        Counteracts PEFT's default `set_adapter` behavior of freezing
        non-active banks. We want a single optimizer over all 4 banks;
        inactive banks contribute no gradient because peft only routes
        forward through the active bank.
        """
        for name, param in self.peft_backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

    def forward(
        self,
        cross_section: CrossSection,
        horizon: str,
    ) -> dict[str, torch.Tensor]:
        """Run tokenizer → LoRA-adapted backbone → 3 heads for the horizon.

        Returns dict with `quantile`, `rank`, `vol` tensors plus the
        FORECAST hidden for downstream diagnostics.
        """
        if horizon not in HORIZONS:
            msg = f"PostTrainModule.forward: unknown horizon {horizon!r}; expected one of {HORIZONS}"
            raise ValueError(msg)
        # In LoRA mode: route forward through the horizon's LoRA bank by
        # flipping `_active_adapter` directly on each cached LoRA layer.
        # This is ~80× faster than `peft_backbone.set_adapter(horizon)` +
        # `_enable_all_lora_grads()` per the H200 diagnosis (Fix-1).
        # requires_grad on all 4 banks was set ONCE at __init__ time.
        # In random-init mode (`_use_lora=False`), skip adapter routing.
        if self._use_lora and self._lora_layers:
            for layer in self._lora_layers:
                layer._active_adapter = [horizon]

        inputs = cross_section.inputs
        # MultiResTokenizer expects {scale: (B, L, C)}. The cross-section
        # input is exactly that, with B = N_symbols.
        tokens = self.tokenizer(inputs)                        # (B, 1+P, d_model)
        # `peft_backbone(...)` dispatches to the wrapped Backbone.forward.
        # Pass non-causal (Phase 3 reads FORECAST hidden from non-causal path).
        hidden = self.peft_backbone(tokens, causal=False)      # (B, 1+P, d_model)
        forecast_hidden = hidden[:, 0]                          # (B, d_model)

        # Aux-feature late-fusion residual (Phase 3.4b iter-4). Only adds
        # signal when both `cfg.use_aux_features` is True AND the cross-
        # section actually carries aux_features (sampler-side toggle).
        if (self.aux_proj is not None
                and cross_section.aux_features is not None):
            aux = cross_section.aux_features.to(forecast_hidden.device,
                                                non_blocking=True)
            forecast_hidden = forecast_hidden + self.aux_proj(self.aux_norm(aux))

        return {
            "quantile": self.quantile_heads[horizon](forecast_hidden),   # (B, K)
            "rank":     self.rank_heads[horizon](forecast_hidden).squeeze(-1),
            "vol":      self.vol_heads[horizon](forecast_hidden).squeeze(-1),
            "forecast_hidden": forecast_hidden,
        }

    # ----- Loss -----

    def per_head_losses(
        self,
        out: dict[str, torch.Tensor],
        cs: CrossSection,
    ) -> dict[str, torch.Tensor]:
        """Compute the three per-head losses for one cross-section."""
        y = cs.labels.to(out["quantile"].device)
        ranks = cs.ranks.to(out["quantile"].device)

        # Vol-head target. Two modes per `cfg.vol_target_mode` (review-finding
        # Major-2). `abs_y_log` is a §3.4 smoke-only proxy that's biased
        # (zero-return bars collapse to log(eps)). `forward_realized`
        # consumes a `vol_target` field on the CrossSection that §3.5 will
        # plumb through — until then, it raises.
        if self.cfg.vol_target_mode == "abs_y_log":
            vol_target = y.abs().clamp(min=1e-6).log()
        elif self.cfg.vol_target_mode == "forward_realized":
            vt = getattr(cs, "vol_target", None)
            if vt is None:
                msg = (
                    "PostTrainModule: vol_target_mode='forward_realized' but "
                    "CrossSection has no `vol_target` field. Phase 3.5 needs "
                    "to plumb this through `daity.data.cross_section`."
                )
                raise RuntimeError(msg)
            vol_target = vt.to(out["quantile"].device).log().clamp(min=-20.0)
        else:
            msg = f"Unknown vol_target_mode: {self.cfg.vol_target_mode!r}"
            raise ValueError(msg)

        quantiles = self._phase3_quantiles.to(out["quantile"].device)
        l_q = pinball_loss(out["quantile"], y, quantiles)
        l_r = listmle_loss(out["rank"], ranks)
        l_v = F.mse_loss(out["vol"], vol_target)
        return {"quantile": l_q, "rank": l_r, "vol": l_v}

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        cs: CrossSection,
        horizon: str,
    ) -> torch.Tensor:
        """Weighted sum of the three per-head losses for this horizon."""
        per_head = self.per_head_losses(out, cs)
        w = self.loss_weights[horizon]
        return (
            w["quantile"] * per_head["quantile"]
            + w["rank"]    * per_head["rank"]
            + w["vol"]     * per_head["vol"]
        )

    # ----- Magnitude-balance heuristic (PHASE3.md §6.2 + amendment 11) -----

    def calibrate_loss_weights(
        self,
        sample_cs_per_horizon: dict[str, CrossSection],
    ) -> dict[str, dict[str, float]]:
        """Set per-head weights so that `w_h * L_h ≈ 1.0` at step 0.

        For each horizon, take one cross-section, forward, compute per-head
        loss values, set weights as `1.0 / max(loss, eps)`. Returns the
        new weights dict for logging.
        """
        eps = 1e-6
        new_weights: dict[str, dict[str, float]] = {}
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for h, cs in sample_cs_per_horizon.items():
                out = self.forward(cs, h)
                losses = self.per_head_losses(out, cs)
                new_weights[h] = {
                    head: float(1.0 / max(float(losses[head]), eps))
                    for head in ("quantile", "rank", "vol")
                }
        if was_training:
            self.train()
        self.loss_weights = new_weights
        return new_weights

    # ----- Trainable param introspection -----

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Returns (name, parameter) pairs for all trainable params.

        Sanity-check the LoRA + heads sum to << backbone size.
        """
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for _, p in self.trainable_parameters())
