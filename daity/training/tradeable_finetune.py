"""Phase 3X — TradeableFineTuneModule: wrap a Phase-3 checkpoint, freeze
encoder, train a fresh `TradeableHead` on the focal loss.

Architecture:
  Phase-3 state.pt → load PostTrainModule (tokenizer + backbone + LoRA + 12 heads)
                  ↓ freeze ALL
  forward(cs, h)  → forecast_hidden = same as Phase 3
                  ↓
  TradeableHead(forecast_hidden) → 3-way logits
                  ↓
  focal_ce_loss(logits, cs.tradeable_label)

This is the cleanest v1 — only the new 3-way head trains, encoder is the
proven F-13 (or any Phase-3) representation. ~50K head params, ~3 min/fold
to train.

v2+ ideas (deferred): unfreeze LoRA banks, add OB conditioning,
per-horizon heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from daity.data.cross_section import CrossSection
from daity.models.tradeable_head import (
    LONG_IDX, NEUTRAL_IDX, SHORT_IDX,
    CrossSectionTradeableHead, TradeableHead, label_tensor_to_index,
)
from daity.training.losses import (
    direct_pnl_loss, focal_ce_loss, listnet_loss, pinball_loss,
)

# 5-quantile levels used by the xsattn quantile aux head.
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

# Forward-import for Stage 2 OB module.
try:
    from daity.training.finetune import FineTuneConfig, FineTuneModule
except Exception:
    FineTuneConfig = None  # type: ignore[misc, assignment]
    FineTuneModule  = None  # type: ignore[misc, assignment]
from daity.training.posttrain import (
    HORIZONS, PostTrainConfig, PostTrainModule,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TradeableFineTuneConfig:
    parent_checkpoint: Path     # Phase-3 *.state.pt

    head_hidden: int | None = None
    head_dropout: float = 0.1
    focal_gamma: float = 2.0

    # Per-class focal weights (alpha). Default: balanced via class frequencies
    # (LONG, NEUTRAL, SHORT) — overrides via `class_freq` at load time.
    class_alpha: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Phase 3X-B: which encoder parameters to unfreeze.
    # `none` = only TradeableHead trains (default, ~300K params)
    # `lora` = also unfreeze LoRA banks for trainable horizons (~3M params)
    # `all`  = also unfreeze tokenizer + backbone (~40M params)
    # The Phase-3 review showed rank head is misaligned with tradeable labels —
    # may need to adapt deeper than just a head.
    unfreeze_mode: str = "none"
    trainable_horizons: tuple[str, ...] = ("30m",)

    # Optional: warm-start the TradeableHead from a prior TradeableFineTune
    # checkpoint (e.g. head_only result). When unfreezing the encoder, this
    # prevents the head from "chasing a moving target" during early training —
    # the head starts already-optimal for the frozen encoder, encoder then
    # fine-tunes around an already-good operating point.
    head_warmstart_path: Path | None = None

    # Head architecture:
    #   "mlp"    : legacy per-symbol 2-layer MLP (TradeableHead)
    #   "xsattn" : cross-symbol attention head with CLS + FiLM
    head_arch: str = "mlp"
    # xsattn-specific:
    head_n_layers: int = 2
    head_n_heads: int  = 4
    head_d_ff: int     = 512
    # aux_features dim: when > 0, head expects aux input and projects to d_model.
    head_aux_dim: int = 0
    # Loss recipe (only used when head_arch="xsattn"):
    #   primary = ListNet listwise rank loss on LONG/SHORT scores (with
    #   continuous forward return `cs.labels` as soft target).
    #   auxiliary = focal CE for calibration, MSE on CLS market-state.
    rank_loss_weight_long:  float = 1.0
    rank_loss_weight_short: float = 1.0
    focal_aux_weight: float = 0.3
    cls_aux_weight:   float = 0.1
    # Pinball-loss on 5-quantile prediction of forward return. Dense
    # per-symbol regression signal — counterbalances tail-concentrated
    # ListNet gradient. Default 0 (off); enable explicitly in iter4+.
    quantile_aux_weight: float = 0.0
    # Per-side temperatures: LONG positives are denser (9.6%) than SHORT
    # (4.4%), so SHORT often benefits from a broader target distribution.
    rank_target_temperature_long:  float = 0.02
    rank_target_temperature_short: float = 0.02

    # Phase 5: Direct PnL loss (offline contextual bandit). When weight > 0,
    # adds `weight * -E_pi[realized_pnl_long]` to the loss. Requires the
    # tradeable_root parquets to carry `realized_pnl_long` column.
    direct_pnl_loss_weight: float = 0.0
    direct_pnl_temperature: float = 0.5
    direct_pnl_top_k: int = 0   # 0 = use all symbols' scores in softmax
    # Bandit regularizers — default off; needed when pure-PnL / no rank anchor
    # (otherwise softmax collapses to one-hot and gradient stops flowing).
    direct_pnl_entropy_coef: float = 0.0
    direct_pnl_uniform_mix: float = 0.0
    direct_pnl_baseline_subtract: bool = False


class TradeableFineTuneModule(nn.Module):
    """Frozen Phase-3 encoder + fresh `TradeableHead`."""

    def __init__(self, cfg: TradeableFineTuneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Load Phase 3 checkpoint (tokenizer + backbone + LoRA + 12 heads).
        if not cfg.parent_checkpoint.exists():
            msg = f"TradeableFineTuneModule: checkpoint not found at {cfg.parent_checkpoint}"
            raise FileNotFoundError(msg)
        payload = torch.load(cfg.parent_checkpoint, map_location="cpu", weights_only=False)
        module_cfg_dict: dict[str, Any] = dict(payload["module_cfg"])
        if isinstance(module_cfg_dict.get("parent_checkpoint"), str):
            module_cfg_dict["parent_checkpoint"] = Path(module_cfg_dict["parent_checkpoint"])
        # Strip Phase-4-only fields if accidentally present.
        valid_keys = {f.name for f in __import__("dataclasses").fields(PostTrainConfig)}
        module_cfg_dict = {k: v for k, v in module_cfg_dict.items() if k in valid_keys}
        ptcfg = PostTrainConfig(**module_cfg_dict)
        self.posttrain = PostTrainModule(ptcfg)
        missing, _ = self.posttrain.load_state_dict(payload["state_dict"], strict=False)
        if missing:
            log.warning(
                "TradeableFineTuneModule: %d missing keys when loading parent: %s",
                len(missing), missing[:3],
            )

        # Freeze every parameter from the checkpoint first.
        for p in self.posttrain.parameters():
            p.requires_grad_(False)
        # Selectively unfreeze per `cfg.unfreeze_mode`.
        if cfg.unfreeze_mode == "lora":
            # Unfreeze LoRA banks for trainable horizons only.
            for name, param in self.posttrain.peft_backbone.named_parameters():
                if "lora_" in name and any(
                    f".{h}." in name for h in cfg.trainable_horizons
                ):
                    param.requires_grad_(True)
        elif cfg.unfreeze_mode == "all":
            # Unfreeze everything (tokenizer + backbone + LoRA banks).
            # Heads from Phase 3 stay frozen — they're rank/quantile/vol;
            # TradeableHead is the only "head" that learns. Existing heads
            # are dead-weight but harmless.
            for p in self.posttrain.parameters():
                p.requires_grad_(True)
            # Re-freeze the original heads — they don't contribute to
            # tradeable loss and unfreezing them would waste compute.
            for h_dict in (
                self.posttrain.quantile_heads,
                self.posttrain.rank_heads,
                self.posttrain.vol_heads,
            ):
                for h_module in h_dict.values():
                    for p in h_module.parameters():
                        p.requires_grad_(False)
        elif cfg.unfreeze_mode == "encoder_only":
            # Train tokenizer + backbone + LoRA — keep head FROZEN.
            # Useful when head is warm-started: it stays at its optimum while
            # the encoder gets gradient (through the frozen head) and learns
            # features the head can map better. Avoids head/encoder
            # co-adaptation oscillation.
            for p in self.posttrain.parameters():
                p.requires_grad_(True)
            for h_dict in (
                self.posttrain.quantile_heads,
                self.posttrain.rank_heads,
                self.posttrain.vol_heads,
            ):
                for h_module in h_dict.values():
                    for p in h_module.parameters():
                        p.requires_grad_(False)
            # tradeable_head freeze happens AFTER head creation/warmstart below.
        elif cfg.unfreeze_mode != "none":
            msg = (
                f"unknown unfreeze_mode={cfg.unfreeze_mode!r}; "
                "expected none|lora|all|encoder_only"
            )
            raise ValueError(msg)
        # Use posttrain in train mode only if any of its params is trainable
        # (else keep it in eval to skip BN/dropout stats updates).
        if cfg.unfreeze_mode == "none":
            self.posttrain.eval()

        # Fresh head (architecture chosen by cfg.head_arch).
        if cfg.head_arch == "mlp":
            self.tradeable_head: nn.Module = TradeableHead(
                d_in=ptcfg.d_model,
                d_hidden=cfg.head_hidden if cfg.head_hidden is not None else ptcfg.d_model,
                dropout=cfg.head_dropout,
            )
        elif cfg.head_arch == "xsattn":
            self.tradeable_head = CrossSectionTradeableHead(
                d_model=ptcfg.d_model,
                n_layers=cfg.head_n_layers,
                n_heads=cfg.head_n_heads,
                d_ff=cfg.head_d_ff,
                dropout=cfg.head_dropout,
                aux_dim=cfg.head_aux_dim,
            )
        else:
            msg = f"unknown head_arch={cfg.head_arch!r}; expected mlp|xsattn"
            raise ValueError(msg)

        # Optional: warm-start the head from a prior TradeableFineTune checkpoint.
        if cfg.head_warmstart_path is not None:
            if not cfg.head_warmstart_path.exists():
                msg = f"head_warmstart_path not found: {cfg.head_warmstart_path}"
                raise FileNotFoundError(msg)
            warm = torch.load(cfg.head_warmstart_path, map_location="cpu", weights_only=False)
            head_sd = {
                k.removeprefix("tradeable_head."): v
                for k, v in warm["state_dict"].items()
                if k.startswith("tradeable_head.")
            }
            if not head_sd:
                msg = (
                    f"head_warmstart_path {cfg.head_warmstart_path} contains no "
                    "tradeable_head.* keys — wrong checkpoint format?"
                )
                raise ValueError(msg)
            missing, unexpected = self.tradeable_head.load_state_dict(
                head_sd, strict=False,
            )
            log.info(
                "TradeableFineTuneModule: head warm-started from %s "
                "(%d tensors, %d missing, %d unexpected)",
                cfg.head_warmstart_path, len(head_sd),
                len(missing), len(unexpected),
            )
            if len(missing) == len(dict(self.tradeable_head.named_parameters())):
                log.warning(
                    "head_warmstart_path loaded zero matching keys — likely "
                    "wrong head_arch. Head will train from scratch.",
                )

        # encoder_only mode: head stays at its (warm-started) values.
        if cfg.unfreeze_mode == "encoder_only":
            for p in self.tradeable_head.parameters():
                p.requires_grad_(False)
            self.tradeable_head.eval()

    def forward(
        self, cs: CrossSection, horizon: str,
    ) -> dict[str, torch.Tensor]:
        """Returns `{"logits": (B, 3), "long_probs": (B,), "short_probs": (B,)}`.

        The encoder forward is wrapped in `torch.no_grad()` to make backward
        cheap (only TradeableHead receives gradient).
        """
        # Route through this horizon's LoRA bank.
        if self.posttrain._use_lora and self.posttrain._lora_layers:
            for layer in self.posttrain._lora_layers:
                layer._active_adapter = [horizon]
        # When encoder is frozen, save compute + memory by skipping grad
        # tracking; when unfreeze_mode != "none", gradients must flow.
        encoder_no_grad = (self.cfg.unfreeze_mode == "none")
        ctx = torch.no_grad() if encoder_no_grad else torch.enable_grad()
        with ctx:
            tokens = self.posttrain.tokenizer(cs.inputs)
            hidden = self.posttrain.peft_backbone(tokens, causal=False)
            forecast_hidden = hidden[:, 0]                # (B, d_model)
        if self.cfg.head_arch == "xsattn":
            device = forecast_hidden.device
            aux_t = (cs.aux_features.to(device)
                     if cs.aux_features is not None and self.cfg.head_aux_dim > 0
                     else None)
            head_out = self.tradeable_head(
                forecast_hidden=forecast_hidden,
                vol_bucket=cs.vol_bucket.to(device).long(),
                t=cs.t,
                valid_mask=cs.tradeable_valid.to(device) if cs.tradeable_valid is not None
                           else torch.ones(forecast_hidden.shape[0], dtype=torch.bool, device=device),
                aux=aux_t,
            )
            logits = head_out["logits"]                   # (B, 3)
            cls_market = head_out["cls_market"]           # scalar
        else:
            logits = self.tradeable_head(forecast_hidden) # (B, 3)
            cls_market = None
        probs = torch.softmax(logits, dim=-1)             # (B, 3)
        return {
            "logits":      logits,
            "long_probs":  probs[..., LONG_IDX],
            "short_probs": probs[..., SHORT_IDX],
            "forecast_hidden": forecast_hidden,
            "cls_market":  cls_market,
        }

    def compute_loss(
        self, out: dict[str, torch.Tensor], cs: CrossSection,
    ) -> torch.Tensor:
        """Compute loss depending on head_arch.

        head_arch="mlp":   focal CE on 3-way logits, filtered to valid rows.
        head_arch="xsattn": ListNet(LONG) + ListNet(SHORT) + α·focal_ce +
                            β·MSE(CLS, cross-section mean return).
        """
        if cs.tradeable_label is None or cs.tradeable_valid is None:
            msg = (
                "TradeableFineTuneModule.compute_loss: cross-section has no "
                "tradeable_label — was the sampler built with tradeable_root?"
            )
            raise RuntimeError(msg)
        device = out["logits"].device
        valid_mask = cs.tradeable_valid.to(device)
        if valid_mask.sum() == 0:
            # Return zero WITH grad attached so backward() doesn't crash on
            # cross-sections that have no valid rows (common for very strict
            # path-conditioned labels).
            return out["logits"].sum() * 0.0
        logits = out["logits"]                          # (N, 3)
        targets_full = label_tensor_to_index(cs.tradeable_label.to(device))
        alpha = torch.tensor(self.cfg.class_alpha, dtype=logits.dtype, device=device)

        if self.cfg.head_arch == "mlp":
            return focal_ce_loss(
                logits[valid_mask], targets_full[valid_mask],
                alpha=alpha, gamma=self.cfg.focal_gamma, reduction="mean",
            )

        # xsattn loss recipe.
        long_scores  = logits[..., LONG_IDX]            # (N,)
        short_scores = logits[..., SHORT_IDX]           # (N,)
        # `cs.labels` carries the continuous forward return per symbol.
        # Higher = more bullish → use as LONG soft target; negate for SHORT.
        soft_long  = cs.labels.to(device)
        soft_short = -soft_long
        l_long = listnet_loss(
            long_scores, soft_long, mask=valid_mask,
            target_temperature=self.cfg.rank_target_temperature_long,
        )
        l_short = listnet_loss(
            short_scores, soft_short, mask=valid_mask,
            target_temperature=self.cfg.rank_target_temperature_short,
        )
        l_focal = focal_ce_loss(
            logits[valid_mask], targets_full[valid_mask],
            alpha=alpha, gamma=self.cfg.focal_gamma, reduction="mean",
        )
        # CLS aux: predict cross-section mean of continuous return.
        cls_pred = out.get("cls_market")
        if cls_pred is not None:
            cs_mean = soft_long[valid_mask].mean()
            l_cls = (cls_pred - cs_mean).pow(2)
        else:
            l_cls = logits.new_zeros(())
        # Quantile aux: pinball loss on per-symbol forward return.
        q_pred = out.get("quantiles")
        if q_pred is not None and self.cfg.quantile_aux_weight > 0:
            q_levels = torch.tensor(QUANTILE_LEVELS, dtype=q_pred.dtype,
                                    device=device)
            l_quant = pinball_loss(
                q_pred[valid_mask], soft_long[valid_mask], q_levels,
            )
        else:
            l_quant = logits.new_zeros(())
        # Direct PnL (offline contextual bandit): policy-gradient over LONG
        # softmax scores w/ realized PnL as the reward.
        l_pnl = logits.new_zeros(())
        if (self.cfg.direct_pnl_loss_weight > 0
                and cs.realized_pnl_long is not None):
            pnl_long = cs.realized_pnl_long.to(device)
            top_k = self.cfg.direct_pnl_top_k if self.cfg.direct_pnl_top_k > 0 else None
            l_pnl = direct_pnl_loss(
                long_scores, pnl_long, mask=valid_mask,
                top_k=top_k, temperature=self.cfg.direct_pnl_temperature,
                entropy_coef=self.cfg.direct_pnl_entropy_coef,
                uniform_mix=self.cfg.direct_pnl_uniform_mix,
                baseline_subtract=self.cfg.direct_pnl_baseline_subtract,
            )

        return (
            self.cfg.rank_loss_weight_long  * l_long
            + self.cfg.rank_loss_weight_short * l_short
            + self.cfg.focal_aux_weight * l_focal
            + self.cfg.cls_aux_weight   * l_cls
            + self.cfg.quantile_aux_weight * l_quant
            + self.cfg.direct_pnl_loss_weight * l_pnl
        )

    def trainable_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Stage 2: TradeableOBFineTuneModule — Phase 4 OB-fused encoder + xsattn head
# ---------------------------------------------------------------------------

@dataclass
class TradeableOBFineTuneConfig:
    """Stage 2 config: wrap a Phase-4 OB-fused encoder (FineTuneModule)
    and train a fresh `CrossSectionTradeableHead` on top.
    """

    parent_ob_checkpoint: Path     # Phase-4 *.state.pt (OB-F1/F2/F3 etc.)

    # Head architecture (xsattn only for Stage 2 — MLP head doesn't see OB).
    head_n_layers: int = 2
    head_n_heads: int  = 4
    head_d_ff: int     = 512
    head_dropout: float = 0.1
    head_aux_dim: int = 0
    # Warm-start the xsattn head from a Stage 1 checkpoint. STRONGLY
    # recommended — Stage 2's tiny OB-rich window (~455 cs) can't train a
    # fresh head from scratch reliably.
    head_warmstart_path: Path | None = None

    focal_gamma: float = 2.0
    class_alpha: tuple[float, float, float] = (1.0, 1.0, 1.0)

    rank_loss_weight_long:  float = 1.0
    rank_loss_weight_short: float = 1.0
    focal_aux_weight: float = 0.3
    cls_aux_weight:   float = 0.1
    quantile_aux_weight: float = 0.0
    rank_target_temperature_long:  float = 0.02
    rank_target_temperature_short: float = 0.05

    # Phase 5: direct-PnL offline contextual bandit (same as Stage 1 config).
    direct_pnl_loss_weight: float = 0.0
    direct_pnl_temperature: float = 0.5
    direct_pnl_top_k: int = 0
    # Bandit regularizers — default off; needed when pure-PnL / no rank anchor.
    direct_pnl_entropy_coef: float = 0.0
    direct_pnl_uniform_mix: float = 0.0
    direct_pnl_baseline_subtract: bool = False

    # Stage 2 encoder unfreeze:
    #   "none"   = encoder frozen (just like Stage 1, gate stays at Phase 4 value)
    #   "fusion" = unfreeze fusion gate + OB encoder + cross-attn fusion layers
    #              (lets the head "tune" how OB signal arrives)
    #   "all"    = everything (tokenizer + backbone + LoRA + fusion + ob_encoder)
    unfreeze_mode: str = "none"
    # If True, force OB fusion gate to remain zero (ablation: equivalent to
    # Phase 3 encoder, used to sanity-check Stage 2 baseline).
    force_gate_zero: bool = False


class TradeableOBFineTuneModule(nn.Module):
    """Phase-4 OB-fused encoder + xsattn `CrossSectionTradeableHead`."""

    def __init__(self, cfg: TradeableOBFineTuneConfig) -> None:
        super().__init__()
        if FineTuneModule is None or FineTuneConfig is None:
            msg = "daity.training.finetune.FineTuneModule unavailable — Stage 2 requires Phase 4"
            raise ImportError(msg)
        self.cfg = cfg

        # Load Phase-4 checkpoint.
        if not cfg.parent_ob_checkpoint.exists():
            msg = f"parent_ob_checkpoint not found at {cfg.parent_ob_checkpoint}"
            raise FileNotFoundError(msg)
        payload = torch.load(cfg.parent_ob_checkpoint, map_location="cpu", weights_only=False)
        ft_cfg_dict: dict[str, Any] = dict(payload["module_cfg"])
        if isinstance(ft_cfg_dict.get("parent_checkpoint"), str):
            ft_cfg_dict["parent_checkpoint"] = Path(ft_cfg_dict["parent_checkpoint"])
        valid_keys = {f.name for f in __import__("dataclasses").fields(FineTuneConfig)}
        ft_cfg_dict = {k: v for k, v in ft_cfg_dict.items() if k in valid_keys}
        ft_cfg = FineTuneConfig(**ft_cfg_dict)
        self.encoder = FineTuneModule(ft_cfg)
        missing, _ = self.encoder.load_state_dict(payload["state_dict"], strict=False)
        if missing:
            log.warning(
                "TradeableOBFineTuneModule: %d missing keys when loading Phase-4 state: %s",
                len(missing), missing[:5],
            )

        # Curriculum: gate-force-zero is a Phase-4 training device. For
        # Stage 2 we want the gate to remain learned (or at least usable).
        self.encoder.set_gate_force_zero(cfg.force_gate_zero)
        # Use OB by default — that's the whole point of Stage 2.
        self.encoder.use_ob = True

        # Freeze encoder per `unfreeze_mode`.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if cfg.unfreeze_mode == "fusion":
            # Unfreeze OB encoder + fusion layers + gate inside fused_backbone.
            for p in self.encoder.ob_encoder.parameters():
                p.requires_grad_(True)
            for name, param in self.encoder.fused_backbone.named_parameters():
                if "fusion" in name or "gate" in name:
                    param.requires_grad_(True)
        elif cfg.unfreeze_mode == "all":
            for p in self.encoder.parameters():
                p.requires_grad_(True)
        elif cfg.unfreeze_mode != "none":
            msg = f"unknown unfreeze_mode={cfg.unfreeze_mode!r}; expected none|fusion|all"
            raise ValueError(msg)
        if cfg.unfreeze_mode == "none":
            self.encoder.eval()

        # Fresh xsattn head (warm-started from Stage 1 if available).
        d_model = self.encoder.posttrain.cfg.d_model
        self.tradeable_head = CrossSectionTradeableHead(
            d_model=d_model,
            n_layers=cfg.head_n_layers,
            n_heads=cfg.head_n_heads,
            d_ff=cfg.head_d_ff,
            dropout=cfg.head_dropout,
            aux_dim=cfg.head_aux_dim,
        )
        if cfg.head_warmstart_path is not None:
            if not cfg.head_warmstart_path.exists():
                msg = f"head_warmstart_path not found: {cfg.head_warmstart_path}"
                raise FileNotFoundError(msg)
            warm = torch.load(cfg.head_warmstart_path, map_location="cpu", weights_only=False)
            head_sd = {
                k.removeprefix("tradeable_head."): v
                for k, v in warm["state_dict"].items()
                if k.startswith("tradeable_head.")
            }
            if not head_sd:
                msg = f"head_warmstart_path {cfg.head_warmstart_path} has no tradeable_head.*"
                raise ValueError(msg)
            missing, unexpected = self.tradeable_head.load_state_dict(head_sd, strict=False)
            log.info(
                "TradeableOBFineTuneModule: head warm-started from %s "
                "(%d tensors, %d missing, %d unexpected)",
                cfg.head_warmstart_path, len(head_sd), len(missing), len(unexpected),
            )

    def forward(self, cs: CrossSection, horizon: str) -> dict[str, torch.Tensor]:
        encoder_no_grad = (self.cfg.unfreeze_mode == "none")
        ctx = torch.no_grad() if encoder_no_grad else torch.enable_grad()
        with ctx:
            ft_out = self.encoder(cs, horizon)
        forecast_hidden = ft_out["forecast_hidden"]                     # (B, d_model)
        device = forecast_hidden.device
        aux_t = (cs.aux_features.to(device)
                 if cs.aux_features is not None and self.cfg.head_aux_dim > 0
                 else None)
        head_out = self.tradeable_head(
            forecast_hidden=forecast_hidden,
            vol_bucket=cs.vol_bucket.to(device).long(),
            t=cs.t,
            valid_mask=cs.tradeable_valid.to(device) if cs.tradeable_valid is not None
                       else torch.ones(forecast_hidden.shape[0], dtype=torch.bool, device=device),
            aux=aux_t,
        )
        logits = head_out["logits"]
        probs = torch.softmax(logits, dim=-1)
        return {
            "logits":          logits,
            "long_probs":      probs[..., LONG_IDX],
            "short_probs":     probs[..., SHORT_IDX],
            "forecast_hidden": forecast_hidden,
            "cls_market":      head_out.get("cls_market"),
            "quantiles":       head_out.get("quantiles"),
        }

    def compute_loss(
        self, out: dict[str, torch.Tensor], cs: CrossSection,
    ) -> torch.Tensor:
        if cs.tradeable_label is None or cs.tradeable_valid is None:
            msg = "TradeableOBFineTuneModule.compute_loss: cross-section has no tradeable_label"
            raise RuntimeError(msg)
        device = out["logits"].device
        valid_mask = cs.tradeable_valid.to(device)
        if valid_mask.sum() == 0:
            # Return zero WITH grad attached so backward() doesn't crash on
            # cross-sections that have no valid rows (common for very strict
            # path-conditioned labels).
            return out["logits"].sum() * 0.0
        logits = out["logits"]
        targets_full = label_tensor_to_index(cs.tradeable_label.to(device))
        alpha = torch.tensor(self.cfg.class_alpha, dtype=logits.dtype, device=device)

        long_scores  = logits[..., LONG_IDX]
        short_scores = logits[..., SHORT_IDX]
        soft_long  = cs.labels.to(device)
        soft_short = -soft_long
        l_long = listnet_loss(
            long_scores, soft_long, mask=valid_mask,
            target_temperature=self.cfg.rank_target_temperature_long,
        )
        l_short = listnet_loss(
            short_scores, soft_short, mask=valid_mask,
            target_temperature=self.cfg.rank_target_temperature_short,
        )
        l_focal = focal_ce_loss(
            logits[valid_mask], targets_full[valid_mask],
            alpha=alpha, gamma=self.cfg.focal_gamma, reduction="mean",
        )
        cls_pred = out.get("cls_market")
        if cls_pred is not None:
            cs_mean = soft_long[valid_mask].mean()
            l_cls = (cls_pred - cs_mean).pow(2)
        else:
            l_cls = logits.new_zeros(())
        q_pred = out.get("quantiles")
        if q_pred is not None and self.cfg.quantile_aux_weight > 0:
            q_levels = torch.tensor(QUANTILE_LEVELS, dtype=q_pred.dtype, device=device)
            l_quant = pinball_loss(
                q_pred[valid_mask], soft_long[valid_mask], q_levels,
            )
        else:
            l_quant = logits.new_zeros(())
        # Direct PnL (offline contextual bandit): policy-gradient over LONG
        # softmax scores w/ realized PnL as the reward.
        l_pnl = logits.new_zeros(())
        if (self.cfg.direct_pnl_loss_weight > 0
                and cs.realized_pnl_long is not None):
            pnl_long = cs.realized_pnl_long.to(device)
            top_k = self.cfg.direct_pnl_top_k if self.cfg.direct_pnl_top_k > 0 else None
            l_pnl = direct_pnl_loss(
                long_scores, pnl_long, mask=valid_mask,
                top_k=top_k, temperature=self.cfg.direct_pnl_temperature,
                entropy_coef=self.cfg.direct_pnl_entropy_coef,
                uniform_mix=self.cfg.direct_pnl_uniform_mix,
                baseline_subtract=self.cfg.direct_pnl_baseline_subtract,
            )

        return (
            self.cfg.rank_loss_weight_long  * l_long
            + self.cfg.rank_loss_weight_short * l_short
            + self.cfg.focal_aux_weight * l_focal
            + self.cfg.cls_aux_weight   * l_cls
            + self.cfg.quantile_aux_weight * l_quant
            + self.cfg.direct_pnl_loss_weight * l_pnl
        )

    def trainable_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]
