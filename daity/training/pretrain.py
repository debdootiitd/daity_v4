"""Pretraining LightningModule for daity_v3 (DESIGN §3.2).

Wires the tokenizer + backbone + 4 SSL heads + 4 losses into a single
`LightningModule` ready for `lightning.Trainer.fit`. Loss is the weighted
sum of the four SSL objectives (defaults: 1.0 / 0.5 / 0.3 / 0.5 for
masked / NTP / contrastive / multi-horizon forecast).

Architecture flow per training step:
  1. `OHLCVPretrainDataset` yields a batch: `{scale: (B, L_scale, C)}`,
     plus an optional `__forecast_future__` key with `(B, F, C)` future
     bars at the forecast scale (when forecast SSL is enabled).
  2. `MultiResTokenizer` produces `(B, T, d_model)` with FORECAST at pos 0
     and per-scale patches following.
  3. **Mask** a fraction of patch positions (DESIGN: 30-40%); replace with
     a learnable `mask_token`. Position 0 is never masked.
  4. **Backbone** runs twice:
       a. non-causal pass over the masked sequence → for masked-recon +
          contrastive head + multi-horizon forecast head.
       b. causal pass over the unmasked sequence → for next-patch
          prediction head.
  5. **Heads + losses**:
       - Masked-recon: vol-weighted MSE on the masked patches' RevIN-
         normalized targets.
       - NTP: unweighted MSE on shifted-by-one prediction.
       - Contrastive: InfoNCE between FORECAST projections of the two
         passes. Phase 2 implementation: positives are the (causal,
         non-causal) pair from the SAME sample. Phase 3+ will replace
         this with the cross-symbol same-date InfoNCE described in DESIGN.
       - Multi-horizon forecast: MSE between H non-overlapping future
         patches at `forecast_scale` and the head's prediction read off
         the FORECAST token's non-causal hidden. Disabled when
         cfg.forecast_n_patches == 0.

Optimizer: AdamW with cosine LR + 5% warmup. EMA: decay 0.999 of every
parameter, evaluated separately for val metrics.

The module is precision-agnostic: pass `precision='bf16-mixed'` to the
Trainer for A100 / Vast and `precision='32-true'` for CPU smoke.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import lightning as L
import torch
from torch import nn

from daity.data.datasets import FORECAST_FUTURE_KEY
from daity.data.preprocess import (
    IDX_CLOSE,
    N_CHANNELS_OHLCV,
    to_log_returns,
    to_log_returns_partial,
)
from daity.data.tokenizer_targets import build_forecast_targets, build_targets
from daity.models.backbone import Backbone
from daity.models.heads import (
    ContrastiveHead,
    CrossAttentionForecastHead,
    MaskedReconstructionHead,
    MultiHorizonForecastHead,
    NextPatchHead,
)
from daity.models.tokenizer import (
    DEFAULT_PATCH_LEN,
    DEFAULT_PATCH_STRIDE,
    MultiResTokenizer,
)
from daity.training.losses import (
    DEFAULT_MASK_RATIO,
    apply_random_patch_mask,
    info_nce_loss,
    next_patch_mse,
    volume_weighted_mse,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class PretrainConfig:
    """All hyperparameters for the SSL pretraining run."""

    # Data
    scales: tuple[str, ...] = ("5m", "15m", "60m", "day")
    num_channels: int = 5
    # Tokenizer / backbone
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 8
    ffn_ratio: int = 4
    patch_len: int = DEFAULT_PATCH_LEN
    patch_stride: int = DEFAULT_PATCH_STRIDE
    max_seq_len: int = 512
    # SSL
    mask_ratio: float = DEFAULT_MASK_RATIO
    contrastive_proj_dim: int = 128
    contrastive_temperature: float = 0.07
    # RevIN affine. Default False for SSL pretraining (DESIGN amendment 12):
    # with `affine=True`, the learnable gamma/beta are present in BOTH the
    # input projections AND the recon/forecast targets (since `build_targets`
    # calls the same RevIN). The optimizer can shrink gamma → 0 to make
    # targets arbitrarily small and drive MSE → 0 without learning anything
    # useful (validated on the H200 run: gamma collapsed to ~0.04 over 30K
    # steps, val_loss hit 1e-8). For classification heads (the original
    # RevIN paper's setup), `affine=True` is fine because the target is the
    # class label, unaffected by RevIN. For our SSL-on-RevIN-output setup,
    # affine=False is mandatory.
    revin_affine: bool = False
    # Multi-horizon forecast on FORECAST token. forecast_n_patches=0
    # disables the head (and the dataset will yield no future bars).
    # Default: 6 non-overlapping patches at 5m/patch_len=16 = 8h ≈ 1
    # trading day forward.
    forecast_scale: str = "5m"
    forecast_n_patches: int = 6
    # Phase 2.4 (per architect review §3.1): forecast head architecture.
    # "mlp" — original single-vector MLP from FORECAST token (legacy; preserves
    #         v1/v2/v2_long/v3_lr/v3_features behavior for back-compat).
    # "cross_attention" — H learnable queries cross-attend over the full
    #         encoder output (canonical Time-FM design, PatchTST/MOIRAI/Chronos).
    #         Removes the single-vector bottleneck that's the leading
    #         hypothesis for the val_forecast plateau.
    forecast_head_type: str = "mlp"
    # Number of channels the FORECAST head predicts. Defaults to None ⇒ same
    # as `num_channels` (back-compat for 5-channel runs). For 18-channel
    # feature_parquet runs, set this to 5 — predict ONLY the bar_channels
    # (OHLCV in log-return form). The 13 derived feature channels (log-return
    # lags, wicks, vol_z, time covariates) include heavy-tailed targets
    # (especially log_volume) that produce ~10⁵-10⁸ MSE spikes on outlier
    # batches even with gradient clipping. Forecasting only OHLCV-log-returns
    # is also the canonical target form for Phase 3's quantile head.
    forecast_num_channels: int | None = None
    # Input / target representation (Phase 2.3). Default "raw" matches v1/v2/v2_long
    # behavior — the tokenizer sees raw OHLCV and RevIN normalizes per-window;
    # forecast targets are RevIN-normalized raw OHLCV.
    #
    # `input_form="log_returns"` + `target_form="log_returns"`: the bars are
    # transformed to a 5-channel stationary log-return form (see
    # `daity.data.preprocess.to_log_returns`) before RevIN. Forecast targets
    # are anchored to the LAST INPUT BAR'S CLOSE so the first future log-return
    # is genuinely predictive (no peeking). Empirically the canonical Time-FM
    # choice; speeds up forecast-head convergence by removing the random-walk
    # component the model would otherwise have to learn to subtract out.
    input_form: str = "raw"          # "raw" | "log_returns"
    target_form: str = "absolute"    # "absolute" | "log_returns"
    # Loss weights (DESIGN §3.2 initial: 1.0 / 0.5 / 0.3 / 0.5)
    w_masked: float = 1.0
    w_ntp: float = 0.5
    w_contrastive: float = 0.3
    w_forecast: float = 0.5
    # Optim
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_pct: float = 0.05
    max_steps: int = 100_000
    # Cosine LR floor as a fraction of peak LR. Default 0 (decay all the way
    # to zero — original v1 behavior). For long runs, a non-zero floor
    # (typically 0.05–0.10) keeps the model learning during the late phase
    # rather than effectively freezing it. Applied as
    #   lr(step) = lr_min_ratio + (1 - lr_min_ratio) * cosine(progress)
    # so the schedule still starts at peak (cosine(0)=1) and ends at
    # `lr_min_ratio × peak` (cosine(1)=0).
    lr_min_ratio: float = 0.0
    # EMA
    ema_decay: float = 0.999


class PretrainModule(L.LightningModule):
    """LightningModule encapsulating the full Phase 2 pretraining graph."""

    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__()
        # Lightning needs hyperparameters as a dict-like for checkpoint
        # serialization. Cast the dataclass.
        self.save_hyperparameters({
            f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()
        })
        self.cfg = cfg

        self.tokenizer = MultiResTokenizer(
            scales=cfg.scales, num_channels=cfg.num_channels,
            d_model=cfg.d_model,
            patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
            revin_affine=cfg.revin_affine,
        )
        self.backbone = Backbone(
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            ffn_ratio=cfg.ffn_ratio, max_seq_len=cfg.max_seq_len,
        )
        self.recon_head = MaskedReconstructionHead(
            d_model=cfg.d_model, num_channels=cfg.num_channels,
            patch_len=cfg.patch_len,
        )
        self.ntp_head = NextPatchHead(
            d_model=cfg.d_model, num_channels=cfg.num_channels,
            patch_len=cfg.patch_len,
        )
        self.contrastive_head = ContrastiveHead(
            d_model=cfg.d_model, proj_dim=cfg.contrastive_proj_dim,
        )
        # Forecast-target channel count: defaults to num_channels (back-compat),
        # but for 18-channel feature_parquet runs we want to forecast only
        # the 5 OHLCV bar_channels (heavy-tailed log_volume / vol_z make the
        # other 13 channels destructively bad MSE targets — see config doc).
        self._forecast_num_channels = (
            cfg.forecast_num_channels
            if cfg.forecast_num_channels is not None
            else cfg.num_channels
        )

        # Multi-horizon forecast head. Disabled (None) when
        # cfg.forecast_n_patches == 0 — keeps the head out of the param
        # count for ablation runs.
        # `forecast_head_type` (Phase 2.4) selects between:
        #   "mlp"             — single-vector MLP from FORECAST token (legacy)
        #   "cross_attention" — H queries cross-attend over full encoder output
        # See `daity/models/heads.py` for the architectural rationale.
        self.forecast_head: nn.Module | None
        if cfg.forecast_n_patches > 0:
            if cfg.forecast_head_type == "mlp":
                self.forecast_head = MultiHorizonForecastHead(
                    d_model=cfg.d_model, n_patches=cfg.forecast_n_patches,
                    num_channels=self._forecast_num_channels,
                    patch_len=cfg.patch_len,
                )
            elif cfg.forecast_head_type == "cross_attention":
                self.forecast_head = CrossAttentionForecastHead(
                    d_model=cfg.d_model, n_patches=cfg.forecast_n_patches,
                    num_channels=self._forecast_num_channels,
                    patch_len=cfg.patch_len,
                    n_heads=cfg.n_heads,
                )
            else:
                msg = (
                    f"Unknown forecast_head_type {cfg.forecast_head_type!r}; "
                    f"expected 'mlp' or 'cross_attention'"
                )
                raise ValueError(msg)
        else:
            self.forecast_head = None

        # Learnable [MASK] embedding, broadcast over masked patch positions.
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Cache patch counts per scale (for slicing the token sequence).
        # Filled lazily in the first forward pass since they depend on
        # the input window lengths the dataset provides.
        self._n_patches_per_scale: dict[str, int] | None = None

    # ----- Forward / loss -----

    def _patches_per_scale(self, batch: dict[str, torch.Tensor]) -> dict[str, int]:
        if self._n_patches_per_scale is None:
            seq_lens = {sc: t.size(1) for sc, t in batch.items()}
            self._n_patches_per_scale = self.tokenizer.n_patches_per_scale(seq_lens)
        return self._n_patches_per_scale

    def _build_targets(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the per-patch targets + per-patch volume.

        Returns:
          targets: (B, P, C, patch_len) — the original-but-RevIN-normalized
                   patches the masked-recon head should reconstruct.
          volumes: (B, P) — per-patch mean volume (used by vol-weighted MSE).
        """
        return build_targets(
            batch=batch,
            scales=self.cfg.scales,
            patch_len=self.cfg.patch_len,
            patch_stride=self.cfg.patch_stride,
            tokenizer=self.tokenizer,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, parts = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        for name, val in parts.items():
            self.log(f"train_{name}", val, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, parts = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for name, val in parts.items():
            self.log(f"val_{name}", val, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _compute_loss(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run all SSL objectives and return weighted total + parts.

        Non-mutating: we work on a shallow copy of `batch` so callers can
        re-use the same dict across multiple steps (e.g. overfit tests).
        """
        # ----- Pop the optional forecast-future tail before tokenization.
        # The dataset packs it under a reserved key so the DataLoader can
        # collate everything at once; the tokenizer must NOT see it. -----
        future_bars: torch.Tensor | None = batch.get(FORECAST_FUTURE_KEY)
        if future_bars is not None:
            batch = {k: v for k, v in batch.items() if k != FORECAST_FUTURE_KEY}

        # ----- (Phase 2.3) Optional log-return transform on inputs +
        # forecast targets. The anchor for the forecast target's first
        # bar is the LAST INPUT BAR'S CLOSE at `forecast_scale`, captured
        # HERE before we transform anything (the transform doesn't keep
        # raw close prices, so we'd lose the anchor downstream). -----
        forecast_anchor_close: torch.Tensor | None = None
        need_anchor = (
            future_bars is not None
            and self.cfg.target_form == "log_returns"
            and self.cfg.forecast_scale in batch
        )
        if need_anchor:
            forecast_anchor_close = batch[self.cfg.forecast_scale][:, -1, IDX_CLOSE].clone()

        if self.cfg.input_form == "log_returns":
            # When num_channels > 5, only the first 5 channels are bar_channels
            # (raw OHLCV); the rest are already-stationary feature-engine
            # outputs (log-return lags, wicks, vol_z, time covariates) — they
            # pass through unchanged. `to_log_returns_partial` handles both
            # the 5-channel and ≥5-channel cases.
            batch = {
                sc: to_log_returns_partial(x, n_bar_channels=N_CHANNELS_OHLCV)
                for sc, x in batch.items()
            }

        if future_bars is not None and self.cfg.target_form == "log_returns":
            if forecast_anchor_close is None:
                msg = (
                    "target_form=log_returns requires forecast_scale to appear "
                    "in batch (it should — check cfg.scales)"
                )
                raise RuntimeError(msg)
            future_bars = to_log_returns_partial(
                future_bars,
                n_bar_channels=N_CHANNELS_OHLCV,
                anchor_close=forecast_anchor_close,
            )

        # ----- Build targets BEFORE feeding tokenizer (we need the
        # post-RevIN patch tensor as the reconstruction target) -----
        targets, volumes = self._build_targets(batch)        # (B, P, C, patch_len), (B, P)

        # ----- Tokenize (this also runs RevIN; same stats now cached) -----
        tokens = self.tokenizer(batch)                       # (B, 1+P, d_model)

        # ----- Masked-recon path: mask + non-causal backbone + recon head -----
        masked_tokens, mask = apply_random_patch_mask(
            tokens, mask_ratio=self.cfg.mask_ratio, mask_token=self.mask_token,
        )
        hidden_nc = self.backbone(masked_tokens, causal=False)
        # Slice off FORECAST + reshape head output to (B, P, C, patch_len).
        recon_out = self.recon_head(hidden_nc[:, 1:])
        B, P, _ = recon_out.shape
        recon_out = recon_out.view(B, P, self.cfg.num_channels, self.cfg.patch_len)
        # mask is (B, 1+P); slice off FORECAST to get (B, P).
        patch_mask = mask[:, 1:]
        loss_recon = volume_weighted_mse(
            recon_out, targets, volume=volumes, mask=patch_mask,
        )

        # ----- Next-patch path: causal backbone over UNMASKED tokens -----
        hidden_c = self.backbone(tokens, causal=True)
        ntp_out = self.ntp_head(hidden_c[:, 1:])             # predictions at every patch
        ntp_out = ntp_out.view(B, P, self.cfg.num_channels, self.cfg.patch_len)
        # Predict patch t+1 from prefix at position t. So pred[:, :-1] vs
        # target[:, 1:]. The mask is "all but the last position".
        pred_shifted = ntp_out[:, :-1]
        target_shifted = targets[:, 1:]
        ntp_mask = torch.ones(B, P - 1, dtype=torch.bool, device=tokens.device)
        loss_ntp = next_patch_mse(pred_shifted, target_shifted, mask=ntp_mask)

        # ----- Contrastive path: positives = (causal, non-causal) FORECASTs -----
        z_nc = self.contrastive_head(hidden_nc[:, 0])
        z_c = self.contrastive_head(hidden_c[:, 0])
        loss_contrast = info_nce_loss(
            z_nc, z_c, temperature=self.cfg.contrastive_temperature,
        )

        # ----- Multi-horizon forecast path. -----
        # MLP head (legacy) reads hidden_nc[:, 0] — the FORECAST token's
        # post-encoder hidden state, a single (B, d_model) bottleneck.
        # Cross-attention head reads hidden_nc[:, 1:] — the full patch-token
        # encoder output (B, T-1, d_model), letting H learnable forecast
        # queries attend over all 81 patch tokens. Architect's review §3.1:
        # the MLP's single-vector bottleneck is the leading hypothesis for
        # the val_forecast plateau; cross-attention removes it.
        loss_forecast: torch.Tensor
        if self.forecast_head is not None and future_bars is not None:
            forecast_targets = build_forecast_targets(
                future_bars=future_bars,
                forecast_scale=self.cfg.forecast_scale,
                n_patches=self.cfg.forecast_n_patches,
                patch_len=self.cfg.patch_len,
                tokenizer=self.tokenizer,
            )                                                 # (B, H, num_channels, patch_len)
            # Slice to forecast_num_channels — for 18-channel feature_parquet
            # runs this drops the heavy-tailed derived feature targets and
            # keeps only the OHLCV bar_channels (channels 0:5). For 5-channel
            # raw OHLCV runs this is a no-op (forecast_num_channels == 5).
            if self._forecast_num_channels < self.cfg.num_channels:
                forecast_targets = forecast_targets[:, :, :self._forecast_num_channels, :]
            if self.cfg.forecast_head_type == "cross_attention":
                forecast_pred = self.forecast_head(hidden_nc[:, 1:])
            else:                                            # "mlp"
                forecast_pred = self.forecast_head(hidden_nc[:, 0])
            loss_forecast = ((forecast_pred - forecast_targets) ** 2).mean()
        else:
            # Forecast head disabled or batch lacks future bars: contribute
            # a zero on the same device/dtype so the optimizer graph is
            # uniform across batches.
            loss_forecast = torch.zeros((), device=tokens.device, dtype=loss_recon.dtype)

        loss = (
            self.cfg.w_masked * loss_recon
            + self.cfg.w_ntp * loss_ntp
            + self.cfg.w_contrastive * loss_contrast
            + self.cfg.w_forecast * loss_forecast
        )
        return loss, {
            "loss_recon": loss_recon.detach(),
            "loss_ntp": loss_ntp.detach(),
            "loss_contrast": loss_contrast.detach(),
            "loss_forecast": loss_forecast.detach(),
        }

    # ----- Optimizer + LR -----

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.95),
        )
        warmup_steps = max(1, int(self.cfg.warmup_pct * self.cfg.max_steps))
        max_steps = self.cfg.max_steps

        min_ratio = float(self.cfg.lr_min_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Cosine from 1.0 → min_ratio over [warmup, max_steps].
            # cos_factor goes 1 → 0 as progress goes 0 → 1.
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_ratio + (1.0 - min_ratio) * cos_factor

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
