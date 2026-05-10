"""Diagnose: v3_features produced NaN train_loss from step 9 onward across
all 1,119 logged steps. The 18-channel + log-return + features path is
hitting NaN somewhere in `_compute_loss`. Bisect to find where.

Steps in `_compute_loss` order — check each tensor for NaN/Inf:
  1. raw input batch (read from features_parquet)
  2. forecast anchor capture (last input close)
  3. to_log_returns_partial on inputs
  4. to_log_returns_partial on future bars
  5. build_targets — RevIN-norm patches
  6. tokenizer(batch) — RevIN + project + concat
  7. apply_random_patch_mask
  8. backbone non-causal forward
  9. backbone causal forward
  10. recon_head, ntp_head outputs
  11. forecast head output
  12. losses

Print the first NaN/Inf-containing tensor.
"""

# %%
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from daity.data.datasets import FORECAST_FUTURE_KEY, OHLCVPretrainDataset
from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.parquet_store import ParquetStore
from daity.data.preprocess import (
    IDX_CLOSE,
    N_CHANNELS_OHLCV,
    to_log_returns_partial,
)
from daity.data.tokenizer_targets import build_forecast_targets, build_targets
from daity.training.losses import apply_random_patch_mask
from daity.training.pretrain import PretrainConfig, PretrainModule

torch.manual_seed(0)


def report(name: str, t: torch.Tensor | None) -> None:
    if t is None:
        print(f"  {name:<40s}  None")
        return
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    finite = t[torch.isfinite(t)]
    if finite.numel() > 0:
        mn, mx, mean, std = finite.min().item(), finite.max().item(), finite.mean().item(), finite.std().item()
    else:
        mn = mx = mean = std = float("nan")
    flag = "🔴" if nan or inf else "  "
    print(
        f"  {flag}{name:<38s} shape={list(t.shape)} "
        f"nan={n_nan} inf={n_inf} "
        f"min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g}"
    )


# %%
# Build the production-config v3_features setup.
cfg = PretrainConfig(
    scales=("5m", "15m", "60m", "day"),
    num_channels=18,
    d_model=480, n_layers=12, n_heads=8, ffn_ratio=4,
    patch_len=16, patch_stride=8, max_seq_len=128,
    mask_ratio=0.4, contrastive_proj_dim=128,
    forecast_scale="5m", forecast_n_patches=6,
    input_form="log_returns",
    target_form="log_returns",
    revin_affine=False,
    w_masked=1.0, w_ntp=0.5, w_contrastive=0.0, w_forecast=0.5,
    lr=7e-4, weight_decay=1e-2, warmup_pct=0.025, max_steps=150_000,
    lr_min_ratio=0.05, ema_decay=0.999,
)
print(f"Config: num_channels={cfg.num_channels}, input_form={cfg.input_form}, target_form={cfg.target_form}")

# %%
# Build dataset over feature_parquet (the 18-channel store).
store = ParquetStore(Path("data/features_parquet"))
on_disk = sorted(p.stem for p in store.list_partitions("day"))
print(f"Universe: {len(on_disk)} symbols on disk")

ds = OHLCVPretrainDataset(
    store=store, symbols=on_disk[:8],
    as_of=datetime(2025, 12, 31, tzinfo=UTC),
    window_bars={"5m": 256, "15m": 192, "60m": 168, "day": 64},
    channels=CHANNEL_ORDER_18,
    seed=42,
    forecast_scale="5m",
    forecast_n_bars=6 * 16,
)
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch = next(iter(loader))

print("\n=== STEP 1: Raw batch from feature_parquet ===")
for k, v in batch.items():
    report(k, v)


# %%
# Pop forecast tail.
future_bars = batch.pop(FORECAST_FUTURE_KEY)
print("\n=== STEP 2: Capture forecast anchor (last input close at forecast_scale) ===")
anchor = batch[cfg.forecast_scale][:, -1, IDX_CLOSE].clone()
report("forecast_anchor", anchor)


# %%
# Apply log-return transform to inputs (channels 0:5 of each scale).
print("\n=== STEP 3: to_log_returns_partial on inputs (per-scale) ===")
batch_lr = {}
for sc, x in batch.items():
    x_lr = to_log_returns_partial(x, n_bar_channels=N_CHANNELS_OHLCV)
    report(f"input[{sc}] post-LR", x_lr)
    batch_lr[sc] = x_lr


# %%
# Apply log-return transform to future bars.
print("\n=== STEP 4: to_log_returns_partial on future_bars (anchored) ===")
future_lr = to_log_returns_partial(
    future_bars, n_bar_channels=N_CHANNELS_OHLCV, anchor_close=anchor,
)
report("future_lr", future_lr)


# %%
# Build module.
mod = PretrainModule(cfg)
mod.eval()  # disable dropout for diagnosis

print("\n=== STEP 5: build_targets (RevIN-norm patches) ===")
targets, volumes = build_targets(
    batch=batch_lr, scales=cfg.scales,
    patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
    tokenizer=mod.tokenizer,
)
report("recon_targets", targets)
report("volumes", volumes)


# %%
print("\n=== STEP 6: tokenizer(batch_lr) ===")
tokens = mod.tokenizer(batch_lr)
report("tokens", tokens)


# %%
# Look at RevIN cache state for each scale — std=0 channels would explain things.
print("\n=== STEP 6b: RevIN cache (mean, std) per scale — looking for std=0 channels ===")
for sc in cfg.scales:
    revin = mod.tokenizer.revins[sc]
    report(f"revin[{sc}]._mean", revin._mean)
    report(f"revin[{sc}]._std", revin._std)
    # Specifically: any std value at or near 0?
    n_near_zero = (revin._std < 1e-6).sum().item()
    if n_near_zero > 0:
        print(
            f"     ⚠ revin[{sc}] has {n_near_zero} (B*C) cells with std < 1e-6 "
            f"(would cause /0 in RevIN _normalize)"
        )


# %%
print("\n=== STEP 7: build_forecast_targets ===")
forecast_targets = build_forecast_targets(
    future_bars=future_lr,
    forecast_scale=cfg.forecast_scale,
    n_patches=cfg.forecast_n_patches,
    patch_len=cfg.patch_len,
    tokenizer=mod.tokenizer,
)
report("forecast_targets", forecast_targets)


# %%
print("\n=== STEP 8: backbone forward (non-causal) on tokens ===")
masked_tokens, mask = apply_random_patch_mask(
    tokens, mask_ratio=cfg.mask_ratio, mask_token=mod.mask_token,
)
report("masked_tokens", masked_tokens)
hidden_nc = mod.backbone(masked_tokens, causal=False)
report("hidden_nc", hidden_nc)


# %%
print("\n=== STEP 9: head outputs ===")
recon_out_flat = mod.recon_head(hidden_nc[:, 1:])
report("recon_out_flat", recon_out_flat)


# %%
print("\n=== STEP 10: full _compute_loss output ===")
batch_full = {**batch, FORECAST_FUTURE_KEY: future_bars}
try:
    loss, parts = mod._compute_loss(batch_full)
    print(f"  loss = {loss.item():.6e}")
    for name, val in parts.items():
        print(f"  {name:<22s} = {val.item():.6e}")
except Exception as exc:
    print(f"  EXCEPTION: {exc}")
