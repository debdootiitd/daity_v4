"""Diagnostic 3: load the H200 checkpoint, predict vs target element-wise.

H1 (structural zeros) and H2 (RevIN cache leakage) were rejected by
`diagnose_loss_collapse.py`. Remaining hypothesis: the model genuinely
predicts targets to ~1.9e-4 per-element error in normalized space.

This script loads `runs/h200/checkpoints/last.ckpt`, runs a real batch,
and computes:
  - per-loss-component MSE (sanity check vs CSV claim of ~1e-8)
  - element-wise pred vs target stats (max abs error, %elements with
    |error| < 1e-3, etc.)
  - separately for: masked-recon, NTP, multi-horizon forecast
  - a "constant-zero baseline" comparison: what would loss be if pred=0?
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
from daity.data.parquet_store import ParquetStore
from daity.data.tokenizer_targets import build_forecast_targets, build_targets
from daity.training.losses import apply_random_patch_mask
from daity.training.pretrain import PretrainConfig, PretrainModule

torch.manual_seed(0)


# %%
CKPT = Path("runs/h200/checkpoints/last.ckpt")
PARQUET_ROOT = Path("data/parquet")

# Load checkpoint to get the hyperparameters used during training.
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")
hparams = ckpt["hyper_parameters"]
print(f"hparams['provenance']: {hparams.get('provenance', 'n/a')}")
print(f"hparams['scales']:     {hparams.get('scales')}")
print(f"hparams['d_model']:    {hparams.get('d_model')}")
print(f"hparams['max_steps']:  {hparams.get('max_steps')}")
print(f"hparams['mask_ratio']: {hparams.get('mask_ratio')}")
print(f"hparams['w_*']: masked={hparams.get('w_masked')} ntp={hparams.get('w_ntp')} "
      f"contrast={hparams.get('w_contrastive')} forecast={hparams.get('w_forecast')}")


# %%
# Reconstruct the config from hparams.  PretrainConfig stores everything we
# need to rebuild the module identically.
cfg_kwargs = {
    f: hparams[f] for f in PretrainConfig.__dataclass_fields__
    if f in hparams
}
# scales/window_bars come back as YAML lists; normalize to tuple/dict.
if isinstance(cfg_kwargs.get("scales"), list):
    cfg_kwargs["scales"] = tuple(cfg_kwargs["scales"])
cfg = PretrainConfig(**cfg_kwargs)
print(f"\nReconstructed PretrainConfig: {cfg}")

module = PretrainModule(cfg)
state = ckpt["state_dict"]
# Strip lightning's "_orig_mod." or similar prefixes if any.
state = {k: v for k, v in state.items()}
missing, unexpected = module.load_state_dict(state, strict=False)
print(f"Loaded state_dict — missing keys: {len(missing)}, unexpected: {len(unexpected)}")
if missing:
    print(f"  first missing: {missing[:5]}")
if unexpected:
    print(f"  first unexpected: {unexpected[:5]}")
module.eval()


# %%
# Build a real validation batch — production window_bars + production
# forecast_n_patches. Same path the H200 used for val_loss computation.
# window_bars isn't on PretrainConfig (lives on YAML/CLI only); use the
# production defaults from configs/train/pretrain.yaml.
WINDOW_BARS = {"5m": 256, "15m": 192, "60m": 168, "day": 64}
PATCH_LEN = cfg.patch_len
FORECAST_SCALE = cfg.forecast_scale
FORECAST_N_PATCHES = cfg.forecast_n_patches
FORECAST_N_BARS = FORECAST_N_PATCHES * PATCH_LEN

store = ParquetStore(PARQUET_ROOT)
on_disk = sorted(p.stem for p in store.list_partitions("day"))
# Use val seed (1) and val_end as the leakage cap, matching what the H200
# Trainer's val_loader saw.
val_end = datetime(2026, 4, 30, tzinfo=UTC)
ds = OHLCVPretrainDataset(
    store=store, symbols=on_disk, as_of=val_end,
    window_bars=WINDOW_BARS, seed=1,
    forecast_scale=FORECAST_SCALE, forecast_n_bars=FORECAST_N_BARS,
)
loader = DataLoader(ds, batch_size=8, num_workers=0)
batch = next(iter(loader))
print(f"\nBatch keys: {list(batch.keys())}")
for k, v in batch.items():
    print(f"  {k:<32s} shape={tuple(v.shape)}  dtype={v.dtype}")


# %%
# Run a forward pass with the trained model — replicate _compute_loss
# step by step but break out predictions so we can inspect them.

# 1. Pop the future tail.
future_bars = batch[FORECAST_FUTURE_KEY]
batch_no_future = {k: v for k, v in batch.items() if k != FORECAST_FUTURE_KEY}

# 2. Build recon targets (this also caches RevIN stats).
targets, volumes = build_targets(
    batch=batch_no_future, scales=cfg.scales,
    patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
    tokenizer=module.tokenizer,
)

# 3. Tokenize → mask → non-causal backbone → recon head.
tokens = module.tokenizer(batch_no_future)
torch.manual_seed(0)  # determinism on the random mask
masked_tokens, mask_full = apply_random_patch_mask(
    tokens, mask_ratio=cfg.mask_ratio, mask_token=module.mask_token,
)
with torch.no_grad():
    hidden_nc = module.backbone(masked_tokens, causal=False)
    recon_out_flat = module.recon_head(hidden_nc[:, 1:])
B, P, _ = recon_out_flat.shape
recon_out = recon_out_flat.view(B, P, cfg.num_channels, cfg.patch_len)
patch_mask = mask_full[:, 1:]

# 4. Causal backbone → NTP head.
with torch.no_grad():
    hidden_c = module.backbone(tokens, causal=True)
    ntp_out_flat = module.ntp_head(hidden_c[:, 1:])
ntp_out = ntp_out_flat.view(B, P, cfg.num_channels, cfg.patch_len)
pred_shifted = ntp_out[:, :-1]
target_shifted = targets[:, 1:]

# 5. Multi-horizon forecast.
with torch.no_grad():
    forecast_pred = module.forecast_head(hidden_nc[:, 0])
forecast_targets = build_forecast_targets(
    future_bars=future_bars, forecast_scale=cfg.forecast_scale,
    n_patches=cfg.forecast_n_patches, patch_len=cfg.patch_len,
    tokenizer=module.tokenizer,
)


# %%
# Pred vs target side-by-side for each loss.

def stats(pred: torch.Tensor, target: torch.Tensor, name: str, mask: torch.Tensor | None = None) -> None:
    if mask is not None:
        # mask is (B, P) bool. Broadcast over channels and patch_len.
        mask_4d = mask[..., None, None].expand_as(pred)
        diff = (pred - target)[mask_4d]
        tgt = target[mask_4d]
    else:
        diff = (pred - target).flatten()
        tgt = target.flatten()
    mse = (diff ** 2).mean().item()
    print(f"\n=== {name} ===")
    print(f"  shape pred={tuple(pred.shape)} target={tuple(target.shape)}")
    print(f"  target stats:  mean={tgt.mean():.4f} std={tgt.std():.4f}  "
          f"min={tgt.min():.4f}  max={tgt.max():.4f}")
    print(f"  pred stats:    mean={pred.mean():.4f} std={pred.std():.4f}  "
          f"min={pred.min():.4f}  max={pred.max():.4f}")
    print(f"  abs(pred-tgt): mean={diff.abs().mean():.4e}  "
          f"max={diff.abs().max():.4e}  "
          f"p99={torch.quantile(diff.abs(), 0.99):.4e}")
    print(f"  MSE={mse:.4e}")
    # Constant-zero baseline.
    baseline_mse = (tgt ** 2).mean().item()
    print(f"  baseline (pred=0) MSE: {baseline_mse:.4e}  "
          f"→ model is {baseline_mse/max(mse,1e-30):.0f}× better than zero")

stats(recon_out, targets, "MASKED RECON  (loss reads masked positions only)", mask=patch_mask)
stats(pred_shifted, target_shifted, "NTP")
stats(forecast_pred, forecast_targets, "MULTI-HORIZON FORECAST")


# %%
# A specific look: pick one batch element, one channel (close=3), one
# patch in the forecast head's output. Print pred vs target side by side.
print("\n\n=== FORECAST head: sample-0, channel=close, patch-by-patch ===")
print(f"{'patch':>5} {'idx':>4} {'target':>10} {'pred':>10} {'diff':>10}")
sample = 0
ch = 3
for p in range(cfg.forecast_n_patches):
    for i in range(cfg.patch_len):
        if i not in (0, cfg.patch_len // 2, cfg.patch_len - 1):
            continue
        t = forecast_targets[sample, p, ch, i].item()
        pr = forecast_pred[sample, p, ch, i].item()
        print(f"{p:>5} {i:>4} {t:>10.4f} {pr:>10.4f} {t-pr:>+10.4e}")


# %%
# Final: is the prediction *zero* in normalized space?
print("\n=== Is the model just outputting zero? ===")
print(f"  forecast_pred:    mean={forecast_pred.mean():.4e}  "
      f"std={forecast_pred.std():.4f}")
print(f"  forecast_target:  mean={forecast_targets.mean():.4e}  "
      f"std={forecast_targets.std():.4f}")
print(f"  recon_out:        mean={recon_out.mean():.4e}  std={recon_out.std():.4f}")
print(f"  recon_targets:    mean={targets.mean():.4e}  std={targets.std():.4f}")
print(f"  ntp_out_full:     mean={ntp_out.mean():.4e}  std={ntp_out.std():.4f}")
