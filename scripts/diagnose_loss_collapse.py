"""Diagnostic: why does Phase 2 SSL training drive losses to ~1e-8?

Three hypotheses (per the H200 run's val_loss curve hitting 4.95e-08 at step 28999):

  H1. Structural data issue — limit-locked / zero-volume / pre-padded
      bars produce near-constant target patches that MSE trivially
      solves. Fix: mask such bars in the loss.

  H2. RevIN cache leakage — the forecast target builder's `apply_cached`
      stats get overwritten by an intervening tokenizer call, so future
      bars get normalized using their OWN stats (not the input window's),
      producing trivial targets. Fix: snapshot stats, or change the
      contract.

  H3. Real but uninteresting — overlapping patches (stride=8, len=16)
      give NTP a 50% free lunch via copy. The forecast head shouldn't
      have this leakage but might be learning a similarly-trivial
      structure on smooth NSE 5m data. Fix: nothing — but verify the
      checkpoint produces useful FORECAST embeddings via a t-SNE.

This script tests H1 and H2 directly (no trained checkpoint required).
H3 verification will follow if H1 + H2 are ruled out.
"""

# %%
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from daity.data.datasets import FORECAST_FUTURE_KEY, OHLCVPretrainDataset
from daity.data.parquet_store import ParquetStore
from daity.data.tokenizer_targets import build_forecast_targets, build_targets
from daity.models.tokenizer import MultiResTokenizer

torch.manual_seed(0)


# %% [markdown]
# ### Step 1: build a batch from production Parquet


# %%
PARQUET_ROOT = Path("data/parquet")
SCALES = ("5m", "15m", "60m", "day")
WINDOW_BARS = {"5m": 256, "15m": 192, "60m": 168, "day": 64}
PATCH_LEN, PATCH_STRIDE = 16, 8
FORECAST_SCALE = "5m"
FORECAST_N_PATCHES = 6
FORECAST_N_BARS = FORECAST_N_PATCHES * PATCH_LEN

store = ParquetStore(PARQUET_ROOT)
on_disk = sorted(p.stem for p in store.list_partitions("day"))
print(f"Universe: {len(on_disk)} symbols on disk")

ds = OHLCVPretrainDataset(
    store=store, symbols=on_disk[:8],   # 8-symbol mini-universe for speed
    as_of=datetime(2025, 12, 31, tzinfo=UTC),
    window_bars=WINDOW_BARS, seed=42,
    forecast_scale=FORECAST_SCALE,
    forecast_n_bars=FORECAST_N_BARS,
)
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch = next(iter(loader))
print("Batch keys:", list(batch.keys()))
for k, v in batch.items():
    print(f"  {k:<32s} shape={tuple(v.shape)}  dtype={v.dtype}")


# %% [markdown]
# ### Step 2: H1 — does the RAW input data have suspicious structural zeros?
#
# Per-channel stats (open, high, low, close, volume) per scale, in RAW space.
# What we're looking for: any column whose stdev is near-zero across the
# whole batch, or where the median/min are zero (indicating padding / halts).


# %%
print("\n=== RAW input stats (B=4, per scale, per channel) ===")
print(f"{'scale':<6} {'channel':<8} {'mean':>12} {'std':>12} {'min':>12} {'max':>12} {'%zeros':>8}")
CHANNELS = ("open", "high", "low", "close", "volume")
for sc in SCALES:
    x = batch[sc].float()                          # (B, L, C)
    for c, name in enumerate(CHANNELS):
        col = x[..., c]
        pct_zero = (col == 0).float().mean().item() * 100
        print(f"{sc:<6} {name:<8} {col.mean().item():>12.4g} "
              f"{col.std().item():>12.4g} {col.min().item():>12.4g} "
              f"{col.max().item():>12.4g} {pct_zero:>7.2f}%")


# %% [markdown]
# Forecast tail (the future bars used as the multi-horizon target):


# %%
ff = batch[FORECAST_FUTURE_KEY].float()
print(f"\n=== FORECAST_FUTURE shape={tuple(ff.shape)} ===")
for c, name in enumerate(CHANNELS):
    col = ff[..., c]
    pct_zero = (col == 0).float().mean().item() * 100
    print(f"  {name:<8} mean={col.mean().item():>12.4g} std={col.std().item():>12.4g} "
          f"min={col.min().item():>12.4g} max={col.max().item():>12.4g} "
          f"%zeros={pct_zero:>5.2f}%")


# %% [markdown]
# ### Step 3: tokenize → build_targets → inspect normalized targets
#
# The recon target is the RevIN-normalized patches. After RevIN, each
# (batch, channel) is mean=0, std=1 over the window. The PATCH-wise
# stats should *not* be 0/1 (they're sub-windows). Look for:
#   - per-patch stdev near 0  → patch is near-constant → trivial MSE
#   - per-patch mean far from 0 → fine, RevIN normalizes the WINDOW
#   - target std globally ≈ 1 → expected (the RevIN baseline)


# %%
tokenizer = MultiResTokenizer(
    scales=SCALES, num_channels=5,
    d_model=64, patch_len=PATCH_LEN, patch_stride=PATCH_STRIDE,
)
batch_no_future = {k: v for k, v in batch.items() if k != FORECAST_FUTURE_KEY}
targets, volumes = build_targets(
    batch=batch_no_future, scales=SCALES,
    patch_len=PATCH_LEN, patch_stride=PATCH_STRIDE, tokenizer=tokenizer,
)
print(f"\n=== build_targets output ===")
print(f"  targets shape: {tuple(targets.shape)}  (B, total_patches, C, patch_len)")
print(f"  volumes shape: {tuple(volumes.shape)}")
print(f"  targets   stats: mean={targets.mean():.4f} std={targets.std():.4f} "
      f"min={targets.min():.4f} max={targets.max():.4f}")

# Per-patch stdev distribution: if many patches have std≈0, that's a smoking gun.
per_patch_std = targets.std(dim=(-1, -2))                           # (B, P)
zero_std_pct = (per_patch_std < 1e-3).float().mean().item() * 100
print(f"  per-patch std < 1e-3:  {zero_std_pct:.2f}% of patches")
print(f"  per-patch std percentiles: "
      f"p1={torch.quantile(per_patch_std, 0.01):.4f} "
      f"p50={torch.quantile(per_patch_std, 0.50):.4f} "
      f"p99={torch.quantile(per_patch_std, 0.99):.4f}")

# Per-patch mean — should be ROUGHLY zero (since RevIN zeros the window
# mean), but spread across the window can be wide.
per_patch_mean = targets.mean(dim=(-1, -2))                          # (B, P)
print(f"  per-patch mean: "
      f"p1={torch.quantile(per_patch_mean, 0.01):.4f} "
      f"p50={torch.quantile(per_patch_mean, 0.50):.4f} "
      f"p99={torch.quantile(per_patch_mean, 0.99):.4f}")


# %% [markdown]
# ### Step 4: H2 — does `apply_cached` actually use input-window stats?
#
# Snapshot the tokenizer's RevIN means/stds AFTER build_targets, then call
# `tokenizer(batch)` (which would re-cache), then build_forecast_targets,
# and check whether the cached stats at the time of forecast-target
# construction match what's expected.


# %%
# State 1: stats AFTER build_targets — these are the input-window stats.
revin = tokenizer.revins[FORECAST_SCALE]
stats_after_build_targets = {
    "mean": revin._mean.clone(),
    "std":  revin._std.clone(),
}

# State 2: tokenize the input. This does ANOTHER norm pass on the same
# input — should produce identical stats since input is unchanged.
_tokens = tokenizer(batch_no_future)
stats_after_tokenize = {
    "mean": revin._mean.clone(),
    "std":  revin._std.clone(),
}

# Verify identity.
mean_diff = (stats_after_build_targets["mean"] - stats_after_tokenize["mean"]).abs().max().item()
std_diff  = (stats_after_build_targets["std"]  - stats_after_tokenize["std"]).abs().max().item()
print(f"\n=== H2: RevIN stats after build_targets vs after tokenizer(batch) ===")
print(f"  max |Δmean| = {mean_diff:.2e}   max |Δstd| = {std_diff:.2e}")
print(f"  → {'IDENTICAL ✓' if (mean_diff < 1e-6 and std_diff < 1e-6) else 'DRIFT ✗'}")

# State 3: now build forecast targets using the live cache.
forecast_targets = build_forecast_targets(
    future_bars=batch[FORECAST_FUTURE_KEY],
    forecast_scale=FORECAST_SCALE,
    n_patches=FORECAST_N_PATCHES, patch_len=PATCH_LEN,
    tokenizer=tokenizer,
)
print(f"\n=== build_forecast_targets output ===")
print(f"  shape: {tuple(forecast_targets.shape)}  (B, H, C, patch_len)")
print(f"  stats: mean={forecast_targets.mean():.4f} std={forecast_targets.std():.4f} "
      f"min={forecast_targets.min():.4f} max={forecast_targets.max():.4f}")

# Per-patch stdev — same diagnostic as before.
ftgt_std = forecast_targets.std(dim=(-1, -2))
ftgt_zero = (ftgt_std < 1e-3).float().mean().item() * 100
print(f"  per-patch std < 1e-3:  {ftgt_zero:.2f}% of forecast patches")
print(f"  per-patch std percentiles: "
      f"p1={torch.quantile(ftgt_std, 0.01):.4f} "
      f"p50={torch.quantile(ftgt_std, 0.50):.4f} "
      f"p99={torch.quantile(ftgt_std, 0.99):.4f}")


# %% [markdown]
# ### Step 5: H1 deep dive — what fraction of the volume column is zero?
#
# NSE intraday data: every 5m bar that closed a session has volume > 0.
# But bars at session boundaries (pre-open auction, post-close) sometimes
# have volume=0. Days with halts have whole sequences of zero-volume bars.
# If the RevIN normalizes a window where >50% of bars have volume=0, the
# normalized volume column has structurally weird stats.


# %%
print("\n=== H1: volume-column zero-fraction per (sample, scale) ===")
for sc in SCALES:
    vol = batch[sc][..., 4].float()                                  # (B, L)
    per_sample_zeros = (vol == 0).float().mean(dim=-1)               # (B,)
    print(f"  {sc:<5} window_bars={WINDOW_BARS[sc]:>4}  "
          f"per-sample %zero-volume: "
          f"min={per_sample_zeros.min().item():.2%} "
          f"mean={per_sample_zeros.mean().item():.2%} "
          f"max={per_sample_zeros.max().item():.2%}")


# %% [markdown]
# ### Verdict
#
# Print a one-line summary of which hypothesis the data supports.


# %%
verdict_lines = []
if zero_std_pct > 5:
    verdict_lines.append(
        f"H1 LIKELY: {zero_std_pct:.1f}% of input patches have std < 1e-3 (constant patches)"
    )
if ftgt_zero > 5:
    verdict_lines.append(
        f"H1 LIKELY (forecast): {ftgt_zero:.1f}% of forecast patches have std < 1e-3"
    )
if mean_diff > 1e-6 or std_diff > 1e-6:
    verdict_lines.append(
        f"H2 LIKELY: tokenizer's RevIN cache drifted between calls (Δmean={mean_diff:.2e})"
    )
if not verdict_lines:
    verdict_lines.append(
        "H1 + H2 not supported by this batch's stats. H3 (real overfit on overlapping "
        "patches + smooth data) is the remaining candidate; needs trained-model check."
    )

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
for line in verdict_lines:
    print(f"  {line}")
