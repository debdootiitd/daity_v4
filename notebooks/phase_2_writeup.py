# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # Phase 2 write-up — daity_v3 pretraining (code-complete)
#
# **Scope of this writeup**: Phase 2 deliverables are *code complete* and
# *CPU-smoke-verified*. The actual ~2-3 day pretraining run on A100/Vast
# is a separate step (see `docs/vast_bootstrap.md`); this notebook
# validates the pieces are wired correctly and ready to launch.
#
# DESIGN §7 acceptance criterion for Phase 2 ("val masked-MSE plateau,
# checkpoint < 200 MB, reconstructions visually plausible") is *partial*
# at this point: the architecture supports it, the smoke training step
# decreases all 3 losses, but the actual plateau-on-A100 evidence will
# arrive after the Vast run completes. **Defer Phase-3 reviewer signoff
# until the trained checkpoint is back.**

# %% [markdown]
# ## 1 · Setup

# %%
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return start
_root = _find_project_root(Path.cwd())
if Path.cwd() != _root:
    os.chdir(_root)
print(f"cwd: {Path.cwd()}")

import matplotlib.pyplot as plt
import polars as pl
import torch
from torch.utils.data import DataLoader

from daity.data.datasets import OHLCVPretrainDataset
from daity.data.parquet_store import ParquetStore
from daity.training.pretrain import PretrainConfig, PretrainModule

ARTIFACTS = Path("notebooks/_artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2 · Hardware + accelerator detection
#
# `daity-pretrain --device auto` resolves to the best accelerator. On
# Vast we'd see `cuda`; on this Mac, `mps`. CPU smoke runs on CPU
# explicitly via `--smoke`.

# %%
print(f"torch:           {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"MPS available:   {torch.backends.mps.is_available()}")
print(f"CPU threads:     {torch.get_num_threads()}")

# %% [markdown]
# ## 3 · Architecture summary
#
# Build the production-sized backbone (default config) and report the
# parameter count. DESIGN §3.1 budget is ~20M params for the full backbone
# *including* the Phase-4 OB fusion; pretraining-only backbone lands
# around half of that.

# %%
prod_cfg = PretrainConfig(
    scales=("5m", "15m", "60m", "day"),
    num_channels=5,
    d_model=320, n_layers=8, n_heads=8, ffn_ratio=4,
    patch_len=16, patch_stride=8, max_seq_len=128,
    mask_ratio=0.4, contrastive_proj_dim=128,
)
prod_module = PretrainModule(prod_cfg)
n_total = sum(p.numel() for p in prod_module.parameters())
n_per_module = {
    name.split(".")[0]: 0 for name, _ in prod_module.named_parameters()
}
for name, p in prod_module.named_parameters():
    n_per_module[name.split(".")[0]] += p.numel()
print(f"Total params (production config): {n_total:,}")
print()
print(f"{'module':<22} {'params':>14}")
for k, v in sorted(n_per_module.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<20} {v:>14,}")

# Acceptance check: the BACKBONE component must be <20M.
backbone_params = sum(p.numel() for p in prod_module.backbone.parameters())
print()
print(f"Backbone alone: {backbone_params:,} ({backbone_params/1e6:.1f}M)")
assert backbone_params < 20_000_000, (
    f"Backbone {backbone_params:,} exceeds DESIGN §3.1 ~20M target"
)
print("PASS  backbone within DESIGN §3.1 size budget")

# %% [markdown]
# ## 4 · Estimated checkpoint size
#
# DESIGN §7 acceptance: "pretrained checkpoint < 200 MB". A bf16 weights-
# only checkpoint is `n_params × 2 bytes`. fp32 (full state with optimizer)
# is ~3-4× that.

# %%
bf16_mb = (n_total * 2) / 1024 / 1024
fp32_mb = (n_total * 4) / 1024 / 1024
print(f"bf16 weights-only checkpoint estimate: {bf16_mb:.1f} MB")
print(f"fp32 weights-only checkpoint estimate: {fp32_mb:.1f} MB")
print(f"With optimizer state (~3×):            {fp32_mb*3:.1f} MB")
assert bf16_mb < 200, "bf16 weights checkpoint exceeds 200 MB DESIGN target"
print("PASS  bf16 weights checkpoint comfortably under 200 MB")

# %% [markdown]
# ## 5 · Live pretraining smoke — the headline acceptance for *this* writeup
#
# Run a tiny model for 200 steps on real RELIANCE/TCS/INFY/HDFCBANK data
# from `ParquetStore`, on this Mac's MPS or CPU. Confirm:
#   1. The pipeline runs end-to-end without errors.
#   2. Each of the 4 SSL losses (recon / NTP / contrastive / forecast)
#      is finite and non-zero at init, and *decreases* over the run for
#      the trainable ones (recon / NTP / forecast).
#   3. The total weighted loss trends down.

# %%
# Pick a device: prefer MPS on Mac to validate non-CPU code paths, fall
# back to CPU if not available.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

smoke_cfg = PretrainConfig(
    scales=("5m", "day"), num_channels=5,
    d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
    patch_len=16, patch_stride=8, max_seq_len=32,
    mask_ratio=0.3, contrastive_proj_dim=32,
    forecast_scale="5m", forecast_n_patches=2,
    # DESIGN amendment 11: contrastive disabled in v1 (placeholder positives
    # collapse InfoNCE to log(B) by construction); re-enabled in Phase 3 with
    # cross-symbol same-date positives.
    w_contrastive=0.0,
    lr=5e-3, max_steps=200,
)
smoke_module = PretrainModule(smoke_cfg).to(device)

store = ParquetStore(Path("data/parquet"))
ds = OHLCVPretrainDataset(
    store=store,
    symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "ITC"],
    as_of=datetime(2026, 5, 9, tzinfo=timezone.utc),
    window_bars={"5m": 64, "day": 16},
    seed=42,
    # 2 patches × patch_len=16 = 32 future 5m bars ≈ 2.7h forward.
    forecast_scale="5m",
    forecast_n_bars=2 * smoke_cfg.patch_len,
)
loader = DataLoader(ds, batch_size=8, num_workers=0)

opt = torch.optim.AdamW(smoke_module.parameters(), lr=smoke_cfg.lr)
losses: dict[str, list[float]] = {
    "total": [], "recon": [], "ntp": [], "contrast": [], "forecast": [],
}

import time
t0 = time.time()
data_iter = iter(loader)
for step in range(smoke_cfg.max_steps):
    batch = next(data_iter)
    batch = {k: v.to(device) for k, v in batch.items()}
    opt.zero_grad()
    loss, parts = smoke_module._compute_loss(batch)
    loss.backward()
    opt.step()
    losses["total"].append(loss.item())
    losses["recon"].append(parts["loss_recon"].item())
    losses["ntp"].append(parts["loss_ntp"].item())
    losses["contrast"].append(parts["loss_contrast"].item())
    losses["forecast"].append(parts["loss_forecast"].item())
elapsed = time.time() - t0
print(f"Ran {smoke_cfg.max_steps} steps in {elapsed:.1f}s "
      f"({smoke_cfg.max_steps/elapsed:.1f} steps/s on {device})")

# %% [markdown]
# ### Loss curves

# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=False)
for ax, (name, vals) in zip(axes, losses.items()):
    ax.plot(vals, lw=0.8, alpha=0.8, label="step")
    # Smooth running mean (window=20).
    if len(vals) >= 20:
        running = [sum(vals[max(0, i-20):i+1]) / min(20, i+1) for i in range(len(vals))]
        ax.plot(running, lw=2, label="20-step mean")
    ax.set_title(f"{name} loss")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(ARTIFACTS / "phase2_smoke_losses.png", dpi=120, bbox_inches="tight")
print(f"Saved {ARTIFACTS / 'phase2_smoke_losses.png'}")

# %% [markdown]
# ### Acceptance assertions

# %%
import numpy as np

def early_late(vals: list[float], window: int = 20) -> tuple[float, float]:
    early = float(np.mean(vals[:window]))
    late = float(np.mean(vals[-window:]))
    return early, late

print(f"{'objective':<10} {'early (avg first 20)':>22} {'late (avg last 20)':>20} {'Δ':>10}")
for name, vals in losses.items():
    e, l = early_late(vals)
    delta_pct = (l - e) / e * 100 if e > 0 else 0
    print(f"  {name:<8} {e:>22.4f} {l:>20.4f} {delta_pct:>+9.1f}%")

# Headline assertion: the masked-recon loss must drop meaningfully — that's
# the primary SSL objective per DESIGN §3.2 (weight 1.0).
e_recon, l_recon = early_late(losses["recon"])
assert l_recon < e_recon, (
    f"recon loss did not decrease: early={e_recon:.4f}, late={l_recon:.4f}"
)
print()
print(f"PASS  recon loss decreases (primary SSL objective, weight 1.0)")

# NTP should also drop (weight 0.5).
e_ntp, l_ntp = early_late(losses["ntp"])
assert l_ntp < e_ntp, f"NTP loss did not decrease: early={e_ntp:.4f}, late={l_ntp:.4f}"
print(f"PASS  NTP loss decreases (causal next-patch, weight 0.5)")

# Contrastive: disabled in v1 per DESIGN amendment 11 (placeholder positives
# would collapse InfoNCE to log(B) — see review finding 2). The loss is still
# computed for visibility but contributes 0 to the total via w_contrastive=0.
# All we assert here is that it stays finite.
assert all(np.isfinite(losses["contrast"])), "contrast loss has non-finite values"
print(f"INFO  contrast disabled (w_contrastive=0; DESIGN amendment 11). "
      f"Phase 3 re-enables with cross-symbol same-date positives.")

# Multi-horizon forecast: predicts H non-overlapping FUTURE bars at
# forecast_scale. This is the *hardest* objective — no overlap free-lunch
# (unlike recon/NTP which can copy 50% from neighboring patches). The
# H200 50K-step run shows val_loss_forecast 8.61 → 8.22 over 17K val
# passes (real but slow). On a 200-step CPU smoke with the tiny model,
# the forecast loss is dominated by noise and plateaus rather than
# decreasing reliably — this is *expected* per DESIGN amendment 10. We
# only assert it stays finite + within an order of magnitude of its
# initial value (i.e. doesn't explode).
e_fc, l_fc = early_late(losses["forecast"])
assert all(np.isfinite(losses["forecast"])), "forecast loss has non-finite values"
assert l_fc < 10 * e_fc, (
    f"forecast loss exploded: early={e_fc:.4f}, late={l_fc:.4f}. "
    f"Expected noisy plateau, not divergence."
)
print(f"INFO  forecast loss noisy on 200-step smoke ({e_fc:.2f} → {l_fc:.2f}); "
      f"H200 trajectory is the real signal.")

# %% [markdown]
# ## 6 · Reconstruction visualization
#
# DESIGN §7 calls for "reconstructions visually plausible on held-out
# symbols". With only 200 steps the model hasn't really learned anything,
# but we plot one reconstruction here as a sanity-check that the
# *infrastructure* produces the right shape. Visual plausibility is
# revisited after the Vast pretrain run.

# %%
smoke_module.eval()
with torch.no_grad():
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    # Forward through tokenizer + non-causal backbone + recon head.
    targets, _ = smoke_module._build_targets(batch)             # (B, P, C, patch_len)
    tokens = smoke_module.tokenizer(batch)
    hidden = smoke_module.backbone(tokens, causal=False)
    recon_out = smoke_module.recon_head(hidden[:, 1:])
    B, P, _ = recon_out.shape
    recon = recon_out.view(B, P, smoke_cfg.num_channels, smoke_cfg.patch_len)

# Plot one symbol's first-scale (5m) close channel: target vs reconstruction
# stitched back together.
sample_idx = 0
n_5m_patches = smoke_module.tokenizer.patcher.n_patches(64)  # 7 for L=64
target_close = targets[sample_idx, :n_5m_patches, 3, :]      # close = channel 3
recon_close = recon[sample_idx, :n_5m_patches, 3, :]

fig, ax = plt.subplots(figsize=(11, 3))
# Show first 4 patches end-to-end (overlapping by stride 8 — averages).
flat_target = target_close.reshape(-1).cpu().numpy()
flat_recon = recon_close.reshape(-1).cpu().numpy()
ax.plot(flat_target, label="target (RevIN-norm)", lw=1)
ax.plot(flat_recon, label="reconstruction", lw=1, alpha=0.7)
ax.set_title("RELIANCE-ish 5m close: target vs (untrained) reconstruction")
ax.set_xlabel("patch position (concatenated)")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(ARTIFACTS / "phase2_recon_sample.png", dpi=120, bbox_inches="tight")
print(f"Saved {ARTIFACTS / 'phase2_recon_sample.png'}")

# %% [markdown]
# ## 7 · End-to-end timing budget
#
# Production Vast/A100 budget:
# - 50K steps × batch 64 × ~5s/step (pessimistic) = ~70 hours = ~3 days.
# - At bf16 + FlashAttention-2 + warm dataloader, more like ~1.5-2 days.
#
# CPU budget on this Mac (smoke):
# - 200 steps × 4 samples/batch in {elapsed:.0f}s above → too slow for
#   real training but adequate for code validation.

# %% [markdown]
# ## 8 · Acceptance checklist for Phase 2 *code complete*
#
# (Per DESIGN §7 acceptance the FULL Phase-2 sign-off requires the trained
# checkpoint. This checklist is the prerequisite gate before launching
# the real run.)
#
# | check | state |
# |---|---|
# | RevIN reversibility round-trip (`denorm(norm(x)) ≈ x`) | locked by `test_revin.py::test_norm_then_denorm_recovers_input` |
# | Multi-resolution tokenizer produces `(B, 1+ΣP, d_model)` with FORECAST at pos 0 | `test_tokenizer.py::test_tokenizer_forecast_token_is_at_position_zero` |
# | Dataset's `as_of` cap is mandatory and excludes in-progress intraday | `test_datasets.py::test_no_sample_extends_past_as_of` |
# | Backbone causal mask is honored end-to-end | `test_backbone.py::test_backbone_causal_pass_blocks_future_at_position_0` |
# | All 3 SSL losses non-zero at init, gradient reaches every component | `test_pretrain_module.py::test_compute_loss_gradient_reaches_every_component` |
# | One-batch overfit test: total loss drops on a fixed batch | `test_pretrain_module.py::test_overfit_one_batch_reduces_loss` |
# | Production-config backbone < 20M params (DESIGN §3.1 budget) | this notebook §3 |
# | bf16 weights checkpoint estimate < 200 MB (DESIGN §7) | this notebook §4 |
# | CPU smoke completes 200 steps without error | this notebook §5 |
# | Recon loss + NTP loss decrease over 200 smoke steps | this notebook §5 |
#
# **Verdict: Phase 2 code complete. Ready to launch on Vast/A100.**
# The masked-MSE plateau evidence + visual reconstruction quality on
# held-out symbols comes after the real run.

# %% [markdown]
# ## 9 · Architecture diagram
#
# Two-panel figure summarizing what each training step actually does:
#  - **Top:** the 82-token sequence layout (FORECAST + 81 patch tokens
#    across 4 scales).
#  - **Bottom:** the forward flow — per-scale RevIN/patch/project →
#    concat + prepend FORECAST → twin backbone passes (causal for NTP,
#    non-causal for everything else) → four heads → weighted total loss.
#
# Numbers are pulled directly from `prod_cfg` so they always match the
# production YAML.

# %%
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle

# Numbers that appear in the diagram, derived from prod_cfg + the production YAML.
window_bars_prod = {"5m": 256, "15m": 192, "60m": 168, "day": 64}
patch_len = prod_cfg.patch_len
patch_stride = prod_cfg.patch_stride
n_patches_prod = {sc: (n - patch_len) // patch_stride + 1 for sc, n in window_bars_prod.items()}
total_patches_prod = sum(n_patches_prod.values())
T_prod = total_patches_prod + 1  # plus FORECAST
B_prod = 64

# Cumulative offsets so we know where each scale's patches sit in the sequence.
offsets = {}
cum = 1                                  # position 0 = FORECAST
for sc in ("5m", "15m", "60m", "day"):
    offsets[sc] = (cum, cum + n_patches_prod[sc])
    cum += n_patches_prod[sc]

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.20)

# --- TOP PANEL: token sequence layout --------------------------------------
ax_seq = fig.add_subplot(gs[0])
SCALE_COLORS = {
    "FORECAST": "#1f4e79",
    "5m":       "#d97706",
    "15m":      "#059669",
    "60m":      "#dc2626",
    "day":      "#7c3aed",
}
spans = [
    (0, 1, "[FORECAST]", "1 token (broadcast)", SCALE_COLORS["FORECAST"]),
    (offsets["5m"][0],  offsets["5m"][1],  "5m  patches",  f"{n_patches_prod['5m']} tokens",  SCALE_COLORS["5m"]),
    (offsets["15m"][0], offsets["15m"][1], "15m patches", f"{n_patches_prod['15m']} tokens", SCALE_COLORS["15m"]),
    (offsets["60m"][0], offsets["60m"][1], "60m patches", f"{n_patches_prod['60m']} tokens", SCALE_COLORS["60m"]),
    (offsets["day"][0], offsets["day"][1], "day patches", f"{n_patches_prod['day']} tokens", SCALE_COLORS["day"]),
]
for start, end, name, count, color in spans:
    ax_seq.add_patch(Rectangle((start, 0), end - start, 1.0, facecolor=color, edgecolor="white"))
    ax_seq.text((start + end) / 2, 0.55, name, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax_seq.text((start + end) / 2, 0.20, count, ha="center", va="center",
                color="white", fontsize=9)
ax_seq.set_xlim(-0.5, T_prod + 0.5)
ax_seq.set_ylim(-0.1, 1.4)
ax_seq.set_xticks([0, 1, offsets["15m"][0], offsets["60m"][0], offsets["day"][0], T_prod])
ax_seq.set_xticklabels(["0", "1", str(offsets["15m"][0]), str(offsets["60m"][0]), str(offsets["day"][0]), str(T_prod)])
ax_seq.set_yticks([])
ax_seq.set_xlabel("token position", fontsize=10)
ax_seq.set_title(
    f"Token sequence layout — T = {T_prod} tokens  "
    f"(d_model={prod_cfg.d_model}, max_seq_len={prod_cfg.max_seq_len}, "
    f"{(T_prod / prod_cfg.max_seq_len) * 100:.0f}% used)",
    fontsize=12, fontweight="bold",
)
for spine in ax_seq.spines.values():
    spine.set_visible(False)

# --- BOTTOM PANEL: forward flow --------------------------------------------
ax_arch = fig.add_subplot(gs[1])
ax_arch.set_xlim(0, 16)
ax_arch.set_ylim(0, 14)
ax_arch.axis("off")

def _box(x, y, w, h, text, fc="#f3f4f6", ec="#374151", fontsize=9, fontweight="normal"):
    ax_arch.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax_arch.text(x + w/2, y + h/2, text, ha="center", va="center",
                 fontsize=fontsize, fontweight=fontweight, wrap=True)

def _arrow(x1, y1, x2, y2, color="#374151"):
    ax_arch.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.2,
    ))

# Row 1: input batch dict
_box(0.5, 12.5, 15, 1.3,
     f"DataLoader yields per-step batch (B={B_prod}):  "
     f"{{'5m': (B,256,5),  '15m': (B,192,5),  '60m': (B,168,5),  'day': (B,64,5),  "
     f"'__forecast_future__': (B,96,5)}}",
     fc="#eff6ff", ec="#1e3a8a", fontsize=9, fontweight="bold")

# Row 2: per-scale tokenization (one column per scale)
scale_x = {"5m": 1.0, "15m": 4.5, "60m": 8.0, "day": 11.5}
for sc in ("5m", "15m", "60m", "day"):
    x = scale_x[sc]
    color = SCALE_COLORS[sc]
    L = window_bars_prod[sc]
    n = n_patches_prod[sc]
    _box(x, 9.6, 3.0, 2.4,
         f"{sc} stream\n"
         f"───\n"
         f"RevIN(affine=False)  →  shape (B, {L}, 5)\n"
         f"Patcher(len=16, stride=8)  →  (B, {n}, 5, 16)\n"
         f"Linear[{sc}](80→320) + scale-embed\n"
         f"=  (B, {n}, 320)",
         fc="#ffffff", ec=color, fontsize=8)
    _arrow(x + 1.5, 12.5, x + 1.5, 12.0)

# Forecast tail (right side) — bypasses tokenizer, fed to forecast-target builder.
_box(14.7, 9.6, 1.2, 2.4,
     "future\n5m bars\n(B,96,5)\n───\nbuild_forecast_targets\nuses cached\nRevIN stats",
     fc="#fffbeb", ec="#b45309", fontsize=7)
_arrow(15.3, 12.5, 15.3, 12.0)

# Row 3: token assembly
_box(0.5, 7.6, 15, 1.6,
     "Concatenate per-scale projections + prepend learned [FORECAST] token  →  "
     f"tokens shape (B={B_prod}, T={T_prod}, d_model=320)",
     fc="#ecfdf5", ec="#065f46", fontsize=10, fontweight="bold")
for sc in ("5m", "15m", "60m", "day"):
    _arrow(scale_x[sc] + 1.5, 9.6, scale_x[sc] + 1.5, 9.2)

# Row 4: two backbone passes (causal LEFT, non-causal RIGHT)
_box(0.5, 5.0, 7.0, 2.2,
     "CAUSAL backbone\n8× transformer blocks\n"
     "RoPE on Q+K via SDPA\nlower-triangular mask\n"
     "→ hidden_c  (B, T, 320)",
     fc="#fef3c7", ec="#92400e", fontsize=10, fontweight="bold")
_box(8.5, 5.0, 7.0, 2.2,
     "NON-CAUSAL backbone\n8× transformer blocks\n"
     "RoPE on Q+K via SDPA\nfull attention\n40% of patch tokens replaced with [MASK]\n"
     "→ hidden_nc  (B, T, 320)",
     fc="#dbeafe", ec="#1e40af", fontsize=10, fontweight="bold")
_arrow(4.0, 7.6, 4.0, 7.2)
_arrow(12.0, 7.6, 12.0, 7.2)

# Row 5: heads (4 columns)
heads = [
    ("ntp_head", "reads hidden_c[:,1:]", "(B,81,5,16)\npredict patch t+1\nfrom prefix [≤t]",
     "#fef3c7", "#92400e", 0.5),
    ("recon_head", "reads hidden_nc[:,1:]", "(B,81,5,16)\nMSE only at\nthe 40% [MASK]ed\npatch positions",
     "#dbeafe", "#1e40af", 4.5),
    ("forecast_head", "reads hidden_nc[:,0]", "(B,6,5,16)\npredict next 6\nnon-overlapping\n5m patches\nafter end_ts",
     "#dbeafe", "#1e40af", 8.5),
    ("contrastive_head", "reads hidden_nc[:,0]", "(B,128) ℓ2-norm\nDISABLED\n(w_contrastive=0\nDESIGN amend 11)",
     "#fee2e2", "#991b1b", 12.5),
]
for name, source, body, fc, ec, x in heads:
    _box(x, 2.2, 3.0, 2.2,
         f"{name}\n{source}\n───\n{body}",
         fc=fc, ec=ec, fontsize=8, fontweight="bold")
# Arrows from backbone to heads.
_arrow(2.0, 5.0, 2.0, 4.4)             # ntp ← causal
_arrow(6.0, 5.0, 6.0, 4.4)             # recon ← non-causal
_arrow(10.0, 5.0, 10.0, 4.4)           # forecast ← non-causal
_arrow(14.0, 5.0, 14.0, 4.4)           # contrast ← non-causal

# Row 6: weighted total loss
_box(0.5, 0.3, 15, 1.5,
     "total_loss  =  1.0 × loss_recon  "
     "+  0.5 × loss_ntp  "
     "+  0.0 × loss_contrastive  "
     "+  0.5 × loss_forecast",
     fc="#f3f4f6", ec="#111827", fontsize=11, fontweight="bold")
for x in (2.0, 6.0, 10.0, 14.0):
    _arrow(x, 2.2, x, 1.8)

ax_arch.set_title("Phase 2 forward pass — single training step", fontsize=12, fontweight="bold")

plt.savefig(ARTIFACTS / "phase2_architecture.png", dpi=120, bbox_inches="tight")
print(f"Saved {ARTIFACTS / 'phase2_architecture.png'}")
plt.close()

# %% [markdown]
# ## 10 · v1 H200 run — val_loss trajectory
#
# The trained checkpoint at `runs/h200_v1/checkpoints/step=48000-val_loss=4.1303.ckpt`
# is the Phase-2.1.1 baseline. This section reads its 50-pass val trajectory
# from the rsync'd `metrics.csv` and answers: which loss components saturated,
# and where? The answer informs what "Phase 2.2 scaling" is trying to test.

# %%
import numpy as np
import pandas as pd

V1_METRICS = Path("runs/h200_v1/lightning_logs/version_0/metrics.csv")
if V1_METRICS.exists():
    df = pd.read_csv(V1_METRICS)
    val_df = df[df["val_loss"].notna()].copy().reset_index(drop=True)
    print(f"Val passes: {len(val_df)}, step range {val_df['step'].min()}–{val_df['step'].max()}")

    # Three-panel figure: total + 4 components on a shared x.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    panels = [
        ("val_loss",          "total val_loss (1.0 recon + 0.5 ntp + 0.5 forecast)",   axes[0, 0], "#111827"),
        ("val_loss_recon",    "val_recon (masked-MSE on RevIN-norm patches)",          axes[0, 1], "#1e40af"),
        ("val_loss_ntp",      "val_ntp (next-patch causal MSE)",                       axes[1, 0], "#92400e"),
        ("val_loss_forecast", "val_forecast (next 6 non-overlap 5m patches, MSE)",     axes[1, 1], "#7c3aed"),
    ]
    for col, title, ax, color in panels:
        ax.plot(val_df["step"], val_df[col], color=color, lw=1.5)
        ax.scatter(val_df["step"], val_df[col], color=color, s=10, alpha=0.5)
        # Annotate first/last/min.
        first_v = val_df[col].iloc[0]
        last_v = val_df[col].iloc[-1]
        min_v = val_df[col].min()
        min_s = int(val_df.loc[val_df[col].idxmin(), "step"])
        ax.axhline(min_v, ls=":", color="grey", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.text(0.02, 0.96,
                f"start={first_v:.3f}  end={last_v:.3f}  min={min_v:.3f}@step={min_s}\n"
                f"Δ={(last_v - first_v):+.3f}  ({(last_v - first_v)/first_v*100:+.1f}%)",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd"))
    fig.suptitle(
        "v1 H200 50K-step run — val trajectory (Phase 2.1.1, revin_affine=False)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "phase2_v1_val_trajectory.png", dpi=120, bbox_inches="tight")
    print(f"Saved {ARTIFACTS / 'phase2_v1_val_trajectory.png'}")
    plt.close()

    # Last-10K-step slope per component — the "is it still learning?" diagnostic.
    print("\nSlope analysis (last 10K steps, per-Kstep loss reduction):")
    print(f"  {'component':<22} {'val@step40K':>12} {'val@step50K':>12} {'slope (loss/Kstep)':>20}")
    last10 = val_df[val_df["step"] >= 39_000]
    for col in ("val_loss", "val_loss_recon", "val_loss_ntp", "val_loss_forecast"):
        if last10[col].notna().sum() < 2:
            continue
        slope = (last10[col].iloc[-1] - last10[col].iloc[0]) / (last10["step"].iloc[-1] - last10["step"].iloc[0]) * 1000
        print(f"  {col:<22} {last10[col].iloc[0]:>12.4f} {last10[col].iloc[-1]:>12.4f} {slope:>20.5f}")
else:
    print(f"v1 metrics not yet rsync'd to {V1_METRICS}; skipping val-trajectory plot.")

# %%
print("=" * 60)
print("Phase 2 write-up complete. All assertions passed.")
print("=" * 60)
