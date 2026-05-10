"""Compare v1 / v2 / v3 scaling-law experiment.

Reads `runs/h200_v{1,2,3}/lightning_logs/version_0/metrics.csv` (rsync'd
from the H200) and produces:

  1. **Trajectory overlay** — val_forecast vs step for all three runs on
     one axis. Shows whether bigger models train faster, deeper, or just
     identically.
  2. **Saturation table** — final val_forecast + slope-of-last-10K-steps
     per run. Quantifies "did it saturate or is it still falling?"
  3. **Scaling-law fit** — log-log plot of `min(val_forecast)` vs
     `total_params`. Fits `loss = A · params^(-α)` and reports the
     exponent α (canonical Time-FM α ≈ 0.05-0.10 per Chronos / MOIRAI).

The plot saves to `notebooks/_artifacts/phase2_scaling_comparison.png`
so it lands alongside the writeup notebook's other figures.

Run from project root:  uv run python scripts/compare_scaling_runs.py
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Each entry: (label, run_dir, total_params, color).
# total_params is read from the run banner if available; for now we hard-code
# the numbers reported by the CLI (these match the model_summary table that
# `daity-pretrain` prints at startup).
RUNS = [
    ("v1 (11.3M params)",  Path("runs/h200_v1"),    11_276_288, "#2563eb"),
    ("v2 (36.0M params)",  Path("runs/h200_v2"),    35_979_648, "#059669"),
    ("v3 (127M params)",   Path("runs/h200_v3"),   127_000_000, "#dc2626"),  # approximate; update from actual banner
]

ARTIFACTS = Path("notebooks/_artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _load_val(run_dir: Path) -> pd.DataFrame | None:
    """Load val rows from the run's metrics.csv (or None if not yet rsync'd)."""
    candidates = list(run_dir.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    val = df[df["val_loss"].notna()].copy().reset_index(drop=True)
    return val


# %%
# Pull what's available locally; missing runs are just skipped (the script
# is idempotent — re-run after each rsync).
loaded: list[tuple[str, pd.DataFrame, int, str]] = []
for label, run_dir, n_params, color in RUNS:
    val = _load_val(run_dir)
    if val is None or val.empty:
        print(f"[skip] {label} — no metrics yet at {run_dir}")
        continue
    loaded.append((label, val, n_params, color))
    last_step = int(val["step"].iloc[-1])
    last_loss = float(val["val_loss_forecast"].iloc[-1])
    min_loss = float(val["val_loss_forecast"].min())
    print(f"[ok]   {label}  steps=0..{last_step}  "
          f"final_forecast={last_loss:.3f}  min_forecast={min_loss:.3f}")

if not loaded:
    print("Nothing to plot yet. Re-run after the first rsync.")
    sys.exit(0)


# %%
# Two-panel figure: trajectory overlay (left) + scaling-law fit (right).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax_traj, ax_law = axes

# ---- Left: val_forecast trajectory overlay -------------------------------
for label, val, n_params, color in loaded:
    ax_traj.plot(val["step"], val["val_loss_forecast"], color=color, lw=1.6, label=label)
    ax_traj.scatter(val["step"], val["val_loss_forecast"], color=color, s=10, alpha=0.4)
ax_traj.set_xlabel("step")
ax_traj.set_ylabel("val_loss_forecast (MSE on RevIN-norm future patches)")
ax_traj.set_title("Multi-horizon forecast loss — scaling experiment", fontsize=11, fontweight="bold")
ax_traj.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax_traj.grid(alpha=0.3)

# ---- Right: scaling law fit ----------------------------------------------
# Use the MIN val_loss_forecast achieved by each run (post-warmup) as the
# point in the (n_params, loss) plane.  Then fit log(loss) = log(A) − α·log(params).
points = []
for label, val, n_params, color in loaded:
    # Skip the first few val passes to avoid the warmup-noisy region.
    val_post_warmup = val[val["step"] >= 5_000]
    if val_post_warmup.empty:
        val_post_warmup = val
    min_fc = val_post_warmup["val_loss_forecast"].min()
    points.append((n_params, min_fc, label, color))
    ax_law.scatter(n_params, min_fc, color=color, s=120, edgecolor="white",
                   linewidth=1.5, zorder=3, label=label)
    ax_law.text(n_params * 1.05, min_fc, f"{min_fc:.2f}", fontsize=9,
                va="center", color=color, fontweight="bold")

ax_law.set_xscale("log")
ax_law.set_xlabel("total params (log scale)")
ax_law.set_ylabel("min val_loss_forecast (post step 5K)")
ax_law.set_title("Scaling law:  loss = A · params^(−α)", fontsize=11, fontweight="bold")
ax_law.grid(alpha=0.3, which="both")

# If we have ≥ 2 points, fit + draw the power law.
if len(points) >= 2:
    n_arr = np.array([p[0] for p in points], dtype=float)
    l_arr = np.array([p[1] for p in points], dtype=float)
    # Fit log-log line: log(loss) = log(A) - α·log(n)
    coef = np.polyfit(np.log(n_arr), np.log(l_arr), 1)
    alpha = -coef[0]
    log_A = coef[1]
    A = np.exp(log_A)
    # Plot the fitted line over the visible x range.
    xs = np.geomspace(n_arr.min() * 0.5, n_arr.max() * 2.0, 100)
    ys = A * xs ** (-alpha)
    ax_law.plot(xs, ys, ls="--", color="#374151", lw=1.2, alpha=0.7,
                label=f"fit: loss = {A:.2f} × n^(−{alpha:.4f})")

    # Annotate the slope on-plot.
    ax_law.text(0.02, 0.05,
                f"α = {alpha:.4f}\n"
                f"(canonical Time-FM α ≈ 0.05–0.10 per Chronos/MOIRAI)\n"
                f"Extrapolation: 1B params → loss ≈ "
                f"{A * (1e9 ** (-alpha)):.2f}",
                transform=ax_law.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa"))
    ax_law.legend(loc="upper right", fontsize=9, framealpha=0.9)

plt.tight_layout()
out = ARTIFACTS / "phase2_scaling_comparison.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved {out}")


# %%
# Saturation slope per run (last 10K steps) — quantifies "still falling" vs "saturated".
print("\nSaturation slopes (forecast loss reduction per Kstep, last 10K steps):")
print(f"  {'run':<22} {'final':>9} {'slope/Kstep':>13} {'verdict':>14}")
for label, val, _, _ in loaded:
    last10 = val[val["step"] >= val["step"].max() - 10_000]
    if len(last10) < 2:
        continue
    delta = last10["val_loss_forecast"].iloc[-1] - last10["val_loss_forecast"].iloc[0]
    span = last10["step"].iloc[-1] - last10["step"].iloc[0]
    slope = delta / max(span, 1) * 1000
    verdict = "saturated" if abs(slope) < 0.005 else "still falling"
    print(f"  {label:<22} {last10['val_loss_forecast'].iloc[-1]:>9.3f} "
          f"{slope:>+13.5f} {verdict:>14}")
