"""Conformal calibration for each prediction head of the cohort model.

For each horizon h:
  1. Split predictions into CALIBRATION (e.g., 2025 H1) and TEST (rest).
  2. On calibration: compute non-conformity scores = |real_lr - pred_lr|.
     Quantile q_α of these scores gives the symmetric ±q_α interval that
     achieves nominal coverage (1-α) on exchangeable data.
  3. On test: for each nominal level α ∈ {0.05, 0.10, ..., 0.50}:
        interval = [pred - q_α, pred + q_α]
        actual coverage = fraction of test points where real ∈ interval
     Plot nominal vs actual (reliability diagram).
  4. Also: scatter of pred vs actual + binned-mean (lift) plot per horizon.

Output: PNG grid (one row per horizon, three columns:
   (a) reliability diagram, (b) pred-vs-actual scatter, (c) binned lift).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--cal-start", type=str, default="2025-01-01")
    ap.add_argument("--cal-end",   type=str, default="2025-06-30")
    ap.add_argument("--test-start", type=str, default="2025-07-01")
    ap.add_argument("--test-end",   type=str, default="2026-04-30")
    ap.add_argument("--horizons", type=str, default="",
                    help="Comma-separated horizons to calibrate. Empty = all in data.")
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pl.read_parquet(args.predictions_path)
    print(f"loaded {df.shape[0]} rows", flush=True)
    cal_start = date.fromisoformat(args.cal_start)
    cal_end   = date.fromisoformat(args.cal_end)
    test_start = date.fromisoformat(args.test_start)
    test_end   = date.fromisoformat(args.test_end)

    horizons_in_data = sorted(df["horizon"].unique().to_list())
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()] or horizons_in_data
    print(f"horizons: {horizons}", flush=True)

    nominal_levels = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    summary_rows = []

    nrows = len(horizons)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 3.2 * nrows), squeeze=False)
    for r, h in enumerate(horizons):
        cal = df.filter(
            (pl.col("horizon") == h)
            & (pl.col("date") >= cal_start) & (pl.col("date") <= cal_end),
        )
        tst = df.filter(
            (pl.col("horizon") == h)
            & (pl.col("date") >= test_start) & (pl.col("date") <= test_end),
        )
        cal_pred = cal["pred_lr"].to_numpy()
        cal_real = cal["real_lr"].to_numpy()
        tst_pred = tst["pred_lr"].to_numpy()
        tst_real = tst["real_lr"].to_numpy()
        if cal.shape[0] < 100 or tst.shape[0] < 100:
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"{h}: insufficient data\n"
                                          f"cal={cal.shape[0]} test={tst.shape[0]}",
                                ha="center", va="center", transform=axes[r, c].transAxes)
                axes[r, c].set_title(f"{h} ({c+1})")
            continue

        # Non-conformity scores (symmetric)
        cal_residuals = np.abs(cal_real - cal_pred)
        # Conformal q-values: for nominal coverage 1-α, we want the (1-α)-th
        # quantile of cal residuals (with small finite-sample correction).
        n_cal = cal_residuals.size
        q_values = []
        nominal_coverage = []   # = 1 - alpha
        actual_coverage = []
        interval_widths = []
        for alpha in nominal_levels:
            target = 1 - alpha
            # Finite-sample-corrected quantile (Romano-Patterson):
            k = int(np.ceil((n_cal + 1) * target))
            k = min(max(k, 1), n_cal)
            q_alpha = float(np.partition(cal_residuals, k - 1)[k - 1])
            q_values.append(q_alpha)
            interval_widths.append(2 * q_alpha)
            cov = ((tst_real >= tst_pred - q_alpha) & (tst_real <= tst_pred + q_alpha)).mean()
            actual_coverage.append(float(cov))
            nominal_coverage.append(float(target))

        summary_rows.append({
            "horizon": h, "n_cal": int(n_cal), "n_test": int(tst.shape[0]),
            "cal_residual_p50": float(np.median(cal_residuals)),
            "cal_residual_p90": float(np.quantile(cal_residuals, 0.9)),
            "coverage@90nominal": float(actual_coverage[1]),  # alpha=0.10 → 90%
            "coverage@50nominal": float(actual_coverage[-1]),
            "interval_width_p90": float(interval_widths[1]),
        })

        # (a) Reliability diagram
        ax = axes[r, 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="ideal")
        ax.plot(nominal_coverage, actual_coverage, "o-", color="tab:blue", lw=2,
                label=f"{h}")
        ax.set_xlabel("nominal coverage (1-α)")
        ax.set_ylabel("empirical coverage on test")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{h}: conformal reliability")

        # (b) Pred-vs-actual scatter
        ax = axes[r, 1]
        # Subsample for visibility
        idx = np.random.default_rng(0).choice(
            tst_pred.size, size=min(5000, tst_pred.size), replace=False,
        )
        ax.scatter(tst_pred[idx] * 10000, tst_real[idx] * 10000,
                   s=2, alpha=0.2, color="tab:gray")
        # Identity line
        lim = max(abs(tst_pred).max(), abs(tst_real).max()) * 10000
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        # Univariate stats
        from scipy.stats import spearmanr, pearsonr
        ic_s, _ = spearmanr(tst_pred, tst_real)
        ic_p, _ = pearsonr(tst_pred, tst_real)
        ax.set_xlabel("pred (bps)"); ax.set_ylabel("real (bps)")
        ax.set_title(f"{h}: pred vs real (spearman={ic_s:+.3f}, pearson={ic_p:+.3f})")
        ax.grid(alpha=0.3)

        # (c) Binned-mean lift chart
        ax = axes[r, 2]
        n_bins = 10
        bin_edges = np.quantile(tst_pred, np.linspace(0, 1, n_bins + 1))
        # Make edges strictly increasing
        bin_edges = np.unique(bin_edges)
        if bin_edges.size < 3:
            ax.text(0.5, 0.5, "deg. bins", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            bin_idx = np.clip(np.digitize(tst_pred, bin_edges[1:-1]),
                              0, bin_edges.size - 2)
            bin_pred = np.zeros(bin_edges.size - 1)
            bin_real = np.zeros(bin_edges.size - 1)
            bin_n = np.zeros(bin_edges.size - 1, dtype=int)
            for b in range(bin_edges.size - 1):
                m = bin_idx == b
                if m.sum() == 0: continue
                bin_pred[b] = tst_pred[m].mean()
                bin_real[b] = tst_real[m].mean()
                bin_n[b] = int(m.sum())
            ax.bar(np.arange(bin_edges.size - 1), bin_real * 10000,
                   color=["tab:green" if r > 0 else "tab:red" for r in bin_real],
                   alpha=0.6, label="mean realized")
            ax.plot(np.arange(bin_edges.size - 1), bin_pred * 10000,
                    "o-", color="tab:blue", lw=1.5, label="mean predicted")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("predicted decile (low → high)")
            ax.set_ylabel("bps")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_title(f"{h}: decile lift (top-decile real = {bin_real[-1]*10000:+.1f} bps)")
            ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out_png}", flush=True)

    if args.out_summary:
        pl.DataFrame(summary_rows).write_parquet(args.out_summary)
        print(f"summary -> {args.out_summary}", flush=True)
    for row in summary_rows:
        print(f"  {row['horizon']:>15} | n_cal={row['n_cal']:>5} n_test={row['n_test']:>5} | "
              f"cal_res_p90={row['cal_residual_p90']*10000:>6.1f}bps | "
              f"cov@90nom={row['coverage@90nominal']*100:.1f}% | "
              f"width_p90={row['interval_width_p90']*10000:.1f}bps",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
