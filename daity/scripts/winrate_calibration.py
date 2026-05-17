"""Binary win-rate calibration per prediction head.

Definition: win = 1 if realized log return × 10000 > win_threshold_bps
            (default: 20 bps in the horizon's time interval).

For each horizon:
  1. Bin predictions into deciles by predicted log return.
  2. Per decile: empirical win rate, mean prediction, count.
  3. Also: cumulative win rate when taking top-N% picks (lift curve).
  4. ROC AUC of predictions as ranker for the binary win label.
  5. Brier score (after isotonic calibration, optional).

Output:
  - PNG grid: rows = horizons, columns = (decile win-rate, lift curve, ROC).
  - Parquet summary table.
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
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.isotonic import IsotonicRegression


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--cal-start", type=str, default="2024-07-01")
    ap.add_argument("--cal-end",   type=str, default="2025-06-30")
    ap.add_argument("--test-start", type=str, default="2025-07-01")
    ap.add_argument("--test-end",   type=str, default="2026-04-30")
    ap.add_argument("--win-threshold-bps", type=float, default=20.0)
    ap.add_argument("--horizons", type=str, default="")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pl.read_parquet(args.predictions_path)
    cal_start = date.fromisoformat(args.cal_start)
    cal_end   = date.fromisoformat(args.cal_end)
    test_start = date.fromisoformat(args.test_start)
    test_end   = date.fromisoformat(args.test_end)
    win_thr = args.win_threshold_bps / 10000.0

    horizons_in_data = sorted(df["horizon"].unique().to_list())
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()] or horizons_in_data

    summary = []
    nrows = len(horizons)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 3.0 * nrows), squeeze=False)

    for r, h in enumerate(horizons):
        cal = df.filter(
            (pl.col("horizon") == h)
            & (pl.col("date") >= cal_start) & (pl.col("date") <= cal_end),
        )
        tst = df.filter(
            (pl.col("horizon") == h)
            & (pl.col("date") >= test_start) & (pl.col("date") <= test_end),
        )
        if tst.shape[0] < 500:
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"{h}: insufficient",
                                ha="center", transform=axes[r, c].transAxes)
            continue

        cal_pred = cal["pred_lr"].to_numpy()
        cal_real = cal["real_lr"].to_numpy()
        cal_y    = (cal_real > win_thr).astype(int)
        tst_pred = tst["pred_lr"].to_numpy()
        tst_real = tst["real_lr"].to_numpy()
        tst_y    = (tst_real > win_thr).astype(int)
        base_rate_cal = cal_y.mean()
        base_rate_tst = tst_y.mean()

        # ROC AUC of raw pred as ranker on test
        try:
            auc = roc_auc_score(tst_y, tst_pred)
        except ValueError:
            auc = float("nan")

        # Isotonic-calibrated probability on test (fit on cal)
        try:
            iso = IsotonicRegression(out_of_bounds="clip").fit(cal_pred, cal_y)
            tst_prob = iso.predict(tst_pred)
            brier = brier_score_loss(tst_y, tst_prob)
        except Exception:
            tst_prob = np.full_like(tst_pred, base_rate_cal)
            brier = float("nan")

        # (a) Decile win-rate
        ax = axes[r, 0]
        bin_edges = np.quantile(tst_pred, np.linspace(0, 1, args.n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if bin_edges.size < 3:
            ax.text(0.5, 0.5, "degenerate bins", ha="center", transform=ax.transAxes)
        else:
            n_bins = bin_edges.size - 1
            idx = np.clip(np.digitize(tst_pred, bin_edges[1:-1]), 0, n_bins - 1)
            wr   = np.zeros(n_bins)
            mp   = np.zeros(n_bins)
            cnt  = np.zeros(n_bins, dtype=int)
            for b in range(n_bins):
                m = (idx == b)
                if m.sum() == 0: continue
                wr[b] = tst_y[m].mean()
                mp[b] = tst_pred[m].mean()
                cnt[b] = int(m.sum())
            colors = ["tab:green" if wr[b] > base_rate_tst else "tab:red"
                      for b in range(n_bins)]
            ax.bar(np.arange(n_bins), wr * 100, color=colors, alpha=0.7)
            ax.axhline(base_rate_tst * 100, color="black", lw=1.0, linestyle="--",
                       label=f"base rate {base_rate_tst*100:.1f}%")
            ax.set_xlabel("predicted decile (low → high)")
            ax.set_ylabel("empirical win rate (%)")
            ax.set_title(f"{h}: decile win-rate (top={wr[-1]*100:.1f}%, base={base_rate_tst*100:.1f}%)")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.3, axis="y")

        # (b) Cumulative win rate of top-N% lift curve
        ax = axes[r, 1]
        order = np.argsort(-tst_pred)
        y_sorted = tst_y[order]
        # Cumulative win rate at the top N % (N = 1..100)
        n = tst_y.size
        cum_wins = np.cumsum(y_sorted)
        cum_count = np.arange(1, n + 1)
        cum_wr = cum_wins / cum_count
        pcts = cum_count / n * 100
        # Plot top 30% slice (most relevant for trading)
        cut = max(int(0.30 * n), 100)
        ax.plot(pcts[:cut], cum_wr[:cut] * 100, color="tab:blue", lw=2,
                label="cum win-rate")
        ax.axhline(base_rate_tst * 100, color="black", lw=1.0, linestyle="--",
                   label=f"base {base_rate_tst*100:.1f}%")
        # Top-K=10 stocks per day approx — show a marker at top-K%
        # Top-10 of ~200 = 5% of universe.
        for top_pct in [1, 5, 10, 20]:
            i = max(int(top_pct / 100 * n) - 1, 0)
            ax.axvline(top_pct, color="gray", alpha=0.3, lw=0.5)
            ax.annotate(f"{cum_wr[i]*100:.1f}%@{top_pct}%",
                        xy=(top_pct, cum_wr[i] * 100),
                        xytext=(top_pct + 0.5, cum_wr[i] * 100 + 1),
                        fontsize=8)
        ax.set_xlabel("top-N% of predictions (sorted desc)")
        ax.set_ylabel("cumulative win rate (%)")
        ax.set_title(f"{h}: lift curve (AUC={auc:.3f})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        # (c) ROC curve
        ax = axes[r, 2]
        try:
            fpr, tpr, _ = roc_curve(tst_y, tst_pred)
            ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{h}: ROC | brier(iso)={brier:.4f}")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {e}", ha="center", transform=ax.transAxes)

        summary.append({
            "horizon": h,
            "n_cal": int(cal.shape[0]),
            "n_test": int(tst.shape[0]),
            "base_rate_test": float(base_rate_tst),
            "auc": float(auc) if not np.isnan(auc) else None,
            "brier_iso": float(brier) if not np.isnan(brier) else None,
            "top1pct_winrate": float(cum_wr[max(int(0.01 * n) - 1, 0)]),
            "top5pct_winrate": float(cum_wr[max(int(0.05 * n) - 1, 0)]),
            "top10pct_winrate": float(cum_wr[max(int(0.10 * n) - 1, 0)]),
        })

    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out_png}", flush=True)

    if args.out_summary:
        pl.DataFrame(summary).write_parquet(args.out_summary)

    print(f"\n=== Win-rate summary (win = real > {args.win_threshold_bps}bps) ===")
    print(f"{'horizon':>15} {'n_test':>7} {'base':>6} {'AUC':>6} "
          f"{'top1%':>7} {'top5%':>7} {'top10%':>7}")
    for s in summary:
        print(f"{s['horizon']:>15} {s['n_test']:>7} "
              f"{s['base_rate_test']*100:>5.1f}% "
              f"{(s.get('auc') or 0)*100:>5.1f} "
              f"{s['top1pct_winrate']*100:>6.1f}% "
              f"{s['top5pct_winrate']*100:>6.1f}% "
              f"{s['top10pct_winrate']*100:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
