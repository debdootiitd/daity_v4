"""Phase 3X — precision@k / recall@k metrics for the tradeable-candidate
classifier framing.

Two evaluation modes:

1. **Cross-section-aggregate** (the main metric for trading):
   For each cross-section at time t (with N=205 symbols), rank symbols by
   predicted score. Top-k symbols. Precision = fraction with label=+1
   (for LONG side) or label=-1 (for SHORT side). Average across cs.

2. **Universe-aggregate** (recall-style):
   For each cs, top-k predicted. Recall = (# label=+1 in top-k) / (# label=+1 in universe).

These align with how a real trading strategy would be evaluated:
"top-k predictions per rebalance, what fraction actually triggered the
move we wanted?"

Random-ranker baseline:
  Base rate of +1 labels = ~7.8% in our v2 set.
  Random precision@k = 7.8% for ANY k.
  Random recall@k = k / N (where N is universe size).

Good models lift precision@k for small k well above 7.8%; recall@k climbs
faster than random's k/N line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class PrecisionRecallAtK:
    """One row of precision/recall@k aggregated across cross-sections."""
    k: int
    precision: float       # mean across cs of P(top-k contains target_label)
    recall: float          # mean across cs of (top-k ∩ target) / (universe ∩ target)
    n_cross_sections: int
    n_target_in_universe: float    # avg target-label count per cs (for context)


@dataclass(slots=True)
class MeanPnlAtK:
    """Realized-PnL of top-K picks, aggregated across cross-sections.

    For each cs: compute mean(realized_pnl[top-k by score]).
    The metric is the mean of these per-cs means across the val window.

    Aligned with the bandit objective (direct_pnl_loss) and with what a
    deployed strategy would actually realize. Continuous-valued — captures
    PnL magnitude, not just label-touch counts like precision@k.
    """
    k: int
    mean_pnl: float        # average per-pick realized return at top-K (bps as fraction)
    pnl_std: float         # cross-cs std of per-cs mean PnL (units same as mean_pnl)
    sharpe: float          # mean_pnl / pnl_std (units-free risk-adjusted)
    n_cross_sections: int


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k by `scores` (descending). Stable on ties."""
    if k >= scores.size:
        return np.argsort(-scores, kind="stable")
    # argpartition is O(N); pull top-k then sort for stability inside.
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def per_cross_section_precision_recall_at_k(
    *,
    scores: np.ndarray,        # (N_symbols,) higher = more likely target_label
    labels: np.ndarray,        # (N_symbols,) int in {-1, 0, +1}
    k: int,
    target: Literal[+1, -1],
) -> tuple[float, float, int]:
    """Precision + recall @ k on one cross-section.

    Returns `(precision, recall, n_target_in_universe)`.

    `precision` = fraction of top-k symbols whose label == target.
    `recall`    = fraction of target-labeled symbols that landed in top-k.
    """
    if scores.shape != labels.shape:
        msg = f"scores {scores.shape} != labels {labels.shape}"
        raise ValueError(msg)
    n = scores.size
    n_target = int((labels == target).sum())
    if n == 0 or k <= 0:
        return 0.0, 0.0, n_target
    k_eff = min(k, n)
    top_idx = _topk_indices(scores, k_eff)
    n_hit = int((labels[top_idx] == target).sum())
    precision = n_hit / k_eff
    recall = n_hit / n_target if n_target > 0 else 0.0
    return precision, recall, n_target


def per_cross_section_mean_pnl_at_k(
    *,
    scores: np.ndarray,        # (N_symbols,)
    pnl: np.ndarray,           # (N_symbols,) realized return per symbol (signed)
    k: int,
) -> float | None:
    """Mean realized PnL of top-K symbols by `scores` for ONE cross-section.

    Returns None if scores or pnl is empty / k <= 0.
    """
    if scores.shape != pnl.shape:
        msg = f"scores {scores.shape} != pnl {pnl.shape}"
        raise ValueError(msg)
    if scores.size == 0 or k <= 0:
        return None
    k_eff = min(k, scores.size)
    top_idx = _topk_indices(scores, k_eff)
    return float(pnl[top_idx].mean())


def aggregate_mean_pnl_at_k(
    *,
    scores_per_cs: list[np.ndarray],
    pnl_per_cs: list[np.ndarray | None],  # None ⇒ skip that cs
    ks: list[int],
) -> list[MeanPnlAtK]:
    """For each k, mean across cs of per-cs `mean(realized_pnl[top-k])`.

    Cross-sections with `pnl_per_cs[i] is None` are skipped (e.g., when
    the dataset doesn't carry realized_pnl_long for that timestamp).
    """
    out: list[MeanPnlAtK] = []
    for k in ks:
        vals: list[float] = []
        for s, p in zip(scores_per_cs, pnl_per_cs, strict=True):
            if p is None:
                continue
            v = per_cross_section_mean_pnl_at_k(scores=s, pnl=p, k=k)
            if v is not None:
                vals.append(v)
        if not vals:
            out.append(MeanPnlAtK(k=k, mean_pnl=0.0, pnl_std=0.0, sharpe=0.0, n_cross_sections=0))
            continue
        arr = np.array(vals, dtype=np.float64)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sh = mu / sd if sd > 1e-12 else 0.0
        out.append(MeanPnlAtK(k=k, mean_pnl=mu, pnl_std=sd, sharpe=sh,
                              n_cross_sections=len(arr)))
    return out


def aggregate_precision_recall_at_k(
    *,
    scores_per_cs: list[np.ndarray],   # length=n_cs, each (N_symbols,)
    labels_per_cs: list[np.ndarray],
    ks: list[int],
    target: Literal[+1, -1] = +1,
) -> list[PrecisionRecallAtK]:
    """For each k in `ks`, average precision/recall across cross-sections.

    Cross-sections with zero target-labeled symbols are EXCLUDED from the
    recall average (undefined denominator) but included in precision.
    """
    out: list[PrecisionRecallAtK] = []
    for k in ks:
        prec_vals: list[float] = []
        rec_vals: list[float] = []
        n_target_total = 0
        n_cs_used = 0
        for s, l in zip(scores_per_cs, labels_per_cs, strict=True):
            p, r, nt = per_cross_section_precision_recall_at_k(
                scores=s, labels=l, k=k, target=target,
            )
            n_target_total += nt
            prec_vals.append(p)
            if nt > 0:
                rec_vals.append(r)
            n_cs_used += 1
        avg_p = float(np.mean(prec_vals)) if prec_vals else 0.0
        avg_r = float(np.mean(rec_vals)) if rec_vals else 0.0
        out.append(PrecisionRecallAtK(
            k=k, precision=avg_p, recall=avg_r,
            n_cross_sections=n_cs_used,
            n_target_in_universe=n_target_total / max(1, n_cs_used),
        ))
    return out


# ----- Plotting (matplotlib, optional dependency) -----


def plot_precision_recall_curves(
    long_series: list[PrecisionRecallAtK],
    short_series: list[PrecisionRecallAtK] | None = None,
    *,
    title: str = "",
    base_rate_long: float | None = None,
    base_rate_short: float | None = None,
    output_path: str | None = None,
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Two-panel plot: precision@k and recall@k vs k.

    `base_rate_*` is the marginal class probability in the eval universe
    (drawn as a horizontal dashed line — random-ranker baseline). For our
    v2 labels: base_rate_long ≈ 0.078, base_rate_short ≈ 0.074.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    def _plot_series(ax, series, label, color, panel):
        ks = [s.k for s in series]
        vals = [getattr(s, panel) for s in series]
        ax.plot(ks, vals, marker="o", color=color, label=label, linewidth=1.8)

    _plot_series(ax_p, long_series, "LONG (label=+1)", "C0", "precision")
    _plot_series(ax_r, long_series, "LONG (label=+1)", "C0", "recall")
    if short_series is not None:
        _plot_series(ax_p, short_series, "SHORT (label=-1)", "C3", "precision")
        _plot_series(ax_r, short_series, "SHORT (label=-1)", "C3", "recall")

    if base_rate_long is not None:
        ax_p.axhline(base_rate_long, color="C0", linestyle="--", alpha=0.4,
                     label=f"LONG random baseline ({base_rate_long:.3f})")
    if base_rate_short is not None and short_series is not None:
        ax_p.axhline(base_rate_short, color="C3", linestyle="--", alpha=0.4,
                     label=f"SHORT random baseline ({base_rate_short:.3f})")

    # Random recall line: k / N where N ≈ 205 — diagonal
    n_universe = long_series[0].n_target_in_universe / max(base_rate_long or 1, 1e-9) if base_rate_long else 200
    ks_max = max(s.k for s in long_series)
    ax_r.plot([1, ks_max], [1/n_universe, ks_max/n_universe], "k--", alpha=0.4,
              label=f"random recall (k/N={1.0/n_universe:.3f}/k)")

    ax_p.set_xlabel("k (top-k predictions per cross-section)")
    ax_p.set_ylabel("precision @ k")
    ax_p.set_title(f"Precision @ k — {title}")
    ax_p.grid(alpha=0.3); ax_p.legend(fontsize=8)

    ax_r.set_xlabel("k")
    ax_r.set_ylabel("recall @ k")
    ax_r.set_title(f"Recall @ k — {title}")
    ax_r.grid(alpha=0.3); ax_r.legend(fontsize=8)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig
