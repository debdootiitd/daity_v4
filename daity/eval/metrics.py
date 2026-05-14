"""Phase 3 evaluation metrics — rank-IC, CRPS, calibration helpers.

This module starts as the home for cross-sectional ranking metrics
used by every Phase 3+ deliverable:

- §3.2 XGBoost baseline (per-fold rank-IC vs labels).
- §3.7 walk-forward sweep (per-horizon per-fold rank-IC for the
  tri-criterion + ≥2-of-4 acceptance gate).
- §3.6 conformal calibration (uses CRPS as an auxiliary measure).
- Phase 5 strategy evaluation (consumes the same rank-IC for trade
  selection).

CRPS and calibration helpers will land in §3.6; for §3.2 we only need
rank-IC and a small bootstrap CI helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class RankICResult:
    """Per-evaluation rank-IC summary.

    `mean_ic` is the average across cross-sections (one IC per `t`).
    `std_ic` is the cross-section-to-cross-section standard deviation
    (gives us a tail estimate for the t-stat).
    `t_stat` is `mean_ic * sqrt(N) / std_ic` — a rough significance
    indicator at large N.
    `n_cross_sections` counts how many distinct `t`s contributed.
    `n_observations` counts total `(sym, t)` rows used.
    """

    mean_ic: float
    std_ic: float
    t_stat: float
    n_cross_sections: int
    n_observations: int


def cross_sectional_rank_ic(
    *,
    predictions: np.ndarray | pl.Series,
    realizations: np.ndarray | pl.Series,
    timestamps: np.ndarray | pl.Series,
    valid: np.ndarray | pl.Series | None = None,
    min_cross_section_size: int = 10,
    return_per_cs: bool = False,
) -> RankICResult | tuple[RankICResult, list[float]]:
    """Spearman rank-IC averaged across cross-sections.

    The standard finance / Time-FM evaluation: for each unique `t`,
    compute Spearman correlation between `predictions` and
    `realizations` across the symbols present at that `t`. Average
    across `t`s.

    Args:
      predictions:    per-(sym, t) predicted scalar (median quantile or
                      ranking score).
      realizations:   per-(sym, t) realized log-return.
      timestamps:     per-(sym, t) anchor timestamp (the column we group
                      on to define a cross-section).
      valid:          optional bool mask; rows with `valid=False` are
                      excluded. If omitted, all rows participate.
      min_cross_section_size: cross-sections with fewer than this many
                      valid rows are dropped (Spearman on too few rows
                      is too noisy to interpret). Default 10.

    Returns:
      `RankICResult`. The cross-section count + observation count are
      reported so the caller can decide whether the IC is statistically
      meaningful for their fold.
    """
    pred = np.asarray(predictions)
    real = np.asarray(realizations)
    ts = np.asarray(timestamps)
    if valid is not None:
        m = np.asarray(valid).astype(bool)
        pred, real, ts = pred[m], real[m], ts[m]

    n = pred.shape[0]
    if n == 0:
        return RankICResult(0.0, 0.0, 0.0, 0, 0)

    # Group by timestamp. Use polars' group_by for stable ordering.
    df = pl.DataFrame({"t": ts, "pred": pred, "real": real})
    ics: list[float] = []
    for _, group in df.group_by("t", maintain_order=True):
        if group.height < min_cross_section_size:
            continue
        # Both columns constant → Spearman is undefined → skip.
        p = group["pred"].to_numpy()
        r = group["real"].to_numpy()
        if np.std(p) == 0.0 or np.std(r) == 0.0:
            continue
        rho, _pval = spearmanr(p, r)
        if np.isnan(rho):
            continue
        ics.append(float(rho))

    n_cs = len(ics)
    if n_cs == 0:
        empty = RankICResult(0.0, 0.0, 0.0, 0, n)
        return (empty, []) if return_per_cs else empty

    arr = np.asarray(ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1)) if n_cs > 1 else 0.0
    t_stat = mean_ic * np.sqrt(n_cs) / std_ic if std_ic > 0 else 0.0
    result = RankICResult(
        mean_ic=mean_ic, std_ic=std_ic, t_stat=t_stat,
        n_cross_sections=n_cs, n_observations=n,
    )
    return (result, ics) if return_per_cs else result


def top_bottom_decile_spread_bps(
    *,
    predictions: np.ndarray,
    realizations: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    """Mean (top-decile realized) − (bottom-decile realized) in bps.

    Reported per `PHASE3.md` §8.2 alongside rank-IC. Positive numbers
    mean the predictor's top-ranked symbols outperform its bottom-ranked
    symbols on average — the basic test of a long-short alpha signal.
    """
    pred = predictions
    real = realizations
    ts = timestamps
    if valid is not None:
        m = valid.astype(bool)
        pred, real, ts = pred[m], real[m], ts[m]
    if pred.shape[0] == 0:
        return 0.0

    df = pl.DataFrame({"t": ts, "pred": pred, "real": real})
    spreads: list[float] = []
    for _, group in df.group_by("t", maintain_order=True):
        if group.height < 10:   # need ≥10 names for a decile to be meaningful
            continue
        p = group["pred"].to_numpy()
        r = group["real"].to_numpy()
        order = np.argsort(p)
        n = order.size
        bottom = r[order[: max(1, n // 10)]].mean()
        top = r[order[-max(1, n // 10):]].mean()
        spreads.append((top - bottom) * 10000.0)   # to bps
    return float(np.mean(spreads)) if spreads else 0.0
