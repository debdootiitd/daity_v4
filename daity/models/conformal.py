"""Phase 3.6 — split-conformal calibration (`PHASE3.md` §7).

Two conformal score functions, with per-`(horizon, vol_bucket)` selection
between them at calibration time (amendment 9):

- **CQR** (Romano-Patterson-Candès 2019, Eq. 6): heteroscedastic score
  using the model's own `(q_0.1, q_0.5, q_0.9)` to modulate the band.
- **Vanilla split-conformal**: constant-width score from the absolute
  residual `|y - q_0.5|`.

At calibration, we compute BOTH scores per residual, find each score's
empirical `(1-α)`-quantile, simulate coverage on the calibration set,
and pick the score whose coverage is closest to `1-α`. The choice is
stored per cell in the output JSON.

At inference, the predicted 80% band depends on the cell's selected
score function (see `predict_band`).

Public surface:

- `compute_cqr_scores(q_pred, y)` / `compute_vanilla_scores(q_pred, y)`
- `ConformalCell` — one cell's calibrated state.
- `calibrate_per_cell(...)` — fit cells from a calibration set.
- `predict_band(...)` — look up the band for a new `(sym, t)` query.

`PHASE3.md` §1.3 (post-amendment 1) — the acceptance gate measures
coverage on the held-out **test fold**, not the calibration set. Use
`empirical_coverage(...)` to compute the gate's input on the test
fold's residuals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from daity.utils.logging import get_logger

log = get_logger(__name__)


# Quantile indices into the 5-quantile head output (matches
# `daity.training.losses.PHASE3_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)`).
IDX_Q_LO: int = 0    # q_0.1
IDX_Q_MED: int = 2   # q_0.5
IDX_Q_HI: int = 4    # q_0.9

# Tiny epsilon to avoid div-by-zero on degenerate quantile predictions.
_EPS: float = 1e-9


def compute_cqr_scores(q_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CQR nonconformity scores (Romano-Patterson-Candès 2019, Eq. 6).

    Args:
        q_pred: (N, 5) predicted quantiles in the order
            `(q_0.1, q_0.25, q_0.5, q_0.75, q_0.9)`.
        y:      (N,) realized targets.

    Returns: (N,) non-conformity scores.

    `s_i = max( (q_0.1 - y_i) / (q_0.5 - q_0.1 + ε),
                (y_i - q_0.9) / (q_0.9 - q_0.5 + ε) )`

    The CQR score is signed-heteroscedastic: if `y` falls below `q_0.1`,
    the LOWER-side ratio is positive and (large) — the model under-
    estimated the lower tail; conversely if `y` falls above `q_0.9`. In
    the well-calibrated case, both ratios are ≤ 0 and `max(.)` is ≤ 0
    too (the inner band already covers `y`).

    Review Major-2: pinball loss doesn't enforce quantile monotonicity,
    so the head can emit rows with `q_lo > q_med` or `q_med > q_hi`. The
    CQR formula assumes monotone quantiles — crossing inverts the
    denominator sign and silently breaks the score. We sort per-row to
    enforce monotonicity before computing the score; rows that needed
    sorting are counted and a warning is emitted (so the §3.7 caller
    can decide whether the underlying quantile head needs more training).
    """
    if q_pred.ndim != 2 or q_pred.shape[1] < 5:
        msg = f"q_pred must be (N, ≥5), got {q_pred.shape}"
        raise ValueError(msg)
    if y.shape[0] != q_pred.shape[0]:
        msg = f"y and q_pred[0] must match: {y.shape}, {q_pred.shape}"
        raise ValueError(msg)

    # Enforce monotonicity (Major-2). Sort the row's quantile values so the
    # CQR formula's directional assumptions hold. Count + warn on crossing.
    q_relevant = q_pred[:, [IDX_Q_LO, IDX_Q_MED, IDX_Q_HI]]
    crossed = (
        (q_relevant[:, 0] > q_relevant[:, 1])
        | (q_relevant[:, 1] > q_relevant[:, 2])
    )
    if crossed.any():
        log.warning(
            "compute_cqr_scores: %d / %d rows have crossed quantiles; "
            "sorting before computing CQR score (this indicates a "
            "miscalibrated quantile head — Major-2 in §3.6 review).",
            int(crossed.sum()), int(crossed.shape[0]),
        )
    q_sorted = np.sort(q_relevant, axis=1)
    q_lo, q_med, q_hi = q_sorted[:, 0], q_sorted[:, 1], q_sorted[:, 2]

    lower_ratio = (q_lo - y) / (q_med - q_lo + _EPS)
    upper_ratio = (y - q_hi) / (q_hi - q_med + _EPS)
    return np.maximum(lower_ratio, upper_ratio)


def compute_vanilla_scores(q_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vanilla split-conformal scores = `|y - q_0.5|`."""
    if q_pred.ndim != 2 or q_pred.shape[1] < 5:
        msg = f"q_pred must be (N, ≥5), got {q_pred.shape}"
        raise ValueError(msg)
    return np.abs(y - q_pred[:, IDX_Q_MED])


@dataclass
class ConformalCell:
    """One `(horizon, vol_bucket)` calibrated cell.

    `score_fn` is the picked function. `s_star` is its `(1-α)`-quantile
    on the calibration set. `coverage_in_sample` is the calibration-set
    coverage for the picked function (sanity assertion per §7.5).
    """
    horizon: str
    vol_bucket: int
    score_fn: Literal["cqr", "vanilla"]
    s_star: float
    n_calibration: int
    coverage_in_sample: float
    alpha: float


def _empirical_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample (1-α)-quantile per split-conformal Eq. 1.

    Use the (`⌈(n+1)(1-α)⌉ / n`)-th order statistic to get the standard
    split-conformal coverage guarantee (Romano 2019 Eq. 1).
    """
    n = scores.shape[0]
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


def calibrate_one_cell(
    *,
    q_pred: np.ndarray,
    y: np.ndarray,
    horizon: str,
    vol_bucket: int,
    alpha: float = 0.2,
) -> ConformalCell:
    """Pick CQR vs vanilla for this cell. Returns the chosen `ConformalCell`.

    Picker rule: compute both score functions' `(1-α)`-quantile, simulate
    the calibration-set coverage each would give, pick the one whose
    coverage is closest to `1-α`. Ties (both equidistant) → CQR by
    convention.

    `PHASE3.md` §7.1 + §7.2.
    """
    s_cqr = compute_cqr_scores(q_pred, y)
    s_van = compute_vanilla_scores(q_pred, y)
    s_star_cqr = _empirical_quantile(s_cqr, alpha)
    s_star_van = _empirical_quantile(s_van, alpha)

    # Simulate in-sample coverage under each.
    cov_cqr = float(np.mean(s_cqr <= s_star_cqr))
    cov_van = float(np.mean(s_van <= s_star_van))
    target = 1.0 - alpha

    dist_cqr = abs(cov_cqr - target)
    dist_van = abs(cov_van - target)
    score_fn: Literal["cqr", "vanilla"] = "cqr" if dist_cqr <= dist_van else "vanilla"

    return ConformalCell(
        horizon=horizon,
        vol_bucket=vol_bucket,
        score_fn=score_fn,
        s_star=s_star_cqr if score_fn == "cqr" else s_star_van,
        n_calibration=int(q_pred.shape[0]),
        coverage_in_sample=cov_cqr if score_fn == "cqr" else cov_van,
        alpha=alpha,
    )


def calibrate_per_cell(
    *,
    q_pred: np.ndarray,
    y: np.ndarray,
    vol_bucket: np.ndarray,
    horizon: str,
    alpha: float = 0.2,
    min_cell_size: int = 1000,
) -> dict[int, ConformalCell]:
    """Fit one `ConformalCell` per unique `vol_bucket` value.

    `min_cell_size` defaults to 1000 per `PHASE3.md` §6.4 + §7.3 (the
    threshold below which split-conformal's empirical `(1-α)`-quantile
    becomes statistically unstable). Cells smaller than this are merged
    into a "global" cell keyed at `vol_bucket=-1` so the conformal
    pipeline still emits a band for them at inference.

    The spec's preferred fallback is `K=5 → K=3` for the WHOLE horizon
    (collapse the binning, not just under-populated buckets). This
    method takes a simpler per-bucket-pooling approach — the caller
    chooses K via the cross-section sampler's `n_vol_buckets`, and if
    some buckets at K=5 are too small they get pooled here. If many
    buckets need pooling, the caller should drop to K=3 in the sampler.
    Logged at `WARNING` per review Major-1.
    """
    cells: dict[int, ConformalCell] = {}
    pooled_mask = np.zeros(q_pred.shape[0], dtype=bool)
    for b in np.unique(vol_bucket):
        mask = vol_bucket == b
        n = int(mask.sum())
        if n < min_cell_size:
            pooled_mask |= mask
            log.warning(
                "calibrate_per_cell %s: bucket %d has %d samples (< min_cell_size=%d); "
                "pooling into bucket=-1. If many buckets need pooling, consider "
                "dropping K=5 → K=3 in the cross-section sampler.",
                horizon, int(b), n, min_cell_size,
            )
            continue
        cells[int(b)] = calibrate_one_cell(
            q_pred=q_pred[mask], y=y[mask], horizon=horizon,
            vol_bucket=int(b), alpha=alpha,
        )
    if pooled_mask.any() and int(pooled_mask.sum()) >= 10:
        cells[-1] = calibrate_one_cell(
            q_pred=q_pred[pooled_mask], y=y[pooled_mask], horizon=horizon,
            vol_bucket=-1, alpha=alpha,
        )
    return cells


def predict_band(
    *,
    q_pred: np.ndarray,
    vol_bucket: np.ndarray,
    cells: dict[int, ConformalCell],
) -> tuple[np.ndarray, np.ndarray]:
    """Look up calibrated band `(band_lo, band_hi)` for each query.

    Args:
        q_pred: (N, 5) quantile predictions in the same order as training.
        vol_bucket: (N,) integer bucket ids.
        cells: mapping `bucket_id -> ConformalCell`. Buckets not in the
            mapping fall back to the `cells[-1]` global cell (if present)
            or raise.

    Returns: `(band_lo, band_hi)` both (N,) arrays.

    For CQR cells:
        band_lo = q_med - (1 + s*) * (q_med - q_lo)
        band_hi = q_med + (1 + s*) * (q_hi - q_med)

    For vanilla cells:
        band_lo = q_med - s*
        band_hi = q_med + s*
    """
    n = q_pred.shape[0]
    band_lo = np.zeros(n, dtype=np.float64)
    band_hi = np.zeros(n, dtype=np.float64)
    q_lo = q_pred[:, IDX_Q_LO]
    q_med = q_pred[:, IDX_Q_MED]
    q_hi = q_pred[:, IDX_Q_HI]

    fallback = cells.get(-1)
    for i in range(n):
        b = int(vol_bucket[i])
        cell = cells.get(b) or fallback
        if cell is None:
            msg = (
                f"predict_band: no calibrated cell for vol_bucket={b} "
                f"and no fallback at vol_bucket=-1"
            )
            raise KeyError(msg)
        if cell.score_fn == "cqr":
            band_lo[i] = q_med[i] - (1.0 + cell.s_star) * (q_med[i] - q_lo[i])
            band_hi[i] = q_med[i] + (1.0 + cell.s_star) * (q_hi[i] - q_med[i])
        else:
            band_lo[i] = q_med[i] - cell.s_star
            band_hi[i] = q_med[i] + cell.s_star
    return band_lo, band_hi


def empirical_coverage(
    *,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    y: np.ndarray,
) -> float:
    """Fraction of `y` that falls in `[band_lo, band_hi]` (inclusive)."""
    if not (band_lo.shape == band_hi.shape == y.shape):
        msg = "band_lo / band_hi / y must all have the same shape"
        raise ValueError(msg)
    if y.shape[0] == 0:
        return 0.0
    inside = (y >= band_lo) & (y <= band_hi)
    return float(inside.mean())


# ----- Persistence -----


def save_cells(
    cells_by_horizon: dict[str, dict[int, ConformalCell]],
    output_root: Path,
) -> None:
    """Write one JSON per `(horizon, vol_bucket)` cell to
    `{output_root}/{horizon}_{vol_bucket}.json` per `PHASE3.md` §7.4.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    for horizon, by_bucket in cells_by_horizon.items():
        for b, cell in by_bucket.items():
            path = output_root / f"{horizon}_{b}.json"
            path.write_text(json.dumps(asdict(cell), indent=2))


def load_cells(input_root: Path) -> dict[str, dict[int, ConformalCell]]:
    """Inverse of `save_cells`. Returns `{horizon: {bucket: ConformalCell}}`."""
    out: dict[str, dict[int, ConformalCell]] = {}
    for path in sorted(input_root.glob("*_*.json")):
        # filename: `{horizon}_{bucket}.json`. horizon can contain digits
        # (`30m`, `120m`), so rsplit on the LAST underscore.
        stem = path.stem
        horizon, bucket_str = stem.rsplit("_", 1)
        bucket = int(bucket_str)
        data = json.loads(path.read_text())
        cell = ConformalCell(**data)
        out.setdefault(horizon, {})[bucket] = cell
    return out
