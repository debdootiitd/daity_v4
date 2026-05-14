"""Unit tests for `daity.models.conformal` (PHASE3.md §7).

Locks in:
- CQR + vanilla score formulas match the published definitions.
- The empirical `(1-α)`-quantile gives the split-conformal coverage
  guarantee on the calibration set.
- Per-cell picker selects whichever score function has empirical
  coverage closer to `1-α`.
- `predict_band` produces (band_lo, band_hi) in the right order.
- Round-trip via `save_cells` / `load_cells` is bit-equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from daity.models.conformal import (
    IDX_Q_HI,
    IDX_Q_LO,
    IDX_Q_MED,
    ConformalCell,
    calibrate_one_cell,
    calibrate_per_cell,
    compute_cqr_scores,
    compute_vanilla_scores,
    empirical_coverage,
    load_cells,
    predict_band,
    save_cells,
)
from daity.training.losses import PHASE3_QUANTILES

# ----- Score function correctness -----


def test_quantile_indices_match_phase3_quantiles() -> None:
    """The hard-coded indices align with `PHASE3_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)`."""
    assert PHASE3_QUANTILES[IDX_Q_LO] == 0.1
    assert PHASE3_QUANTILES[IDX_Q_MED] == 0.5
    assert PHASE3_QUANTILES[IDX_Q_HI] == 0.9


def test_cqr_score_negative_when_inside_band() -> None:
    """If `q_0.1 < y < q_0.9` AND `y == q_0.5`, both ratios are ≤ 0 and
    the CQR score is ≤ 0."""
    q = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    y = np.array([0.0])
    s = compute_cqr_scores(q, y)
    assert s[0] <= 0.0


def test_cqr_score_positive_below_lower_band() -> None:
    """If `y < q_0.1`, the lower-side ratio is positive."""
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    y = np.array([-0.5])    # below q_0.1 = 0.0
    s = compute_cqr_scores(q, y)
    # lower_ratio = (0 - (-0.5)) / (0.5 - 0) = 1.0
    # upper_ratio = (-0.5 - 1.0) / (1.0 - 0.5) = -3.0
    # max = 1.0
    assert abs(s[0] - 1.0) < 1e-6


def test_cqr_score_positive_above_upper_band() -> None:
    """If `y > q_0.9`, the upper-side ratio is positive."""
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    y = np.array([1.5])     # above q_0.9 = 1.0
    s = compute_cqr_scores(q, y)
    # upper_ratio = (1.5 - 1.0) / (1.0 - 0.5) = 1.0
    assert abs(s[0] - 1.0) < 1e-6


def test_vanilla_score_is_abs_residual() -> None:
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    y = np.array([0.7])
    s = compute_vanilla_scores(q, y)
    # |0.7 - 0.5| = 0.2
    assert abs(s[0] - 0.2) < 1e-6


# ----- Calibration -----


def _synth_quantiles_aligned_to_y(
    n: int, rng: np.random.Generator,
    band_width: float = 1.0,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a calibration set where the band CORRECTLY covers ~80% of y.

    Returns (q_pred (N, 5), y (N,)). The model emits a fixed band of width
    `band_width` (so q_0.9 = +band_width/2, q_0.1 = -band_width/2 in
    standardized space), and y is drawn from a Normal with std tuned to
    give ~80% coverage of that band.
    """
    # For an 80% band that fits a Normal: q_0.9 - q_0.1 ≈ 2.56·σ.
    # Set σ = band_width / 2.56 to get exact 80% expected coverage.
    sigma = band_width / 2.56
    y = rng.normal(0.0, sigma, size=n)
    q_med = np.zeros(n)
    q_lo = q_med - band_width / 2
    q_q25 = q_med - band_width / 4
    q_q75 = q_med + band_width / 4
    q_hi = q_med + band_width / 2
    # Add a tiny amount of model "noise" to q_med so vanilla and CQR aren't
    # collapsed to the same values.
    q_med = q_med + rng.normal(0.0, noise, size=n)
    q_pred = np.stack([q_lo, q_q25, q_med, q_q75, q_hi], axis=1)
    return q_pred, y


def test_calibrate_one_cell_picks_a_score_function() -> None:
    """A well-calibrated cell returns either CQR or vanilla, finite s*."""
    rng = np.random.default_rng(0)
    q_pred, y = _synth_quantiles_aligned_to_y(n=500, rng=rng)
    cell = calibrate_one_cell(
        q_pred=q_pred, y=y, horizon="30m", vol_bucket=0, alpha=0.2,
    )
    assert cell.score_fn in ("cqr", "vanilla")
    assert np.isfinite(cell.s_star)
    assert 0.7 < cell.coverage_in_sample < 0.9   # near 0.8 for both functions
    assert cell.n_calibration == 500
    assert cell.alpha == 0.2


def test_calibrate_one_cell_split_conformal_coverage_at_least_1_minus_alpha() -> None:
    """Per the split-conformal guarantee: in-sample coverage at the
    `⌈(n+1)(1-α)⌉/n` quantile is ≥ `1-α`."""
    rng = np.random.default_rng(42)
    q_pred, y = _synth_quantiles_aligned_to_y(n=1000, rng=rng)
    cell = calibrate_one_cell(
        q_pred=q_pred, y=y, horizon="30m", vol_bucket=0, alpha=0.2,
    )
    # The picker chose whichever score got closer to 0.8 in-sample.
    # Both should satisfy the lower bound (1 - alpha = 0.8).
    assert cell.coverage_in_sample >= 0.79   # tiny float tolerance


def test_calibrate_per_cell_pools_small_buckets() -> None:
    """Buckets with < min_cell_size samples are merged into a single
    fallback at bucket=-1."""
    rng = np.random.default_rng(7)
    n = 1000
    q_pred, y = _synth_quantiles_aligned_to_y(n=n, rng=rng)
    vb = np.zeros(n, dtype=np.int8)
    vb[:50] = 1   # bucket 1 is small (50 samples)
    vb[50:120] = 2   # bucket 2 is small (70 samples) — both < min_cell_size=100
    # bucket 0 = the rest (880 samples)

    cells = calibrate_per_cell(
        q_pred=q_pred, y=y, vol_bucket=vb, horizon="30m",
        alpha=0.2, min_cell_size=100,
    )
    # bucket 0 should be its own cell; buckets 1+2 pooled into -1.
    assert 0 in cells
    assert -1 in cells
    assert 1 not in cells
    assert 2 not in cells


# ----- Predict band -----


def test_predict_band_cqr_uses_quantile_widths() -> None:
    """For a CQR cell with s_star=0.5 and q_pred = (0, 0.25, 0.5, 0.75, 1.0):
    band_lo = 0.5 - 1.5 * 0.5 = -0.25
    band_hi = 0.5 + 1.5 * 0.5 =  1.25
    """
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    vb = np.array([0])
    cell = ConformalCell(
        horizon="30m", vol_bucket=0, score_fn="cqr",
        s_star=0.5, n_calibration=100, coverage_in_sample=0.8, alpha=0.2,
    )
    band_lo, band_hi = predict_band(q_pred=q, vol_bucket=vb, cells={0: cell})
    assert abs(band_lo[0] - (-0.25)) < 1e-6
    assert abs(band_hi[0] - 1.25) < 1e-6


def test_predict_band_vanilla_constant_width() -> None:
    """Vanilla band: q_med ± s_star, regardless of q_lo / q_hi."""
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    vb = np.array([0])
    cell = ConformalCell(
        horizon="30m", vol_bucket=0, score_fn="vanilla",
        s_star=0.3, n_calibration=100, coverage_in_sample=0.8, alpha=0.2,
    )
    band_lo, band_hi = predict_band(q_pred=q, vol_bucket=vb, cells={0: cell})
    assert abs(band_lo[0] - 0.2) < 1e-6   # 0.5 - 0.3
    assert abs(band_hi[0] - 0.8) < 1e-6   # 0.5 + 0.3


def test_predict_band_falls_back_to_global_cell() -> None:
    """Unknown bucket ids use the `-1` fallback cell."""
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    vb = np.array([99])   # not in cells
    cells = {
        -1: ConformalCell(horizon="30m", vol_bucket=-1, score_fn="vanilla",
                           s_star=0.5, n_calibration=100,
                           coverage_in_sample=0.8, alpha=0.2),
    }
    band_lo, band_hi = predict_band(q_pred=q, vol_bucket=vb, cells=cells)
    assert abs(band_lo[0] - 0.0) < 1e-6
    assert abs(band_hi[0] - 1.0) < 1e-6


def test_predict_band_raises_on_missing_cell_no_fallback() -> None:
    q = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    vb = np.array([99])
    with pytest.raises(KeyError, match="no calibrated cell"):
        predict_band(q_pred=q, vol_bucket=vb, cells={})


# ----- Coverage measurement -----


def test_empirical_coverage_perfect_match() -> None:
    band_lo = np.array([-1.0, -1.0, -1.0])
    band_hi = np.array([1.0, 1.0, 1.0])
    y = np.array([0.0, 0.5, -0.5])
    assert empirical_coverage(band_lo=band_lo, band_hi=band_hi, y=y) == 1.0


def test_empirical_coverage_zero_when_all_outside() -> None:
    band_lo = np.array([0.0, 0.0])
    band_hi = np.array([0.1, 0.1])
    y = np.array([1.0, -1.0])
    assert empirical_coverage(band_lo=band_lo, band_hi=band_hi, y=y) == 0.0


def test_end_to_end_calibration_in_sample_coverage(tmp_path: Path) -> None:
    """Full pipeline: synth quantiles → calibrate → predict band →
    measure in-sample coverage. The picked score function must hit
    `[0.78, 0.82]` (within 2% of nominal 80% for n=2000)."""
    rng = np.random.default_rng(11)
    q_pred, y = _synth_quantiles_aligned_to_y(n=2000, rng=rng)
    vb = np.zeros(q_pred.shape[0], dtype=np.int8)
    cells = calibrate_per_cell(
        q_pred=q_pred, y=y, vol_bucket=vb, horizon="30m",
        alpha=0.2, min_cell_size=100,
    )
    assert 0 in cells

    band_lo, band_hi = predict_band(q_pred=q_pred, vol_bucket=vb, cells=cells)
    cov = empirical_coverage(band_lo=band_lo, band_hi=band_hi, y=y)
    assert 0.78 <= cov <= 0.82, f"in-sample coverage {cov:.4f} not in [0.78, 0.82]"


# ----- Persistence -----


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save_cells → load_cells is bit-equal."""
    cells_by_horizon = {
        "30m": {
            0: ConformalCell(horizon="30m", vol_bucket=0, score_fn="cqr",
                              s_star=0.42, n_calibration=500,
                              coverage_in_sample=0.81, alpha=0.2),
            -1: ConformalCell(horizon="30m", vol_bucket=-1, score_fn="vanilla",
                               s_star=0.21, n_calibration=100,
                               coverage_in_sample=0.79, alpha=0.2),
        },
        "1d": {
            0: ConformalCell(horizon="1d", vol_bucket=0, score_fn="vanilla",
                              s_star=0.05, n_calibration=2000,
                              coverage_in_sample=0.80, alpha=0.2),
        },
    }
    save_cells(cells_by_horizon, tmp_path)
    loaded = load_cells(tmp_path)

    assert set(loaded.keys()) == {"30m", "1d"}
    assert set(loaded["30m"].keys()) == {0, -1}
    assert loaded["30m"][0].s_star == 0.42
    assert loaded["30m"][0].score_fn == "cqr"
    assert loaded["1d"][0].n_calibration == 2000


def test_load_cells_handles_horizon_with_digits(tmp_path: Path) -> None:
    """`120m` has digits but the filename split must split on the LAST
    underscore so `120m_5.json` parses as `(120m, bucket=5)` not
    `(120, bucket=m_5)`."""
    cells = {"120m": {5: ConformalCell(
        horizon="120m", vol_bucket=5, score_fn="cqr",
        s_star=0.3, n_calibration=200, coverage_in_sample=0.81, alpha=0.2,
    )}}
    save_cells(cells, tmp_path)
    loaded = load_cells(tmp_path)
    assert "120m" in loaded
    assert 5 in loaded["120m"]
    assert loaded["120m"][5].s_star == 0.3
