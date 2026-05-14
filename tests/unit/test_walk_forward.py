"""Unit tests for `daity.eval.walk_forward` (PHASE3.md §8.1 / §1).

Locks in:
- The per-horizon tri-criterion: ALL of (mean delta ≥ 0.005, ≥3 of 4
  folds individually positive, ≥3 of 4 folds individually beat XGB).
- Edge cases: borderline mean delta, 4-of-4 vs 3-of-4 vs 2-of-4 positive,
  empty results.
- The ≥2-of-4-horizons rule.
- The §1.4 random-init gate's best-of-seeds per-fold logic.
- §1.3 conformal coverage gate.
- JSON schema round-trip + survivorship-audit / fold-metadata wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

from daity.eval.walk_forward import (
    MIN_FOLDS_BEAT_XGB,
    MIN_FOLDS_POSITIVE,
    RANK_IC_DELTA_THRESHOLD,
    FoldHorizonResult,
    emit_walk_forward_json,
    evaluate_conformal_gate,
    evaluate_horizon_gate,
    evaluate_random_init_gate,
    evaluate_walk_forward,
)

# ----- Helpers -----


def _fhr(fold_id: str, horizon: str, mode: str, ic: float,
         seed: int = 0, n_cs: int = 100) -> FoldHorizonResult:
    return FoldHorizonResult(
        fold_id=fold_id, horizon=horizon, mode=mode,  # type: ignore[arg-type]
        seed=seed, rank_ic_mean=ic, n_cross_sections=n_cs,
    )


# ----- Tri-criterion edge cases -----


def test_tri_criterion_4_of_4_positive_clearly_passes() -> None:
    ssl = [
        _fhr("F-9",  "30m", "ssl_heads", 0.015),
        _fhr("F-10", "30m", "ssl_heads", 0.020),
        _fhr("F-11", "30m", "ssl_heads", 0.018),
        _fhr("F-12", "30m", "ssl_heads", 0.012),
    ]
    xgb = [
        _fhr("F-9",  "30m", "xgboost", 0.005),
        _fhr("F-10", "30m", "xgboost", 0.008),
        _fhr("F-11", "30m", "xgboost", 0.006),
        _fhr("F-12", "30m", "xgboost", 0.007),
    ]
    res = evaluate_horizon_gate(horizon="30m", ssl_results=ssl, xgb_results=xgb)
    assert res.passed
    assert res.passed_mean_delta
    assert res.passed_folds_positive
    assert res.passed_folds_beat_xgb
    assert res.folds_ssl_positive == 4
    assert res.folds_ssl_beat_xgb == 4


def test_tri_criterion_3_of_4_positive_passes_b_and_c() -> None:
    """≥3 of 4 is the threshold per amendment 8 — exactly 3 passes."""
    ssl = [
        _fhr("F-9",  "30m", "ssl_heads", 0.020),
        _fhr("F-10", "30m", "ssl_heads", 0.015),
        _fhr("F-11", "30m", "ssl_heads", 0.010),
        _fhr("F-12", "30m", "ssl_heads", -0.005),   # one negative fold
    ]
    xgb = [
        _fhr("F-9",  "30m", "xgboost", 0.000),
        _fhr("F-10", "30m", "xgboost", 0.000),
        _fhr("F-11", "30m", "xgboost", 0.000),
        _fhr("F-12", "30m", "xgboost", 0.000),
    ]
    res = evaluate_horizon_gate(horizon="30m", ssl_results=ssl, xgb_results=xgb)
    assert res.folds_ssl_positive == 3
    assert res.folds_ssl_beat_xgb == 3
    assert res.passed_folds_positive   # 3 ≥ 3 → pass
    assert res.passed_folds_beat_xgb   # 3 ≥ 3 → pass


def test_tri_criterion_2_of_4_positive_fails_b() -> None:
    ssl = [
        _fhr("F-9",  "30m", "ssl_heads", 0.020),
        _fhr("F-10", "30m", "ssl_heads", 0.015),
        _fhr("F-11", "30m", "ssl_heads", -0.005),
        _fhr("F-12", "30m", "ssl_heads", -0.005),
    ]
    xgb = [_fhr(f, "30m", "xgboost", 0.000) for f in ("F-9", "F-10", "F-11", "F-12")]
    res = evaluate_horizon_gate(horizon="30m", ssl_results=ssl, xgb_results=xgb)
    assert res.folds_ssl_positive == 2
    assert not res.passed_folds_positive   # 2 < 3 → fail
    assert not res.passed                  # any one fail → tri-criterion fail


def test_tri_criterion_mean_delta_at_threshold() -> None:
    """Mean delta exactly 0.005 should pass (we use ≥, not strictly >)."""
    ssl = [_fhr(f, "30m", "ssl_heads", 0.005) for f in ("F-9", "F-10", "F-11", "F-12")]
    xgb = [_fhr(f, "30m", "xgboost",   0.000) for f in ("F-9", "F-10", "F-11", "F-12")]
    res = evaluate_horizon_gate(horizon="30m", ssl_results=ssl, xgb_results=xgb)
    assert abs(res.mean_delta - 0.005) < 1e-9
    assert res.passed_mean_delta


def test_tri_criterion_mean_delta_below_threshold_fails_a() -> None:
    ssl = [_fhr(f, "30m", "ssl_heads", 0.004) for f in ("F-9", "F-10", "F-11", "F-12")]
    xgb = [_fhr(f, "30m", "xgboost",   0.000) for f in ("F-9", "F-10", "F-11", "F-12")]
    res = evaluate_horizon_gate(horizon="30m", ssl_results=ssl, xgb_results=xgb)
    assert not res.passed_mean_delta
    assert not res.passed


def test_tri_criterion_lucky_fold_dominates_mean_but_fails_monotonicity() -> None:
    """Mean = 0.02 (passes a) but only 1 fold positive (fails b/c).

    This is the architect-review §A.7 single-fold-dominance pathology
    the tri-criterion was DESIGNED to catch.
    """
    ssl = [
        _fhr("F-9",  "30m", "ssl_heads", 0.085),    # one huge lucky fold
        _fhr("F-10", "30m", "ssl_heads", -0.001),
        _fhr("F-11", "30m", "ssl_heads", -0.002),
        _fhr("F-12", "30m", "ssl_heads", -0.002),
    ]
    xgb = [_fhr(f, "30m", "xgboost", 0.000) for f in ("F-9", "F-10", "F-11", "F-12")]
    res = evaluate_horizon_gate(horizon="30m", ssl_results=ssl, xgb_results=xgb)
    # Mean delta ≈ 0.02 > 0.005 → passes (a) ...
    assert res.passed_mean_delta
    # ... but only 1 of 4 folds positive → fails (b)/(c) → overall fail.
    assert not res.passed_folds_positive
    assert not res.passed
    # Doc the catch: this is exactly what amendment 8 was designed to prevent.


def test_tri_criterion_empty_results_returns_failure() -> None:
    res = evaluate_horizon_gate(horizon="30m", ssl_results=[], xgb_results=[])
    assert res.n_folds == 0
    assert not res.passed


# ----- §1.4 random-init gate -----


def test_random_init_takes_best_of_seeds() -> None:
    """Best-of-seeds per fold, then mean across folds, then delta ≥0.005."""
    ssl = [_fhr(f, "30m", "ssl_heads", 0.015)
           for f in ("F-9", "F-10", "F-11", "F-12")]
    # Random-init: 3 seeds × 4 folds. The MAX seed per fold is what counts.
    rand = []
    for f in ("F-9", "F-10", "F-11", "F-12"):
        for s, ic in zip((0, 1, 2), (0.005, 0.008, 0.009), strict=True):
            rand.append(_fhr(f, "30m", "random_init", ic, seed=s))
    res = evaluate_random_init_gate(
        horizon="30m", ssl_results=ssl, random_init_results=rand,
    )
    # Per fold best = 0.009; mean = 0.009.
    assert abs(res.mean_random_init_ic - 0.009) < 1e-9
    # SSL mean = 0.015. Delta = 0.006 ≥ 0.005 → passes.
    assert res.passed


def test_random_init_fails_when_delta_below_threshold() -> None:
    ssl = [_fhr(f, "30m", "ssl_heads", 0.012)
           for f in ("F-9", "F-10", "F-11", "F-12")]
    # Best-of-seeds = 0.010; SSL mean = 0.012 → delta 0.002 < 0.005.
    rand = []
    for f in ("F-9", "F-10", "F-11", "F-12"):
        for s in (0, 1, 2):
            rand.append(_fhr(f, "30m", "random_init", 0.010, seed=s))
    res = evaluate_random_init_gate(
        horizon="30m", ssl_results=ssl, random_init_results=rand,
    )
    assert not res.passed


# ----- ≥2-of-4 horizons-passed rule -----


def test_two_of_four_horizons_passes_gate_1() -> None:
    """If 2 horizons pass tri-criterion and 2 fail, gate 1 still passes."""
    ssl: list[FoldHorizonResult] = []
    xgb: list[FoldHorizonResult] = []
    for h, ic in (("30m", 0.020), ("120m", 0.020), ("1d", -0.005), ("2d", -0.005)):
        for f in ("F-9", "F-10", "F-11", "F-12"):
            ssl.append(_fhr(f, h, "ssl_heads", ic))
            xgb.append(_fhr(f, h, "xgboost", 0.000))
    out = evaluate_walk_forward(
        ssl_results=ssl, xgb_results=xgb, random_init_results=[],
    )
    assert out.n_horizons_passed == 2
    assert out.phase3_gate_1_passed
    assert set(out.horizons_passed) == {"30m", "120m"}


def test_one_of_four_horizons_fails_gate_1() -> None:
    ssl: list[FoldHorizonResult] = []
    xgb: list[FoldHorizonResult] = []
    for h, ic in (("30m", 0.020), ("120m", -0.005), ("1d", -0.005), ("2d", -0.005)):
        for f in ("F-9", "F-10", "F-11", "F-12"):
            ssl.append(_fhr(f, h, "ssl_heads", ic))
            xgb.append(_fhr(f, h, "xgboost", 0.000))
    out = evaluate_walk_forward(
        ssl_results=ssl, xgb_results=xgb, random_init_results=[],
    )
    assert out.n_horizons_passed == 1
    assert not out.phase3_gate_1_passed


def test_all_four_horizons_pass() -> None:
    ssl: list[FoldHorizonResult] = []
    xgb: list[FoldHorizonResult] = []
    for h in ("30m", "120m", "1d", "2d"):
        for f in ("F-9", "F-10", "F-11", "F-12"):
            ssl.append(_fhr(f, h, "ssl_heads", 0.020))
            xgb.append(_fhr(f, h, "xgboost", 0.000))
    out = evaluate_walk_forward(
        ssl_results=ssl, xgb_results=xgb, random_init_results=[],
    )
    assert out.n_horizons_passed == 4
    assert out.phase3_gate_1_passed


# ----- §1.3 conformal -----


def test_conformal_gate_in_band() -> None:
    res = evaluate_conformal_gate(horizon="30m", per_fold_coverage=[0.80, 0.79, 0.81, 0.82])
    assert abs(res.mean_coverage_across_folds - 0.805) < 1e-6
    assert res.in_acceptance_band


def test_conformal_gate_below_band() -> None:
    res = evaluate_conformal_gate(horizon="30m", per_fold_coverage=[0.70, 0.72, 0.74, 0.73])
    assert not res.in_acceptance_band


def test_conformal_gate_empty() -> None:
    res = evaluate_conformal_gate(horizon="30m", per_fold_coverage=[])
    assert not res.in_acceptance_band
    assert res.n_folds == 0


# ----- JSON emit + round-trip -----


def test_emit_walk_forward_json_writes_full_schema(tmp_path: Path) -> None:
    """Smoke: emit_walk_forward_json produces a JSON with all top-level keys."""
    ssl, xgb, rand = [], [], []
    for h, ic in (("30m", 0.015), ("120m", 0.015), ("1d", -0.005), ("2d", -0.005)):
        for f in ("F-9", "F-10", "F-11", "F-12"):
            ssl.append(_fhr(f, h, "ssl_heads", ic))
            xgb.append(_fhr(f, h, "xgboost", 0.005))
            for s in range(3):
                rand.append(_fhr(f, h, "random_init", 0.000, seed=s))

    out_path = tmp_path / "phase3_walk_forward.json"
    emit_walk_forward_json(
        ssl_results=ssl, xgb_results=xgb, random_init_results=rand,
        conformal_coverage={"30m": [0.80, 0.81, 0.79, 0.82], "120m": [0.78, 0.80, 0.79, 0.81]},
        fold_metadata=[{"fold_id": f} for f in ("F-9", "F-10", "F-11", "F-12")],
        survivorship_audit=[{"fold_id": "F-9", "n_symbols_at_train_start": 148}],
        output_path=out_path,
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    # Top-level keys per `PHASE3.md` §8.4 schema (post-amendment 16).
    for key in ("schema_version", "generated_at", "provenance",
                "folds", "tri_criterion", "random_init_gates",
                "conformal_gates", "horizons_passed", "n_horizons_passed",
                "phase3_gate_1_passed", "phase3_gate_3_passed",
                "phase3_gate_4_passed", "survivorship_audit"):
        assert key in data, f"missing key {key!r}"

    # Sanity: 30m + 120m pass tri-criterion → horizons_passed has both.
    assert set(data["horizons_passed"]) == {"30m", "120m"}
    assert data["phase3_gate_1_passed"]


def test_emit_json_gate_3_requires_all_passing_horizons_to_be_in_band(tmp_path: Path) -> None:
    """If 30m passes tri-criterion AND its conformal coverage is in band,
    but 120m passes tri-criterion AND its coverage is OUT of band, gate 3
    fails."""
    ssl, xgb, rand = [], [], []
    for h in ("30m", "120m"):
        for f in ("F-9", "F-10", "F-11", "F-12"):
            ssl.append(_fhr(f, h, "ssl_heads", 0.020))
            xgb.append(_fhr(f, h, "xgboost", 0.000))
    # 30m coverage in band; 120m way out of band.
    payload = emit_walk_forward_json(
        ssl_results=ssl, xgb_results=xgb, random_init_results=rand,
        conformal_coverage={"30m": [0.80, 0.80, 0.80, 0.80],
                            "120m": [0.50, 0.50, 0.50, 0.50]},
        output_path=tmp_path / "phase3_walk_forward.json",
    )
    assert not payload.phase3_gate_3_passed


def test_thresholds_are_locked() -> None:
    """Sanity: the magic-number thresholds match PHASE3.md."""
    assert RANK_IC_DELTA_THRESHOLD == 0.005
    assert MIN_FOLDS_POSITIVE == 3
    assert MIN_FOLDS_BEAT_XGB == 3
