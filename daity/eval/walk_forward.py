"""Phase 3.7 — walk-forward sweep aggregator + tri-criterion gate.

Aggregates per-fold per-horizon results from §3.5 (`run_single_fold` in
LoRA + random-init modes) and §3.2 (XGBoost baseline) and applies the
Phase 3 acceptance rules from `PHASE3.md` §1 (post-amendment 8 + 16):

- §1.1 / §1.2: per-horizon tri-criterion vs XGBoost.
  Pass requires ALL three of:
    (a) mean rank-IC across 4 folds beats XGBoost mean by ≥0.005
    (b) ≥3 of 4 folds individually positive on rank-IC
    (c) ≥3 of 4 folds individually beat XGBoost on rank-IC
- §1.2 horizon-set: ≥2 of 4 horizons pass tri-criterion.
- §1.4 random-init: ≥2 of 4 horizons beat random-init (best of 3 seeds) by ≥0.005.

Emits `reports/phase3_walk_forward.json` per `PHASE3.md` §8.4 schema.

The MODULE is the decision-rule + aggregation engine; the actual
training runs happen in §3.5's `run_single_fold` and §3.2's
`fit_predict_one_fold`. The §3.7 CLI orchestrates them.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from daity.data.labels import ALL_HORIZONS as HORIZONS   # torch-free import (Major-3)
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Acceptance-gate thresholds — locked to PHASE3.md §1 + amendments 8, 17.
RANK_IC_DELTA_THRESHOLD: float = 0.005      # kept for diagnostic continuity
DELTA_T_STAT_THRESHOLD: float = 1.5         # PHASE3.md amendment 17 — primary
                                            # criterion replacing the fixed
                                            # `Δ ≥ 0.005` margin (the latter
                                            # is statistically toothless at
                                            # cross-fold std 0.007-0.017 with
                                            # n=4; see `reports/reviews/
                                            # phase_3_xgboost_baseline_review.md`
                                            # §D.2).
MIN_FOLDS_POSITIVE: int = 3                 # tri-criterion (b) at 4 folds
MIN_FOLDS_BEAT_XGB: int = 3                 # tri-criterion (c) at 4 folds
MIN_HORIZONS_PASSED: int = 2                # ≥2 of 4 per §1.2
N_RANDOM_INIT_SEEDS: int = 3                # §1.4 best-of-3


def _t_stat(values: list[float]) -> float:
    """One-sample t-statistic of `values` against zero.

    `t = mean / (std / sqrt(n))` with `std` the sample (Bessel) std.
    Degenerate cases:
    - `n < 2`: returns 0.0 (no SE estimate possible).
    - `std == 0`: returns +inf / -inf with the sign of `mean`, or 0.0 if
      `mean == 0` (perfectly degenerate).
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        if mean > 0: return float("inf")
        if mean < 0: return float("-inf")
        return 0.0
    se = std / math.sqrt(n)
    return mean / se


# ----- Inputs from §3.2 / §3.5 -----


@dataclass(frozen=True, slots=True)
class FoldHorizonResult:
    """One (fold, horizon) outcome from a single training run.

    The `mode` field distinguishes which model produced this number:
    `ssl_heads` from `run_single_fold(mode='lora')`,
    `xgboost` from `xgboost_baseline.fit_predict_one_fold`,
    `random_init` from `run_single_fold(mode='random_init', seed=s)`.
    """
    fold_id: str
    horizon: str
    mode: Literal["ssl_heads", "xgboost", "random_init"]
    seed: int                       # only meaningful for random_init
    rank_ic_mean: float
    n_cross_sections: int


# ----- Tri-criterion -----


@dataclass(slots=True)
class HorizonGateResult:
    """Per-horizon tri-criterion outcome (`PHASE3.md` §1.1 / §1.2 / §8.1,
    amendment 17).

    Criterion (a) is now the t-stat test, NOT the fixed `Δ ≥ 0.005` margin
    (amendment 17 — that margin was statistically toothless at our 4-fold
    std). `mean_delta` and `passed_mean_delta` are retained as
    DIAGNOSTIC fields so the report shows both views.
    """
    horizon: str
    mean_ssl_ic: float
    mean_xgb_ic: float
    mean_delta: float                       # ssl - xgb (diagnostic)
    delta_t_stat: float                     # one-sample t-stat of per-fold deltas
    folds_ssl_positive: int                 # of n_folds
    folds_ssl_beat_xgb: int                 # of n_folds
    n_folds: int
    passed_t_stat: bool                     # (a) — primary, amendment 17
    passed_mean_delta: bool                 # (a-diagnostic) — fixed-Δ view
    passed_folds_positive: bool             # (b)
    passed_folds_beat_xgb: bool             # (c)
    passed: bool                            # (a) AND (b) AND (c)


def evaluate_horizon_gate(
    *,
    horizon: str,
    ssl_results: list[FoldHorizonResult],
    xgb_results: list[FoldHorizonResult],
    delta_threshold: float = RANK_IC_DELTA_THRESHOLD,
    t_stat_threshold: float = DELTA_T_STAT_THRESHOLD,
    min_folds_positive: int = MIN_FOLDS_POSITIVE,
    min_folds_beat_xgb: int = MIN_FOLDS_BEAT_XGB,
) -> HorizonGateResult:
    """Apply the per-horizon tri-criterion to one horizon's SSL+heads vs
    XGBoost rank-IC numbers across the walk-forward folds.

    Args expected: `ssl_results` and `xgb_results` are lists of length
    `n_folds` (e.g., 4 for §3.7). Each must carry the same set of
    `fold_id`s; we pair them by fold.
    """
    ssl_by_fold = {r.fold_id: r for r in ssl_results}
    xgb_by_fold = {r.fold_id: r for r in xgb_results}
    fold_ids = sorted(set(ssl_by_fold) & set(xgb_by_fold))
    n_folds = len(fold_ids)
    if n_folds == 0:
        return HorizonGateResult(
            horizon=horizon, mean_ssl_ic=0.0, mean_xgb_ic=0.0, mean_delta=0.0,
            delta_t_stat=0.0,
            folds_ssl_positive=0, folds_ssl_beat_xgb=0, n_folds=0,
            passed_t_stat=False,
            passed_mean_delta=False, passed_folds_positive=False,
            passed_folds_beat_xgb=False, passed=False,
        )

    ssl_ics = [ssl_by_fold[f].rank_ic_mean for f in fold_ids]
    xgb_ics = [xgb_by_fold[f].rank_ic_mean for f in fold_ids]
    deltas = [s - x for s, x in zip(ssl_ics, xgb_ics, strict=True)]

    mean_ssl = sum(ssl_ics) / n_folds
    mean_xgb = sum(xgb_ics) / n_folds
    mean_delta = mean_ssl - mean_xgb
    t_stat = _t_stat(deltas)

    folds_positive = sum(1 for ic in ssl_ics if ic > 0)
    folds_beat = sum(1 for d in deltas if d > 0)

    passed_t = t_stat > t_stat_threshold     # amendment 17 — primary
    passed_a = mean_delta >= delta_threshold  # diagnostic only
    passed_b = folds_positive >= min_folds_positive
    passed_c = folds_beat >= min_folds_beat_xgb
    return HorizonGateResult(
        horizon=horizon, mean_ssl_ic=mean_ssl, mean_xgb_ic=mean_xgb,
        mean_delta=mean_delta, delta_t_stat=t_stat,
        folds_ssl_positive=folds_positive, folds_ssl_beat_xgb=folds_beat,
        n_folds=n_folds,
        passed_t_stat=passed_t,
        passed_mean_delta=passed_a, passed_folds_positive=passed_b,
        passed_folds_beat_xgb=passed_c,
        passed=passed_t and passed_b and passed_c,
    )


# ----- §1.4 random-init ablation -----


@dataclass(slots=True)
class RandomInitGateResult:
    """Per-horizon §1.4 random-init delta outcome (post-amendment 17).

    Random-init's per-horizon-per-fold rank-IC is the BEST across
    `N_RANDOM_INIT_SEEDS` (amendment 3). The primary acceptance criterion
    is the one-sample t-stat of per-fold deltas > `DELTA_T_STAT_THRESHOLD`
    (1.5). `mean_delta` + `passed_mean_delta` retained as diagnostics.
    """
    horizon: str
    mean_ssl_ic: float
    mean_random_init_ic: float                # best-of-seeds mean across folds
    mean_delta: float
    delta_t_stat: float
    passed_t_stat: bool                       # primary (amendment 17)
    passed_mean_delta: bool                   # diagnostic
    passed: bool                              # = passed_t_stat


def evaluate_random_init_gate(
    *,
    horizon: str,
    ssl_results: list[FoldHorizonResult],
    random_init_results: list[FoldHorizonResult],
    delta_threshold: float = RANK_IC_DELTA_THRESHOLD,
    t_stat_threshold: float = DELTA_T_STAT_THRESHOLD,
) -> RandomInitGateResult:
    """SSL mean rank-IC must beat best-of-seeds random_init by t-stat > 1.5
    (amendment 17). Fixed-Δ result reported as diagnostic.
    """
    # Group random_init by fold, take best across seeds per fold.
    rand_by_fold: dict[str, list[float]] = {}
    for r in random_init_results:
        rand_by_fold.setdefault(r.fold_id, []).append(r.rank_ic_mean)

    ssl_by_fold = {r.fold_id: r.rank_ic_mean for r in ssl_results}
    fold_ids = sorted(set(rand_by_fold) & set(ssl_by_fold))
    if not fold_ids:
        return RandomInitGateResult(
            horizon=horizon, mean_ssl_ic=0.0, mean_random_init_ic=0.0,
            mean_delta=0.0, delta_t_stat=0.0,
            passed_t_stat=False, passed_mean_delta=False, passed=False,
        )

    ssl_ics = [ssl_by_fold[f] for f in fold_ids]
    rand_ics = [max(rand_by_fold[f]) for f in fold_ids]   # best-of-seeds per fold
    deltas = [s - r for s, r in zip(ssl_ics, rand_ics, strict=True)]
    mean_ssl = sum(ssl_ics) / len(fold_ids)
    mean_rand = sum(rand_ics) / len(fold_ids)
    delta = mean_ssl - mean_rand
    t_stat = _t_stat(deltas)
    passed_t = t_stat > t_stat_threshold
    passed_m = delta >= delta_threshold
    return RandomInitGateResult(
        horizon=horizon, mean_ssl_ic=mean_ssl, mean_random_init_ic=mean_rand,
        mean_delta=delta, delta_t_stat=t_stat,
        passed_t_stat=passed_t, passed_mean_delta=passed_m,
        passed=passed_t,    # amendment 17 — t-stat is primary
    )


# ----- Top-level acceptance -----


@dataclass(slots=True)
class WalkForwardOutcome:
    """Full Phase 3 acceptance outcome for the walk-forward sweep."""
    tri_criterion: dict[str, HorizonGateResult]
    random_init_gates: dict[str, RandomInitGateResult]
    horizons_passed: list[str]
    n_horizons_passed: int
    phase3_gate_1_passed: bool                # ≥2 of 4 tri-criterion
    phase3_gate_4_passed: bool                # ≥2 of 4 random-init delta
    # §1.3 (conformal OOS coverage) is computed separately from coverage
    # numbers — see `evaluate_conformal_gate`.


def evaluate_walk_forward(
    *,
    ssl_results: list[FoldHorizonResult],
    xgb_results: list[FoldHorizonResult],
    random_init_results: list[FoldHorizonResult],
    horizons: list[str] | None = None,
    delta_threshold: float = RANK_IC_DELTA_THRESHOLD,
    min_horizons_passed: int = MIN_HORIZONS_PASSED,
) -> WalkForwardOutcome:
    """Apply §1.1 / §1.2 + §1.4 across all horizons."""
    hs = horizons or list(HORIZONS)
    tri: dict[str, HorizonGateResult] = {}
    rand: dict[str, RandomInitGateResult] = {}
    for h in hs:
        ssl_h = [r for r in ssl_results if r.horizon == h]
        xgb_h = [r for r in xgb_results if r.horizon == h]
        rand_h = [r for r in random_init_results if r.horizon == h]
        tri[h] = evaluate_horizon_gate(
            horizon=h, ssl_results=ssl_h, xgb_results=xgb_h,
            delta_threshold=delta_threshold,
        )
        rand[h] = evaluate_random_init_gate(
            horizon=h, ssl_results=ssl_h, random_init_results=rand_h,
            delta_threshold=delta_threshold,
        )

    horizons_passed = [h for h in hs if tri[h].passed]
    rand_passed = sum(1 for h in hs if rand[h].passed)
    return WalkForwardOutcome(
        tri_criterion=tri,
        random_init_gates=rand,
        horizons_passed=horizons_passed,
        n_horizons_passed=len(horizons_passed),
        phase3_gate_1_passed=len(horizons_passed) >= min_horizons_passed,
        phase3_gate_4_passed=rand_passed >= min_horizons_passed,
    )


# ----- §1.3 conformal out-of-sample coverage -----


@dataclass(slots=True)
class ConformalGateResult:
    """`PHASE3.md` §1.3 — out-of-sample coverage averaged across the 4 folds.

    For each horizon that passed §1.1 (`horizons_to_check`), compute the
    mean test-fold 80%-band coverage and require it to fall in [0.75, 0.85].
    """
    horizon: str
    mean_coverage_across_folds: float
    in_acceptance_band: bool
    n_folds: int


def evaluate_conformal_gate(
    *,
    horizon: str,
    per_fold_coverage: list[float],
    lower_bound: float = 0.75,
    upper_bound: float = 0.85,
) -> ConformalGateResult:
    """Mean coverage across the test folds in `[lower_bound, upper_bound]`."""
    if not per_fold_coverage:
        return ConformalGateResult(
            horizon=horizon, mean_coverage_across_folds=0.0,
            in_acceptance_band=False, n_folds=0,
        )
    mean_cov = sum(per_fold_coverage) / len(per_fold_coverage)
    return ConformalGateResult(
        horizon=horizon, mean_coverage_across_folds=mean_cov,
        in_acceptance_band=(lower_bound <= mean_cov <= upper_bound),
        n_folds=len(per_fold_coverage),
    )


# ----- Provenance + JSON serialization -----


def _git_sha(default: str = "unknown") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return default


def _checkpoint_hash(path: Path | None) -> str:
    if path is None or not path.exists():
        return "unknown"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class WalkForwardJSON:
    """Top-level JSON schema for `reports/phase3_walk_forward.json`
    per `PHASE3.md` §8.4 (post-amendment 16 — 4 horizons)."""
    schema_version: int = 1
    generated_at: str = ""
    provenance: dict = field(default_factory=dict)
    folds: list[dict] = field(default_factory=list)
    tri_criterion: dict = field(default_factory=dict)
    random_init_gates: dict = field(default_factory=dict)
    conformal_gates: dict = field(default_factory=dict)
    horizons_passed: list[str] = field(default_factory=list)
    n_horizons_passed: int = 0
    phase3_gate_1_passed: bool = False
    phase3_gate_3_passed: bool = False
    phase3_gate_4_passed: bool = False
    survivorship_audit: list[dict] = field(default_factory=list)


def emit_walk_forward_json(
    *,
    ssl_results: list[FoldHorizonResult],
    xgb_results: list[FoldHorizonResult],
    random_init_results: list[FoldHorizonResult],
    conformal_coverage: dict[str, list[float]] | None = None,
    fold_metadata: list[dict] | None = None,
    survivorship_audit: list[dict] | None = None,
    parent_checkpoint: Path | None = None,
    config_hash: str | None = None,            # Major-2: provenance triple
    as_of: str | None = None,                  # Major-2
    output_path: Path = Path("reports/phase3_walk_forward.json"),
    horizons: list[str] | None = None,
    delta_threshold: float = RANK_IC_DELTA_THRESHOLD,
    min_horizons_passed: int = MIN_HORIZONS_PASSED,
) -> WalkForwardJSON:
    """Compute the full acceptance outcome and write the §8.4 JSON.

    Per PHASE3.md §8.4 (review Major-1), the `folds` payload is the
    full per-(fold, horizon, mode) data array — NOT just fold-window
    metadata. The Phase-5 tearsheet reads this directly. We assemble
    `folds_data` from the three input lists; `fold_metadata` (an
    optional pre-computed array of fold-window dicts) is carried as a
    separate `fold_windows` key.

    Per PHASE3.md §6.10 (review Major-2), provenance is the
    `(git_sha, config_hash, as_of)` triple plus `parent_checkpoint_hash`.

    Per PHASE3.md §1.4 (review Major-4), gate 4 must check that the
    horizons clearing the random-init delta INTERSECT with the horizons
    that passed gate 1 — "the SSL backbone must help at the same
    horizons it produces alpha at".
    """
    outcome = evaluate_walk_forward(
        ssl_results=ssl_results, xgb_results=xgb_results,
        random_init_results=random_init_results,
        horizons=horizons, delta_threshold=delta_threshold,
        min_horizons_passed=min_horizons_passed,
    )

    # §1.3 — only gate horizons that passed §1.1.
    conformal_gates: dict[str, ConformalGateResult] = {}
    if conformal_coverage is not None:
        for h in outcome.horizons_passed:
            conformal_gates[h] = evaluate_conformal_gate(
                horizon=h, per_fold_coverage=conformal_coverage.get(h, []),
            )
    gate_3 = (
        all(g.in_acceptance_band for g in conformal_gates.values())
        if conformal_gates else False
    )

    # Major-4: gate 4 also requires the random-init-clearing horizons to
    # OVERLAP with the gate-1-passing horizons. Per PHASE3.md §1.4
    # ("the set of horizons satisfying §1.4 must overlap with the set
    # satisfying §1.1's tri-criterion — i.e., 'the SSL backbone helps at
    # the same horizons it produces alpha at'").
    random_init_passing_horizons = [
        h for h, g in outcome.random_init_gates.items() if g.passed
    ]
    intersection = set(random_init_passing_horizons) & set(outcome.horizons_passed)
    gate_4 = (
        outcome.phase3_gate_4_passed
        and len(intersection) >= min_horizons_passed
    )

    # Major-1: build the 16-row per-(fold, horizon, mode) folds payload.
    folds_data: list[dict] = []
    for r in (*ssl_results, *xgb_results, *random_init_results):
        folds_data.append(asdict(r))

    # Major-2: full provenance triple.
    provenance = {
        "git_sha": _git_sha(),
        "config_hash": config_hash or "unknown",
        "as_of": as_of or "unknown",
        "parent_checkpoint_hash": _checkpoint_hash(parent_checkpoint),
    }

    payload = WalkForwardJSON(
        schema_version=1,
        generated_at=datetime.now().isoformat(),
        provenance=provenance,
        folds=folds_data,
        tri_criterion={h: asdict(g) for h, g in outcome.tri_criterion.items()},
        random_init_gates={h: asdict(g) for h, g in outcome.random_init_gates.items()},
        conformal_gates={h: asdict(g) for h, g in conformal_gates.items()},
        horizons_passed=outcome.horizons_passed,
        n_horizons_passed=outcome.n_horizons_passed,
        phase3_gate_1_passed=outcome.phase3_gate_1_passed,
        phase3_gate_3_passed=gate_3,
        phase3_gate_4_passed=gate_4,
        survivorship_audit=survivorship_audit or [],
    )
    # Optionally carry the fold-time-window metadata as a separate key
    # (consumed by tearsheets / debug). Not part of the dataclass — written
    # alongside the rest of the payload.
    payload_dict = asdict(payload)
    if fold_metadata:
        payload_dict["fold_windows"] = fold_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload_dict, indent=2, default=str))
    return payload
