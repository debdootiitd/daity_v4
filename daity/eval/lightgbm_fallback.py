"""Phase 3.2 — LightGBM stronger-fallback baseline (skeleton).

This module is the **gated escape hatch** for `PHASE3.md` §1's failure-
mode budget: if the SSL+heads model fails the §1.1 / §1.2 tri-criterion
against the default XGBoost baseline by ≤ 0.003, we run this stronger
fallback before declaring failure.

Per `PHASE3.md` §5.6 (architect review B.MINOR.1):

- **LightGBM with `objective='rank_xendcg'`** as the rank-aware head
  (more recent than XGBoost's rank objective; sometimes outperforms).
- **Interaction features** between top-decile importance pairs from
  XGBoost's first-pass fit. Explicit ratios like `vol_z_5m_last /
  vol_z_60m_last` typically add 1-2% IC on cross-sectional tasks.
- **Optuna sweep, 50 trials** over `(num_leaves, learning_rate,
  feature_fraction, bagging_fraction, min_child_samples,
  lambda_l1, lambda_l2)`. ~4 hours CPU.

This file is a **skeleton** in the Phase 3.2 deliverable: the
acceptance gate that triggers it doesn't fire until §3.7's walk-forward
sweep produces numbers. The skeleton:

1. Defines the `LightGBMFallbackConfig` (knobs) and
   `LightGBMFallbackResult` (outputs).
2. Provides a runtime guard: `lightgbm` and `optuna` are optional
   imports — if either is missing, raises a clear error with the exact
   `uv add` command to install. CI doesn't currently install them, so
   they're imported only when this module is actually used.
3. The `run_fallback(...)` entrypoint is a stub that documents the
   intended flow but currently raises `NotImplementedError`. We'll fill
   it in only if the gate fires (§3.7's failure-margin trigger). No
   point implementing optimization machinery we won't run.

When the gate fires, this is ~1 day of work to wire up: the feature-
flatten layer is shared with `xgboost_baseline.py`, only the model fit
+ Optuna search are new.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Optional deps — guarded; imported lazily on call.
try:
    import lightgbm  # noqa: F401
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    import optuna  # noqa: F401
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class LightGBMFallbackConfig:
    """Knobs for the stronger fallback. Defaults match §5.6 closely."""

    # Number of Optuna trials per (horizon, fold).
    n_trials: int = 50

    # Top-K features to use for interaction-pair generation. After XGBoost's
    # first-pass fit, the top-K most-important features get pairwise ratios
    # added (K * (K-1) / 2 new interactions per scale).
    top_k_for_interactions: int = 12

    # LightGBM rank objective. Other supported: "lambdarank", "rank_xentropy".
    rank_objective: str = "rank_xendcg"

    # Inner-CV folds for the Optuna study (TimeSeriesSplit).
    inner_cv_folds: int = 5

    # Random seed for reproducibility.
    seed: int = 0


@dataclass(slots=True)
class LightGBMFallbackResult:
    """Output of one fallback run; mirrors `FoldResult` for tearsheet
    compatibility but adds best-trial metadata.
    """
    horizon: str
    fold_id: str
    rank_ic_mean: float
    rank_ic_std: float
    decile_spread_bps: float
    n_trials_completed: int
    best_trial_params: dict[str, float | int] = field(default_factory=dict)
    best_trial_score: float = 0.0
    interaction_features_added: int = 0
    notes: str = ""


def _check_runtime() -> None:
    """Raise a clear error if optional deps are missing."""
    missing: list[str] = []
    if not _LIGHTGBM_AVAILABLE:
        missing.append("lightgbm")
    if not _OPTUNA_AVAILABLE:
        missing.append("optuna")
    if missing:
        msg = (
            f"LightGBM fallback requires optional deps: {missing}. "
            f"Install with: `uv add {' '.join(missing)}` (or "
            f"`pip install {' '.join(missing)}`). The fallback is gated "
            f"by `PHASE3.md` §1.1 failure margin ≤ 0.003 — only install "
            f"once the gate fires."
        )
        raise ImportError(msg)


def run_fallback(
    *,
    horizon: str,
    fold_id: str,
    config: LightGBMFallbackConfig | None = None,
    **kwargs: object,
) -> LightGBMFallbackResult:
    """Run the LightGBM-rank + Optuna fallback for one (horizon, fold).

    The intended flow (per `PHASE3.md` §5.6) is:

    1. Reuse `xgboost_baseline._build_train_test_xy` to assemble the
       457-feature design matrix.
    2. Fit a quick XGBoost first-pass to identify top-K important
       features; emit pairwise-ratio interactions of those.
    3. Concatenate base features + interaction features into the
       LightGBM input (~457 + ~K*(K-1)/2 ≈ 530 features at K=12).
    4. Run an Optuna study (TPE sampler, MedianPruner) with N_trials =
       `config.n_trials` over the standard LightGBM-rank hyperparam
       grid. Inner CV uses `TimeSeriesSplit(n_splits=config.inner_cv_folds)`.
       Objective: maximize cross-sectional rank-IC on the inner-val
       fold (using `daity.eval.metrics.cross_sectional_rank_ic`).
    5. Refit on the full train window with the best trial's params.
    6. Predict on the test window; report rank-IC + decile spread.

    Currently a stub — the §1 acceptance gate hasn't fired so we
    deliberately don't ship the implementation. This will be filled
    in only if the trigger condition is met during Phase 3.7's
    walk-forward sweep.
    """
    if config is None:
        config = LightGBMFallbackConfig()
    _check_runtime()
    msg = (
        "LightGBM fallback is a skeleton in Phase 3.2. The full "
        "implementation will land if `PHASE3.md` §1.1 / §1.2 fail the "
        "tri-criterion by a margin ≤ 0.003 against the XGBoost baseline. "
        "See module docstring for the intended flow."
    )
    raise NotImplementedError(msg)


def is_available() -> bool:
    """True iff `lightgbm` + `optuna` are importable.

    Used by the walk-forward CLI to decide whether to *attempt* the
    fallback or skip it (with a one-line note in the tearsheet).
    Returning False here is non-fatal — the fallback is always optional.
    """
    return _LIGHTGBM_AVAILABLE and _OPTUNA_AVAILABLE
