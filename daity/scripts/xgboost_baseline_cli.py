"""`daity-xgboost-baseline` — Phase 3.2: produce the XGBoost-on-Phase-1-features
rank-IC numbers that Phase 3's acceptance gate (`PHASE3.md` §1) compares
against.

Output: `reports/phase3_xgboost_baseline.json` with per-horizon × per-fold
rank-IC, top-bottom decile spread, and feature-importance top-50.

The 4 walk-forward folds (F-9..F-12 per PHASE3.md §8.1) are passed in via
`--folds-json` (a precomputed JSON of fold boundaries from Phase 1's
`WalkForwardSplitter`). For Phase 3.2 we hard-code a sensible fold scheme
matching §8.1; Phase 3.7's walk-forward CLI will own the canonical fold list.

Run:
    daity-xgboost-baseline                          # all 4 horizons × all 4 folds
    daity-xgboost-baseline --horizons 30m,1d        # subset
    daity-xgboost-baseline --symbols RELIANCE,TCS   # subset of universe
    daity-xgboost-baseline --max-symbols 50         # cap universe (debug)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import numpy as np
from rich.console import Console

from daity.data.labels import ALL_HORIZONS
from daity.data.parquet_store import ParquetStore
from daity.eval.xgboost_baseline import (
    XGBHyperparams,
    fit_predict_one_fold,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _default_folds() -> list[dict[str, str]]:
    """Phase 3.7 walk-forward fold scheme — 4 most recent folds.

    Mirrors `PHASE3.md` §8.1. Each fold trains on `[2019-01-01, train_end]`,
    validates on `(train_end, val_end]`, tests on `(val_end, test_end]`.
    The XGBoost baseline trains on train+val and tests on test (the val
    is reserved for conformal calibration in §3.6, not used by the
    baseline).
    """
    return [
        {
            "fold_id": "F-9",
            "train_start": "2019-01-01T00:00:00Z",
            "train_end":   "2024-09-30T15:30:00Z",
            "val_end":     "2024-12-31T15:30:00Z",
            "test_start":  "2025-01-01T03:45:00Z",
            "test_end":    "2025-03-31T15:30:00Z",
        },
        {
            "fold_id": "F-10",
            "train_start": "2019-01-01T00:00:00Z",
            "train_end":   "2024-12-31T15:30:00Z",
            "val_end":     "2025-03-31T15:30:00Z",
            "test_start":  "2025-04-01T03:45:00Z",
            "test_end":    "2025-06-30T15:30:00Z",
        },
        {
            "fold_id": "F-11",
            "train_start": "2019-01-01T00:00:00Z",
            "train_end":   "2025-03-31T15:30:00Z",
            "val_end":     "2025-06-30T15:30:00Z",
            "test_start":  "2025-07-01T03:45:00Z",
            "test_end":    "2025-09-30T15:30:00Z",
        },
        {
            "fold_id": "F-12",
            "train_start": "2019-01-01T00:00:00Z",
            "train_end":   "2025-06-30T15:30:00Z",
            "val_end":     "2025-09-30T15:30:00Z",
            "test_start":  "2025-10-01T03:45:00Z",
            "test_end":    "2025-12-31T15:30:00Z",
        },
    ]


@click.command()
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True,
              help="Root of the 18-channel feature parquet store.")
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True,
              help="Root of the per-horizon label store (output of `daity-build-labels`).")
@click.option("--report-path", type=click.Path(path_type=Path),
              default=Path("reports/phase3_xgboost_baseline.json"), show_default=True,
              help="Where to write the per-fold per-horizon results.")
@click.option("--horizons", default=",".join(ALL_HORIZONS), show_default=True,
              help="Comma-separated horizons.")
@click.option("--symbols", default=None,
              help="Comma-separated symbols (default: all on disk).")
@click.option("--max-symbols", type=int, default=None,
              help="Cap the universe size (debug aid).")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--folds-json", type=click.Path(path_type=Path), default=None,
              help="Optional override for fold boundaries (JSON list).")
@click.option("--as-of", default=None,
              help="Hard upper-bound cap on input data. Default = today + 1d (UTC).")
@click.option("--train-anchors-per-day", type=int, default=5, show_default=True,
              help="Sub-sample intraday train anchors to this many per "
                   "(symbol, day). Per review B.3, 5 is the recommended "
                   "default — recovers intra-session feature dynamics that "
                   "1/day under-samples, at modest RAM cost (~8 GB peak). "
                   "Pass 0 or use --no-subsample to disable entirely.")
@click.option("--no-subsample", is_flag=True,
              help="Disable train anchor sub-sampling (needs ~150 GB RAM on "
                   "30m × 6-yr × 205-symbol training).")
def main(
    feature_root: Path, label_root: Path, report_path: Path,
    horizons: str, symbols: str | None, max_symbols: int | None,
    seed: int, folds_json: Path | None, as_of: str | None,
    train_anchors_per_day: int, no_subsample: bool,
) -> None:
    """Run the XGBoost baseline across all (horizon, fold) pairs."""
    horizons_list = [h.strip() for h in horizons.split(",") if h.strip()]
    unknown = [h for h in horizons_list if h not in ALL_HORIZONS]
    if unknown:
        raise click.UsageError(
            f"Unknown horizon(s): {unknown}. Allowed: {list(ALL_HORIZONS)}"
        )

    feature_store = ParquetStore(feature_root)
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        # The day partitions (in raw parquet) are the canonical roster, but
        # for XGBoost we need a feature_parquet partition — use those.
        symbol_list = sorted(p.stem for p in feature_store.list_partitions("day"))
    if not symbol_list:
        raise click.UsageError(
            f"No symbols found at {feature_root}/day/. Run "
            f"`daity-precompute-features` first."
        )
    if max_symbols is not None:
        symbol_list = symbol_list[:max_symbols]

    folds = (
        json.loads(folds_json.read_text())
        if folds_json is not None
        else _default_folds()
    )

    as_of_dt = (
        datetime.fromisoformat(as_of).replace(tzinfo=UTC)
        if as_of is not None
        else datetime.now(UTC) + timedelta(days=1)
    )

    console.print(
        f"[bold]XGBoost baseline:[/bold] "
        f"{len(horizons_list)} horizons × {len(folds)} folds × "
        f"{len(symbol_list)} symbols\n"
        f"  feature_root: {feature_root}\n"
        f"  label_root:   {label_root}\n"
        f"  report_path:  {report_path}\n"
        f"  as_of:        {as_of_dt.isoformat()}"
    )

    hparams = XGBHyperparams()
    effective_subsample = None if no_subsample else max(0, train_anchors_per_day) or None
    results: list[dict] = []
    t0 = time.time()
    for horizon in horizons_list:
        for fold in folds:
            t_fold = time.time()
            train_start = datetime.fromisoformat(fold["train_start"].replace("Z", "+00:00"))
            # train_end intentionally not pulled — XGBoost folds val_end into
            # train (val is reserved for conformal in §3.6, not used here).
            val_end     = datetime.fromisoformat(fold["val_end"].replace("Z", "+00:00"))
            test_start  = datetime.fromisoformat(fold["test_start"].replace("Z", "+00:00"))
            test_end    = datetime.fromisoformat(fold["test_end"].replace("Z", "+00:00"))

            # XGBoost baseline trains on `[train_start, val_end]` (val
            # is folded into train; conformal in §3.6 owns its own split).
            res = fit_predict_one_fold(
                horizon=horizon, fold_id=fold["fold_id"],
                feature_store=feature_store, label_root=label_root,
                symbols=symbol_list,
                train_start=train_start, train_end=val_end,   # val folded in
                test_start=test_start, test_end=test_end,
                as_of=as_of_dt, hparams=hparams, seed=seed,
                train_anchors_per_day=effective_subsample,
            )
            elapsed = time.time() - t_fold
            console.print(
                f"  {horizon:>5s} {fold['fold_id']}: "
                f"rank_ic={res.rank_ic.mean_ic:+.4f} "
                f"(n_cs={res.rank_ic.n_cross_sections}, "
                f"n_obs={res.rank_ic.n_observations})  "
                f"decile_bps={res.decile_spread_bps:+.1f}  "
                f"[{elapsed:.1f}s]"
            )
            results.append({
                "horizon": horizon,
                "fold_id": fold["fold_id"],
                "fold_boundaries": fold,
                "rank_ic": asdict(res.rank_ic),
                "per_cross_section_ic": list(res.per_cross_section_ic),
                "decile_spread_bps": res.decile_spread_bps,
                "feature_importance_top50": res.feature_importance,
                "n_train": res.n_train,
                "n_test": res.n_test,
                "best_hparams": res.best_hparams,
                "best_iteration": res.best_iteration,
                "n_boost_rounds_used": res.n_boost_rounds_used,
                "elapsed_seconds": round(elapsed, 1),
            })

    # Cross-fold summary per horizon (review D.2: gate uses delta-t-stat,
    # so the cross-fold std + t-stat of the mean must be persisted).
    per_horizon: dict[str, dict[str, float]] = {}
    for h in horizons_list:
        h_means = [r["rank_ic"]["mean_ic"] for r in results if r["horizon"] == h]
        if not h_means:
            continue
        arr = np.asarray(h_means, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        # cross-fold t-stat of the mean: mean / (std / sqrt(n_folds))
        t_stat = mean / (std / np.sqrt(arr.size)) if std > 0 else 0.0
        per_horizon[h] = {
            "mean_ic": mean,
            "cross_fold_std": std,
            "cross_fold_t_stat": float(t_stat),
            "folds_positive": int((arr > 0).sum()),
            "n_folds": int(arr.size),
        }

    # Provenance triple per CLAUDE.md "Reproducibility" + review E.1.
    git_sha = "unknown"
    git_dirty = False
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()[:12]
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        git_dirty = bool(dirty)
    except Exception:
        pass

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of_dt.isoformat(),
        "provenance": {
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "as_of": as_of_dt.isoformat(),
            "train_anchors_per_day": effective_subsample,
            "seed": seed,
            "feature_root": str(feature_root),
            "label_root": str(label_root),
        },
        "n_symbols": len(symbol_list),
        "horizons": horizons_list,
        "folds": [f["fold_id"] for f in folds],
        "results": results,
        "per_horizon_summary": per_horizon,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, default=str))

    elapsed_total = time.time() - t0
    console.print(
        f"\n[bold]Done in {elapsed_total:.1f}s.[/bold]  "
        f"Report → {report_path}"
    )

    # Per-horizon cross-fold summary (review D.2 — gate consumes t-stat).
    console.print("\n[bold]Per-horizon cross-fold summary:[/bold]")
    for h in horizons_list:
        s = per_horizon.get(h)
        if s is None:
            continue
        console.print(
            f"  {h:>5s}: mean={s['mean_ic']:+.4f}  "
            f"std={s['cross_fold_std']:.4f}  "
            f"cross_fold_t={s['cross_fold_t_stat']:+.2f}  "
            f"folds_positive={s['folds_positive']}/{s['n_folds']}"
        )

    # Phase 3.2 doesn't gate; we just report.
    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
