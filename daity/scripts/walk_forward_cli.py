"""`daity-walk-forward` — Phase 3.7 full walk-forward sweep.

Orchestrates the 4-fold × 2-mode (LoRA + random_init × 3 seeds) ×
4-horizon sweep + XGBoost baseline, runs conformal calibration on
each fold's val window, evaluates §1.3 out-of-sample coverage on the
test window, and emits `reports/phase3_walk_forward.json` per
`PHASE3.md` §8.4.

This CLI is intentionally a thin orchestrator on top of:
- `daity.training.posttrain_loop.run_single_fold` (§3.5 — LoRA / random_init)
- `daity.eval.xgboost_baseline.fit_predict_one_fold` (§3.2)
- `daity.models.conformal` (§3.6)
- `daity.eval.walk_forward` (§3.7 — aggregation + tri-criterion)

Compute footprint (post-amendment 16):
- 4 folds × (1 LoRA + 3 random-init seeds) = 16 training runs
- 4 folds × 4 horizons = 16 XGBoost fits
- Per fold: 4 horizons × conformal calibration + OOS coverage
- Estimated H200 wall-clock: ~30K SSL steps × 4 folds + 30K × 12 random-init
  + XGBoost 1.5h ≈ 60-90 H200-hours per PHASE3.md §11.

Run:
    daity-walk-forward --checkpoint <ssl.ckpt>

The default fold set matches `PHASE3.md` §8.1 (F-9..F-12 walk-forward).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from daity.data.parquet_store import ParquetStore
from daity.eval.walk_forward import (
    FoldHorizonResult,
    emit_walk_forward_json,
)
from daity.scripts.xgboost_baseline_cli import _default_folds
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


@click.command()
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True,
              help="SSL checkpoint path.")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--symbols", default=None,
              help="Comma-separated symbols. Default: all in feature store.")
@click.option("--max-symbols", type=int, default=None,
              help="Cap universe size (debug).")
@click.option("--max-steps", type=int, default=30_000, show_default=True,
              help="Posttrain steps per fold × per mode.")
@click.option("--n-random-seeds", type=int, default=3, show_default=True,
              help="Number of random-init seeds for §1.4 best-of comparison.")
@click.option("--ssl-results-json", type=click.Path(path_type=Path), default=None,
              help="Pre-computed SSL per-(fold,horizon) results (skip training).")
@click.option("--xgb-results-json", type=click.Path(path_type=Path), default=None,
              help="Pre-computed XGBoost per-(fold,horizon) results.")
@click.option("--random-init-results-json", type=click.Path(path_type=Path), default=None,
              help="Pre-computed random-init per-(fold,horizon,seed) results.")
@click.option("--conformal-coverage-json", type=click.Path(path_type=Path), default=None,
              help="Pre-computed per-horizon test-fold coverage list.")
@click.option("--report-path", type=click.Path(path_type=Path),
              default=Path("reports/phase3_walk_forward.json"), show_default=True)
def main(
    checkpoint: Path, feature_root: Path, label_root: Path,
    symbols: str | None, max_symbols: int | None,
    max_steps: int, n_random_seeds: int,
    ssl_results_json: Path | None,
    xgb_results_json: Path | None,
    random_init_results_json: Path | None,
    conformal_coverage_json: Path | None,
    report_path: Path,
) -> None:
    """Run the walk-forward sweep + emit acceptance JSON.

    The `--*-results-json` flags let you skip already-run sub-phases and
    plumb in pre-computed numbers. This is how Phase 3.7 will be operated
    in practice: run `daity-posttrain` × 4 folds × 4 modes separately
    (each with its own JSON output), run `daity-xgboost-baseline` once,
    then run THIS CLI with all four `--*-json` flags pointed at the
    per-run outputs to aggregate.
    """
    feature_store = ParquetStore(feature_root)
    if symbols:
        universe = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if not universe:
        raise click.UsageError(
            f"No symbols found at {feature_root}/day/.",
        )
    if max_symbols is not None:
        universe = universe[:max_symbols]

    folds = _default_folds()

    # ----- Load pre-computed results (the orchestrate-by-flags path) -----
    ssl_results = _load_fhr_json(ssl_results_json)
    xgb_results = _load_fhr_json(xgb_results_json)
    random_init_results = _load_fhr_json(random_init_results_json)
    conformal_coverage = (
        json.loads(conformal_coverage_json.read_text())
        if conformal_coverage_json is not None else None
    )

    # ----- Inline-orchestration path NOT YET WIRED -----
    # The full orchestration (4 folds × 16 training runs + 16 XGB fits +
    # conformal per horizon) takes 60-90 H200-hours per PHASE3.md §11.
    # The CLI deliberately keeps the inline path absent so a user can't
    # accidentally launch it on the wrong machine. The H200 workflow is:
    #
    #   for fold in F-9 F-10 F-11 F-12; do
    #     daity-posttrain --mode lora --train-start ... --train-end ... \
    #       --val-start ... --val-end ... --report-path runs/$fold/ssl.json
    #     for seed in 0 1 2; do
    #       daity-posttrain --mode random_init --seed $seed \
    #         --report-path runs/$fold/rand_$seed.json ...
    #     done
    #   done
    #   daity-xgboost-baseline --report-path runs/xgb.json ...
    #   # then aggregate the per-run JSONs into FoldHorizonResult lists
    #   # and pass them to THIS CLI.

    if not ssl_results or not xgb_results:
        raise click.UsageError(
            "Inline orchestration not implemented (cost: 60-90 H200-hours). "
            "Run `daity-posttrain` (×4 folds × {1+3} modes) and "
            "`daity-xgboost-baseline` separately, then re-run this CLI with "
            "--ssl-results-json --xgb-results-json --random-init-results-json "
            "pointed at the aggregated per-run outputs.",
        )

    console.print(
        f"[bold]Walk-forward aggregation:[/bold]\n"
        f"  SSL results:        {len(ssl_results)} (fold, horizon) entries\n"
        f"  XGBoost results:    {len(xgb_results)} entries\n"
        f"  Random-init results: {len(random_init_results)} entries\n"
        f"  Output: {report_path}",
    )

    payload = emit_walk_forward_json(
        ssl_results=ssl_results,
        xgb_results=xgb_results,
        random_init_results=random_init_results,
        conformal_coverage=conformal_coverage,
        fold_metadata=folds,
        parent_checkpoint=checkpoint,
        output_path=report_path,
    )

    # Print per-horizon summary.
    console.print("\n[bold]Per-horizon tri-criterion (post-amendment 17 — t-stat primary):[/bold]")
    for h, gate in payload.tri_criterion.items():
        status = "[green]PASS[/green]" if gate["passed"] else "[red]FAIL[/red]"
        console.print(
            f"  {h:>5s} {status}  "
            f"t_stat={gate.get('delta_t_stat', 0):+.2f}  "
            f"mean_delta={gate['mean_delta']:+.4f}  "
            f"folds_positive={gate['folds_ssl_positive']}/{gate['n_folds']}  "
            f"folds_beat_xgb={gate['folds_ssl_beat_xgb']}/{gate['n_folds']}",
        )

    console.print("\n[bold]Per-horizon §1.4 vs random-init:[/bold]")
    for h, gate in payload.random_init_gates.items():
        status = "[green]PASS[/green]" if gate["passed"] else "[red]FAIL[/red]"
        console.print(
            f"  {h:>5s} {status}  "
            f"t_stat={gate.get('delta_t_stat', 0):+.2f}  "
            f"mean_delta={gate['mean_delta']:+.4f}"
        )

    console.print(
        f"\n[bold]Phase 3 acceptance:[/bold]\n"
        f"  Gate 1 (≥2 horizons pass tri-criterion, t>1.5): "
        f"{'[green]PASS[/green]' if payload.phase3_gate_1_passed else '[red]FAIL[/red]'} "
        f"({payload.n_horizons_passed} of 4 horizons passed)\n"
        f"  Gate 3 (conformal OOS coverage in [0.75, 0.85]): "
        f"{'[green]PASS[/green]' if payload.phase3_gate_3_passed else '[red]FAIL[/red]'}\n"
        f"  Gate 4 (≥2 horizons beat random-init, t>1.5): "
        f"{'[green]PASS[/green]' if payload.phase3_gate_4_passed else '[red]FAIL[/red]'}\n"
        f"  Report → {report_path}",
    )

    # Exit code reflects COMPOUND Phase-3 acceptance (Major-4): all of
    # gate 1 + gate 3 (if conformal coverage was provided) + gate 4 must
    # pass for the project to take a clean "go" decision into Phase 4.
    # If conformal coverage was NOT supplied, gate 3 is excluded from
    # the compound check (the run is gate-1+4 only — still informative).
    needs_gate_3 = payload.conformal_gates and len(payload.conformal_gates) > 0
    accept = (
        payload.phase3_gate_1_passed
        and payload.phase3_gate_4_passed
        and (payload.phase3_gate_3_passed or not needs_gate_3)
    )
    sys.exit(0 if accept else 1)


def _load_fhr_json(path: Path | None) -> list[FoldHorizonResult]:
    """Read a JSON list of `FoldHorizonResult` dicts."""
    if path is None:
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return [FoldHorizonResult(**row) for row in data]


if __name__ == "__main__":
    main()
