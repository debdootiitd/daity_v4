"""`daity-conformal-calibrate` — Phase 3.6 split-conformal calibration.

For each fold's saved LoRA `state.pt`:
- reload `PostTrainModule`,
- iterate the val sampler per horizon,
- collect `(q_pred[5], y, vol_bucket)` per row,
- split 50/50 chronologically into calibration + test,
- calibrate per `(horizon, vol_bucket)` cells (CQR vs vanilla picker),
- compute coverage on the test half,
- emit per-horizon coverage to a JSON suitable for `daity-walk-forward`.

Output schema (matches `--conformal-coverage-json` input of §3.7 CLI):
    {"30m": [cov_F9, cov_F10, cov_F11, cov_F12],
     "120m": [...], "1d": [...], "2d": [...]}

The driver is INFERENCE-ONLY: loads `state.pt`, no training. GPU is
needed for the forward pass; calibration + coverage are CPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import torch
from rich.console import Console

from daity.data.cross_section import CrossSection, CrossSectionalSampler
from daity.data.labels import ALL_HORIZONS as HORIZONS
from daity.data.parquet_store import ParquetStore
from daity.models.conformal import (
    calibrate_per_cell,
    empirical_coverage,
    predict_band,
)
from daity.training.posttrain import PostTrainConfig, PostTrainModule
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _load_module_from_state(state_pt_path: Path) -> tuple[PostTrainModule, dict]:
    """Reload a trained PostTrainModule + return its saved run_config."""
    payload = torch.load(state_pt_path, map_location="cpu", weights_only=False)
    module_cfg_dict = payload["module_cfg"]
    if "parent_checkpoint" in module_cfg_dict and isinstance(
        module_cfg_dict["parent_checkpoint"], str
    ):
        module_cfg_dict["parent_checkpoint"] = Path(module_cfg_dict["parent_checkpoint"])
    module_cfg = PostTrainConfig(**module_cfg_dict)
    module = PostTrainModule(module_cfg)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module, payload["run_config"]


def _collect_predictions(
    module: PostTrainModule,
    val_sampler: CrossSectionalSampler,
    horizon: str,
    *,
    device: torch.device,
    limit_cross_sections: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on the val sampler for one horizon.

    Returns `(q_pred, y, vol_bucket)`:
      - q_pred:     (N_rows, 5) — full 5-quantile output
      - y:          (N_rows,)   — labels
      - vol_bucket: (N_rows,)   — per-row volatility quintile (K=5)
    """
    q_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    vb_chunks: list[np.ndarray] = []
    n = 0
    with torch.no_grad():
        for cs in getattr(val_sampler, f"iter_{horizon}")():
            if limit_cross_sections is not None and n >= limit_cross_sections:
                break
            inputs_on_device = {sc: t.to(device) for sc, t in cs.inputs.items()}
            cs_dev = CrossSection(
                t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
                inputs=inputs_on_device,
                labels=cs.labels.to(device),
                ranks=cs.ranks.to(device),
                valid_mask=cs.valid_mask.to(device),
                vol_bucket=cs.vol_bucket.to(device),
            )
            out = module(cs_dev, horizon=horizon)
            q = out["quantile"].detach().float().cpu().numpy()    # (B, 5)
            q_chunks.append(q)
            y_chunks.append(cs.labels.numpy())
            vb_chunks.append(cs.vol_bucket.numpy().astype(np.int64))
            n += 1
    if not q_chunks:
        return np.zeros((0, 5)), np.zeros(0), np.zeros(0, dtype=np.int64)
    return (
        np.concatenate(q_chunks, axis=0),
        np.concatenate(y_chunks, axis=0),
        np.concatenate(vb_chunks, axis=0),
    )


def _chronological_split(
    n_rows_per_cs: list[int], split_frac: float = 0.5,
) -> tuple[int, int]:
    """Return `(end_of_cal, start_of_test) = (k, k)` such that the first k
    rows go to calibration and the rest go to test. Split honors
    cross-section boundaries so we don't tear a single time slice in half.
    """
    total = sum(n_rows_per_cs)
    target = int(total * split_frac)
    cum = 0
    for i, n in enumerate(n_rows_per_cs):
        cum += n
        if cum >= target:
            return cum, cum   # split between cs i and i+1
    return total, total


def _calibrate_and_score_one_horizon(
    q_pred: np.ndarray, y: np.ndarray, vol_bucket: np.ndarray,
    horizon: str, alpha: float = 0.2, split_frac: float = 0.5,
) -> dict:
    """Return per-horizon calibration + OOS coverage summary."""
    n = q_pred.shape[0]
    if n < 200:
        return {"horizon": horizon, "n_total": n, "coverage": 0.0,
                "error": "insufficient data"}
    split = int(n * split_frac)
    cal_q, test_q = q_pred[:split], q_pred[split:]
    cal_y, test_y = y[:split], y[split:]
    cal_vb, test_vb = vol_bucket[:split], vol_bucket[split:]

    cells = calibrate_per_cell(
        q_pred=cal_q, y=cal_y, vol_bucket=cal_vb,
        horizon=horizon, alpha=alpha,
    )
    if not cells:
        return {"horizon": horizon, "n_total": n, "coverage": 0.0,
                "error": "no cells calibrated"}
    band_lo, band_hi = predict_band(
        q_pred=test_q, vol_bucket=test_vb, cells=cells,
    )
    cov = empirical_coverage(band_lo=band_lo, band_hi=band_hi, y=test_y)
    return {
        "horizon": horizon,
        "n_total": int(n),
        "n_calibration": int(split),
        "n_test": int(n - split),
        "coverage": float(cov),
        "target_coverage": 1.0 - alpha,
        "n_cells": len(cells),
        "cells": {str(b): asdict(c) for b, c in cells.items()},
    }


@click.command()
@click.option("--state-pt", "state_pts", type=click.Path(path_type=Path),
              multiple=True, required=True,
              help="Saved LoRA state.pt files. Repeat once per fold.")
@click.option("--fold-id", "fold_ids", type=str, multiple=True, required=True,
              help="Fold ids matching state.pt order (e.g. F-9 F-10 F-11 F-12).")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--max-symbols", type=int, default=None)
@click.option("--limit-cross-sections", type=int, default=2000, show_default=True,
              help="Cap per-horizon cs count (mirror posttrain eval default).")
@click.option("--alpha", type=float, default=0.2, show_default=True,
              help="Conformal miscoverage level (alpha=0.2 -> 80%% band).")
@click.option("--output-coverage", type=click.Path(path_type=Path),
              default=Path("reports/phase3_conformal_coverage.json"), show_default=True)
@click.option("--output-details", type=click.Path(path_type=Path),
              default=Path("reports/phase3_conformal_details.json"), show_default=True)
def main(
    state_pts: tuple[Path, ...], fold_ids: tuple[str, ...],
    feature_root: Path, label_root: Path,
    max_symbols: int | None, limit_cross_sections: int,
    alpha: float, output_coverage: Path, output_details: Path,
) -> None:
    """Phase 3.6 conformal calibration driver for all walk-forward folds."""
    if len(state_pts) != len(fold_ids):
        raise click.UsageError(
            f"--state-pt and --fold-id must have same count: "
            f"got {len(state_pts)} states, {len(fold_ids)} fold-ids"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_store = ParquetStore(feature_root)
    universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if max_symbols is not None:
        universe = universe[:max_symbols]

    # Coverage by horizon -> list of per-fold coverages (one per fold).
    coverage_by_h: dict[str, list[float]] = {h: [] for h in HORIZONS}
    details: list[dict] = []

    for state_pt, fold_id in zip(state_pts, fold_ids, strict=True):
        console.print(f"\n[bold]{fold_id}[/bold] state={state_pt}")
        module, run_cfg = _load_module_from_state(state_pt)
        module = module.to(device)

        val_start = _parse_dt(run_cfg["val_start"])
        val_end   = _parse_dt(run_cfg["val_end"])
        as_of     = _parse_dt(run_cfg["as_of"])

        val_sampler = CrossSectionalSampler(
            feature_store=feature_store, label_root=label_root,
            universe=universe,
            as_of=as_of,                   # = val_end; caps inputs + labels
            min_end_ts=val_start,
        )

        fold_summary: dict[str, dict] = {}
        for h in HORIZONS:
            q_pred, y, vb = _collect_predictions(
                module, val_sampler, h, device=device,
                limit_cross_sections=limit_cross_sections,
            )
            res = _calibrate_and_score_one_horizon(
                q_pred=q_pred, y=y, vol_bucket=vb, horizon=h, alpha=alpha,
            )
            fold_summary[h] = res
            coverage_by_h[h].append(res["coverage"])
            console.print(
                f"  {h:>5s}  n={res.get('n_total',0):>5d}  "
                f"coverage={res.get('coverage', 0):.3f}  "
                f"target={res.get('target_coverage', 1-alpha):.3f}  "
                f"n_cells={res.get('n_cells', 0)}"
            )
        details.append({
            "fold_id": fold_id,
            "state_pt": str(state_pt),
            "val_start": run_cfg["val_start"],
            "val_end":   run_cfg["val_end"],
            "horizons": fold_summary,
        })

        # Free GPU memory before next fold.
        del module
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    output_coverage.parent.mkdir(parents=True, exist_ok=True)
    output_coverage.write_text(json.dumps(coverage_by_h, indent=2))
    output_details.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "folds": details,
    }, indent=2, default=str))

    console.print(f"\n[bold]Coverage summary across {len(state_pts)} folds:[/bold]")
    for h, covs in coverage_by_h.items():
        mean_cov = sum(covs) / len(covs) if covs else 0.0
        in_band = 0.75 <= mean_cov <= 0.85
        status = "[green]PASS[/green]" if in_band else "[red]FAIL[/red]"
        console.print(
            f"  {h:>5s}  mean_coverage={mean_cov:.3f}  per_fold={[round(c,3) for c in covs]}  {status}"
        )
    console.print(f"\n  coverage -> {output_coverage}")
    console.print(f"  details  -> {output_details}")


if __name__ == "__main__":
    main()
