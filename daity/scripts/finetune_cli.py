"""`daity-finetune` — Phase 4 OB fine-tune CLI.

One fold at a time. Pass `--fold-id OB-F1|OB-F2|OB-F3|OB-F4` to look up
the train/val dates from `daity.data.splits.OB_WALK_FORWARD_FOLDS`.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console

from daity.data.parquet_store import ParquetStore
from daity.data.splits import OB_WALK_FORWARD_FOLDS
from daity.training.finetune import FineTuneConfig
from daity.training.finetune_loop import FineTuneRunConfig, run_finetune_fold
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


@click.command()
@click.option("--parent-checkpoint", type=click.Path(path_type=Path), required=True,
              help="Phase 3 `*.state.pt` to fine-tune from "
                   "(typically `reports/exp_ushape_F-12_lora_seed0.state.pt`).")
@click.option("--fold-id", required=True,
              type=click.Choice([f.name for f in OB_WALK_FORWARD_FOLDS]),
              help="OB-Fx fold name (defined in daity.data.splits).")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--ob-root", type=click.Path(path_type=Path),
              default=Path("data/ob_parquet"), show_default=True)
@click.option("--max-symbols", type=int, default=None)
@click.option("--max-steps", type=int, default=8000, show_default=True)
@click.option("--gate-zero-steps", type=int, default=1500, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--report-path", type=click.Path(path_type=Path), default=None)
@click.option("--save-model", type=click.Path(path_type=Path), default=None)
@click.option("--no-save-model", is_flag=True)
def main(
    parent_checkpoint: Path, fold_id: str,
    feature_root: Path, label_root: Path, ob_root: Path,
    max_symbols: int | None, max_steps: int, gate_zero_steps: int, seed: int,
    report_path: Path | None,
    save_model: Path | None, no_save_model: bool,
) -> None:
    """Run one Phase 4 OB fine-tune fold + emit twin-eval rank-IC."""
    fold = next((f for f in OB_WALK_FORWARD_FOLDS if f.name == fold_id), None)
    if fold is None:
        raise click.UsageError(f"unknown fold {fold_id!r}")

    feature_store = ParquetStore(feature_root)
    universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if not universe:
        raise click.UsageError(f"no symbols at {feature_root}/day/")
    if max_symbols is not None:
        universe = universe[:max_symbols]

    if report_path is None:
        report_path = Path(f"reports/phase4_finetune_{fold_id}_seed{seed}.json")
    if no_save_model:
        save_model_path = None
    elif save_model is not None:
        save_model_path = save_model
    else:
        save_model_path = report_path.with_suffix(".state.pt")

    module_cfg = FineTuneConfig(parent_checkpoint=parent_checkpoint)
    run_cfg = FineTuneRunConfig(
        fold=fold, feature_root=feature_root, label_root=label_root,
        ob_root=ob_root, universe=universe,
        parent_checkpoint=parent_checkpoint,
        module_cfg=module_cfg,
        max_steps=max_steps, gate_zero_steps=gate_zero_steps, seed=seed,
        save_model_path=save_model_path,
    )

    console.print(
        f"[bold]Phase 4 fine-tune:[/bold] fold={fold_id} seed={seed} steps={max_steps}\n"
        f"  parent_checkpoint: {parent_checkpoint}\n"
        f"  train: {fold.train_start} → {fold.train_end}\n"
        f"  val:   {fold.val_start} → {fold.val_end}\n"
        f"  universe: {len(universe)} symbols\n"
        f"  ob_root: {ob_root}\n",
    )

    _module, result = run_finetune_fold(run_cfg)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fold_id": result.fold_id,
        "final_step": result.final_step,
        "per_horizon_rank_ic_with_ob":    result.per_horizon_rank_ic_with_ob,
        "per_horizon_rank_ic_without_ob": result.per_horizon_rank_ic_without_ob,
        "per_horizon_delta_ic":            result.per_horizon_delta_ic,
        "per_horizon_n_cross_sections":    result.per_horizon_n_cross_sections,
        "final_loss_by_horizon":           result.final_loss_by_horizon,
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str))

    console.print(f"\n[bold]Done.[/bold]  Report → {report_path}")
    console.print(f"[bold]Per-horizon twin-eval:[/bold]")
    for h in result.per_horizon_rank_ic_with_ob.keys():
        w = result.per_horizon_rank_ic_with_ob[h]
        wo = result.per_horizon_rank_ic_without_ob[h]
        d = result.per_horizon_delta_ic[h]
        n = result.per_horizon_n_cross_sections.get(h, 0)
        console.print(f"  {h:>5s}: with_ob={w:+.4f}  without_ob={wo:+.4f}  Δ={d:+.4f}  n_cs={n}")

    sys.exit(0)


if __name__ == "__main__":
    main()
