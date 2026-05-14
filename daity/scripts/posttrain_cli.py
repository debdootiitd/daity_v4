"""`daity-posttrain` — Phase 3.5 single-fold training CLI.

Runs `run_single_fold` against a user-supplied (or default) fold definition
and emits a JSON summary with per-horizon rank-IC. Two modes:

- `--mode lora` (default): LoRA + heads only, backbone frozen, init from
  the v3_xattn SSL checkpoint.
- `--mode random_init`: full fine-tune of a random backbone, heads only
  (LoRA disabled). Used for the §1.4 hard gate. Run with `--seed 0/1/2`
  for the 3-seed best-of comparison.

The CLI is INTENTIONALLY thin: most of the configuration is locked in
`posttrain_loop.RunConfig` defaults. Override what you need via flags.

Run examples:
    # LoRA mode, full single-fold training:
    daity-posttrain --checkpoint runs/h200_v3_xattn/checkpoints/step=10000-val_loss=4.7956.ckpt \\
        --train-start 2019-01-01 --train-end 2024-12-31 \\
        --val-start  2025-01-01 --val-end   2025-03-31

    # Random-init ablation, seed 1:
    daity-posttrain --mode random_init --seed 1 \\
        --train-start 2019-01-01 --train-end 2024-12-31 \\
        --val-start  2025-01-01 --val-end   2025-03-31
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
from rich.console import Console

from daity.data.parquet_store import ParquetStore
from daity.training.posttrain import PostTrainConfig
from daity.training.posttrain_loop import (
    RunConfig,
    run_single_fold,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


@click.command()
@click.option("--checkpoint", type=click.Path(path_type=Path), required=False, default=None,
              help="SSL checkpoint path (required in mode=lora; ignored in random_init).")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--symbols", default=None,
              help="Comma-separated symbols. Default: all in feature store.")
@click.option("--max-symbols", type=int, default=None,
              help="Cap universe size (debug aid).")
@click.option("--train-start", required=True)
@click.option("--train-end",   required=True)
@click.option("--val-start",   required=True)
@click.option("--val-end",     required=True)
@click.option("--as-of", default=None,
              help="Leakage cap. Default = today + 1d (UTC).")
@click.option("--mode", type=click.Choice(["lora", "random_init"]),
              default="lora", show_default=True)
@click.option("--max-steps", type=int, default=30_000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--report-path", type=click.Path(path_type=Path), default=None,
              help="Where to write the per-horizon JSON summary. "
                   "Default: reports/phase3_posttrain_{mode}_seed{seed}.json")
@click.option("--save-model", type=click.Path(path_type=Path), default=None,
              help="Where to write the trained state_dict + run-config "
                   "tarball for future eval-only runs. Default: same path "
                   "as --report-path with .state.pt extension.")
@click.option("--no-save-model", is_flag=True,
              help="Disable model-state save entirely.")
@click.option("--grad-accum-steps", type=int, default=1, show_default=True,
              help="Gradient accumulation: K cross-sections per opt step. "
                   "Amortizes the ~50ms-per-opt-step Python overhead K-fold "
                   "at the cost of larger effective batch. With K=1 the "
                   "loop is unchanged. Bigger K helps GPU util when the "
                   "bottleneck is per-opt-step overhead at small batch.")
@click.option("--pad-n-for-compile", type=int, default=0, show_default=True,
              help="Pad cross-section N to this fixed value so static-shape "
                   "torch.compile (mode='reduce-overhead') can capture CUDA "
                   "Graphs. Set to universe size (typically 205) to enable. "
                   "0 = disable padding (use eager forward).")
@click.option("--cadence-weights", default=None,
              help="JSON dict of horizon->weight overriding "
                   "DEFAULT_CADENCE_WEIGHTS. Default natural cadence is "
                   "{30m:70,120m:52,1d:1,2d:1} which gives 1d/2d only ~240 "
                   "gradient updates out of 30K. Pass e.g. "
                   "'{\"30m\":30,\"120m\":30,\"1d\":30,\"2d\":30}' for equal "
                   "weighting (each horizon ~7500 updates).")
@click.option("--use-aux-features", is_flag=True,
              help="Phase 3.4b iter-4 — enable late-fusion of top-50 XGB "
                   "features into the FORECAST hidden state. Requires "
                   "--aux-root to point at a pre-built aux-feature parquet "
                   "store (see `daity-precompute-aux-features`).")
@click.option("--aux-root", type=click.Path(path_type=Path), default=None,
              help="Root of the per-symbol aux-feature parquet store "
                   "(`{root}/{symbol}.parquet`, columns anchor_ts + f_0..f_49).")
@click.option("--label-mode", type=click.Choice(["raw", "resid"]),
              default="raw", show_default=True,
              help="Phase 3.4c iter-5 — `resid` uses sector-residualized "
                   "labels for `--resid-horizons` (default 1d, 2d); other "
                   "horizons stay on raw. Build resid labels first with "
                   "`daity-build-residual-labels`.")
@click.option("--label-root-resid", type=click.Path(path_type=Path),
              default=Path("data/labels_resid"), show_default=True,
              help="Root of the per-horizon, per-symbol residualized-label "
                   "parquet store (consumed when --label-mode=resid).")
@click.option("--resid-horizons", default="1d,2d", show_default=True,
              help="Comma-separated horizons that use the resid path "
                   "(others stay raw).")
def main(
    checkpoint: Path | None, feature_root: Path, label_root: Path,
    symbols: str | None, max_symbols: int | None,
    train_start: str, train_end: str, val_start: str, val_end: str,
    as_of: str | None,
    mode: str, max_steps: int, seed: int,
    report_path: Path | None,
    save_model: Path | None, no_save_model: bool,
    grad_accum_steps: int, pad_n_for_compile: int,
    cadence_weights: str | None,
    use_aux_features: bool, aux_root: Path | None,
    label_mode: str, label_root_resid: Path,
    resid_horizons: str,
) -> None:
    """Run one Phase 3.5 single-fold training + emit per-horizon rank-IC."""
    if mode == "lora" and checkpoint is None:
        raise click.UsageError("--checkpoint is required when --mode=lora")
    if mode == "random_init" and checkpoint is None:
        # Build a dummy synth checkpoint that random_init will overwrite
        # anyway. The PostTrainModule still needs a valid file to "load".
        raise click.UsageError(
            "--checkpoint is required (random_init ignores its weights but "
            "still uses it to instantiate the module's shape). For a true "
            "no-checkpoint random_init, run `daity-pretrain` first to write "
            "any valid checkpoint and pass it here."
        )

    feature_store = ParquetStore(feature_root)
    if symbols:
        universe = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if not universe:
        raise click.UsageError(
            f"No symbols found at {feature_root}/day/. Run "
            f"`daity-precompute-features` first."
        )
    if max_symbols is not None:
        universe = universe[:max_symbols]

    as_of_dt = (
        _parse_dt(as_of) if as_of is not None
        else datetime.now(UTC) + timedelta(days=1)
    )

    # Resolve report + save-model paths up front so we can pass save_model
    # into the run config (the loop saves state inside `run_single_fold`).
    if report_path is None:
        report_path = Path(f"reports/phase3_posttrain_{mode}_seed{seed}.json")
    if no_save_model:
        save_model_path: Path | None = None
    elif save_model is not None:
        save_model_path = save_model
    else:
        save_model_path = report_path.with_suffix(".state.pt")

    parsed_cadence = json.loads(cadence_weights) if cadence_weights else None
    if use_aux_features and aux_root is None:
        raise click.UsageError("--use-aux-features requires --aux-root")

    module_cfg = PostTrainConfig(
        parent_checkpoint=checkpoint, use_aux_features=use_aux_features,
    )
    run_cfg_kwargs = dict(
        feature_root=feature_root, label_root=label_root, universe=universe,
        train_start=_parse_dt(train_start), train_end=_parse_dt(train_end),
        val_start=_parse_dt(val_start), val_end=_parse_dt(val_end),
        as_of=as_of_dt,
        mode=mode,
        module_cfg=module_cfg,
        max_steps=max_steps,
        seed=seed,
        save_model_path=save_model_path,
        grad_accum_steps=grad_accum_steps,
        pad_n_for_compile=pad_n_for_compile,
    )
    if parsed_cadence:
        run_cfg_kwargs["cadence_weights"] = parsed_cadence
    if aux_root is not None:
        run_cfg_kwargs["aux_root"] = aux_root
    if label_mode == "resid":
        run_cfg_kwargs["label_mode"] = "resid"
        run_cfg_kwargs["label_root_resid"] = label_root_resid
        run_cfg_kwargs["resid_horizons"] = tuple(
            h.strip() for h in resid_horizons.split(",") if h.strip()
        )
    run_cfg = RunConfig(**run_cfg_kwargs)

    console.print(
        f"[bold]Phase 3.5 posttrain:[/bold] mode={mode} seed={seed} "
        f"max_steps={max_steps}\n"
        f"  feature_root: {feature_root}\n"
        f"  label_root:   {label_root}\n"
        f"  universe:     {len(universe)} symbols\n"
        f"  train:        {train_start} → {train_end}\n"
        f"  val:          {val_start} → {val_end}\n"
        f"  as_of:        {as_of_dt.isoformat()}",
    )

    _module, result = run_single_fold(run_cfg)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": result.mode,
        "seed": result.seed,
        "final_step": result.final_step,
        "per_horizon_rank_ic": result.per_horizon_rank_ic,
        "per_horizon_n_cross_sections": result.per_horizon_n_cross_sections,
        "final_loss_by_horizon": result.final_loss_by_horizon,
        "run_config": {
            "max_steps": run_cfg.max_steps,
            "train_start": train_start, "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "as_of": as_of_dt.isoformat(),
            "n_symbols": len(universe),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str))

    console.print(
        f"\n[bold]Done.[/bold]  Report → {report_path}\n"
        f"[bold]Per-horizon rank-IC:[/bold]",
    )
    for h, ic in result.per_horizon_rank_ic.items():
        n = result.per_horizon_n_cross_sections[h]
        console.print(f"  {h:>5s}: rank_ic={ic:+.4f}  (n_cross_sections={n})")

    sys.exit(0)


if __name__ == "__main__":
    main()
