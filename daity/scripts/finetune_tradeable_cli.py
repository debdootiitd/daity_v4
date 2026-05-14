"""`daity-finetune-tradeable` — Phase 3X CLI.

Trains a TradeableHead on top of a frozen Phase-3 encoder.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console

from daity.data.parquet_store import ParquetStore
from daity.eval.classifier_metrics import (
    PrecisionRecallAtK, plot_precision_recall_curves,
)
from daity.training.tradeable_finetune import (
    TradeableFineTuneConfig, TradeableOBFineTuneConfig,
)
from daity.training.tradeable_finetune_loop import (
    TradeableRunConfig, run_tradeable_finetune,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


@click.command()
@click.option("--parent-checkpoint", type=click.Path(path_type=Path), required=True,
              help="Phase 3 *.state.pt (encoder source).")
@click.option("--parent-ob-checkpoint", type=click.Path(path_type=Path), default=None,
              help="Stage 2: Phase 4 OB-fused *.state.pt. When set, "
                   "TradeableOBFineTuneModule is used (xsattn head only).")
@click.option("--ob-root", type=click.Path(path_type=Path), default=None,
              help="Stage 2: OB parquet root.")
@click.option("--ob-unfreeze",
              type=click.Choice(["none", "fusion", "all"]),
              default="none", show_default=True,
              help="Stage 2 only: encoder unfreeze mode for Phase 4 encoder.")
@click.option("--force-gate-zero", is_flag=True,
              help="Stage 2: force OB fusion gate to zero (ablation = Phase 3).")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--tradeable-root", type=click.Path(path_type=Path), required=True,
              help="Tradeable-labels config dir.")
@click.option("--train-start", required=True)
@click.option("--train-end",   required=True)
@click.option("--val-start",   required=True)
@click.option("--val-end",     required=True)
@click.option("--as-of",       required=True)
@click.option("--max-symbols", type=int, default=None)
@click.option("--max-steps", type=int, default=4000, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--warmup-steps-pct", type=float, default=0.05, show_default=True,
              help="Linear warmup as fraction of max_steps before cosine decay.")
@click.option("--focal-gamma", type=float, default=2.0, show_default=True)
@click.option("--class-alpha", default="1.0,1.0,1.0", show_default=True,
              help="Comma-separated focal alpha for (LONG, NEUTRAL, SHORT).")
@click.option("--unfreeze",
              type=click.Choice(["none", "lora", "all", "encoder_only"]),
              default="none", show_default=True,
              help="Phase 3X-B: encoder unfreeze mode. `none`=head only "
                   "(~300K params), `lora`=head+30m LoRA banks (~3M), "
                   "`all`=everything (~40M), `encoder_only`=tokenizer+"
                   "backbone+LoRA but head FROZEN (use with --head-warmstart "
                   "to fine-tune encoder around a fixed head).")
@click.option("--head-warmstart", type=click.Path(path_type=Path), default=None,
              help="Warm-start TradeableHead from a prior TradeableFineTune "
                   "checkpoint (e.g. head_only state.pt). Required when "
                   "unfreezing the encoder to avoid 'head chasing moving target' "
                   "instability.")
@click.option("--head-hidden", type=int, default=None,
              help="TradeableHead hidden width. Default: d_model (256).")
@click.option("--head-dropout", type=float, default=0.1, show_default=True,
              help="TradeableHead dropout.")
@click.option("--head-arch", type=click.Choice(["mlp", "xsattn"]),
              default="mlp", show_default=True,
              help="Head architecture. `mlp` = legacy per-symbol 2-layer; "
                   "`xsattn` = cross-symbol attention head with CLS + FiLM "
                   "(uses ListNet rank loss).")
@click.option("--head-n-layers", type=int, default=2, show_default=True,
              help="xsattn: number of Pre-LN Transformer blocks.")
@click.option("--head-n-heads", type=int, default=4, show_default=True,
              help="xsattn: number of attention heads.")
@click.option("--head-d-ff", type=int, default=512, show_default=True,
              help="xsattn: feed-forward hidden width per block.")
@click.option("--aux-root", type=click.Path(path_type=Path), default=None,
              help="When set, sampler attaches aux_features per cross-section "
                   "and head_aux_dim must match feature width (50).")
@click.option("--head-aux-dim", type=int, default=0, show_default=True,
              help="xsattn: dim of aux features (50 for Phase 3 XGB features). "
                   "0 disables aux input even if --aux-root is set.")
@click.option("--rank-loss-weight-long",  type=float, default=1.0, show_default=True)
@click.option("--rank-loss-weight-short", type=float, default=1.0, show_default=True)
@click.option("--focal-aux-weight", type=float, default=0.3, show_default=True)
@click.option("--cls-aux-weight",   type=float, default=0.1, show_default=True)
@click.option("--quantile-aux-weight", type=float, default=0.0, show_default=True,
              help="xsattn: pinball-loss weight on 5-quantile prediction of "
                   "forward return. 0 disables (default). Enable in iter4+.")
@click.option("--rank-target-temperature-long",  type=float, default=0.02,
              show_default=True,
              help="Softmax temperature applied to LONG soft target (returns).")
@click.option("--rank-target-temperature-short", type=float, default=0.02,
              show_default=True,
              help="Softmax temperature applied to SHORT soft target (-returns).")
@click.option("--direct-pnl-loss-weight", type=float, default=0.0,
              show_default=True,
              help="Weight for direct-PnL (offline bandit) loss term. "
                   "Requires realized_pnl_long column in tradeable_root parquet.")
@click.option("--direct-pnl-temperature", type=float, default=0.5,
              show_default=True, help="Softmax temperature for direct-PnL policy.")
@click.option("--direct-pnl-top-k", type=int, default=0, show_default=True,
              help="If >0, masks REWARD outside top-K (gradient still flows everywhere).")
@click.option("--direct-pnl-entropy-coef", type=float, default=0.0, show_default=True,
              help="Entropy bonus on the policy softmax — discourages collapse. "
                   "Try 0.01-0.05 (entropy magnitude ~ log(N) ≈ 6 for N=600).")
@click.option("--direct-pnl-uniform-mix", type=float, default=0.0, show_default=True,
              help="Behaviour-policy smoothing: convex-combine policy with "
                   "uniform 1/N. Keeps gradient flowing to all symbols. Try 0.05-0.20.")
@click.option("--direct-pnl-baseline-subtract", is_flag=True, default=False,
              help="REINFORCE control-variate: subtract cross-section mean PnL.")
@click.option("--gradient-clip-val", type=float, default=1.0, show_default=True,
              help="Total-gradient-norm clip. Higher value allows bigger steps "
                   "on informative spike batches.")
@click.option("--weight-decay", type=float, default=1e-2, show_default=True,
              help="AdamW weight decay on head + backbone groups (lora gets 0).")
@click.option("--lr-ratio-lora", type=float, default=1.0/3, show_default=True,
              help="LoRA group LR as fraction of head LR. Default 1/3 (~0.333).")
@click.option("--lr-ratio-backbone", type=float, default=1.0/10, show_default=True,
              help="Backbone group LR as fraction of head LR. Default 1/10 (~0.1).")
@click.option("--lr-min-ratio", type=float, default=0.05, show_default=True,
              help="Final LR (post-cosine-decay) as fraction of peak LR.")
@click.option("--gradient-accumulation-steps", type=int, default=1,
              show_default=True,
              help="Cross-sections sampled per optimizer step (variance ↓ √N).")
@click.option("--wandb-project", default=None,
              help="W&B project name. Enables W&B logging. Requires "
                   "WANDB_API_KEY env var or `wandb login` to be done.")
@click.option("--wandb-run-name", default=None,
              help="W&B run name. Default: auto-generated by W&B.")
@click.option("--wandb-entity", default=None,
              help="W&B entity (team/user). Optional.")
@click.option("--wandb-tags", default=None,
              help="Comma-separated tags for the W&B run.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--report-path", type=click.Path(path_type=Path), default=None)
@click.option("--save-model", type=click.Path(path_type=Path), default=None)
@click.option("--no-save-model", is_flag=True)
@click.option("--plot-path", type=click.Path(path_type=Path), default=None)
@click.option("--eval-every-n-steps", type=int, default=None,
              help="Run intermediate val every N steps; track best checkpoint. "
                   "Default disables intermediate eval.")
@click.option("--best-metric", default="long_precision_at_10", show_default=True,
              help="Metric to track for best-checkpoint: "
                   "{long|short}_precision_at_{k}.")
def main(
    parent_checkpoint: Path,
    parent_ob_checkpoint: Path | None,
    ob_root: Path | None, ob_unfreeze: str, force_gate_zero: bool,
    feature_root: Path, label_root: Path,
    tradeable_root: Path,
    train_start: str, train_end: str, val_start: str, val_end: str, as_of: str,
    max_symbols: int | None, max_steps: int, lr: float, warmup_steps_pct: float,
    focal_gamma: float, class_alpha: str, unfreeze: str,
    head_warmstart: Path | None,
    head_hidden: int | None, head_dropout: float,
    head_arch: str, head_n_layers: int, head_n_heads: int, head_d_ff: int,
    aux_root: Path | None, head_aux_dim: int,
    rank_loss_weight_long: float, rank_loss_weight_short: float,
    focal_aux_weight: float, cls_aux_weight: float, quantile_aux_weight: float,
    rank_target_temperature_long: float, rank_target_temperature_short: float,
    direct_pnl_loss_weight: float, direct_pnl_temperature: float,
    direct_pnl_top_k: int,
    direct_pnl_entropy_coef: float, direct_pnl_uniform_mix: float,
    direct_pnl_baseline_subtract: bool,
    gradient_clip_val: float,
    weight_decay: float,
    lr_ratio_lora: float, lr_ratio_backbone: float,
    lr_min_ratio: float,
    gradient_accumulation_steps: int,
    wandb_project: str | None, wandb_run_name: str | None,
    wandb_entity: str | None, wandb_tags: str | None,
    seed: int,
    report_path: Path | None, save_model: Path | None, no_save_model: bool,
    plot_path: Path | None,
    eval_every_n_steps: int | None, best_metric: str,
) -> None:
    """Train TradeableHead + emit precision/recall@k."""
    alpha = tuple(float(x) for x in class_alpha.split(","))
    if len(alpha) != 3:
        raise click.UsageError("--class-alpha must be 3 floats: LONG,NEUTRAL,SHORT")

    feature_store = ParquetStore(feature_root)
    universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if max_symbols is not None:
        universe = universe[:max_symbols]

    if report_path is None:
        report_path = Path(f"reports/phase3x_tradeable_seed{seed}.json")
    if no_save_model:
        save_model_path = None
    elif save_model is not None:
        save_model_path = save_model
    else:
        save_model_path = report_path.with_suffix(".state.pt")

    if parent_ob_checkpoint is not None:
        module_cfg = TradeableOBFineTuneConfig(
            parent_ob_checkpoint=parent_ob_checkpoint,
            head_n_layers=head_n_layers,
            head_n_heads=head_n_heads,
            head_d_ff=head_d_ff,
            head_dropout=head_dropout,
            head_aux_dim=head_aux_dim,
            head_warmstart_path=head_warmstart,
            focal_gamma=focal_gamma,
            class_alpha=alpha,
            rank_loss_weight_long=rank_loss_weight_long,
            rank_loss_weight_short=rank_loss_weight_short,
            focal_aux_weight=focal_aux_weight,
            cls_aux_weight=cls_aux_weight,
            quantile_aux_weight=quantile_aux_weight,
            rank_target_temperature_long=rank_target_temperature_long,
            rank_target_temperature_short=rank_target_temperature_short,
            direct_pnl_loss_weight=direct_pnl_loss_weight,
            direct_pnl_temperature=direct_pnl_temperature,
            direct_pnl_top_k=direct_pnl_top_k,
            direct_pnl_entropy_coef=direct_pnl_entropy_coef,
            direct_pnl_uniform_mix=direct_pnl_uniform_mix,
            direct_pnl_baseline_subtract=direct_pnl_baseline_subtract,
            unfreeze_mode=ob_unfreeze,
            force_gate_zero=force_gate_zero,
        )
    else:
        module_cfg = TradeableFineTuneConfig(
            parent_checkpoint=parent_checkpoint,
            focal_gamma=focal_gamma,
            class_alpha=alpha,
            unfreeze_mode=unfreeze,
            head_warmstart_path=head_warmstart,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
            head_arch=head_arch,
            head_n_layers=head_n_layers,
            head_n_heads=head_n_heads,
            head_d_ff=head_d_ff,
            head_aux_dim=head_aux_dim,
            rank_loss_weight_long=rank_loss_weight_long,
            rank_loss_weight_short=rank_loss_weight_short,
            focal_aux_weight=focal_aux_weight,
            cls_aux_weight=cls_aux_weight,
            quantile_aux_weight=quantile_aux_weight,
            rank_target_temperature_long=rank_target_temperature_long,
            rank_target_temperature_short=rank_target_temperature_short,
            direct_pnl_loss_weight=direct_pnl_loss_weight,
            direct_pnl_temperature=direct_pnl_temperature,
            direct_pnl_top_k=direct_pnl_top_k,
            direct_pnl_entropy_coef=direct_pnl_entropy_coef,
            direct_pnl_uniform_mix=direct_pnl_uniform_mix,
            direct_pnl_baseline_subtract=direct_pnl_baseline_subtract,
        )
    wandb_tag_tuple = tuple(
        t.strip() for t in (wandb_tags or "").split(",") if t.strip()
    )
    run_cfg = TradeableRunConfig(
        feature_root=feature_root, label_root=label_root,
        tradeable_root=tradeable_root, universe=universe,
        parent_checkpoint=parent_checkpoint,
        parent_ob_checkpoint=parent_ob_checkpoint,
        ob_root=ob_root,
        aux_root=aux_root,
        module_cfg=module_cfg,
        train_start=_parse_dt(train_start), train_end=_parse_dt(train_end),
        val_start=_parse_dt(val_start), val_end=_parse_dt(val_end),
        as_of=_parse_dt(as_of),
        max_steps=max_steps, lr=lr,
        weight_decay=weight_decay,
        warmup_steps_pct=warmup_steps_pct,
        lr_min_ratio=lr_min_ratio,
        lr_ratio_lora=lr_ratio_lora,
        lr_ratio_backbone=lr_ratio_backbone,
        gradient_clip_val=gradient_clip_val,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        save_model_path=save_model_path,
        eval_every_n_steps=eval_every_n_steps,
        best_metric=best_metric,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tag_tuple,
    )

    console.print(
        f"[bold]Phase 3X tradeable-head fine-tune (unfreeze={unfreeze}):[/bold]\n"
        f"  parent: {parent_checkpoint}\n"
        f"  tradeable_root: {tradeable_root}\n"
        f"  train: {train_start} → {train_end}\n"
        f"  val:   {val_start} → {val_end}\n"
        f"  universe: {len(universe)} symbols\n"
        f"  steps={max_steps} lr={lr} γ={focal_gamma} α={alpha}",
    )
    _module, result = run_tradeable_finetune(run_cfg)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "final_step": result.final_step,
        "n_val_cross_sections": result.n_val_cross_sections,
        "base_rate_long":  result.base_rate_long,
        "base_rate_short": result.base_rate_short,
        "final_loss": result.final_loss,
        "long_precision_at_k":  result.long_precision_at_k,
        "long_recall_at_k":     result.long_recall_at_k,
        "short_precision_at_k": result.short_precision_at_k,
        "short_recall_at_k":    result.short_recall_at_k,
    }
    report_path.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[bold]Done.[/bold]  Report → {report_path}")

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        long_series = [
            PrecisionRecallAtK(
                k=k, precision=result.long_precision_at_k[k],
                recall=result.long_recall_at_k[k],
                n_cross_sections=result.n_val_cross_sections,
                n_target_in_universe=0.0,
            )
            for k in sorted(result.long_precision_at_k.keys())
        ]
        short_series = [
            PrecisionRecallAtK(
                k=k, precision=result.short_precision_at_k[k],
                recall=result.short_recall_at_k[k],
                n_cross_sections=result.n_val_cross_sections,
                n_target_in_universe=0.0,
            )
            for k in sorted(result.short_precision_at_k.keys())
        ]
        plot_precision_recall_curves(
            long_series=long_series, short_series=short_series,
            title=f"TradeableHead | val {val_start}→{val_end}",
            base_rate_long=result.base_rate_long,
            base_rate_short=result.base_rate_short,
            output_path=str(plot_path),
        )
        console.print(f"  Plot → {plot_path}")


if __name__ == "__main__":
    main()
