"""`daity-pretrain` — CLI entry point for the Phase 2 SSL pretraining run.

Wires `OHLCVPretrainDataset` → `DataLoader` → `PretrainModule` →
`lightning.Trainer.fit`. The CLI takes a base YAML config (default:
`configs/train/pretrain.yaml`) and a handful of CLI flags so smoke tests
can override `--max-steps`, `--d-model`, etc. without editing the file.

Run:
    daity-pretrain                                    # full A100 run from base config
    daity-pretrain --smoke                            # 100-step CPU smoke
    daity-pretrain --max-steps 1000 --d-model 64      # arbitrary overrides
    daity-pretrain --device mps --precision 32-true   # Mac M-series smoke
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import click
import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from rich.console import Console

from daity.data.bq import BQClient
from daity.data.datasets import DEFAULT_CHANNELS, OHLCVPretrainDataset
from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster
from daity.training.callbacks import (
    EMACallback,
    build_provenance,
    write_provenance_sidecar,
)
from daity.training.pretrain import PretrainConfig, PretrainModule
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _select_device(name: str) -> str:
    """Resolve 'auto' to the best available accelerator."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_universe(parquet_root: Path, universe_cache: Path) -> list[str]:
    """Symbols to train on — the partitions present in the Parquet store.

    Uses the live SymbolMaster as the universe of record but filters to
    only those symbols actually ingested locally (so a missing partition
    doesn't crash the dataloader mid-batch).
    """
    store = ParquetStore(parquet_root)
    on_disk = {p.stem for p in store.list_partitions("day")}
    try:
        cfg = load_env(strict=False)
        sm = SymbolMaster.from_bq(BQClient(cfg), cache_root=universe_cache)
        master_syms = set(sm.symbols)
        usable = sorted(master_syms & on_disk)
    except (EnvError, Exception) as exc:  # noqa: BLE001
        # Offline / no BQ creds: just use whatever's on disk.
        log.warning("Falling back to on-disk symbols (no SymbolMaster): %s", exc)
        usable = sorted(on_disk)
    if not usable:
        msg = (
            f"No symbols found in {parquet_root}. Run `daity-ingest-parquet` first."
        )
        raise click.UsageError(msg)
    return usable


def _build_pretrain_config(yaml_cfg: dict, overrides: dict) -> PretrainConfig:
    """Produce a `PretrainConfig` from the YAML + CLI overrides.

    Casts string scalars to the dataclass field types — PyYAML doesn't
    parse `1e-2` / `5e-4` as floats by default, so we coerce here. Field
    annotations are stored as strings (because `from __future__ import
    annotations`), so we match by name.
    """
    merged = {**yaml_cfg, **overrides}
    # Annotations are stringified; "float" / "int" are the type names we care about.
    fields_map = {f.name: str(f.type) for f in dataclasses.fields(PretrainConfig)}
    pretrain_kwargs: dict = {}
    for name, typ_str in fields_map.items():
        if name not in merged:
            continue
        val = merged[name]
        if isinstance(val, str):
            if "float" in typ_str:
                val = float(val)
            elif "int" in typ_str:
                val = int(val)
        pretrain_kwargs[name] = val
    return PretrainConfig(**pretrain_kwargs)


@click.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path),
              default=Path("configs/train/pretrain.yaml"), show_default=True,
              help="Base YAML config.")
@click.option("--smoke", is_flag=True,
              help="100-step CPU smoke run with a tiny model. "
                   "Overrides max_steps, d_model, n_layers, etc.")
@click.option("--max-steps", type=int, default=None, help="Override max_steps.")
@click.option("--d-model", type=int, default=None, help="Override d_model.")
@click.option("--n-layers", type=int, default=None, help="Override n_layers.")
@click.option("--batch-size", type=int, default=None, help="Override batch_size.")
@click.option("--device", default=None, help="Override device (auto/cpu/mps/cuda).")
@click.option("--precision", default=None,
              help='Override precision ("bf16-mixed" / "32-true" / "16-mixed").')
@click.option("--symbols", default=None,
              help="Comma-separated subset of symbols (default: all on disk).")
@click.option("--allow-dirty", is_flag=True,
              help="Allow run to proceed with uncommitted local changes. "
                   "Without this flag, dirty checkouts hard-fail to enforce "
                   "the CLAUDE.md §6 reproducibility-tag requirement on the "
                   "trained checkpoint (git_sha, config_hash, as_of).")
@click.option("--lr", type=float, default=None, help="Override lr (peak LR).")
@click.option("--log-dir", type=click.Path(path_type=Path), default=None,
              help="Override log_dir (used by LR sweep to give each run its own dir).")
@click.option("--wandb-run-name", default=None,
              help="Override wandb_run_name (for LR sweep tagging).")
def main(config_path: Path, smoke: bool, max_steps: int | None, d_model: int | None,  # noqa: PLR0915
         n_layers: int | None, batch_size: int | None, device: str | None,
         precision: str | None, symbols: str | None, allow_dirty: bool,
         lr: float | None, log_dir: Path | None,
         wandb_run_name: str | None) -> None:
    """Phase 2 SSL pretraining."""
    if not config_path.exists():
        raise click.UsageError(f"Config not found: {config_path}")
    yaml_cfg = yaml.safe_load(config_path.read_text())

    overrides: dict = {}
    if smoke:
        overrides.update(
            max_steps=100, d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
            batch_size=4, num_workers=0,
            precision="32-true", device="cpu",
            wandb_offline=True,
            window_bars={"5m": 64, "15m": 48, "60m": 32, "day": 16},
            warmup_pct=0.10, lr=1e-3,
            # Smoke uses a tiny forecast horizon to keep CPU runs fast.
            forecast_n_patches=2,
            # Contrastive disabled for smoke too (DESIGN amendment 11):
            # adding a constant log(B) to a 100-step CPU loss curve makes
            # the smoke output noisier without buying signal.
            w_contrastive=0.0,
        )
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    if d_model is not None:
        overrides["d_model"] = d_model
    if n_layers is not None:
        overrides["n_layers"] = n_layers
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if device is not None:
        overrides["device"] = device
    if precision is not None:
        overrides["precision"] = precision
    if lr is not None:
        overrides["lr"] = lr
    if log_dir is not None:
        overrides["log_dir"] = str(log_dir)
    if wandb_run_name is not None:
        overrides["wandb_run_name"] = wandb_run_name

    merged = {**yaml_cfg, **overrides}
    pretrain_cfg = _build_pretrain_config(yaml_cfg, overrides)

    # Resolve universe + dataloader.
    parquet_root = Path(merged["parquet_root"])
    universe_cache = Path(merged["universe_path"])
    if symbols:
        universe = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        universe = _resolve_universe(parquet_root, universe_cache)

    train_end = datetime.fromisoformat(str(merged["train_end"])).replace(tzinfo=UTC)
    val_start = datetime.fromisoformat(str(merged["val_start"])).replace(tzinfo=UTC)
    val_end = datetime.fromisoformat(str(merged["val_end"])).replace(tzinfo=UTC)

    # Feature parquet path (Phase 2.3 v3_features). When `feature_root` is
    # set in the YAML, the dataset reads from data/features_parquet/ (18
    # channels per `CHANNEL_ORDER_18`) instead of data/parquet/ (5 raw OHLCV).
    # Channel count must match `cfg.num_channels` (the tokenizer's per-scale
    # Linear is sized to it). Mismatches surface immediately at the first
    # batch via shape errors.
    feature_root_str = merged.get("feature_root")
    if feature_root_str:
        store = ParquetStore(Path(feature_root_str))
        channels = CHANNEL_ORDER_18
        if pretrain_cfg.num_channels != len(channels):
            raise click.UsageError(
                f"feature_root is set but num_channels={pretrain_cfg.num_channels} "
                f"!= {len(channels)} (the precomputed feature channel count). "
                f"Set num_channels={len(channels)} in the YAML."
            )
        console.print(
            f"[bold]Feature mode:[/bold] reading from {feature_root_str} "
            f"with {len(channels)}-channel input"
        )
    else:
        store = ParquetStore(parquet_root)
        channels = DEFAULT_CHANNELS
    window_bars = merged["window_bars"]
    # Forecast horizon in raw bars at the forecast scale: n_patches * patch_len
    # non-overlapping bars after end_ts. The dataset rejects samples that
    # don't have enough future bars within the as_of cap.
    forecast_scale = merged.get("forecast_scale", "5m")
    forecast_n_patches = int(merged.get("forecast_n_patches", 0))
    forecast_n_bars = forecast_n_patches * pretrain_cfg.patch_len
    train_ds = OHLCVPretrainDataset(
        store=store, symbols=universe, as_of=train_end,
        window_bars=window_bars, channels=channels, seed=0,
        forecast_scale=forecast_scale if forecast_n_bars > 0 else None,
        forecast_n_bars=forecast_n_bars,
    )
    val_ds = OHLCVPretrainDataset(
        store=store, symbols=universe, as_of=val_end,
        window_bars=window_bars, channels=channels, seed=1,
        forecast_scale=forecast_scale if forecast_n_bars > 0 else None,
        forecast_n_bars=forecast_n_bars,
        # DESIGN §2.5 — strict walk-forward val: end_ts must be >= val_start
        # so val samples are genuinely after the train cutoff. Without this
        # bound the val_ds samples uniformly from [data_start, val_end] which
        # heavily overlaps train and makes val_loss ≈ train_loss.
        min_end_ts=val_start,
    )
    pin_memory = bool(merged.get("pin_memory", False))
    nw = int(merged.get("num_workers", 0))
    # PyArrow's threadpool is fork-unsafe (the silent wedge we hit on the
    # first H200 attempt with num_workers=4 + default fork). Spawn re-imports
    # the parent state in each worker — slower startup (a few seconds per
    # worker, one-time), but no fork-state inheritance bugs. `persistent_workers=True`
    # amortizes the spawn startup cost across the whole training run.
    train_kwargs: dict = {
        "batch_size": merged["batch_size"],
        "num_workers": nw,
        "pin_memory": pin_memory,
    }
    if nw > 0:
        train_kwargs["multiprocessing_context"] = "spawn"
        train_kwargs["persistent_workers"] = True
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    # Val keeps num_workers=0 deliberately — val passes are short (50 batches
    # via limit_val_batches), spawning workers would cost more than it saves.
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=merged["batch_size"], num_workers=0,
        pin_memory=pin_memory,
    )

    # Build module + trainer.
    module = PretrainModule(pretrain_cfg)
    n_params = sum(p.numel() for p in module.parameters())
    console.print(
        f"[bold]Pretrain config:[/bold] d_model={pretrain_cfg.d_model} "
        f"n_layers={pretrain_cfg.n_layers} max_steps={pretrain_cfg.max_steps:,} "
        f"params={n_params:,}"
    )
    console.print(
        f"[bold]Universe:[/bold] {len(universe)} symbols  "
        f"train_end={train_end.date()}  val=[{val_start.date()}, {val_end.date()}]"
    )

    chosen_device = _select_device(merged.get("device", "auto"))
    console.print(f"[bold]Device:[/bold] {chosen_device}  "
                  f"precision={merged.get('precision', 'bf16-mixed')}")

    log_dir = Path(merged.get("log_dir", "runs/pretrain"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # ----- Provenance triple (CLAUDE.md §6) -----
    # Captured BEFORE training starts and stamped into both a sidecar JSON
    # and the LightningModule's hparams (so it lands inside every checkpoint).
    repo_root = Path(__file__).resolve().parents[2]
    provenance = build_provenance(
        repo=repo_root, merged_cfg=merged, train_end=train_end,
    )
    if provenance.git_dirty and not allow_dirty:
        raise click.UsageError(
            f"Working tree has uncommitted changes (git_sha={provenance.git_sha}). "
            "Re-run with --allow-dirty to override, or commit/stash first. "
            "This guard exists so the (git_sha, config_hash, as_of) triple "
            "stamped on the trained checkpoint is meaningful."
        )
    sidecar = write_provenance_sidecar(provenance, log_dir)
    module.hparams["provenance"] = provenance.to_dict()
    console.print(
        f"[bold]Provenance:[/bold] git_sha={provenance.git_sha[:12]} "
        f"dirty={provenance.git_dirty} config_hash={provenance.config_hash} "
        f"as_of={train_end.date()}  → {sidecar}"
    )

    # ----- Callbacks (Findings 1 + 3 of phase_2_review) -----
    # ModelCheckpoint protects multi-day runs from preemption: top-3 by val
    # loss + a `last.ckpt` sentinel that always points at the latest. Cadence
    # is step-driven (deterministic across resumes) rather than wall-clock
    # minutes — DESIGN §3.2's "every 2 hours" is encoded here as
    # `checkpoint_every_n_steps` (defaults to 2000, ≈ 2h on A100-40G batch 64
    # at observed throughput; H200 sees this more often, which is fine).
    ckpt_dir = log_dir / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="step={step}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=int(merged.get("save_top_k", 3)),
        save_last=True,
        every_n_train_steps=int(merged.get("checkpoint_every_n_steps", 2000)),
        auto_insert_metric_name=False,
    )
    # EMA-of-weights (DESIGN §3.3): used for validation pass + checkpoint
    # selection. Decay comes from the config (defaults 0.999).
    ema_cb = EMACallback(decay=float(merged.get("ema_decay", 0.999)))

    # Gradient clipping: protects training from heavy-tailed losses (real
    # NSE log-return targets occasionally produce huge gradients on outlier
    # batches → forecast head explodes if unclipped). Default 0.0 = no
    # clipping (preserves v1/v2 behavior). Production runs with log-return
    # targets should set this to 1.0 (canonical).
    # Loggers — CSVLogger always on as a local fallback. WandbLogger
    # additionally if the config sets `wandb_project` (Phase 2.5+).
    csv_logger = L.pytorch.loggers.CSVLogger(save_dir=str(log_dir))
    loggers: list = [csv_logger]
    wandb_project = merged.get("wandb_project")
    if wandb_project:
        try:
            from lightning.pytorch.loggers import WandbLogger
            wandb_run_name = merged.get("wandb_run_name") or log_dir.name
            wandb_offline = bool(merged.get("wandb_offline", False))
            wandb_tags = merged.get("wandb_tags", [])
            wb = WandbLogger(
                project=wandb_project,
                name=wandb_run_name,
                save_dir=str(log_dir),
                offline=wandb_offline,
                tags=list(wandb_tags) if wandb_tags else None,
            )
            wb.log_hyperparams({
                "config_hash": provenance.config_hash,
                "git_sha": provenance.git_sha,
                "git_dirty": provenance.git_dirty,
                "n_universe_symbols": len(universe),
                "train_end": str(train_end.date()),
                "val_start": str(val_start.date()),
                "val_end": str(val_end.date()),
            })
            loggers.append(wb)
            console.print(
                f"[bold]W&B:[/bold] project={wandb_project} run={wandb_run_name} "
                f"offline={wandb_offline}"
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning:[/yellow] WandbLogger init failed ({e}); "
                "continuing with CSV only."
            )

    grad_clip = float(merged.get("gradient_clip_val", 0.0))
    trainer_kwargs: dict = {
        "max_steps": pretrain_cfg.max_steps,
        "accelerator": chosen_device,
        "devices": 1,
        "precision": merged.get("precision", "32-true"),
        "val_check_interval": merged.get("val_every_n_steps", 1000),
        "limit_val_batches": merged.get("limit_val_batches", 50),
        "log_every_n_steps": 10,
        "default_root_dir": log_dir,
        "logger": loggers,
        "callbacks": [checkpoint_cb, ema_cb],
        "enable_progress_bar": True,
    }
    if grad_clip > 0:
        trainer_kwargs["gradient_clip_val"] = grad_clip
        trainer_kwargs["gradient_clip_algorithm"] = "norm"   # L2 norm clip
        console.print(
            f"[bold]Gradient clipping:[/bold] L2 norm ≤ {grad_clip} (Trainer-level)"
        )
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    console.print(
        f"[green]Pretrain finished.[/green] Logs at {log_dir}  "
        f"checkpoints at {ckpt_dir}  "
        f"best={Path(checkpoint_cb.best_model_path).name if checkpoint_cb.best_model_path else 'n/a'}"
    )


if __name__ == "__main__":
    main()
