"""`daity-eval-tradeable` — Phase 3X eval CLI.

For a saved Phase-3 `*.state.pt` (or Phase-4 OB fine-tune state):
  1. Build a val sampler at the given fold (or arbitrary window).
  2. Forward through the model; collect the **rank head's per-symbol score**
     (proxy for P(good long) — higher score = more likely to be a real long
     candidate).
  3. Look up each row's tradeable label from the v2 parquet store.
  4. Compute precision@k and recall@k for LONG (label=+1) using the rank
     score, and SHORT (label=-1) using the NEGATIVE rank score.
  5. Plot the curves; save JSON of the metric values.

Random-ranker baseline:
  precision@k_random = base rate of target label (≈ 7.8% LONG, 7.4% SHORT).
  recall@k_random    = k / |universe|.

Used to baseline existing Phase-3 / Phase-4 models against the
tradeable-label framing BEFORE we train a dedicated classifier head.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import polars as pl
import torch
from rich.console import Console

from daity.data.cross_section import CrossSection, CrossSectionalSampler
from daity.data.parquet_store import ParquetStore
from daity.eval.classifier_metrics import (
    aggregate_precision_recall_at_k,
    plot_precision_recall_curves,
)
from daity.training.posttrain import PostTrainConfig, PostTrainModule
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _load_module(state_pt: Path) -> tuple[PostTrainModule, dict]:
    """Reload a PostTrainModule from a *.state.pt file (Phase 3 or 4)."""
    payload = torch.load(state_pt, map_location="cpu", weights_only=False)
    module_cfg_dict = dict(payload["module_cfg"])
    if isinstance(module_cfg_dict.get("parent_checkpoint"), str):
        module_cfg_dict["parent_checkpoint"] = Path(module_cfg_dict["parent_checkpoint"])
    # Strip Phase-4-only fields if present.
    valid_keys = {f.name for f in __import__("dataclasses").fields(PostTrainConfig)}
    module_cfg_dict = {k: v for k, v in module_cfg_dict.items() if k in valid_keys}
    cfg = PostTrainConfig(**module_cfg_dict)
    m = PostTrainModule(cfg)
    m.load_state_dict(payload["state_dict"], strict=False)
    m.eval()
    return m, payload.get("run_config", {})


def _load_tradeable_labels_for_symbol(
    tradeable_root: Path, symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a symbol's tradeable label time-series.

    Returns `(t: datetime64[us], label: int8)`.
    """
    path = tradeable_root / f"{symbol}.parquet"
    if not path.exists():
        return (
            np.zeros(0, dtype="datetime64[us]"),
            np.zeros(0, dtype=np.int8),
        )
    df = pl.read_parquet(path).filter(pl.col("valid_label"))
    if df.height == 0:
        return (
            np.zeros(0, dtype="datetime64[us]"),
            np.zeros(0, dtype=np.int8),
        )
    # `t` column already tz-naive UTC.
    t = df["t"].to_numpy().astype("datetime64[us]")
    lab = df["tradeable_label"].to_numpy().astype(np.int8)
    return t, lab


def _label_at(
    t_arr: np.ndarray, lab_arr: np.ndarray, anchor: np.datetime64,
) -> int | None:
    """Exact binary-search lookup. Returns None if anchor not present."""
    if t_arr.size == 0: return None
    pos = int(np.searchsorted(t_arr, anchor, side="left"))
    if pos < t_arr.size and t_arr[pos] == anchor:
        return int(lab_arr[pos])
    return None


@click.command()
@click.option("--state-pt", type=click.Path(path_type=Path), required=True,
              help="Saved Phase-3/4 *.state.pt to evaluate.")
@click.option("--horizon", default="30m", show_default=True,
              type=click.Choice(["30m", "120m", "1d", "2d"]),
              help="Which val sampler to use. 30m matches the v2 label "
                   "trigger-window scale.")
@click.option("--val-start", required=True)
@click.option("--val-end",   required=True)
@click.option("--as-of",     required=True)
@click.option("--tradeable-root", type=click.Path(path_type=Path), required=True,
              help="Path to a tradeable-labels config dir, e.g. "
                   "data/tradeable_labels/v1_trig5_floor3_dmin10_dmax30_k60_5m_<hash>")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True,
              help="Phase-3 forward-return labels (needed to drive sampler).")
@click.option("--max-symbols", type=int, default=None)
@click.option("--max-cross-sections", type=int, default=500, show_default=True)
@click.option("--ks", default="1,3,5,10,20,50", show_default=True,
              help="Comma-separated k values to evaluate.")
@click.option("--output-json", type=click.Path(path_type=Path), default=None)
@click.option("--output-plot", type=click.Path(path_type=Path), default=None)
@click.option("--title", default="", help="Title for plot.")
def main(
    state_pt: Path, horizon: str,
    val_start: str, val_end: str, as_of: str,
    tradeable_root: Path,
    feature_root: Path, label_root: Path,
    max_symbols: int | None, max_cross_sections: int,
    ks: str,
    output_json: Path | None, output_plot: Path | None, title: str,
) -> None:
    """Evaluate rank-head ranking against tradeable labels — precision@k + recall@k."""
    k_list = [int(k.strip()) for k in ks.split(",") if k.strip()]

    # 1. Build samplers.
    feature_store = ParquetStore(feature_root)
    universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if max_symbols is not None:
        universe = universe[:max_symbols]
    sampler = CrossSectionalSampler(
        feature_store=feature_store, label_root=label_root,
        universe=universe, as_of=_parse_dt(as_of),
        min_end_ts=_parse_dt(val_start),
        drop_invalid_threshold=0.3,
    )

    # 2. Pre-load tradeable labels per symbol.
    console.print(f"Loading tradeable labels from {tradeable_root}")
    label_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym in universe:
        label_index[sym] = _load_tradeable_labels_for_symbol(tradeable_root, sym)

    # Base rates (across val window, valid rows only).
    cum_long = cum_short = cum_total = 0
    for sym in universe:
        t_arr, lab_arr = label_index[sym]
        if t_arr.size == 0: continue
        in_window = (t_arr >= np.datetime64(_parse_dt(val_start).replace(tzinfo=None))) & \
                    (t_arr < np.datetime64(_parse_dt(val_end).replace(tzinfo=None)))
        sl = lab_arr[in_window]
        cum_long += int((sl == 1).sum())
        cum_short += int((sl == -1).sum())
        cum_total += sl.size
    base_long = cum_long / max(1, cum_total)
    base_short = cum_short / max(1, cum_total)

    # 3. Load model + run forward, collect per-cs (scores, tradeable_labels).
    console.print(f"Loading model from {state_pt}")
    module, _ = _load_module(state_pt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    long_scores_per_cs: list[np.ndarray] = []
    long_labels_per_cs: list[np.ndarray] = []
    short_scores_per_cs: list[np.ndarray] = []
    short_labels_per_cs: list[np.ndarray] = []
    n_used = 0
    with torch.no_grad():
        for cs in getattr(sampler, f"iter_{horizon}")():
            if n_used >= max_cross_sections: break
            cs_dev = CrossSection(
                t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
                inputs={s: t.to(device) for s, t in cs.inputs.items()},
                labels=cs.labels.to(device), ranks=cs.ranks.to(device),
                valid_mask=cs.valid_mask.to(device),
                vol_bucket=cs.vol_bucket.to(device),
            )
            out = module(cs_dev, horizon=horizon)
            scores = out["rank"].detach().cpu().numpy()
            # Look up tradeable label for each symbol at this anchor.
            anchor = np.datetime64(cs.t.replace(tzinfo=None))
            labs = np.zeros(len(cs.symbols), dtype=np.int8)
            valid = np.zeros(len(cs.symbols), dtype=bool)
            for i, sym in enumerate(cs.symbols):
                t_arr, lab_arr = label_index.get(sym, (np.zeros(0, dtype="datetime64[us]"), np.zeros(0, dtype=np.int8)))
                lab = _label_at(t_arr, lab_arr, anchor)
                if lab is not None:
                    labs[i] = lab
                    valid[i] = True
            # Drop rows with no tradeable label (e.g. end-of-data warmup at val_end).
            if valid.sum() < 10:  # need at least 10 symbols
                continue
            scores_v = scores[valid]
            labs_v = labs[valid]
            long_scores_per_cs.append(scores_v)
            long_labels_per_cs.append(labs_v)
            short_scores_per_cs.append(-scores_v)   # invert for SHORT
            short_labels_per_cs.append(labs_v)
            n_used += 1
    console.print(f"Evaluated on {n_used} cross-sections")

    # 4. Aggregate metrics.
    long_metrics = aggregate_precision_recall_at_k(
        scores_per_cs=long_scores_per_cs, labels_per_cs=long_labels_per_cs,
        ks=k_list, target=+1,
    )
    short_metrics = aggregate_precision_recall_at_k(
        scores_per_cs=short_scores_per_cs, labels_per_cs=short_labels_per_cs,
        ks=k_list, target=-1,
    )

    console.print(f"\n[bold]LONG  (target=+1, base rate={base_long:.3f}):[/bold]")
    for m in long_metrics:
        lift = m.precision / base_long if base_long > 0 else 0
        console.print(f"  k={m.k:>3d}  precision={m.precision:.3f}  "
                      f"recall={m.recall:.3f}  lift={lift:.2f}×  n_cs={m.n_cross_sections}")
    console.print(f"\n[bold]SHORT (target=-1, base rate={base_short:.3f}):[/bold]")
    for m in short_metrics:
        lift = m.precision / base_short if base_short > 0 else 0
        console.print(f"  k={m.k:>3d}  precision={m.precision:.3f}  "
                      f"recall={m.recall:.3f}  lift={lift:.2f}×  n_cs={m.n_cross_sections}")

    from dataclasses import asdict
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({
            "state_pt": str(state_pt),
            "horizon": horizon,
            "val_start": val_start, "val_end": val_end,
            "n_cross_sections": n_used,
            "base_rate_long": base_long,
            "base_rate_short": base_short,
            "long":  [asdict(m) for m in long_metrics],
            "short": [asdict(m) for m in short_metrics],
        }, indent=2))
        console.print(f"  → {output_json}")

    if output_plot is not None:
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_precision_recall_curves(
            long_series=long_metrics, short_series=short_metrics,
            title=title or f"{state_pt.stem} | {horizon} | {val_start}→{val_end}",
            base_rate_long=base_long, base_rate_short=base_short,
            output_path=str(output_plot),
        )
        console.print(f"  → {output_plot}")


if __name__ == "__main__":
    main()
