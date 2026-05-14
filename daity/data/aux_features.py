"""Phase 3.4b — auxiliary features for LoRA late-fusion.

The XGBoost baseline (§3.2) uses 457 hand-crafted aggregations of the 18
backbone channels across 4 scales. XGB beats LoRA on 120m horizon
(Δ −0.0129), suggesting the backbone's tokenizer can't recover all the
information that explicit aggregations carry.

This module exposes the top-50 most-important features (aggregated by
mean `feature_importance_top50` across all 16 fold-horizon XGB runs)
as `TOP_50_FEATURES`, and a builder that emits a per-symbol parquet
of (`anchor_ts`, top-50 columns) ready to be loaded by the cross-section
sampler as a static-covariate prefix for the FORECAST token.

The actual feature-building logic is *re-used* from `xgboost_baseline.py`'s
`flatten_anchors_to_features` — we just select the top-50 columns.

Output format (per symbol):
    `data/aux_features/{symbol}.parquet`:
      anchor_ts   datetime[ms, UTC]
      f_0..f_49   float32 (50 columns; column-order matches `TOP_50_FEATURES`)

`PostTrainModule` projects the 50-dim vector through a small MLP
(`aux_proj: Linear(50, hidden_dim)`) and adds it to the FORECAST token's
hidden state. Late-fusion — no backbone modification.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from daity.data.parquet_store import ParquetStore
from daity.eval.xgboost_baseline import (
    DEFAULT_WINDOW_BARS,
    flatten_anchors_to_features,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Top-50 features by mean importance across all 16 (fold, horizon) XGB runs
# (computed 2026-05-12 from `reports/phase3_xgboost_baseline.json`). The
# ranking is by `sum(feature_importance_top50)` across fold-horizon
# results. Locked here so the LoRA model's aux input is reproducible.
TOP_50_FEATURES: tuple[str, ...] = (
    "15m__min_of_day_sin__std",
    "15m__bar_range__std",
    "5m__low__last5_b0",
    "5m__low__last",
    "5m__vol_z_20__last",
    "5m__upper_wick_norm__std",
    "60m__volume__last",
    "5m__min_of_day_cos__mean",
    "15m__volume__last",
    "15m__high__std",
    "15m__close__std",
    "15m__body__last",
    "15m__bar_range__max",
    "15m__min_of_day_cos__std",
    "day__body__last",
    "5m__volume__last5_b0",
    "5m__day_of_week_cos__mean",
    "5m__volume__last5_b1",
    "60m__ret_15__std",
    "5m__bar_range__trend",
    "60m__lower_wick_norm__last",
    "5m__high__max",
    "60m__low__last",
    "5m__volume__last",
    "15m__min_of_day_sin__mean",
    "day__vol_z_20__last",
    "5m__vol_z_20__min",
    "60m__ret_60__std",
    "5m__day_of_week_cos__std",
    "5m__bar_range__mean",
    "5m__day_of_week_sin__last",
    "5m__day_of_week_sin__min",
    "5m__volume__trend",
    "5m__min_of_day_sin__mean",
    "5m__bar_range__last",
    "15m__ret_1__max",
    "60m__ret_5__std",
    "5m__volume__last5_b3",
    "day__bar_range__last",
    "5m__open__max",
    "15m__open__std",
    "5m__open__last",
    "day__ret_1__min",
    "5m__day_of_week_sin__mean",
    "15m__min_of_day_cos__trend",
    "15m__day_of_week_cos__std",
    "15m__upper_wick_norm__max",
    "15m__day_of_week_sin__mean",
    "60m__close__std",
    "15m__low__std",
)

assert len(TOP_50_FEATURES) == 50, f"expected 50 features, got {len(TOP_50_FEATURES)}"

# Feature columns in canonical order: f_0..f_49.
AUX_FEATURE_COLS: tuple[str, ...] = tuple(f"f_{i}" for i in range(50))


def build_aux_features_for_symbol(
    *,
    feature_store: ParquetStore,
    symbol: str,
    as_of: datetime,
    window_bars: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Build aux-feature matrix for one symbol at every 5m bar's effective
    close (i.e., every possible 30m/120m anchor and every 1d/2d anchor's
    close-of-day).

    The anchor grid is taken from the 5m feature store — this is a
    SUPERSET of every anchor the cross-section sampler will ever query.

    Returns: `pl.DataFrame` with columns `(anchor_ts, f_0, …, f_49)`.
    Rows that lack any of the 50 features (insufficient warmup at one
    scale) are dropped.
    """
    if window_bars is None:
        window_bars = DEFAULT_WINDOW_BARS
    df_5m = feature_store.read(symbol=symbol, scale="5m", as_of=as_of)
    if df_5m.height == 0:
        return pl.DataFrame({"anchor_ts": [], **{c: [] for c in AUX_FEATURE_COLS}})
    # Anchor grid: every 5m bar's effective_close. The flatten function
    # joins backward (strictly less than the anchor), so the bar that
    # closed at anchor_ts is the most recent one we can use.
    anchor_ts = (df_5m["ts"] + timedelta(minutes=5)).to_list()

    flat = flatten_anchors_to_features(
        feature_store=feature_store, symbol=symbol,
        anchor_ts=anchor_ts, as_of=as_of, window_bars=window_bars,
    )
    # Project to the 50 columns; drop rows where any are missing.
    missing = [c for c in TOP_50_FEATURES if c not in flat.columns]
    if missing:
        msg = (
            f"build_aux_features_for_symbol({symbol}): {len(missing)} of 50 "
            f"top features missing from XGB feature matrix; first 3: "
            f"{missing[:3]}"
        )
        raise ValueError(msg)
    select_cols = ["anchor_ts"] + list(TOP_50_FEATURES)
    out = flat.select(select_cols)
    # Rename to canonical f_0..f_49.
    rename_map = {orig: f"f_{i}" for i, orig in enumerate(TOP_50_FEATURES)}
    out = out.rename(rename_map)
    # Cast to float32 for compactness; drop any rows with any NaN.
    out = out.with_columns([
        pl.col(c).cast(pl.Float32) for c in AUX_FEATURE_COLS
    ])
    out = out.drop_nulls()
    return out


def write_aux_features(
    *,
    out_root: Path, feature_store: ParquetStore,
    symbols: list[str], as_of: datetime,
    window_bars: dict[str, int] | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write per-symbol aux-feature parquet to `{out_root}/{symbol}.parquet`.

    Returns: `{symbol: n_rows_written}`.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for sym in symbols:
        out_path = out_root / f"{sym}.parquet"
        if out_path.exists() and not overwrite:
            counts[sym] = 0
            continue
        df = build_aux_features_for_symbol(
            feature_store=feature_store, symbol=sym, as_of=as_of,
            window_bars=window_bars,
        )
        df.write_parquet(out_path)
        counts[sym] = df.height
        log.info("aux_features %s -> %s (%d rows)", sym, out_path, df.height)
    return counts


def load_aux_features(
    *, aux_root: Path, symbol: str, as_of: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one symbol's aux features capped at `as_of`.

    Returns: `(anchor_ts: np.ndarray[datetime64], features: np.ndarray[N,50])`.
    """
    path = aux_root / f"{symbol}.parquet"
    if not path.exists():
        return np.zeros(0, dtype="datetime64[us]"), np.zeros((0, 50), dtype=np.float32)
    df = pl.read_parquet(path)
    # Apply as_of cap.
    as_of_naive = as_of.replace(tzinfo=None) if as_of.tzinfo else as_of
    df = df.filter(pl.col("anchor_ts") < as_of_naive)
    if df.height == 0:
        return np.zeros(0, dtype="datetime64[us]"), np.zeros((0, 50), dtype=np.float32)
    ts = df["anchor_ts"].to_numpy()
    feats = df.select(AUX_FEATURE_COLS).to_numpy().astype(np.float32)
    return ts, feats


def lookup_aux_features_at_ts(
    *, anchor_ts_arr: np.ndarray, features: np.ndarray,
    query_ts: np.datetime64,
) -> np.ndarray | None:
    """Return the row of aux features at `query_ts` (must be in `anchor_ts_arr`).

    Returns `None` if `query_ts` isn't found (e.g., warmup gap).
    """
    if anchor_ts_arr.size == 0:
        return None
    # Binary search for exact match.
    idx = np.searchsorted(anchor_ts_arr, query_ts)
    if idx < anchor_ts_arr.size and anchor_ts_arr[idx] == query_ts:
        return features[idx]
    return None
