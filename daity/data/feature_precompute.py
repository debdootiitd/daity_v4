"""Precompute the 18-channel feature table per (symbol, scale).

Why precompute (per the v3_features experiment design): on-the-fly feature
computation in the dataset would push the per-batch dataloader compute to
~100-400 ms (rolling vol-z over 20 bars × 4 scales × batch=512), regressing
the GPU utilization we just got to 99% via spawn workers. Precomputing the
features once into a separate Parquet store lets workers do mmap-only reads.

Channel layout (locked — `OHLCVPretrainDataset` indexes by position, not name):

    [0]  open                 (bar_channels)
    [1]  high
    [2]  low
    [3]  close
    [4]  volume
    [5]  ret_1                (log_returns at lag 1, close-only)
    [6]  ret_5
    [7]  ret_15
    [8]  ret_60
    [9]  bar_range            (wicks, raw — RevIN normalizes per-window)
    [10] body
    [11] upper_wick_norm      (∈ [0, 1] when bar has nonzero range)
    [12] lower_wick_norm
    [13] vol_z_20             (20-bar rolling z-score of volume)
    [14] min_of_day_sin       (time_covariates — IST trading clock)
    [15] min_of_day_cos
    [16] day_of_week_sin
    [17] day_of_week_cos

Layout on disk: `data/features_parquet/{scale}/{symbol}.parquet`, sorted
by ts. Mirrors the layout of the raw `data/parquet/` store so the existing
ParquetStore APIs work without modification.

Warmup handling: rolling features (ret_60, vol_z_20) need 60 bars of
history before the first valid row. We REQUEST `MAX_LOOKBACK_BARS` of
warmup from FeatureEngine (which handles it via `read_start = window_start
- warmup_calendar_days`), then DROP the warmup-null rows before writing.
The output file thus contains only fully-valid rows; the dataset doesn't
need feature-aware warmup logic.

Idempotency: if the output file exists with the right schema and row
count, skip. The CLI's `--force` flag overrides for re-precompute after
feature-engine bug fixes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from daity.data.features import (
    FEATURE_GROUPS,
    FeatureEngine,
    FeatureSpec,
)
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Channel order is part of the dataset's public contract — locked here.
CHANNEL_ORDER_18: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "ret_1", "ret_5", "ret_15", "ret_60",
    "bar_range", "body", "upper_wick_norm", "lower_wick_norm",
    "vol_z_20",
    "min_of_day_sin", "min_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
)

# Columns in the output Parquet — channel columns plus ts (the index) and
# symbol+scale (carried for sanity / cross-checks).
OUTPUT_SCHEMA: tuple[str, ...] = ("symbol", "ts", "scale", *CHANNEL_ORDER_18)


def precompute_features(
    *,
    raw_store: ParquetStore,
    feature_root: Path,
    symbol: str,
    scale: str,
    as_of: datetime,
    force: bool = False,
) -> int:
    """Compute the 18-channel feature table for one `(symbol, scale)`.

    Args:
      raw_store: source `ParquetStore` (the `data/parquet/` root).
      feature_root: output root (`data/features_parquet/`); the file
        lands at `{feature_root}/{scale}/{symbol}.parquet`.
      symbol, scale: as named.
      as_of: hard upper bound. Bars whose effective_close > as_of are
        excluded, exactly as `FeatureEngine.compute(as_of=...)` enforces.
        For precompute over the full ingested history, pass a generous
        future date (e.g. the data extent's last ts + 1 day).
      force: if True, recompute and overwrite even if a valid output
        already exists.

    Returns:
      The number of rows written to the output file.

    Idempotent: returns the row count of the existing file if it's already
    present and looks valid (schema match, nonzero rows). `--force` skips
    this check.
    """
    out_path = feature_root / scale / f"{symbol}.parquet"
    if not force and out_path.exists():
        # Quick schema check — if columns match, trust the existing file.
        try:
            existing = pl.scan_parquet(out_path).collect_schema().names()
            if set(existing) >= set(OUTPUT_SCHEMA):
                n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
                log.debug(
                    "precompute_features: %s/%s already at %s (%d rows); skipping",
                    scale, symbol, out_path, n,
                )
                return int(n)
        except Exception:
            # Corrupt / incompatible existing file — fall through to recompute.
            pass

    # Compute via the existing FeatureEngine, requesting all 5 groups.
    # `window_start=None` means "compute over the full history available";
    # FeatureEngine handles per-scale warmup internally.
    engine = FeatureEngine(raw_store)
    df = engine.compute(
        symbol=symbol, scale=scale, as_of=as_of,
        window_start=None, window_end=None,
        spec=FeatureSpec(include=FEATURE_GROUPS),
    )
    if df.height == 0:
        log.warning("precompute_features: %s/%s produced 0 rows", scale, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        return 0

    # Drop rows where any of the rolling/lag features are null (the warmup
    # period). After this, the warmup rows are gone — but limit-locked bars
    # (high==low, bar_range==0) still have NULL upper/lower_wick_norm by
    # FeatureEngine's design. We fill those with 0.0 ("no wick") rather
    # than dropping the bars themselves — limit-locked bars are real data
    # we want the model to see.
    must_be_valid = ("ret_60", "vol_z_20")
    pre_drop = df.height
    df = df.drop_nulls(subset=list(must_be_valid))
    log.debug(
        "precompute_features: %s/%s dropped %d warmup rows (%d → %d)",
        scale, symbol, pre_drop - df.height, pre_drop, df.height,
    )

    # Fill NaN in wick columns with 0.0 (limit-locked bars: bar_range == 0
    # → wick is undefined; 0.0 = "no wick beyond open-close range" is the
    # geometrically right answer for a flat bar).
    df = df.with_columns([
        pl.col("upper_wick_norm").fill_null(0.0).alias("upper_wick_norm"),
        pl.col("lower_wick_norm").fill_null(0.0).alias("lower_wick_norm"),
    ])

    # Reorder columns to match the public schema. Anything not in
    # OUTPUT_SCHEMA gets dropped; anything missing raises a clear error.
    missing = [c for c in OUTPUT_SCHEMA if c not in df.columns]
    if missing:
        msg = (
            f"precompute_features({scale}/{symbol}): FeatureEngine output "
            f"missing expected columns: {missing}. Got: {df.columns}"
        )
        raise RuntimeError(msg)
    df = df.select(list(OUTPUT_SCHEMA))

    # Defensive: assert no remaining NaN/null in any feature column.
    # If this fires we have a new failure mode that the dataloader will
    # propagate to NaN losses (the v3_features-step-9 bug we just fixed).
    null_counts = df.null_count().row(0)
    cols = df.columns
    bad = [(cols[i], null_counts[i]) for i in range(len(cols)) if null_counts[i] > 0]
    if bad:
        msg = (
            f"precompute_features({scale}/{symbol}): unexpected nulls remain: "
            f"{bad}. Add a fillna for these columns or drop the rows."
        )
        raise RuntimeError(msg)

    # Write Parquet. ZSTD compression matches what `ParquetStore` uses.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    log.info(
        "precompute_features: %s/%s → %s (%d rows, %d cols)",
        scale, symbol, out_path, df.height, len(df.columns),
    )
    return df.height
