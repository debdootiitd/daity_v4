"""Phase 3.2 — XGBoost-on-engineered-features baseline.

The bar to beat for the Phase 3 acceptance gate (`PHASE3.md` §1, §5).
Why this is the right baseline (architect review §C.5, §6):

- It's the strongest cheap baseline for tabular cross-sectional prediction.
- Zero-deep-learning, so any rank-IC delta vs. SSL+heads is direct
  evidence the foundation-model layer added value.
- It shares its feature inputs with the SSL backbone (the 18-channel
  feature engine), so the comparison is apples-to-apples on the
  *information* axis — the only thing that differs is who does the
  representation learning.

Feature flattening (`PHASE3.md` §5.2): for each anchor `(symbol, t)`
we read each scale's 18-channel window-of-K bars ending strictly
before `t` and reduce it to:

- 4 scales × 18 channels × 6 stats = 432 features
  (last, mean, std, min, max, normalized_trend = (last-first)/abs(first))
- Plus the 5 most recent 5m-bar OHLCV values of channels 0..4 = 25 features
  (gives the tree models direct access to the last 5 minutes of price action)

Total: 457 features per `(symbol, t)`.

Implementation notes:

1. **Vectorized per (symbol, scale).** Rather than slice the feature
   parquet for each anchor (`O(N_anchors × K)` reads), we compute
   rolling-K aggregates over the full feature series once
   (`pl.col(...).rolling_*`), then look up the AT-row aggregates at
   each anchor's `effective_close` boundary via `join_asof`. This is
   ~50× faster than naive per-anchor slicing.

2. **Strict `[t-window, t)` semantics** (architect review §5.7 / B.MINOR.1):
   the slope / mean / etc are computed on bars whose
   `effective_close ≤ anchor_t - 1_step` — never on the bar AT
   `effective_close == t` itself. The leakage gate
   `tests/leakage/test_xgboost_features_leakage.py` verifies this with
   100 random `(sym, t)` triples.

3. **One model per horizon × per fold.** Phase 3.7 walk-forward sweeps
   over 4 folds; this module is fold-agnostic — the caller supplies
   the train/test boundaries and we fit + predict.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from daity.data.candles import INTERVAL_DURATIONS
from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.parquet_store import ParquetStore
from daity.eval.metrics import (
    RankICResult,
    cross_sectional_rank_ic,
    top_bottom_decile_spread_bps,
)
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Feature flattening parameters — `PHASE3.md` §5.2.
# Stats applied per (scale, channel) over the lookback window.
SUMMARY_STATS: tuple[str, ...] = ("last", "mean", "std", "min", "max", "trend")

# Last-N raw 5m OHLCV bar values added on top of the per-scale summaries.
LAST_N_5M_BAR_CHANNELS: int = 5         # last 5 5m bars
LAST_N_5M_FEATURES: int = 5             # the 5 OHLCV channels

# Scales in canonical order — matches Phase 2's tokenizer ordering.
SCALES_ORDERED: tuple[str, ...] = ("5m", "15m", "60m", "day")

# Default lookback window per scale — same as Phase 2's pretrain config.
DEFAULT_WINDOW_BARS: dict[str, int] = {
    "5m":  256,
    "15m": 192,
    "60m": 168,
    "day": 64,
}


@dataclass(frozen=True, slots=True)
class XGBHyperparams:
    """Hyperparameter grid for the per-fold sweep (`PHASE3.md` §5.4).

    The grid is small by design: 3 × 3 × 2 = 18 configurations × 5-fold
    inner CV = 90 fits per outer fold per horizon. At ~10s per fit
    that's ~15 min per outer fold per horizon = ~4h total for 4 folds ×
    4 horizons. Larger grids are deferred to the LightGBM fallback
    (§5.6) which is gated on §1.1 failure margin ≤ 0.003.
    """
    max_depth: tuple[int, ...] = (4, 6, 8)
    learning_rate: tuple[float, ...] = (0.01, 0.03, 0.1)
    n_estimators: tuple[int, ...] = (500, 1500)
    subsample: float = 0.8
    colsample_bytree: float = 0.5
    min_child_weight: int = 100


@dataclass(slots=True)
class FoldResult:
    """One fold's per-horizon outputs (used by Phase 3.7's walk-forward).

    The `per_cross_section_ic` field carries one rank-IC value per
    distinct test-window `t` — required by §3.7's tri-criterion
    `delta_t_stat` calculation (review D.2). Empty list if not computed.
    """

    horizon: str
    fold_id: str
    rank_ic: RankICResult
    decile_spread_bps: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    per_cross_section_ic: list[float] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    best_hparams: dict[str, float | int] = field(default_factory=dict)
    best_iteration: int | None = None
    n_boost_rounds_used: int | None = None


# ---------- Feature flattening ----------


def _rolling_summary_per_scale(
    feature_df: pl.DataFrame,
    *,
    scale: str,
    window_bars: int,
    channels: tuple[str, ...] = CHANNEL_ORDER_18,
) -> pl.DataFrame:
    """Compute per-row rolling-window summary stats for every channel.

    Returns a frame with one row per input bar, carrying:
      - `ts` (the bar's open / close time per `daity.data.candles`)
      - `effective_close` (the moment the bar's data becomes available)
      - For each channel × each stat: a `{channel}__{stat}` column
      - Plus `{ch}__last5_b{0..4}` for the 5m bar_channels (last-5
        OHLCV in the 5 most recent bars)

    The stats are computed over a STRICTLY-CAUSAL window
    `[i-K+1, i]` — the bar at `i` is the most recent included bar.
    A consumer that wants "stats over bars STRICTLY before anchor t"
    must take the row at `i-1` (the bar before the anchor's
    effective_close). The `join_asof` strategy="backward" in
    `_assemble_anchor_features` handles this — we look up the most
    recent `effective_close < anchor_t`, never `≤ anchor_t`.
    """
    if feature_df.height == 0:
        return feature_df

    df = feature_df.sort("ts")
    # Per `effective_close`: this is the moment-when-known. For 5m it's
    # `ts + 5m`; for `day` it's `ts` itself (already the close).
    if scale == "day":
        df = df.with_columns(pl.col("ts").alias("effective_close"))
    else:
        delta = INTERVAL_DURATIONS[scale]
        df = df.with_columns((pl.col("ts") + delta).alias("effective_close"))

    exprs: list[pl.Expr] = []
    for ch in channels:
        exprs.extend([
            pl.col(ch).alias(f"{ch}__last"),
            pl.col(ch).rolling_mean(window_size=window_bars, min_samples=window_bars)
              .alias(f"{ch}__mean"),
            pl.col(ch).rolling_std(window_size=window_bars, min_samples=window_bars)
              .alias(f"{ch}__std"),
            pl.col(ch).rolling_min(window_size=window_bars, min_samples=window_bars)
              .alias(f"{ch}__min"),
            pl.col(ch).rolling_max(window_size=window_bars, min_samples=window_bars)
              .alias(f"{ch}__max"),
            # `trend = (last - first_in_window) / |first_in_window|`,
            # a tree-friendly normalized slope. `shift(window-1)` gives the
            # first-in-window value at each row.
            ((pl.col(ch) - pl.col(ch).shift(window_bars - 1))
             / (pl.col(ch).shift(window_bars - 1).abs() + 1e-12))
              .alias(f"{ch}__trend"),
        ])

    out = df.select([pl.col("ts"), pl.col("effective_close"), *exprs])

    # 5m-only: last-5 bar_channels (open, high, low, close, volume) at lags 0..4.
    if scale == "5m":
        last5_exprs: list[pl.Expr] = []
        for ch in CHANNEL_ORDER_18[:LAST_N_5M_FEATURES]:  # OHLCV
            for k in range(LAST_N_5M_BAR_CHANNELS):
                last5_exprs.append(
                    pl.col(ch).shift(k).alias(f"{ch}__last5_b{k}"),
                )
        # Recompute `last5` from the original df (we only need ts as the
        # join key; `df` has ts in the same order as `out`).
        last5 = df.select([pl.col("ts"), *last5_exprs])
        out = out.join(last5, on="ts", how="left")

    return out


def _assemble_anchor_features(
    *,
    anchor_ts: pl.Series,
    summaries: dict[str, pl.DataFrame],
    window_bars: dict[str, int],
) -> pl.DataFrame:
    """For a list of anchor timestamps, look up each scale's summary
    via `join_asof` at the most recent bar with
    `effective_close < anchor_ts` (STRICTLY less than).

    Polars' `join_asof(strategy='backward')` finds the most recent
    right-side row with `right_key ≤ left_key`. To get strict less-than
    semantics — required by the leakage gate (architect review B.MINOR.1
    / `PHASE3.md` §5.7) — we shift the left key by `-1μs` before the
    join. At microsecond resolution this is the same as `<` and avoids
    matching the bar AT the anchor boundary.
    """
    base = pl.DataFrame({"anchor_ts": anchor_ts}).sort("anchor_ts")
    # Shift left key for strict-`<` semantics; drop after the join.
    base_shifted = base.with_columns(
        (pl.col("anchor_ts") - pl.duration(microseconds=1))
            .alias("_anchor_minus_eps"),
    )
    out: pl.DataFrame | None = None

    for scale in SCALES_ORDERED:
        s = summaries[scale]
        if s.height == 0:
            continue
        s_sorted = s.sort("effective_close")
        joined = base_shifted.join_asof(
            s_sorted, left_on="_anchor_minus_eps", right_on="effective_close",
            strategy="backward",
        )
        # Strict-`<` is now baked into the join — no extra filter needed.
        # Drop right-side helper columns.
        rename = {
            c: f"{scale}__{c}" for c in s.columns
            if c not in ("ts", "effective_close")
        }
        joined = joined.rename(rename)
        joined = joined.drop(["_anchor_minus_eps", "ts", "effective_close"])

        out = joined if out is None else out.join(joined, on="anchor_ts", how="left")

    return out if out is not None else base


def flatten_anchors_to_features(
    *,
    feature_store: ParquetStore,
    symbol: str,
    anchor_ts: list[datetime],
    as_of: datetime,
    window_bars: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Build the 457-feature design matrix for `(symbol, anchor_ts)` rows.

    Args:
      feature_store: a `ParquetStore` rooted at the 18-channel
        `data/features_parquet/` (NOT the raw `data/parquet/`).
      symbol: the symbol whose features we load.
      anchor_ts: timestamps to flatten.
      as_of: leakage cap. Bars whose `effective_close > as_of` are
        excluded at read time.
      window_bars: per-scale lookback. Defaults to `DEFAULT_WINDOW_BARS`.

    Returns: a polars frame with `anchor_ts` plus 457 feature columns
    (or fewer if a scale dropped out due to insufficient warmup). Rows
    with insufficient warmup at any scale are dropped.
    """
    if window_bars is None:
        window_bars = DEFAULT_WINDOW_BARS

    summaries: dict[str, pl.DataFrame] = {}
    for scale in SCALES_ORDERED:
        df = feature_store.read(symbol=symbol, scale=scale, as_of=as_of)
        # Apply effective_close gate (same pattern labels.py uses for 5m).
        if df.height > 0 and scale != "day":
            df = df.filter(pl.col("ts") + INTERVAL_DURATIONS[scale] <= as_of)
        summaries[scale] = _rolling_summary_per_scale(
            df, scale=scale, window_bars=window_bars[scale],
        )

    # Coerce anchor_ts to a polars Series with consistent dtype (UTC).
    anchor_series = pl.Series(
        "anchor_ts", anchor_ts, dtype=pl.Datetime("us", "UTC"),
    )
    feats = _assemble_anchor_features(
        anchor_ts=anchor_series, summaries=summaries, window_bars=window_bars,
    )
    # Drop rows with any null feature. Nulls arise from two sources:
    # (a) rolling-window warmup at the head of the feature_parquet,
    # (b) join_asof misses when `anchor_ts` is before the earliest
    #     bar's `effective_close` for some scale.
    feature_cols = [c for c in feats.columns if c != "anchor_ts"]
    if feature_cols:
        feats = feats.drop_nulls(subset=feature_cols)
    return feats


# ---------- Training + evaluation ----------


def _build_train_test_xy(
    *,
    feature_store: ParquetStore,
    label_root: Path,
    horizon: str,
    symbols: Iterable[str],
    train_start: datetime, train_end: datetime,
    test_start: datetime, test_end: datetime,
    as_of: datetime,
    window_bars: dict[str, int] | None = None,
    train_anchors_per_day: int | None = 5,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assemble `(X_train, y_train)` and `(X_test, y_test)` frames.

    Returns one wide polars frame per split, with columns
    `[symbol, anchor_ts, y_log_return, valid_label, <457 features>]`.

    `train_anchors_per_day` (default 5; review B.3 amendment): for
    intraday horizons (30m/120m), train can have ~70–52 anchors per
    `(symbol, day)` — heavily autocorrelated within a day. Sub-sample
    to this many anchors per `(symbol, day)` for training (uniform
    random per day, seeded deterministically by `(seed, hash(symbol))`).
    The test window is never sub-sampled — every anchor contributes to
    the IC. For daily horizons (1d/2d), there is only 1 anchor per
    `(symbol, day)` so sub-sampling is a no-op.

    Pass `train_anchors_per_day=None` for no sub-sampling (matches the
    original behavior but OOMs on intraday × 6-year training windows on
    a 128 GB box — see issue noted in `xgboost_baseline_cli.py`).
    """
    train_rows: list[pl.DataFrame] = []
    test_rows: list[pl.DataFrame] = []
    label_dir = label_root / horizon
    is_intraday = horizon in {"30m", "120m"}

    # Per-symbol RNG seed so sub-sampling is deterministic AND
    # order-invariant. Hash by symbol name (not by enumeration index)
    # so swapping `--symbols A,B` ↔ `--symbols B,A` produces bit-equal
    # train sets given the same `--seed`. Caught in review B.4.
    base_seed = int(seed) % (2**31 - 1)

    for sym in symbols:
        label_path = label_dir / f"{sym}.parquet"
        if not label_path.exists():
            continue
        labels = pl.scan_parquet(label_path).filter(
            (pl.col("t") >= train_start) & (pl.col("t") <= test_end),
        ).collect()
        if labels.height == 0:
            continue

        anchor_ts_list = labels["t"].to_list()
        feats = flatten_anchors_to_features(
            feature_store=feature_store, symbol=sym,
            anchor_ts=anchor_ts_list, as_of=as_of,
            window_bars=window_bars,
        )
        if feats.height == 0:
            continue

        merged = feats.join(
            labels.select(["t", "y_log_return", "valid_label"])
                  .rename({"t": "anchor_ts"}),
            on="anchor_ts", how="inner",
        )
        merged = merged.with_columns(pl.lit(sym).alias("symbol"))
        merged = merged.filter(pl.col("valid_label"))

        in_train = merged.filter(
            (pl.col("anchor_ts") >= train_start)
            & (pl.col("anchor_ts") <= train_end),
        )
        in_test = merged.filter(
            (pl.col("anchor_ts") >= test_start)
            & (pl.col("anchor_ts") <= test_end),
        )

        # Sub-sample train anchors per (symbol, day) for intraday horizons.
        # `in_test` is NEVER sub-sampled — eval rank-IC needs full coverage.
        if (
            is_intraday
            and train_anchors_per_day is not None
            and in_train.height > 0
        ):
            sym_hash = int.from_bytes(
                hashlib.sha1(sym.encode()).digest()[:4], "big",
            )
            sym_seed = (base_seed + sym_hash) % (2**31 - 1)
            in_train = (
                in_train
                .with_columns(pl.col("anchor_ts").dt.date().alias("_anchor_date"))
                .with_columns(
                    pl.int_range(pl.len(), dtype=pl.UInt32)
                      .shuffle(seed=sym_seed)
                      .over("_anchor_date")
                      .alias("_per_day_rank"),
                )
                .filter(pl.col("_per_day_rank") < train_anchors_per_day)
                .drop("_anchor_date", "_per_day_rank")
            )

        if in_train.height > 0:
            train_rows.append(in_train)
        if in_test.height > 0:
            test_rows.append(in_test)

    X_train = pl.concat(train_rows) if train_rows else pl.DataFrame()
    X_test = pl.concat(test_rows) if test_rows else pl.DataFrame()
    return X_train, X_test


def _split_xy(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull (X, y, anchor_ts, symbol) numpy views from a wide frame."""
    meta_cols = {"symbol", "anchor_ts", "y_log_return", "valid_label"}
    feat_cols = [c for c in frame.columns if c not in meta_cols]
    X = frame.select(feat_cols).to_numpy()
    y = frame["y_log_return"].to_numpy().astype(np.float32)
    ts = frame["anchor_ts"].to_numpy()
    syms = frame["symbol"].to_numpy()
    return X, y, ts, syms


def fit_predict_one_fold(
    *,
    horizon: str,
    fold_id: str,
    feature_store: ParquetStore,
    label_root: Path,
    symbols: Iterable[str],
    train_start: datetime, train_end: datetime,
    test_start: datetime, test_end: datetime,
    as_of: datetime,
    hparams: XGBHyperparams | None = None,
    seed: int = 0,
    window_bars: dict[str, int] | None = None,
    train_anchors_per_day: int | None = 5,
) -> FoldResult:
    """Fit XGBoost on the train window, evaluate on the test window.

    Returns a `FoldResult` carrying rank-IC + top-bottom decile spread
    + feature importances. The 18-config × 5-CV grid search is currently
    deferred — we use the grid's *median* hyperparams as a fixed
    baseline. Phase 3.7's walk-forward CLI may opt into the full sweep
    via a flag once the baseline plumbing is verified.

    `train_anchors_per_day` (default 1): see `_build_train_test_xy`.
    Sub-samples intraday train anchors to 1 per (symbol, day) to keep
    a 6-year train window's design matrix within the 128 GB box's RAM.
    Pass `None` to disable sub-sampling (requires significantly more RAM).
    """
    if hparams is None:
        hparams = XGBHyperparams()
    X_train_df, X_test_df = _build_train_test_xy(
        feature_store=feature_store, label_root=label_root,
        horizon=horizon, symbols=symbols,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        as_of=as_of, window_bars=window_bars,
        train_anchors_per_day=train_anchors_per_day,
        seed=seed,
    )
    if X_train_df.height == 0 or X_test_df.height == 0:
        log.warning(
            "fit_predict_one_fold: empty train (%d) or test (%d) for %s/%s",
            X_train_df.height, X_test_df.height, horizon, fold_id,
        )
        return FoldResult(
            horizon=horizon, fold_id=fold_id,
            rank_ic=RankICResult(0.0, 0.0, 0.0, 0, 0),
            decile_spread_bps=0.0, n_train=X_train_df.height, n_test=X_test_df.height,
        )

    X_train, y_train, ts_train, _syms_train = _split_xy(X_train_df)
    X_test, y_test, ts_test, _syms_test = _split_xy(X_test_df)

    # Fixed baseline hyperparams — median of the grid. Phase 3.2 ships
    # the no-sweep version; the inner grid-search hook lives in §3.7.
    fixed = {
        "objective": "reg:squarederror",
        "max_depth": int(np.median(hparams.max_depth)),
        "learning_rate": float(np.median(hparams.learning_rate)),
        "n_estimators": int(np.median(hparams.n_estimators)),
        "subsample": hparams.subsample,
        "colsample_bytree": hparams.colsample_bytree,
        "min_child_weight": hparams.min_child_weight,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": seed,
        "n_jobs": -1,
        # Early-stopping (review C.1): halt boosting when the held-out
        # val tail's RMSE doesn't improve for `early_stopping_rounds`
        # iterations. Cheap regularization on noisy targets.
        "early_stopping_rounds": 50,
    }
    model = xgb.XGBRegressor(**fixed)
    # Carve a 10% tail of the (time-sorted) train window as the
    # early-stopping val set. Last-fold-of-train = closest in
    # distribution to test, and we don't touch the actual test set.
    sort_order = np.argsort(ts_train)
    n_val = max(1, X_train.shape[0] // 10)
    val_idx = sort_order[-n_val:]
    fit_idx = sort_order[:-n_val] if X_train.shape[0] > n_val else sort_order
    model.fit(
        X_train[fit_idx], y_train[fit_idx],
        eval_set=[(X_train[val_idx], y_train[val_idx])],
        verbose=False,
    )
    preds = model.predict(X_test)

    rank_ic, per_cs = cross_sectional_rank_ic(
        predictions=preds, realizations=y_test, timestamps=ts_test,
        return_per_cs=True,
    )
    decile_bps = top_bottom_decile_spread_bps(
        predictions=preds, realizations=y_test, timestamps=ts_test,
    )

    # Feature importance — top-50 by 'gain'.
    feat_cols = [c for c in X_train_df.columns
                 if c not in {"symbol", "anchor_ts", "y_log_return", "valid_label"}]
    imp = model.feature_importances_
    importance = {
        feat_cols[i]: float(imp[i])
        for i in np.argsort(imp)[::-1][:50]
    }

    best_iter = getattr(model, "best_iteration", None)
    n_boost = getattr(model, "best_iteration", None)
    if n_boost is None:
        # `best_iteration` is xgb 2.x+; fall back to n_estimators if unavailable.
        n_boost = int(fixed["n_estimators"])

    return FoldResult(
        horizon=horizon, fold_id=fold_id, rank_ic=rank_ic,
        decile_spread_bps=decile_bps, feature_importance=importance,
        per_cross_section_ic=per_cs,
        n_train=X_train_df.height, n_test=X_test_df.height,
        best_hparams={k: v for k, v in fixed.items()
                      if k in {"max_depth", "learning_rate", "n_estimators"}},
        best_iteration=int(best_iter) if best_iter is not None else None,
        n_boost_rounds_used=int(n_boost) if n_boost is not None else None,
    )
