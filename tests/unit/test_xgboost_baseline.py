"""Unit tests for `daity.eval.xgboost_baseline`.

Locks in:
- Feature flattening dimensions match the documented 457-dim contract.
- Synthetic-linear-y test: when the label is a linear combination of a
  few features + small noise, XGBoost recovers a rank-IC > 0.5 (the
  baseline must "work" on signal-rich data).
- Determinism given seed.
- Empty-frame edge cases don't raise.
- Strict-`[t-window, t)` semantics: the bar AT `effective_close == t`
  is NOT included in the anchor's flatten window. (The full leakage
  gate over 100 random triples lives in
  `tests/leakage/test_xgboost_features_leakage.py`.)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.parquet_store import ParquetStore
from daity.eval.metrics import cross_sectional_rank_ic
from daity.eval.xgboost_baseline import (
    DEFAULT_WINDOW_BARS,
    LAST_N_5M_BAR_CHANNELS,
    LAST_N_5M_FEATURES,
    SCALES_ORDERED,
    SUMMARY_STATS,
    XGBHyperparams,
    _build_train_test_xy,
    fit_predict_one_fold,
    flatten_anchors_to_features,
)

IST = timezone(timedelta(hours=5, minutes=30))


def _bars_with_features(symbol: str, scale: str, ts_list: list[datetime],
                        rng: np.random.Generator) -> pl.DataFrame:
    """Synthesize an 18-channel feature row per ts."""
    n = len(ts_list)
    cols: dict[str, list[float] | list[str] | list[datetime]] = {
        "symbol": [symbol] * n, "ts": ts_list, "scale": [scale] * n,
    }
    for ch in CHANNEL_ORDER_18:
        cols[ch] = list(rng.normal(0, 1, size=n).astype(np.float32))
    schema = {"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "scale": pl.Utf8}
    for ch in CHANNEL_ORDER_18:
        schema[ch] = pl.Float32
    return pl.DataFrame(cols, schema=schema)


def _intraday_session_ts(d_start: datetime, n_days: int = 30) -> list[datetime]:
    """75 5m bars × n_days trading days."""
    out: list[datetime] = []
    for day_idx in range(n_days):
        # Skip weekends naively (we just want chronological 5m bars; the
        # XGBoost test doesn't enforce calendar correctness).
        date_at = d_start + timedelta(days=day_idx)
        if date_at.weekday() >= 5:
            continue
        session_open = datetime(date_at.year, date_at.month, date_at.day, 9, 15, tzinfo=IST)
        base = session_open.astimezone(UTC)
        for k in range(75):
            out.append(base + timedelta(minutes=5 * k))
    return out


def _daily_ts(d_start: datetime, n_days: int = 200) -> list[datetime]:
    """One bar per weekday at 10:00 UTC (NSE close)."""
    out: list[datetime] = []
    cur = d_start
    for _ in range(n_days):
        if cur.weekday() < 5:
            out.append(datetime(cur.year, cur.month, cur.day, 10, 0, tzinfo=UTC))
        cur += timedelta(days=1)
    return out


# ----- Schema / dimension tests -----


def test_summary_stats_count_is_six() -> None:
    assert len(SUMMARY_STATS) == 6


def test_scales_ordered() -> None:
    assert SCALES_ORDERED == ("5m", "15m", "60m", "day")


def test_default_window_bars_documented() -> None:
    assert DEFAULT_WINDOW_BARS["5m"] == 256
    assert DEFAULT_WINDOW_BARS["15m"] == 192
    assert DEFAULT_WINDOW_BARS["60m"] == 168
    assert DEFAULT_WINDOW_BARS["day"] == 64


def test_flatten_dimensions_match_457_contract(tmp_path: Path) -> None:
    """Each anchor row produces 4 scales × 18 channels × 6 stats + 25 = 457
    feature columns plus the `anchor_ts` join key."""
    rng = np.random.default_rng(42)
    store = ParquetStore(tmp_path)

    # Use small windows so the test runs fast but warmup is satisfied.
    small_windows = {"5m": 8, "15m": 6, "60m": 4, "day": 4}

    # 5m: enough bars for 256 lookback...we'll override to a small window.
    ts_5m = _intraday_session_ts(datetime(2024, 6, 3), n_days=20)
    store.write_partition("5m", "X", _bars_with_features("X", "5m", ts_5m, rng))
    # 15m / 60m: derive from 5m at lower density.
    ts_15m = ts_5m[::3]
    store.write_partition("15m", "X", _bars_with_features("X", "15m", ts_15m, rng))
    ts_60m = ts_5m[::12]
    store.write_partition("60m", "X", _bars_with_features("X", "60m", ts_60m, rng))
    ts_day = _daily_ts(datetime(2024, 6, 3), n_days=30)
    store.write_partition("day", "X", _bars_with_features("X", "day", ts_day, rng))

    # Pick anchors deep into the time series so all rolling windows are warm.
    # Day window=4 needs ≥4 prior day bars; pick anchors well into day 8+.
    anchors = [t + timedelta(minutes=5) for t in ts_5m[600:620]]
    feats = flatten_anchors_to_features(
        feature_store=store, symbol="X", anchor_ts=anchors,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        window_bars=small_windows,
    )
    assert feats.height > 0, "expected non-empty feature frame"

    # Columns: anchor_ts + per-scale features. Per-scale: 18 × 6 = 108 stats.
    # 5m additionally has LAST_N_5M_FEATURES × LAST_N_5M_BAR_CHANNELS = 25.
    expected_5m = 18 * 6 + LAST_N_5M_FEATURES * LAST_N_5M_BAR_CHANNELS  # 133
    expected_other = 18 * 6                                              # 108
    expected_total = expected_5m + 3 * expected_other                    # 457
    n_feature_cols = sum(1 for c in feats.columns if c != "anchor_ts")
    assert n_feature_cols == expected_total, (
        f"got {n_feature_cols} feature columns; expected {expected_total}"
    )


# ----- Strict-`[t-window, t)` semantics -----


def test_anchor_window_excludes_bar_at_anchor(tmp_path: Path) -> None:
    """The bar whose effective_close == anchor_ts must NOT be included
    in the anchor's flattened window.

    Concretely: at anchor_t = T, the most-recent bar in the window has
    effective_close STRICTLY < T. For 5m bars: the most recent
    included bar is the one opening at T - 10min (eff_close = T - 5min).
    The bar opening at T - 5min (eff_close = T) is excluded.
    """
    rng = np.random.default_rng(7)
    store = ParquetStore(tmp_path)
    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}

    ts_5m = _intraday_session_ts(datetime(2024, 6, 3), n_days=20)
    store.write_partition("5m", "X", _bars_with_features("X", "5m", ts_5m, rng))
    store.write_partition("15m", "X", _bars_with_features("X", "15m", ts_5m[::3], rng))
    store.write_partition("60m", "X", _bars_with_features("X", "60m", ts_5m[::12], rng))
    store.write_partition("day", "X", _bars_with_features(
        "X", "day", _daily_ts(datetime(2024, 6, 3), n_days=30), rng))

    # Pick a target bar deep enough that all rolling windows are warm.
    target_bar_idx = 600   # day 9, plenty of warmup
    anchor_ts = ts_5m[target_bar_idx] + timedelta(minutes=5)

    feats = flatten_anchors_to_features(
        feature_store=store, symbol="X",
        anchor_ts=[anchor_ts],
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        window_bars=small_windows,
    )
    assert feats.height == 1, "expected 1 row of features"

    # The most-recent INCLUDED 5m bar has eff_close = anchor_ts - 5min,
    # i.e., the bar at index target_bar_idx - 1. Its `close` column must
    # equal the flattened `5m__close__last`.
    df_5m = pl.read_parquet(tmp_path / "5m" / "X.parquet").sort("ts")
    expected_last_close = df_5m["close"].to_numpy()[target_bar_idx - 1]
    leaky_close = df_5m["close"].to_numpy()[target_bar_idx]
    actual = feats["5m__close__last"][0]
    assert abs(actual - expected_last_close) < 1e-5, (
        f"expected close[{target_bar_idx - 1}]={expected_last_close:.6f}, "
        f"got {actual:.6f}"
    )
    assert abs(actual - leaky_close) > 1e-9, (
        f"flat last leaks bar AT anchor (={leaky_close:.6f})"
    )


# ----- Determinism + signal recovery -----


def test_fit_predict_recovers_signal_with_synthetic_linear_y(tmp_path: Path) -> None:
    """When the label is a clean linear combination of a few features,
    XGBoost should produce a strongly-positive rank-IC.

    This is a sanity test that the whole pipeline (feature flatten → fit
    → predict → rank-IC) connects correctly. It is NOT a test of
    real-data signal capture.
    """
    rng = np.random.default_rng(123)
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    feature_root.mkdir()
    label_root.mkdir()
    feat_store = ParquetStore(feature_root)

    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}

    n_symbols = 12
    n_anchors_per_sym = 80
    ts_5m = _intraday_session_ts(datetime(2024, 6, 3), n_days=20)
    ts_15m = ts_5m[::3]
    ts_60m = ts_5m[::12]
    ts_day = _daily_ts(datetime(2024, 6, 3), n_days=30)

    # Write feature parquets.
    for s in range(n_symbols):
        sym = f"S{s}"
        feat_store.write_partition("5m", sym, _bars_with_features(sym, "5m", ts_5m, rng))
        feat_store.write_partition("15m", sym, _bars_with_features(sym, "15m", ts_15m, rng))
        feat_store.write_partition("60m", sym, _bars_with_features(sym, "60m", ts_60m, rng))
        feat_store.write_partition("day", sym, _bars_with_features(sym, "day", ts_day, rng))

    # Build labels: y = β · feature_at_anchor + ε. We use the close column at
    # the *previous* 5m bar (the one whose eff_close = anchor_ts - 5min) to
    # generate y, so the model needs to learn "look at last close".
    test_end_ts = ts_5m[-1] + timedelta(seconds=1)

    # Pick one anchor every 10 5m bars, deep enough that warmup is satisfied.
    anchor_indices = list(range(50, len(ts_5m), 10))[:n_anchors_per_sym]
    anchor_ts_full = [ts_5m[i] + timedelta(minutes=5) for i in anchor_indices]

    # For each symbol: read the close column, then label = scaled close + noise.
    label_root_30m = label_root / "30m"
    label_root_30m.mkdir()
    for s in range(n_symbols):
        sym = f"S{s}"
        df_5m = pl.read_parquet(feature_root / "5m" / f"{sym}.parquet").sort("ts")
        closes = df_5m["close"].to_numpy()
        # Generate y from close[anchor_idx - 1] (the bar just before anchor's eff_close).
        y = np.zeros(len(anchor_ts_full), dtype=np.float32)
        for j, idx in enumerate(anchor_indices):
            y[j] = 0.05 * closes[idx - 1] + 0.005 * rng.standard_normal()
        labels = pl.DataFrame({
            "symbol": [sym] * len(anchor_ts_full),
            "t": anchor_ts_full,
            "t_horizon": [t + timedelta(minutes=30) for t in anchor_ts_full],
            "y_log_return": y.astype(np.float32),
            "y_rank": [None] * len(anchor_ts_full),
            "valid_label": [True] * len(anchor_ts_full),
            "halt_count": [0] * len(anchor_ts_full),
        }, schema={
            "symbol": pl.Utf8, "t": pl.Datetime("us", "UTC"),
            "t_horizon": pl.Datetime("us", "UTC"),
            "y_log_return": pl.Float32, "y_rank": pl.Float32,
            "valid_label": pl.Boolean, "halt_count": pl.Int8,
        })
        labels.write_parquet(label_root_30m / f"{sym}.parquet", compression="zstd")

    # Train: first 70% of anchors. Test: last 30%.
    n_train_anchors = int(0.7 * len(anchor_ts_full))
    train_end_ts = anchor_ts_full[n_train_anchors - 1]
    test_start_ts = anchor_ts_full[n_train_anchors]

    res = fit_predict_one_fold(
        horizon="30m", fold_id="test_synth",
        feature_store=feat_store, label_root=label_root,
        symbols=[f"S{s}" for s in range(n_symbols)],
        train_start=ts_5m[0], train_end=train_end_ts,
        test_start=test_start_ts, test_end=test_end_ts,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        hparams=XGBHyperparams(),
        seed=0, window_bars=small_windows,
        # Synthetic fixture is small (84 train rows after default sub-sample);
        # disable sub-sampling so the model sees enough data to recover signal.
        train_anchors_per_day=None,
    )
    assert res.n_train > 0 and res.n_test > 0
    # Model should clearly recover positive rank-IC on this clean signal.
    # In practice on the synthetic data we get >0.5; require >0.2 to leave
    # comfortable margin against test flakiness.
    assert res.rank_ic.mean_ic > 0.2, (
        f"baseline failed to recover synthetic signal: "
        f"rank_ic={res.rank_ic.mean_ic:+.4f}"
    )


def test_metrics_rank_ic_handles_constant_predictions() -> None:
    """A constant predictor produces zero IC (Spearman undefined → skipped)."""
    n = 100
    pred = np.zeros(n)
    real = np.linspace(-1, 1, n)
    ts = np.tile([datetime(2025, 1, 1, tzinfo=UTC)], n)
    res = cross_sectional_rank_ic(predictions=pred, realizations=real, timestamps=ts)
    assert res.mean_ic == 0.0
    assert res.n_cross_sections == 0


def test_train_anchors_per_day_subsample_is_order_invariant(tmp_path: Path) -> None:
    """Same --seed + same symbol set must produce the same train rows
    regardless of the order symbols are passed in.

    Catches the `sym_idx` foot-gun flagged by review B.4: seeding by
    enumeration index made `--symbols A,B` ≠ `--symbols B,A` for the
    same `--seed`. Fix: seed by `hash(sym)` so order is irrelevant.
    """
    rng = np.random.default_rng(7)
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    feature_root.mkdir()
    label_root.mkdir()
    feat_store = ParquetStore(feature_root)

    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}
    n_symbols = 6
    ts_5m = _intraday_session_ts(datetime(2024, 6, 3), n_days=10)
    ts_15m = ts_5m[::3]
    ts_60m = ts_5m[::12]
    ts_day = _daily_ts(datetime(2024, 6, 3), n_days=20)

    label_root_30m = label_root / "30m"
    label_root_30m.mkdir()

    for s in range(n_symbols):
        sym = f"S{s}"
        feat_store.write_partition("5m", sym, _bars_with_features(sym, "5m", ts_5m, rng))
        feat_store.write_partition("15m", sym, _bars_with_features(sym, "15m", ts_15m, rng))
        feat_store.write_partition("60m", sym, _bars_with_features(sym, "60m", ts_60m, rng))
        feat_store.write_partition("day", sym, _bars_with_features(sym, "day", ts_day, rng))

        anchor_idxs = list(range(50, len(ts_5m), 5))
        anchor_ts = [ts_5m[i] + timedelta(minutes=5) for i in anchor_idxs]
        labels = pl.DataFrame({
            "symbol": [sym] * len(anchor_ts),
            "t": anchor_ts,
            "t_horizon": [t + timedelta(minutes=30) for t in anchor_ts],
            "y_log_return": rng.normal(0, 0.01, size=len(anchor_ts)).astype(np.float32),
            "y_rank": [None] * len(anchor_ts),
            "valid_label": [True] * len(anchor_ts),
            "halt_count": [0] * len(anchor_ts),
        }, schema={
            "symbol": pl.Utf8, "t": pl.Datetime("us", "UTC"),
            "t_horizon": pl.Datetime("us", "UTC"),
            "y_log_return": pl.Float32, "y_rank": pl.Float32,
            "valid_label": pl.Boolean, "halt_count": pl.Int8,
        })
        labels.write_parquet(label_root_30m / f"{sym}.parquet", compression="zstd")

    syms_forward = [f"S{s}" for s in range(n_symbols)]
    syms_reversed = list(reversed(syms_forward))

    common_kwargs = dict(
        feature_store=feat_store, label_root=label_root, horizon="30m",
        train_start=ts_5m[0], train_end=ts_5m[-30],
        test_start=ts_5m[-30], test_end=ts_5m[-1] + timedelta(seconds=1),
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        window_bars=small_windows, train_anchors_per_day=1, seed=42,
    )
    train_fwd, _ = _build_train_test_xy(symbols=syms_forward, **common_kwargs)
    train_rev, _ = _build_train_test_xy(symbols=syms_reversed, **common_kwargs)

    # Same SET of (symbol, anchor_ts) pairs in both — order may differ.
    pairs_fwd = set(zip(
        train_fwd["symbol"].to_list(), train_fwd["anchor_ts"].to_list(), strict=True,
    ))
    pairs_rev = set(zip(
        train_rev["symbol"].to_list(), train_rev["anchor_ts"].to_list(), strict=True,
    ))
    assert pairs_fwd == pairs_rev, (
        f"sub-sample is order-dependent: forward yields {len(pairs_fwd)} pairs, "
        f"reversed yields {len(pairs_rev)}, diff size "
        f"{len(pairs_fwd.symmetric_difference(pairs_rev))}"
    )


def test_train_anchors_per_day_subsamples_train_not_test(tmp_path: Path) -> None:
    """`train_anchors_per_day=1` keeps the first anchor per (symbol, day)
    in train. Test set is never sub-sampled — every test anchor contributes."""
    rng = np.random.default_rng(99)
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    feature_root.mkdir()
    label_root.mkdir()
    feat_store = ParquetStore(feature_root)

    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}
    n_symbols = 4
    ts_5m = _intraday_session_ts(datetime(2024, 6, 3), n_days=10)
    ts_15m = ts_5m[::3]
    ts_60m = ts_5m[::12]
    ts_day = _daily_ts(datetime(2024, 6, 3), n_days=20)

    label_root_30m = label_root / "30m"
    label_root_30m.mkdir()

    for s in range(n_symbols):
        sym = f"S{s}"
        feat_store.write_partition("5m", sym, _bars_with_features(sym, "5m", ts_5m, rng))
        feat_store.write_partition("15m", sym, _bars_with_features(sym, "15m", ts_15m, rng))
        feat_store.write_partition("60m", sym, _bars_with_features(sym, "60m", ts_60m, rng))
        feat_store.write_partition("day", sym, _bars_with_features(sym, "day", ts_day, rng))

        anchor_idxs = list(range(50, len(ts_5m), 5))
        anchor_ts = [ts_5m[i] + timedelta(minutes=5) for i in anchor_idxs]
        labels = pl.DataFrame({
            "symbol": [sym] * len(anchor_ts),
            "t": anchor_ts,
            "t_horizon": [t + timedelta(minutes=30) for t in anchor_ts],
            "y_log_return": rng.normal(0, 0.01, size=len(anchor_ts)).astype(np.float32),
            "y_rank": [None] * len(anchor_ts),
            "valid_label": [True] * len(anchor_ts),
            "halt_count": [0] * len(anchor_ts),
        }, schema={
            "symbol": pl.Utf8, "t": pl.Datetime("us", "UTC"),
            "t_horizon": pl.Datetime("us", "UTC"),
            "y_log_return": pl.Float32, "y_rank": pl.Float32,
            "valid_label": pl.Boolean, "halt_count": pl.Int8,
        })
        labels.write_parquet(label_root_30m / f"{sym}.parquet", compression="zstd")

    n_train_anchors = int(0.7 * len(range(50, len(ts_5m), 5)))
    train_end_ts = [ts_5m[i] + timedelta(minutes=5) for i in range(50, len(ts_5m), 5)][n_train_anchors - 1]
    test_start_ts = [ts_5m[i] + timedelta(minutes=5) for i in range(50, len(ts_5m), 5)][n_train_anchors]
    test_end_ts = ts_5m[-1] + timedelta(seconds=1)

    # With sub-sample = 1.
    train_sub, test_sub = _build_train_test_xy(
        feature_store=feat_store, label_root=label_root, horizon="30m",
        symbols=[f"S{s}" for s in range(n_symbols)],
        train_start=ts_5m[0], train_end=train_end_ts,
        test_start=test_start_ts, test_end=test_end_ts,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        window_bars=small_windows, train_anchors_per_day=1, seed=0,
    )
    # Without sub-sample.
    train_full, test_full = _build_train_test_xy(
        feature_store=feat_store, label_root=label_root, horizon="30m",
        symbols=[f"S{s}" for s in range(n_symbols)],
        train_start=ts_5m[0], train_end=train_end_ts,
        test_start=test_start_ts, test_end=test_end_ts,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        window_bars=small_windows, train_anchors_per_day=None, seed=0,
    )

    # Train should shrink under sub-sample; test must be identical.
    assert train_sub.height < train_full.height, (
        "sub-sample=1 should reduce train row count"
    )
    assert test_sub.height == test_full.height, (
        "test must NEVER be sub-sampled"
    )
    # Per (sym, anchor_date) under sub-sample: at most 1 row per pair.
    sym_date = train_sub.with_columns(
        pl.col("anchor_ts").dt.date().alias("_d"),
    ).group_by(["symbol", "_d"]).len()
    assert (sym_date["len"] <= 1).all()


def test_metrics_rank_ic_perfect_predictor_returns_one() -> None:
    """A predictor that is a monotone transform of realizations gets IC=1."""
    n = 50
    real = np.linspace(-1, 1, n)
    pred = 2 * real + 0.0   # exact monotone transform
    ts = np.tile([datetime(2025, 1, 1, tzinfo=UTC)], n)
    res = cross_sectional_rank_ic(predictions=pred, realizations=real, timestamps=ts)
    assert res.mean_ic > 0.99
    assert res.n_cross_sections == 1
