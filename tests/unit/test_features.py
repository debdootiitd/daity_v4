"""Unit tests for `daity.data.features`.

The full leakage gate (100 random `(symbol, scale, as_of)` triples) lives
in `tests/leakage/test_features_leakage.py`. Here we lock in the per-feature
arithmetic + edge cases.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from daity.data.features import (
    FEATURE_GROUPS,
    LOG_RETURN_LAGS,
    MAX_LOOKBACK_BARS,
    VOLUME_Z_WINDOW,
    FeatureEngine,
    FeatureSpec,
    _add_log_returns,
    _add_time_covariates,
    _add_volume_z,
    _add_wick_features,
    _filter_by_effective_close,
)
from daity.data.parquet_store import ParquetStore

# ----- Helpers -----


def _bars(symbol: str, scale: str, ts_list: list[datetime],
          opens: list[float] | None = None,
          highs: list[float] | None = None,
          lows: list[float] | None = None,
          closes: list[float] | None = None,
          vols: list[int] | None = None) -> pl.DataFrame:
    n = len(ts_list)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n, "ts": ts_list, "scale": [scale] * n,
            "open":   opens   if opens   is not None else [100.0] * n,
            "high":   highs   if highs   is not None else [102.0] * n,
            "low":    lows    if lows    is not None else [99.0]  * n,
            "close":  closes  if closes  is not None else [101.0] * n,
            "volume": vols    if vols    is not None else [1000]  * n,
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )


def _seq(start: datetime, n: int, step: timedelta) -> list[datetime]:
    return [start + i * step for i in range(n)]


# ----- FeatureSpec -----


def test_feature_spec_default_includes_all_groups() -> None:
    spec = FeatureSpec()
    assert set(spec.include) == set(FEATURE_GROUPS)


def test_feature_spec_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Unknown feature groups"):
        FeatureSpec(include=("returns_typo",))


def test_max_lookback_covers_all_rolling_windows() -> None:
    """Sanity: MAX_LOOKBACK_BARS must >= every individual rolling window so the
    warm-up read is always sufficient."""
    assert max(LOG_RETURN_LAGS) <= MAX_LOOKBACK_BARS
    assert MAX_LOOKBACK_BARS >= VOLUME_Z_WINDOW


# ----- _filter_by_effective_close (point-in-time leakage gate) -----


def test_filter_by_effective_close_drops_intraday_in_progress(tmp_path: Path) -> None:
    """A 5m bar opening at 09:30 has effective_close 09:35; with as_of=09:32
    it must be dropped (still in progress)."""
    base = datetime(2024, 1, 1, 9, 15, tzinfo=UTC)
    ts_list = _seq(base, 6, timedelta(minutes=5))   # 09:15, 09:20, 09:25, 09:30, 09:35, 09:40
    df = _bars("X", "5m", ts_list)

    as_of = datetime(2024, 1, 1, 9, 32, tzinfo=UTC)
    out = _filter_by_effective_close(df, scale="5m", as_of=as_of)
    # Bars eligible: those with effective_close <= 09:32.
    #   09:15 ec=09:20 OK  | 09:20 ec=09:25 OK  | 09:25 ec=09:30 OK
    #   09:30 ec=09:35 NO  | 09:35 ec=09:40 NO  | 09:40 ec=09:45 NO
    assert out["ts"].to_list() == ts_list[:3]


def test_filter_by_effective_close_includes_bar_at_exact_boundary() -> None:
    """A bar whose effective_close equals as_of is INCLUDED — fully observed."""
    base = datetime(2024, 1, 1, 9, 15, tzinfo=UTC)
    df = _bars("X", "5m", [base])  # ec = 09:20
    as_of = datetime(2024, 1, 1, 9, 20, tzinfo=UTC)
    assert _filter_by_effective_close(df, scale="5m", as_of=as_of).height == 1


def test_filter_by_effective_close_daily_uses_ts_directly() -> None:
    """Daily ts already encodes the close (15:30 IST = 10:00 UTC). The filter
    is ts <= as_of, no duration shift."""
    ts_close = datetime(2024, 1, 5, 10, 0, tzinfo=UTC)
    df = _bars("X", "day", [ts_close])
    assert _filter_by_effective_close(df, scale="day",
        as_of=ts_close).height == 1
    assert _filter_by_effective_close(df, scale="day",
        as_of=ts_close - timedelta(seconds=1)).height == 0


# ----- _add_log_returns -----


def test_log_returns_arithmetic() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    closes = [100.0, 101.0, 99.0, 102.0]
    df = _bars("X", "day", _seq(base, 4, timedelta(days=1)), closes=closes)
    out = _add_log_returns(df, lags=(1,))
    rets = out["ret_1"].to_list()
    assert rets[0] is None
    assert rets[1] == pytest.approx(math.log(101.0 / 100.0))
    assert rets[2] == pytest.approx(math.log(99.0 / 101.0))
    assert rets[3] == pytest.approx(math.log(102.0 / 99.0))


def test_log_returns_emit_one_column_per_lag() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = _bars("X", "day", _seq(base, 70, timedelta(days=1)),
               closes=[100.0 + i * 0.1 for i in range(70)])
    out = _add_log_returns(df, lags=(1, 5, 15, 60))
    for k in (1, 5, 15, 60):
        assert f"ret_{k}" in out.columns
    # Last row's ret_60 = log(close[69]/close[9]) = log(106.9/100.9)
    assert out["ret_60"][69] == pytest.approx(math.log(106.9 / 100.9))


def test_log_returns_warmup_rows_are_null() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = _bars("X", "day", _seq(base, 5, timedelta(days=1)),
               closes=[100.0] * 5)
    out = _add_log_returns(df, lags=(1, 5))
    # ret_1 null at index 0; ret_5 null at indices 0..4.
    assert out["ret_1"][0] is None
    assert out["ret_1"][1] is not None
    assert out["ret_5"].null_count() == 5  # not enough history


# ----- _add_wick_features -----


def test_wick_decomposition_basic() -> None:
    """Bar with O=100, H=110, L=95, C=105:
       range=15, body=5, upper=110-105=5, lower=100-95=5,
       upper_norm=5/15=0.333, lower_norm=5/15=0.333."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = _bars("X", "day", [base], opens=[100.0], highs=[110.0],
               lows=[95.0], closes=[105.0])
    out = _add_wick_features(df)
    assert out["bar_range"][0] == 15.0
    assert out["body"][0] == 5.0
    assert out["upper_wick_norm"][0] == pytest.approx(5.0 / 15.0)
    assert out["lower_wick_norm"][0] == pytest.approx(5.0 / 15.0)


def test_wick_zero_range_yields_null_norms() -> None:
    """Limit-locked bar (H==L) → range=0 → wick norms are null, not inf."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = _bars("X", "day", [base], opens=[100.0], highs=[100.0],
               lows=[100.0], closes=[100.0])
    out = _add_wick_features(df)
    assert out["bar_range"][0] == 0.0
    assert out["upper_wick_norm"][0] is None
    assert out["lower_wick_norm"][0] is None


def test_wick_red_candle() -> None:
    """C < O: body still = |C - O|, upper wick is from H down to O, lower from C down to L."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = _bars("X", "day", [base], opens=[105.0], highs=[110.0],
               lows=[95.0], closes=[100.0])
    out = _add_wick_features(df)
    assert out["body"][0] == 5.0  # |C-O|
    # upper = H - max(O,C) = 110 - 105 = 5
    # lower = min(O,C) - L = 100 - 95 = 5
    assert out["upper_wick_norm"][0] == pytest.approx(5.0 / 15.0)
    assert out["lower_wick_norm"][0] == pytest.approx(5.0 / 15.0)


# ----- _add_volume_z -----


def test_volume_z_warmup_rows_are_null() -> None:
    """For the first `window-1` rows, the rolling stats can't be computed →
    `vol_z_20` is null. Use varying volumes so std > 0 once the window fills,
    otherwise we'd hit the (correct) zero-std null path instead."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    vols = [1000 + 10 * i for i in range(25)]   # strictly varying
    df = _bars("X", "day", _seq(base, 25, timedelta(days=1)), vols=vols)
    out = _add_volume_z(df, window=20)
    assert out["vol_z_20"][:19].null_count() == 19
    assert out["vol_z_20"][19] is not None


def test_volume_z_zero_std_yields_null() -> None:
    """Constant volume → std=0 → z is null (no division)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = _bars("X", "day", _seq(base, 25, timedelta(days=1)),
               vols=[1000] * 25)
    out = _add_volume_z(df, window=20)
    assert out["vol_z_20"][20] is None  # std=0 over the trailing window


def test_volume_z_arithmetic() -> None:
    """Volume = [1000]*20 + [2000] → z = (2000 - mean) / std at index 20.

    The trailing window for index 20 is rows 1..20: 19 of 1000s + one 2000."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    n = 21
    vols = [1000] * 20 + [2000]
    df = _bars("X", "day", _seq(base, n, timedelta(days=1)), vols=vols)
    out = _add_volume_z(df, window=20)
    # Compute manually: trailing 20 of row 20 is rows 1..20: 19 × 1000 + 1 × 2000.
    trailing = vols[1:21]
    import statistics
    mean = statistics.mean(trailing)
    std = statistics.stdev(trailing)
    expected = (2000 - mean) / std
    assert out["vol_z_20"][20] == pytest.approx(expected, rel=1e-6)


# ----- _add_time_covariates -----


def test_time_covariates_intraday_session_open() -> None:
    """09:15 IST = market open → normalized minute-of-day = 0 → sin=0, cos=1."""
    open_ist = datetime(2024, 1, 1, 9, 15, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    open_utc = open_ist.astimezone(UTC)
    df = _bars("X", "5m", [open_utc])
    out = _add_time_covariates(df, scale="5m")
    assert out["min_of_day_sin"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["min_of_day_cos"][0] == pytest.approx(1.0, abs=1e-9)


def test_time_covariates_intraday_session_close() -> None:
    """15:30 IST close → normalized = 1.0 → sin(2π) ≈ 0, cos ≈ 1."""
    close_ist = datetime(2024, 1, 1, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    close_utc = close_ist.astimezone(UTC)
    df = _bars("X", "5m", [close_utc])
    out = _add_time_covariates(df, scale="5m")
    assert out["min_of_day_sin"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["min_of_day_cos"][0] == pytest.approx(1.0, abs=1e-9)


def test_time_covariates_day_of_week_distinct() -> None:
    """Mon vs Fri must produce different (sin, cos) pairs."""
    ist = timezone(timedelta(hours=5, minutes=30))
    mon = datetime(2024, 1, 1, 12, 0, tzinfo=ist).astimezone(UTC)
    fri = datetime(2024, 1, 5, 12, 0, tzinfo=ist).astimezone(UTC)
    df = _bars("X", "5m", [mon, fri])
    out = _add_time_covariates(df, scale="5m")
    assert out["day_of_week_sin"][0] != out["day_of_week_sin"][1]


# ----- End-to-end via FeatureEngine -----


def test_feature_engine_round_trip(tmp_path: Path) -> None:
    """Build a small store, write a frame, compute features, check schema."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)  # 15:30 IST close
    n = 100
    closes = [100.0 + i * 0.5 for i in range(n)]
    frame = _bars("X", "day", _seq(base, n, timedelta(days=1)), closes=closes)
    store.write_partition("day", "X", frame)

    engine = FeatureEngine(store)
    out = engine.compute(symbol="X", scale="day",
                         as_of=datetime(2026, 12, 31, tzinfo=UTC))
    assert out.height == n
    # Headline columns present.
    for col in ("ret_1", "ret_60", "bar_range", "vol_z_20",
                "min_of_day_sin", "day_of_week_sin"):
        assert col in out.columns


def test_feature_engine_window_trimming_and_warmup(tmp_path: Path) -> None:
    """Output is trimmed to [window_start, window_end] but rolling features
    are well-defined from the first row — the engine reads the whole partition
    so warm-up is automatic."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    n = 200
    closes = [100.0 + i * 0.1 for i in range(n)]
    vols = [1000 + 10 * i for i in range(n)]   # varying so vol_z std > 0
    store.write_partition(
        "day", "X",
        _bars("X", "day", _seq(base, n, timedelta(days=1)), closes=closes, vols=vols),
    )

    engine = FeatureEngine(store)
    win_start = base + timedelta(days=100)
    win_end = base + timedelta(days=110)
    out = engine.compute(
        symbol="X", scale="day",
        as_of=datetime(2026, 12, 31, tzinfo=UTC),
        window_start=win_start, window_end=win_end,
    )
    assert out.height == 11
    # All rolling features must be defined on the first window row.
    assert out["ret_60"][0] is not None
    assert out["vol_z_20"][0] is not None


def test_feature_engine_respects_as_of_cap(tmp_path: Path) -> None:
    """No row past `as_of` (or in-progress at as_of) should appear."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 9, 15, tzinfo=UTC)
    df = _bars("X", "5m", _seq(base, 20, timedelta(minutes=5)))
    store.write_partition("5m", "X", df)

    engine = FeatureEngine(store)
    cap = datetime(2024, 1, 1, 9, 35, tzinfo=UTC)  # exactly on a bar's effective_close
    out = engine.compute(symbol="X", scale="5m", as_of=cap)
    # Eligible bars: ts in {09:15, 09:20, 09:25, 09:30} (ec ∈ {09:20..09:35}).
    assert out["ts"].max() == datetime(2024, 1, 1, 9, 30, tzinfo=UTC)


def test_feature_engine_returns_empty_when_no_data(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    engine = FeatureEngine(store)
    out = engine.compute(symbol="MISSING", scale="day",
                         as_of=datetime(2026, 12, 31, tzinfo=UTC))
    assert out.height == 0
    # Schema includes the canonical feature columns.
    for col in ("ret_1", "bar_range", "vol_z_20", "min_of_day_sin"):
        assert col in out.columns


def test_feature_engine_bounded_read_does_not_materialize_whole_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Major #11 fix: when `window_start` is set, the engine must pass a
    `start` lower bound to `ParquetStore.read` (not None) so the Parquet
    reader's row-group pruning kicks in.

    We monkeypatch `ParquetStore.read` and capture the `start` arg.
    """
    from daity.data import features as features_mod
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    n = 200
    store.write_partition("day", "X",
                          _bars("X", "day", _seq(base, n, timedelta(days=1)),
                                vols=[1000 + 10 * i for i in range(n)]))

    captured: dict = {}
    real_read = ParquetStore.read

    def spy_read(self, *, symbol, scale, as_of, start=None, end=None):
        captured["start"] = start
        captured["end"] = end
        return real_read(self, symbol=symbol, scale=scale, as_of=as_of,
                         start=start, end=end)

    monkeypatch.setattr(ParquetStore, "read", spy_read)

    engine = FeatureEngine(store)
    win_start = base + timedelta(days=120)
    engine.compute(symbol="X", scale="day",
                   as_of=datetime(2026, 12, 31, tzinfo=UTC),
                   window_start=win_start,
                   window_end=win_start + timedelta(days=10))
    assert captured["start"] is not None, (
        "engine must bound the read with a start lower-bound when "
        "window_start is provided"
    )
    # The bounded start must be EARLIER than window_start (warmup) but not
    # so early that we're effectively reading the whole partition.
    expected_start = win_start - timedelta(days=features_mod.WARMUP_CALENDAR_DAYS["day"])
    assert captured["start"] == expected_start, (
        f"warmup window mis-sized: got {captured['start']}, expected {expected_start}"
    )


def test_feature_engine_unbounded_read_when_window_start_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When `window_start` is None, the engine must read the whole partition
    (no usable lower bound) — we don't want to artificially clip history."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    store.write_partition("day", "X",
                          _bars("X", "day", _seq(base, 50, timedelta(days=1))))

    captured: dict = {}
    real_read = ParquetStore.read
    def spy(self, **kw):
        captured.update(kw)
        return real_read(self, **kw)
    monkeypatch.setattr(ParquetStore, "read", spy)

    FeatureEngine(store).compute(
        symbol="X", scale="day",
        as_of=datetime(2026, 12, 31, tzinfo=UTC),
    )
    assert captured.get("start") is None


def test_feature_engine_spec_filters_groups(tmp_path: Path) -> None:
    """`include=` should restrict which feature groups appear in the output."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    store.write_partition("day", "X",
                          _bars("X", "day", _seq(base, 100, timedelta(days=1))))

    engine = FeatureEngine(store)
    out = engine.compute(
        symbol="X", scale="day",
        as_of=datetime(2026, 12, 31, tzinfo=UTC),
        spec=FeatureSpec(include=("bar_channels", "log_returns")),
    )
    assert "ret_1" in out.columns
    # Wicks / vol-z / time covariates should NOT have been added.
    assert "bar_range" not in out.columns
    assert "vol_z_20" not in out.columns
    assert "min_of_day_sin" not in out.columns
