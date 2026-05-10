"""Feature engineering for daity_v3 (Phase 1.6).

A feature is a deterministic function of `(symbol, ts)` keyed off the
`ParquetStore`. Every public entry point takes an `as_of: datetime` and
respects the project-wide point-in-time correctness rule: no input bar with
`effective_close(scale, ts) > as_of` is ever consulted, and no output row's
`ts > as_of` is ever returned.

Features computed in Phase 1 (intentionally minimal — enough for pretraining):

  - **Bar channels** (passthrough): `open`, `high`, `low`, `close`, `volume`.
  - **Log returns**: `ret_1`, `ret_5`, `ret_15`, `ret_60` — `log(close[t] / close[t-k])`.
  - **Wick decomposition**: `bar_range`, `body`, `upper_wick_norm`, `lower_wick_norm`
    — body and wicks normalized to the bar's range; null when range == 0.
  - **Volume z-score**: `vol_z_20` — z-scored against the trailing 20-bar
    mean+std (`null` while the std is 0 to avoid divide-by-zero).
  - **Time covariates**: `min_of_day_sin`, `min_of_day_cos`,
    `day_of_week_sin`, `day_of_week_cos` — IST trading-clock encoding.

Deferred to later phases (each is its own `_add_*` function when it lands):

  - Microstructure-lite (Lee–Ready trade-direction proxy, Roll's effective
    spread, Amihud illiquidity) — DESIGN §2.4. Phase 3 supervised heads
    benefit from these; pretraining doesn't need them.
  - Static covariates (sector / industry / mcap-bucket / fno_eligible) —
    DESIGN §2.4. SymbolMaster has the columns; once mcap-bucket etc. are
    populated this becomes a simple join.
  - Days-to-expiry — needs the F&O expiry calendar.
  - **RevIN** (Reversible Instance Normalization). Per DESIGN §2.4 this is
    a per-`(symbol, window)` normalization; conceptually it lives in the
    model tokenizer (Phase 2), not here. Adding it as a `transform` rather
    than a `feature` keeps the feature engine deterministic + tz-stable.

Rolling-window features need `--max-lookback` bars of warm-up before
`window_start`. The engine pulls extra history from `ParquetStore` and trims
the output to the requested window so callers always get well-defined
feature values from the first row.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

import polars as pl

from daity.data.candles import (
    CLOSE_TIME_INTERVALS,
    INTERVAL_DURATIONS,
)
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)

# Pre-defined feature group labels for `include=` filtering.
FEATURE_GROUPS: tuple[str, ...] = (
    "bar_channels",
    "log_returns",
    "wicks",
    "volume_z",
    "time_covariates",
)

# Lags used for log returns. Loud-and-stable so a typo elsewhere can't add
# a phantom column.
LOG_RETURN_LAGS: tuple[int, ...] = (1, 5, 15, 60)

# Window for the volume z-score.
VOLUME_Z_WINDOW: int = 20

# Max bars of history any rolling feature looks back. Used to size the
# `read` warmup so the first output row still has a well-defined value.
MAX_LOOKBACK_BARS: int = max(LOG_RETURN_LAGS + (VOLUME_Z_WINDOW,))

# Per-scale calendar-day extent for `MAX_LOOKBACK_BARS` bars.
# Sized to overshoot for safety (weekends + holidays + irregular sessions);
# the IO cost of reading a few extra row groups is negligible vs the
# correctness cost of under-reading and producing null rolling features.
#
# Math (60-bar lookback):
#   5m  : 60 bars × 5min ÷ 75 bars/day ≈ 0.8 trading days × 1.5 (weekend) ≈ 2 cal days
#         + 5d cushion for holiday clusters → 7 calendar days
#   15m : same trading-day count, same cushion → 7
#   60m : 60 bars × 1h ÷ ~6 hours-per-session ≈ 10 trading days × 1.5 → 21 cal days
#   day : 60 trading days × 1.5 (weekends) ≈ 90 cal days; round up for safety
WARMUP_CALENDAR_DAYS: dict[str, int] = {
    "5m":  7,
    "15m": 7,
    "60m": 21,
    "day": 100,
}

# IST is the trading-clock zone; time covariates encode minute-of-day in
# IST so the same encoding works across DST and across timezones for any
# downstream consumer.
IST_TZ = timezone(timedelta(hours=5, minutes=30))

# NSE regular session: 09:15 IST (= 555 minutes after midnight) to 15:30 IST
# (= 930). Used to scale `minute_of_day` for the sin/cos encoding so the
# encoding is dense within the session and zero outside it (intraday) /
# trivial (daily).
SESSION_MIN_START = 9 * 60 + 15
SESSION_MIN_END = 15 * 60 + 30
SESSION_DURATION_MIN = SESSION_MIN_END - SESSION_MIN_START  # 375


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """What to compute. Used by `FeatureEngine.compute(spec=...)`."""

    include: tuple[str, ...] = field(default=FEATURE_GROUPS)

    def __post_init__(self) -> None:
        bad = [g for g in self.include if g not in FEATURE_GROUPS]
        if bad:
            msg = f"Unknown feature groups: {bad}; expected subset of {list(FEATURE_GROUPS)}"
            raise ValueError(msg)


class FeatureEngine:
    """Computes features for one (symbol, scale) over a time window.

    Given the project's per-symbol Parquet partitioning, all rolling features
    can be computed without crossing symbol boundaries — no leakage path
    through cross-symbol joins. Cross-sectional features (e.g., return
    rank within the universe) are computed by a higher-level pipeline that
    fans out per-timestamp reads via `ParquetStore.read_cross_section`.
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    def compute(
        self,
        *,
        symbol: str,
        scale: str,
        as_of: datetime,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        spec: FeatureSpec | None = None,
    ) -> pl.DataFrame:
        """Compute features for `(symbol, scale)` over `[window_start, window_end]`.

        - `as_of`: hard upper-bound on which input bars are eligible. A bar
          with `effective_close > as_of` is excluded entirely (no leakage).
        - `window_start` / `window_end`: optional output bounds. Rolling
          features get a warm-up read of `MAX_LOOKBACK_BARS` bars before
          `window_start` so the first returned row is fully defined.
        - `spec`: which feature groups to include. Default: all of them.

        Returns a frame keyed by `ts`, with the schema described in the
        module docstring.
        """
        spec = spec or FeatureSpec()
        # Bound the read to (window_start - per-scale warmup) ⤳ window_end so
        # rolling features have enough warm-up but we don't materialize the
        # entire 7-year partition for a one-week window. With Parquet's row
        # group pruning (1 row group ≈ 1 month), this turns a "scan 80 row
        # groups" read into "scan 1-3 row groups" for typical training/
        # inference windows. When `window_start` is None we read everything
        # (no usable lower bound).
        if window_start is None:
            read_start = None
        else:
            warmup = timedelta(days=WARMUP_CALENDAR_DAYS[scale])
            read_start = _ensure_utc(window_start) - warmup
        df = self.store.read(
            symbol=symbol, scale=scale,
            start=read_start, end=window_end, as_of=as_of,
        )
        if df.height == 0:
            return _empty_feature_frame()

        # Strict point-in-time gate: any bar whose effective_close exceeds
        # `as_of` is excluded. For intraday, this catches the bar straddling
        # `as_of`; for daily, this is a no-op vs the as_of filter the store
        # already applied.
        df = _filter_by_effective_close(df, scale=scale, as_of=as_of)
        if df.height == 0:
            return _empty_feature_frame()

        # Apply each requested group. Order matters only for chained features.
        if "bar_channels" in spec.include:
            pass  # already in df from the store
        if "log_returns" in spec.include:
            df = _add_log_returns(df, lags=LOG_RETURN_LAGS)
        if "wicks" in spec.include:
            df = _add_wick_features(df)
        if "volume_z" in spec.include:
            df = _add_volume_z(df, window=VOLUME_Z_WINDOW)
        if "time_covariates" in spec.include:
            df = _add_time_covariates(df, scale=scale)

        # Trim to the requested output window. Rolling features computed
        # earlier already saw their warmup rows; trimming now is safe.
        if window_start is not None:
            df = df.filter(pl.col("ts") >= _ensure_utc(window_start))
        if window_end is not None:
            df = df.filter(pl.col("ts") <= _ensure_utc(window_end))
        return df


# ----- Module helpers (each feature group; pure on a sorted-by-ts frame) -----


def _filter_by_effective_close(
    df: pl.DataFrame, *, scale: str, as_of: datetime,
) -> pl.DataFrame:
    """Drop any bar whose effective_close is strictly after `as_of`."""
    cap = _ensure_utc(as_of)
    if scale in CLOSE_TIME_INTERVALS:
        # ts is already the close → ts <= as_of is the gate.
        return df.filter(pl.col("ts") <= cap)
    # Intraday: effective_close = ts + interval_duration. A bar with
    # effective_close == as_of is on the boundary and counts as "available".
    duration = INTERVAL_DURATIONS[scale]
    return df.filter(pl.col("ts") + duration <= cap)


def _add_log_returns(df: pl.DataFrame, *, lags: Iterable[int]) -> pl.DataFrame:
    """Add `ret_<k>` columns: `log(close[t] / close[t-k])`. Null for warm-up rows."""
    exprs = []
    for k in lags:
        prev = pl.col("close").shift(k)
        # log(close / prev). Polars 1.x has `.log` on Expr; using ln() for clarity.
        exprs.append(((pl.col("close") / prev).log()).alias(f"ret_{k}"))
    return df.with_columns(exprs)


def _add_wick_features(df: pl.DataFrame) -> pl.DataFrame:
    """Decompose each bar into range/body/wicks, normalized to range.

    `bar_range = high - low`. When range == 0 (limit-locked or stale tick)
    body+wick normalizations are null rather than infinite.
    """
    rng = (pl.col("high") - pl.col("low")).alias("bar_range")
    body = (pl.col("close") - pl.col("open")).abs().alias("body")
    upper = (pl.col("high") - pl.max_horizontal("open", "close")).alias("_upper")
    lower = (pl.min_horizontal("open", "close") - pl.col("low")).alias("_lower")
    df = df.with_columns([rng, body, upper, lower])
    # Normalize wicks to range, with null when range == 0.
    df = df.with_columns(
        [
            pl.when(pl.col("bar_range") > 0)
              .then(pl.col("_upper") / pl.col("bar_range"))
              .otherwise(None)
              .alias("upper_wick_norm"),
            pl.when(pl.col("bar_range") > 0)
              .then(pl.col("_lower") / pl.col("bar_range"))
              .otherwise(None)
              .alias("lower_wick_norm"),
        ]
    ).drop(["_upper", "_lower"])
    return df


def _add_volume_z(df: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Z-score volume against a trailing `window`-bar mean + std. Null for warm-up."""
    mean = pl.col("volume").rolling_mean(window_size=window, min_samples=window)
    std = pl.col("volume").rolling_std(window_size=window, min_samples=window)
    z = pl.when(std > 0).then((pl.col("volume") - mean) / std).otherwise(None)
    return df.with_columns(z.alias(f"vol_z_{window}"))


def _add_time_covariates(df: pl.DataFrame, *, scale: str) -> pl.DataFrame:
    """Add sin/cos of minute-of-day-IST and day-of-week.

    Daily bars all carry `min_of_day_*` set to the session-close encoding
    (15:30 IST), since intraday-clock features collapse for them.
    `day_of_week_*` is the IST trading day's weekday (Mon=0..Fri=4 only —
    Sat/Sun are not in the data).
    """
    ist = pl.col("ts").dt.convert_time_zone("Asia/Kolkata")
    # minute-of-day in IST. Polars `dt.hour()` returns i8 — cast to i32 before
    # multiplying by 60 or `9 * 60 = 540` overflows silently.
    minute_of_day = ist.dt.hour().cast(pl.Int32) * 60 + ist.dt.minute().cast(pl.Int32)
    if scale in CLOSE_TIME_INTERVALS:
        # Daily: ts is at 15:30 IST close. The covariate adds no signal but
        # we keep the column so the schema is uniform across scales.
        normalized = pl.lit(1.0)  # SESSION_MIN_END / SESSION_DURATION_MIN — sin(2π) = 0, cos = 1
    else:
        normalized = (
            (minute_of_day - SESSION_MIN_START) / SESSION_DURATION_MIN
        )
    angle_min = normalized * (2 * math.pi)
    angle_dow = ist.dt.weekday() * (2 * math.pi / 7)
    return df.with_columns(
        [
            angle_min.sin().alias("min_of_day_sin"),
            angle_min.cos().alias("min_of_day_cos"),
            angle_dow.sin().alias("day_of_week_sin"),
            angle_dow.cos().alias("day_of_week_cos"),
        ]
    )


# ----- Tiny helpers -----


def _ensure_utc(dt: datetime) -> datetime:
    """Coerce naive datetime to tz-aware UTC; pass tz-aware through."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _empty_feature_frame() -> pl.DataFrame:
    """An empty frame with the canonical schema. Used when a window has no data."""
    base = {
        "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "scale": pl.Utf8,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64,
    }
    derived = {
        **{f"ret_{k}": pl.Float64 for k in LOG_RETURN_LAGS},
        "bar_range": pl.Float64, "body": pl.Float64,
        "upper_wick_norm": pl.Float64, "lower_wick_norm": pl.Float64,
        f"vol_z_{VOLUME_Z_WINDOW}": pl.Float64,
        "min_of_day_sin": pl.Float64, "min_of_day_cos": pl.Float64,
        "day_of_week_sin": pl.Float64, "day_of_week_cos": pl.Float64,
    }
    return pl.DataFrame(schema={**base, **derived})
