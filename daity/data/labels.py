"""Phase 3.1 — forward-return label construction.

For each `(symbol, t)` we emit forward log-returns at four horizons
(`PHASE3.md` §3.1, post-amendment 16):

    H ∈ {"30m", "120m", "1d", "2d"}

The label is a close-to-close log-return:

    y_H(sym, t) = log( P_close(sym, t + H) / P_close(sym, t) )

Two implementation paths:

- **Intraday horizons** (`30m`, `120m`) read from the 5m parquet store.
  An anchor at `t` is the moment `effective_close(5m, anchor.ts) == t`,
  i.e., the anchor bar opens at `t - 5m` and closes at `t`. The forward
  bar opens at `t + (N-1)·5m` and closes at `t + H`, where N is the
  bar count for the horizon (6 bars for 30m, 24 for 120m).
- **Daily horizons** (`1d`, `2d`) read from the `day` store. The anchor
  bar's `ts` IS the close time (15:30 IST = 10:00 UTC per
  `daity.data.candles`). The forward bar is the close `n` trading days
  later, located via `NSECalendar.shift`.

Output schema (per `PHASE3.md` §3.6):

    (symbol str, t timestamp, t_horizon timestamp, y_log_return float32,
     y_rank float32, valid_label bool, halt_count int8)

`y_rank` is left null at per-symbol-file write time. The cross-section
sampler in Phase 3.3 fills it at batch-construction time once it has
all symbols at the same `t`. (The ranks are not stable as a per-symbol
artifact — they depend on the universe at `as_of`.)

Halt / circuit detection follows the standard OHLCV pattern: a bar is
"untradeable" if `high == low` (limit-locked) OR `volume == 0` (halt).
`halt_count` counts untradeable bars in the FORWARD window (anchor
itself excluded — the prediction is conditional on observing the
anchor). `valid_label` is `True` iff `halt_count == 0`.

Leakage contract:

- `build_labels(..., as_of=T)` only emits rows where the forward bar's
  `effective_close ≤ T`. Anchors whose forward bar lies past `T` are
  silently dropped, never emitted with a future label.
- The function never reads any row whose `effective_close > T`.
- Bit-equality: a label at `(sym, t)` produced under `as_of=T1` is
  bit-equal to the same label produced under any `as_of=T2 ≥ t + H +
  small_eps`. The leakage test in `tests/leakage/test_label_leakage.py`
  checks 100 random triples for both guarantees.
"""

from __future__ import annotations

from datetime import date as _date_cls
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from daity.data.calendar_nse import NSECalendar
from daity.data.candles import INTERVAL_DURATIONS
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Horizon registries — single source of truth.
HORIZONS_INTRADAY: tuple[str, ...] = ("30m", "120m")
HORIZONS_DAILY: tuple[str, ...] = ("1d", "2d")
ALL_HORIZONS: tuple[str, ...] = HORIZONS_INTRADAY + HORIZONS_DAILY

# Number of forward 5m bars per intraday horizon (each bar = 5min).
HORIZON_BAR_COUNT: dict[str, int] = {
    "30m": 6,    # 6 × 5m = 30m
    "120m": 24,  # 24 × 5m = 120m
}

# Number of forward trading days per daily horizon.
HORIZON_DAY_COUNT: dict[str, int] = {
    "1d": 1,
    "2d": 2,
}

# The scale we read inputs from, per horizon.
HORIZON_SCALE: dict[str, str] = {
    "30m": "5m",
    "120m": "5m",
    "1d": "day",
    "2d": "day",
}

# Output schema. `y_rank` is reserved for cross-section-time fill (Phase 3.3).
OUTPUT_SCHEMA: tuple[str, ...] = (
    "symbol", "t", "t_horizon",
    "y_log_return", "y_rank",
    "valid_label", "halt_count",
)

# Bar duration for 5m, used in many places.
_FIVE_MIN: timedelta = INTERVAL_DURATIONS["5m"]


def _is_untradeable(df: pl.DataFrame) -> pl.Series:
    """Boolean mask for halt/limit-locked bars.

    A bar is untradeable if either:
    - `volume == 0` (no trades — exchange halt or extreme illiquidity)
    - `high == low` AND `volume > 0` (limit-locked: price stuck at circuit)

    Both kinds count toward `halt_count` per `PHASE3.md` §3.5.
    """
    return (df["volume"] == 0) | (df["high"] == df["low"])


def _build_intraday_labels(
    *,
    symbol: str,
    horizon: str,
    raw_store: ParquetStore,
    as_of: datetime,
) -> pl.DataFrame:
    """Forward log-return labels for `30m` / `120m` horizons (5m bars)."""
    if horizon not in HORIZON_BAR_COUNT:
        msg = f"_build_intraday_labels: unknown horizon {horizon!r}"
        raise ValueError(msg)

    n_forward = HORIZON_BAR_COUNT[horizon]

    # Read 5m bars and apply the effective_close gate. ParquetStore.read
    # filters by `ts <= as_of`, but for intraday scales the bar at `ts`
    # only closes at `ts + 5min`. Same pattern as `OHLCVPretrainDataset`
    # and `FeatureEngine`: enforce `effective_close <= as_of` here.
    df = raw_store.read(symbol=symbol, scale="5m", as_of=as_of)
    if df.height == 0:
        return _empty_output(symbol)
    df = df.filter(pl.col("ts") + _FIVE_MIN <= as_of)
    if df.height < n_forward + 1:
        return _empty_output(symbol)

    # Sort defensively — store should already be sorted but pretraining
    # depends on it and this is cheap.
    df = df.sort("ts")

    # Anchor index range: 0..(height-n_forward-1) such that forward index
    # i+n_forward exists. The anchor's effective_close is `anchor.ts + 5m`;
    # the forward bar's effective_close is `forward.ts + 5m`. We additionally
    # require both anchor and forward to lie in the same trading session
    # (no overnight crossing) — enforced via exact-delta check below.
    ts_arr = df["ts"].to_numpy()
    close_arr = df["close"].to_numpy().astype(np.float64)
    untradeable_arr = _is_untradeable(df).to_numpy()

    # Same-session check: forward.ts == anchor.ts + n_forward * 5m EXACTLY.
    # If the data has a session-boundary gap (e.g., last bar of Mon to first
    # bar of Tue), the diff will be ~17h45m instead of n_forward*5m, and we
    # drop the anchor.
    expected_delta = np.timedelta64(n_forward * 5, "m")
    actual_delta = ts_arr[n_forward:] - ts_arr[:-n_forward]

    anchor_idx = np.where(actual_delta == expected_delta)[0]
    if anchor_idx.size == 0:
        return _empty_output(symbol)

    forward_idx = anchor_idx + n_forward
    anchor_close = close_arr[anchor_idx]
    forward_close = close_arr[forward_idx]

    # Drop rows where either close is non-positive (log undefined).
    valid_close = (anchor_close > 0) & (forward_close > 0)
    if not valid_close.any():
        return _empty_output(symbol)

    anchor_idx = anchor_idx[valid_close]
    forward_idx = forward_idx[valid_close]
    anchor_close = anchor_close[valid_close]
    forward_close = forward_close[valid_close]

    y_log_return = np.log(forward_close / anchor_close).astype(np.float32)

    # Halt count over the FORWARD window (anchor.ts+5m .. forward.ts).
    # That's bars at indices [anchor_idx + 1, ..., forward_idx]. The anchor
    # itself is excluded — the prediction is conditional on observing it.
    halt_count = np.zeros(anchor_idx.size, dtype=np.int8)
    for offset in range(1, n_forward + 1):
        halt_count = halt_count + untradeable_arr[anchor_idx + offset].astype(np.int8)
    halt_count = np.minimum(halt_count, 127).astype(np.int8)
    valid_label = halt_count == 0

    # `t` is the anchor's effective_close. `t_horizon` is the forward's.
    five_min_us = np.timedelta64(5, "m").astype("timedelta64[us]")
    t_us = (ts_arr[anchor_idx].astype("datetime64[us]") + five_min_us)
    t_horizon_us = (ts_arr[forward_idx].astype("datetime64[us]") + five_min_us)

    return pl.DataFrame({
        "symbol":       pl.Series([symbol] * anchor_idx.size, dtype=pl.Utf8),
        "t":            pl.Series(t_us).dt.replace_time_zone("UTC"),
        "t_horizon":    pl.Series(t_horizon_us).dt.replace_time_zone("UTC"),
        "y_log_return": pl.Series(y_log_return, dtype=pl.Float32),
        "y_rank":       pl.Series([None] * anchor_idx.size, dtype=pl.Float32),
        "valid_label":  pl.Series(valid_label, dtype=pl.Boolean),
        "halt_count":   pl.Series(halt_count, dtype=pl.Int8),
    }).select(list(OUTPUT_SCHEMA))


def _build_daily_labels(
    *,
    symbol: str,
    horizon: str,
    raw_store: ParquetStore,
    calendar: NSECalendar,
    as_of: datetime,
) -> pl.DataFrame:
    """Forward log-return labels for `1d` / `2d` horizons (daily bars)."""
    if horizon not in HORIZON_DAY_COUNT:
        msg = f"_build_daily_labels: unknown horizon {horizon!r}"
        raise ValueError(msg)

    n_forward = HORIZON_DAY_COUNT[horizon]
    df = raw_store.read(symbol=symbol, scale="day", as_of=as_of)
    if df.height < n_forward + 1:
        return _empty_output(symbol)
    df = df.sort("ts")

    ts_arr = df["ts"].to_numpy()           # tz-aware UTC datetime64[us]
    close_arr = df["close"].to_numpy().astype(np.float64)
    untradeable_arr = _is_untradeable(df).to_numpy()

    # Build an anchor → forward index map by walking the calendar. For each
    # anchor at trading-day D, the forward is the close `n_forward` trading
    # days later (per NSE calendar — skips weekends, holidays, exchange
    # closures). The map is positional, not by-date, so out-of-roster
    # holiday gaps are honored.
    n = df.height
    anchor_idx_list: list[int] = []
    forward_idx_list: list[int] = []

    # ts_arr[i] is the close timestamp (15:30 IST) of trading day D_i. Build
    # a date → index map for O(1) forward lookup.
    ts_dates = [_ts_to_utc_date(ts) for ts in ts_arr]
    date_to_idx = {d: i for i, d in enumerate(ts_dates)}

    cap_date = _ts_to_utc_date(np.datetime64(as_of.replace(tzinfo=None)))

    for i, anchor_date in enumerate(ts_dates):
        # forward_date = `n_forward` trading days strictly forward.
        forward_date = calendar.shift(anchor_date, n_forward, as_of=cap_date)
        if forward_date is None:
            continue
        j = date_to_idx.get(forward_date)
        if j is None:
            continue
        anchor_idx_list.append(i)
        forward_idx_list.append(j)

    if not anchor_idx_list:
        return _empty_output(symbol)

    anchor_idx = np.asarray(anchor_idx_list, dtype=np.int64)
    forward_idx = np.asarray(forward_idx_list, dtype=np.int64)

    anchor_close = close_arr[anchor_idx]
    forward_close = close_arr[forward_idx]
    valid_close = (anchor_close > 0) & (forward_close > 0)
    if not valid_close.any():
        return _empty_output(symbol)
    anchor_idx = anchor_idx[valid_close]
    forward_idx = forward_idx[valid_close]
    anchor_close = anchor_close[valid_close]
    forward_close = forward_close[valid_close]

    y_log_return = np.log(forward_close / anchor_close).astype(np.float32)

    # Halt count over the forward trading days (forward bar inclusive,
    # anchor exclusive). For 1d that's just the forward bar; for 2d it's
    # the bar at `anchor_idx + 1` plus the bar at `forward_idx`. We use
    # the positional indices in the dataframe directly.
    halt_count = np.zeros(anchor_idx.size, dtype=np.int8)
    for offset in range(1, n_forward + 1):
        idx = anchor_idx + offset
        # Guard: idx <= forward_idx by construction; idx < n by the
        # earlier `+1` slack on `df.height`.
        in_bounds = idx < n
        halt_count = halt_count + np.where(
            in_bounds, untradeable_arr[np.minimum(idx, n - 1)].astype(np.int8), 0,
        ).astype(np.int8)
    halt_count = np.minimum(halt_count, 127).astype(np.int8)
    valid_label = halt_count == 0

    # For daily, `ts` IS the close time — no offset needed.
    t_us = ts_arr[anchor_idx].astype("datetime64[us]")
    t_horizon_us = ts_arr[forward_idx].astype("datetime64[us]")

    return pl.DataFrame({
        "symbol":       pl.Series([symbol] * anchor_idx.size, dtype=pl.Utf8),
        "t":            pl.Series(t_us).dt.replace_time_zone("UTC"),
        "t_horizon":    pl.Series(t_horizon_us).dt.replace_time_zone("UTC"),
        "y_log_return": pl.Series(y_log_return, dtype=pl.Float32),
        "y_rank":       pl.Series([None] * anchor_idx.size, dtype=pl.Float32),
        "valid_label":  pl.Series(valid_label, dtype=pl.Boolean),
        "halt_count":   pl.Series(halt_count, dtype=pl.Int8),
    }).select(list(OUTPUT_SCHEMA))


def build_labels(
    *,
    symbol: str,
    horizon: str,
    raw_store: ParquetStore,
    calendar: NSECalendar,
    as_of: datetime,
) -> pl.DataFrame:
    """Public entry point — dispatches by horizon family.

    Returns an empty (well-typed) frame on insufficient data. Never raises
    on a missing partition (the symbol may legitimately have no data).
    """
    if horizon in HORIZONS_INTRADAY:
        return _build_intraday_labels(
            symbol=symbol, horizon=horizon, raw_store=raw_store, as_of=as_of,
        )
    if horizon in HORIZONS_DAILY:
        return _build_daily_labels(
            symbol=symbol, horizon=horizon, raw_store=raw_store,
            calendar=calendar, as_of=as_of,
        )
    msg = (
        f"build_labels: unknown horizon {horizon!r}; "
        f"expected one of {list(ALL_HORIZONS)}"
    )
    raise ValueError(msg)


def precompute_labels(
    *,
    raw_store: ParquetStore,
    label_root: Path,
    symbol: str,
    horizon: str,
    calendar: NSECalendar,
    as_of: datetime,
    force: bool = False,
) -> int:
    """Compute and write the per-`(symbol, horizon)` label parquet.

    Output: `{label_root}/{horizon}/{symbol}.parquet`. Idempotent: if the
    output exists with a matching schema, returns the existing row count
    without recomputing (unless `force=True`).
    """
    out_path = label_root / horizon / f"{symbol}.parquet"

    if not force and out_path.exists():
        try:
            existing_cols = pl.scan_parquet(out_path).collect_schema().names()
            if set(existing_cols) >= set(OUTPUT_SCHEMA):
                n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
                log.debug(
                    "precompute_labels: %s/%s already at %s (%d rows); skipping",
                    horizon, symbol, out_path, n,
                )
                return int(n)
        except Exception:
            pass

    df = build_labels(
        symbol=symbol, horizon=horizon,
        raw_store=raw_store, calendar=calendar, as_of=as_of,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    log.info(
        "precompute_labels: %s/%s → %s (%d rows; %d invalid)",
        horizon, symbol, out_path,
        df.height,
        int((~df["valid_label"]).sum()) if df.height > 0 else 0,
    )
    return df.height


# ----- helpers -----

def _empty_output(symbol: str) -> pl.DataFrame:
    """A well-typed, zero-row frame matching `OUTPUT_SCHEMA`."""
    return pl.DataFrame(
        schema={
            "symbol":       pl.Utf8,
            "t":            pl.Datetime("us", "UTC"),
            "t_horizon":    pl.Datetime("us", "UTC"),
            "y_log_return": pl.Float32,
            "y_rank":       pl.Float32,
            "valid_label":  pl.Boolean,
            "halt_count":   pl.Int8,
        },
    ).select(list(OUTPUT_SCHEMA))


def _ts_to_utc_date(ts: np.datetime64) -> _date_cls:
    """Best-effort numpy datetime64 → date conversion (tz-naive UTC)."""
    py = ts.astype("datetime64[us]").astype("int64")
    sec = py // 1_000_000
    dt = datetime.fromtimestamp(sec, tz=None)
    return _date_cls(dt.year, dt.month, dt.day)
