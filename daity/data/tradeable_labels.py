"""Phase 3X — tradeable-candidate ternary labels.

Recsys-style label mining for a 3-way classifier head:
  +1 → good LONG candidate  (price rises ≥ trigger AND stays up for k minutes)
   0 → neutral
  −1 → good SHORT candidate (symmetric)

The recipe encodes the trader's REAL question: "Can I open this position
at time t, ride the move, and exit cleanly before the move reverses?"

Algorithm per (symbol, anchor t):
  ref = close at t
  trigger_window = bars [t + d_min, t + d_max]  — when the move must happen
  hold_window    = bars [t + d_max, t + d_max + k]  — must persist
  long  iff: max(trigger_window) / ref - 1 > +trigger_threshold
             AND min(hold_window)    / ref - 1 > +floor_threshold
  short iff: min(trigger_window) / ref - 1 < -trigger_threshold
             AND max(hold_window)    / ref - 1 < -floor_threshold
  else neutral (0)

`d_min` is the slippage cushion — ensures predictions are made well before
the trigger fires, so the model can't cheat by mostly predicting events
imminent at `t`. `k` is the hold horizon — ensures the move isn't a
1-tick spike that immediately reverts.

The label store mirrors `daity/data/labels.py` conventions:
  data/tradeable_labels/{config_tag}/{symbol}.parquet
  columns: ts, t, tradeable_label (int8), realized_max_trigger_ret,
           realized_min_trigger_ret, realized_max_hold_ret,
           realized_min_hold_ret, valid_label
The aux *_ret columns let us re-label with different thresholds offline
without re-reading raw OHLCV.

Default config (v1):
  trigger_threshold = 0.005   (0.5% trigger)
  floor_threshold   = 0.003   (0.3% persistence floor)
  d_min_minutes     = 5       (slippage cushion)
  d_max_minutes     = 30      (predict up to 30min ahead)
  k_minutes         = 60      (must hold for 60min after d_max)
  interval          = "5m"    (raw bar granularity)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from daity.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TradeableLabelConfig:
    """One label-mining configuration. Hashing it gives a stable tag for
    the on-disk parquet directory."""
    trigger_threshold: float = 0.005
    floor_threshold:   float = 0.003
    d_min_minutes:     int   = 5
    d_max_minutes:     int   = 30
    k_minutes:         int   = 60
    interval:          str   = "5m"
    version:           int   = 1   # bump on schema/algorithm change

    @property
    def tag(self) -> str:
        """Filesystem-safe tag for this config (used as directory name)."""
        h = hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:8]
        return (
            f"v{self.version}_"
            f"trig{int(self.trigger_threshold*1000)}_"
            f"floor{int(self.floor_threshold*1000)}_"
            f"dmin{self.d_min_minutes}_dmax{self.d_max_minutes}_"
            f"k{self.k_minutes}_{self.interval}_{h}"
        )

    def bars_per_minute(self) -> int:
        if self.interval == "5m": return 5
        if self.interval == "15m": return 15
        if self.interval == "60m": return 60
        if self.interval == "day": return 60 * 24
        msg = f"unknown interval {self.interval!r}"
        raise ValueError(msg)


def _compute_labels_vectorized(
    close: np.ndarray,
    *,
    d_min_bars: int, d_max_bars: int, k_bars: int,
    trigger_threshold: float, floor_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fully-vectorized label computation over a 1-D `close` array.

    Returns `(labels, max_trig_ret, min_trig_ret, max_hold_ret,
              min_hold_ret, valid)`.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    max_trig = np.full(n, np.nan, dtype=np.float32)
    min_trig = np.full(n, np.nan, dtype=np.float32)
    max_hold = np.full(n, np.nan, dtype=np.float32)
    min_hold = np.full(n, np.nan, dtype=np.float32)
    valid    = np.zeros(n, dtype=bool)

    valid_end = n - d_max_bars - k_bars
    if valid_end <= 0:
        return labels, max_trig, min_trig, max_hold, min_hold, valid

    trig_win_size = d_max_bars - d_min_bars + 1
    hold_win_size = k_bars + 1

    if n < trig_win_size + d_min_bars or n < hold_win_size + d_max_bars:
        return labels, max_trig, min_trig, max_hold, min_hold, valid

    trig_windows = sliding_window_view(close, trig_win_size)
    trig_max_per = trig_windows.max(axis=1)
    trig_min_per = trig_windows.min(axis=1)

    hold_windows = sliding_window_view(close, hold_win_size)
    hold_max_per = hold_windows.max(axis=1)
    hold_min_per = hold_windows.min(axis=1)

    ref  = close[:valid_end].astype(np.float64)
    tmax = trig_max_per[d_min_bars : d_min_bars + valid_end].astype(np.float64)
    tmin = trig_min_per[d_min_bars : d_min_bars + valid_end].astype(np.float64)
    hmax = hold_max_per[d_max_bars : d_max_bars + valid_end].astype(np.float64)
    hmin = hold_min_per[d_max_bars : d_max_bars + valid_end].astype(np.float64)

    ref_safe = np.where(ref > 0, ref, np.nan)
    max_trig_ret = tmax / ref_safe - 1.0
    min_trig_ret = tmin / ref_safe - 1.0
    max_hold_ret = hmax / ref_safe - 1.0
    min_hold_ret = hmin / ref_safe - 1.0

    long_mask  = (max_trig_ret >  trigger_threshold) & (min_hold_ret >  floor_threshold)
    short_mask = (min_trig_ret < -trigger_threshold) & (max_hold_ret < -floor_threshold)
    labels[:valid_end] = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)

    max_trig[:valid_end] = max_trig_ret.astype(np.float32)
    min_trig[:valid_end] = min_trig_ret.astype(np.float32)
    max_hold[:valid_end] = max_hold_ret.astype(np.float32)
    min_hold[:valid_end] = min_hold_ret.astype(np.float32)
    valid[:valid_end]    = ~np.isnan(max_trig_ret)
    return labels, max_trig, min_trig, max_hold, min_hold, valid


def build_tradeable_labels_for_symbol(
    *,
    parquet_root: Path,
    symbol: str,
    as_of: datetime,
    cfg: TradeableLabelConfig | None = None,
) -> pl.DataFrame:
    """Build tradeable {-1, 0, +1} labels for one symbol's bars.

    Returns empty frame if no OHLCV data available. Anchor `t` is the bar's
    effective close (matches `daity.data.candles.effective_close`).
    """
    if cfg is None:
        cfg = TradeableLabelConfig()
    path = parquet_root / cfg.interval / f"{symbol}.parquet"
    if not path.exists():
        return pl.DataFrame()
    df = pl.read_parquet(path).sort("ts")
    if df.height == 0:
        return pl.DataFrame()
    # Normalize `ts` to tz-naive (some symbols stored tz-aware, others tz-naive
    # — the parquet store isn't consistent on this).
    if "ts" in df.columns:
        ts_dtype = df.schema["ts"]
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
            df = df.with_columns(
                pl.col("ts").dt.replace_time_zone(None).alias("ts")
            )
    bar_min = cfg.bars_per_minute()
    horizon_min = bar_min + cfg.d_max_minutes + cfg.k_minutes
    cutoff = as_of - timedelta(minutes=horizon_min)
    cutoff_naive = cutoff.astimezone().replace(tzinfo=None) if cutoff.tzinfo else cutoff
    df = df.filter(pl.col("ts") <= cutoff_naive)
    if df.height == 0:
        return pl.DataFrame()

    close = df["close"].to_numpy().astype(np.float32)
    d_min_bars = cfg.d_min_minutes // bar_min
    d_max_bars = cfg.d_max_minutes // bar_min
    k_bars     = cfg.k_minutes     // bar_min
    if d_min_bars < 1 or d_max_bars < d_min_bars or k_bars < 1:
        msg = (
            f"build_tradeable_labels: bad config bars d_min={d_min_bars} "
            f"d_max={d_max_bars} k={k_bars} for interval={cfg.interval!r}"
        )
        raise ValueError(msg)

    labels, max_trig, min_trig, max_hold, min_hold, valid = (
        _compute_labels_vectorized(
            close, d_min_bars=d_min_bars, d_max_bars=d_max_bars, k_bars=k_bars,
            trigger_threshold=cfg.trigger_threshold,
            floor_threshold=cfg.floor_threshold,
        )
    )
    eff_close = df["ts"] + timedelta(minutes=bar_min)
    out = df.select([pl.col("ts")]).with_columns([
        eff_close.alias("t"),
        pl.lit(labels).alias("tradeable_label"),
        pl.lit(max_trig).alias("realized_max_trigger_ret"),
        pl.lit(min_trig).alias("realized_min_trigger_ret"),
        pl.lit(max_hold).alias("realized_max_hold_ret"),
        pl.lit(min_hold).alias("realized_min_hold_ret"),
        pl.lit(valid).alias("valid_label"),
    ])
    return out


def build_tradeable_labels(
    *,
    parquet_root: Path,
    out_root: Path,
    symbols: Iterable[str],
    as_of: datetime,
    cfg: TradeableLabelConfig | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build tradeable labels for many symbols. Returns `{symbol: n_rows}`."""
    if cfg is None:
        cfg = TradeableLabelConfig()
    out_dir = out_root / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sidecar config for reproducibility.
    (out_dir / "_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    counts: dict[str, int] = {}
    for sym in symbols:
        out_path = out_dir / f"{sym}.parquet"
        if out_path.exists() and not overwrite:
            counts[sym] = 0
            continue
        df = build_tradeable_labels_for_symbol(
            parquet_root=parquet_root, symbol=sym, as_of=as_of, cfg=cfg,
        )
        if df.height == 0:
            counts[sym] = 0
            continue
        df.write_parquet(out_path)
        counts[sym] = df.height
    return counts


def class_distribution(out_root: Path, cfg: TradeableLabelConfig) -> dict[str, int | float]:
    """Aggregate the label distribution across all symbols for a config."""
    in_dir = out_root / cfg.tag
    n_long = n_short = n_neutral = n_total = 0
    n_invalid = 0
    for path in in_dir.glob("*.parquet"):
        df = pl.read_parquet(path)
        n_total += df.height
        n_long  += int((df["tradeable_label"] == 1).sum())
        n_short += int((df["tradeable_label"] == -1).sum())
        n_neutral += int((df["tradeable_label"] == 0).sum())
        n_invalid += int((~df["valid_label"]).sum())
    return {
        "n_total": n_total,
        "n_long":  n_long,
        "n_short": n_short,
        "n_neutral": n_neutral,
        "n_invalid": n_invalid,
        "pct_long":  100.0 * n_long  / max(1, n_total),
        "pct_short": 100.0 * n_short / max(1, n_total),
        "pct_neutral": 100.0 * n_neutral / max(1, n_total),
    }
