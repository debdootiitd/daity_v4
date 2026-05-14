"""Phase 3.3 — cross-symbol batch construction (`PHASE3.md` §4).

Four horizon-specific iterators, each yielding one cross-section per call:

- `iter_30m()`  — ~69 cross-sections per trading day (5m anchors within
  09:20..15:00 UTC effective-close, after the 30m forward window fits).
- `iter_120m()` — ~52 per day (anchors within 09:20..13:35).
- `iter_1d()`   — 1 per trading day (anchor = 15:30 IST close).
- `iter_2d()`   — 1 per trading day (anchor = 15:30 IST close).

The training loop in Phase 3.4 (`posttrain.py`) draws horizons via
weighted-random sampling proportional to natural cadence (70:52:1:1 ≈
the per-trading-day anchor counts) — see `PHASE3.md` §4.3.

Each yielded `CrossSection` carries:

- `t`            — the anchor's effective_close (the moment all inputs are known).
- `horizon`      — one of `{"30m", "120m", "1d", "2d"}`.
- `symbols`      — the in-universe-at-`t` valid subset (insufficient-history
                   or halt-in-forward-window symbols are dropped).
- `inputs`       — `dict[scale, Tensor(N, window_bars[scale], 18)]`.
- `labels`       — `Tensor(N,)` of forward log-returns at this horizon.
- `ranks`        — `Tensor(N,)` of cross-sectional rank ∈ [0, 1] of labels.
- `valid_mask`   — `Tensor(N,)` bools (always True for emitted rows; the
                   sampler already filters before yielding).
- `vol_bucket`   — `Tensor(N,)` int8 ∈ [0, K-1], the realized-vol quintile
                   per `PHASE3.md` §6.4. K=5 by default.

Leakage contract (locked):

- Every input bar in the per-scale window has `effective_close <
  anchor_t` (STRICT less-than; same -1μs trick the XGBoost baseline
  uses to make `join_asof(backward)` correct).
- The label at `(sym, anchor_t)` was produced by `daity-build-labels`
  under the same `as_of`; the label module already enforces
  `t_horizon ≤ as_of`.
- Vol-bucket lookback uses only bars with `effective_close <
  anchor_t`.

Cross-sections with fewer than `drop_invalid_threshold * |universe|`
valid symbols are skipped (default threshold: 0.5).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from daity.data.candles import INTERVAL_DURATIONS
from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

log = get_logger(__name__)


# Vol-bucket lookback per horizon (in bars at the canonical scale).
# `PHASE3.md` §6.4 + amendment 16:
# - 30m  uses ~1 session of 5m bars (78 bars at the 5m scale).
# - 120m uses ~2 sessions of 5m bars (156 bars).
# - 1d/2d use 20 trading days (read from the `day` scale).
VOL_LOOKBACK_BARS: dict[str, int] = {
    "30m":  78,
    "120m": 156,
    "1d":   20,
    "2d":   20,
}

# Which scale the vol lookback uses (5m for intraday, day for daily).
VOL_LOOKBACK_SCALE: dict[str, str] = {
    "30m":  "5m",
    "120m": "5m",
    "1d":   "day",
    "2d":   "day",
}

# Default per-scale window lengths in input bars. Matches Phase 2's
# pretrain config so the SSL backbone sees the same shape at inference.
DEFAULT_WINDOW_BARS: dict[str, int] = {
    "5m":  256,
    "15m": 192,
    "60m": 168,
    "day": 64,
}

# Scales in canonical order — same as the SSL tokenizer ordering.
SCALES_ORDERED: tuple[str, ...] = ("5m", "15m", "60m", "day")

# Hard floor on cross-section size. Below this, Spearman rank-IC is too
# noisy (architect-review Minor-3 / `daity.eval.metrics.cross_sectional_rank_ic`
# uses `min_cross_section_size=10`) and K=5 vol-buckets can't be populated.
MIN_SYMBOLS_FOR_CROSS_SECTION: int = 10


@dataclass(slots=True)
class CrossSection:
    """One emitted cross-section. See module docstring for field semantics."""

    t: datetime
    horizon: str
    symbols: list[str]
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    ranks: torch.Tensor
    valid_mask: torch.Tensor
    vol_bucket: torch.Tensor
    # Auxiliary engineered features (top-50 XGB importance), shape (N, 50).
    # Populated only when the sampler is constructed with `aux_root` set.
    # Otherwise: zeros tensor of shape (N, 50) — model code can decide to
    # gate on aux being all-zero or check a separate "use_aux" flag.
    aux_features: torch.Tensor | None = None
    # Phase 4: OB snapshot windows per symbol, shape (N, K, OB_N_CHANNELS).
    # Populated only when the sampler is constructed with `ob_root` set.
    # Per-row `ob_mask: (N,)` boolean indicates which symbols actually
    # carry OB at this anchor (vs symbols that lack OB or have stale data).
    # `ob_snapshot_window` is zero-padded for masked-off rows.
    ob_snapshot_window: torch.Tensor | None = None
    ob_mask: torch.Tensor | None = None
    # Phase 3X: tradeable-candidate ternary labels per row, shape (N,) int8
    # in {-1, 0, +1}. Populated only when the sampler is constructed with
    # `tradeable_root` set. Used by TradeableHead training + eval.
    # `tradeable_valid: (N,)` bool — True if the anchor has a valid label
    # (i.e., the forward horizon was observable + valid_label=True at build).
    tradeable_label: torch.Tensor | None = None
    tradeable_valid: torch.Tensor | None = None
    # Phase 5: realistic per-symbol realized PnL under target/stop/timeout exit
    # rule + entry slippage (e.g. entry at high(+5m bar) for LONG).
    # Net of cost. Populated when the tradeable parquet has these columns.
    realized_pnl_long:  torch.Tensor | None = None
    realized_pnl_short: torch.Tensor | None = None


@dataclass(slots=True)
class _ScaleCache:
    """Per-(symbol, scale) cached numpy array of 18-channel features
    plus the effective_close index for binary lookup."""

    eff_close: np.ndarray   # datetime64[us], shape (N,)
    features: np.ndarray    # float32, shape (N, 18)


@dataclass(slots=True)
class _LabelCache:
    """Per-(symbol, horizon) cached label arrays."""

    t: np.ndarray              # datetime64[us], anchor timestamps
    y_log_return: np.ndarray   # float32
    valid_label: np.ndarray    # bool


@dataclass(slots=True)
class _AuxCache:
    """Per-symbol cached auxiliary feature array (top-50 XGB features)."""
    t: np.ndarray         # datetime64[us], anchor timestamps
    features: np.ndarray  # float32, shape (N, 50)


@dataclass(slots=True)
class _OBCache:
    """Per-symbol cached OB snapshot stream."""
    t: np.ndarray         # datetime64[us], snapshot timestamps
    features: np.ndarray  # float32, shape (N_snap, OB_N_CHANNELS)


@dataclass(slots=True)
class _TradeableCache:
    """Per-symbol cached tradeable-label time-series."""
    t: np.ndarray         # datetime64[us], anchor times
    label: np.ndarray     # int8, shape (N,) in {-1, 0, +1}
    valid: np.ndarray     # bool
    # Optional realized PnL columns (Phase 5 direct-PnL training). May be
    # None when the parquet doesn't include them.
    pnl_long: np.ndarray | None = None    # float32
    pnl_short: np.ndarray | None = None   # float32


@dataclass
class CrossSectionalSampler:
    """Builds per-anchor cross-sections from the universe of symbols.

    Construct once per (fold, train|val|test window); iterate via the
    four `iter_*` methods. Caching is per-symbol; the first lookup
    reads the parquet, subsequent lookups are O(1) on cached numpy.

    The sampler is single-process. Multi-worker DataLoader integration
    lives in `posttrain.py` — workers can be spawned with disjoint
    universe shards if memory pressure ever bites. At full scale the
    feature cache is ~13 MB / symbol × 205 symbols ≈ 2.7 GB across all
    four scales (5m dominates), plus ~150 MB of labels — comfortable
    on H200 (143 GB) and fine on local 64 GB RAM machines.
    """

    feature_store: ParquetStore
    label_root: Path
    universe: list[str]
    as_of: datetime
    window_bars: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_WINDOW_BARS))
    # Lower bound on `t`. Must be UTC tz-aware (or any tz-aware datetime;
    # we normalize via .astimezone(UTC) before stripping tzinfo).
    min_end_ts: datetime | None = None
    drop_invalid_threshold: float = 0.5
    n_vol_buckets: int = 5
    # Phase 3.4c: which label column to use as the supervised target.
    # `raw` reads `y_log_return` from `data/labels/{h}/{sym}.parquet`.
    # `resid` reads `y_resid` from `data/labels_resid/{h}/{sym}.parquet`
    # (built by `daity-build-residual-labels`). Per-horizon: 30m/120m
    # stay raw, 1d/2d use resid by default in the iter-5 ship config.
    label_mode: str = "raw"
    label_root_resid: Path | None = None
    # Horizons for which to use `resid` mode (others fall back to `raw`).
    # Only consulted when `label_mode == "resid"`.
    resid_horizons: tuple[str, ...] = ("1d", "2d")
    # Optional aux-feature root (top-50 XGB features per symbol). When set,
    # the sampler loads `{aux_root}/{symbol}.parquet` per symbol and the
    # emitted CrossSection has `aux_features: Tensor(N, 50)`. When None,
    # `aux_features` is `None` on every emitted CrossSection.
    aux_root: Path | None = None
    # Phase 4: OB parquet root (`{ob_root}/{symbol}.parquet`). When set,
    # the sampler attempts to load + engineer OB features per symbol
    # and packs an (N, K, OB_N_CHANNELS) tensor on each CrossSection
    # plus an (N,) ob_mask. K-snapshot window per symbol is filtered
    # for staleness (>5 min from anchor = mask off).
    ob_root: Path | None = None
    ob_k_snapshots: int = 64
    ob_max_staleness_minutes: int = 5
    # Phase 3X: tradeable-label root (`{root}/{symbol}.parquet` with
    # columns t, tradeable_label, valid_label). When set, the sampler
    # attaches `tradeable_label` and `tradeable_valid` to each CrossSection.
    tradeable_root: Path | None = None

    # Lazy per-symbol caches.
    _feat_cache: dict[tuple[str, str], _ScaleCache] = field(
        default_factory=dict, init=False, repr=False,
    )
    _label_cache: dict[tuple[str, str], _LabelCache] = field(
        default_factory=dict, init=False, repr=False,
    )
    _aux_cache: dict[str, _AuxCache] = field(
        default_factory=dict, init=False, repr=False,
    )
    _ob_cache: dict[str, _OBCache] = field(
        default_factory=dict, init=False, repr=False,
    )
    _tradeable_cache: dict[str, _TradeableCache] = field(
        default_factory=dict, init=False, repr=False,
    )

    # ----- Caches -----

    def _scale(self, symbol: str, scale: str) -> _ScaleCache:
        key = (symbol, scale)
        cached = self._feat_cache.get(key)
        if cached is not None:
            return cached
        df = self.feature_store.read(symbol=symbol, scale=scale, as_of=self.as_of)
        if df.height == 0:
            cached = _ScaleCache(
                eff_close=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, 18), dtype=np.float32),
            )
            self._feat_cache[key] = cached
            return cached
        df = df.sort("ts")
        # Apply the effective_close gate explicitly for intraday scales
        # (matches labels.py + xgboost_baseline.py).
        if scale != "day":
            df = df.filter(pl.col("ts") + INTERVAL_DURATIONS[scale] <= self.as_of)
        ts_arr = df["ts"].to_numpy()
        if scale == "day":
            eff = ts_arr.astype("datetime64[us]")
        else:
            duration_us = int(INTERVAL_DURATIONS[scale].total_seconds() * 1_000_000)
            eff = ts_arr.astype("datetime64[us]") + np.timedelta64(duration_us, "us")
        feats = np.stack(
            [df[c].to_numpy().astype(np.float32) for c in CHANNEL_ORDER_18],
            axis=-1,
        )
        cached = _ScaleCache(eff_close=eff, features=feats)
        self._feat_cache[key] = cached
        return cached

    def _labels(self, symbol: str, horizon: str) -> _LabelCache:
        key = (symbol, horizon)
        cached = self._label_cache.get(key)
        if cached is not None:
            return cached
        # Phase 3.4c: choose label root + target column based on label_mode.
        use_resid = (
            self.label_mode == "resid"
            and self.label_root_resid is not None
            and horizon in self.resid_horizons
        )
        if use_resid:
            path = Path(self.label_root_resid) / horizon / f"{symbol}.parquet"
            target_col = "y_resid"
        else:
            path = self.label_root / horizon / f"{symbol}.parquet"
            target_col = "y_log_return"
        if not path.exists():
            cached = _LabelCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                y_log_return=np.zeros(0, dtype=np.float32),
                valid_label=np.zeros(0, dtype=bool),
            )
            self._label_cache[key] = cached
            return cached
        df = pl.read_parquet(path).sort("t")
        # Apply as_of cap (labels are pre-cut at build time, but be defensive).
        df = df.filter(pl.col("t_horizon") <= self.as_of)
        if self.min_end_ts is not None:
            df = df.filter(pl.col("t") >= self.min_end_ts)
        if target_col not in df.columns:
            # resid parquet missing the y_resid column — fall back to raw.
            log.warning(
                "_labels(%s, %s): %s missing %s column; falling back to "
                "y_log_return.", symbol, horizon, path, target_col,
            )
            target_col = "y_log_return"
        cached = _LabelCache(
            t=df["t"].to_numpy().astype("datetime64[us]"),
            y_log_return=df[target_col].to_numpy().astype(np.float32),
            valid_label=df["valid_label"].to_numpy().astype(bool),
        )
        self._label_cache[key] = cached
        return cached

    def _aux(self, symbol: str) -> _AuxCache:
        """Load + cache one symbol's aux features (top-50 XGB columns).

        Returns an empty `_AuxCache` if `self.aux_root` is None or the
        symbol's aux parquet is missing (caller must handle gracefully).
        """
        cached = self._aux_cache.get(symbol)
        if cached is not None:
            return cached
        if self.aux_root is None:
            cached = _AuxCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, 50), dtype=np.float32),
            )
            self._aux_cache[symbol] = cached
            return cached
        path = Path(self.aux_root) / f"{symbol}.parquet"
        if not path.exists():
            cached = _AuxCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, 50), dtype=np.float32),
            )
            self._aux_cache[symbol] = cached
            return cached
        df = pl.read_parquet(path)
        df = df.filter(pl.col("anchor_ts") < self.as_of)
        if df.height == 0:
            cached = _AuxCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, 50), dtype=np.float32),
            )
        else:
            ts = df["anchor_ts"].to_numpy().astype("datetime64[us]")
            f_cols = [f"f_{i}" for i in range(50)]
            feats = df.select(f_cols).to_numpy().astype(np.float32)
            cached = _AuxCache(t=ts, features=feats)
        self._aux_cache[symbol] = cached
        return cached

    def _ob(self, symbol: str) -> _OBCache:
        """Load + engineer + cache OB features for one symbol.

        Returns an empty `_OBCache` if `self.ob_root` is None or no OB
        parquet for this symbol (most symbols outside the OB-coverage
        subset of ~206/208).
        """
        cached = self._ob_cache.get(symbol)
        if cached is not None:
            return cached
        from daity.data.ob_features import (
            OB_N_CHANNELS, compute_ob_features,
        )
        if self.ob_root is None:
            cached = _OBCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, OB_N_CHANNELS), dtype=np.float32),
            )
            self._ob_cache[symbol] = cached
            return cached
        path = Path(self.ob_root) / f"{symbol}.parquet"
        if not path.exists():
            cached = _OBCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, OB_N_CHANNELS), dtype=np.float32),
            )
            self._ob_cache[symbol] = cached
            return cached
        raw = pl.read_parquet(path)
        # Apply as_of cap on the raw OB stream (defensive — the BQ window
        # may have shifted between ingest and now). OB parquet's `ts` is
        # tz-naive (UTC by convention); normalize as_of to naive for the
        # comparison.
        if "ts" in raw.columns:
            as_of_naive = self.as_of.astimezone(UTC).replace(tzinfo=None) if self.as_of.tzinfo else self.as_of
            raw = raw.filter(pl.col("ts") < as_of_naive)
        feats = compute_ob_features(raw)
        if feats.height == 0:
            cached = _OBCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                features=np.zeros((0, OB_N_CHANNELS), dtype=np.float32),
            )
        else:
            ts = feats["ts"].to_numpy().astype("datetime64[us]")
            f_cols = [f"f_{i}" for i in range(OB_N_CHANNELS)]
            arr = feats.select(f_cols).to_numpy().astype(np.float32)
            cached = _OBCache(t=ts, features=arr)
        self._ob_cache[symbol] = cached
        return cached

    def _tradeable(self, symbol: str) -> _TradeableCache:
        """Load + cache one symbol's tradeable-label time-series.

        Returns empty cache if `self.tradeable_root` is None or the symbol's
        parquet is missing.
        """
        cached = self._tradeable_cache.get(symbol)
        if cached is not None:
            return cached
        if self.tradeable_root is None:
            cached = _TradeableCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                label=np.zeros(0, dtype=np.int8),
                valid=np.zeros(0, dtype=bool),
            )
            self._tradeable_cache[symbol] = cached
            return cached
        path = Path(self.tradeable_root) / f"{symbol}.parquet"
        if not path.exists():
            cached = _TradeableCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                label=np.zeros(0, dtype=np.int8),
                valid=np.zeros(0, dtype=bool),
            )
            self._tradeable_cache[symbol] = cached
            return cached
        df = pl.read_parquet(path)
        if "t" in df.columns:
            t_dtype = df.schema["t"]
            if isinstance(t_dtype, pl.Datetime) and t_dtype.time_zone is not None:
                df = df.with_columns(
                    pl.col("t").dt.replace_time_zone(None).alias("t")
                )
        as_of_naive = (
            self.as_of.astimezone(UTC).replace(tzinfo=None)
            if self.as_of.tzinfo else self.as_of
        )
        df = df.filter(pl.col("t") < as_of_naive)
        if df.height == 0:
            cached = _TradeableCache(
                t=np.zeros(0, dtype="datetime64[us]"),
                label=np.zeros(0, dtype=np.int8),
                valid=np.zeros(0, dtype=bool),
            )
        else:
            t = df["t"].to_numpy().astype("datetime64[us]")
            lab = df["tradeable_label"].to_numpy().astype(np.int8)
            valid = df["valid_label"].to_numpy().astype(bool)
            pnl_l = (df["realized_pnl_long"].to_numpy().astype(np.float32)
                     if "realized_pnl_long" in df.columns else None)
            pnl_s = (df["realized_pnl_short"].to_numpy().astype(np.float32)
                     if "realized_pnl_short" in df.columns else None)
            cached = _TradeableCache(t=t, label=lab, valid=valid,
                                      pnl_long=pnl_l, pnl_short=pnl_s)
        self._tradeable_cache[symbol] = cached
        return cached

    def _tradeable_at(
        self, symbol: str, anchor_ts: np.datetime64,
    ) -> tuple[int, bool, float, float] | None:
        """Return `(label, valid, pnl_long, pnl_short)` at this anchor or None.

        `pnl_long`/`pnl_short` are 0.0 if the parquet doesn't carry them.
        """
        if self.tradeable_root is None:
            return None
        tc = self._tradeable(symbol)
        if tc.t.size == 0:
            return None
        pos = int(np.searchsorted(tc.t, anchor_ts, side="left"))
        if pos >= tc.t.size or tc.t[pos] != anchor_ts:
            return None
        pnl_l = float(tc.pnl_long[pos])  if tc.pnl_long  is not None else 0.0
        pnl_s = float(tc.pnl_short[pos]) if tc.pnl_short is not None else 0.0
        return int(tc.label[pos]), bool(tc.valid[pos]), pnl_l, pnl_s

    def _ob_at(
        self, symbol: str, anchor_ts: np.datetime64,
    ) -> np.ndarray | None:
        """Return the K most-recent OB snapshots strictly before `anchor_ts`,
        or `None` if symbol lacks OB / has warmup gap / latest stale.

        Output shape (when not None): `(K, OB_N_CHANNELS)` float32.
        """
        if self.ob_root is None:
            return None
        oc = self._ob(symbol)
        if oc.t.size == 0:
            return None
        from daity.data.ob_features import lookup_snapshot_window
        return lookup_snapshot_window(
            feature_ts=oc.t, feature_mat=oc.features,
            anchor_ts=anchor_ts, k_snapshots=self.ob_k_snapshots,
            max_staleness_us=self.ob_max_staleness_minutes * 60 * 1_000_000,
        )

    def _aux_at(self, symbol: str, anchor_ts: np.datetime64) -> np.ndarray | None:
        """Return the 50-dim aux feature row at `anchor_ts` for `symbol`,
        or None if the timestamp isn't present (warmup gap or missing file).

        Exact-match lookup via binary search — aux features are pre-computed
        on a 5m grid, so anchors at 30m / 120m / 1d boundaries land on
        that grid exactly.
        """
        if self.aux_root is None:
            return None
        ac = self._aux(symbol)
        if ac.t.size == 0:
            return None
        pos = int(np.searchsorted(ac.t, anchor_ts, side="left"))
        if pos >= ac.t.size or ac.t[pos] != anchor_ts:
            return None
        return ac.features[pos]

    # ----- Per-symbol anchor lookup -----

    def _strict_last_idx(self, eff_close: np.ndarray, anchor_ts: np.datetime64) -> int:
        """Largest index `i` with `eff_close[i] < anchor_ts`, or -1 if none.

        STRICT less-than — matches the xgboost_baseline `-1μs` trick.
        Implemented via `searchsorted(side='left')` which returns the
        first index ≥ anchor_ts; the one before that is the last with `<`.
        """
        if eff_close.size == 0:
            return -1
        pos = int(np.searchsorted(eff_close, anchor_ts, side="left"))
        return pos - 1   # may be -1 if anchor_ts ≤ eff_close[0]

    def _build_input_tensor(
        self, symbol: str, anchor_ts: np.datetime64,
    ) -> dict[str, np.ndarray] | None:
        """Slice the last `window_bars[scale]` features ending strictly
        before `anchor_ts` at every scale. Returns None if any scale lacks
        sufficient history.
        """
        out: dict[str, np.ndarray] = {}
        for scale in SCALES_ORDERED:
            sc = self._scale(symbol, scale)
            last = self._strict_last_idx(sc.eff_close, anchor_ts)
            need = self.window_bars[scale]
            if last < need - 1:   # not enough bars before anchor
                return None
            start = last - need + 1
            window = sc.features[start : last + 1]
            if window.shape[0] != need:
                return None
            out[scale] = window
        return out

    def _vol_bucket_value(
        self, symbol: str, anchor_ts: np.datetime64, horizon: str,
    ) -> float | None:
        """Realized vol of close-to-close log-returns over the horizon's
        lookback, strictly before `anchor_ts`. Returns None if insufficient
        history. Bucketing into quintiles happens at the cross-section
        level (relative to peers at the same `t`).
        """
        scale = VOL_LOOKBACK_SCALE[horizon]
        lookback = VOL_LOOKBACK_BARS[horizon]
        sc = self._scale(symbol, scale)
        last = self._strict_last_idx(sc.eff_close, anchor_ts)
        if last < lookback:   # need lookback + 1 bars for `lookback` returns
            return None
        # Close column = CHANNEL_ORDER_18 index 3.
        closes = sc.features[last - lookback : last + 1, 3]
        # Drop non-positive (shouldn't happen on real OHLCV but be safe).
        if np.any(closes <= 0):
            return None
        log_rets = np.diff(np.log(closes))   # length `lookback`
        if log_rets.size == 0:
            return None
        return float(np.std(log_rets, ddof=1) if log_rets.size > 1 else 0.0)

    def _label_at(
        self, symbol: str, anchor_ts: np.datetime64, horizon: str,
    ) -> float | None:
        """Look up the label at exactly `anchor_ts`. Returns None if the
        symbol has no label at this `t` or `valid_label=False`."""
        lc = self._labels(symbol, horizon)
        if lc.t.size == 0:
            return None
        # Exact match on t. Labels are emitted at every valid anchor; if
        # this t isn't in the labels file, the symbol's forward window
        # was invalid here.
        pos = int(np.searchsorted(lc.t, anchor_ts, side="left"))
        if pos >= lc.t.size or lc.t[pos] != anchor_ts:
            return None
        if not lc.valid_label[pos]:
            return None
        return float(lc.y_log_return[pos])

    # ----- Anchor enumeration -----

    def _anchor_timestamps(self, horizon: str) -> np.ndarray:
        """Union of all symbols' label-`t` values for this horizon,
        deduplicated and sorted. These are the candidate cross-section
        anchors; the per-anchor build will drop ones with too few valid
        symbols."""
        ts_set: set[int] = set()
        for sym in self.universe:
            lc = self._labels(sym, horizon)
            if lc.t.size == 0:
                continue
            # int64 view for hashing.
            ts_set.update(lc.t.astype("int64").tolist())
        if not ts_set:
            return np.zeros(0, dtype="datetime64[us]")
        arr = np.array(sorted(ts_set), dtype="int64").astype("datetime64[us]")
        if self.min_end_ts is not None:
            # Normalize to UTC before stripping tzinfo (callers may pass any tz).
            min_utc = self.min_end_ts.astimezone(UTC).replace(tzinfo=None)
            cutoff = np.datetime64(min_utc)
            arr = arr[arr >= cutoff]
        return arr

    # ----- Cross-section assembly -----

    def _build_cross_section(
        self, anchor_ts: np.datetime64, horizon: str,
    ) -> CrossSection | None:
        """Assemble one cross-section. Returns None if the fraction of
        valid symbols falls below `drop_invalid_threshold`."""
        valid_symbols: list[str] = []
        valid_inputs: list[dict[str, np.ndarray]] = []
        valid_labels: list[float] = []
        valid_vols: list[float] = []
        valid_aux: list[np.ndarray] = []
        valid_ob: list[np.ndarray | None] = []
        valid_trad_label: list[int] = []
        valid_trad_valid: list[bool] = []
        valid_pnl_long:  list[float] = []
        valid_pnl_short: list[float] = []
        use_aux = self.aux_root is not None
        use_ob = self.ob_root is not None
        use_tradeable = self.tradeable_root is not None

        for sym in self.universe:
            label = self._label_at(sym, anchor_ts, horizon)
            if label is None:
                continue
            inputs = self._build_input_tensor(sym, anchor_ts)
            if inputs is None:
                continue
            vol = self._vol_bucket_value(sym, anchor_ts, horizon)
            if vol is None:
                continue
            aux = self._aux_at(sym, anchor_ts) if use_aux else None
            if use_aux and aux is None:
                # Skip symbol if aux missing — preserves per-cs alignment.
                continue
            # OB: don't filter the symbol out on missing OB (most symbols
            # lack OB at any given anchor). Instead, record None and let
            # the fusion-side gate mask it off.
            ob = self._ob_at(sym, anchor_ts) if use_ob else None
            trad = self._tradeable_at(sym, anchor_ts) if use_tradeable else None
            valid_symbols.append(sym)
            valid_inputs.append(inputs)
            valid_labels.append(label)
            valid_vols.append(vol)
            if aux is not None:
                valid_aux.append(aux)
            if use_ob:
                valid_ob.append(ob)
            if use_tradeable:
                if trad is not None:
                    valid_trad_label.append(trad[0])
                    valid_trad_valid.append(trad[1])
                    valid_pnl_long.append(trad[2])
                    valid_pnl_short.append(trad[3])
                else:
                    valid_trad_label.append(0)
                    valid_trad_valid.append(False)
                    valid_pnl_long.append(0.0)
                    valid_pnl_short.append(0.0)

        # Floor: rank-IC stability needs ≥10 names, K=5 vol bucketing needs ≥5.
        # Use 10 as the hard minimum (matches `metrics.cross_sectional_rank_ic`'s
        # `min_cross_section_size` default).
        floor = max(MIN_SYMBOLS_FOR_CROSS_SECTION,
                    int(self.drop_invalid_threshold * len(self.universe)))
        if len(valid_symbols) < floor:
            return None

        n = len(valid_symbols)
        # Stack inputs per scale.
        inputs_t: dict[str, torch.Tensor] = {}
        for scale in SCALES_ORDERED:
            stacked = np.stack([row[scale] for row in valid_inputs], axis=0)
            inputs_t[scale] = torch.from_numpy(stacked).contiguous()

        labels_arr = np.asarray(valid_labels, dtype=np.float32)
        vols_arr = np.asarray(valid_vols, dtype=np.float32)

        # Cross-sectional rank ∈ [0, 1] of labels. Ties broken by stable
        # argsort order (ordinal ranks, not average). Real float32 returns
        # don't tie in practice, and ListMLE (the downstream consumer per
        # PHASE3.md §6) treats ranks as an ordering, not a magnitude.
        order = np.argsort(np.argsort(labels_arr))   # rank (0..n-1)
        ranks = order.astype(np.float32) / max(1, n - 1)

        # Vol bucket: K-quantile bin of vols within this cross-section.
        bucket = _quantile_bin(vols_arr, self.n_vol_buckets)

        aux_t: torch.Tensor | None = None
        if use_aux and valid_aux:
            aux_stack = np.stack(valid_aux, axis=0).astype(np.float32)
            aux_t = torch.from_numpy(aux_stack).contiguous()

        ob_t: torch.Tensor | None = None
        ob_mask_t: torch.Tensor | None = None
        if use_ob:
            from daity.data.ob_features import OB_N_CHANNELS
            K = self.ob_k_snapshots
            ob_arr = np.zeros((n, K, OB_N_CHANNELS), dtype=np.float32)
            ob_mask = np.zeros(n, dtype=bool)
            for i, win in enumerate(valid_ob):
                if win is not None:
                    ob_arr[i] = win
                    ob_mask[i] = True
            ob_t = torch.from_numpy(ob_arr).contiguous()
            ob_mask_t = torch.from_numpy(ob_mask)

        trad_label_t: torch.Tensor | None = None
        trad_valid_t: torch.Tensor | None = None
        pnl_long_t:   torch.Tensor | None = None
        pnl_short_t:  torch.Tensor | None = None
        if use_tradeable:
            trad_label_t = torch.tensor(valid_trad_label, dtype=torch.int8)
            trad_valid_t = torch.tensor(valid_trad_valid, dtype=torch.bool)
            pnl_long_t  = torch.tensor(valid_pnl_long,  dtype=torch.float32)
            pnl_short_t = torch.tensor(valid_pnl_short, dtype=torch.float32)

        return CrossSection(
            t=_np_to_utc(anchor_ts),
            horizon=horizon,
            symbols=valid_symbols,
            inputs=inputs_t,
            labels=torch.from_numpy(labels_arr),
            ranks=torch.from_numpy(ranks),
            valid_mask=torch.ones(n, dtype=torch.bool),
            vol_bucket=torch.from_numpy(bucket),
            aux_features=aux_t,
            ob_snapshot_window=ob_t,
            ob_mask=ob_mask_t,
            tradeable_label=trad_label_t,
            tradeable_valid=trad_valid_t,
            realized_pnl_long=pnl_long_t,
            realized_pnl_short=pnl_short_t,
        )

    # ----- Public iterators -----

    def iter_30m(self) -> Iterator[CrossSection]:
        yield from self._iter_horizon("30m")

    def iter_120m(self) -> Iterator[CrossSection]:
        yield from self._iter_horizon("120m")

    def iter_1d(self) -> Iterator[CrossSection]:
        yield from self._iter_horizon("1d")

    def iter_2d(self) -> Iterator[CrossSection]:
        yield from self._iter_horizon("2d")

    def _iter_horizon(self, horizon: str) -> Iterator[CrossSection]:
        anchors = self._anchor_timestamps(horizon)
        n_skipped = 0
        n_emitted = 0
        for anchor in anchors:
            cs = self._build_cross_section(anchor, horizon)
            if cs is None:
                n_skipped += 1
                continue
            n_emitted += 1
            yield cs
        log.info(
            "CrossSectionalSampler.iter_%s: emitted=%d skipped=%d (universe=%d)",
            horizon, n_emitted, n_skipped, len(self.universe),
        )


# ----- helpers -----


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin `values` into `n_bins` quantiles. Returns int8 in [0, n_bins-1].

    Empty / degenerate inputs (all values equal) return all zeros. We
    use `np.quantile` rather than `pd.qcut` to keep the dep surface
    minimal and avoid pandas-dependent NaN handling.
    """
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if np.unique(values).size == 1:
        # All equal → put everyone in the middle bucket so downstream
        # conformal calibration doesn't see an empty stratum.
        mid = n_bins // 2
        return np.full(n, mid, dtype=np.int8)
    # Quantile edges (n_bins - 1 internal edges).
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(values, qs)
    return np.searchsorted(edges, values, side="right").astype(np.int8)


def _np_to_utc(ts: np.datetime64) -> datetime:
    """Convert tz-naive numpy datetime64[us] back to a UTC `datetime`."""
    us = int(ts.astype("datetime64[us]").astype("int64"))
    sec, micro = divmod(us, 1_000_000)
    return datetime.fromtimestamp(sec, tz=UTC).replace(microsecond=micro)
