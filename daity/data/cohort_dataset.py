"""Per-day cohort dataset for the cohort architecture.

One sample = one (date, anchor_time) pair. The sample carries:
  - Per-stock multi-scale windows for ALL alive stocks in the universe
  - Per-stock 10-horizon log-return targets
  - Per-cohort regime features (universe 5d return, breadth, cross-sectional vol)
  - Anchor-level metadata (anchor_ts, stock_ids, sector_ids)
  - Per-(cohort, horizon, stock) validity masks

We DO NOT eagerly stack into one giant tensor across cohorts at construction —
each `__iter__` step pulls the data lazily. This keeps memory low and lets
the walk-forward trainer adjust the (date, anchor) iteration order.

Key correctness invariants:
  - All data is `as_of`-gated. The dataset's `as_of` is the latest anchor's
    UTC timestamp; ParquetStore enforces no row past `as_of` leaks.
  - Targets are only included when both endpoints satisfy `effective_close <=
    target_endpoint_ts`; otherwise the (cohort, horizon) is masked out.
  - Per-anchor head-validity is determined by `head_validity_mask(anchor)`
    (calendar mask). The dataset additionally enforces "label-exists" by
    checking each target bar is present.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset

from daity.data.candles import INTERVAL_DURATIONS
from daity.data.calendar_nse import NSECalendar
from daity.data.datasets import DEFAULT_WINDOW_BARS
from daity.data.intraday_horizons import (
    HORIZONS,
    N_HORIZONS,
    NSE_OPEN_IST_MINUTES,
    head_validity_mask,
)
from daity.data.parquet_store import ParquetStore

# Channels the SSL champion used. Cohort reuses the same set.
DEFAULT_COHORT_CHANNELS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "ret_1", "ret_5", "ret_15", "ret_60",
    "bar_range", "body",
    "upper_wick_norm", "lower_wick_norm", "vol_z_20",
    "min_of_day_sin", "min_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
)
N_DEFAULT_COHORT_CHANNELS: int = len(DEFAULT_COHORT_CHANNELS)
IDX_CLOSE: int = DEFAULT_COHORT_CHANNELS.index("close")

# Regime features: keep small for MVP. Order is load-bearing — the model's
# regime-type embedding is per-index.
REGIME_FEAT_NAMES: tuple[str, ...] = (
    "univ_5d_return",   # equal-weight 5-day return across alive universe
    "univ_breadth_1d",  # % of stocks up over last trading day
    "univ_xs_vol",      # cross-sectional std of last day's returns
    "univ_1d_return",   # signed equal-weight 1-day return (market avg)
    "sector_disp_1d",   # cross-sectional std of per-sector mean 1d returns
                         # (high = sectors diverging; low = market is one-way)
    "cohort_mean_ret_30m",         # cohort-mean log-return over last 30m
    "cohort_mean_ret_1h",          # cohort-mean log-return over last 60m
    "cohort_mean_ret_today_so_far",# cohort-mean log-return from today's first
                                    # 5m close (9:20 IST) to anchor. 0 at 9:15.
    "anchor_min_of_day_sin",       # sin(2π × minutes_since_open / 375). Lets
    "anchor_min_of_day_cos",       # the bias head distinguish "all-zero = early
                                    # day" from "all-zero = flat market".
)
N_REGIME_FEATS: int = len(REGIME_FEAT_NAMES)


def _ist_anchor_to_utc(d: date, anchor_ist: dtime) -> datetime:
    """Convert IST date+time-of-day to UTC datetime."""
    ist_dt = datetime.combine(d, anchor_ist)
    return (ist_dt - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def _target_endpoint_ts(anchor_utc: datetime, h_idx: int, cal: NSECalendar) -> datetime | None:
    """Compute UTC timestamp of the bar whose close == horizon target endpoint.

    Returns None if the endpoint lies outside the trading calendar (e.g.
    +5 days when the calendar runs out).

    For intraday horizons the endpoint is exactly `anchor + N minutes`.
    For overnight / multi-day horizons the endpoint is a future trading day
    at a specific minute-of-day in IST.
    """
    h = HORIZONS[h_idx]
    if h.kind == "intraday":
        return anchor_utc + timedelta(minutes=h.minutes_intraday)
    if h.kind == "to_close":
        # 15:30 IST = 10:00 UTC on the anchor's trading day.
        ist = anchor_utc + timedelta(hours=5, minutes=30)
        close_ist = ist.replace(hour=15, minute=30, second=0, microsecond=0)
        return (close_ist - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)
    # overnight / next_day / multi_day
    ist = anchor_utc + timedelta(hours=5, minutes=30)
    target_day = cal.shift(ist.date(), h.trading_days_offset)
    if target_day is None:
        return None
    minutes_into = h.target_minutes_into_day
    # IST minute-of-day = open + minutes_into.
    ist_target = datetime.combine(target_day, dtime(0, 0)) + timedelta(
        minutes=NSE_OPEN_IST_MINUTES + minutes_into,
    )
    return (ist_target - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class CohortSample:
    """One cohort assembled at (date, anchor_time_utc).

    Fields:
      anchor_ts:    UTC datetime of the anchor.
      symbols:      list of N alive symbols (post-filtering for data).
      stock_ids:    (N,) long — index into the full universe table.
      sector_ids:   (N,) long — index into sectors table.
      x_by_scale:   dict scale → (N, L, C) float32 tensor.
      labels:       (N, H) float32 — log returns for each (stock, horizon).
                    Invalid entries are NaN; the validity mask zeros them.
      validity:     (H,) bool — calendar validity per horizon (anchor-driven).
                    Per-stock label existence is folded in (NaN labels → mask 0
                    only for that stock; otherwise mask is shared across cohort).
      label_validity_per_stock: (N, H) bool — fine-grained validity that
                    additionally checks the target bar exists for that stock.
      regime_feats: (n_regime_feats,) float32 — market-wide aggregates.
    """

    anchor_ts: datetime
    symbols: tuple[str, ...]
    stock_ids: torch.Tensor
    sector_ids: torch.Tensor
    x_by_scale: dict[str, torch.Tensor]
    labels: torch.Tensor
    validity: torch.Tensor
    label_validity_per_stock: torch.Tensor
    regime_feats: torch.Tensor


class CohortAssembler:
    """Build cohort samples from a ParquetStore + universe + calendar.

    Construction is cheap; per-sample assembly opens and slices parquet
    files. Per-symbol channel arrays are cached internally to amortize
    repeat anchors on the same trading day.
    """

    def __init__(
        self,
        *,
        store: ParquetStore,
        calendar: NSECalendar,
        universe: tuple[str, ...],
        sector_id_by_symbol: dict[str, int],
        as_of: datetime,
        window_bars: dict[str, int] | None = None,
        channels: tuple[str, ...] = DEFAULT_COHORT_CHANNELS,
    ) -> None:
        if not universe:
            msg = "universe must be non-empty"
            raise ValueError(msg)
        self.store = store
        self.calendar = calendar
        self.universe = tuple(universe)
        self.stock_id_by_symbol: dict[str, int] = {
            s: i for i, s in enumerate(self.universe)
        }
        self.sector_id_by_symbol = sector_id_by_symbol
        self.as_of = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)
        self.window_bars: dict[str, int] = window_bars or dict(DEFAULT_WINDOW_BARS)
        self.channels = tuple(channels)
        self.idx_close = self.channels.index("close")
        # Per-(symbol, scale) cached numpy view of (ts, channels). Holds the
        # FULL series up to as_of so window assembly is just an index slice.
        self._series_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    # ------------- Internal helpers -------------

    def _load_series(self, symbol: str, scale: str) -> tuple[np.ndarray, np.ndarray]:
        """Return cached `(ts_us[ndim=1], channels[ndim=2])` for (symbol, scale)."""
        key = (symbol, scale)
        cached = self._series_cache.get(key)
        if cached is not None:
            return cached
        df = self.store.read(symbol=symbol, scale=scale, as_of=self.as_of)
        if df.height == 0:
            ts_arr = np.empty(0, dtype="datetime64[us]")
            ch_arr = np.empty((0, len(self.channels)), dtype=np.float32)
            self._series_cache[key] = (ts_arr, ch_arr)
            return self._series_cache[key]
        # effective_close gate for intraday scales (consistent with sampler).
        if scale in {"5m", "15m", "60m"}:
            duration = INTERVAL_DURATIONS[scale]
            df = df.filter((pl.col("ts") + duration) <= self.as_of)
        ts_arr = (
            df["ts"]
            .dt.replace_time_zone(None)
            .to_numpy()
            .astype("datetime64[us]")
        )
        # Channels must exist; missing column = caller error.
        missing = [c for c in self.channels if c not in df.columns]
        if missing:
            msg = (
                f"Symbol {symbol!r} scale {scale!r} missing required channels: "
                f"{missing}. Available: {list(df.columns)[:30]}"
            )
            raise ValueError(msg)
        ch_arr = np.stack(
            [df[c].to_numpy() for c in self.channels], axis=-1,
        ).astype(np.float32)
        self._series_cache[key] = (ts_arr, ch_arr)
        return self._series_cache[key]

    def _slice_window(
        self,
        symbol: str,
        scale: str,
        end_ts_utc: datetime,
        n_bars: int,
    ) -> np.ndarray | None:
        """Return the last `n_bars` bars at `scale` whose effective_close <= end_ts.

        Returns None if the window cannot be satisfied.
        """
        ts_arr, ch_arr = self._load_series(symbol, scale)
        if ts_arr.size == 0:
            return None
        # effective_close <= end_ts: for intraday subtract one bar duration
        # to find the right index (since ts is open-time).
        cap_dt = end_ts_utc
        if scale in {"5m", "15m", "60m"}:
            cap_dt = cap_dt - INTERVAL_DURATIONS[scale]
        cap_npts = np.datetime64(cap_dt.astimezone(UTC).replace(tzinfo=None), "us")
        idx_right = int(np.searchsorted(ts_arr, cap_npts, side="right"))
        idx_left = idx_right - n_bars
        if idx_left < 0:
            return None
        return ch_arr[idx_left:idx_right]

    def _close_at(self, symbol: str, scale: str, ts_utc: datetime) -> float | None:
        """Return the close of the bar whose effective_close == ts_utc.

        - For intraday scales: bar_open == ts_utc - duration.
        - For daily: bar_ts == ts_utc (daily ts already encodes the close).

        Returns None if no such bar exists in the cache (gap / holiday / future).
        """
        ts_arr, ch_arr = self._load_series(symbol, scale)
        if ts_arr.size == 0:
            return None
        if scale in {"5m", "15m", "60m"}:
            target_open = ts_utc - INTERVAL_DURATIONS[scale]
        else:
            target_open = ts_utc
        target_npts = np.datetime64(
            target_open.astimezone(UTC).replace(tzinfo=None), "us",
        )
        idx = int(np.searchsorted(ts_arr, target_npts, side="left"))
        if idx >= len(ts_arr):
            return None
        if ts_arr[idx] != target_npts:
            return None
        return float(ch_arr[idx, self.idx_close])

    # ------------- Public API -------------

    def assemble(self, anchor_utc: datetime) -> CohortSample | None:
        """Assemble one cohort at the anchor. Returns None if too few alive
        stocks (< 10) at this anchor.

        Filters: stocks must have:
          1. window_bars[scale] bars of history at every scale, with the
             last bar's effective_close <= anchor.
          2. A valid close at the anchor's 5m endpoint (== anchor close).
        """
        alive_syms: list[str] = []
        x_by_scale: dict[str, list[np.ndarray]] = {sc: [] for sc in self.window_bars}
        anchor_close_by_sym: dict[str, float] = {}
        for sym in self.universe:
            wins: dict[str, np.ndarray] = {}
            ok = True
            for sc, n_bars in self.window_bars.items():
                w = self._slice_window(sym, sc, anchor_utc, n_bars)
                if w is None or w.shape[0] != n_bars:
                    ok = False
                    break
                wins[sc] = w
            if not ok:
                continue
            # Anchor close: the last 5m bar's close.
            anchor_close = self._close_at(sym, "5m", anchor_utc)
            if anchor_close is None or anchor_close <= 0:
                continue
            alive_syms.append(sym)
            anchor_close_by_sym[sym] = anchor_close
            for sc in self.window_bars:
                x_by_scale[sc].append(wins[sc])
        if len(alive_syms) < 10:
            return None

        # Tensor-ify per-scale windows: (N, L, C).
        x_tensors = {
            sc: torch.from_numpy(np.stack(x_by_scale[sc], axis=0))
            for sc in self.window_bars
        }
        N = len(alive_syms)

        # Stock & sector IDs.
        stock_ids = torch.tensor(
            [self.stock_id_by_symbol[s] for s in alive_syms],
            dtype=torch.long,
        )
        sector_ids = torch.tensor(
            [self.sector_id_by_symbol[s] for s in alive_syms],
            dtype=torch.long,
        )

        # Per-horizon validity mask (calendar-only, before per-stock checks).
        cal_mask = head_validity_mask(anchor_utc)                  # list[bool] len 10
        validity = torch.tensor(cal_mask, dtype=torch.bool)        # (H,)

        # Labels: for each horizon, log(close_endpoint / close_anchor).
        # Per-stock label_validity = calendar validity AND endpoint bar exists.
        labels = np.full((N, N_HORIZONS), np.nan, dtype=np.float32)
        label_valid = np.zeros((N, N_HORIZONS), dtype=bool)
        for h_idx in range(N_HORIZONS):
            if not cal_mask[h_idx]:
                continue
            endpoint_ts = _target_endpoint_ts(anchor_utc, h_idx, self.calendar)
            if endpoint_ts is None:
                # Calendar can't produce the target trading day (end of data).
                # Mark the WHOLE horizon as invalid in the validity vec.
                validity[h_idx] = False
                continue
            # Use 5m close for intraday + to_close + overnight (high-frequency
            # endpoints); for multi-day endpoints (day_plus_3 / day_plus_5),
            # we'd ideally use the 'day' bar's close, but to keep code uniform
            # we just use 5m close at the target IST minute since intraday
            # bars are available on every trading day.
            label_scale = "5m"
            for i, sym in enumerate(alive_syms):
                close_end = self._close_at(sym, label_scale, endpoint_ts)
                if close_end is None or close_end <= 0:
                    continue
                anchor_close = anchor_close_by_sym[sym]
                labels[i, h_idx] = float(np.log(close_end / anchor_close))
                label_valid[i, h_idx] = True

        labels_t = torch.from_numpy(labels)
        label_valid_t = torch.from_numpy(label_valid)

        # Regime features: equal-weight 5d return, breadth, cross-sectional vol.
        regime_feats = self._compute_regime_feats(alive_syms, anchor_utc, anchor_close_by_sym)

        return CohortSample(
            anchor_ts=anchor_utc,
            symbols=tuple(alive_syms),
            stock_ids=stock_ids,
            sector_ids=sector_ids,
            x_by_scale=x_tensors,
            labels=labels_t,
            validity=validity,
            label_validity_per_stock=label_valid_t,
            regime_feats=regime_feats,
        )

    def _compute_regime_feats(
        self,
        alive_syms: list[str],
        anchor_utc: datetime,
        anchor_close_by_sym: dict[str, float],
    ) -> torch.Tensor:
        """Compute (n_regime_feats,) regime tensor: 5d_return, breadth, xs_vol.

        Uses 'day' scale closes (more stable than intraday for these
        aggregates). Skips stocks missing the 5d-prior close (rare).
        """
        # Find the date 5 trading days before the anchor's trading day.
        anchor_date = (anchor_utc + timedelta(hours=5, minutes=30)).date()
        prev_5td = self.calendar.shift(anchor_date, -5)
        prev_1td = self.calendar.shift(anchor_date, -1)

        rets_5d: list[float] = []
        rets_1d: list[float] = []
        if prev_5td is not None:
            # 15:30 IST close of prev_5td = 10:00 UTC.
            ts5 = _ist_anchor_to_utc(prev_5td, dtime(15, 30))
            for s in alive_syms:
                c0 = self._close_at(s, "5m", ts5)
                if c0 is None or c0 <= 0:
                    continue
                c1 = anchor_close_by_sym[s]
                rets_5d.append(float(np.log(c1 / c0)))
        if prev_1td is not None:
            ts1 = _ist_anchor_to_utc(prev_1td, dtime(15, 30))
            for s in alive_syms:
                c0 = self._close_at(s, "5m", ts1)
                if c0 is None or c0 <= 0:
                    continue
                c1 = anchor_close_by_sym[s]
                rets_1d.append(float(np.log(c1 / c0)))

        univ_5d_ret = float(np.mean(rets_5d)) if rets_5d else 0.0
        univ_breadth = (
            float(np.mean([1.0 if r > 0 else 0.0 for r in rets_1d]))
            if rets_1d else 0.5
        )
        univ_xs_vol = float(np.std(rets_1d)) if len(rets_1d) > 1 else 0.0
        univ_1d_ret = float(np.mean(rets_1d)) if rets_1d else 0.0
        # Per-sector mean 1d return (only when we have rets_1d AND mapping).
        sector_disp_1d = 0.0
        if rets_1d:
            sector_rets: dict[int, list[float]] = {}
            for sym, r in zip(alive_syms, rets_1d, strict=False):
                sec = self.sector_id_by_symbol.get(sym)
                if sec is None:
                    continue
                sector_rets.setdefault(sec, []).append(r)
            per_sector_means = [
                float(np.mean(v)) for v in sector_rets.values() if v
            ]
            if len(per_sector_means) > 1:
                sector_disp_1d = float(np.std(per_sector_means))

        # Trailing cohort-mean intraday returns. Causally valid: all closes
        # are at ts <= anchor_utc. Returns 0.0 when the lookback close is
        # missing for a stock (skipped); 0.0 overall if no stock has both.
        cohort_30m = self._cohort_trailing_return(
            alive_syms, anchor_utc, anchor_close_by_sym, lookback_min=30,
        )
        cohort_1h = self._cohort_trailing_return(
            alive_syms, anchor_utc, anchor_close_by_sym, lookback_min=60,
        )
        cohort_today = self._cohort_today_so_far_return(
            alive_syms, anchor_utc, anchor_close_by_sym, anchor_date,
        )
        # Minute-of-day encoding (9:15 IST = 0, 15:30 IST = 375). Lets the
        # bias head distinguish "early-day zeros" from "flat-market zeros".
        anchor_ist_time = (anchor_utc + timedelta(hours=5, minutes=30)).time()
        min_since_open = (anchor_ist_time.hour * 60 + anchor_ist_time.minute) - (9 * 60 + 15)
        anchor_min_sin = float(np.sin(2 * np.pi * min_since_open / 375))
        anchor_min_cos = float(np.cos(2 * np.pi * min_since_open / 375))
        return torch.tensor(
            [univ_5d_ret, univ_breadth, univ_xs_vol,
             univ_1d_ret, sector_disp_1d,
             cohort_30m, cohort_1h, cohort_today,
             anchor_min_sin, anchor_min_cos],
            dtype=torch.float32,
        )

    def _cohort_trailing_return(
        self,
        alive_syms: list[str],
        anchor_utc: datetime,
        anchor_close_by_sym: dict[str, float],
        lookback_min: int,
    ) -> float:
        """Cohort-mean log-return from (anchor - lookback_min) to anchor."""
        ts_lookback = anchor_utc - timedelta(minutes=lookback_min)
        rets: list[float] = []
        for s in alive_syms:
            c0 = self._close_at(s, "5m", ts_lookback)
            if c0 is None or c0 <= 0:
                continue
            c1 = anchor_close_by_sym.get(s)
            if c1 is None:
                continue
            rets.append(float(np.log(c1 / c0)))
        return float(np.mean(rets)) if rets else 0.0

    def _cohort_today_so_far_return(
        self,
        alive_syms: list[str],
        anchor_utc: datetime,
        anchor_close_by_sym: dict[str, float],
        anchor_date: date,
    ) -> float:
        """Cohort-mean log-return from today's first 5m close (9:20 IST)
        to the anchor. Returns 0.0 at the 9:15 anchor (no prior intraday bar).
        """
        anchor_ist_time = (anchor_utc + timedelta(hours=5, minutes=30)).time()
        first_5m_close_ist = dtime(9, 20)
        if anchor_ist_time <= first_5m_close_ist:
            return 0.0
        ts_open = _ist_anchor_to_utc(anchor_date, first_5m_close_ist)
        rets: list[float] = []
        for s in alive_syms:
            c0 = self._close_at(s, "5m", ts_open)
            if c0 is None or c0 <= 0:
                continue
            c1 = anchor_close_by_sym.get(s)
            if c1 is None:
                continue
            rets.append(float(np.log(c1 / c0)))
        return float(np.mean(rets)) if rets else 0.0


class CohortDataset(IterableDataset):
    """`IterableDataset` yielding one CohortSample per iteration.

    The driver supplies an iterable of (date, anchor_time_ist) pairs; the
    dataset shuffles them per-epoch (seedable) and yields assembled samples
    in dict form ready to feed `CohortModel`.

    Empty/insufficient cohorts are silently skipped.
    """

    def __init__(
        self,
        assembler: CohortAssembler,
        anchor_pairs: list[tuple[date, dtime]],
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.assembler = assembler
        self.anchor_pairs = list(anchor_pairs)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        order = list(range(len(self.anchor_pairs)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)
        for i in order:
            d, anchor_ist = self.anchor_pairs[i]
            anchor_utc = _ist_anchor_to_utc(d, anchor_ist)
            sample = self.assembler.assemble(anchor_utc)
            if sample is None:
                continue
            yield {
                "x_by_scale":   {sc: v.unsqueeze(0) for sc, v in sample.x_by_scale.items()},
                "stock_ids":    sample.stock_ids.unsqueeze(0),
                "sector_ids":   sample.sector_ids.unsqueeze(0),
                "anchor_ts":    torch.tensor(
                    [int(anchor_utc.timestamp() * 1_000_000)], dtype=torch.int64,
                ),
                "regime_feats": sample.regime_feats.unsqueeze(0),
                "labels":       sample.labels.unsqueeze(0),
                "validity":     sample.validity.unsqueeze(0),
                "label_validity_per_stock": sample.label_validity_per_stock.unsqueeze(0),
                "symbols":      sample.symbols,
            }
