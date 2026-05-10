"""PyTorch datasets that yield per-symbol multi-resolution OHLCV windows.

Phase 2 pretraining batches are `(symbol, end_ts)` pairs: for each pair we
pull the last `window_bars[scale]` bars at every requested scale, ending
at `end_ts`. The model concatenates the per-scale patches via
`MultiResTokenizer`. See DESIGN §3.1 / §3.2.

This module ships two related classes:

  - `OHLCVWindowSampler` — pure-Python sampler. Given a `ParquetStore`,
    a symbol list, an `as_of` cap, and per-scale window lengths, yields
    `(symbol, end_ts, channels)` tensors lazily. Easy to test against
    fixtures and reuse outside PyTorch (e.g. for evaluation).

  - `OHLCVPretrainDataset` — `torch.utils.data.IterableDataset` wrapping
    the sampler so a `DataLoader` can shuffle / batch / pin-memory it.
    Implements **regime-aware sampling**: the trailing 18 months get 2×
    weight per DESIGN §3.2.

Leakage gate inheritance: `ParquetStore.read` already filters by
`as_of` (and the project-wide `effective_close` rule). The dataset reads
from there, so no row whose effective_close > as_of can end up in a
batch. Any window whose right edge is past `as_of` is silently skipped
rather than truncated, so the caller never sees a "partial" sample.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import IterableDataset

from daity.data.candles import INTERVAL_DURATIONS
from daity.data.parquet_store import ParquetStore
from daity.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Channels we feed the backbone. Order matters — used as the column index.
DEFAULT_CHANNELS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Default per-scale window lengths (bars). Sized so each scale covers a
# meaningful slice of history without blowing up the per-batch tensor:
#   5m × 256 bars  ≈ 3.4 trading days
#   15m × 192 bars ≈ 8 trading days
#   60m × 168 bars ≈ 4 trading weeks
#   day × 64 bars  ≈ 3 trading months
DEFAULT_WINDOW_BARS: dict[str, int] = {
    "5m":  256,
    "15m": 192,
    "60m": 168,
    "day": 64,
}

# DESIGN §3.2: trailing 18 months upweighted 2× during pretraining sampling.
RECENT_REGIME_DAYS: int = int(18 * 30.5)
RECENT_REGIME_WEIGHT: float = 2.0


@dataclass(frozen=True, slots=True)
class WindowSample:
    """One training sample: aligned per-scale tensors + metadata.

    Fields:
      symbol, end_ts: provenance.
      channels: map scale → (window_bars, n_channels) float32 tensor of
                INPUT bars ending at or before end_ts.
      forecast: optional (forecast_n_bars, n_channels) tensor of FUTURE
                bars at the forecast scale, taken from immediately after
                end_ts (still gated by as_of). None when forecasting is
                disabled. The target builder unfolds this into patches.
    """

    symbol: str
    end_ts: datetime
    # Map scale → (window_bars, n_channels) float32 tensor.
    channels: dict[str, torch.Tensor]
    forecast: torch.Tensor | None = None


@dataclass(slots=True)
class OHLCVWindowSampler:
    """Pull `(symbol, end_ts) → per-scale tensor` samples from a ParquetStore.

    `as_of` is the project-wide leakage cap. `end_ts` for each sample is
    drawn from the eligible interval `[earliest_supported_ts, as_of]`,
    where `earliest_supported_ts` is the first ts at which all scales
    have at least `window_bars[scale]` bars of history available.

    Iteration is unbounded: `__iter__` yields samples forever; callers
    decide how many to consume.
    """

    store: ParquetStore
    symbols: list[str]
    as_of: datetime
    window_bars: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_WINDOW_BARS),
    )
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    seed: int = 0
    recent_regime_weight: float = RECENT_REGIME_WEIGHT
    recent_regime_days: int = RECENT_REGIME_DAYS
    # Multi-horizon forecast (look-forward SSL objective on the FORECAST
    # token). When forecast_n_bars > 0 the sample also carries the next
    # `forecast_n_bars` bars at `forecast_scale`, immediately after end_ts.
    # All future bars must still satisfy effective_close <= as_of, otherwise
    # the sample is rejected (no partial windows).
    forecast_scale: str | None = None
    forecast_n_bars: int = 0
    # Lower bound on `end_ts` — defaults to None (no lower bound; use full
    # available history). Set this on val/test datasets to enforce a strict
    # temporal split: e.g., for val, pass `min_end_ts=val_start` so val
    # samples can only have `end_ts in [val_start, as_of]`. Without this the
    # val_ds samples uniformly from [earliest_data_ts, as_of] which heavily
    # overlaps the train range — measured val_loss is mostly train-distribution
    # loss, not held-out generalization. DESIGN §2.5 ("walk-forward expanding
    # window") expects strict-after-train val sampling.
    min_end_ts: datetime | None = None
    # Per-symbol-per-scale series cache so we don't re-read the partition for
    # every sample. The leakage cap is applied at first-fetch time. Declared
    # as a field with default_factory so `slots=True` is happy on @dataclass.
    _cache: dict[tuple[str, str], Series] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if not self.symbols:
            msg = "symbols must be non-empty"
            raise ValueError(msg)
        if not self.window_bars:
            msg = "window_bars must be non-empty"
            raise ValueError(msg)
        for sc, n in self.window_bars.items():
            if n <= 0:
                msg = f"window_bars[{sc!r}] must be positive, got {n}"
                raise ValueError(msg)
        if self.forecast_n_bars < 0:
            msg = f"forecast_n_bars must be >= 0, got {self.forecast_n_bars}"
            raise ValueError(msg)
        if self.forecast_n_bars > 0:
            if self.forecast_scale is None:
                msg = "forecast_scale must be set when forecast_n_bars > 0"
                raise ValueError(msg)
            if self.forecast_scale not in self.window_bars:
                msg = (
                    f"forecast_scale {self.forecast_scale!r} must also appear in "
                    f"window_bars (got window_bars keys: {list(self.window_bars)})"
                )
                raise ValueError(msg)

    def _series(self, symbol: str, scale: str) -> Series:
        """Cached per-(symbol, scale) ts column + channel matrix."""
        key = (symbol, scale)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        df = self.store.read(symbol=symbol, scale=scale, as_of=self.as_of)
        if df.height == 0:
            self._cache[key] = _empty_series(self.channels)
            return self._cache[key]
        # Apply effective_close gate explicitly (consistent with FeatureEngine).
        # ParquetStore.read uses ts <= as_of; for intraday this is too lax.
        cap = self.as_of
        ts = df["ts"]
        if scale in {"5m", "15m", "60m"}:
            duration = INTERVAL_DURATIONS[scale]
            mask = (ts + duration) <= cap
            df = df.filter(mask)
        # Strip tz before converting (pl→np datetime64 has no tz concept).
        ts_array = (
            df["ts"]
            .dt.replace_time_zone(None)
            .to_numpy()
            .astype("datetime64[us]")
        )
        chan_matrix = np.stack(
            [df[c].to_numpy() for c in self.channels], axis=-1,
        ).astype(np.float32)
        ser = Series(ts=ts_array, channels=chan_matrix)
        self._cache[key] = ser
        return ser

    def _earliest_end_ts(self, symbol: str) -> datetime | None:
        """The earliest end_ts at which every scale has window_bars of history."""
        candidates: list[datetime] = []
        for sc, n in self.window_bars.items():
            ser = self._series(symbol, sc)
            if len(ser.ts) < n:
                return None
            ts_at_n = ser.ts[n - 1]
            candidates.append(_npts_to_datetime(ts_at_n))
        if not candidates:
            return None
        return max(candidates)

    def _build_sample(self, symbol: str, end_ts: datetime) -> WindowSample | None:
        """Pull the last `window_bars[scale]` bars at or before `end_ts`.

        When `forecast_n_bars > 0`, also pulls the next `forecast_n_bars`
        bars at `forecast_scale` strictly *after* end_ts (still gated by
        as_of). Returns None if the future tail can't be satisfied — no
        partial forecast windows.
        """
        per_scale: dict[str, torch.Tensor] = {}
        # Remember the start index of the last input window at the forecast
        # scale, so the forecast slice picks up exactly where the input ends.
        forecast_input_idx_right: int | None = None
        for sc, n in self.window_bars.items():
            ser = self._series(symbol, sc)
            if len(ser.ts) == 0:
                return None
            # Effective close <= end_ts. For intraday, ec = ts + duration;
            # for daily, ec = ts. We've already gated by self.as_of; here
            # we further restrict to <= end_ts.
            cap_ts = end_ts - INTERVAL_DURATIONS[sc] if sc in {"5m", "15m", "60m"} else end_ts
            cap_npts = np.datetime64(cap_ts.astimezone(UTC).replace(tzinfo=None), "us")
            idx_right = int(np.searchsorted(ser.ts, cap_npts, side="right"))
            idx_left = idx_right - n
            if idx_left < 0:
                return None
            window = ser.channels[idx_left:idx_right]            # (n, C)
            per_scale[sc] = torch.from_numpy(np.ascontiguousarray(window))
            if sc == self.forecast_scale:
                forecast_input_idx_right = idx_right

        # Look-forward forecast tail at the configured scale.
        forecast: torch.Tensor | None = None
        if self.forecast_n_bars > 0:
            assert self.forecast_scale is not None  # __post_init__ enforces this
            assert forecast_input_idx_right is not None
            ser = self._series(symbol, self.forecast_scale)
            f_left = forecast_input_idx_right
            f_right = f_left + self.forecast_n_bars
            # ser is already gated by as_of (effective_close <= as_of). We
            # need at least forecast_n_bars bars after end_ts.
            if f_right > len(ser.ts):
                return None
            future = ser.channels[f_left:f_right]                 # (forecast_n_bars, C)
            forecast = torch.from_numpy(np.ascontiguousarray(future))

        return WindowSample(
            symbol=symbol, end_ts=end_ts,
            channels=per_scale, forecast=forecast,
        )

    def _latest_end_ts(self, symbol: str | None = None) -> datetime:
        """Upper bound for `end_ts` so the forecast tail still fits.

        When forecasting is disabled the latest end_ts is just `as_of`. With
        `forecast_n_bars > 0` the bound is the *minimum* of:
          - as_of - forecast_horizon (leakage cap), and
          - actual_last_ts_at_forecast_scale - forecast_horizon (data
            availability — protects against `as_of` running ahead of the
            ingested data, which would make the iterator spin forever).

        Without the second bound the sampler would systematically reject
        end_ts within one forecast-horizon of `as_of` (and `__iter__` would
        retry forever), biasing AWAY from the recent regime DESIGN says we
        want upweighted.

        `symbol` is optional only for the no-forecast fast path; when
        forecasting is enabled it's required to look up the per-symbol
        last-bar ts.
        """
        if self.forecast_n_bars == 0 or self.forecast_scale is None:
            return self.as_of
        # forecast_n_bars bars at forecast_scale ≈ horizon * one-bar duration
        # (daily uses an effective duration of 1 day for this purpose).
        if self.forecast_scale in INTERVAL_DURATIONS:
            bar_duration = INTERVAL_DURATIONS[self.forecast_scale]
        else:
            bar_duration = timedelta(days=1)
        horizon = bar_duration * self.forecast_n_bars
        leakage_bound = self.as_of - horizon
        if symbol is None:
            return leakage_bound
        # Data-availability bound: the forecast scale's last ingested ts
        # minus the horizon. We cache this per-symbol once, since the
        # series doesn't change during sampling.
        ser = self._series(symbol, self.forecast_scale)
        if len(ser.ts) == 0:
            return leakage_bound  # no data; let _draw_end_ts return None upstream
        last_ts = _npts_to_datetime(ser.ts[-1])
        data_bound = last_ts - horizon + bar_duration  # last bar IS available
        return min(leakage_bound, data_bound)

    def _draw_end_ts(self, rng: random.Random, symbol: str) -> datetime | None:
        """Pick a random end_ts in the symbol's eligible range, with the
        trailing `recent_regime_days` upweighted by `recent_regime_weight`.

        Upper bound is `_latest_end_ts(symbol)` (== as_of when no forecast
        tail; otherwise the min of leakage cap and data-availability cap).
        """
        earliest = self._earliest_end_ts(symbol)
        latest = self._latest_end_ts(symbol)
        if earliest is None or earliest >= latest:
            return None
        # Apply optional lower bound (val/test temporal split — DESIGN §2.5).
        if self.min_end_ts is not None and self.min_end_ts > earliest:
            earliest = self.min_end_ts
            if earliest >= latest:
                return None
        boundary = latest - timedelta(days=self.recent_regime_days)
        if boundary <= earliest:
            # Whole window is "recent" — uniform over the full range.
            return _uniform_between(rng, earliest, latest)

        # Pick the regime first via weighted Bernoulli.
        recent_window_days = max((latest - boundary).days, 1)
        prior_window_days = max((boundary - earliest).days, 1)
        # Weighted sampling: P(recent) = recent_weight*recent_days / (recent_weight*recent_days + prior_days).
        w_recent = self.recent_regime_weight * recent_window_days
        w_prior = prior_window_days
        if rng.random() < w_recent / (w_recent + w_prior):
            return _uniform_between(rng, boundary, latest)
        return _uniform_between(rng, earliest, boundary)

    def __iter__(self) -> Iterator[WindowSample]:
        rng = random.Random(self.seed)
        while True:
            sym = rng.choice(self.symbols)
            end_ts = self._draw_end_ts(rng, sym)
            if end_ts is None:
                continue
            sample = self._build_sample(sym, end_ts)
            if sample is None:
                continue
            yield sample


@dataclass(frozen=True, slots=True)
class Series:
    """Internal cached series: parallel ts + channel arrays."""

    ts: np.ndarray
    channels: np.ndarray


"""Reserved key for the future-bars tensor inside the per-sample dict.

The dataset yields the same dict its `MultiResTokenizer` expects (one
key per scale) plus this special key when forecast targets are enabled.
The PretrainModule pops it out before tokenizing the input.
"""
FORECAST_FUTURE_KEY: str = "__forecast_future__"


class OHLCVPretrainDataset(IterableDataset):
    """`IterableDataset` wrapping `OHLCVWindowSampler` for `DataLoader`.

    Each iteration yields a `dict` with one key per scale and a tensor of
    shape `(window_bars[scale], n_channels)`. When forecast targets are
    enabled (`forecast_n_bars > 0`) the dict also carries a key
    `FORECAST_FUTURE_KEY` with a `(forecast_n_bars, n_channels)` tensor of
    future bars at `forecast_scale`. The DataLoader's default `collate_fn`
    stacks across batch correctly.
    """

    def __init__(
        self,
        store: ParquetStore,
        symbols: list[str],
        as_of: datetime,
        *,
        window_bars: dict[str, int] | None = None,
        channels: tuple[str, ...] = DEFAULT_CHANNELS,
        seed: int = 0,
        forecast_scale: str | None = None,
        forecast_n_bars: int = 0,
        min_end_ts: datetime | None = None,
    ) -> None:
        super().__init__()
        self.sampler = OHLCVWindowSampler(
            store=store, symbols=symbols, as_of=as_of,
            window_bars=window_bars or dict(DEFAULT_WINDOW_BARS),
            channels=channels, seed=seed,
            forecast_scale=forecast_scale,
            forecast_n_bars=forecast_n_bars,
            min_end_ts=min_end_ts,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        # Per-worker seed split so multi-worker DataLoaders don't see
        # identical sample streams. Worker None (synchronous) uses base seed.
        sampler = (
            self.sampler if worker is None
            else _reseeded(self.sampler, base_seed=self.sampler.seed + worker.id)
        )
        for sample in sampler:
            # DataLoader collates dict-of-tensors, so we attach the future
            # tail (if any) under a reserved key. The training module pops
            # it before tokenizing.
            if sample.forecast is None:
                yield sample.channels
            else:
                out = dict(sample.channels)
                out[FORECAST_FUTURE_KEY] = sample.forecast
                yield out


# ----- Helpers -----


def _empty_series(channels: tuple[str, ...]) -> Series:
    return Series(
        ts=np.array([], dtype="datetime64[us]"),
        channels=np.zeros((0, len(channels)), dtype=np.float32),
    )


def _uniform_between(rng: random.Random, lo: datetime, hi: datetime) -> datetime:
    """Random datetime in `[lo, hi]` with second-level resolution."""
    if hi <= lo:
        return lo
    span_sec = int((hi - lo).total_seconds())
    return lo + timedelta(seconds=rng.randint(0, max(span_sec, 1)))


def _npts_to_datetime(t: np.datetime64) -> datetime:
    """Convert a numpy datetime64 (us) to a tz-aware UTC `datetime`."""
    epoch_us = int(t.astype("datetime64[us]").astype("int64"))
    return datetime.fromtimestamp(epoch_us / 1_000_000, tz=UTC)


def _reseeded(s: OHLCVWindowSampler, *, base_seed: int) -> OHLCVWindowSampler:
    """Clone a sampler with a different seed but otherwise-identical state."""
    return OHLCVWindowSampler(
        store=s.store, symbols=list(s.symbols), as_of=s.as_of,
        window_bars=dict(s.window_bars), channels=tuple(s.channels),
        seed=base_seed,
        recent_regime_weight=s.recent_regime_weight,
        recent_regime_days=s.recent_regime_days,
        forecast_scale=s.forecast_scale,
        forecast_n_bars=s.forecast_n_bars,
        min_end_ts=s.min_end_ts,
    )
