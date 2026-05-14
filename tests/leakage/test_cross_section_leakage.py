"""Leakage gate for `daity.data.cross_section` — `PHASE3.md` §4.6.

Two guarantees, spot-checked across emitted cross-sections:

1. **No input bar's effective_close ≥ anchor_t.** Every per-scale window
   in the cross-section was sliced STRICTLY before the anchor. This is
   the same `[t-window, t)` semantics the XGBoost baseline already
   enforces; the sampler shares the index lookup via
   `_strict_last_idx`. We verify by reading the underlying feature
   parquet, locating the bars that should be in the window, and
   asserting the LAST row of `cs.inputs[scale]` matches the bar BEFORE
   the anchor (whose effective_close < anchor_t).

2. **No label uses a forward bar past `as_of`.** Inherited from
   `daity.data.labels`'s leakage gate, but verified end-to-end here: the
   sampler must not silently bypass the label module's filter.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from daity.data.calendar_nse import NSECalendar
from daity.data.cross_section import CrossSectionalSampler
from daity.data.feature_precompute import CHANNEL_ORDER_18
from daity.data.labels import precompute_labels
from daity.data.parquet_store import ParquetStore

pytestmark = pytest.mark.leakage

IST = timezone(timedelta(hours=5, minutes=30))


# Reuse the fixture pattern from the unit test, simplified inline.


def _intraday_ts(days: list[date]) -> list[datetime]:
    out: list[datetime] = []
    for d in days:
        so = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST).astimezone(UTC)
        for k in range(75):
            out.append(so + timedelta(minutes=5 * k))
    return out


def _daily_ts(days: list[date]) -> list[datetime]:
    return [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]


def _bars_feat(symbol: str, scale: str, ts_list: list[datetime],
               closes: np.ndarray, rng: np.random.Generator) -> pl.DataFrame:
    n = len(ts_list)
    cols: dict[str, list] = {"symbol": [symbol]*n, "ts": ts_list, "scale": [scale]*n}
    schema = {"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "scale": pl.Utf8}
    for ch in CHANNEL_ORDER_18:
        if ch == "close":
            cols[ch] = closes.astype(np.float32).tolist()
        elif ch in ("open", "high", "low"):
            cols[ch] = (closes.astype(np.float32) + 0.05).tolist()
        elif ch == "volume":
            cols[ch] = list(rng.integers(900, 1100, size=n).astype(np.float32))
        else:
            cols[ch] = list(rng.normal(0, 1, size=n).astype(np.float32))
        schema[ch] = pl.Float32
    return pl.DataFrame(cols, schema=schema)


def _bars_raw(symbol: str, scale: str, ts_list: list[datetime],
              closes: np.ndarray, rng: np.random.Generator) -> pl.DataFrame:
    n = len(ts_list)
    return pl.DataFrame(
        {
            "symbol": [symbol]*n, "ts": ts_list, "scale": [scale]*n,
            "open":  (closes - 0.05).astype(np.float64),
            "high":  (closes + 0.10).astype(np.float64),
            "low":   (closes - 0.10).astype(np.float64),
            "close": closes.astype(np.float64),
            "volume": list(rng.integers(900, 1100, size=n).astype(np.int64)),
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )


def _build_fixture(tmp_path: Path, n_symbols: int = 12, n_days: int = 30):
    feature_root = tmp_path / "features"
    raw_root = tmp_path / "raw"
    label_root = tmp_path / "labels"
    feature_root.mkdir()
    raw_root.mkdir()
    label_root.mkdir()
    rng = np.random.default_rng(20260511)

    days: list[date] = []
    cur = date(2024, 6, 3)
    while len(days) < n_days:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    cal = NSECalendar(trading_days=days, holidays=[])

    ts_5m = _intraday_ts(days)
    ts_day = _daily_ts(days)
    feat_store = ParquetStore(feature_root)
    raw_store = ParquetStore(raw_root)

    universe = [f"S{s:02d}" for s in range(n_symbols)]
    for sym in universe:
        closes_5m = np.cumsum(rng.normal(0.0, 0.1, size=len(ts_5m))) + 100.0
        closes_5m = np.maximum(closes_5m, 1.0)
        closes_day = np.array(
            [closes_5m[(d_idx + 1) * 75 - 1] for d_idx in range(len(days))],
            dtype=np.float32,
        )
        feat_store.write_partition("5m", sym,
            _bars_feat(sym, "5m", ts_5m, closes_5m, rng))
        feat_store.write_partition("15m", sym,
            _bars_feat(sym, "15m", ts_5m[::3], closes_5m[::3], rng))
        feat_store.write_partition("60m", sym,
            _bars_feat(sym, "60m", ts_5m[::12], closes_5m[::12], rng))
        feat_store.write_partition("day", sym,
            _bars_feat(sym, "day", ts_day, closes_day, rng))
        raw_store.write_partition("5m", sym,
            _bars_raw(sym, "5m", ts_5m, closes_5m, rng))
        raw_store.write_partition("day", sym,
            _bars_raw(sym, "day", ts_day, closes_day, rng))

    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    for sym in universe:
        for h in ("30m", "120m", "1d", "2d"):
            precompute_labels(
                raw_store=raw_store, label_root=label_root,
                symbol=sym, horizon=h, calendar=cal, as_of=as_of,
            )
    return feature_root, label_root, universe


def test_no_input_bar_at_or_past_anchor(tmp_path: Path) -> None:
    """For every yielded cross-section, the LAST 5m bar in
    `inputs["5m"]` must have effective_close STRICTLY < anchor_t.
    """
    feature_root, label_root, universe = _build_fixture(tmp_path)
    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}
    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    n_checked = 0
    for cs in sampler.iter_30m():
        # Reconstruct per-symbol: the last close in the input matches
        # the close of the bar with eff_close = anchor_t - 5min.
        anchor_np = np.datetime64(cs.t.replace(tzinfo=None))
        for i, sym in enumerate(cs.symbols):
            df_5m = pl.read_parquet(feature_root / "5m" / f"{sym}.parquet").sort("ts")
            ts_arr = df_5m["ts"].to_numpy()
            eff_arr = ts_arr + np.timedelta64(5, "m")
            # The last valid row has eff_close STRICTLY less than anchor.
            mask = eff_arr < anchor_np
            assert mask.any(), f"sym={sym} anchor={cs.t}: no bar before anchor"
            last_idx = int(np.where(mask)[0][-1])
            truth = float(df_5m["close"][last_idx])
            # cs.inputs["5m"] shape (N, K, 18); close is channel 3.
            tensor_last_close = float(cs.inputs["5m"][i, -1, 3])
            assert abs(tensor_last_close - truth) < 1e-4, (
                f"sym={sym} anchor={cs.t}: tensor_last_close={tensor_last_close} "
                f"truth={truth}"
            )
        n_checked += 1
        if n_checked >= 50:   # 50 anchors × N symbols is plenty
            break
    assert n_checked > 0, "sampler emitted nothing — fixture too small?"


def test_no_label_past_as_of(tmp_path: Path) -> None:
    """If we cap `as_of` in the middle of the data range, no emitted
    cross-section has `t` past the cap. (Labels are pre-cut at build
    time; this verifies the sampler doesn't bypass the cap.)
    """
    feature_root, label_root, universe = _build_fixture(tmp_path)
    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}
    # Mid-range cap.
    as_of = datetime(2024, 6, 20, tzinfo=UTC)
    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=as_of,
    )
    for cs in sampler.iter_1d():
        # For 1d, t_horizon = t + 1 trading day. Both must be ≤ as_of.
        # Sampler only checks `t_horizon <= as_of`, which is the right
        # constraint — we verify directly.
        # Look up the (sym, t) row in the labels file; its t_horizon
        # must be ≤ as_of.
        sym = cs.symbols[0]
        df = pl.read_parquet(label_root / "1d" / f"{sym}.parquet")
        row = df.filter(pl.col("t") == cs.t)
        assert row.height == 1
        assert row["t_horizon"][0] <= as_of


def test_sampler_strict_less_than_at_session_boundary(tmp_path: Path) -> None:
    """A 30m anchor at exactly some 5m bar's effective_close must NOT
    include that bar in its input window — the LAST bar in the window
    is the one whose eff_close == anchor - 5min.
    """
    feature_root, label_root, universe = _build_fixture(tmp_path)
    small_windows = {"5m": 4, "15m": 3, "60m": 2, "day": 2}
    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root),
        label_root=label_root,
        universe=universe, window_bars=small_windows,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    sections = list(sampler.iter_30m())
    assert len(sections) > 0
    # Pick the first emitted cross-section; verify the strict-< property
    # via the (anchor_t, anchor_t - 5min) close comparison.
    cs = sections[5] if len(sections) > 5 else sections[0]
    anchor_np = np.datetime64(cs.t.replace(tzinfo=None))
    for i, sym in enumerate(cs.symbols[:3]):   # spot check 3 symbols
        df = pl.read_parquet(feature_root / "5m" / f"{sym}.parquet").sort("ts")
        ts_arr = df["ts"].to_numpy()
        eff_arr = ts_arr + np.timedelta64(5, "m")
        # The bar opening at anchor - 5min has eff_close == anchor.
        # That bar must NOT appear in the input window.
        leak_idx = np.where(eff_arr == anchor_np)[0]
        if leak_idx.size == 0:
            continue   # no exact match — already strict by chance
        leak_close = float(df["close"][int(leak_idx[0])])
        # Confirm the tensor's last close is NOT this value.
        tensor_last = float(cs.inputs["5m"][i, -1, 3])
        # If they ARE equal, it's a leakage bug.
        # (Numerically distinct synth data → near-zero probability of
        # coincident equality.)
        if abs(tensor_last - leak_close) < 1e-9:
            raise AssertionError(
                f"sym={sym}: input window's last 5m close ({tensor_last}) "
                f"equals the leaking bar's close ({leak_close}) — strict-< violated"
            )
