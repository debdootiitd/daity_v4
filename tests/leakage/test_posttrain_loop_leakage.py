"""Leakage gate for `daity.training.posttrain_loop` — review Critical-1.

The §3.5 review caught a real bug: `run_single_fold` was building the
train sampler with `as_of=cfg.as_of` and `min_end_ts=None`, so the train
stream could emit anchors from the VAL window. That would silently leak
the acceptance gate's IC numbers (§1.1 tri-criterion + §1.4 random-init
delta computed against val labels the model had already trained on).

This test verifies the fix: with `train_end < val_start < val_end`,
every cross-section yielded by the train sampler has `t ≤ train_end`,
and every cross-section yielded by the val sampler has `val_start ≤ t
≤ val_end`. The two streams are disjoint.
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
from daity.training.posttrain import SCALES, PostTrainConfig
from daity.training.posttrain_loop import RunConfig, _build_module

pytestmark = pytest.mark.leakage

IST = timezone(timedelta(hours=5, minutes=30))


def _build_universe(tmp_path: Path, n_symbols: int = 12, n_days: int = 60):
    """Mirror of `tests/unit/test_posttrain_loop.py::_build_synth_universe`."""
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

    ts_5m: list[datetime] = []
    for d in days:
        so = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST).astimezone(UTC)
        for k in range(75):
            ts_5m.append(so + timedelta(minutes=5 * k))
    ts_day = [datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC) for d in days]

    feat_store = ParquetStore(feature_root)
    raw_store = ParquetStore(raw_root)
    universe = [f"S{s:02d}" for s in range(n_symbols)]
    for sym in universe:
        closes_5m = np.cumsum(rng.normal(0.0, 0.05, size=len(ts_5m))) + 100.0
        closes_5m = np.maximum(closes_5m, 1.0)
        closes_day = np.array(
            [closes_5m[(d_idx + 1) * 75 - 1] for d_idx in range(len(days))],
            dtype=np.float32,
        )
        for scale, tslist, closes in (
            ("5m",  ts_5m,         closes_5m),
            ("15m", ts_5m[::3],    closes_5m[::3]),
            ("60m", ts_5m[::12],   closes_5m[::12]),
            ("day", ts_day,        closes_day),
        ):
            n = len(tslist)
            cols: dict[str, list] = {
                "symbol": [sym]*n, "ts": tslist, "scale": [scale]*n,
            }
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
            feat_store.write_partition(scale, sym, pl.DataFrame(cols, schema=schema))
        # Raw OHLCV for label generation.
        for scale, tslist, closes in (("5m", ts_5m, closes_5m), ("day", ts_day, closes_day)):
            n = len(tslist)
            raw_store.write_partition(scale, sym, pl.DataFrame(
                {
                    "symbol": [sym]*n, "ts": tslist, "scale": [scale]*n,
                    "open":  (closes - 0.05).astype(np.float64),
                    "high":  (closes + 0.10).astype(np.float64),
                    "low":   (closes - 0.10).astype(np.float64),
                    "close": closes.astype(np.float64),
                    "volume": list(rng.integers(900, 1100, size=n).astype(np.int64)),
                },
                schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                        "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                        "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
            ))

    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    for sym in universe:
        for h in ("30m", "120m", "1d", "2d"):
            precompute_labels(
                raw_store=raw_store, label_root=label_root,
                symbol=sym, horizon=h, calendar=cal, as_of=as_of,
            )
    return feature_root, label_root, universe, days


def test_train_sampler_never_emits_anchors_past_train_end(tmp_path: Path) -> None:
    """Critical-1 from §3.5 review: with `as_of=train_end`, the train
    sampler must NEVER emit a cross-section whose `t > train_end`.

    Synthesizes a universe spanning [day_0..day_59], picks `train_end =
    day_40 close`, `val_start = day_45 close`. Iterates the train
    sampler and asserts `cs.t <= train_end` for every emitted section.
    """
    feature_root, label_root, universe, days = _build_universe(tmp_path)

    train_start = datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC)
    train_end   = datetime(days[40].year, days[40].month, days[40].day, 23, 59, 59, tzinfo=UTC)
    val_start   = datetime(days[45].year, days[45].month, days[45].day, tzinfo=UTC)

    small_windows = {"5m": 32, "15m": 32, "60m": 32, "day": 16}
    train_sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root), label_root=label_root,
        universe=universe, as_of=train_end,
        window_bars=small_windows, min_end_ts=train_start,
        drop_invalid_threshold=0.0,
    )

    for h in ("30m", "120m", "1d", "2d"):
        n = 0
        for cs in getattr(train_sampler, f"iter_{h}")():
            assert cs.t <= train_end, (
                f"train sampler {h}: emitted cs.t={cs.t} > train_end={train_end}"
            )
            assert cs.t < val_start, (
                f"train sampler {h}: emitted cs.t={cs.t} ≥ val_start={val_start} — LEAKAGE"
            )
            n += 1
        # Each horizon should emit AT LEAST one cross-section in the
        # 40-day train window (otherwise the test isn't actually
        # exercising the train path).
        assert n > 0, f"train sampler {h}: emitted nothing — fixture too small?"


def test_val_sampler_only_emits_anchors_in_val_window(tmp_path: Path) -> None:
    """The complementary check: val sampler emits only `val_start ≤ t ≤ val_end`."""
    feature_root, label_root, universe, days = _build_universe(tmp_path)

    val_start = datetime(days[45].year, days[45].month, days[45].day, tzinfo=UTC)
    val_end   = datetime(days[-1].year, days[-1].month, days[-1].day, 23, 59, 59, tzinfo=UTC)

    small_windows = {"5m": 32, "15m": 32, "60m": 32, "day": 16}
    val_sampler = CrossSectionalSampler(
        feature_store=ParquetStore(feature_root), label_root=label_root,
        universe=universe, as_of=val_end,
        window_bars=small_windows, min_end_ts=val_start,
        drop_invalid_threshold=0.0,
    )

    for h in ("30m", "120m", "1d", "2d"):
        for cs in getattr(val_sampler, f"iter_{h}")():
            assert val_start <= cs.t <= val_end, (
                f"val sampler {h}: cs.t={cs.t} outside [{val_start}, {val_end}]"
            )


def test_random_init_resets_tokenizer_too(tmp_path: Path) -> None:
    """Review Major-2: in random_init mode, the tokenizer's weights must
    differ from the SSL-loaded weights — otherwise the §1.4 ablation
    leaves the (load-bearing) patch projection at SSL init.
    """
    feature_root, label_root, universe, days = _build_universe(tmp_path)
    from daity.models.backbone import Backbone
    from daity.models.tokenizer import MultiResTokenizer
    import torch

    # Build a synthetic SSL checkpoint with deterministic non-random weights.
    tokenizer = MultiResTokenizer(
        scales=SCALES, num_channels=18, d_model=64,
        patch_len=16, patch_stride=8, revin_affine=False,
    )
    backbone = Backbone(d_model=64, n_layers=2, n_heads=4, ffn_ratio=2)
    # Set tokenizer weights to a known pattern.
    with torch.no_grad():
        for p in tokenizer.parameters():
            p.fill_(0.42)
        for p in backbone.parameters():
            p.fill_(0.42)
    state = {f"tokenizer.{k}": v for k, v in tokenizer.state_dict().items()}
    state.update({f"backbone.{k}": v for k, v in backbone.state_dict().items()})
    ckpt_path = tmp_path / "ssl.ckpt"
    torch.save({
        "state_dict": state,
        "hyper_parameters": {
            "d_model": 64, "n_layers": 2, "n_heads": 4, "ffn_ratio": 2,
            "patch_len": 16, "patch_stride": 8, "num_channels": 18,
            "revin_affine": False, "scales": list(SCALES),
        },
    }, ckpt_path)

    module_cfg = PostTrainConfig(
        parent_checkpoint=ckpt_path,
        d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18,
        revin_affine=False, scales=SCALES,
        lora_r=4, head_hidden=32, head_dropout=0.0,
    )
    cfg = RunConfig(
        feature_root=feature_root, label_root=label_root, universe=universe,
        train_start=datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC),
        train_end=datetime(days[40].year, days[40].month, days[40].day, tzinfo=UTC),
        val_start=datetime(days[45].year, days[45].month, days[45].day, tzinfo=UTC),
        val_end=datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=UTC),
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        mode="random_init",
        module_cfg=module_cfg,
        max_steps=1,
        seed=0,
    )
    module = _build_module(cfg)
    # The tokenizer's parameters should NOT equal 0.42 anymore — they were
    # reset to fresh random init.
    for n, p in module.tokenizer.named_parameters():
        if "revin" in n:
            # RevIN has no learnable params when revin_affine=False; skip.
            continue
        if p.numel() == 0:
            continue
        # The 0.42-init was uniform; the random init should not coincidentally
        # also be uniform 0.42.
        assert not torch.allclose(p, torch.full_like(p, 0.42)), (
            f"tokenizer.{n}: random_init failed to reset (still at SSL init 0.42)"
        )
        assert p.requires_grad, f"tokenizer.{n}: random_init must unfreeze"
