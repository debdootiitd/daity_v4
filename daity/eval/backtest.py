"""Phase 5 v1: Backtest engine for tradeable-direction models.

Simple per-trade independence model (no position carry, no compounding):
  - At each 30-min timestamp, compute model's P(LONG), P(SHORT) per symbol.
  - Filter to top-k by score, apply confidence threshold.
  - Open trade at bar close, exit at +30 min (matches label horizon).
  - PnL = direction × forward_return − round_trip_cost.
  - Aggregate to tearsheet (cumulative PnL, win rate, Sharpe, max DD).

Friction model:
  Indian intraday equity ~15 bps round-trip (5-10 bps slippage + STT 2.5 bps
  on sell + brokerage 1-3 bps × 2 + GST/stamp 0.5-1 bps). Conservative for
  retail; institutional could be lower.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daity.data.cross_section import CrossSection, CrossSectionalSampler
from daity.data.parquet_store import ParquetStore
from daity.training.tradeable_finetune import (
    TradeableFineTuneConfig, TradeableFineTuneModule,
    TradeableOBFineTuneConfig, TradeableOBFineTuneModule,
)
from daity.utils.logging import get_logger

UTC = timezone.utc
log = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Knobs for the backtest run."""

    # Universe + test window
    universe: list[str] = field(default_factory=list)
    test_start: datetime = datetime(2026, 4, 20, tzinfo=UTC)
    test_end:   datetime = datetime(2026, 5, 4, tzinfo=UTC)
    as_of:      datetime = datetime(2026, 5, 4, tzinfo=UTC)
    feature_root: Path = Path("data/features_parquet")
    label_root:   Path = Path("data/labels")
    tradeable_root: Path = Path("data/tradeable_labels/v1_trig5_floor3_dmin10_dmax30_k60_5m_c9cd47f8")
    ob_root: Path | None = None
    # Optional secondary checkpoint for ensemble scoring. When set, both models
    # run on every cross-section and their LONG scores are converted to within-
    # cross-section percentile ranks then averaged (with ensemble_secondary_weight
    # on the secondary, default 0.5). The averaged rank replaces combined_long
    # for entry/exit. Improves conviction calibration when the two checkpoints
    # disagree on weak signals but agree on strong ones.
    secondary_checkpoint: Path | None = None
    ensemble_secondary_weight: float = 0.5

    # Trade rules
    long_top_k:  int = 5
    short_top_k: int = 5
    # Score thresholds. 0 = no filter (always take top-k).
    long_score_threshold:  float = 0.0
    short_score_threshold: float = 0.0
    # Percentile-based entry gate. When > 0, only consider the top X fraction
    # of cross-sectional scores (e.g., 0.05 = top 5%). Robust to score-
    # distribution shifts across model checkpoints, unlike absolute threshold.
    long_score_pct_gate:  float = 0.0
    short_score_pct_gate: float = 0.0
    # Skip cross-sections with fewer than this many valid symbols.
    min_valid_per_cs: int = 50
    # Side selection:
    #   "long_only"  — only LONG trades, SHORT scores can still filter
    #   "short_only" — only SHORT trades
    #   "both"       — symmetric LONG + SHORT
    side: str = "both"
    # Long-only score formula:
    #   combined_score = P(LONG) − long_short_alpha * P(SHORT)
    # α=0 → pure P(LONG); α=1 → spread P(LONG)−P(SHORT). Default α=0.
    long_short_alpha: float = 0.0
    # Hard filter: skip a LONG candidate when its P(SHORT) > this value.
    # 1.0 = disabled (default).
    long_short_avoidance_max: float = 1.0

    # Position sizing
    initial_capital: float = 100_000.0   # ₹1 lakh
    position_size_pct: float = 0.02      # 2% per trade (flat default)
    # Conviction-weighted sizing. When "flat" (default), every trade uses
    # `position_size_pct`. When "score_pct_rank", linearly interpolates per-trade
    # notional from `position_size_min_pct` (worst-ranked entry) to
    # `position_size_max_pct` (top of cross-section) by score percentile.
    position_size_mode: str = "flat"
    position_size_min_pct: float = 0.02
    position_size_max_pct: float = 0.20

    # Frictions — modelled as the sum of three components per round trip:
    #   1. Fixed brokerage in INR per LEG (buy or sell). Indian retail discount
    #      brokers charge a flat fee, typically ₹20 per executed order.
    #   2. Statutory costs as a fraction of notional, round-trip total.
    #      (STT 2.5 bps sell + stamp 1.5 bps buy + exchange/GST/SEBI ~1 bps
    #      → ~5 bps round-trip on Indian intraday equity).
    #   3. Slippage as a fraction, round-trip. Adds market-impact + bid-ask.
    # Total round-trip cost in INR = 2 * fixed_brokerage + (statutory + slippage) * notional
    # Total round-trip cost as fraction = above / notional
    fixed_brokerage_per_leg_inr: float = 20.0
    statutory_cost_pct: float = 0.0005     # 5 bps round-trip
    slippage_pct: float = 0.0010           # 10 bps round-trip
    # Legacy fixed round-trip override (if > 0, supersedes the components above).
    round_trip_cost: float = 0.0

    # Hold period (used for reporting only; PnL comes from `cs.labels`,
    # which is the realized forward return at the sampler's horizon).
    hold_horizon_minutes: int = 30

    # ---- Triple-barrier (first-touch) exit ----
    # When use_first_touch=True, simulate trade exit at first barrier touch
    # using 5m bars in [t, t+h_max_minutes]. PnL reflects the realized barrier
    # return, NOT cs.labels. This matches the train-time triple-barrier label.
    use_first_touch: bool = False
    # Per-side barriers as fractional returns (positive). LONG profit target =
    # target_long, LONG stop = stop_long (downside). SHORT side mirrors.
    target_long_pct: float = 0.0075
    stop_long_pct:   float = 0.004
    target_short_pct: float = 0.0075
    stop_short_pct:   float = 0.004
    h_max_minutes:   int = 60
    # Ambiguous-bar rule when both target and stop hit in same 5m bar.
    # "stop_first" | "target_first" | "midbar" | "proportional"
    ambiguous_rule: str = "stop_first"
    # Where to find 5m bars for the first-touch simulator.
    feature_5m_subdir: str = "5m"

    # ---- Score-driven exit ----
    # Replaces or supplements the triple-barrier STOP leg with a model-driven
    # exit. Target and timeout legs are unchanged.
    #   "triple_barrier"  : current — target + stop + timeout (no model re-check)
    #   "score_rerank"    : target + rerank-out-of-top-K (no price stop) + timeout
    #   "score_floor"     : target + score-below-floor (no price stop) + timeout
    #   "hybrid_rerank"   : target + stop + rerank-out-of-top-K + timeout
    #   "hybrid_floor"    : target + stop + score-below-floor + timeout
    # In score-driven modes, the held symbol's score is re-checked at each
    # 30-min cross-section anchor within the [entry, entry+h_max] window.
    # If exit_mode != "triple_barrier", `use_first_touch` is implied True.
    exit_mode: str = "triple_barrier"
    # For "*_rerank" modes: exit when held symbol's rank in current cross-section
    # exceeds this value. Default 3 = exit if not in top-3 anymore (entry was top-1).
    score_rerank_k: int = 3
    # For "*_floor" modes: exit when held symbol's score in current cs < this.
    score_floor: float = 0.0
    # ---- 2-D exit curve: hold decision based on (|current_return|, rank) ----
    # When set (non-empty), takes precedence over score_rerank_k AND disables
    # the static price-target. The curve is a list of (|r| breakpoint,
    # max_rank_to_hold) tuples ascending in |r|. At every bar in the walk:
    #   - compute |current_return|
    #   - find smallest breakpoint > |current_return|; use its max_rank
    #   - if |r| exceeds all breakpoints, use the LAST bucket's max_rank
    #   - exit at bar close when (rank-from-most-recent-anchor) > max_rank
    # This realizes "higher |return|, higher conviction (lower rank) required
    # to keep holding". Subsumes both stop replacement AND target gating.
    # Example: [(0.001, 8), (0.003, 4), (0.005, 2), (0.01, 1)]
    #   |r|<0.1% → hold if rank ≤ 8       (very forgiving on small moves)
    #   0.1%-0.3% → hold if rank ≤ 4
    #   0.3%-0.5% → hold if rank ≤ 2      (at-target gating)
    #   0.5%-1.0% → hold if rank ≤ 1
    #   >1.0%   → hold if rank ≤ 1      (carries last bucket)
    # Static stop still applies if exit_mode is hybrid_*.
    score_exit_curve: list[tuple[float, int]] = field(default_factory=list)

    drop_invalid_threshold: float = 0.3


@dataclass
class Trade:
    t: datetime
    symbol: str
    direction: int           # +1 LONG, -1 SHORT
    score: float             # model's P(LONG) or P(SHORT)
    raw_return: float        # signed forward return at sampler horizon
    cost: float              # round-trip frictions (fraction)
    net_return: float        # raw_return * direction − cost
    notional: float          # position size in currency
    pnl: float               # net_return * notional


@dataclass
class BacktestResult:
    cfg: BacktestConfig
    trades: list[Trade]
    n_trades: int
    n_long: int
    n_short: int
    total_pnl: float
    total_return_pct: float
    win_rate: float
    avg_winner: float
    avg_loser: float
    win_loss_ratio: float
    sharpe_daily: float
    max_drawdown_pct: float
    daily_pnl: dict[date, float]
    cumulative_pnl: list[tuple[datetime, float]]

    def _cost_line(self) -> str:
        notional = self.cfg.initial_capital * self.cfg.position_size_pct
        if self.cfg.round_trip_cost > 0:
            return f"Round-trip cost:  {self.cfg.round_trip_cost*10000:.0f} bps  (legacy override)"
        rt_inr = 2 * self.cfg.fixed_brokerage_per_leg_inr + (
            self.cfg.statutory_cost_pct + self.cfg.slippage_pct
        ) * notional
        rt_bps = rt_inr / notional * 10000
        brk_bps = 2 * self.cfg.fixed_brokerage_per_leg_inr / notional * 10000
        return (f"Round-trip cost:  {rt_bps:.1f} bps "
                f"(brk={brk_bps:.1f} + stat={self.cfg.statutory_cost_pct*10000:.1f} "
                f"+ slip={self.cfg.slippage_pct*10000:.1f})")

    def tearsheet(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST TEARSHEET",
            "=" * 60,
            f"Window:           {self.cfg.test_start.date()} → {self.cfg.test_end.date()}",
            f"Universe size:    {len(self.cfg.universe)}",
            f"Trade rules:      top-{self.cfg.long_top_k} LONG, top-{self.cfg.short_top_k} SHORT",
            f"                  long_T={self.cfg.long_score_threshold}, short_T={self.cfg.short_score_threshold}",
            f"Position size:    {self.cfg.position_size_pct:.1%} of ₹{self.cfg.initial_capital:,.0f} "
            f"= ₹{self.cfg.initial_capital * self.cfg.position_size_pct:,.0f}/trade",
            self._cost_line(),
            "-" * 60,
            f"Total trades:     {self.n_trades:5d}  ({self.n_long} LONG / {self.n_short} SHORT)",
            f"Win rate:         {self.win_rate:.2%}",
            f"Avg winner:       {self.avg_winner*10000:+7.1f} bps",
            f"Avg loser:        {self.avg_loser*10000:+7.1f} bps",
            f"Win/Loss ratio:   {self.win_loss_ratio:.2f}",
            "-" * 60,
            f"Total PnL:        ₹ {self.total_pnl:+10,.0f}",
            f"Total return:     {self.total_return_pct:+.2%}  on capital",
            f"Sharpe (daily):   {self.sharpe_daily:+.2f}",
            f"Max drawdown:     {self.max_drawdown_pct:.2%}",
            "=" * 60,
        ]
        if self.daily_pnl:
            lines.append("Daily PnL:")
            for d, p in sorted(self.daily_pnl.items()):
                lines.append(f"  {d}: ₹ {p:+9,.0f}")
            lines.append("=" * 60)
        return "\n".join(lines)


def _to_dev(cs: CrossSection, device: torch.device) -> CrossSection:
    return CrossSection(
        t=cs.t, horizon=cs.horizon, symbols=cs.symbols,
        inputs={s: t.to(device) for s, t in cs.inputs.items()},
        labels=cs.labels.to(device), ranks=cs.ranks.to(device),
        valid_mask=cs.valid_mask.to(device), vol_bucket=cs.vol_bucket.to(device),
        aux_features=(cs.aux_features.to(device) if cs.aux_features is not None else None),
        ob_snapshot_window=(cs.ob_snapshot_window.to(device) if cs.ob_snapshot_window is not None else None),
        ob_mask=(cs.ob_mask.to(device) if cs.ob_mask is not None else None),
        tradeable_label=cs.tradeable_label.to(device) if cs.tradeable_label is not None else None,
        tradeable_valid=cs.tradeable_valid.to(device) if cs.tradeable_valid is not None else None,
    )


class _BarCache:
    """Lazy per-symbol 5m bars cache for the first-touch simulator."""

    def __init__(self, feature_root: Path, subdir: str) -> None:
        self.root = Path(feature_root) / subdir
        self._cache: dict[str, tuple] = {}

    def get(self, symbol: str):
        if symbol in self._cache:
            return self._cache[symbol]
        path = self.root / f"{symbol}.parquet"
        if not path.exists():
            self._cache[symbol] = None
            return None
        import polars as pl
        df = pl.read_parquet(path).sort("ts").select(["ts", "high", "low", "close"])
        ts_arr = df["ts"].to_numpy()
        ts_arr_dt = np.array([t.astype("datetime64[us]").astype(object).replace(tzinfo=UTC)
                              for t in ts_arr])
        highs  = df["high"].to_numpy().astype(np.float64)
        lows   = df["low"].to_numpy().astype(np.float64)
        closes = df["close"].to_numpy().astype(np.float64)
        self._cache[symbol] = (ts_arr_dt, highs, lows, closes)
        return self._cache[symbol]


def _ambiguous_pnl(target_pct: float, stop_pct: float,
                    bar_high: float, bar_low: float, entry_px: float,
                    rule: str) -> float:
    """PnL when both target AND stop are touched within a single 5m bar."""
    if rule == "stop_first":
        return -stop_pct
    if rule == "target_first":
        return target_pct
    if rule == "midbar":
        return (target_pct - stop_pct) / 2.0
    if rule == "proportional":
        # Split based on distance from entry to high vs low.
        up   = (bar_high - entry_px) / entry_px
        down = (entry_px - bar_low) / entry_px
        total = up + down
        if total <= 0:
            return 0.0
        # P(target_first) ≈ down / total (closer barrier hits first on RANDOM
        # walk within the bar — i.e., the smaller distance is more likely to
        # be reached first). LONG: high distance = up, low distance = down.
        p_target = down / total
        return p_target * target_pct + (1 - p_target) * (-stop_pct)
    # Fallback to most conservative
    return -stop_pct


def _first_touch_pnl(
    bar_cache: _BarCache, symbol: str, entry_t: datetime,
    direction: int,
    target_pct: float, stop_pct: float, h_max_minutes: int,
    ambiguous_rule: str = "stop_first",
) -> tuple[float, str, float] | None:
    """Simulate first-touch exit. Returns (raw_return, outcome, time_minutes)
    or None if data is missing.

    `ambiguous_rule` governs the case when a single 5m bar's [low, high]
    range crosses BOTH the target and the stop (in-bar order unknown):
      - "stop_first"   : assume stop hit first (conservative; default)
      - "target_first" : assume target hit first (optimistic)
      - "midbar"       : assume avg-of-barriers exit (= (target - stop) / 2)
      - "proportional" : split by distance from entry to high vs low

    direction = +1 LONG, -1 SHORT.
    target_pct / stop_pct are POSITIVE fractions.
    """
    from datetime import timedelta
    data = bar_cache.get(symbol)
    if data is None:
        return None
    ts_arr_dt, highs, lows, closes = data
    n = len(ts_arr_dt)
    # Find first bar AT OR AFTER entry_t.
    i_entry = None
    for k, t in enumerate(ts_arr_dt):
        if t >= entry_t:
            i_entry = k; break
    if i_entry is None:
        return None
    entry_px = closes[i_entry]
    if entry_px <= 0:
        return None
    t_end = entry_t + timedelta(minutes=h_max_minutes)
    # Walk forward bars; on first-touch return.
    for k in range(i_entry + 1, n):
        t_k = ts_arr_dt[k]
        if t_k > t_end:
            break
        # Same trading day check: if hour rolled over to a new day, stop.
        if t_k.date() != entry_t.date():
            break
        if direction == +1:
            target_px = entry_px * (1 + target_pct)
            stop_px   = entry_px * (1 - stop_pct)
            hit_t = highs[k] >= target_px
            hit_s = lows[k]  <= stop_px
            if hit_t and hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                pnl = _ambiguous_pnl(target_pct, stop_pct,
                                      highs[k], lows[k], entry_px,
                                      ambiguous_rule)
                return (pnl, "ambig", mins)
            if hit_t:
                mins = (t_k - entry_t).total_seconds() / 60
                return (target_pct, "target", mins)
            if hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                return (-stop_pct, "stop", mins)
        else:  # SHORT
            target_px = entry_px * (1 - target_pct)
            stop_px   = entry_px * (1 + stop_pct)
            hit_t = lows[k]  <= target_px
            hit_s = highs[k] >= stop_px
            if hit_t and hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                pnl = _ambiguous_pnl(target_pct, stop_pct,
                                      highs[k], lows[k], entry_px,
                                      ambiguous_rule)
                return (pnl, "ambig", mins)
            if hit_t:
                mins = (t_k - entry_t).total_seconds() / 60
                return (target_pct, "target", mins)
            if hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                return (-stop_pct, "stop", mins)
    # Timeout — exit at last available close within window.
    last_k = i_entry + 1
    for k in range(i_entry + 1, n):
        if ts_arr_dt[k] > t_end or ts_arr_dt[k].date() != entry_t.date():
            break
        last_k = k
    final_ret = (closes[last_k] - entry_px) / entry_px
    if direction == -1:
        final_ret = -final_ret
    mins = (ts_arr_dt[last_k] - entry_t).total_seconds() / 60
    return (final_ret, "timeout", mins)


def _rank_at_or_before(
    t_k: datetime, anchor_ts_sorted: list[datetime],
    scores_by_t: dict[datetime, dict[str, float]], symbol: str,
) -> int | None:
    """Rank (1-indexed, strict-greater-than) of `symbol` in the most-recent
    cross-section with anchor <= t_k. Returns None if no data."""
    import bisect
    i = bisect.bisect_right(anchor_ts_sorted, t_k) - 1
    if i < 0:
        return None
    anchor = anchor_ts_sorted[i]
    cs_scores = scores_by_t.get(anchor)
    if cs_scores is None or symbol not in cs_scores:
        return None
    held_score = cs_scores[symbol]
    return sum(1 for s in cs_scores.values() if s > held_score) + 1


def _curve_lookup(curve: list[tuple[float, int]], r: float) -> int | None:
    """Return max_rank_to_hold for the bucket containing r.

    Curve is ascending in r-breakpoint. Two modes (auto-detected):
      - SYMMETRIC: all breakpoints non-negative → lookup key is abs(r).
        E.g. [(0.001,8),(0.005,3),(0.01,1)] treats +50bps and -50bps identically.
      - ASYMMETRIC: at least one negative breakpoint → lookup key is signed r.
        E.g. [(-0.005,2),(-0.001,8),(0.001,8),(0.005,3),(0.01,1)] gives
        different policy for losers vs winners.
    If r is past the largest breakpoint, the last bucket's value carries.
    """
    if not curve:
        return None
    has_neg = curve[0][0] < 0   # curve is sorted ascending; first < 0 ⇒ asymmetric
    test_r = r if has_neg else abs(r)
    for r_bp, max_k in curve:
        if test_r < r_bp:
            return max_k
    return curve[-1][1]


def _score_driven_pnl(
    bar_cache: _BarCache, symbol: str, entry_t: datetime, direction: int,
    target_pct: float, stop_pct: float, h_max_minutes: int,
    scores_by_t: dict[datetime, dict[str, float]],
    anchor_ts_sorted: list[datetime],
    exit_mode: str,
    score_rerank_k: int,
    score_floor: float,
    score_exit_curve: list[tuple[float, int]] | None = None,
    ambiguous_rule: str = "stop_first",
) -> tuple[float, str, float] | None:
    """Triple-barrier walk with model-driven exit replacing/supplementing stop.

    Walks 5m bars [entry_t, entry_t+h_max). At each bar:
      - Always check target hit (and stop hit if exit_mode starts with "hybrid_").
      - If the bar timestamp coincides with a cross-section anchor inside the
        hold window, look up the symbol's score in that cross-section and
        apply the exit rule:
          * "score_rerank" / "hybrid_rerank" : exit if rank > score_rerank_k
          * "score_floor"  / "hybrid_floor"  : exit if score < score_floor
        On exit, realize the bar-close return (signed by direction).
      - On timeout, realize last available close.
    """
    from datetime import timedelta
    data = bar_cache.get(symbol)
    if data is None:
        return None
    ts_arr_dt, highs, lows, closes = data
    n = len(ts_arr_dt)
    i_entry = None
    for k, t in enumerate(ts_arr_dt):
        if t >= entry_t:
            i_entry = k
            break
    if i_entry is None:
        return None
    entry_px = closes[i_entry]
    if entry_px <= 0:
        return None
    t_end = entry_t + timedelta(minutes=h_max_minutes)
    # Exit anchors in (entry_t, t_end] — score-check timestamps.
    exit_anchors = {a for a in anchor_ts_sorted if entry_t < a <= t_end}

    has_price_stop   = exit_mode in ("hybrid_rerank", "hybrid_floor")
    score_mode_rerank = exit_mode in ("score_rerank", "hybrid_rerank")
    score_mode_floor  = exit_mode in ("score_floor",  "hybrid_floor")
    use_curve = bool(score_exit_curve)
    # When curve is in use: static price-target is disabled (curve handles target
    # gating). Stop still active if hybrid mode. Anchor-only rerank/floor checks
    # are also superseded by the curve (which runs every bar).

    for k in range(i_entry + 1, n):
        t_k = ts_arr_dt[k]
        if t_k > t_end:
            break
        if t_k.date() != entry_t.date():
            break
        # ---- Target / optional Stop ----
        if direction == +1:
            target_px = entry_px * (1 + target_pct)
            stop_px   = entry_px * (1 - stop_pct)
            hit_t = (not use_curve) and (highs[k] >= target_px)
            hit_s = has_price_stop and (lows[k] <= stop_px)
            if hit_t and hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                pnl = _ambiguous_pnl(target_pct, stop_pct,
                                     highs[k], lows[k], entry_px, ambiguous_rule)
                return (pnl, "ambig", mins)
            if hit_t:
                mins = (t_k - entry_t).total_seconds() / 60
                return (target_pct, "target", mins)
            if hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                return (-stop_pct, "stop", mins)
        else:  # SHORT
            target_px = entry_px * (1 - target_pct)
            stop_px   = entry_px * (1 + stop_pct)
            hit_t = (not use_curve) and (lows[k]  <= target_px)
            hit_s = has_price_stop and (highs[k] >= stop_px)
            if hit_t and hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                pnl = _ambiguous_pnl(target_pct, stop_pct,
                                     highs[k], lows[k], entry_px, ambiguous_rule)
                return (pnl, "ambig", mins)
            if hit_t:
                mins = (t_k - entry_t).total_seconds() / 60
                return (target_pct, "target", mins)
            if hit_s:
                mins = (t_k - entry_t).total_seconds() / 60
                return (-stop_pct, "stop", mins)
        # ---- 2-D Curve exit (every bar; uses carry-forward rank) ----
        if use_curve:
            current_r = direction * (closes[k] - entry_px) / entry_px
            # _curve_lookup auto-detects symmetric (|r|) vs asymmetric (signed r).
            max_rank = _curve_lookup(score_exit_curve, current_r)
            if max_rank is not None:
                rank = _rank_at_or_before(t_k, anchor_ts_sorted, scores_by_t, symbol)
                if rank is not None and rank > max_rank:
                    mins = (t_k - entry_t).total_seconds() / 60
                    return (current_r, "curve_exit", mins)
        # ---- Anchor-based score-driven exit (skipped when curve is in use) ----
        if (not use_curve) and t_k in exit_anchors:
            cs_scores = scores_by_t.get(t_k)
            if cs_scores is not None and symbol in cs_scores:
                held_score = cs_scores[symbol]
                if score_mode_rerank:
                    # Strict greater-than → ties favour the held position.
                    rank = sum(1 for s in cs_scores.values() if s > held_score) + 1
                    if rank > score_rerank_k:
                        ret = (closes[k] - entry_px) / entry_px
                        if direction == -1:
                            ret = -ret
                        mins = (t_k - entry_t).total_seconds() / 60
                        return (ret, "score_rerank_exit", mins)
                if score_mode_floor:
                    if held_score < score_floor:
                        ret = (closes[k] - entry_px) / entry_px
                        if direction == -1:
                            ret = -ret
                        mins = (t_k - entry_t).total_seconds() / 60
                        return (ret, "score_floor_exit", mins)
    # Timeout — exit at last close within window.
    last_k = i_entry + 1
    for k in range(i_entry + 1, n):
        if ts_arr_dt[k] > t_end or ts_arr_dt[k].date() != entry_t.date():
            break
        last_k = k
    final_ret = (closes[last_k] - entry_px) / entry_px
    if direction == -1:
        final_ret = -final_ret
    mins = (ts_arr_dt[last_k] - entry_t).total_seconds() / 60
    return (final_ret, "timeout", mins)


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(payload["module_cfg"])
    # Path field fix-ups.
    for path_field in ("parent_checkpoint", "parent_ob_checkpoint",
                       "head_warmstart_path"):
        if isinstance(cfg_dict.get(path_field), str):
            cfg_dict[path_field] = Path(cfg_dict[path_field])
    # Don't try to re-load warmstart at backtest time.
    cfg_dict["head_warmstart_path"] = None

    # Dispatch: Stage 2 OB module vs Stage 1 module.
    is_ob = "parent_ob_checkpoint" in cfg_dict and cfg_dict["parent_ob_checkpoint"] is not None
    if is_ob:
        valid = {f.name for f in dataclasses.fields(TradeableOBFineTuneConfig)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
        cfg = TradeableOBFineTuneConfig(**cfg_dict)
        m = TradeableOBFineTuneModule(cfg)
    else:
        valid = {f.name for f in dataclasses.fields(TradeableFineTuneConfig)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
        cfg = TradeableFineTuneConfig(**cfg_dict)
        m = TradeableFineTuneModule(cfg)
    missing, unexpected = m.load_state_dict(payload["state_dict"], strict=False)
    log.info("backtest: loaded model from %s (%d missing, %d unexpected, ob=%s)",
             ckpt_path.name, len(missing), len(unexpected), is_ob)
    return m.to(device).eval()


def run_backtest(ckpt_path: Path, cfg: BacktestConfig) -> BacktestResult:
    """Execute the backtest end-to-end."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, device)
    secondary_model = None
    if cfg.secondary_checkpoint is not None:
        log.info("Loading ensemble secondary checkpoint: %s", cfg.secondary_checkpoint)
        secondary_model = _load_model(cfg.secondary_checkpoint, device)

    # Build sampler (auto-picks universe if cfg.universe is empty).
    feature_store = ParquetStore(cfg.feature_root)
    if not cfg.universe:
        cfg.universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    sampler = CrossSectionalSampler(
        feature_store=feature_store, label_root=cfg.label_root,
        universe=cfg.universe, as_of=cfg.as_of,
        window_bars={"5m": 256, "15m": 192, "60m": 168, "day": 64},
        min_end_ts=cfg.test_start,
        drop_invalid_threshold=cfg.drop_invalid_threshold,
        tradeable_root=cfg.tradeable_root,
        ob_root=cfg.ob_root,
    )

    score_driven = cfg.exit_mode != "triple_barrier"
    # Score-driven exits require 5m bar simulation regardless of use_first_touch.
    need_bars = cfg.use_first_touch or score_driven
    bar_cache = _BarCache(cfg.feature_root, cfg.feature_5m_subdir) if need_bars else None
    notional_per_trade = cfg.initial_capital * cfg.position_size_pct
    # Round-trip cost expressed as a fraction of notional.
    if cfg.round_trip_cost > 0:
        rt_cost_frac = cfg.round_trip_cost
    else:
        rt_cost_inr = 2 * cfg.fixed_brokerage_per_leg_inr + (
            cfg.statutory_cost_pct + cfg.slippage_pct
        ) * notional_per_trade
        rt_cost_frac = rt_cost_inr / notional_per_trade
    log.info("backtest cost: notional=₹%.0f  round-trip=%.1f bps  "
             "(fixed=₹%.0f/leg + stat=%.1f bps + slip=%.1f bps)",
             notional_per_trade, rt_cost_frac * 10000,
             cfg.fixed_brokerage_per_leg_inr,
             cfg.statutory_cost_pct * 10000, cfg.slippage_pct * 10000)
    trades: list[Trade] = []
    # Deferred-resolution candidates (used only when score_driven).
    # Each: (entry_t, sym, direction, score, raw_return_at_entry_anchor)
    # Candidate tuple now carries per-trade notional (conviction-sized).
    # Tuple: (entry_t, sym, direction, score, label_return, notional_per_trade)
    candidates: list[tuple[datetime, str, int, float, float, float]] = []
    # Per-cross-section score maps for re-rank / floor exits.
    scores_by_t: dict[datetime, dict[str, float]] = {}
    anchor_ts: list[datetime] = []
    n_cs = 0
    with torch.no_grad():
        for cs in sampler.iter_30m():
            if cs.t > cfg.test_end:
                break
            if cs.tradeable_valid is None:
                continue
            valid = cs.tradeable_valid.numpy().astype(bool)
            if valid.sum() < cfg.min_valid_per_cs:
                continue
            out = model(_to_dev(cs, device), horizon="30m")
            long_p  = out["long_probs"].cpu().numpy()
            short_p = out["short_probs"].cpu().numpy()
            # Ensemble: combine with secondary model via within-cs rank averaging.
            if secondary_model is not None:
                out2 = secondary_model(_to_dev(cs, device), horizon="30m")
                long_p2 = out2["long_probs"].cpu().numpy()
                # Per-cs rank percentile of each score. Use argsort-of-argsort
                # to convert raw scores → rank in [0, 1].
                n = len(long_p)
                rank1 = np.argsort(np.argsort(long_p)).astype(np.float32) / max(n - 1, 1)
                rank2 = np.argsort(np.argsort(long_p2)).astype(np.float32) / max(n - 1, 1)
                w2 = cfg.ensemble_secondary_weight
                ensemble_long = (1.0 - w2) * rank1 + w2 * rank2
                # Replace long_p with ensemble rank (range [0, 1]).
                long_p = ensemble_long
            rets   = cs.labels.numpy()                  # forward return at sampler horizon
            syms   = cs.symbols

            valid_idx = np.where(valid)[0]
            # Stash scores for every VALID symbol at this anchor — needed for
            # rerank / floor exit decisions in the score-driven modes.
            if score_driven:
                # Use long combined score (same expression as the LONG entry path).
                combined_long_full = long_p - cfg.long_short_alpha * short_p
                cs_scores: dict[str, float] = {}
                for vi in valid_idx:
                    cs_scores[syms[vi]] = float(combined_long_full[vi])
                scores_by_t[cs.t] = cs_scores
                anchor_ts.append(cs.t)

            # ---- LONG side ----
            if cfg.side in ("long_only", "both"):
                # Combined score: P(LONG) − α * P(SHORT). α=0 → pure P(LONG).
                combined_long = long_p - cfg.long_short_alpha * short_p
                # Hard filter: skip if P(SHORT) too high.
                short_ok = short_p <= cfg.long_short_avoidance_max
                eligible_long = valid_idx[short_ok[valid_idx]]
                order_long = eligible_long[np.argsort(-combined_long[eligible_long])]
                k_long = min(cfg.long_top_k, len(order_long))
                # Percentile-gate threshold: e.g. pct_gate=0.05 → require score
                # >= 95th percentile of cross-sectional combined_long over
                # eligible_long. Robust to score-distribution shift.
                pct_gate_thresh_long = -np.inf
                if cfg.long_score_pct_gate > 0.0 and len(eligible_long) > 1:
                    pct_gate_thresh_long = float(np.quantile(
                        combined_long[eligible_long],
                        1.0 - cfg.long_score_pct_gate,
                    ))
                # Pre-compute pct-rank lookup for sizing-by-conviction. Sort
                # combined_long over eligible set once; rank for each pick.
                use_pct_rank_size = (
                    cfg.position_size_mode == "score_pct_rank"
                    and len(eligible_long) > 1
                )
                if use_pct_rank_size:
                    sorted_combined_l = np.sort(combined_long[eligible_long])
                for i in order_long[:k_long]:
                    score = float(combined_long[i])
                    if score < cfg.long_score_threshold:
                        continue
                    if score < pct_gate_thresh_long:
                        continue
                    # Per-trade notional (flat or pct-rank conviction-weighted).
                    if use_pct_rank_size:
                        pct_rank = float(np.searchsorted(sorted_combined_l, score)) / len(sorted_combined_l)
                        per_trade_pct = (
                            cfg.position_size_min_pct
                            + (cfg.position_size_max_pct - cfg.position_size_min_pct) * pct_rank
                        )
                        per_trade_notional = cfg.initial_capital * per_trade_pct
                    else:
                        per_trade_notional = notional_per_trade
                    if score_driven:
                        # Defer resolution — need score_map across all anchors.
                        candidates.append((cs.t, syms[i], +1, score, float(rets[i]), per_trade_notional))
                        continue
                    if cfg.use_first_touch and bar_cache is not None:
                        ft = _first_touch_pnl(
                            bar_cache, syms[i], cs.t, direction=+1,
                            target_pct=cfg.target_long_pct,
                            stop_pct=cfg.stop_long_pct,
                            h_max_minutes=cfg.h_max_minutes,
                            ambiguous_rule=cfg.ambiguous_rule,
                        )
                        if ft is None:
                            continue
                        r, _outcome, _t_min = ft
                    else:
                        r = float(rets[i])
                    # Cost per trade: fixed brokerage + pct slippage of THIS trade's notional.
                    trade_cost_inr = 2 * cfg.fixed_brokerage_per_leg_inr + (
                        cfg.statutory_cost_pct + cfg.slippage_pct
                    ) * per_trade_notional
                    trade_cost_frac = trade_cost_inr / per_trade_notional if per_trade_notional > 0 else rt_cost_frac
                    net = r * 1 - trade_cost_frac
                    trades.append(Trade(
                        t=cs.t, symbol=syms[i], direction=+1, score=score,
                        raw_return=r, cost=trade_cost_frac,
                        net_return=net, notional=per_trade_notional,
                        pnl=net * per_trade_notional,
                    ))

            # ---- SHORT side ----
            if cfg.side in ("short_only", "both"):
                order_short = valid_idx[np.argsort(-short_p[valid_idx])]
                k_short = min(cfg.short_top_k, len(order_short))
                pct_gate_thresh_short = -np.inf
                if cfg.short_score_pct_gate > 0.0 and len(valid_idx) > 1:
                    pct_gate_thresh_short = float(np.quantile(
                        short_p[valid_idx], 1.0 - cfg.short_score_pct_gate,
                    ))
                for i in order_short[:k_short]:
                    score = float(short_p[i])
                    if score < cfg.short_score_threshold:
                        continue
                    if score < pct_gate_thresh_short:
                        continue
                    # Per-trade notional (flat or pct-rank for SHORT score).
                    if cfg.position_size_mode == "score_pct_rank" and len(valid_idx) > 1:
                        sorted_short = np.sort(short_p[valid_idx])
                        s_pct_rank = float(np.searchsorted(sorted_short, score)) / len(sorted_short)
                        per_trade_pct = (
                            cfg.position_size_min_pct
                            + (cfg.position_size_max_pct - cfg.position_size_min_pct) * s_pct_rank
                        )
                        per_trade_notional = cfg.initial_capital * per_trade_pct
                    else:
                        per_trade_notional = notional_per_trade
                    if score_driven:
                        candidates.append((cs.t, syms[i], -1, score, float(rets[i]), per_trade_notional))
                        continue
                    if cfg.use_first_touch and bar_cache is not None:
                        ft = _first_touch_pnl(
                            bar_cache, syms[i], cs.t, direction=-1,
                            target_pct=cfg.target_short_pct,
                            stop_pct=cfg.stop_short_pct,
                            h_max_minutes=cfg.h_max_minutes,
                            ambiguous_rule=cfg.ambiguous_rule,
                        )
                        if ft is None:
                            continue
                        r, _outcome, _t_min = ft
                    else:
                        r = float(rets[i])
                        r = r * (-1)
                    trade_cost_inr = 2 * cfg.fixed_brokerage_per_leg_inr + (
                        cfg.statutory_cost_pct + cfg.slippage_pct
                    ) * per_trade_notional
                    trade_cost_frac = trade_cost_inr / per_trade_notional if per_trade_notional > 0 else rt_cost_frac
                    net = r - trade_cost_frac
                    trades.append(Trade(
                        t=cs.t, symbol=syms[i], direction=-1, score=score,
                        raw_return=r, cost=trade_cost_frac,
                        net_return=net, notional=per_trade_notional,
                        pnl=net * per_trade_notional,
                    ))
            n_cs += 1

    # ---- Score-driven exit resolution (deferred until all scores are known) ----
    if score_driven and candidates:
        anchor_ts_sorted = sorted(anchor_ts)
        log.info("score-driven exit resolution: %d candidates over %d anchors "
                 "(mode=%s rerank_k=%d floor=%.4f)",
                 len(candidates), len(anchor_ts_sorted),
                 cfg.exit_mode, cfg.score_rerank_k, cfg.score_floor)
        n_resolved = 0
        n_dropped  = 0
        for (entry_t, sym, direction, score, r_label, per_trade_notional) in candidates:
            ft = _score_driven_pnl(
                bar_cache, sym, entry_t, direction,
                target_pct=cfg.target_long_pct if direction == +1 else cfg.target_short_pct,
                stop_pct=cfg.stop_long_pct  if direction == +1 else cfg.stop_short_pct,
                h_max_minutes=cfg.h_max_minutes,
                scores_by_t=scores_by_t, anchor_ts_sorted=anchor_ts_sorted,
                exit_mode=cfg.exit_mode,
                score_rerank_k=cfg.score_rerank_k,
                score_floor=cfg.score_floor,
                score_exit_curve=list(cfg.score_exit_curve) if cfg.score_exit_curve else None,
                ambiguous_rule=cfg.ambiguous_rule,
            )
            if ft is None:
                n_dropped += 1
                continue
            r, _outcome, _t_min = ft
            # Per-trade cost (size-aware).
            trade_cost_inr = 2 * cfg.fixed_brokerage_per_leg_inr + (
                cfg.statutory_cost_pct + cfg.slippage_pct
            ) * per_trade_notional
            trade_cost_frac = trade_cost_inr / per_trade_notional if per_trade_notional > 0 else rt_cost_frac
            net = r - trade_cost_frac
            trades.append(Trade(
                t=entry_t, symbol=sym, direction=direction, score=score,
                raw_return=r, cost=trade_cost_frac,
                net_return=net, notional=per_trade_notional,
                pnl=net * per_trade_notional,
            ))
            n_resolved += 1
        log.info("score-driven exit: resolved=%d dropped_no_bars=%d", n_resolved, n_dropped)

    return _summarize(trades, cfg, n_cs)


def _summarize(trades: list[Trade], cfg: BacktestConfig, n_cs: int) -> BacktestResult:
    if not trades:
        log.warning("backtest: no trades generated")
        return BacktestResult(
            cfg=cfg, trades=[], n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_return_pct=0.0, win_rate=0.0,
            avg_winner=0.0, avg_loser=0.0, win_loss_ratio=0.0,
            sharpe_daily=0.0, max_drawdown_pct=0.0,
            daily_pnl={}, cumulative_pnl=[],
        )

    n_long  = sum(1 for t in trades if t.direction == +1)
    n_short = sum(1 for t in trades if t.direction == -1)
    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    total_pnl = float(pnls.sum())
    winners = pnls[pnls > 0]
    losers  = pnls[pnls <= 0]
    win_rate = float(len(winners)) / len(pnls)
    avg_winner = float(winners.mean()) / cfg.initial_capital if len(winners) else 0.0
    avg_loser  = float(losers.mean())  / cfg.initial_capital if len(losers)  else 0.0
    win_loss_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else float("inf")

    # Daily aggregation
    daily: dict[date, float] = defaultdict(float)
    for t in trades:
        daily[t.t.date()] += t.pnl
    daily_arr = np.array([daily[d] for d in sorted(daily)], dtype=np.float64)
    # Sharpe (daily, no annualization here — small sample)
    sharpe = (daily_arr.mean() / daily_arr.std()) if daily_arr.std() > 0 else 0.0

    # Equity curve + max DD
    cum: list[tuple[datetime, float]] = []
    running = 0.0
    for t in sorted(trades, key=lambda x: x.t):
        running += t.pnl
        cum.append((t.t, running))
    peaks = np.maximum.accumulate([c[1] for c in cum])
    drawdowns = (peaks - [c[1] for c in cum])
    max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0
    max_dd_pct = max_dd / cfg.initial_capital

    return BacktestResult(
        cfg=cfg, trades=trades, n_trades=len(trades), n_long=n_long, n_short=n_short,
        total_pnl=total_pnl, total_return_pct=total_pnl / cfg.initial_capital,
        win_rate=win_rate, avg_winner=avg_winner, avg_loser=avg_loser,
        win_loss_ratio=win_loss_ratio,
        sharpe_daily=float(sharpe), max_drawdown_pct=max_dd_pct,
        daily_pnl=dict(daily), cumulative_pnl=cum,
    )
