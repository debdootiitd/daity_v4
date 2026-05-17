"""Intraday simulator V2: consumes multi-anchor predictions.

Differences from V1:
  - Strategy parquet has rows per (date, anchor_us, stock, horizon).
  - At each 5-min anchor, train a small per-anchor GBM gate on the last N
    days of the SAME anchor's predictions (so the calibration learns
    intraday-anchor-specific patterns).
  - Pick top-K candidates each anchor, open positions subject to cash + slot.
  - Walk-forward exits with stop-loss check on every 5m bar in-between.

Usage:
  python -m daity.scripts.intraday_simulator_v2 \
      --predictions-path runs/cohort_modeB_v12_multianchor_5min/predictions.parquet \
      --feature-root data/features_parquet --cache-root data/cache \
      --test-start 2025-02-01 --test-end 2026-04-30 \
      --target-horizon next_day_eod \
      --n-lookback-days 10 --holdout-days 3 \
      --top-k-per-anchor 1 --max-positions 10 --position-frac 0.10 \
      --stop-bps 400 --cost-bps-round-trip 15 \
      --out-prefix reports/v12_strategy/intraday_v2 \
      --wandb-project daity_v12_rerun
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


HORIZON_EXIT = {
    "to_close":      ("15:30", 0),
    "next_day_eod":  ("15:30", 1),
    "next_day_1h":   ("10:15", 1),
    "overnight":     ("09:15", 1),
    "intraday_30m":  ("anchor+30m", 0),
    "intraday_60m":  ("anchor+60m", 0),
    "intraday_120m": ("anchor+120m", 0),
    "intraday_180m": ("anchor+180m", 0),
}

WIN_THRESHOLD_BPS = 30.0
PROB_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
GBM_PARAMS = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}


@dataclass
class Position:
    stock: str
    entry_ts: datetime
    entry_price: float
    shares: float
    stop_price: float
    target_ts: datetime
    exit_ts: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""

    @property
    def pnl_bps(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 10000.0


@dataclass
class Book:
    initial_capital: float
    cash: float = 0.0
    open: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    def held_stocks(self) -> set[str]:
        return {p.stock for p in self.open}

    def equity(self, prices: dict[str, float]) -> float:
        eq = self.cash
        for p in self.open:
            eq += p.shares * prices.get(p.stock, p.entry_price)
        return eq

    def open_position(self, stock: str, ts: datetime, raw_px: float,
                       size_dollars: float, stop_pct: float, target_ts: datetime,
                       cost_bps: float) -> Position | None:
        eff_px = raw_px * (1.0 + cost_bps / 2.0 / 10000.0)
        shares = size_dollars / eff_px
        if shares <= 0:
            return None
        p = Position(stock=stock, entry_ts=ts, entry_price=eff_px,
                     shares=shares, stop_price=raw_px * stop_pct, target_ts=target_ts)
        self.cash -= shares * eff_px
        self.open.append(p)
        return p

    def close_position(self, p: Position, ts: datetime, raw_exit_px: float,
                       reason: str, cost_bps: float) -> None:
        eff_px = raw_exit_px * (1.0 - cost_bps / 2.0 / 10000.0)
        p.exit_ts = ts
        p.exit_price = eff_px
        p.exit_reason = reason
        self.cash += p.shares * eff_px
        self.open.remove(p)
        self.closed.append(p)


# Per-stock 5m bar cache
_bar_cache: dict[str, pl.DataFrame] = {}

def _load_bars(feature_root: Path, sym: str) -> pl.DataFrame | None:
    if sym not in _bar_cache:
        path = feature_root / "5m" / f"{sym}.parquet"
        if not path.exists():
            _bar_cache[sym] = pl.DataFrame()
            return None
        try:
            _bar_cache[sym] = pl.read_parquet(path).sort("ts")
        except Exception:
            _bar_cache[sym] = pl.DataFrame()
            return None
    df = _bar_cache[sym]
    return df if df.height > 0 else None


def _ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--cache-root",   type=Path, default=Path("data/cache"))
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end",   type=str, required=True)
    ap.add_argument("--target-horizon", type=str, default="next_day_eod")
    ap.add_argument("--n-lookback-days", type=int, default=10)
    ap.add_argument("--holdout-days", type=int, default=3)
    ap.add_argument("--top-k-per-anchor", type=int, default=1)
    ap.add_argument("--max-positions", type=int, default=10)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--initial-capital", type=float, default=1_000_000.0)
    ap.add_argument("--stop-bps", type=float, default=400.0)
    ap.add_argument("--cost-bps-round-trip", type=float, default=15.0)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    return ap.parse_args()


def _next_trading_day(cal, d: date, n: int) -> date | None:
    if n == 0: return d
    cur = d
    for _ in range(n):
        cur += timedelta(days=1)
        while not cal.is_trading_day(cur):
            cur += timedelta(days=1)
            if (cur - d).days > 30: return None
    return cur


def main() -> int:
    args = parse_args()
    from daity.data.calendar_nse import NSECalendar
    cal = NSECalendar.from_cache(args.cache_root)

    print(f"loading {args.predictions_path}...", flush=True)
    df = pl.read_parquet(args.predictions_path)
    print(f"  rows={df.shape[0]}, cols={df.columns}", flush=True)

    # Filter target horizon and test window
    target_h = args.target_horizon
    df = df.filter(pl.col("horizon") == target_h)
    if df.shape[0] == 0:
        print(f"ERROR: no rows for horizon {target_h}", file=sys.stderr); return 1
    win_thr = WIN_THRESHOLD_BPS / 10000.0
    df = df.with_columns([
        pl.col("date").cast(pl.Date),
        (pl.col("real_lr") > win_thr).cast(pl.Int64).alias("win"),
    ])

    # All distinct (date, anchor_us) — these are decision moments
    anchors_df = df.select(["date", "anchor_us"]).unique().sort(["date", "anchor_us"])

    test_start = date.fromisoformat(args.test_start)
    test_end = date.fromisoformat(args.test_end)
    test_anchors = anchors_df.filter((pl.col("date") >= test_start) & (pl.col("date") <= test_end))
    print(f"  test anchors: {test_anchors.shape[0]}", flush=True)

    book = Book(initial_capital=args.initial_capital)
    pos_dollars = args.initial_capital * args.position_frac
    stop_pct = 1.0 - args.stop_bps / 10000.0

    exit_t_str, n_fwd = HORIZON_EXIT[target_h]

    equity_curve = []
    trades = []
    last_anchor_utc = None

    # Pre-build (anchor_us -> rows) index for quick lookup
    print("indexing predictions by (date, anchor_us)...", flush=True)
    df_by_anchor = df.partition_by(["date", "anchor_us"], as_dict=True)

    # All test anchors, in time order
    anchor_list = test_anchors.iter_rows()
    n_iters = 0
    for (d_anchor, anchor_us) in anchor_list:
        n_iters += 1
        anchor_utc = datetime.fromtimestamp(anchor_us / 1e6, tz=UTC)

        # Step 1: walk all open positions through bars in (last_anchor, current_anchor]
        if last_anchor_utc is not None:
            for p in list(book.open):
                bars = _load_bars(args.feature_root, p.stock)
                if bars is None: continue
                window = bars.filter((pl.col("ts") > last_anchor_utc) & (pl.col("ts") <= anchor_utc))
                if window.height == 0: continue
                for row in window.iter_rows(named=True):
                    ts = row["ts"]
                    if ts >= p.target_ts:
                        book.close_position(p, ts, float(row["close"]), "target", args.cost_bps_round_trip)
                        break
                    if float(row["low"]) <= p.stop_price:
                        book.close_position(p, ts, p.stop_price, "stop", args.cost_bps_round_trip)
                        break

        # Step 2: this anchor's GBM gate decision
        # Build training data: the last N days' rows at the SAME anchor-time (anchor_ist).
        ist_t = (anchor_utc + timedelta(hours=5, minutes=30)).time()
        # Get historical anchors with same IST time, before this anchor
        same_ist = anchors_df.filter(
            (pl.col("date") < d_anchor)
            & (pl.col("anchor_us") % (24 * 3600 * 1_000_000) ==
               (ist_t.hour * 3600 + ist_t.minute * 60) * 1_000_000 -
               (5 * 3600 + 30 * 60) * 1_000_000)  # convert IST to UTC seconds-of-day
        ).sort("date").tail(args.n_lookback_days + args.holdout_days)
        if same_ist.shape[0] < args.n_lookback_days + args.holdout_days:
            # Not enough history → skip this anchor (no decision)
            last_anchor_utc = anchor_utc
            continue

        fit_dates = same_ist["date"].head(args.n_lookback_days).to_list()
        hold_dates = same_ist["date"].tail(args.holdout_days).to_list()

        fit_rows_list = [df_by_anchor.get((fd, a), None)
                         for fd in fit_dates
                         for a in same_ist.filter(pl.col("date") == fd)["anchor_us"].to_list()]
        fit_rows = pl.concat([r for r in fit_rows_list if r is not None]) if any(r is not None for r in fit_rows_list) else None
        if fit_rows is None or fit_rows.shape[0] < 50:
            last_anchor_utc = anchor_utc
            continue

        # Just use pred_lr alone as the single feature (multi-horizon merge would need a wide-pivot of all 8 horizons)
        # For V2 simplicity: bin pred_lr percentile into a single threshold strategy without GBM.
        # Pick top-K stocks by pred_lr if their pred_lr is above a chosen threshold.
        # Threshold is selected on the holdout window to maximize P&L.
        hold_rows_list = [df_by_anchor.get((hd, a), None)
                          for hd in hold_dates
                          for a in same_ist.filter(pl.col("date") == hd)["anchor_us"].to_list()]
        hold_rows = pl.concat([r for r in hold_rows_list if r is not None]) if any(r is not None for r in hold_rows_list) else None
        # Simple percentile-based threshold: try pick top-X% from each holdout day, find best X
        best_pct = None; best_score = -1e9
        if hold_rows is not None and hold_rows.shape[0] > 0:
            for pct in (0.5, 1, 2, 5):  # top-X% by pred_lr
                day_pnls = []
                for hd in hold_dates:
                    sub = hold_rows.filter(pl.col("date") == hd).sort("pred_lr", descending=True)
                    if sub.shape[0] == 0: continue
                    top_n = max(1, int(sub.shape[0] * pct / 100))
                    picks = sub.head(top_n)
                    avg_real = float(picks["real_lr"].mean())
                    day_pnls.append((np.exp(avg_real) - 1) * 10000.0 - args.cost_bps_round_trip)
                if len(day_pnls) >= 2:
                    mean_p = float(np.mean(day_pnls))
                    if mean_p > best_score:
                        best_score = mean_p; best_pct = pct
        if best_pct is None or best_score <= 0:
            last_anchor_utc = anchor_utc
            continue

        # Step 3: apply to today's anchor — pick top-K stocks above the percentile threshold
        today_rows = df_by_anchor.get((d_anchor, anchor_us), None)
        if today_rows is None or today_rows.shape[0] == 0:
            last_anchor_utc = anchor_utc
            continue
        sorted_today = today_rows.sort("pred_lr", descending=True)
        cand_n = max(args.top_k_per_anchor, int(sorted_today.shape[0] * best_pct / 100))
        cand = sorted_today.head(cand_n)["stock"].to_list()[:args.top_k_per_anchor]

        # Step 4: open positions if cash + slot available + not already held
        target_d = _next_trading_day(cal, d_anchor, n_fwd)
        if target_d is None: last_anchor_utc = anchor_utc; continue
        if exit_t_str.startswith("anchor+"):
            mins = int(exit_t_str.removeprefix("anchor+").removesuffix("m"))
            target_utc = anchor_utc + timedelta(minutes=mins)
        else:
            eh, em = (int(x) for x in exit_t_str.split(":"))
            target_utc = _ist_to_utc(target_d, dtime(eh, em))
        held = book.held_stocks()
        for sym in cand:
            if len(book.open) >= args.max_positions: break
            if sym in held: continue
            if book.cash < pos_dollars: continue
            bars = _load_bars(args.feature_root, sym)
            if bars is None: continue
            entry_bar = bars.filter(pl.col("ts") >= anchor_utc).head(1)
            if entry_bar.height == 0: continue
            entry_px = float(entry_bar["open"][0])
            if entry_px <= 0: continue
            book.open_position(sym, anchor_utc, entry_px, pos_dollars, stop_pct,
                                target_utc, args.cost_bps_round_trip)

        # Step 5: mark to market
        prices_now = {}
        for p in book.open:
            bars = _load_bars(args.feature_root, p.stock)
            if bars is None: continue
            recent = bars.filter(pl.col("ts") <= anchor_utc).tail(1)
            if recent.height > 0:
                prices_now[p.stock] = float(recent["close"][0])
        eq = book.equity(prices_now)
        equity_curve.append({
            "ts_us": anchor_us,
            "date": d_anchor,
            "n_open": len(book.open),
            "cash": book.cash,
            "equity": eq,
            "return_pct": (eq / args.initial_capital - 1) * 100,
        })
        last_anchor_utc = anchor_utc

        if n_iters % 500 == 0:
            print(f"  iter {n_iters}: {d_anchor} {ist_t} | open={len(book.open)} eq=${eq:,.0f}", flush=True)

    # Force-close remaining
    for p in list(book.open):
        bars = _load_bars(args.feature_root, p.stock)
        if bars is None: continue
        last = bars.tail(1)
        if last.height > 0:
            book.close_position(p, last["ts"][0], float(last["close"][0]),
                                "forced", args.cost_bps_round_trip)

    # Write outputs
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_trades.parquet")
    equity_path = args.out_prefix.with_name(f"{args.out_prefix.name}_equity.parquet")
    pl.DataFrame([{
        "stock": p.stock,
        "entry_ts_us": int(p.entry_ts.timestamp() * 1e6),
        "exit_ts_us":  int(p.exit_ts.timestamp() * 1e6) if p.exit_ts else 0,
        "entry_price": p.entry_price, "exit_price": p.exit_price or 0.0,
        "shares": p.shares, "pnl_bps": p.pnl_bps,
        "exit_reason": p.exit_reason,
    } for p in book.closed]).write_parquet(trades_path)
    pl.DataFrame(equity_curve).write_parquet(equity_path)

    # Summary
    eq_arr = np.array([r["equity"] for r in equity_curve]) if equity_curve else np.array([args.initial_capital])
    n = len(eq_arr)
    if n > 1:
        rets = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = float(rets.mean() / (rets.std() or 1e-9) * np.sqrt(252 * 75))  # 75 anchors/day
    else:
        sharpe = 0.0
    total_ret = (eq_arr[-1] / args.initial_capital - 1) * 100
    rmax = np.maximum.accumulate(eq_arr)
    mdd = float(((eq_arr / rmax) - 1).min() * 100)
    n_trades = len(book.closed)
    pnls = np.array([p.pnl_bps for p in book.closed]) if n_trades > 0 else np.array([0.0])
    stops = sum(1 for p in book.closed if p.exit_reason == "stop")
    targets = sum(1 for p in book.closed if p.exit_reason == "target")
    hit = float((pnls > 0).sum() / max(n_trades, 1) * 100)
    avg_bps = float(pnls.mean())
    summary = {
        "n_anchors": n, "n_trades": n_trades, "stops": stops, "targets": targets,
        "hit_pct": hit, "avg_pnl_per_trade_bps": avg_bps,
        "total_return_pct": total_ret,
        "sharpe_anchor_freq": sharpe,
        "max_drawdown_pct": mdd,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or args.out_prefix.name,
                   config={**vars(args)})
        for i, r in enumerate(equity_curve):
            wandb.log({"anchor_idx": i, "equity": r["equity"], "return_pct": r["return_pct"],
                       "n_open": r["n_open"], "cash": r["cash"]}, step=i)
        for k, v in summary.items():
            wandb.summary[k] = v
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
