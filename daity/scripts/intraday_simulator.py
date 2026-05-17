"""Intraday position-management backtest with cash + stop-loss + target-expire.

Consumes a strategy parquet (with `chosen_stocks` column) + 5m bars per stock.
Simulates a real position book: max K concurrent positions, equal $ per
position, cash constraint, mark-to-market on 5m bars.

For each test day D, the strategy's chosen_stocks for D are processed at
the configured anchor time. Open positions are checked every 5m bar for
stop-loss; they exit at the target endpoint (e.g. next-day 15:30) otherwise.

Output:
  - trades parquet: per-trade entry/exit details
  - equity parquet: per-day equity, cash, # open positions
  - summary scalars to stdout + (optional) wandb
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


HORIZON_EXIT = {
    "to_close":      ("15:30", 0),
    "overnight":     ("09:15", 1),
    "next_day_1h":   ("10:15", 1),
    "next_day_eod":  ("15:30", 1),
    "day_plus_3":    ("15:30", 3),
    "day_plus_5":    ("15:30", 5),
}


def _ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


@dataclass
class Position:
    stock: str
    entry_ts: datetime
    entry_price: float
    shares: float
    stop_price: float
    target_ts: datetime
    # filled on exit
    exit_ts: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""

    @property
    def cost_value(self) -> float:
        return self.entry_price * self.shares

    def value_at(self, price: float) -> float:
        return price * self.shares

    @property
    def realized_pnl_bps(self) -> float:
        assert self.exit_price is not None
        return (self.exit_price - self.entry_price) / self.entry_price * 10000.0

    def __repr__(self) -> str:
        return f"<Pos {self.stock} {self.entry_ts.date()}->{self.target_ts.date()} entry={self.entry_price:.2f}>"


@dataclass
class Book:
    """Tracks cash and open positions."""
    initial_capital: float
    cash: float = 0.0
    open_positions: list[Position] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    equity_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    def n_open(self) -> int:
        return len(self.open_positions)

    def is_held(self, stock: str) -> bool:
        return any(p.stock == stock for p in self.open_positions)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Mark-to-market equity given per-stock prices."""
        equity = self.cash
        for p in self.open_positions:
            px = prices.get(p.stock, p.entry_price)
            equity += p.value_at(px)
        return equity

    def open_position(self, stock: str, ts: datetime, price: float,
                      position_dollars: float, stop_pct: float,
                      target_ts: datetime, cost_bps: float) -> Position | None:
        # Apply half of round-trip cost as a slippage to entry price
        entry_eff = price * (1.0 + cost_bps / 2.0 / 10000.0)
        shares = position_dollars / entry_eff
        if shares <= 0:
            return None
        pos = Position(
            stock=stock, entry_ts=ts,
            entry_price=entry_eff,
            shares=shares,
            stop_price=price * stop_pct,  # stop on RAW price, not slippage-adjusted
            target_ts=target_ts,
        )
        self.cash -= shares * entry_eff
        self.open_positions.append(pos)
        return pos

    def close_position(self, pos: Position, ts: datetime, raw_exit_price: float,
                       reason: str, cost_bps: float) -> None:
        exit_eff = raw_exit_price * (1.0 - cost_bps / 2.0 / 10000.0)
        pos.exit_ts = ts
        pos.exit_price = exit_eff
        pos.exit_reason = reason
        self.cash += pos.shares * exit_eff
        self.open_positions.remove(pos)
        self.closed_positions.append(pos)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-parquet", type=Path, required=True,
                    help="Per-day P&L parquet with chosen_stocks column")
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--cache-root",   type=Path, default=Path("data/cache"))
    ap.add_argument("--anchor-ist", type=str, default="10:15")
    ap.add_argument("--horizon", type=str, default="next_day_eod")
    ap.add_argument("--initial-capital", type=float, default=1_000_000.0)
    ap.add_argument("--max-k", type=int, default=5,
                    help="Max concurrent positions")
    ap.add_argument("--position-frac", type=float, default=0.2,
                    help="Fraction of initial capital per position")
    ap.add_argument("--stop-bps", type=float, default=400.0)
    ap.add_argument("--cost-bps-round-trip", type=float, default=15.0,
                    help="Total round-trip cost; split half on entry, half on exit")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    return ap.parse_args()


def _next_trading_day(cal, d: date, n_forward: int) -> date | None:
    if n_forward == 0:
        return d
    cur = d
    for _ in range(n_forward):
        cur = cur + timedelta(days=1)
        while not cal.is_trading_day(cur):
            cur = cur + timedelta(days=1)
            if (cur - d).days > 30:
                return None
    return cur


# 5m bar cache to avoid reloading per anchor
_bars_cache: dict[str, pl.DataFrame] = {}

def _load_bars(feature_root: Path, sym: str) -> pl.DataFrame | None:
    if sym not in _bars_cache:
        path = feature_root / "5m" / f"{sym}.parquet"
        if not path.exists():
            _bars_cache[sym] = pl.DataFrame()  # empty marker
            return None
        try:
            _bars_cache[sym] = pl.read_parquet(path).sort("ts")
        except Exception:
            _bars_cache[sym] = pl.DataFrame()
            return None
    df = _bars_cache[sym]
    return df if df.height > 0 else None


def _bars_in_window(df: pl.DataFrame, ts_lo: datetime, ts_hi: datetime) -> pl.DataFrame:
    return df.filter((pl.col("ts") >= ts_lo) & (pl.col("ts") <= ts_hi))


def simulate(args: argparse.Namespace) -> dict:
    from daity.data.calendar_nse import NSECalendar
    cal = NSECalendar.from_cache(args.cache_root)

    df = pl.read_parquet(args.strategy_parquet).sort("test_date")
    if "chosen_stocks" not in df.columns:
        print("ERROR: strategy parquet missing chosen_stocks. Re-run strategy.", file=sys.stderr)
        return {}

    h, m = (int(x) for x in args.anchor_ist.split(":"))
    anchor_t = dtime(h, m)
    if args.horizon not in HORIZON_EXIT:
        print(f"ERROR: unknown horizon {args.horizon}", file=sys.stderr)
        return {}
    exit_t_str, n_fwd = HORIZON_EXIT[args.horizon]
    stop_pct = 1.0 - args.stop_bps / 10000.0

    book = Book(initial_capital=args.initial_capital)
    pos_dollars = args.initial_capital * args.position_frac

    # Process days in order. For each day, at the anchor:
    #   1. Walk 5m bars from prior-anchor → this-anchor to check stop/exit on open positions
    #   2. Open new positions for this day's picks (if cash + slot available)
    #   3. Mark-to-market at anchor's close
    prev_anchor_utc = None
    trades = []
    equity_rows = []

    for r in df.iter_rows(named=True):
        d = r["test_date"]
        anchor_utc = _ist_to_utc(d, anchor_t)
        # Step 1: process open positions through bars in (prev_anchor, anchor]
        if prev_anchor_utc is not None:
            for p in list(book.open_positions):
                bars = _load_bars(args.feature_root, p.stock)
                if bars is None: continue
                window = _bars_in_window(bars, prev_anchor_utc, anchor_utc)
                if window.height == 0: continue
                # Walk forward; check stop first then target
                for row in window.iter_rows(named=True):
                    ts = row["ts"]
                    if ts >= p.target_ts:
                        # Exit at this bar's close at target
                        book.close_position(p, ts, float(row["close"]),
                                            "target", args.cost_bps_round_trip)
                        break
                    if float(row["low"]) <= p.stop_price:
                        book.close_position(p, ts, p.stop_price,
                                            "stop", args.cost_bps_round_trip)
                        break

        # Step 2: open new positions from this day's picks
        syms = (r.get("chosen_stocks") or "").split(",")
        syms = [s for s in syms if s]
        if r["n_today_trades"] > 0 and syms:
            target_d = _next_trading_day(cal, d, n_fwd)
            if target_d is None:
                target_utc = None
            else:
                eh, em = (int(x) for x in exit_t_str.split(":"))
                target_utc = _ist_to_utc(target_d, dtime(eh, em))
            for sym in syms:
                if book.n_open() >= args.max_k: break
                if book.is_held(sym): continue
                if book.cash < pos_dollars: continue
                if target_utc is None: continue
                bars = _load_bars(args.feature_root, sym)
                if bars is None: continue
                # Find the first bar at or after anchor
                entry_bar = bars.filter(pl.col("ts") >= anchor_utc).head(1)
                if entry_bar.height == 0: continue
                entry_px = float(entry_bar["open"][0])
                if entry_px <= 0: continue
                book.open_position(sym, anchor_utc, entry_px, pos_dollars,
                                   stop_pct, target_utc, args.cost_bps_round_trip)

        # Step 3: mark-to-market at anchor close
        prices_now = {}
        for p in book.open_positions:
            bars = _load_bars(args.feature_root, p.stock)
            if bars is None: continue
            entry_bar = bars.filter(pl.col("ts") <= anchor_utc).tail(1)
            if entry_bar.height > 0:
                prices_now[p.stock] = float(entry_bar["close"][0])
        equity_now = book.mark_to_market(prices_now)
        equity_rows.append({
            "date": d,
            "anchor_utc": int(anchor_utc.timestamp() * 1_000_000),
            "n_open": book.n_open(),
            "cash": book.cash,
            "equity": equity_now,
            "return_pct_from_initial": (equity_now / args.initial_capital - 1.0) * 100,
        })
        prev_anchor_utc = anchor_utc

    # Forced close at end: liquidate remaining open positions at their last bar before final anchor
    for p in list(book.open_positions):
        bars = _load_bars(args.feature_root, p.stock)
        if bars is None: continue
        last = bars.filter(pl.col("ts") >= p.entry_ts).tail(1)
        if last.height > 0:
            book.close_position(p, last["ts"][0], float(last["close"][0]),
                                "forced", args.cost_bps_round_trip)

    # Write outputs
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_trades.parquet")
    equity_path = args.out_prefix.with_name(f"{args.out_prefix.name}_equity.parquet")

    trade_rows = []
    for p in book.closed_positions:
        trade_rows.append({
            "stock": p.stock,
            "entry_ts_us": int(p.entry_ts.timestamp() * 1_000_000),
            "exit_ts_us":  int(p.exit_ts.timestamp() * 1_000_000) if p.exit_ts else 0,
            "entry_price": p.entry_price,
            "exit_price":  p.exit_price or 0.0,
            "shares":      p.shares,
            "pnl_bps":     p.realized_pnl_bps if p.exit_price else 0.0,
            "exit_reason": p.exit_reason,
            "hold_seconds": (p.exit_ts - p.entry_ts).total_seconds() if p.exit_ts else 0,
        })
    pl.DataFrame(trade_rows).write_parquet(trades_path)
    pl.DataFrame(equity_rows).write_parquet(equity_path)

    # Summary
    equity = np.array([r["equity"] for r in equity_rows])
    n = len(equity)
    daily_rets = np.diff(equity) / equity[:-1]
    mean_daily = float(daily_rets.mean()) if len(daily_rets) > 0 else 0.0
    sd_daily = float(daily_rets.std()) or 1e-9
    sharpe = mean_daily / sd_daily * np.sqrt(252)
    total_return_pct = (equity[-1] / args.initial_capital - 1.0) * 100
    n_trades = len(book.closed_positions)
    if n_trades > 0:
        pnls = np.array([p.realized_pnl_bps for p in book.closed_positions])
        hit = float((pnls > 0).sum() / n_trades * 100)
        avg_pnl = float(pnls.mean())
        stops = sum(1 for p in book.closed_positions if p.exit_reason == "stop")
        targets = sum(1 for p in book.closed_positions if p.exit_reason == "target")
    else:
        hit = avg_pnl = stops = targets = 0
    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity / running_max - 1.0) * 100
    max_dd = float(drawdowns.min())

    summary = {
        "test_days": n,
        "n_trades": n_trades,
        "stops": stops, "targets": targets,
        "hit_pct": hit,
        "avg_pnl_per_trade_bps": avg_pnl,
        "total_return_pct": total_return_pct,
        "sharpe_daily": sharpe,
        "max_drawdown_pct": max_dd,
        "annualized_return_pct": total_return_pct / (n / 252.0) if n > 0 else 0.0,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v:+10.3f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    print(f"  trades → {trades_path}")
    print(f"  equity → {equity_path}")

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or args.out_prefix.name,
                   config={**vars(args), "stop_pct": stop_pct})
        for i, r in enumerate(equity_rows):
            wandb.log({
                "test_day_index": i,
                "equity": r["equity"],
                "return_pct": r["return_pct_from_initial"],
                "n_open": r["n_open"],
                "cash": r["cash"],
            }, step=i)
        for k, v in summary.items():
            wandb.summary[k] = v
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        eq_pct = [r["return_pct_from_initial"] for r in equity_rows]
        a1.plot(range(n), eq_pct, lw=2, color="C0")
        a1.fill_between(range(n), 0, eq_pct, where=[v > 0 for v in eq_pct], color="C0", alpha=0.2)
        a1.fill_between(range(n), 0, eq_pct, where=[v < 0 for v in eq_pct], color="red", alpha=0.15)
        a1.axhline(0, color="black", lw=0.5); a1.grid(alpha=0.3)
        a1.set_ylabel("Equity return from initial (%)")
        a1.set_title(f"{args.out_prefix.name}: total {total_return_pct:+.2f}%  Sharpe={sharpe:+.2f}  trades={n_trades}  hit={hit:.0f}%  MDD={max_dd:.2f}%")
        a2.plot(range(n), [r["n_open"] for r in equity_rows], lw=2, color="C2")
        a2.set_ylabel("# open positions"); a2.set_xlabel("Test day index"); a2.grid(alpha=0.3)
        plt.tight_layout()
        wandb.log({"equity_chart": wandb.Image(fig)})
        plt.close(fig)
        wandb.finish()
    return summary


def main() -> int:
    args = parse_args()
    summary = simulate(args)
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
