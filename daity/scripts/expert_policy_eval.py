"""Rule-based expert policy on top of multi-head cohort predictions.

Each trading day at the configured anchor:
  1. Score each stock by combining horizons + consistency.
  2. Decide exits, holds, new entries with portfolio + sector constraints.
  3. Execute (mark new positions at anchor close; exit at hold-period end).
  4. Aggregate daily P&L.

This is the deterministic baseline that the LLM trader needs to beat.

Inputs: a predictions.parquet from `online_walk_forward.py` (with columns
date, anchor_us, stock, horizon, pred_lr, real_lr).

Outputs: per-day P&L parquet + summary stats.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class Position:
    stock: str
    sector_id: int | None
    entry_date: date
    entry_anchor_us: int
    size_pct: float                    # of total book (0..1)
    horizon_chosen: str                # which horizon's prediction we keyed off
    hold_days_remaining: int

    def age_days(self, today: date) -> int:
        return (today - self.entry_date).days


@dataclass
class Portfolio:
    cash_pct: float = 1.0
    positions: dict[str, Position] = field(default_factory=dict)

    def used_pct(self) -> float:
        return sum(p.size_pct for p in self.positions.values())

    def sector_exposure(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for p in self.positions.values():
            if p.sector_id is None:
                continue
            out[p.sector_id] = out.get(p.sector_id, 0.0) + p.size_pct
        return out


def score_candidates(
    day_df: pl.DataFrame,
    horizons: list[str],
    primary_weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """For each stock at this anchor: pivot horizon predictions, compute:
        score = weighted mean(pred across horizons)
        consistency = fraction of horizons with sign matching score
        max_horizon_name / max_horizon_pred
    Returns one row per stock with the scored columns.
    """
    w = primary_weights or {h: 1.0 for h in horizons}
    pivoted = (
        day_df.filter(pl.col("horizon").is_in(horizons))
        .pivot(values="pred_lr", index="stock", on="horizon", aggregate_function="first")
    )
    # Compute weighted mean.
    cols_in = [c for c in pivoted.columns if c != "stock" and c in horizons]
    if not cols_in:
        return pivoted.with_columns([
            pl.lit(0.0).alias("score"),
            pl.lit(0.0).alias("consistency"),
        ])
    wsum = sum(w.get(c, 1.0) for c in cols_in) or 1.0
    score_expr = sum((pl.col(c) * w.get(c, 1.0)) for c in cols_in) / wsum
    pivoted = pivoted.with_columns(score_expr.alias("score"))
    # Consistency: fraction of horizons with same sign as score.
    cons_expr = sum(
        (pl.col(c).sign() == pl.col("score").sign()).cast(pl.Float32)
        for c in cols_in
    ) / float(len(cols_in))
    pivoted = pivoted.with_columns(cons_expr.alias("consistency"))
    return pivoted


def get_realized_return(
    day_df: pl.DataFrame,
    stock: str,
    horizon: str,
) -> float | None:
    """Realized log return for this stock+horizon (from the same day_df).
    Returns None if not present."""
    sub = day_df.filter(
        (pl.col("stock") == stock) & (pl.col("horizon") == horizon),
    )
    if sub.shape[0] == 0:
        return None
    return float(sub["real_lr"][0])


@dataclass
class ExpertConfig:
    horizons: tuple[str, ...] = (
        "intraday_60m", "intraday_120m", "to_close", "next_day_eod",
    )
    # Primary horizon for hold-period / scoring weight.
    primary_horizon: str = "next_day_eod"
    horizon_weights: dict[str, float] = field(default_factory=lambda: {
        "intraday_60m":  0.5, "intraday_120m": 0.7,
        "to_close":      1.0, "next_day_eod":  1.5,
    })
    # Stock-pick filters
    top_k_pick: int = 5                     # number of new entries to evaluate
    score_threshold_bps: float = 30.0       # require score > this bps to enter
    consistency_threshold: float = 0.75     # at least 75% horizons aligned
    # Portfolio constraints
    max_positions: int = 5
    max_size_per_pos: float = 0.20
    max_sector_exposure: float = 0.40
    # Hold-period & exit
    default_hold_days: int = 1              # hold N trading days
    cost_bps_per_leg: float = 2.5           # per-leg cost = 2.5bps each side = 5bps round-trip


def run_expert(
    preds: pl.DataFrame,
    cfg: ExpertConfig,
    test_start: date,
    test_end: date,
    sector_by_stock: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Run the expert through trading days. Output per-day P&L rows.

    Mechanics:
      For each trading day D at the anchor:
        1. Decrement hold_days_remaining for held positions; exit those <= 0.
        2. Compute realized P&L for exited positions.
        3. Score new candidates; rank by score × consistency.
        4. Pick top-N respecting position cap + sector cap.
        5. Enter at this anchor (size = max_size_per_pos).
        6. Carry positions to next day.
    """
    # Build day index.
    all_dates = sorted(
        preds.filter(
            (pl.col("date") >= test_start) & (pl.col("date") <= test_end),
        )["date"].unique().to_list(),
    )

    pf = Portfolio()
    daily_rows: list[dict] = []
    cost_per_leg = cfg.cost_bps_per_leg

    for d in all_dates:
        day_df = preds.filter(pl.col("date") == d)

        # 1) Process exits — anything whose hold_days_remaining drops to 0 today.
        exits = []
        for sym, pos in list(pf.positions.items()):
            pos.hold_days_remaining -= 1
            if pos.hold_days_remaining <= 0:
                # Realized return: use the horizon we entered with.
                # Approximate: the realized for that horizon is the actual outcome
                # of holding from entry anchor to anchor+horizon_period.
                # For day_plus_1 strategies we approximate the realized at exit as
                # the realized that was associated with the entry anchor's
                # next_day_eod label.
                # Since we already have realized data in `pred_rows`, we use that.
                # NOTE: this approximates "hold from entry anchor to today" with
                # the realized label that the model was trained on.
                # Get the entry day's preds for this stock+horizon
                entry_df = preds.filter(
                    (pl.col("date") == pos.entry_date)
                    & (pl.col("stock") == sym)
                    & (pl.col("horizon") == pos.horizon_chosen),
                )
                if entry_df.shape[0] > 0:
                    real_lr = float(entry_df["real_lr"][0])
                    net_bps = (
                        (np.exp(real_lr) - 1.0) * 10000.0
                        - 2 * cost_per_leg
                    )
                    exits.append((sym, pos, net_bps))
            # Else: still held
        for sym, pos, _ in exits:
            del pf.positions[sym]

        # 2) Score candidates
        scored = score_candidates(day_df, list(cfg.horizons), cfg.horizon_weights)
        if scored.shape[0] == 0:
            daily_rows.append({"date": d, "n_exits": len(exits),
                               "exits_bps": float(sum(e[2] for e in exits)),
                               "n_entries": 0, "entries_cost": 0.0,
                               "n_held": len(pf.positions), "exposure": pf.used_pct()})
            continue
        ranked = scored.filter(
            (pl.col("score").abs() * 10000.0 > cfg.score_threshold_bps)
            & (pl.col("consistency") >= cfg.consistency_threshold),
        ).sort("score", descending=True)

        # 3) Apply portfolio constraints & enter
        slots_left = cfg.max_positions - len(pf.positions)
        sec_exp = pf.sector_exposure()
        entries: list[str] = []
        for row in ranked.iter_rows(named=True):
            if slots_left <= 0:
                break
            sym = row["stock"]
            if sym in pf.positions:
                continue  # already held; don't double down
            sec = sector_by_stock.get(sym) if sector_by_stock else None
            cur_sec_exp = sec_exp.get(sec, 0.0) if sec is not None else 0.0
            if cur_sec_exp + cfg.max_size_per_pos > cfg.max_sector_exposure:
                continue
            if pf.used_pct() + cfg.max_size_per_pos > 1.0:
                continue
            new_pos = Position(
                stock=sym, sector_id=sec, entry_date=d,
                entry_anchor_us=int(day_df["anchor_us"][0]),
                size_pct=cfg.max_size_per_pos,
                horizon_chosen=cfg.primary_horizon,
                hold_days_remaining=cfg.default_hold_days,
            )
            pf.positions[sym] = new_pos
            entries.append(sym)
            slots_left -= 1
            if sec is not None:
                sec_exp[sec] = sec_exp.get(sec, 0.0) + new_pos.size_pct

        # Aggregate daily numbers
        # Net P&L for the day = exits' realized bps × position_size (already 20%)
        # Note: we treat each exit's realized as the per-leg bps; weight by size_pct
        # and book size assumption of 100% capital.
        day_pnl_bps = 0.0
        for sym, pos, net_bps in exits:
            day_pnl_bps += net_bps * pos.size_pct
        # Entry costs are paid TODAY for new entries (per-leg).
        entry_cost = cfg.max_size_per_pos * len(entries) * cost_per_leg
        day_pnl_bps -= entry_cost
        daily_rows.append({
            "date": d,
            "n_exits": len(exits),
            "exits_bps": float(sum(e[2] for e in exits)),
            "n_entries": len(entries),
            "entries_cost": float(entry_cost),
            "n_held": len(pf.positions),
            "exposure": pf.used_pct(),
            "net_bps": float(day_pnl_bps),
        })

    return pl.DataFrame(daily_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end", type=str, required=True)
    ap.add_argument("--primary-horizon", type=str, default="next_day_eod")
    ap.add_argument("--horizons", type=str, default="",
                    help="Comma-separated horizon names to score on. "
                         "Empty = use ExpertConfig default.")
    ap.add_argument("--horizon-weights", type=str, default="",
                    help="Comma-separated weights matching --horizons. "
                         "Empty = uniform weight per horizon.")
    ap.add_argument("--top-k-pick", type=int, default=5)
    ap.add_argument("--score-threshold-bps", type=float, default=30.0)
    ap.add_argument("--consistency-threshold", type=float, default=0.75)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--max-size-per-pos", type=float, default=0.20)
    ap.add_argument("--max-sector-exposure", type=float, default=0.40)
    ap.add_argument("--default-hold-days", type=int, default=1)
    ap.add_argument("--cost-bps-per-leg", type=float, default=2.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    args = ap.parse_args()

    # Load sector mapping
    from daity.data.symbols import SymbolMaster
    master = SymbolMaster.from_cache(args.cache_root)
    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in master.frame["symbol"].to_list()
    })
    sec_to_id = {s: i for i, s in enumerate(all_sectors)}
    sec_by_stock = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in master.frame["symbol"].to_list()
    }

    preds = pl.read_parquet(args.predictions_path)
    print(f"loaded {preds.shape[0]} prediction rows", flush=True)

    cfg = ExpertConfig(
        primary_horizon=args.primary_horizon,
        top_k_pick=args.top_k_pick,
        score_threshold_bps=args.score_threshold_bps,
        consistency_threshold=args.consistency_threshold,
        max_positions=args.max_positions,
        max_size_per_pos=args.max_size_per_pos,
        max_sector_exposure=args.max_sector_exposure,
        default_hold_days=args.default_hold_days,
        cost_bps_per_leg=args.cost_bps_per_leg,
    )
    daily = run_expert(
        preds, cfg,
        date.fromisoformat(args.test_start),
        date.fromisoformat(args.test_end),
        sec_by_stock,
    )
    daily.write_parquet(args.out)

    # Summary
    nb = daily["net_bps"].drop_nulls()
    n = nb.shape[0]
    sd = nb.std() or 1e-9
    print(f"\ndays: {n}")
    print(f"mean bps/day: {nb.mean():+.2f}")
    print(f"median bps:   {nb.median():+.2f}")
    print(f"hit rate:     {(nb > 0).sum() / n * 100:.1f}%")
    print(f"sharpe:       {nb.mean() / sd * np.sqrt(252):+.2f}")
    print(f"total bps:    {nb.sum():+.1f}")
    print(f"avg n_held:   {daily['n_held'].mean():.2f}")
    print(f"avg exposure: {daily['exposure'].mean():.2%}")
    print(f"avg entries/day: {daily['n_entries'].mean():.2f}")
    print(f"avg exits/day:   {daily['n_exits'].mean():.2f}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
