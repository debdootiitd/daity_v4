"""Daily adaptive strategy: grid-search horizon weights + buy/sell thresholds
on the last `lookback_days`; apply winning config to the next day.

For each test day D:
  1. Tuning window = D-lookback .. D-1 (realized labels available).
  2. Sweep over (horizon weight subset, K, buy_threshold_bps, hold_days).
  3. Pick the (weights, K, threshold) that maximize lookback Sharpe.
  4. On day D: compute score(stock) = sum(weight[h] × pred[h]) / sum(weights).
     BUY rule: enter LONG top-K stocks where score > buy_threshold AND
               stock not already held.
     SELL rule: exit any held position where current-day score < sell_threshold
                OR hold_days_remaining ≤ 0.
  5. Record realized P&L for exits + new entries.

Long-only, no shorting. Each position equal-weighted at 1/K of book.

Output: daily P&L parquet + summary.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


# ----------------- Strategy definition -----------------

@dataclass
class StratConfig:
    weights: tuple[tuple[str, float], ...]   # ((horizon, weight), ...)
    top_k: int
    buy_threshold_bps: float
    sell_threshold_bps: float
    hold_days: int

    def key(self) -> str:
        hkey = "+".join(f"{h}*{w:.1f}" for h, w in self.weights)
        return (f"K{self.top_k}_buy{self.buy_threshold_bps:.0f}"
                f"_sell{self.sell_threshold_bps:.0f}_hold{self.hold_days}_{hkey}")


def score_stocks(day_df: pl.DataFrame, weights: tuple[tuple[str, float], ...]) -> pl.DataFrame:
    """Return per-stock score = weighted mean of horizon preds (log-return units).

    Defensive: groupby-stock aggregation avoids brittle pivot when some
    horizons are missing for some stocks. Computes the weighted sum
    arithmetically as `sum(pred[h] × w[h]) / sum(w_present)`.
    """
    hns_req = {h: w for h, w in weights if w != 0}
    if not hns_req:
        return pl.DataFrame({"stock": [], "score": []})
    sub = day_df.filter(pl.col("horizon").is_in(list(hns_req.keys())))
    if sub.shape[0] == 0:
        return pl.DataFrame({"stock": [], "score": []})
    # weighted_pred per row = pred_lr × w[horizon]
    df = sub.with_columns(
        pl.col("horizon").replace_strict(hns_req, default=0.0).alias("__w"),
    )
    df = df.with_columns((pl.col("pred_lr") * pl.col("__w")).alias("__wp"))
    agg = df.group_by("stock").agg([
        pl.col("__wp").sum().alias("wp_sum"),
        pl.col("__w").sum().alias("w_sum"),
    ])
    return agg.with_columns(
        (pl.col("wp_sum") / pl.when(pl.col("w_sum") > 0)
                              .then(pl.col("w_sum"))
                              .otherwise(1.0)).alias("score"),
    ).select(["stock", "score"])


@dataclass
class Holding:
    stock: str
    entry_date: date
    entry_pred_lr: float
    horizon_chosen: str
    hold_days_remaining: int


def simulate_window(
    preds: pl.DataFrame,
    cfg: StratConfig,
    test_dates: list[date],
    cost_bps: float,
    realized_horizon: str,
) -> tuple[list[float], int, int]:
    """Simulate the strategy over `test_dates`. Returns (per_day_bps, n_entries, n_exits).

    For lookback evaluation we use the SAME realized_horizon for exits as on day D.
    """
    holdings: dict[str, Holding] = {}
    daily_bps: list[float] = []
    cost_per_leg = cost_bps / 2.0
    total_entries = 0
    total_exits = 0

    for d in test_dates:
        day_df = preds.filter(pl.col("date") == d)
        if day_df.shape[0] == 0:
            daily_bps.append(0.0)
            continue
        scored = score_stocks(day_df, cfg.weights)
        scored_map: dict[str, float] = (
            {row["stock"]: row["score"] for row in scored.to_dicts()}
            if scored.shape[0] > 0 else {}
        )

        # 1) Exit rule — hold_days expired OR score < sell_threshold
        exits: list[Holding] = []
        for sym in list(holdings.keys()):
            h = holdings[sym]
            h.hold_days_remaining -= 1
            cur_score_bps = scored_map.get(sym, 0.0) * 10000.0
            if (h.hold_days_remaining <= 0
                    or cur_score_bps < cfg.sell_threshold_bps):
                # Realize using entry day's preds for the chosen horizon.
                ent_df = preds.filter(
                    (pl.col("date") == h.entry_date)
                    & (pl.col("stock") == sym)
                    & (pl.col("horizon") == h.horizon_chosen),
                )
                if ent_df.shape[0] > 0:
                    real_lr = float(ent_df["real_lr"][0])
                    bps_size = ((np.exp(real_lr) - 1.0) * 10000.0
                                - 2 * cost_per_leg)
                    exits.append((sym, h, bps_size))
                else:
                    exits.append((sym, h, 0.0))
                del holdings[sym]
        total_exits += len(exits)

        # 2) Entry rule — top-K above buy_threshold
        new_entries: list[str] = []
        if scored.shape[0] > 0:
            cands = scored.filter(
                pl.col("score") * 10000.0 > cfg.buy_threshold_bps,
            ).sort("score", descending=True)
            slots = cfg.top_k - len(holdings)
            for row in cands.iter_rows(named=True):
                if slots <= 0: break
                sym = row["stock"]
                if sym in holdings: continue
                holdings[sym] = Holding(
                    stock=sym, entry_date=d, entry_pred_lr=float(row["score"]),
                    horizon_chosen=realized_horizon,
                    hold_days_remaining=cfg.hold_days,
                )
                new_entries.append(sym)
                slots -= 1
        total_entries += len(new_entries)

        # 3) Aggregate P&L for the day
        size_pct = 1.0 / cfg.top_k
        day_bps = 0.0
        for sym, h, bps in exits:
            day_bps += bps * size_pct
        day_bps -= len(new_entries) * size_pct * cost_per_leg
        daily_bps.append(day_bps)

    return daily_bps, total_entries, total_exits


def sharpe_of(bps: list[float]) -> float:
    arr = np.asarray(bps, dtype=float)
    if arr.size < 3:
        return -1e9
    sd = arr.std() or 1e-9
    return arr.mean() / sd * np.sqrt(252)


# ----------------- Tuning loop -----------------

def build_configs(args) -> list[StratConfig]:
    """Build the grid of configs to evaluate on the lookback."""
    horizons_list = [h.strip() for h in args.horizons.split(",") if h.strip()]
    weight_choices = [float(w) for w in args.weight_choices.split(",")]
    k_choices       = [int(k)   for k in args.k_choices.split(",")]
    buy_thr_choices = [float(t) for t in args.buy_thresholds.split(",")]
    sell_thr_choices = [float(t) for t in args.sell_thresholds.split(",")]
    hold_choices   = [int(h)   for h in args.hold_choices.split(",")]

    # Each horizon gets a weight from weight_choices (independent).
    # Skip the all-zero combo.
    configs: list[StratConfig] = []
    weight_grids = list(itertools.product(weight_choices, repeat=len(horizons_list)))
    for ws in weight_grids:
        if sum(ws) == 0: continue
        weights = tuple(zip(horizons_list, ws))
        for k, bt, st, hd in itertools.product(
            k_choices, buy_thr_choices, sell_thr_choices, hold_choices,
        ):
            configs.append(StratConfig(
                weights=weights, top_k=k,
                buy_threshold_bps=bt, sell_threshold_bps=st,
                hold_days=hd,
            ))
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end",   type=str, required=True)
    ap.add_argument("--lookback-days", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--realized-horizon", type=str, default="next_day_eod",
                    help="Which horizon's realized return to use at exit time.")
    # Grid
    ap.add_argument("--horizons", type=str,
                    default="intraday_120m,to_close,next_day_eod,day_plus_3,day_plus_5")
    ap.add_argument("--weight-choices", type=str, default="0,1")
    ap.add_argument("--k-choices", type=str, default="5,10")
    ap.add_argument("--buy-thresholds", type=str, default="0,20,50")
    ap.add_argument("--sell-thresholds", type=str, default="-9999,-50,0")  # -9999 = no early exit
    ap.add_argument("--hold-choices", type=str, default="1,3")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    preds = pl.read_parquet(args.predictions_path)
    print(f"loaded {preds.shape[0]} pred rows; "
          f"date range {preds['date'].min()}..{preds['date'].max()}", flush=True)

    configs = build_configs(args)
    print(f"sweep {len(configs)} configs per day", flush=True)

    test_start = date.fromisoformat(args.test_start)
    test_end   = date.fromisoformat(args.test_end)
    all_dates = sorted(
        preds.filter(
            (pl.col("date") >= test_start) & (pl.col("date") <= test_end),
        )["date"].unique().to_list(),
    )
    # Need data BEFORE test_start for the first lookback.
    lookback_start = test_start - timedelta(days=args.lookback_days * 3)
    candidate_dates = sorted(
        preds.filter(
            (pl.col("date") >= lookback_start) & (pl.col("date") <= test_end),
        )["date"].unique().to_list(),
    )

    out_rows: list[dict] = []
    rolling: dict[str, Holding] = {}    # carries across test days
    cost_per_leg = args.cost_bps / 2.0
    t0 = time.time()
    for ti, d in enumerate(all_dates):
        prior = [pd for pd in candidate_dates if pd < d]
        if len(prior) < args.lookback_days:
            continue
        lookback = prior[-args.lookback_days:]

        # Tune (independent simulation per config on the lookback window).
        best_cfg = None
        best_sharpe = -1e9
        for cfg in configs:
            bps, _, _ = simulate_window(preds, cfg, lookback,
                                         args.cost_bps, args.realized_horizon)
            non_zero = [b for b in bps if abs(b) > 1e-9]
            if len(non_zero) < 3:
                continue
            sharpe = sharpe_of(non_zero)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_cfg = cfg

        if best_cfg is None:
            # Carry rolling state; record zero-P&L day.
            out_rows.append({"date": d, "tuned_K": 0, "tuned_buy_thr": 0,
                             "tuned_sell_thr": 0, "tuned_hold": 0,
                             "tuned_weights": "", "lookback_sharpe": 0.0,
                             "n_entries": 0, "n_exits": 0, "net_bps": 0.0})
            continue

        # Apply tuned cfg to ROLLING state on day d.
        day_df = preds.filter(pl.col("date") == d)
        scored = score_stocks(day_df, best_cfg.weights)
        scored_map: dict[str, float] = {row["stock"]: row["score"]
                                          for row in scored.to_dicts()}

        # 1) Exits
        exits_bps = 0.0
        n_exits_today = 0
        for sym in list(rolling.keys()):
            h = rolling[sym]
            h.hold_days_remaining -= 1
            cur_score_bps = scored_map.get(sym, 0.0) * 10000.0
            if (h.hold_days_remaining <= 0
                    or cur_score_bps < best_cfg.sell_threshold_bps):
                ent_df = preds.filter(
                    (pl.col("date") == h.entry_date)
                    & (pl.col("stock") == sym)
                    & (pl.col("horizon") == h.horizon_chosen),
                )
                if ent_df.shape[0] > 0:
                    real_lr = float(ent_df["real_lr"][0])
                    bps_pos = ((np.exp(real_lr) - 1.0) * 10000.0
                               - 2 * cost_per_leg)
                    exits_bps += bps_pos * (1.0 / best_cfg.top_k)
                del rolling[sym]
                n_exits_today += 1

        # 2) Entries
        n_entries_today = 0
        if scored.shape[0] > 0:
            cands = scored.filter(
                pl.col("score") * 10000.0 > best_cfg.buy_threshold_bps,
            ).sort("score", descending=True)
            slots = best_cfg.top_k - len(rolling)
            for row in cands.iter_rows(named=True):
                if slots <= 0: break
                sym = row["stock"]
                if sym in rolling: continue
                rolling[sym] = Holding(
                    stock=sym, entry_date=d,
                    entry_pred_lr=float(row["score"]),
                    horizon_chosen=args.realized_horizon,
                    hold_days_remaining=best_cfg.hold_days,
                )
                n_entries_today += 1
                slots -= 1

        entry_cost = (n_entries_today * (1.0 / best_cfg.top_k) * cost_per_leg)
        day_bps = exits_bps - entry_cost

        out_rows.append({
            "date": d,
            "tuned_K":       best_cfg.top_k,
            "tuned_buy_thr": best_cfg.buy_threshold_bps,
            "tuned_sell_thr": best_cfg.sell_threshold_bps,
            "tuned_hold":    best_cfg.hold_days,
            "tuned_weights": best_cfg.key(),
            "lookback_sharpe": best_sharpe,
            "n_entries": n_entries_today,
            "n_exits":   n_exits_today,
            "n_held":    len(rolling),
            "net_bps":   float(day_bps),
        })
        if (ti + 1) % 25 == 0:
            print(f"  processed {ti+1}/{len(all_dates)} days, "
                  f"rolling={len(rolling)}, elapsed {time.time()-t0:.0f}s",
                  flush=True)

    df = pl.DataFrame(out_rows)
    df.write_parquet(args.out)

    print(f"\nsaved -> {args.out}")
    print(f"days: {df.shape[0]}")
    if df.shape[0] == 0:
        return 2
    nb = df["net_bps"]
    sd = nb.std() or 1e-9
    print(f"mean bps/day: {nb.mean():+.2f}")
    print(f"hit rate:     {(nb > 0).sum() / df.shape[0] * 100:.1f}%")
    print(f"sharpe:       {nb.mean() / sd * np.sqrt(252):+.2f}")
    print(f"total bps:    {nb.sum():+.1f}")
    # Most-frequent tuned configs
    tc = df.group_by("tuned_weights").len().sort("len", descending=True)
    print(f"\nTop-5 most-picked weight combos:")
    for row in tc.head(5).to_dicts():
        print(f"  {row['len']:>3} days: {row['tuned_weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
