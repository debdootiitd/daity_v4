"""Adaptive calibrated strategy.

For each test day D:
  1. ASSEMBLE TRAINING DATA whose labels are realized by day D:
      - intraday_*  predictions made on D-1  → realized D-1 intraday
      - to_close   predictions made on D-1  → realized D-1 EOD
      - next_day_* predictions made on D-1  → realized D
      - day_plus_3 predictions made on D-3  → realized D
      - day_plus_5 predictions made on D-5  → realized D
     Each training row = (features = all horizon predictions at the anchor +
     universe regime features) + label (binary: real return > win_threshold_bps).
  2. TRAIN a calibration model (logistic regression) on the last N such days.
  3. APPLY the model to TODAY's predictions for the target horizon — produces a
     calibrated win probability per stock.
  4. SWEEP thresholds; compute lookback P&L using each threshold on the past
     N days.
  5. GATE: if any threshold yielded positive lookback P&L, apply it to D+1's
     predictions. Else, sit out.
  6. SWEEP over N ∈ {5, 10, 20, 40}.

Output: per-N daily P&L parquet + summary table.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# Horizon → number of TRADING days after the anchor when label realizes.
HORIZON_REALIZATION_TRADING_DAYS = {
    "intraday_30m": 0, "intraday_60m": 0, "intraday_120m": 0, "intraday_180m": 0,
    "to_close": 0, "overnight": 1, "next_day_1h": 1, "next_day_eod": 1,
    "day_plus_3": 3, "day_plus_5": 5,
}


def _to_date(x) -> date:
    if isinstance(x, date): return x
    return date.fromisoformat(str(x))


def build_wide_table(df: pl.DataFrame, horizons: list[str]) -> pl.DataFrame:
    """Pivot per (anchor_date, stock): wide row with all horizon preds + reals."""
    keep = df.filter(pl.col("horizon").is_in(horizons))
    p = keep.pivot(values="pred_lr", index=["date", "stock"], on="horizon",
                   aggregate_function="first")
    r = keep.pivot(values="real_lr", index=["date", "stock"], on="horizon",
                   aggregate_function="first")
    # Rename so we can join
    p = p.rename({h: f"pred_{h}" for h in horizons if h in p.columns})
    r = r.rename({h: f"real_{h}" for h in horizons if h in r.columns})
    return p.join(r, on=["date", "stock"], how="inner")


def add_regime_features(wide: pl.DataFrame, horizons: list[str]) -> pl.DataFrame:
    """Per anchor_date: mean and std of predictions across the universe.

    Adds univ_mean_<h>, univ_std_<h> as features (regime context).
    """
    aggs = []
    for h in horizons:
        col = f"pred_{h}"
        if col in wide.columns:
            aggs.append(pl.col(col).mean().alias(f"univ_mean_pred_{h}"))
            aggs.append(pl.col(col).std().alias(f"univ_std_pred_{h}"))
    if not aggs:
        return wide
    regime = wide.group_by("date").agg(aggs)
    return wide.join(regime, on="date", how="left")


@dataclass
class StratResult:
    test_date: date
    N: int
    chosen_threshold: float | None
    lookback_pnl_bps: float
    lookback_sharpe: float
    n_lookback_trades: int
    n_today_trades: int
    today_pnl_bps: float


def _make_clf(clf_type: str):
    if clf_type == "gbm":
        return GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                          learning_rate=0.1, random_state=0)
    return LogisticRegression(max_iter=500, C=1.0)


def run_one_N(
    wide: pl.DataFrame,
    test_dates: list[date],
    feature_cols: list[str],
    target_horizon: str,
    realization_offset: int,
    win_threshold_bps: float,
    N_lookback: int,
    thresholds: list[float],
    top_K: int,
    cost_bps: float,
    holdout_days: int = 3,
    clf_type: str = "lr",
) -> list[StratResult]:
    """Run the adaptive calibrated strategy for one value of N.

    Threshold gating uses a proper train/holdout split within the N-day lookback:
      - Train calibration model on the first (N - holdout_days) anchor dates.
      - Evaluate each threshold's P&L on the held-out last holdout_days dates.
      - Gate: only trade if any threshold yields positive mean held-out P&L.
      - Final model for D+1: retrain on all N days.
    This prevents in-sample threshold selection that inflates the trade rate.
    """
    label_col = f"real_{target_horizon}"
    pred_col  = f"pred_{target_horizon}"
    win_thr   = win_threshold_bps / 10000.0

    results: list[StratResult] = []
    valid = wide.filter(pl.col(label_col).is_not_null()
                        & pl.col(pred_col).is_not_null())

    for d in test_dates:
        anchor_cap_realization = d - timedelta(days=realization_offset)
        train = valid.filter(pl.col("date") < anchor_cap_realization)
        if train.shape[0] < 100:
            continue
        train_dates = sorted(train["date"].unique().to_list())
        if len(train_dates) < N_lookback + holdout_days:
            continue

        # Split: fit window = first N days; holdout window = last holdout_days days
        fit_dates  = train_dates[-(N_lookback + holdout_days):-holdout_days]
        hold_dates = train_dates[-holdout_days:]
        recent_dates = train_dates[-(N_lookback + holdout_days):]  # all N+h days

        fit_data  = train.filter(pl.col("date").is_in(fit_dates))
        hold_data = train.filter(pl.col("date").is_in(hold_dates))

        X_fit = fit_data.select(feature_cols).to_numpy()
        y_fit = (fit_data[label_col].to_numpy() > win_thr).astype(int)
        if y_fit.sum() < 3 or (1 - y_fit).sum() < 3:
            continue

        # Fit calibration model on fit window only
        scaler = StandardScaler()
        Xs_fit = scaler.fit_transform(np.nan_to_num(X_fit))
        try:
            clf_gate = _make_clf(clf_type)
            clf_gate.fit(Xs_fit, y_fit)
        except Exception:
            continue

        # Evaluate each threshold on held-out days (OOS gate)
        best_thr = None
        best_lookback_pnl = -1e9
        best_lookback_sharpe = -1e9
        best_n_trades = 0
        lookback_bps_log: dict[float, list[float]] = {t: [] for t in thresholds}

        for ad in hold_dates:
            ad_rows = hold_data.filter(pl.col("date") == ad)
            if ad_rows.shape[0] < top_K:
                continue
            X_ad = scaler.transform(np.nan_to_num(ad_rows.select(feature_cols).to_numpy()))
            prob = clf_gate.predict_proba(X_ad)[:, 1]
            real = ad_rows[label_col].to_numpy()
            for thr in thresholds:
                mask = prob > thr
                if mask.sum() < top_K:
                    lookback_bps_log[thr].append(0.0)
                    continue
                top_idx = np.argsort(-prob[mask])[:top_K]
                orig_idx = np.where(mask)[0][top_idx]
                chosen_real = real[orig_idx].mean()
                bps_val = (np.exp(chosen_real) - 1.0) * 10000.0 - cost_bps
                lookback_bps_log[thr].append(float(bps_val))

        for thr, log in lookback_bps_log.items():
            traded = [b for b in log if abs(b) > 1e-9]
            if not traded:
                continue
            mean_b = float(np.mean(traded))
            sd = float(np.std(traded)) or 1e-9
            sharpe_lb = mean_b / sd * np.sqrt(252 / max(1, realization_offset))
            if mean_b > best_lookback_pnl:
                best_lookback_pnl = mean_b
                best_lookback_sharpe = sharpe_lb
                best_thr = thr
                best_n_trades = len(traded)

        # GATE: only trade today if held-out P&L positive for some threshold
        if best_thr is None or best_lookback_pnl <= 0:
            results.append(StratResult(test_date=d, N=N_lookback,
                                       chosen_threshold=None,
                                       lookback_pnl_bps=best_lookback_pnl,
                                       lookback_sharpe=best_lookback_sharpe,
                                       n_lookback_trades=best_n_trades,
                                       n_today_trades=0,
                                       today_pnl_bps=0.0))
            continue

        # Retrain on ALL N+holdout days for the final D+1 model
        all_recent = train.filter(pl.col("date").is_in(recent_dates))
        X_all = all_recent.select(feature_cols).to_numpy()
        y_all = (all_recent[label_col].to_numpy() > win_thr).astype(int)
        scaler_final = StandardScaler()
        Xs_all = scaler_final.fit_transform(np.nan_to_num(X_all))
        try:
            clf_final = _make_clf(clf_type)
            clf_final.fit(Xs_all, y_all)
        except Exception:
            results.append(StratResult(d, N_lookback, best_thr, best_lookback_pnl,
                                       best_lookback_sharpe, best_n_trades, 0, 0.0))
            continue

        # APPLY to today (d): pick top-K with prob > best_thr
        today_rows = valid.filter(pl.col("date") == d)
        if today_rows.shape[0] < top_K:
            results.append(StratResult(d, N_lookback, best_thr, best_lookback_pnl,
                                       best_lookback_sharpe, best_n_trades, 0, 0.0))
            continue
        X_today = scaler_final.transform(np.nan_to_num(today_rows.select(feature_cols).to_numpy()))
        prob_today = clf_final.predict_proba(X_today)[:, 1]
        mask = prob_today > best_thr
        if mask.sum() < top_K:
            results.append(StratResult(d, N_lookback, best_thr, best_lookback_pnl,
                                       best_lookback_sharpe, best_n_trades, 0, 0.0))
            continue
        top_idx = np.argsort(-prob_today[mask])[:top_K]
        orig_idx = np.where(mask)[0][top_idx]
        chosen_real_today = today_rows[label_col].to_numpy()[orig_idx].mean()
        today_bps = (np.exp(chosen_real_today) - 1.0) * 10000.0 - cost_bps
        results.append(StratResult(d, N_lookback, best_thr, best_lookback_pnl,
                                   best_lookback_sharpe, best_n_trades,
                                   int(top_K), float(today_bps)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-path", type=Path, required=True)
    ap.add_argument("--test-start", type=str, required=True)
    ap.add_argument("--test-end",   type=str, required=True)
    ap.add_argument("--target-horizon", type=str, default="day_plus_5")
    ap.add_argument("--horizons", type=str,
                    default="intraday_60m,intraday_120m,to_close,next_day_eod,day_plus_3,day_plus_5")
    ap.add_argument("--win-threshold-bps", type=float, default=30.0)
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--n-sweep", type=str, default="5,10,20,40")
    ap.add_argument("--prob-thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--holdout-days", type=int, default=3,
                    help="Days held out from end of lookback for threshold evaluation (OOS gate)")
    ap.add_argument("--clf-type", type=str, default="lr", choices=["lr", "gbm"],
                    help="Calibration model: lr=LogisticRegression, gbm=GradientBoosting")
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    horizons = [h.strip() for h in args.horizons.split(",")]
    n_values = [int(n) for n in args.n_sweep.split(",")]
    thresholds = [float(t) for t in args.prob_thresholds.split(",")]

    print(f"loading {args.predictions_path}...", flush=True)
    df = pl.read_parquet(args.predictions_path)
    wide = build_wide_table(df, horizons)
    wide = add_regime_features(wide, horizons)
    print(f"wide table: {wide.shape[0]} rows, {wide.shape[1]} cols", flush=True)

    # Feature columns: all pred + univ_mean + univ_std
    feature_cols = [c for c in wide.columns
                    if c.startswith("pred_")
                    or c.startswith("univ_mean_")
                    or c.startswith("univ_std_")]
    print(f"features: {len(feature_cols)} (sample: {feature_cols[:5]})", flush=True)
    print(f"clf={args.clf_type}  holdout_days={args.holdout_days}", flush=True)

    test_start = _to_date(args.test_start)
    test_end   = _to_date(args.test_end)
    test_dates = sorted(
        wide.filter((pl.col("date") >= test_start)
                    & (pl.col("date") <= test_end))["date"].unique().to_list(),
    )
    print(f"test dates: {len(test_dates)}", flush=True)

    realization_offset = HORIZON_REALIZATION_TRADING_DAYS.get(args.target_horizon, 1)
    summary_rows = []
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for N in n_values:
        t0 = time.time()
        print(f"\n=== N={N} ===", flush=True)
        results = run_one_N(
            wide=wide,
            test_dates=test_dates,
            feature_cols=feature_cols,
            target_horizon=args.target_horizon,
            realization_offset=realization_offset,
            win_threshold_bps=args.win_threshold_bps,
            N_lookback=N,
            thresholds=thresholds,
            top_K=args.top_k,
            cost_bps=args.cost_bps,
            holdout_days=args.holdout_days,
            clf_type=args.clf_type,
        )
        out = pl.DataFrame([{
            "test_date":         r.test_date,
            "N":                 r.N,
            "chosen_threshold":  r.chosen_threshold,
            "lookback_pnl_bps":  r.lookback_pnl_bps,
            "lookback_sharpe":   r.lookback_sharpe,
            "n_lookback_trades": r.n_lookback_trades,
            "n_today_trades":    r.n_today_trades,
            "today_pnl_bps":     r.today_pnl_bps,
        } for r in results])
        out.write_parquet(args.out_prefix.with_name(f"{args.out_prefix.name}_N{N}.parquet"))

        traded = out.filter(pl.col("n_today_trades") > 0)
        if traded.shape[0] > 0:
            tb = traded["today_pnl_bps"]
            sd = float(tb.std()) or 1e-9
            sharpe_traded = float(tb.mean()) / sd * np.sqrt(252)
            ab = out["today_pnl_bps"]
            sharpe_all = float(ab.mean()) / (float(ab.std()) or 1e-9) * np.sqrt(252)
            summary_rows.append({
                "N": N,
                "test_days": out.shape[0],
                "traded_days": traded.shape[0],
                "trade_frac": traded.shape[0] / out.shape[0],
                "mean_bps_traded": float(tb.mean()),
                "sharpe_traded": sharpe_traded,
                "hit_traded": float((tb > 0).sum() / traded.shape[0] * 100),
                "mean_bps_all": float(ab.mean()),
                "sharpe_all": sharpe_all,
                "elapsed_sec": time.time() - t0,
            })
            print(f"  test_days={out.shape[0]} traded={traded.shape[0]} "
                  f"({100*traded.shape[0]/out.shape[0]:.1f}%) | "
                  f"mean_traded={float(tb.mean()):+.2f} sharpe_traded={sharpe_traded:+.2f} | "
                  f"mean_all={float(ab.mean()):+.2f} sharpe_all={sharpe_all:+.2f}",
                  flush=True)
        else:
            print(f"  no trades on N={N}", flush=True)
            summary_rows.append({"N": N, "test_days": out.shape[0],
                                 "traded_days": 0, "elapsed_sec": time.time() - t0})

    pl.DataFrame(summary_rows).write_parquet(
        args.out_prefix.with_name(f"{args.out_prefix.name}_summary.parquet"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
