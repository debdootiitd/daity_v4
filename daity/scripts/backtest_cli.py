"""`daity-backtest` — Phase 5 v1 backtest CLI.

Runs a per-trade-independence backtest of a TradeableFineTune (Stage 1) or
TradeableOBFineTune (Stage 2) checkpoint on the given test window.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console

from daity.data.parquet_store import ParquetStore
from daity.eval.backtest import BacktestConfig, run_backtest
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _parse_curve(s: str) -> list[tuple[float, int]]:
    """Parse '0.001:8,0.003:4,0.005:2' → [(0.001,8),(0.003,4),(0.005,2)] sorted."""
    if not s.strip():
        return []
    pairs: list[tuple[float, int]] = []
    for tok in s.split(","):
        r_str, k_str = tok.strip().split(":")
        pairs.append((float(r_str), int(k_str)))
    return sorted(pairs)


@click.command()
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True,
              help="TradeableFineTune/OB *.state.pt to backtest.")
@click.option("--secondary-checkpoint", type=click.Path(path_type=Path), default=None,
              help="Optional ensemble: a SECOND checkpoint whose LONG scores "
                   "are rank-averaged with the primary's (per cross-section). "
                   "Use --ensemble-secondary-weight to set the secondary's weight.")
@click.option("--ensemble-secondary-weight", type=float, default=0.5, show_default=True,
              help="Weight on secondary checkpoint in rank-averaging (0..1).")
@click.option("--feature-root", type=click.Path(path_type=Path),
              default=Path("data/features_parquet"), show_default=True)
@click.option("--label-root", type=click.Path(path_type=Path),
              default=Path("data/labels"), show_default=True)
@click.option("--tradeable-root", type=click.Path(path_type=Path), required=True,
              help="Tradeable-labels config dir.")
@click.option("--ob-root", type=click.Path(path_type=Path), default=None,
              help="OB parquet root (required for OB-conditioned models).")
@click.option("--test-start", required=True)
@click.option("--test-end",   required=True)
@click.option("--as-of",      default=None,
              help="Latest visible ts. Defaults to --test-end.")
@click.option("--max-symbols", type=int, default=None)
@click.option("--side", type=click.Choice(["long_only", "short_only", "both"]),
              default="both", show_default=True,
              help="Trading sides to enable.")
@click.option("--long-top-k",  type=int, default=5, show_default=True)
@click.option("--short-top-k", type=int, default=5, show_default=True)
@click.option("--long-score-threshold",  type=float, default=0.0, show_default=True)
@click.option("--short-score-threshold", type=float, default=0.0, show_default=True)
@click.option("--long-score-pct-gate", type=float, default=0.0, show_default=True,
              help="Percentile entry gate. 0.05 = require top 5%% of cross-section "
                   "combined_long. Robust to score distribution shift across ckpts.")
@click.option("--short-score-pct-gate", type=float, default=0.0, show_default=True,
              help="Symmetric percentile gate for SHORT side.")
@click.option("--long-short-alpha", type=float, default=0.0, show_default=True,
              help="LONG combined score formula: P(LONG) - alpha * P(SHORT).")
@click.option("--long-short-avoidance-max", type=float, default=1.0,
              show_default=True,
              help="Skip LONG candidate when its P(SHORT) > this value.")
@click.option("--initial-capital", type=float, default=100_000.0, show_default=True)
@click.option("--position-size-pct", type=float, default=0.02, show_default=True,
              help="Fraction of initial_capital per trade (flat mode; e.g. 0.02 = 2%).")
@click.option("--position-size-mode",
              type=click.Choice(["flat", "score_pct_rank"]),
              default="flat", show_default=True,
              help="flat = uniform position_size_pct. score_pct_rank = linearly "
                   "interp between min/max by cross-section score percentile.")
@click.option("--position-size-min-pct", type=float, default=0.02, show_default=True,
              help="Min position size when position_size_mode='score_pct_rank'.")
@click.option("--position-size-max-pct", type=float, default=0.20, show_default=True,
              help="Max position size when position_size_mode='score_pct_rank'.")
@click.option("--round-trip-cost-bps", type=float, default=0.0, show_default=True,
              help="Override total round-trip cost in bps. If 0, computed from "
                   "fixed-brokerage + statutory + slippage components.")
@click.option("--fixed-brokerage-per-leg", type=float, default=20.0,
              show_default=True, help="Flat brokerage in INR per executed leg.")
@click.option("--statutory-cost-bps", type=float, default=5.0, show_default=True,
              help="Statutory costs (STT + stamp + exchange + GST) as bps "
                   "of notional, round-trip.")
@click.option("--slippage-bps", type=float, default=10.0, show_default=True,
              help="Slippage / market impact as bps of notional, round-trip.")
@click.option("--use-first-touch", is_flag=True,
              help="Use triple-barrier first-touch exit on 5m bars (matches "
                   "triple-barrier training label). Otherwise uses cs.labels "
                   "(30-min forward return).")
@click.option("--target-long-pct",  type=float, default=0.0075, show_default=True)
@click.option("--stop-long-pct",    type=float, default=0.004,  show_default=True)
@click.option("--target-short-pct", type=float, default=0.0075, show_default=True)
@click.option("--stop-short-pct",   type=float, default=0.004,  show_default=True)
@click.option("--h-max-minutes",    type=int,   default=60,     show_default=True)
@click.option("--ambiguous-rule",
              type=click.Choice(["stop_first", "target_first", "midbar", "proportional"]),
              default="stop_first", show_default=True,
              help="How to resolve a 5m bar that crosses BOTH target and stop.")
@click.option("--exit-mode",
              type=click.Choice(["triple_barrier", "score_rerank", "score_floor",
                                  "hybrid_rerank", "hybrid_floor"]),
              default="triple_barrier", show_default=True,
              help="Trade exit logic. triple_barrier=current (tgt+stop+timeout). "
                   "score_*=replace stop with model re-rank/floor. "
                   "hybrid_*=add model exit on top of price stop.")
@click.option("--score-rerank-k", type=int, default=3, show_default=True,
              help="Exit when held symbol's rank in current cross-section "
                   "exceeds this. Only used in *_rerank modes.")
@click.option("--score-floor", type=float, default=0.0, show_default=True,
              help="Exit when held symbol's score < this. Only used in *_floor modes.")
@click.option("--score-exit-curve", type=str, default="",
              help="2-D exit curve as 'r1:k1,r2:k2,...' pairs ascending in r. "
                   "When set, every-bar curve exit replaces flat score_rerank_k "
                   "AND disables static target. Example: "
                   "'0.001:8,0.003:4,0.005:2,0.01:1' = hold loose at small |r|, "
                   "tighter at higher |r|.")
@click.option("--report-path", type=click.Path(path_type=Path), default=None)
@click.option("--trades-path",  type=click.Path(path_type=Path), default=None,
              help="Optional: write per-trade detail CSV.")
def main(
    checkpoint: Path, secondary_checkpoint: Path | None,
    ensemble_secondary_weight: float,
    feature_root: Path, label_root: Path,
    tradeable_root: Path, ob_root: Path | None,
    test_start: str, test_end: str, as_of: str | None,
    max_symbols: int | None,
    side: str,
    long_top_k: int, short_top_k: int,
    long_score_threshold: float, short_score_threshold: float,
    long_score_pct_gate: float, short_score_pct_gate: float,
    long_short_alpha: float, long_short_avoidance_max: float,
    initial_capital: float, position_size_pct: float,
    position_size_mode: str, position_size_min_pct: float, position_size_max_pct: float,
    round_trip_cost_bps: float,
    fixed_brokerage_per_leg: float,
    statutory_cost_bps: float, slippage_bps: float,
    use_first_touch: bool,
    target_long_pct: float, stop_long_pct: float,
    target_short_pct: float, stop_short_pct: float,
    h_max_minutes: int, ambiguous_rule: str,
    exit_mode: str, score_rerank_k: int, score_floor: float,
    score_exit_curve: str,
    report_path: Path | None, trades_path: Path | None,
) -> None:
    """Backtest a tradeable-direction model."""
    feature_store = ParquetStore(feature_root)
    universe = sorted(p.stem for p in feature_store.list_partitions("day"))
    if max_symbols is not None:
        universe = universe[:max_symbols]

    cfg = BacktestConfig(
        universe=universe,
        test_start=_parse_dt(test_start),
        test_end=_parse_dt(test_end),
        as_of=_parse_dt(as_of) if as_of else _parse_dt(test_end),
        feature_root=feature_root, label_root=label_root,
        tradeable_root=tradeable_root, ob_root=ob_root,
        secondary_checkpoint=secondary_checkpoint,
        ensemble_secondary_weight=ensemble_secondary_weight,
        side=side,
        long_top_k=long_top_k, short_top_k=short_top_k,
        long_score_threshold=long_score_threshold,
        short_score_threshold=short_score_threshold,
        long_score_pct_gate=long_score_pct_gate,
        short_score_pct_gate=short_score_pct_gate,
        long_short_alpha=long_short_alpha,
        long_short_avoidance_max=long_short_avoidance_max,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        position_size_mode=position_size_mode,
        position_size_min_pct=position_size_min_pct,
        position_size_max_pct=position_size_max_pct,
        round_trip_cost=round_trip_cost_bps / 10000.0,
        fixed_brokerage_per_leg_inr=fixed_brokerage_per_leg,
        statutory_cost_pct=statutory_cost_bps / 10000.0,
        slippage_pct=slippage_bps / 10000.0,
        use_first_touch=use_first_touch,
        target_long_pct=target_long_pct, stop_long_pct=stop_long_pct,
        target_short_pct=target_short_pct, stop_short_pct=stop_short_pct,
        h_max_minutes=h_max_minutes,
        ambiguous_rule=ambiguous_rule,
        exit_mode=exit_mode,
        score_rerank_k=score_rerank_k,
        score_floor=score_floor,
        score_exit_curve=_parse_curve(score_exit_curve),
    )

    console.print(
        f"[bold]Phase 5 v1 backtest:[/bold]\n"
        f"  ckpt:     {checkpoint}\n"
        f"  window:   {test_start} → {test_end}\n"
        f"  universe: {len(universe)} symbols\n"
        f"  rules:    top-{long_top_k} LONG / top-{short_top_k} SHORT, "
        f"thresh L>{long_score_threshold:.3f} S>{short_score_threshold:.3f}\n"
        f"  capital:  ₹{initial_capital:,.0f}, position size {position_size_pct:.1%}\n"
        f"  cost:     {round_trip_cost_bps:.0f} bps round-trip",
    )

    result = run_backtest(checkpoint, cfg)
    console.print("\n" + result.tearsheet())

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid serializing the model+sampler refs.
        cfg_dict = {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in asdict(cfg).items()
                    if k not in ("universe",)}
        cfg_dict["universe_size"] = len(cfg.universe)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "checkpoint": str(checkpoint),
            "config": cfg_dict,
            "n_trades": result.n_trades,
            "n_long":  result.n_long,
            "n_short": result.n_short,
            "total_pnl": result.total_pnl,
            "total_return_pct": result.total_return_pct,
            "win_rate": result.win_rate,
            "avg_winner": result.avg_winner,
            "avg_loser":  result.avg_loser,
            "win_loss_ratio": result.win_loss_ratio,
            "sharpe_daily": result.sharpe_daily,
            "max_drawdown_pct": result.max_drawdown_pct,
            "daily_pnl": {str(d): v for d, v in result.daily_pnl.items()},
        }
        report_path.write_text(json.dumps(payload, indent=2, default=str))
        console.print(f"\nReport written → {report_path}")

    if trades_path is not None and result.trades:
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        # Write per-trade detail as CSV.
        import csv
        with trades_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "symbol", "direction", "score",
                             "raw_return", "cost", "net_return",
                             "notional", "pnl"])
            for t in result.trades:
                writer.writerow([t.t.isoformat(), t.symbol, t.direction,
                                 f"{t.score:.4f}", f"{t.raw_return:.6f}",
                                 f"{t.cost:.6f}", f"{t.net_return:.6f}",
                                 f"{t.notional:.2f}", f"{t.pnl:.2f}"])
        console.print(f"Per-trade detail → {trades_path}")


if __name__ == "__main__":
    main()
