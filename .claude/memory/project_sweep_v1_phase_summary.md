---
name: Sweep v1 strategy-experimentation phase (closeout summary)
description: What we learned from 100+ backtest variants on existing checkpoints. Strategy ceiling reached at -0.68% retail / -0.83% inst on A2 alone. Pivot to encoder retraining 2026-05-14.
type: project
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
# Strategy phase reached its ceiling 2026-05-14

After mapping the lever space on A2 (and partially S1) checkpoints, the BEST achievable result is:
- **A2 + Curve F + T=0.23 = -0.68% retail (27 trades, DD 0.61%)**
- **A2 + Curve F (no threshold) = -0.83% inst (621 trades)**

Gap to retail break-even is **0.68pp**. All single-checkpoint backtest-time levers exhausted. **Path forward = better encoder/model**, NOT more strategy tuning.

**Why:** All models trained so far (A1/A2/S1) produce tightly clustered top-1 scores in [0.20, 0.25]. Score VALUE is uninformative about per-trade quality — only the RANKING is. Strict threshold filtering reduces trade count but doesn't improve per-trade EV. Therefore retail cost (19bps RT) eats the edge regardless of which exit/sizing policy is applied.

**Why:** retail cost adds ~1.4 bps capital overhead per trade. The model's per-trade EV at inst is -0.5 bps capital. We're 1.9 bps from per-trade profitability under retail conditions. No backtest-time lever closes this — only a stronger encoder will.

## Bottom-line lever ranking (empirical, on A2 ckpt)

WORKED (apply to all future ckpts):
1. **Score-driven exit + 2D curve (Curve F asymmetric)** — replaces static stop with rank-based exit. -2.3pp improvement on A2 (from -3.14% to -0.83% inst). Code shipped: `score_rerank` exit_mode + `score_exit_curve` with negative breakpoints.
2. **Strict absolute threshold (T=0.23 on A2's score distribution)** — keeps only 27 high-conviction trades, cuts retail cost overhead proportionally. -11pp improvement at retail.
3. **Hybrid_rerank (static stop + Curve F)** for wider-target models like S1 — keeps stop protection while adding rank exits. S1 best inst -0.79%.

DIDN'T WORK:
1. **Conviction-weighted position sizing (score_pct_rank on top-5)** — picks deeper into top-K, lower-conviction trades have bad EV and fixed brokerage hits them worse. -19% inst.
2. **A1+A2 rank-averaged ensemble** — A1's weaker signal dilutes A2's. Monotonic degradation as A1 weight increases.
3. **Percentile entry gate at small top_k** — no-op (top-K always passes top-X% cutoff).
4. **Score floor exit (M3)** — removes static stop. Losers leak past -30bps to -50bps. Win rate UP but AvgL grows more.
5. **Extended hold time (60 → 120 min)** — no help. Winners that don't materialize in 60min don't materialize in 120.
6. **Tight-target retraining (K1, tgt=0.30%)** — L@10 dropped BELOW random base rate. Tighter target = more noise relative to signal. K1 base broken.
7. **Multi-target ensemble at backtest** — not tested.

## Best-performing config recipes (to retry on better encoder)

**Recipe A2-CHAMPION-RETAIL:**
```
labels: tgt=0.50% / stop=0.30% / h=60min / slip=5min / cost=19bps
loss: rank=1.0/1.0, focal=0.3, cls=0.1, direct_pnl=100, T=1.0, top_k=10, rank_target_temp=0.02
backtest: top-1, Curve F asymmetric, score_rerank K=3,
  score_exit_curve = [[-0.01,1],[-0.005,1],[-0.003,2],[-0.001,5],[0.001,8],[0.005,5],[0.01,3]]
  long_score_threshold = 0.23  (filter to top-1 picks with high absolute score)
  cost = retail (Rs 20 brokerage/leg + 5bps statutory + 10bps slippage = 19bps RT)
```
Result on A2: retail -0.68%, n=27 trades, win 14.8%, AvgW +0.76bps, AvgL -3.09bps, DD 0.61%.

**Recipe S1-WIDE-RETAIL:**
```
labels: tgt=0.75% / stop=0.40% / h=180min / slip=5min / cost=19bps
loss: rank=1.0/0.0, focal=0.5, cls=0.3 (stability recipe), direct_pnl=0
training: lr=3e-5, max_steps=18000, unfreeze=none
backtest: top-1, hybrid_rerank K=3 + Curve F scaled 1.5x,
  score_exit_curve = [[-0.015,1],[-0.0075,1],[-0.0045,2],[-0.0015,5],[0.0015,8],[0.0075,5],[0.015,3]]
  long_score_threshold = 0.24
```
Result on S1: retail -0.92%, n=26 trades, win 7.7%, AvgW +24.83bps (massive!), AvgL -5.90, DD 0.86%.

S1 has higher upside (big winners) but lower win rate. Different signal character from A2.

## Score distribution finding

Top-1 picks across A2 and S1 cluster in 0.20-0.25 score range. **The encoder produces uniform-confidence "binary" picks** — either a stock IS a top candidate or it isn't, with no usable conviction gradient.

This is the root cause of why threshold filtering can only reduce trade count, not improve per-trade EV. A more conviction-calibrated encoder would have wider score distribution and DIFFERENTIATED top-1 quality.
