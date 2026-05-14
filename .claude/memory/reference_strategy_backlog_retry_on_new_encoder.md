---
name: Strategy backlog — retry on better encoder
description: Strategies queued for retry once a higher-quality encoder ckpt becomes available. Currently deprioritized in favor of encoder training.
type: reference
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
# Strategies to retry once a better-encoder ckpt lands

A "better encoder" = one whose top-1 scores show wider distribution (>0.10 score range) AND/OR higher L@10/base_rate ratio than A2's 1.35×.

## Strategies CONFIRMED working on A2 — apply directly to new ckpt

1. **Curve F asymmetric exit** — `exit_mode=score_rerank score_rerank_k=3` + `score_exit_curve=[[-0.01,1],[-0.005,1],[-0.003,2],[-0.001,5],[0.001,8],[0.005,5],[0.01,3]]` (scale breakpoints to match label barriers if not tgt=0.50%/stop=0.30%).
2. **Strict absolute threshold T≈0.23** — apply at retail cost. Cuts trade count to ~27, retains high-conviction picks.
3. **Hybrid_rerank (stop + curve)** — for wider-target ckpts (tgt > 0.50%), keep static stop to bound losses. Used by S1.

## Strategies NEEDING new encoder to work

1. **Conviction-weighted sizing** — failed on A2 because score is uninformative. With a confidence-calibrated encoder, sizing by score percentile should improve EV.
2. **A1+A2 ensemble** — failed because A1 was weaker. With TWO comparable-quality ckpts (e.g., PU_RECOVERY_2v2 + S1), agreement-based ensemble may add value. Code already shipped (`secondary_checkpoint` + `ensemble_secondary_weight`).
3. **Percentile entry gate** — meaningless at small top_k. Useful only with WIDE top_k + new encoder that differentiates within top-K.

## Configs to RE-RUN on each new ckpt

Once a new ckpt completes training, manually backtest these 4 variants in parallel (each ~2 min on vast.ai):
1. Top-1 baseline (inst + retail)
2. Top-1 + Curve F (score_rerank K=3) + retail
3. Top-1 + Curve F + threshold sweep T ∈ {0.20, 0.22, 0.23, 0.24} (probe score cliff)
4. Top-1 + hybrid_rerank K=3 + retail

Report best inst and best retail, compare to A2+T=0.23 retail = -0.68% champion.

## Other strategies queued but never tested due to engineering cost

- **Multi-target ensemble** — train ONE ckpt, backtest at multiple (target, stop) pairs, pick best Sharpe
- **Multi-day holding** — h_max ∈ {1d, 2d} vs current 60min
- **Regime conditioning** — only fire at certain intraday times (first hour, last hour) or vol regimes
- **Cost-aware label generation** — rebuild labels to require realized return > retail cost
- **Top-K portfolio ensemble** — combine top-1 + top-3 + top-5 picks

If encoder retraining unlocks meaningful improvement, these become worth implementing.
