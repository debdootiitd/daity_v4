---
name: A2-class encoder lacks conviction calibration
description: Root cause of strategy ceiling — top-1 scores cluster tightly in [0.20, 0.25] with no per-trade EV gradient. Fixable only via encoder retraining (PU bandit + entropy regularizers, or unfreeze).
type: feedback
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
**The encoder is rank-trained, not confidence-trained.** A2's top-1 picks all score in [0.20, 0.25] — score value is uninformative about per-trade outcome quality. Empirically verified across thresholds T=0.10 → 0.22 (all 621 trades pass) → T=0.23 (cliff at 27 trades) → T=0.235 (zero trades).

**Why:** ListNet ranking loss optimizes the RELATIVE ORDERING of scores across the cross-section but doesn't constrain the absolute magnitude. The classifier head's `w_LONG` weights move uniformly across high-confidence picks. Combined with frozen encoder (default `--unfreeze none`), the head saturates into a "binary classifier" — pick is top-K or it isn't.

**How to apply:** Don't expect score-magnitude-based filters to improve per-trade EV on A1/A2/S1/F1m/K1-family checkpoints. They only reduce trade count. Per-trade EV at retail (~-0.5 bps capital) is bounded by the encoder's ranking quality.

**Fixes (must be tried in encoder retraining, not at backtest time):**
1. **Pure direct_pnl_loss with entropy/uniform_mix regularizers** (PU_RECOVERY_2v2 recipe, queued). Trains scores to predict realized PnL magnitude not just rank.
2. **Encoder_only unfreeze** (K1_CurveF_unfreeze_last2, U1_lora_A2_baseline) — give the encoder capacity to differentiate weak vs strong top-1 candidates.
3. **Higher rank_target_temperature_long** (currently 0.02) — peaked targets force the head to make sharper distinctions.
4. **Cost-aware labels** — label LONG only when realized return survives retail cost (would shrink positive base rate but eliminate marginal touches).

The 5 PU and 2 unfreeze configs queued as of 2026-05-14 directly test these. **Until one of these produces wider top-1 score distribution OR higher per-trade EV at threshold filter, retail break-even is unreachable.**
