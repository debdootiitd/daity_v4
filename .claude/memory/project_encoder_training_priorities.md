---
name: Encoder training priorities (2026-05-14 pivot)
description: Which sweep configs to focus on for encoder retraining. Strategy experimentation paused. Goal: produce a ckpt with wider score distribution or higher L@10/base ratio than A2's 1.35x.
type: project
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
# CRITICAL BUG DISCOVERED 2026-05-14 ~07:00 UTC

**ALL prior `--unfreeze` experiments were silently head-only.** When Stage 2 module is used (parent_ob_checkpoint set — ALWAYS in our sweep), the `--unfreeze` flag is IGNORED. Only `--ob-unfreeze` controls encoder unfreezing. The OB module supports `unfreeze_mode ∈ {none, fusion, all}`:
- `none`: only head trains (default)
- `fusion`: OB encoder + fusion layers trainable
- `all`: EVERYTHING (main encoder + OB + fusion + head) trainable — ~40M backbone params

This means K1_CurveF_unfreeze_last2 (L@10=0.167), U1_lora_A2_baseline (0.159), PU5v2, PU6v2, PU_RECOVERY_2/2v2/2v3, B1, B3 ALL ran head-only training. The "L@10≈0.18 ceiling" we attributed to encoder capacity was actually the HEAD-only ceiling. **Encoder unfreeze was never tested before 2026-05-14 07:00 UTC.**

To use encoder unfreeze in sweep configs: set `train_ob_unfreeze: "all"` (NOT `train_unfreeze`).

# Pivot direction: stop tuning strategies, train better encoders

After mapping the strategy lever space on A2/S1, the bottleneck is the encoder. Best retail = -0.68%, gap to break-even = 0.68pp. No backtest-time lever closes this. The model produces uniform-confidence top-1 picks; per-trade EV is bounded.

## Top-priority configs (training)

These configs target ENCODER improvement via loss recipe or unfreeze:

1. **PU_RECOVERY_2v2_fullsoftmax_entropy** — pure direct_pnl + entropy + uniform_mix + baseline_subtract regularizers. The full "Fix 1" recipe from the ML scientist diagnosis. If it produces a model whose top-1 scores have wider distribution OR higher per-trade EV at strict thresholds, it's the breakthrough we need.
2. **PU5v2_unfreezeLast2_pnl100_T07_k20** — pure direct_pnl + encoder_only unfreeze. PU5's partial best (step 2000) gave -1.93% inst. Full training to 18000 may unlock more.
3. **PU6v2_unfreezeAll_lr1e5_pnl200** — pure direct_pnl + FULL encoder unfreeze. Maximum model capacity. Partial training at step 4000 showed L@10=0.1526 (best of PU family).
4. **K1_CurveF_unfreeze_last2** — encoder_only unfreeze on tight-target labels. Could rescue the K1 family if added capacity is what's needed.
5. **U1_lora_A2_baseline** — LoRA unfreeze diagnostic on A2's working labels. Tests whether LoRA capacity alone breaks L@10 ceiling.

## DROP from queue (strategy experiments, will fail or be redundant)

K1-family variants using frozen encoder on tight (0.30%) target labels:
- K1_CurveF (frozen, redundant with K1)
- K1_CurveF_scaled (frozen, redundant)
- K1_CurveF_entryGated_T05 (frozen, redundant)
- K1_CurveG_sharper (frozen, redundant)
- K1m_tightTgt_M1_rerank3 (already trained — weak L@K)

Why drop: K1 base trained with L@10=0.119 vs base=0.130 (BELOW random). All variants using same labels + frozen encoder will produce same broken model. Only K1_CurveF_unfreeze_last2 has a chance because it unfreezes.

Other strategy-only configs (also low priority):
- F1m_wideTgt_M1_rerank3_stable (already trained — L@10 below base rate, broken)
- K5m_NIFTY50_M1_rerank3_top3 (already trained — weak L@K = 1.04x, NIFTY-50 didn't help)
- K5m_sized_NIFTY50_pctrank (same labels — likely weak)

## Strategy backlog (defer until new encoder exists)

See reference_strategy_backlog_retry_on_new_encoder.md for full list. Key items:
- Curve F + T=0.23 threshold combo (proven on A2, retry on new ckpt)
- Hybrid_rerank for wider-target models (proven on S1)
- Agreement-based ensemble (requires 2 comparable-quality ckpts)
- Multi-target backtest sweep
- Score percentile entry gate (only useful with wider score distribution)

## Decision rule

After PU_RECOVERY_2v2 / PU5v2 / PU6v2 / K1_CurveF_unfreeze_last2 / U1_lora complete:
- If any has L@10/base ratio > 1.35× (beats A2): promote to "champion encoder", retry full strategy stack
- If none beat A2: try LARGER architecture, MORE training data, MORE TRAINING STEPS, or different feature inputs (OB encoder, longer history)
