---
name: PU* training trajectory baselines (2026-05-14)
description: Reference eval trajectories for A2 / PU6v4 / PU7 / PU10 — PU10 broke L@10=0.167 with flat LR ratios
type: project
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---

## 2026-05-14 ~10:00 UTC: PU10 BROKE THE PU* CEILING

**PU10_NoLRRatio_baseline** (lr=2e-5, lr_ratio_lora=1.0, lr_ratio_backbone=1.0):
- step 5000: 0.1266 → step 9000: **0.1670** ← new PU* family best
- Beats PU6v4 peak (0.158) and PU7 plateau (0.13)

**Lesson:** Architect's hand-tuned 1:0.5:0.3 LR ratios HURT backbone learning. Adam's 2nd-moment normalization already auto-scales per-param. Flat LR ratios (head=lora=backbone=1.0) with conservative global LR is the winning recipe.

**Why:** Adam update = lr × m̂ / √v̂ — backbone's small v̂ already produces relatively larger updates per gradient magnitude. Manual LR ratio = 0.3 then additionally scales backbone updates down, fighting the autoscale.

**How to apply:** Future encoder training — flat LR ratios + lr ∈ [1e-5, 5e-5]. Skip hand-tuned per-group LR. If layer-specific LR truly needed, prefer layer-wise depth decay (γ=0.9) over flat group ratios.

PU8 (variance reduction accum=4) plateaued at ~0.14 — variance was not the bottleneck.
PU7 (architect's LR ratios) plateaued at ~0.13 — KILLED.

---

## Original PU7 launch notes (kept for context)

PU7_BalancedLR + PU7_AggressiveBackboneLR launched 2026-05-14 08:08 UTC. 15k steps, eval every 1000.

**Step rate:** ~133 steps/min/worker with 40M trainable backbone (vs 158 in prior estimate). ETA finish ~09:55 UTC.

## Baseline L@10 trajectories (val_start 2026-04-20)

| Step | A2 (head-only) | PU6v4 (TRUE unfreeze, lr=1e-5) | PU_RECOVERY_2v4 (lr=1e-5+regs) |
|------|----------------|--------------------------------|--------------------------------|
| 1000 | (no eval — every 2000) | (no eval) | (no eval) |
| 2000 | 0.0952         | 0.1542                         | 0.1252                         |
| 4000 | 0.1674         | 0.1576                         | 0.1304                         |
| 6000 | 0.1492         | 0.1508 (peaked at 4k)          | 0.1332                         |
| 8000 | 0.1454         | killed                         | killed                         |
| 10000| **0.1816 (peak)** | — | — |

## PU7 step-1000 data (first eval)
- PU7_BalancedLR (back_ratio=0.3): 0.1404
- PU7_AggressiveBackboneLR (back_ratio=0.5): 0.1436

## Why **Why:** Architect predicted +0.04–0.07 above PU6v4's 0.158 peak from {LR ratios, focal/rank, weight decay}. Target: L@10 > 0.20 by step 6000.

## How to apply: At each PU7 eval point (steps 2k/3k/...): compare against PU6v4 column (same row by step) and A2's profile (peaks at 10k). Decision rule:
- step 4000 L@10 < 0.16 → architect's diagnosis wrong; backbone still effectively frozen; pivot to higher back_ratio (0.7-1.0) or much higher global LR
- step 4000 L@10 ∈ [0.16, 0.18] → on PU6v4 trajectory; wait for step 10k
- step 4000 L@10 > 0.18 → ALREADY beating A2 at A2's peak step; champion candidate
- step 10000 L@10 > 0.18 → final-pass beat A2 → run full strategy retry stack

## Grad ratio diagnostic (step 1200)
PU7_BalancedLR: head=0.6, lora=0.01, back=0.02 → head:back ≈ 30:1 still
- LR ratios apply to *updates*, not gradients. back updates = 0.02 * 5e-5 * 0.3 = 3e-7 per step
- head updates = 0.6 * 5e-5 = 3e-5 per step → 100x larger
- So while head:back grad ratio is 30:1, head:back update ratio is 100:1. Encoder still moves slowly relative to head.
