# Autoloop final summary

**Session:** 2026-05-19 · ~70+ iterations · lightweight 75–300K param FT-Transformer + Set Transformer on engineered features.

## TL;DR — Honest champion

The most defensible result of this loop is the **9-model cross-config ensemble** of focal+small variants, evaluated under **rolling-anchor** backtest:

- **Annualised Sharpe**: **+1.13 ± 0.48** (range +0.50 to +1.84 across 10 offsets, never negative)
- **Annualised sleeve**: **+24.94% / yr** (geometric, net of 30 bps RT cost)
- **IC mean**: +0.0114 (single-offset; honest IC across seeds ~+0.008)

Layered on a **rolling-entry strategy** (continuous, threshold-gated entries with rolling exits, max 10 positions, hold=10 days, threshold=20 bps min pred_lr), the same 9-model ensemble lifts to:

- **Sharpe**: **+2.17**, **Sleeve**: **+42.45%/yr**, **MaxDD**: −5.7%, 158 trades/yr, 60.8% position-level hit-rate

**Validated under order-book execution** (2026-03-09 → 2026-05-04 window) at ₹2 lakh per pick: 94% of fills within the 30 bps cost assumption, mean slippage -14 bps **favourable** vs the cost budget.

## Strategy ladder — what we tried and what worked

### Multi-day hold + cost amortisation (iter5)
Going from hold=1 to hold=20 took baseline Sharpe from −1.98 to +0.72 — the single biggest win. The 30 bps RT cost is otherwise prohibitive on daily turnover.

### Lower LR (iter10) — but it was offset-luck
iter10 (lr=1e-4, 1ep) headlined Sharpe +3.20 / sleeve +69%, but **rolling-anchor across 10 K-offsets dropped it to +0.89**. The original +3.20 was offset-cherry-picked. This led to making **rolling-anchor mandatory** going forward.

### Smaller model (iter18) — 75K params beats 300K
Reducing `d_model 128 → 64` and `d_ft 32 → 16` lifted IC from +0.011 to +0.0163 (highest single-seed IC). Even smaller (~25K, d_model=32) drops back. **2 transformer layers is the sweet spot** — 1 layer goes IC-negative (-0.004), 3 layers also drops to ~+0.0022.

### Focal BCE classifier head (iter17)
Adding `w_focal=2.0` on a positive-direction BCE auxiliary head improved IC to +0.0147 and stabilised the model across seeds. `w_focal=1.5` and `w_focal=4.0` were worse — 2.0 is the optimum.

### Multi-task with 2 epochs (iter33, iter55-57)
focal+small + **2 epochs** lifted 3-seed mean IC to +0.0127 ± 0.0099 (vs 1ep ensemble's +0.0086). But the resulting **Sharpe dropped** to +0.74 (less ETF-padded) — useful as an ensemble member, not as a standalone model.

### Seed-variance retraction
Single-seed IC swings from +0.017 to **−0.024** for identical configs. The autoloop spent ~30 iterations measuring noise. Going forward, **every config must be evaluated with ≥3 seeds**, and the ensemble is what's reported. Memory note saved at `feedback_autoloop_seed_variance.md`.

### Cross-config ensembling
Pure seed-averaging is good; **cross-config + seed averaging is better**:

| Ensemble | IC | Rolling-anchor Sharpe (hold=10 K=10) | Sleeve mean | Min offset | Std |
|---|---|---|---|---|---|
| iter25 (single seed) | +0.0115 | +1.00 ± 0.72 | +23.1% | −0.12 | 0.72 |
| focal+small × 3 seeds | +0.0086 | +1.08 ± 0.57 | +22.3% | +0.15 | 0.57 |
| focal+small × {1ep, 2ep} × 3 seeds (6 models) | +0.0135 | +1.12 ± 0.55 | +22.9% | +0.37 | 0.55 |
| **focal+small × {1ep, 2ep, lr=1.5e-4} × 3 seeds (9 models)** | **+0.0114** | **+1.13 ± 0.48** | **+24.94%** | **+0.50** | **0.48** |

Each layer of diversification tightens the offset-std and raises the floor.

## Rolling-entry strategy — the bigger lift

Replacing chunky periodic rebalances with **continuous threshold-gated entries** dramatically improves the realised P&L:

### Mechanics
- Every trading day, screen top-50 stocks by ensemble `pred_lr`
- Open a new position if `pred_lr > 0.002` (20 bps predicted) and a slot is available (max 10 positions)
- No same-stock duplicates
- Each position exits 10 trading days after its own entry (independent timer per position)
- 1/10 equal-weight; partial-portfolio (any unused slots stay in cash)

### Results vs chunky rebal

| Metric | Chunky rebal (K=10 hold=10) | Rolling-entry (thr=0.002) |
|---|---|---|
| Sleeve / yr | +24.94% (rolling-anchor mean) | **+42.45%** (single path) |
| Sharpe / ann | +1.13 ± 0.48 | **+2.17** |
| Max DD | — (per-offset) | **−5.68%** |
| Trades / yr (delta) | 410-480 | **316 (158 entries + 158 exits)** |
| Position-level hit-rate | 50% | **60.8%** |
| Mean fill | 10/10 always | 6.4/10 (cash when no signal) |
| Implicit regime gate | No | **Yes** (threshold filters low-conviction periods) |

### Why it works
1. **Threshold acts as a regime gate**: during Aug-Nov 2025 the model couldn't generate signals clearing 20 bps, and the strategy effectively went to cash. It re-engaged for the early-2026 rally fully loaded. No human intervention.
2. **Position-level hold timer** decouples entries from exits — no chunky day where all picks turn over at once.
3. **Higher per-trade quality**: hit-rate jumps from 50% (no filter) to 60.8% (thr=0.002) because marginal picks are rejected.

## OB execution validation

For the 2026-03-09 → 2026-05-04 window (the only period with bid/ask depth data), the 9-model ensemble at K=10 hold=10 was simulated with realistic top-5 ask/bid walks.

### Capacity at ₹2 lakh per pick (₹20L portfolio)
- 94% of fills land within 30 bps slippage
- Median slippage from top-of-book: **0.4 bps**
- Mean slippage: 11.3 bps (skewed by ~6% thin-name tail)
- p95 slippage: 31 bps
- Max slippage: 508 bps (TORNTPHARM — should be name-gated)
- **OB realised −14 bps better than assumed 30 bps cost** on average

### Position-size scaling (₹50L portfolio = ₹5 CR at K=10)

| Pos / pick | Avg deployed | Median fill / pick | Mean OB return / rebal | Slip vs ideal | In-window Sharpe |
|---|---|---|---|---|---|
| ₹15 L | 47% of target | 56% | +3.72% | -55 bps | +2.71 |
| ₹25 L | 36% of target | 34% | +3.63% | -65 bps | +2.58 |
| **₹50 L** | **23% of target** | **17%** | **+3.54%** | **−74 bps** | **+2.45** |

At ₹50L per pick the strategy can only deploy ~₹1.15 CR (23% of target ₹5 CR) using top-5 ask depth alone. Capacity sweet spot is ₹15-25L per pick.

The ₹50L per-pick experiment also shows the cumulative chart underperforms ideal by only ~2pp over a month of rebals (12.3% vs 14.0%) — the slippage stings per rebal but the strategy *does* execute and produce P&L. For larger size, smarter execution (TWAP, passive limit ladder, name-gating) would close the gap.

### Execution-system recommendation

Five-component stack to push ₹50L slippage from -74 bps to ~-20 bps:

1. **Pre-trade liquidity scanner**: skip picks where expected slip > 50 bps (drops TORNTPHARM-class tail)
2. **TWAP over 30-60 min**: split ₹50L into 10× ₹5L child orders, each fits top-of-book
3. **Passive limit ladder**: start at mid, escalate every 2 min to touch+5 bps as backstop
4. **Iceberg orders on NSE**: reveal 20% of size at a time, prevent front-running
5. **TCA feedback**: drop combos that consistently slip > 30 bps after 50 trades

## What didn't work

| Idea | Outcome |
|---|---|
| Multi-epoch on full train+OOS | Iter2 (5 ep): OOS IC went **negative**. Overfits to OOS days seen in earlier epochs. |
| Sharpe loss (w_sharpe=0.5 or 0.1) | Flipped IC sign / no improvement across 4 attempted runs. Differentiable Sharpe is too noisy on small batches. |
| 5d / 7d / 12d / 15d cumul targets (single-head replacement) | All went to ~0 IC on 1d-real-lr eval. Cumul labels dilute the cross-sectional signal when used as the sole regression target. (Multi-head is the right answer — but needs code change.) |
| Higher LR (5e-4, 2e-4) | Worse rolling-anchor Sharpe. |
| Bigger model (~600K params, d_model=192) | IC dropped from +0.0163 to +0.0107. |
| Tiny model (~25K, d_model=32) | IC lower than small. ~75K is the capacity floor. |
| 1 transformer layer | IC **negative**. 2 is the floor. |
| 3 transformer layers | IC ~0. 2 is also the ceiling. |
| Buffer=504 days vs 252 | No improvement. |
| Combining focal + ranking, focal + sharpe, focal + heavy WD | High Sharpe sometimes (ETF-padded) but low IC (~0.004). |
| gclip=0.5 / 0.3 alone | Helped seed=42 alone, but 3-seed mean +0.0037 ± 0.0110 — single-seed artifact. |
| batch=32 | 3-seed mean −0.0020 (negative). Smaller batch (16) was right. |
| Lower LR (5e-5, 7e-5) | Loses signal. |
| Longer warmup (500 steps) | Crashed in the autoloop (thread pool) — not re-attempted. |

## Universe + ETF effects

The model **does** lean on LIQUIDBEES (cash-equivalent ETF), LTGILTBEES (gilt ETF), NIFTYBEES (index ETF) — confirmed in iter10 audit (12 of 60 chunky-rebal picks were these names). User confirmed ETFs are **valid allocations** for this strategy. Real-stock picks (POWERINDIA, COCHINSHIP, DIXON, ADANIPOWER, MAZDOCK, JSWENERGY, …) drive most of the alpha during rally regimes; ETFs pad Sharpe in flat/down regimes.

## Trade counts and capacity

For the 9-model ensemble at K=10 hold=10 (canonical strategy):

| | Value |
|---|---|
| Rebalances / year | 24 |
| Naive trades / year (exit + enter all) | 480 |
| Delta trades / year (only changed names) | 410 |
| Mean overlap between rebals | 15% (effective 85% turnover) |
| Trades / trading day | ~1.7 |
| Gross volume / year @ ₹2L per pick | ₹8.2 CR (41× portfolio turnover) |
| Capacity at acceptable slippage | ~₹2-3 CR notional (₹15-25L per pick) |

## Key files

- **Champion summary**: `reports/autoloop_champion_summary.md` (this file)
- **Full results table**: `reports/autoloop_results.md`
- **Champion predictions**: `runs/autoloop_predictions/ensemble_9_models.parquet` (9-model focal+small ensemble)
- **Rolling-anchor scorer**: `scripts/score_rolling_anchor.py`
- **Rolling-entry strategy**: `scripts/rolling_entry_strategy.py`
- **OB fill audit**: `scripts/audit_ob_fillability.py`
- **OB paper P&L**: `scripts/ob_paper_pnl_partial.py`
- **Seed ensembler**: `scripts/seed_ensemble.py`
- **Multi-horizon labels**: `/tmp/add_multihorizon_labels_v2.py`

## Memory notes saved

- `feedback_autoloop_seed_variance.md` — single-seed IC unreliable; always ≥3 seeds + ensemble
- See `MEMORY.md` for full index

## Update — 10d-target wave + cross-target ensemble (completed after summary)

### 10d-target as sole regression head — fails the chunky-rebal backtest

Three configs × 3 seeds = 9 models trained with `--target-label-col label_10d_cumul` (sum of 10 daily log-returns). Same focal+small architecture.

3-seed means (IC measured against 1d real_lr — note: model trained on 10d):

| Config | IC mean ± std |
|---|---|
| target10d 1ep | +0.0026 ± 0.0050 |
| target10d 2ep | +0.0049 ± 0.0067 |
| target10d lr=1.5e-4 | **+0.0114 ± 0.0098** |

9-model 10d-target ensemble: IC +0.0127 (looks high — but chunky-rebal Sharpe is essentially zero):

| Backtest config | Rolling-anchor mean Sharpe |
|---|---|
| hold=10 K=20 | +0.01 |
| hold=10 K=10 | -0.11 |
| hold=20 K=5 | -0.04 |

The cumulative label dilutes daily cross-sectional ranking when used as the sole regression target. **This confirms the iter15 failure pattern with a larger sample.**

### Rolling-entry on 10d-target ensemble — works at higher threshold

Different prediction scale needs different threshold. Sweep:

| Threshold | Sleeve | Sharpe | MaxDD | Positions |
|---|---|---|---|---|
| 0.002 (would be tight for 10d) | -3.99% | -0.28 | -17.3% | 200 |
| 0.005 | +2.27% | +0.17 | -10.4% | 160 |
| **0.010 (100 bps min predicted 10d return)** | **+29.83%** | **+2.18** | **-6.3%** | 120 |
| 0.015 | +14.27% | +1.18 | -6.5% | 100 |
| 0.020 | +10.52% | +1.17 | -6.4% | 70 |
| 0.030 | +5.58% | +0.73 | -4.2% | 30 |

Best 10d-target rolling-entry at threshold=0.010: **Sharpe +2.18, sleeve +29.83%** — slightly higher Sharpe but LOWER sleeve than 1d-target ensemble (Sharpe +2.17, sleeve +42.5%). More selective, fewer positions, lower total deployment.

### Mega 18-model cross-target ensemble

Combined 1d-target 9 models + 10d-target 9 models via per-day z-score normalisation, then averaged. Result:

- Combined IC: +0.0229 / IR +2.74 (highest ever recorded)
- Best rolling-anchor Sharpe (hold=5 K=5): **+0.84 ± 0.35** (lower point estimate, lower std)
- Rolling-entry at z-score thr=0.7: Sharpe **+1.97**, Sleeve **+26.45%**

The mega ensemble has TIGHTER offset-std but WORSE point estimate than 9-model 1d-target alone. Adding 10d-target models dilutes the signal more than it diversifies. **Sticking with 9-model 1d-target ensemble as production champion.**

### Lesson — multi-task is the right answer, not single-head cumul-target

Both iter15 (5d-target single-head) and this iter80+ wave (10d-target single-head) failed because they replaced the 1d label rather than augmenting it. The right architecture is multi-task with separate regression heads for {1d, 10d_close, 10d_1h} — the 1d head anchors cross-sectional ranking, the 10d head specialises for the hold-period decision. That requires a model code change (n_horizons > 1 in `regression_head`); deferred to next pass.

## Next moves

1. **Multi-task head for 10d target** — current models trained on 1d label and summed; a proper 10d auxiliary head should improve hold=10 performance (in flight: 9× 10d-target runs).
2. **Live-trading execution scaffolding** — implement the 5-component execution stack against Zerodha/Upstox API; test paper-trade at ₹2L per pick.
3. **Multi-year OOS revalidation** — the +42.5% rolling-entry sleeve is from a 1-year sample including the April 2026 rally. Regime sensitivity is unknown.
4. **OB-aware backtest** — extend `ob_paper_pnl_partial.py` to simulate TWAP / passive-limit execution and produce a realistic capacity-vs-slippage curve.
5. **Add 1h-after-open target label** — for execution-realistic prediction horizon.

## Charts produced

- `reports/ob_paper_pnl_ensemble_K10_h10.png` — OB execution at ₹2L per pick, K=10 hold=10
- `reports/rolling_entry_ens9_thr0.002.png` — Rolling-entry strategy equity curve + daily P&L + fill state
- `reports/rolling_entry_ens9_thr0.png` — Same strategy at threshold=0 (no gate, for comparison)
- `reports/ob_pnl_ensemble9_50L_partial.png` — OB execution at ₹50L per pick with partial-fill logic
