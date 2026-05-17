# Adaptive Calibrated Strategy — v1

**Status:** **REVISED 2026-05-17 — earlier headline numbers retracted after independent
review identified a critical lookahead bug. See §0 below.**

**Test window:** 2025-02-01 to 2026-04-30 (303 trading days, NSE ~200-stock universe)  
**Best honest config (post-N-resweep + split-window verified):** K=1, N=**12**, holdout=3, GBM, day_plus_5, at realistic 15 bps round-trip cost  
**Honest result:** Sharpe **0.97** · +45.68 bps/day · **+30.98% sleeve-adjusted return** (14 months, ~26% annualized)  

**N=12 robustness — verified on a 50/50 split.** N=12 wins H1 (Feb–Sep'25) unambiguously
(Sharpe 0.61 vs N=10's 0.11) and ties for best in H2 (Oct'25–Apr'26, Sharpe 1.37 vs N=8's 1.41).

**Regime warning.** H2 is ~2× more profitable than H1 for every N (H1 Sharpe 0.61 / sleeve +9.35% vs
H2 Sharpe 1.37 / sleeve +19.78% at N=12). The +26% annualized headline averages a favorable
Q4'25–Q1'26. Forward expectations should anchor closer to **~15–20% annualized** in regime-neutral conditions.

**Ablations from follow-up sweep:**
- Drop GBM, rank by raw `pred_day_plus_5`: Sharpe drops to **0.42**, sleeve +11.40%. The GBM is doing real work — cross-horizon and regime-feature interactions are not replaceable by raw pred rank.
- Extend threshold sweep to {0.45..0.80}: Sharpe +0.03 (1.00 vs 0.97). Harmless but no new edge — strategy already finds the optimum in the legacy [0.45–0.65] range.
- High-only thresholds [0.65–0.80] alone: Sharpe drops to 0.66 (gate fires less often).

The strategy still **beats baseline at all cost levels tested** (5/15/30 bps round-trip), but the
margin is modest. Earlier "Sharpe 1.91 / +65% return" claims were inflated ~3–5× by a
calendar-day / trading-day mismatch in the lookahead gate.

## 0. Bug fix and corrected results (2026-05-17)

### The bug

`adaptive_calibrated_strategy.py:134` (now fixed) used `timedelta(days=realization_offset)`
to compute the cutoff before which training rows' labels must be realized. But
`realization_offset` is in **trading days** (e.g., 5 for `day_plus_5`), while
`timedelta(days=5)` is **5 calendar days**. Every weekend creates a 2-day leak; holidays
make it worse. The leak corrupted:

1. The GBM training set (last ~2 anchors per test day had unrealized labels)
2. The OOS-gate holdout window (same 2 anchors went into the held-out 3-day window)

In other words, the very mechanism designed to be the OOS gate was contaminated.

### The fix

Use the data's distinct dates as the trading calendar; cap admitted anchors by **trading-day
index**, not calendar-day delta:

```python
all_trading_dates = sorted(wide["date"].unique().to_list())
td_idx = {td: i for i, td in enumerate(all_trading_dates)}
...
d_idx = td_idx.get(d)
if d_idx is None or d_idx < realization_offset:
    continue
anchor_cap_realization = all_trading_dates[d_idx - realization_offset]
train = valid.filter(pl.col("date") < anchor_cap_realization)
```

Sharpe annualization also corrected from `sqrt(252)` (treats overlapping 5-day positions
as independent daily picks) to `sqrt(252 / realization_offset) ≈ 7.10` for d5.

### Before vs after (K=2, N=10, holdout=3, cost=5 bps)

| Metric | Before fix (claimed) | After fix (honest) | Inflation |
|--------|----------------------:|--------------------:|----------:|
| Sharpe | 1.91 | 0.56 | 3.4× |
| Mean bps/day (all) | +83.81 | +20.66 | 4.1× |
| Sleeve return (14mo) | +65.14% | +12.87% | 5.1× |
| Trade rate | 62% | 59% | — |
| Hit rate (traded) | 64% | 55% | — |

### Honest K sweep (corrected gate, N=10, holdout=3, GBM, d5)

**Cost = 5 bps (legacy / optimistic):**
| K | Traded | Mean_all | Sharpe | Sleeve | Hit |
|---|-------:|---------:|-------:|-------:|----:|
| **1** ★ | 62% | +39.00 | +0.76 | **+25.64%** | 54% |
| 2 | 59% | +20.66 | +0.56 | +12.87% | 55% |
| 3 | 52% | +19.72 | +0.63 | +12.35% | 52% |
| 5 | 49% | +10.79 | +0.40 | +6.52%  | 55% |

**Cost = 15 bps (realistic NSE delivery):**
| K | Traded | Mean_all | Sharpe | Sleeve | Hit |
|---|-------:|---------:|-------:|-------:|----:|
| **1** ★ | 62% | +32.33 | **+0.63** | **+20.67%** | 53% |
| 2 | 58% | +13.86 | +0.38 | +8.32% | 53% |
| 3 | 51% | +12.78 | +0.41 | +7.74% | 51% |
| 5 | 48% | +5.46 | +0.21 | +3.14% | 54% |

**Cost = 30 bps (conservative):**
| K | Traded | Mean_all | Sharpe | Sleeve | Hit |
|---|-------:|---------:|-------:|-------:|----:|
| **1** ★ | 60% | +20.03 | +0.41 | +12.07% | 51% |
| 2 | 57% | +8.27 | +0.23 | +4.72% | 52% |
| 3 | 49% | +6.36 | +0.21 | +3.63% | 49% |
| 5 | 47% | +0.95 | +0.04 | +0.36% | 53% |

### Baseline comparison (top-K every day, no gate, corrected)

| Strategy | Cost | Sharpe | Sleeve | Hit |
|----------|-----:|-------:|-------:|----:|
| Baseline K=2 | 5 bps | +0.22 | +6.14% | 49% |
| Baseline K=2 | 15 bps | +0.02 | -0.08% | 47% |
| Baseline K=2 | 30 bps | -0.27 | -8.74% | 45% |
| Baseline K=7 | 5 bps | +0.55 | +13.50% | 53% |
| Baseline K=7 | 15 bps | +0.30 | +6.85% | 51% |
| Baseline K=7 | 30 bps | -0.08 | -2.40% | 49% |
| **GBM K=1 (honest)** | **15 bps** | **+0.63** | **+20.67%** | **53%** |
| **GBM K=1 (honest)** | **30 bps** | **+0.41** | **+12.07%** | **51%** |

The GBM gate **still adds value over baseline at all cost levels** — at 15 bps, K=1 GBM
posts Sharpe 0.63 vs baseline K=7's 0.30. The improvement is real, just not the 7×
multiplier earlier claimed.

### Other bugs identified by reviewers (not yet quantified)

- **Contrastive pair mining uses realized future returns** to define positive pairs
  (`cohort_pair_miner.py:154-204`, `mine_sector_alpha_pairs`). Encoder is label-conditioned.
  Bounded by `train_end=2024-11-30` so not a direct OOS leak, but generalization
  guarantees weaker than "pure self-supervised" framing implies.
- **Cost model 5 bps round-trip optimistic for NSE delivery**: real ≈ STT 20 bps + ~5
  exchange/GST + ~5–15 slippage + ~5 brokerage = **30–50 bps**. Recommended baseline:
  15 bps minimum; 30 bps for conservative planning.
- **P&L formula** averages log-returns then converts to simple return (small bias).
- **Threshold-sweep retrain** retrains the final model on the same holdout window used
  for threshold selection. Modest residual leakage; harder to quantify.

---

---

## 1. End-to-end pipeline

```
Raw NSE OHLCV (2019 – present)
    │
    ▼  Stage 1 — Contrastive pretraining (GPU required)
StockContextEncoder pretrained weights
    │
    ▼  Stage 2 — Cohort model fine-tuning (GPU required)
v11 Cohort checkpoint  runs/cohort_modeA_v11_from_contrastive/
    │
    ▼  Stage 3 — Static prediction dump (GPU optional, CPU feasible)
reports/v11_static_predictions_2025_2026.parquet
    │
    ▼  Stage 4 — GBM adaptive calibration (CPU only)
Daily stock picks (K=2 per day, 62% frequency)
```

---

## 2. Stage 1 — Contrastive pretraining

**Script:** `daity/scripts/pretrain_contrastive.py`  
**GPU:** Required (H200 ~2–4h at 4,000 steps; A100 ~4–6h)  
**Data:** NSE OHLCV 2019-01-01 to 2024-11-30, multi-resolution (5m/15m/60m/day)

### Architecture

- **Two-tower InfoNCE** (symmetric, temperature τ=0.1)
- **Shared encoder:** `StockContextEncoder` — multi-resolution patch tokenizer → RevIN → Transformer backbone (d_model=480, 1 backbone layer, 8 heads)
- **Projection head:** Linear(d→d) → GELU → Linear(d→proj_dim=128) → L2-norm
- **Pairs:** Same stock on two randomly-offset anchor dates (same regime window) = positive pair; all other stocks in batch = negatives
- **Loss collapses at batch=8** (known issue at 3-loss phase 2 config; resolved by honest disabling of contrastive head in the Phase-2 pretrain and re-enabling here with real pairs)

### Output

Pretrained encoder checkpoint loaded by `train_cohort.py` via `--init-ckpt`.

### Command

```bash
python -m daity.scripts.pretrain_contrastive \
    --pairs-path runs/pair_dataset_v1/pairs.parquet \
    --train-start 2019-01-01 --train-end 2024-11-30 \
    --universe-end 2024-12-31 \
    --batch-pairs 64 --max-steps 4000 --lr 3e-4 \
    --tau 0.1 --proj-dim 128 \
    --log-dir runs/contrastive_pretrain_v1
```

---

## 3. Stage 2 — Cohort model fine-tuning

**Script:** `daity/scripts/train_cohort.py`  
**GPU:** Required (H200 ~2–4 days for full walk-forward; A100 longer)  
**Init:** Contrastive pretrained encoder from Stage 1

### Architecture (`daity/models/cohort/`)

| Component | File | Role |
|-----------|------|------|
| `StockContextEncoder` | `stock_encoder.py` | Per-stock multi-resolution patch embedder + Transformer |
| `MarketContextTransformer` | `market_context.py` | Attends over all stocks to build market-level context |
| Cross-attention layers | `cross_attn.py` | Each stock attends to market context |
| FiLM conditioning | `film_conditioning.py` | Injects regime features into hidden states |
| Top-K attention | `top_k_attn.py` | Sparse attention for large stock universes |
| `CohortModel` | `cohort_module.py` | Full end-to-end module |
| Multi-horizon head | `cohort_heads.py` | 10-horizon prediction head |
| Contrastive loss | `cohort_loss.py` | Auxiliary contrastive loss during fine-tune |

### Training modes

| Mode | Description | Data range |
|------|-------------|-----------|
| A | Walk-forward (train on historical, predict next window) | 2019 → 2025 |
| B | Short-window fine-tune on recent months | 2024–2025 |
| C | Cross-sectional ranking loss only | 2024–2025 |

Walk-forward dataset: `daity/data/cohort_dataset.py` (`CohortAssembler`).  
Pair mining for contrastive: `daity/data/cohort_pair_dataset.py`, `cohort_pair_miner.py`.

### Output horizons (10 total)

Defined in `daity/data/intraday_horizons.py`:

| Horizon | Description | Realization offset |
|---------|-------------|-------------------|
| `intraday_30m` | 30-min return from anchor | 0 days |
| `intraday_60m` | 60-min return | 0 days |
| `intraday_120m` | 2h return | 0 days |
| `intraday_180m` | 3h return | 0 days |
| `to_close` | Return to market close (15:30 IST) | 0 days |
| `overnight` | Overnight gap (close → next open) | 1 day |
| `next_day_1h` | Return in first hour of next day | 1 day |
| `next_day_eod` | Next-day end-of-day return | 1 day |
| `day_plus_3` | 3-day compounded return | 3 days |
| `day_plus_5` | 5-day compounded return ← **strategy target** | 5 days |

### Champion checkpoint

`runs/cohort_modeA_v11_from_contrastive/`  
Training: two-tower contrastive pretrain → Mode A walk-forward regression+rank  
First cohort model to show positive OOS P&L on the full 2025 test set.

### Command (Mode A)

```bash
python -m daity.scripts.train_cohort \
    --init-ckpt runs/contrastive_pretrain_v1/best.ckpt \
    --cache-root data/cache \
    --feature-root data/features_parquet \
    --train-start 2019-01-01 --train-end 2024-11-30 \
    --universe-end 2024-12-31 \
    --mode A --d-model 480 --n-market-layers 4 --n-cross-layers 4 \
    --log-dir runs/cohort_modeA_v11_from_contrastive
```

---

## 4. Stage 3 — Static prediction dump

**Script:** `daity/scripts/dump_static_predictions.py`  
**GPU:** Optional. CUDA used if available; falls back to CPU (slow: ~2–4h vs ~30 min on GPU)  
**Input:** Frozen v11 checkpoint  
**Output:** `reports/v11_static_predictions_2025_2026.parquet`

### Output schema

```
date        : Date       — trading day (NSE calendar)
anchor_us   : Int64      — anchor timestamp (μs, 15:25 IST = 10:00 UTC - 5min)
stock       : String     — NSE symbol
horizon     : String     — one of the 10 horizons above
pred_lr     : Float64    — predicted log-return
real_lr     : Float64    — realized log-return (NaN if not yet realized)
```

64,880 rows covering Feb 2025 – Apr 2026 (303 trading days × ~215 stocks × 10 horizons).

### Command

```bash
python -m daity.scripts.dump_static_predictions \
    --ckpt runs/cohort_modeA_v11_from_contrastive/best.ckpt \
    --cache-root data/cache \
    --feature-root data/features_parquet \
    --start 2025-02-01 --end 2026-04-30 \
    --out reports/v11_static_predictions_2025_2026.parquet
```

---

## 5. Stage 4 — GBM Adaptive Calibration Strategy

**Script:** `daity/scripts/adaptive_calibrated_strategy.py`  
**GPU:** None — fully CPU-only (sklearn GBM)  
**Input:** `reports/v11_static_predictions_2025_2026.parquet`  
**Dependencies:** `scikit-learn`, `polars`, `numpy` — no PyTorch/CUDA

This is the component you run **after shutting down the GPU instance**. All further strategy experimentation requires only the static prediction parquet.

### Algorithm (per test day D)

```
1. ASSEMBLE TRAINING DATA
   Collect all (anchor, stock) rows whose labels have realized by D:
   - intraday/to_close rows: anchor_date + 0 days < D
   - next_day_* rows:        anchor_date + 1 day  < D
   - day_plus_3 rows:        anchor_date + 3 days < D
   - day_plus_5 rows:        anchor_date + 5 days < D   ← target horizon

2. FEATURE ENGINEERING
   Per (anchor_date, stock) row:
     - 5 horizon predictions: pred_{h} for h in {to_close, next_day_1h,
       next_day_eod, day_plus_3, day_plus_5}
     - Universe regime: univ_mean_pred_{h}, univ_std_pred_{h} per day
       (cross-sectional mean/std of all stocks' predictions for that day)
   Total: 15 features per row

3. BINARY LABEL
   win = 1  if  real_day_plus_5 > 30 bps  else  0

4. HOLDOUT SPLIT (OOS gate — CRITICAL)
   Take the last (N + holdout_days) anchor dates before the label-realization cap:
     fit_dates  = anchor_dates[-(N + holdout_days) : -holdout_days]   ← N days
     hold_dates = anchor_dates[-holdout_days:]                         ← holdout_days days
   Train GBM on fit_dates only → produce calibrated win probabilities.

5. THRESHOLD SWEEP (on held-out days only)
   For each threshold t ∈ {0.45, 0.50, 0.55, 0.60, 0.65}:
     On each hold_date: pick top-K stocks with prob > t; compute P&L.
   Select threshold t* with highest mean P&L on hold_dates.

6. GATE
   If best mean held-out P&L ≤ 0 → sit out (record 0 bps for day D).
   If > 0 → continue.

7. FINAL MODEL
   Retrain GBM on all (N + holdout_days) anchor dates.

8. PICK STOCKS FOR D+1
   Apply final model + threshold t* to today's predictions.
   Long top-K stocks by descending win probability.
   Hold for 5 trading days.
```

### Champion hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target horizon | `day_plus_5` | 5-day compounded return |
| K | **2** | Stocks per trading day |
| N | **10** | GBM fit window (trading days) |
| holdout_days | **3** | OOS gate evaluation window |
| win_threshold_bps | 30 | Binary label threshold |
| cost_bps | 5 | Round-trip transaction cost |
| GBM n_estimators | 50 | Gradient boosting trees |
| GBM max_depth | 3 | Tree depth |
| GBM learning_rate | 0.1 | Shrinkage |
| Prob thresholds swept | 0.45–0.65 (step 0.05) | 5 candidates |
| Hold period | 5 trading days | Matches day_plus_5 horizon |

### Calibration model comparison (K=3, N=10, holdout=3)

| Model | Trade% | Mean_all (bps) | Sharpe* | Sleeve (14mo) | Hit |
|-------|-------:|---------------:|--------:|--------------:|----:|
| LR    | 35%    | +17.25         | +1.26   | ~+12%         | 58% |
| RF    | 45%    | +11.59         | +0.31   | ~+4%          | 52% |
| **GBM** ★ | **54%** | **+53.92** | **+1.56** | **+38%** | **64%** |

GBM chosen: better regime-change tracking, non-linear interaction capture, stable OOS.

### K sweep (N=10, holdout=3, GBM, Feb'25–Apr'26, 303 days)

| K | Traded | Mean_all (bps) | Sharpe* | Sleeve (14mo) | Hit | Mean on traded |
|---|-------:|---------------:|--------:|--------------:|----:|---------------:|
| 1 | 64%    | +86.31         | +1.45   | +66.85%       | 58% | +134.11 bps    |
| **2** ★ | **62%** | **+83.81** | **+1.91** | **+65.14%** | **64%** | **+135.80 bps** |
| 3 | 54%    | +53.92         | +1.56   | +38.12%       | 64% | +100.23 bps    |
| 5 | 48%    | +26.09         | +0.93   | +16.84%       | 58% | +54.53 bps     |
| 7 | 47%    | +14.65         | +0.56   | +9.05%        | 55% | +31.48 bps     |
| 10| 47%    | +5.91          | +0.23   | +3.44%        | 53% | +12.61 bps     |

K=2 chosen: highest Sharpe (1.91) with best hit rate (64%) vs K=1's 58%.

### N sweep (K=2, holdout=3, GBM, Feb'25–Apr'26, 303 days)

| N | Traded | Mean_all (bps) | Sharpe* | Sleeve (14mo) |
|---|-------:|---------------:|--------:|--------------:|
| 5 | 61%    | +55.44         | +1.27   | +39.10%       |
| 8 | 59%    | +60.46         | +1.45   | +43.47%       |
| **10** ★ | **62%** | **+83.81** | **+1.91** | **+65.14%** |
| 12| 59%    | +59.18         | +1.38   | +42.31%       |
| 15| 52%    | +31.45         | +0.83   | +20.45%       |
| 20| 50%    | +42.30         | +1.12   | +28.32%       |
| 30| 49%    | +54.56         | +1.39   | +36.58%       |

N=10 chosen: clear winner on Sharpe and sleeve.

### Holdout sensitivity (K=3, N=10, GBM)

| Holdout | Traded | Mean_all (bps) | Sharpe* | Sleeve |
|--------:|-------:|---------------:|--------:|-------:|
| 2 | 53% | +23.21 | +0.67 | +14.68% |
| **3** ★ | **54%** | **+53.92** | **+1.56** | **+38.12%** |
| 5 | 52% | +71.55 | +2.10 | +53.69% |

Note: holdout=5 is better at K=3 but worse at K=2 (1.73 vs 1.91). Champion uses K=2/holdout=3.

*Sharpe uses sqrt(252/5) annualization (5-day independent hold periods).

### Capital accounting (CRITICAL — no double-counting)

A 5-day hold with **daily** entry creates 5 overlapping sleeves. Naïve compounding
`prod(1 + bps/10000)` overstates returns by ~5x (implies 5× leverage).

**Correct sleeve-fragmented accounting** (no leverage, 1 account):
- Allocate 1/5 of capital to each "sleeve" (day 1 positions, day 2 positions, ..., day 5 positions)
- Each sleeve compounds at `prod(1 + bps/10000/5)` where bps/5 = single-sleeve contribution
- Total return = `prod(1 + bps/10000/5) - 1` applied to all test days

| K | Mean bps/day | Sleeve-adjusted return (14 months) | Annualized |
|---|-------------:|-----------------------------------:|-----------|
| 1 | +86.31 | +66.85% | ~57% |
| **2** ★ | **+83.81** | **+65.14%** | **~56%** |
| 3 | +53.92 | +38.12% | ~33% |
| 7 (baseline) | +14.65 | +9.05% | ~8% |

### Baseline comparison

| Strategy | Mean bps/day | Sharpe | Sleeve (14mo) |
|----------|-------------:|-------:|---------------|
| Top-K=7 every day (no gate) | +14.65 | +0.56 | +9.05% |
| Top-K=2 every day (no gate) | ~+30 | ~+0.7 | ~+20% |
| **GBM gated K=2 N=10** ★ | **+83.81** | **+1.91** | **+65.14%** |

The GBM gate filters ~38% of days where calibration lookback was unprofitable, turning those into 0-bps (capital parked elsewhere).

---

## 6. Running the strategy (CPU-only, no GPU)

All of Stage 4 runs on any modern laptop. The static predictions parquet is the only artifact needed from the GPU phase.

### Prerequisites

```bash
cd /path/to/daity
uv sync          # or: pip install scikit-learn polars numpy
```

### Champion run

```bash
python -m daity.scripts.adaptive_calibrated_strategy \
    --predictions-path reports/v11_static_predictions_2025_2026.parquet \
    --test-start 2025-02-01 --test-end 2026-04-30 \
    --target-horizon day_plus_5 \
    --clf-type gbm \
    --top-k 2 \
    --n-sweep 10 \
    --holdout-days 3 \
    --win-threshold-bps 30 \
    --cost-bps 5 \
    --out-prefix reports/gbm_k2_n10_h3
```

Runtime: ~85 seconds per N value on a 4-core laptop.  
Output: `reports/gbm_k2_n10_h3_N10.parquet` (per-day P&L) + `_summary.parquet`.

### Output columns

```
test_date           : Date      — trading day D
N                   : Int       — lookback window used
chosen_threshold    : Float     — prob threshold that passed the OOS gate (null = sit out)
lookback_pnl_bps    : Float     — mean P&L on holdout window (gate signal)
lookback_sharpe     : Float     — Sharpe on holdout window
n_lookback_trades   : Int       — number of holdout days with trades
n_today_trades      : Int       — 0 = sat out, K = traded
today_pnl_bps       : Float     — realized P&L for day D (0 = sat out)
```

---

## 7. GPU dependency map

| Stage | Script | Needs GPU? | Can run on CPU? | Notes |
|-------|--------|-----------|-----------------|-------|
| 1. Contrastive pretrain | `pretrain_contrastive.py` | Yes | No (too slow) | ~2–4h on H200 |
| 2. Cohort fine-tune | `train_cohort.py` | Yes | No (too slow) | ~2–4 days on H200 |
| 3. Prediction dump | `dump_static_predictions.py` | Preferred | Yes (~2–4h) | One-time per ckpt |
| **4. GBM strategy** | **`adaptive_calibrated_strategy.py`** | **No** | **Yes (~85s)** | **Daily use** |
| Eval/analysis | `eval_cohort_pnl.py`, `conformal_calibration.py` | No | Yes | Post-hoc only |

**To shut down the GPU instance:** pull these four artifacts first, then the instance can be destroyed:

| Artifact | Size | Local path | Status |
|----------|------|-----------|--------|
| `reports/v11_static_predictions_2025_2026.parquet` | 2 MB | `reports/` | ✅ local |
| `runs/pair_dataset_v1/` (pairs + facts) | 9 MB | `runs/pair_dataset_v1/` | ✅ local |
| `runs/contrastive_pretrain_v1/checkpoints/last=2000.pt` | 13 MB | `runs/contrastive_pretrain_v1/checkpoints/` | ✅ local |
| `runs/cohort_modeA_v11_from_contrastive/checkpoints/best.pt` | 248 MB | `runs/cohort_modeA_v11_from_contrastive/checkpoints/` | ✅ local |

All pulled as of 2026-05-17. **GPU instance can be shut down.** Stages 1–3 are complete for v11. Stage 4 (daily strategy) runs locally with no GPU.

---

## 8. Known limitations and next steps

1. **Static predictions:** The v11 checkpoint is frozen. If the market regime shifts, predictions degrade. Next: re-run contrastive pretrain + cohort fine-tune quarterly.

2. **Transaction cost sensitivity:** 5 bps assumed. If actual costs are higher (e.g., 10 bps), K=1 mean of +86 bps still clears; K=2 at +83 bps clears; both remain profitable.

3. **Single-stock concentration (K=2):** On any given day, K=2 means concentrated single-name risk. Tail of individual stock days: likely high (~1–2% daily swings). The hit rate of 64% is the primary risk buffer.

4. **No exit rule:** Strategy assumes a clean 5-day hold from entry. A mid-period exit (e.g., stop-loss at -200 bps intraday) is not modeled and could improve or hurt results depending on implementation.

5. **Universe size:** ~200 liquid NSE stocks. Expanding the universe or reweighting by liquidity tier is an open experiment.

6. **Walk-forward validation:** The GBM is refitted daily on N=10 days. For a rolling 6-month forward test from 2026-05-01, run `dump_static_predictions.py` on new data and rerun Stage 4.

---

## 9. File inventory

```
daity/scripts/
  adaptive_calibrated_strategy.py    ← Stage 4 (champion strategy, CPU-only)
  pretrain_contrastive.py            ← Stage 1
  train_cohort.py                    ← Stage 2
  dump_static_predictions.py         ← Stage 3
  eval_cohort_pnl.py                 ← OOS P&L evaluation
  conformal_calibration.py           ← Conformal prediction intervals
  online_walk_forward.py             ← Live walk-forward (not used for static)
  winrate_calibration.py             ← Win-rate calibration analysis

daity/models/cohort/
  cohort_module.py                   ← CohortModel (full model)
  stock_encoder.py                   ← StockContextEncoder
  market_context.py                  ← Market-level transformer
  cross_attn.py                      ← Cross-attention (stock ↔ market)
  film_conditioning.py               ← Regime feature injection
  top_k_attn.py                      ← Sparse attention
  cohort_heads.py                    ← Multi-horizon prediction head
  cohort_loss.py                     ← Contrastive + regression losses

daity/data/
  cohort_dataset.py                  ← CohortAssembler (data pipeline)
  cohort_pair_dataset.py             ← Contrastive pair dataset
  cohort_pair_miner.py               ← Pair mining logic
  intraday_horizons.py               ← HORIZONS enum (10 prediction targets)

reports/
  v11_static_predictions_2025_2026.parquet  ← Stage 3 output (gitignored, keep local)
  liquid_universe_100stocks.txt             ← Universe definition
```
