# Reproduction guide — Autoloop champion (2026-05-19)

End-to-end recipe to reproduce the **9-model 1d-target focal+small ensemble** that produced rolling-anchor Sharpe +1.13 / sleeve +24.94% (chunky rebal) and Sharpe +2.17 / sleeve +42.5% (rolling-entry, threshold=0.002).

## 0. Prerequisites

- **Hardware**: 1 GPU with ≥ 16 GB VRAM (any modern card; the model is ~75K params so even a T4 is fine). Training runs use ~1.5 GB VRAM each, so a single 40 GB GPU can run 6-9 in parallel comfortably.
- **OS / CPU**: enough fork() headroom for parallel runs — set `ulimit -u` to at least 4096 if running 8+ models concurrently. We hit the default 1024 limit twice during the autoloop.
- **Python**: 3.11
- **Env**: `uv` venv at `/root/daity/.venv/` (or local equivalent). Required packages: `torch`, `polars`, `numpy`, `scipy`, `matplotlib`, `wandb` (optional logging).
- **Repo branch**: `phase-2.4-cross-attention-forecast-head` at commit `5a66a28` or later.

## 1. Data inputs

### Features parquet
`/root/daity/data/lightweight_features.parquet` — must contain:
- Columns: `date`, `stock` (string), `label_next_day_eod` (1-day log-return label), `sector_id`, `stock_id`, plus 64 numerical engineered features (lag returns, momentum, vol, calendar).
- Coverage: ~205 NSE equities + ETFs, 2019-06-26 through latest train cutoff.
- Built by `/tmp/build_lightweight_features.py` (not in repo — schema is documented in `daity/models/lightweight/ft_set_transformer.py:200-250`).

### Multi-horizon labels (optional, only needed if exploring 10d-target wave)
`/root/daity/data/lightweight_features_mh.parquet` — built by `scripts/add_multihorizon_labels_v2.py` (referenced from `/tmp/add_multihorizon_labels_v2.py`).

Adds: `label_5d_cumul`, `label_7d_cumul`, `label_10d_cumul`, `label_12d_cumul`, `label_15d_cumul`, `label_20d_cumul` — each is `log(close[d+N] / close[d])` computed as a per-stock rolling sum of `label_next_day_eod`.

### Order-book parquets
`/Users/debdoot/Documents/Claude/Projects/daity/data/ob_parquet/<SYMBOL>.parquet` — one per stock, 27 columns: top-5 bid/ask prices + qty + LTP + volume + total_buy/sell_qty + spread_bps. Date range 2026-03-09 to 2026-05-04. Used only for execution validation, not training.

## 2. Model architecture

`daity/models/lightweight/ft_set_transformer.py` — `LightweightCohortModel`.

- **FT-Transformer per stock** (`d_ft=16`, `ft_layers=2`): tokenises each of 64 numerical features into a `d_ft=16` token, runs a 2-layer transformer encoder over the (66 = 64 num + 2 cat) tokens, takes the CLS pooled output. Output: `(B, N_stocks, d_ft)` projected to `d_model=64`.
- **Set Transformer across stocks** (`d_model=64`, `set_layers=2`): cross-stock attention over N_stocks tokens. Output: `(B, N_stocks, d_model)`.
- **Regression head**: `Linear(d_model, n_horizons=1)` → per-stock next-day log-return prediction.
- **Classifier head**: `Linear(d_model, n_horizons=1)` → per-stock binary direction logit (positive next-day return).
- **Optional Portfolio head** with STE top-K (unused in this loop).

Total parameters: **~75 K** at d_ft=16, d_model=64 (the "small" config).

## 3. Training recipe — single run

The lightweight model is trained walk-forward via `/tmp/online_wf_lightweight.py` (also reachable from the lightweight model dir).

For ONE of the 9 ensemble members, e.g., `focal+small + 1ep + seed=42`:

```bash
.venv/bin/python /tmp/online_wf_lightweight.py \
  --features /root/daity/data/lightweight_features.parquet \
  --online-start 2019-06-26 \
  --oos-start-date 2025-05-01 \
  --online-end 2026-04-30 \
  --n-epochs 1 \
  --batch-days 16 \
  --buffer-max-days 252 \
  --steps-per-day 3 \
  --label-lag-trading-days 2 \
  --ckpt-every 50 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --warmup-steps 200 \
  --label-clip 0.20 \
  --w-reg 1.0 \
  --w-clf 1.0 \
  --w-focal 2.0 \
  --w-sharpe 0.0 \
  --d-ft 16 \
  --d-model 64 \
  --seed 42 \
  --log-dir runs/lightweight_wf_iter25_focal_small
```

Two training phases:

- **TRAIN-MULTI** (multi-epoch on past): days 2019-06-26 → 2025-04-30 (~1700 train days), 1 pass at lr=1e-4 with 200-step linear warmup. Each day's batch combines 16 most-recent train days (current observation) + 16 randomly sampled older days from the 252-day replay buffer. 3 gradient steps per day.
- **OOS** (single pass): days 2025-05-01 → 2026-04-30 (~248 days). Predictions written to `predictions.parquet`. The model *does* continue to update weights during OOS via online walk-forward — but only ONCE per day, single pass.

Loss = `1.0 × SmoothL1(reg_pred, label_next_day_eod_clipped) + 1.0 × BCEFocal(clf_pred, sign(label_next_day_eod), γ=2)`. The `--w-focal=2.0` flag enables focal weighting with γ=2 (standard).

**Key knobs** (all from sweep, do not change):
- `--lr 1e-4` (5e-4 overfits, 5e-5 loses signal)
- `--n-epochs 1` (2 epochs = good for ensembling but lowers solo Sharpe due to less ETF-padding)
- `--d-ft 16 --d-model 64` (sweet spot; 32/128 default overfits, 8/32 tiny under-fits)
- `--w-focal 2.0` (1.5 worse, 4.0 worse)
- `--ft-layers 2 --set-layers 2` (defaults; 1 layer = IC-negative, 3 layers = IC-zero)
- `--label-lag-trading-days 2` (point-in-time correctness)
- `--label-clip 0.20` (single-day return capped at ±20% before loss)

Runtime: **~75 sec single-run on H100**, ~5-10 min when running 8 in parallel on a 40 GB instance.

## 4. The 9-model ensemble — what to train

To reproduce `runs/autoloop_predictions/ensemble_9_models.parquet` you need **9 separate training runs** = 3 configs × 3 seeds. Same flags as above, only these three knobs vary:

| Run | Config | Seed | Flags differing from base |
|---|---|---|---|
| 1 | 1ep, lr=1e-4 | 42 | (base) |
| 2 | 1ep, lr=1e-4 | 7 | `--seed 7` |
| 3 | 1ep, lr=1e-4 | 123 | `--seed 123` |
| 4 | 2ep, lr=1e-4 | 42 | `--n-epochs 2` |
| 5 | 2ep, lr=1e-4 | 7 | `--n-epochs 2 --seed 7` |
| 6 | 2ep, lr=1e-4 | 123 | `--n-epochs 2 --seed 123` |
| 7 | 1ep, lr=1.5e-4 | 42 | `--lr 1.5e-4` |
| 8 | 1ep, lr=1.5e-4 | 7 | `--lr 1.5e-4 --seed 7` |
| 9 | 1ep, lr=1.5e-4 | 123 | `--lr 1.5e-4 --seed 123` |

All runs write to separate `runs/lightweight_wf_<tag>/predictions.parquet`. Each run produces ~570 KB of `(date, stock, pred_lr, real_lr)` rows.

## 5. Build the 9-model ensemble

`scripts/seed_ensemble.py` averages `pred_lr` across the 9 prediction parquets:

```bash
.venv/bin/python scripts/seed_ensemble.py \
  --inputs runs/autoloop_predictions/lightweight_wf_iter25_focal_small.parquet \
           runs/autoloop_predictions/iter46_focal_small_seed7.parquet \
           runs/autoloop_predictions/iter47_focal_small_seed123.parquet \
           runs/autoloop_predictions/focal_small_2ep_seed42.parquet \
           runs/autoloop_predictions/focal_small_2ep_seed7.parquet \
           runs/autoloop_predictions/focal_small_2ep_seed123.parquet \
           runs/autoloop_predictions/focal_small_lr15e5_seed42.parquet \
           runs/autoloop_predictions/focal_small_lr15e5_seed7.parquet \
           runs/autoloop_predictions/focal_small_lr15e5_seed123.parquet \
  --output runs/autoloop_predictions/ensemble_9_models.parquet \
  --label "9-model: focal+small × {1ep,2ep,lr1.5e-4} × {3 seeds}"
```

Expected output: `IC mean: +0.0114, IR: +1.19` (verifies the parquets matched the training above).

## 6. Evaluation — rolling-anchor backtest

The single-offset `score_iter.py` is misleading (offset=0 luck — iter10 looked +3.20 but the rolling mean was +0.77). Always use:

```bash
.venv/bin/python scripts/score_rolling_anchor.py \
  runs/autoloop_predictions/ensemble_9_models.parquet \
  "ensemble_9_models"
```

For each `(hold, K)` combination it runs the chunky-rebal backtest at all `hold` offsets (0, 1, ..., hold-1) and reports `mean ± std`, `min`, `max` across offsets.

**Expected key numbers**:

| hold | K | mean Sharpe | std | min | max | sleeve_mean |
|---|---|---|---|---|---|---|
| 10 | 10 | **+1.13** | 0.48 | +0.50 | +1.84 | +24.94% |
| 10 | 5 | +1.03 | 0.47 | +0.09 | +1.59 | +23.15% |
| 10 | 20 | +0.77 | 0.26 | +0.40 | +1.14 | +14.78% |
| 5 | 10 | +0.68 | 0.55 | -0.16 | +1.24 | +14.48% |
| 20 | 5 | +0.65 | 0.94 | -1.51 | +2.05 | +13.05% |

If your reproduction's `hold=10 K=10` mean is within ±0.1 of +1.13, the model trained correctly. Larger deviations mean either (a) a different feature parquet, (b) a different code path in `/tmp/online_wf_lightweight.py`, or (c) different PyTorch version causing non-determinism on the GPU.

## 7. Evaluation — rolling-entry strategy (the production strategy)

This is the strategy with the headline +42.5% sleeve. Daily threshold-gated entries with rolling per-position 10-day exits:

```bash
.venv/bin/python scripts/rolling_entry_strategy.py \
  runs/autoloop_predictions/ensemble_9_models.parquet \
  --max-pos 10 \
  --hold 10 \
  --threshold 0.002 \
  --top-n 50 \
  --cost-bps 30 \
  --out reports/rolling_entry_ens9_thr0.002.png
```

**Expected key numbers**:

| Metric | Value |
|---|---|
| Sleeve (net) | **+42.45%** |
| Annualised Sharpe | **+2.17** |
| Max drawdown | **−5.68%** |
| Positions opened | 158 |
| Positions closed | 158 |
| Position hit-rate | 60.8% |
| Mean position return | +2.39% |
| Mean active fill | 6.4 / 10 |

Threshold=0.002 (20 bps min predicted 1-day log-return) is critical — at threshold=0 sleeve drops to +21%, at threshold=0.005 it drops to +12%. The threshold acts as an implicit regime gate.

## 8. Evaluation — OB execution simulation

Validates that picks are actually tradeable at production size (₹2L–₹50L per pick) using top-5 ask depth.

```bash
.venv/bin/python scripts/ob_paper_pnl_partial.py \
  runs/autoloop_predictions/ensemble_9_models.parquet \
  --K 10 \
  --hold 10 \
  --pos-inr 200000 \
  --all-offsets \
  --out reports/ob_paper_pnl_ensemble_K10_h10.png
```

Window: 2026-03-09 → 2026-05-04 (only days with OB data).

**Expected at ₹2L/pick**:
- 94% of fills within 30 bps slippage of top-of-book
- Mean OB-executed return per rebal: +4.44% (in window)
- OB realises **−14 bps better than the 30 bps assumed cost** (favourable)

**Capacity scaling** — repeat with `--pos-inr 1500000` (₹15L), `--pos-inr 2500000` (₹25L), `--pos-inr 5000000` (₹50L). Expected sleeves: 47%, 36%, 23% of target deployed; slippage -55, -65, -74 bps respectively.

## 9. Common pitfalls

- **Don't use `score_iter.py`** for headline numbers. It's the single-offset chunky rebal which is biased upward by offset choice. Use `score_rolling_anchor.py` for the honest mean.
- **Don't trust single-seed runs.** A specific seed can produce IC +0.017 or IC -0.002 for *identical* config (we observed this 3× in the autoloop). Always train ≥ 3 seeds and ensemble.
- **Don't train with `--target-label-col label_10d_cumul`** as a sole-head replacement. That kills cross-sectional ranking (iter15 and iter80+ both confirmed this). Use 1d label for training; multi-day evaluation is for the BACKTEST stage only.
- **Watch out for fork() limits** when running ≥ 9 parallel runs. Symptom: `pyo3_runtime.PanicException: could not spawn threads`. Raise `ulimit -u`, or split into smaller waves.
- **Label_lag_trading_days=2** is mandatory for point-in-time correctness. Without it, day-d prediction can see day-d EOD label which is forward-leaking.
- The 1-year OOS sample (2025-05-02 → 2026-04-30) **contains the strong early-2026 rally**. The +42.5% rolling-entry sleeve has regime-dependence; do not extrapolate to multi-year without revalidation.

## 10. Hash verification (after reproduction)

After your reproduction completes, the canonical `ensemble_9_models.parquet` should report:

```
=== ensemble_9_models ===
rows: 47,864  days: 248
IC mean: +0.0114  IR: +1.19
```

If your `IC mean` is within ±0.0015 of +0.0114 and `IR` within ±0.15 of +1.19, the reproduction matches.

## 11. GCS artifact location

All artifacts at: `gs://atomic-vault-476315-m3-tradiate/autoloop_2026_05_19/`

- `predictions/` — 28 parquets including individual seeds, ensembles, and the mega cross-target
- `scripts/` — 7 evaluation + ensemble scripts
- `reports/` — summary doc, results table, 4 charts

To pull the reference predictions:

```bash
gsutil cp gs://atomic-vault-476315-m3-tradiate/autoloop_2026_05_19/predictions/ensemble_9_models.parquet runs/autoloop_predictions/
```

Then run the rolling-anchor and rolling-entry scripts above to verify the reference numbers.

## 12. What's next (deferred work)

- **Multi-task model with 1d + 10d heads** — requires `n_horizons > 1` in `regression_head` Linear layer. Currently the model only outputs 1 regression scalar per stock. The 10d-target single-head experiments (iter80+) confirmed that augmenting > replacing is the right pattern.
- **Add 10d+1h label** — for execution-realistic predictions. Compute from intraday 60m parquets at `/root/daity/data/features_parquet/60m/`.
- **Multi-year OOS revalidation** — current sample is 12 months only.
- **Execution-aware backtest** — extend `ob_paper_pnl_partial.py` with TWAP + passive-limit-ladder simulation; produces a realistic slippage-vs-size curve.
- **Live-trading scaffolding** — implement the 5-component execution stack in `serve/` against Zerodha/Upstox API.
