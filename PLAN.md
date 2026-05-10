# daity_v3 — Foundation-Model Algo Trading System

**Scope**: NSE equities (cash + listed F&O underlyings), intraday (minutes–hours) and swing (days–weeks) horizons. Pretrain a time-series foundation model on `curated_ohlcv`, post-train it as a supervised forecaster, then fine-tune with order-book conditioning on the limited `order_book_depth` data.

**Decided constraints**
- Data lake: `atomic-vault-476315-m3.daity_v3.*` in BigQuery.
- Compute: BQ for storage/queries, training on local/own GPU (likely H100 or 2×A100 80GB tier).
- Modeling philosophy: SSL-pretrained patch transformer backbone, supervised heads, LoRA adapters for the order-book fine-tune.
- Excluded for now: pure sub-minute HFT, multi-asset generalization beyond NSE.

---

## 1. System Goals & Non-Goals

**Goals**
1. A single backbone that learns transferable representations from the entire NSE universe of OHLCV history.
2. Strong cross-sectional ranking signals at the chosen horizons (intraday 5–60 min; swing 1–10 days).
3. Effective use of the small order-book sample as a *conditioning* signal that sharpens short-horizon predictions, without overfitting.
4. Honest backtesting that respects NSE frictions, calendar, corporate actions, and survivorship.
5. A clean path from research notebook → reproducible training run → live/paper trading.

**Non-goals (v1)**
- True tick-level / sub-second microstructure trading.
- Options pricing and Greeks-based strategies (F&O underlyings only as cash equities).
- Cross-market arbitrage (BSE, US, crypto).

---

## 2. Data Layer

### 2.1 Tables (assumed; will validate during Phase 1)
- `daity_v3.curated_ohlcv` — bar data. Expected schema includes `(symbol, ts, timeframe, open, high, low, close, volume, ...)`. We'll confirm timeframes available and adjustment status.
- `daity_v3.order_book_depth` — L2 snapshots or events. Expected `(symbol, ts, side, level, price, size, ...)`. Volume and time coverage TBD — this drives whether order-book is a fine-tune signal vs. just an evaluation overlay.

### 2.2 Ingestion & cache
- BigQuery Storage Read API (Arrow) → Parquet partitioned by `(symbol, year, month)` on local SSD.
- DuckDB on top of Parquet for ad-hoc analytics; PyTorch `IterableDataset` reads Parquet directly via PyArrow.
- All reads parameterized by `as_of` to prevent leakage in walk-forward training.

### 2.3 NSE-specific data quality work
- Symbol master with ISINs to handle ticker changes and merge histories.
- **Survivorship**: pull delisted symbols from `curated_ohlcv` if present; flag and exclude survivorship-only symbol lists in evaluation.
- **Corporate actions**: split / bonus / dividend adjustment. Verify whether `curated_ohlcv` is already adjusted; if not, build adjustment factors.
- **Calendar**: NSE trading-day calendar (holidays, muhurat, special sessions), session segments (pre-open 09:00–09:15, regular 09:15–15:30, post-close).
- **Circuit filters & halts**: detect from OHLCV (zero-range bars, no volume) and mask appropriately.
- **F&O expiry effects**: monthly/weekly expiry markers as features.

### 2.4 Feature engineering
- **Raw channels**: O, H, L, C, V, plus VWAP if computable, returns at multiple lags.
- **Microstructure-lite from OHLCV**: bar range, body/wick ratios, volume z-scores, intrabar volatility.
- **Cross-sectional**: sector / industry id (one-hot embed), market-cap bucket, free-float bucket — slow-moving, used as static covariates.
- **Time covariates**: minute-of-day, day-of-week, days-to-expiry, days-from-event (results, RBI policy if available later).
- **Reversible Instance Normalization (RevIN)** per (symbol, window) — critical for non-stationary equity series.

### 2.5 Splits
- Walk-forward / expanding window. E.g. for swing: train through 2022-12, validate 2023, test 2024+, then roll.
- **Embargo / purging** of length ≥ max forecast horizon between train/val and val/test to kill overlap leakage.
- Cross-symbol splits *not* used (don't hold out symbols) — universe is fixed; we hold out *time*.

---

## 3. Modeling

### 3.1 Backbone — patch-based time-series transformer
- Inspired by PatchTST + Chronos + MOMENT. Channel-independent: each numeric channel (O,H,L,C,V,returns,etc.) tokenized separately, sharing weights.
- **Input**: window of `L` bars (e.g. 512 minute-bars or 128 daily-bars), patched into chunks of `P` (e.g. 16) → ~32 tokens per channel.
- **Multi-resolution**: parallel input streams at 1m / 5m / 15m / 1h / 1d. Streams share the backbone but get a learned resolution embedding.
- **Architecture knobs (initial)**: `d_model=384`, `n_heads=6`, `n_layers=8`, dropout 0.1, RoPE on time axis, FlashAttention 2.
- **Static covariates** (sector, mcap bucket) injected via prefix tokens.
- **Calendar tokens** (minute-of-day, expiry-distance) injected via additive embeddings on each patch.

### 3.2 Pretraining objectives (SSL)
Three losses, weighted:
1. **Masked patch reconstruction** (primary) — BERT-style masking of 30–40% patches, predict via lightweight MLP head on the patch embedding. MSE on normalized values.
2. **Next-patch prediction** (causal head with attention mask) — gives an autoregressive prior the heads can later exploit.
3. **Cross-sectional contrastive** — same-day-different-symbol vs. different-day same-symbol, InfoNCE. Encourages regime-aware representations.

Pretraining runs over the full NSE universe × full history available, multi-resolution. No labels.

### 3.3 Post-training — supervised forecasting heads
Backbone is frozen for the first epoch, then unfrozen with low LR. New heads attached at the `[FORECAST]` token (a learned summary token):
- **Quantile regression head**: predicts forward returns at a set of horizons (intraday: 5/15/30/60 min; swing: 1/3/5/10 d) with multiple quantiles (e.g. 0.1/0.25/0.5/0.75/0.9). Pinball loss.
- **Cross-sectional ranking head**: ListMLE / pairwise ranking loss on the median return — this is what most equity strategies actually need.
- **Auxiliary direction classifier**: 3-class (down/flat/up) with class-balanced cross-entropy.

Output a *distribution* per (symbol, horizon), not a point estimate. Strategy layer consumes quantiles directly for sizing.

### 3.4 Fine-tuning — order-book conditioning (the part that uses limited data)
- **Order-book encoder**: a small transformer that consumes the last K snapshots of L2 depth → bid/ask price-vs-mid, size, cumulative imbalance, weighted mid-price, microprice. Output a fixed-length context vector per timestamp.
- **Fusion**: cross-attention block in the upper layers of the backbone, queries are time-series tokens, keys/values from the order-book encoder. Gated so the model can ignore OB when it's stale or absent.
- **LoRA adapters** on backbone attention layers; full training on the OB encoder and fusion block. Total trainable params kept to single-digit % of the backbone — this is the lever against overfitting on the small dataset.
- **Curriculum**: only the intraday head trains in this stage; swing head is frozen.
- **Regularization**: heavy dropout in the fusion block, EMA of fine-tuned weights, early stop on val rank-IC.

### 3.5 Why this layering
Pretraining gets the model to *understand price dynamics* across the universe before it ever sees a label. Post-training turns that understanding into forecasts using all OHLCV labels available (much more data than OB). Fine-tuning then uses the precious OB data to correct only what it's actually informative about — short-horizon direction and impact — without disturbing the base.

---

## 4. Strategy & Evaluation Layer

### 4.1 Metrics (research-grade)
- **Rank IC** and **Rank IC decay** by horizon.
- **Top-bottom decile spread**, hit rate, asymmetric profit factor.
- **Distributional**: CRPS on the quantile head, calibration plots.

### 4.2 Backtest engine
- Vectorized Python backtester (custom; vectorbt is option but we want NSE-specific charges). Event-driven version for live parity.
- **Friction model (NSE)**:
  - STT: 0.025% sell-side intraday cash, 0.1% both sides delivery.
  - Exchange transaction charges (NSE), SEBI turnover fee, stamp duty, GST on brokerage + charges.
  - Brokerage as configurable (Zerodha-style flat ₹20/order is the default profile).
  - **Slippage**: linear-in-size against best-quote when OB unavailable; impact-adjusted from OB depth when available.
- **Strategies built on top**:
  - *Intraday cross-sectional*: long top decile, short bottom decile of 30-min forward median return, rebalance every 30 min during regular session, square off by 15:20 to avoid auction. Long-only variant for short-sell-restricted names.
  - *Swing cross-sectional*: same idea at 5-day horizon, daily rebalance at next-day open.
- **Risk overlays**: sector neutralization, market-beta neutralization, per-name and per-sector caps, daily VaR cap, kill-switch on intraday DD breach.

### 4.3 Capacity & realism
- Track ADV-fraction of every fill; cap participation at e.g. 5–10% of bar volume.
- Report metrics at multiple AUM levels (₹1Cr / ₹10Cr / ₹100Cr) — capacity decays sharply for intraday on NSE small/mid-caps.

---

## 5. Repo Structure

```
daity_v3/
├── README.md
├── pyproject.toml                # uv-managed, py3.11
├── configs/                      # hydra configs
│   ├── data.yaml
│   ├── pretrain.yaml
│   ├── posttrain.yaml
│   ├── finetune.yaml
│   └── backtest.yaml
├── daity/
│   ├── data/
│   │   ├── bq.py                 # BQ Storage Read → Parquet
│   │   ├── parquet_store.py
│   │   ├── calendar_nse.py
│   │   ├── corporate_actions.py
│   │   ├── features.py
│   │   ├── splits.py             # walk-forward + embargo
│   │   └── datasets.py           # IterableDataset implementations
│   ├── models/
│   │   ├── backbone.py           # PatchTST-style transformer
│   │   ├── tokenizer.py          # patching + RevIN
│   │   ├── heads.py              # quantile / ranking / classifier
│   │   ├── orderbook.py          # OB encoder + fusion
│   │   └── adapters.py           # LoRA
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── posttrain.py
│   │   ├── finetune.py
│   │   ├── losses.py
│   │   ├── schedulers.py
│   │   └── callbacks.py
│   ├── eval/
│   │   ├── metrics.py            # IC, NDCG, CRPS
│   │   ├── backtest.py
│   │   ├── frictions_nse.py
│   │   └── tearsheet.py
│   ├── serve/
│   │   ├── infer.py
│   │   └── paper_trade.py
│   └── utils/
├── scripts/                      # thin CLI entry points
├── notebooks/                    # EDA, ablations, postmortems
└── tests/
```

**Stack**: PyTorch 2.x + Lightning, HuggingFace `peft` for LoRA, Polars for FE, DuckDB + Parquet for cache, `google-cloud-bigquery-storage`, Hydra for configs, Weights & Biases for tracking, `uv` for env management.

---

## 6. Implementation Phases

| Phase | Deliverable | Rough effort |
|---|---|---|
| 0. Bootstrap | Repo skeleton, BQ auth, schema dump of both tables, sample-rate sanity checks | 1–2 d |
| 1. Data layer | BQ → Parquet ingest, NSE calendar, corp-action handling, feature engine, walk-forward splits, leakage tests | 3–4 d |
| 2. Pretraining | Backbone + SSL losses + multi-res tokenizer, first end-to-end pretrain run on a sample, scaling test | 4–5 d |
| 3. Post-training | Supervised heads, training loop, IC/CRPS eval harness, baseline comparison vs. XGBoost on engineered features | 3–4 d |
| 4. OB fine-tune | OB encoder, cross-attention fusion, LoRA wiring, fine-tune run, ablation: with/without OB | 3–4 d |
| 5. Backtest | Vectorized engine, NSE friction model, intraday + swing strategies, risk overlays, tearsheets | 3–4 d |
| 6. Serving | Batch inference, paper-trading harness, monitoring hooks (drift, IC live) | 2–3 d |

Total ≈ 3–4 weeks of focused work. Each phase ends with a runnable artifact and a short writeup committed to `notebooks/`.

---

## 7. Open Questions / Risks (will resolve in Phase 0–1)

1. **OB data volume.** How many symbol-days of order-book do we actually have? If under ~50 symbol-days, OB becomes a feature for an evaluation overlay rather than a fine-tune signal — we'd shift to using OB only for slippage modeling.
2. **Adjustment status of `curated_ohlcv`.** Adjusted-close vs. raw matters a lot; mis-adjustment will silently break swing labels.
3. **Survivorship.** If delisted names are missing, swing returns will be optimistically biased — need to confirm and flag.
4. **Universe size.** ~2000 NSE-listed names is fine; if `curated_ohlcv` covers only F&O 200 we'll narrow accordingly and revisit.
5. **Target leakage in features.** Aggressive engineered features (e.g. rolling stats spanning forecast horizon) need automated leakage tests as part of CI.
6. **Capacity.** Intraday alpha on Indian small/mid-caps is heavily capacity-constrained — every backtest must report ADV-participation and capacity-adjusted Sharpe.
7. **Regime shifts.** Indian markets had material structural changes (e.g. 2023 derivatives lot-size revisions, monthly→weekly expiry shifts). Walk-forward must be long enough to span regimes.

---

## 8. Decision points for you before I implement

These would change concrete code, not the architecture, so flagging now:

A. **Universe scope for v1**: F&O 200 (smaller, cleaner, capacity-friendly) vs. full Nifty 500 vs. all NSE cash. I'd recommend **F&O 200** for v1 because it's the realistic tradeable universe and OB is most likely to exist for these names.

B. **Pretraining corpus span**: last 5 years vs. all available history. Recommend **all available**, with a regime-aware sampling weight that upsamples recent data.

C. **Backbone size**: small (~20M params) vs. medium (~80M) for v1. Recommend **small** — equity time-series is data-limited; bigger models overfit unless we add cross-asset data.

D. **Live trading target**: paper-only for v1 vs. wire up to a broker (Zerodha Kite / Dhan). Recommend **paper-only** for v1; broker integration is its own milestone.

E. **Experiment tracking**: W&B (cloud) vs. self-hosted MLflow. Recommend **W&B** for speed; switch later if data residency becomes a concern.
