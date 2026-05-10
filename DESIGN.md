# daity_v3 — Design Document

**Status**: Draft v1, approved 2026-05-09; **post-audit amendments 2026-05-09 — see callout below**.
**Date**: 2026-05-09.
**Decided constraints (locked in conversation)**: single-GPU pretraining (≤A100 40GB); v1 deployment is research + paper trading only; order-book sample is medium (50–500 symbol-days), so OB enters as a fine-tune signal, not a pretraining corpus.

> **Post-audit amendments (2026-05-09).** Phase 0 audit (see `STATUS.md`) revealed several facts that supersede assumptions below. **Where this callout disagrees with text further down, the callout wins.**
>
> 1. **OB sample is ~13× larger than assumed.** §1, §3.4, and §8.1 estimated 50–500 symbol-days. Actual: **6,510 symbol-days with full L2 book** over a 2-month window (98.9% fill). OB stage stays a fine-tune (LoRA + small OB encoder), but the regularization can be relaxed and the curriculum's early-epoch zero-gate phase shortened to 1 epoch tops; the sample is no longer the binding constraint.
> 2. **Prices are pre-adjusted.** §2.3 planned to build a corp-action adjustment table; not needed in v1. `curated_ohlcv` is adjusted; `raw_ohlcv.scale` is available for verification. Data layer exposes `read_adjusted={True|False}` (defaults True). Bhavcopy ingestion deferred unless drift surfaces.
> 3. **Calendar already exists in BQ.** §2.3 implied building from scratch. Use `nse_trading_calendar` (1,717 rows) + `nse_holidays` (172 rows). The Python wrapper is a thin reader, not a builder.
> 4. **Survivorship bias is real.** Universe is the current-active F&O 200-ish roster (208 names in OHLCV, 206 in OB); delisted symbols are absent. Per §2.3 we tag every tearsheet with a survivorship warning and report a survivorship-adjusted Sharpe range.
> 5. **OHLCV resolutions are native.** §2.4 implied deriving multi-resolution from a base bar; instead `interval` already provides `5m`, `15m`, `60m`, `day` natively.
> 6. **Symbol master is partial.** `symbol_list` + `sector_metadata` give us `(symbol, sector, industry, macro_sector)`. Missing: ISIN, `valid_from/valid_to`, `mcap_bucket`, `free_float_bucket`, `fno_eligible`, `listed_on/delisted_on`. Phase 1 builds the missing columns; ISIN-based identity (§2.2) still applies once ISINs land.
> 7. **Candle `ts` convention (point-in-time critical).** Intraday intervals (`5m`/`15m`/`60m`) store `ts` as the candle's *open* time; `day` stores `ts` as the *close* time, **15:30 IST = 10:00 UTC**. The asymmetry is deliberate — it ensures any naive `ts <= as_of` join can never see same-day daily data before NSE has actually closed. All cross-interval as-of joins must go through `daity.data.candles.effective_close(interval, ts)` rather than reasoning about open/close per call site. Source-side data (e.g., Kite, which emits daily at 00:00 IST) is rewritten to the close-time convention at ingest by `daity.scripts.refresh_ohlcv`. Locked in by `tests/unit/test_candles.py` and `test_refresh_ohlcv_frame.py`.
> 8. **Parquet cache layout — per `(scale, symbol)`, monthly row groups.** §2.2 originally proposed `(symbol, year, month)`; in practice the dominant access pattern is pretraining (random window per random symbol, batched, millions of reads), where one-mmap-per-symbol wins decisively. `data/parquet/{scale}/{symbol}.parquet`, sorted by `ts`, with row group sizes tuned per scale (≈1 month of bars). 820 files total, 735 MB. Cross-sectional reads at strategy-rebalance frequency (every 30 min during regular session) go through `ParquetStore.read_cross_section`, which uses DuckDB to glob all symbol files for one scale; row group ts-stat pruning brings each query to ~200-500ms. Ingest is via BigQuery's `EXPORT DATA` to a GCS staging bucket (server-side parallelism, no client-side OOM, no readsessions IAM grant required) — see `daity.scripts.ingest_parquet`. The leakage gate `tests/leakage/test_parquet_leakage.py` spot-checks 100 random `(symbol, scale, as_of)` triples — the test DESIGN §2.5 calls out.
> 9. **Feature engine scope (Phase 1).** §2.4 lists ~20 features across 5 layers; the Phase-1 implementation ships **5 feature groups** that are sufficient for pretraining (Phase 2): bar channels (passthrough), log returns at lags `{1, 5, 15, 60}`, wick decomposition (range/body/upper/lower normalized to range), 20-bar volume z-score, IST trading-clock time covariates (sin/cos of minute-of-day + day-of-week). Microstructure-lite proxies, static covariates (sector embedding etc.), days-to-expiry, and RevIN are deferred — each is a single `_add_*` function when the next phase needs it; RevIN in particular is moved into the model tokenizer. The leakage gate `tests/leakage/test_features_leakage.py` covers three guarantees: no row past `as_of` is emitted, no in-progress intraday bar leaks (effective_close > as_of), and rolling features computed at row R are bit-equal whether the engine ran on full history or on history truncated at R (the no-future-peek invariant).
> 10. **Fourth SSL objective added — multi-horizon forecast on FORECAST token (Phase 2.1).** §3.2 originally listed three objectives + excluded a synthetic-order-flow fourth. Phase 2.1 adds a different fourth: a multi-horizon point-forecast head that reads the FORECAST token's non-causal hidden and predicts the next `H = forecast_n_patches` non-overlapping patches at a single `forecast_scale` (default 6 patches × 5m × patch_len 16 = 8h ≈ 1 trading day forward). Loss is plain MSE on RevIN-normalized targets (using the input-window's cached stats via `RevIN.apply_cached`). Loss weight 0.5. Rationale: the FORECAST token was otherwise consumed only by the placeholder contrastive head (see amendment 11) and was effectively unused; the multi-horizon head trains it as a useful summary representation that Phase 3's supervised quantile head can fine-tune.
> 11. **Contrastive head deferred to Phase 3 (Phase 2.1).** §3.2's cross-symbol same-date InfoNCE requires batch-construction infrastructure (multiple symbols at the same `as_of`) that lands in Phase 3. The Phase-2 implementation pairs `(causal-FORECAST, non-causal-FORECAST)` of the same sample as a placeholder, but with the broadcast FORECAST parameter and a causal mask, the causal-pass position-0 hidden is identical across the batch — InfoNCE then collapses to `log(B)` by construction with zero useful gradient. Phase 2.1 sets `w_contrastive: 0.0` in the production config (`configs/train/pretrain.yaml`); the head and code path remain so Phase 3 can re-enable with a fix once cross-symbol pairing exists. STATUS line 64's "expected to plateau at log(B)" describes the symptom; this amendment names the cause.
> 12. **RevIN affine OFF for SSL pretraining (Phase 2.1.1).** `revin_affine=True` (the original RevIN paper's recipe, suitable for time-series classification heads) is *broken* for SSL where MSE on the RevIN output is the loss. With `affine=True`, the tokenizer's `build_targets` and the input projection share the same learnable `(gamma, beta)`. The optimizer can shrink gamma → 0 to make the targets arbitrarily small, driving MSE → 0 without learning anything. The H200 first attempt confirmed this empirically: across 30K steps, gamma collapsed to ~0.04 of init, val_loss hit `4.95e-08`, and the trained checkpoint had no usable representation for Phase 3. `weight_decay` accelerates the collapse (L2 reg pulls gamma toward 0). Phase 2.1.1 sets `cfg.revin_affine: False` (and pins it in the YAML) so RevIN just standardizes — per-channel scaling is left to the linear layers in the tokenizer/backbone, which already provide ample expressivity. Regression locked in by `tests/unit/test_pretrain_module.py::test_revin_targets_do_not_collapse_during_overfit`.

This document is intentionally a fresh design rather than an extension of the existing `PLAN.md`. The two share NSE focus and a transformer backbone, but differ in: (a) pretraining objective design and how OB-derived synthetic signals are folded in pre-fine-tune, (b) explicit conformal calibration of the forecast distribution, (c) capacity-aware backtesting as a first-class concern rather than an overlay, and (d) a much sharper definition of what counts as "done" at each phase.

---

## 1. North Star and Non-Goals

The system should produce **calibrated, cross-sectional forecasts of forward returns at intraday and swing horizons for the NSE tradeable universe**, derived from a single foundation model that is pretrained on OHLCV self-supervision, post-trained as a supervised quantile/ranker, and fine-tuned with order-book conditioning where data permits.

Concretely, "best in class" here means four things, in order of priority:

1. **Honesty of the alpha number.** Walk-forward, embargoed, friction-aware, capacity-adjusted, multi-AUM. If the system reports Sharpe > 2.0 at any meaningful AUM after honest accounting, it is real.
2. **Calibrated uncertainty.** The 80% predicted band should contain the realized return ~80% of the time, in-sample and out-of-sample. Sizing depends on this, so we calibrate it explicitly with conformal methods rather than trusting raw quantile heads.
3. **Reproducibility.** A single command from a clean checkout produces every artifact, deterministically up to GPU non-determinism. CI catches data leakage and silent label corruption before anyone trains.
4. **Paper-trade fidelity.** The paper-trading harness shares ≥95% of its code path with the (future) live path — same order-management, same risk circuits, only the broker adapter differs.

**Non-goals for v1.** Sub-minute / tick HFT. Options, futures, or any derivatives strategy beyond using F&O underlyings as cash equities. Cross-market (BSE/US/crypto). Reinforcement learning for execution. Multi-account portfolio construction (assume single book).

---

## 2. Data Layer

### 2.1 Sources (BigQuery)

Two source tables in `atomic-vault-476315-m3.daity_v3`:

- `curated_ohlcv` — assumed to contain `(symbol, ts, timeframe, open, high, low, close, volume)` plus possibly VWAP and trade-count. Confirm in Phase 0 whether prices are corporate-action-adjusted, what timeframes exist, what universe is covered, and whether delisted symbols are present. These four answers reshape the data layer.
- `order_book_depth` — assumed L2 snapshots `(symbol, ts, side, level, price, size)`. Confirm in Phase 0: snapshot rate, number of levels, symbol-day coverage, and whether it's snapshots or events. Coverage drives the fine-tune sample size.

### 2.2 Cache and read path

BigQuery is the lake; we don't train against it directly. Phase-1 ingest copies the relevant slices to a local Parquet store partitioned by `(symbol, year, month)`, using the BigQuery Storage Read API (Arrow path, not the REST one — it's an order of magnitude faster). DuckDB sits on top of the Parquet for ad-hoc analytics. PyTorch loads via PyArrow datasets.

Two design choices worth flagging:

- **Point-in-time correctness via `as_of`.** Every reader takes an `as_of: datetime`. No row whose `ts > as_of` is ever returned. This is the single mechanism that prevents lookahead leakage — features, labels, splits, and backtests all funnel through it.
- **Symbol identity on ISIN, not ticker.** Tickers change in NSE (mergers, suffix flips). The internal `symbol_id` is ISIN-based. A symbol-master table keeps `(symbol_id, ticker, valid_from, valid_to, isin, sector, industry, mcap_bucket, listed_on, delisted_on, fno_eligible)` and is the join key for everything downstream.

### 2.3 NSE-specific data hygiene (non-negotiable)

- **Calendar.** NSE trading days, holidays, Muhurat sessions, special sessions. Within a day: pre-open (09:00–09:15), regular (09:15–15:30), post-close. Auction periods (last 10 min) are flagged separately.
- **Corporate actions.** If `curated_ohlcv` is unadjusted, we build an adjustment-factor table from a corporate-actions feed (NSE bhavcopy plus a manual override table for bonuses/splits) and apply factors at read time so we can switch between adjusted and raw without regenerating cache. Swing labels are computed on adjusted prices.
- **Survivorship.** If delisted names are absent, swing performance will be optimistically biased by an estimated 1–3% per year. We confirm coverage in Phase 0; if delisted are missing we flag the leak in eval and quote a survivorship-adjusted Sharpe range.
- **Halts and circuits.** Detected from OHLCV (zero range, zero volume, or limit-locked bars where high == low == circuit boundary). Bars are tagged with a `tradable` mask the model receives as input and the strategy enforces.
- **F&O expiry calendar.** Days-to-expiry (weekly and monthly), expiry-day flag, and rollover windows are features.

### 2.4 Feature engineering

A feature is a deterministic function of `(symbol, as_of)`. Three layers:

- **Raw bar channels** — O, H, L, C, V, plus VWAP and trade-count where available. Derived: log returns at 1, 5, 15, 60-bar lags; bar range, body, upper-wick, lower-wick (each normalized to bar range); volume z-score against a 20-bar trailing mean.
- **Microstructure-lite proxies from OHLCV.** Lee–Ready-style trade-direction proxy, Roll's effective spread estimator, Amihud illiquidity at various windows. These are weak but cheap and cover the >99% of bar-time we don't have OB.
- **Static covariates.** Sector and industry IDs (one-hot embeddings), market-cap bucket, free-float bucket, F&O eligibility flag, listing age. Slow-moving — emitted as prefix tokens, not per-bar.
- **Time covariates.** Minute-of-day, day-of-week, days-to-expiry. Sin/cos encoded, added to patch embeddings.
- **Reversible Instance Normalization (RevIN)** per `(symbol, window)`. Critical: equity series are non-stationary, vol regimes shift, and RevIN lets the same backbone parameters work across small-caps and Nifty-50 names. Statistics are stored alongside outputs so we can de-normalize to absolute returns at inference.

### 2.5 Splits and leakage controls

Walk-forward expanding window. Default: train through `T-2yr`, validate `T-2yr` to `T-1yr`, test `T-1yr` to `T`, then roll the window forward by one quarter and refit. This is the only setup that survives regime change in Indian markets (post-2023 derivatives changes, expiry shifts, etc.).

Embargoes: the validation window starts `H` bars after train end, where `H` is the longest forecast horizon (10 days for swing → 10 trading days embargo). Same gap between val and test. This kills overlap leakage from rolling labels.

Cross-symbol holdouts are *not* used. The universe is fixed; we hold out time. (Holding out symbols leaks regime information through correlated returns.)

A **leakage CI test** runs as part of every PR: it constructs a feature, asserts that for any `as_of`, no upstream value comes from `ts > as_of`. The test spot-checks 100 random `(symbol, as_of)` triples and is the cheapest way to catch the worst class of bug we can introduce.

---

## 3. Modeling

### 3.1 Backbone — multi-resolution channel-independent patch transformer

The choice of backbone family is PatchTST/Chronos-style: each numeric channel is patched and tokenized independently, sharing weights across channels and across symbols. This is the most data-efficient family for time-series and the only one that handles channel-dropout cleanly (which we'll need for OB conditioning).

Multi-resolution input: parallel streams at 1m, 5m, 15m, 1h, 1d. Each stream is patched separately (patch length 16, stride 8) and gets a learned resolution embedding. The backbone sees the concatenation of patches across resolutions plus a `[FORECAST]` summary token whose final hidden state is the input to all downstream heads.

Initial sizing (single-GPU constrained):
- `d_model = 320`, `n_heads = 8`, `n_layers = 8`, FFN ratio 4×.
- ~20M parameters. Fits A100-40GB at batch 256 with gradient checkpointing and bf16. FlashAttention-2 mandatory.
- RoPE on the time axis. Channel attention is restricted to upper layers (last 2) — channel-independent for the first 6, channel-mixing for the last 2.

Why upper-layer channel mixing rather than full channel-mixing? Pure channel-independent (PatchTST original) is the strongest data-efficiency baseline; pure channel-mixing models tend to overfit on equity data. Mixing only in the upper layers preserves data efficiency while letting the model learn cross-channel context (e.g., volume × return sign) where it actually pays.

### 3.2 Pretraining — multi-task SSL

Three objectives, weighted sum of losses (initial weights `1.0 / 0.5 / 0.3`, tuned on val):

1. **Masked patch reconstruction** (primary). 30–40% of patches masked, lightweight MLP head reconstructs the patch from its post-encoder embedding. MSE on RevIN-normalized values, but **volume-weighted** — bars with volume below the 20th percentile contribute 0.25× weight, bars in the top decile contribute 1.5×. The model spends its capacity on liquid, tradeable bars.

2. **Next-patch prediction** (causal). A separate causal head with attention mask predicts the next patch from the prefix. Gives an autoregressive prior that downstream forecast heads can exploit.

3. **Cross-sectional contrastive** (InfoNCE). For a given trading minute (or day), pull together representations of the same date across different symbols *only if* their returns are co-moving above a threshold; push apart unrelated days. This teaches the backbone regime-awareness without explicit regime labels.

A fourth objective — synthetic order-flow reconstruction from OHLCV — was considered and excluded from v1: it adds engineering cost for unclear marginal value, and the same signal can be added cleanly in the fine-tune stage when we have real OB data.

Pretraining corpus: full available NSE history on `curated_ohlcv`, with a **regime-aware sampling weight** that upsamples the trailing 18 months by 2× to bias the backbone toward current dynamics without losing long-horizon coverage.

Compute budget: at the sized backbone, full pretraining is ~2–3 days on a single A100-40GB. We checkpoint every 2 hours and keep the best three by val MSE for downstream initialization.

### 3.3 Post-training — supervised forecasting

The pretrained backbone is the starting point. New heads attach to the `[FORECAST]` token:

- **Quantile regression head**, multi-horizon, multi-quantile. Pinball loss across `{0.1, 0.25, 0.5, 0.75, 0.9}` quantiles for horizons `{5, 15, 30, 60 min, 1, 3, 5, 10 d}`. Output a *distribution* per (symbol, horizon).
- **Cross-sectional ranking head** (ListMLE on the median-quantile prediction). Most equity strategies actually consume rankings, not point forecasts. This head provides the ranking objective directly so we don't lose information at strategy time.
- **Direction classifier**, 3-class (down / flat / up) with class-balanced cross-entropy. The "flat" class is defined per-horizon as `|return| < 0.5σ` of horizon volatility. Auxiliary, mainly a regularizer that improves quantile head calibration.
- **Realized-vol head**, predicts the next-horizon realized vol. Used by the strategy layer for sizing; also acts as another regularizer.

Training schedule: backbone frozen for the first epoch, then unfrozen with a 10× lower LR than heads (`5e-5` vs `5e-4`), cosine schedule with 5% warmup. EMA of weights (decay 0.999) used for evaluation.

**Conformal calibration** is run on the val window after each post-train completes. We use split conformal on residuals stratified by horizon and by realized-vol bucket. Output: per-horizon, per-vol-bucket residual quantile bands that are added to the raw quantile-head outputs to produce calibrated bands. This is the part that turns a good forecaster into one whose confidence we can size against. Mis-calibration is monitored in production.

### 3.4 Fine-tuning — order-book conditioning (the "limited data" stage)

Working assumption (your answer): 50–500 symbol-days of L2 OB data. That is enough for parameter-efficient fine-tuning of an intraday-only head, with strong regularization. It is not enough to retrain the backbone, and not enough for a swing head.

Architecture additions:

- **OB encoder**: a small (4 layers, `d_model=192`) transformer that consumes the most recent K snapshots (K ≈ 64) of L2 depth. The input per snapshot is engineered features rather than raw price-size pairs:
  - microprice and weighted mid-price,
  - cumulative OFI (Order Flow Imbalance, Cont/Stoikov definition) at 1, 5, 10 levels,
  - book pressure (bid_size − ask_size) / (bid_size + ask_size) at top 1, 3, 5 levels,
  - queue-imbalance asymmetry,
  - spread in bps and relative to its 5-min trailing mean,
  - depth at multiple price offsets (depth available within 1bp, 5bp, 10bp of mid).
  Engineered features dominate raw L2 in low-data regimes; we revisit raw-input design only if we get >1k symbol-days.
- **Fusion**: gated cross-attention block in the upper two backbone layers. Queries are time-series tokens, keys/values from the OB encoder. The gate is a learned per-token sigmoid that lets the model ignore OB when stale or absent — essential because at inference time, OB will be absent for most of the universe.
- **LoRA adapters** on backbone attention (`r=8`, `alpha=16`), full training on the OB encoder and fusion gate. Trainable params ~3% of backbone. This is the lever against overfitting on small N.

Training details:

- Only the intraday quantile and direction heads train in this stage; swing heads are frozen.
- Strong regularization: dropout 0.3 on fusion, weight decay 1e-2, EMA decay 0.9999, early-stop on val rank-IC at 30-min horizon.
- Curriculum: epoch 1 trains with the OB-gate forced to zero (so the model first relearns the head with the new training distribution); epoch 2+ unfreezes the gate.
- Evaluation must include the **with-OB / without-OB ablation** at every checkpoint. The fine-tuned model is only deployed if (a) with-OB beats without-OB on val rank-IC by a meaningful margin and (b) without-OB is no worse than the post-train baseline. This catches the failure mode where the fusion path silently degrades the core forecaster.

### 3.5 Why this layering, in one paragraph

Pretraining teaches the backbone the structure of equity price dynamics across the universe before it sees any label. Post-training converts that structure into forecasts, calibrated honestly via conformal residuals. Fine-tuning then uses the precious OB sample to correct the *short-horizon* head where microstructure information actually matters, without touching the parts of the model where it doesn't. Each stage uses the data type it is best matched to and is small enough to fit a single GPU.

---

## 4. Strategy and Evaluation Layer

### 4.1 Research metrics

- **Rank IC** by horizon (Spearman correlation between predicted-median and realized return, daily). Plus rank IC decay across horizons.
- **Top-bottom decile spread** in basis points, with t-stat across the test window.
- **CRPS** on the quantile head, plus calibration plots (predicted quantile vs. empirical coverage).
- **Hit rate** by direction class, profit factor, asymmetric upside/downside ratio.
- **Capacity curves**: Sharpe and turnover as a function of AUM, with ADV-fraction caps applied.

### 4.2 Backtest engine

Two implementations sharing a strategy interface:
- **Vectorized** for sweeps and ablations. Handles cross-sectional rank strategies in seconds across years.
- **Event-driven** for paper-trade parity. Same code path the paper-trader will use.

NSE friction model (the design of which is the difference between a real Sharpe and a paper one):
- STT: 0.025% sell-side intraday cash, 0.1% both sides delivery.
- Exchange transaction charges (NSE), SEBI turnover fee, stamp duty, GST on (brokerage + exchange charges).
- Brokerage profile is configurable; default = Zerodha-style flat ₹20/order. Discount-broker assumptions documented; we don't assume zero-fee.
- **Slippage** has two modes:
  - When OB is unavailable (the common case): linear-in-size against the next bar's open, with the slope estimated empirically from the OB-coverage subset and conditional on `(mcap_bucket, vol_bucket, time_of_day)`.
  - When OB is available: walk the book to compute realized impact for each fill.
- Auction-period orders rejected; orders submitted within 10 min of close are squared off in the next session unless the strategy is overnight.

### 4.3 Strategies

Two reference strategies, each with a long-only and long-short variant:

- **Intraday cross-sectional** (primary). Long top decile, short bottom decile of 30-min forward median return. Rebalance every 30 min during regular session, square off by 15:20. Long-only variant for SLB-restricted names. Sector-neutralized.
- **Swing cross-sectional**. 5-day horizon, daily rebalance at next-day open. Sector- and beta-neutralized.

Risk overlays applied uniformly:
- Per-name cap at `min(2% of book, 5% of 20-day ADV)`.
- Per-sector cap at 25% gross.
- Daily VaR cap at 1.5% of book (95th percentile of forecast-horizon-aligned vol).
- Kill-switch: if intraday MTM drawdown exceeds 1.0% before 11:00, square off and stand down.

### 4.4 Capacity and realism

Every backtest reports metrics at three AUM levels (₹1Cr, ₹10Cr, ₹100Cr) with capacity-adjusted slippage applied. Intraday alpha on Indian small/mid-caps degrades sharply at scale — reporting one-AUM Sharpe is a lie. The default tearsheet shows the capacity curve.

---

## 5. Infra, MLOps, and Repo

### 5.1 Repo layout

```
daity_v3/
├── README.md
├── DESIGN.md                        # this document
├── pyproject.toml                   # uv-managed, Python 3.11
├── configs/                         # Hydra
│   ├── data/{ohlcv.yaml, ob.yaml, splits.yaml}
│   ├── model/{backbone.yaml, heads.yaml, ob_encoder.yaml}
│   ├── train/{pretrain.yaml, posttrain.yaml, finetune.yaml}
│   ├── strategy/{intraday.yaml, swing.yaml}
│   └── backtest/{frictions_nse.yaml, capacity.yaml}
├── daity/
│   ├── data/
│   │   ├── bq.py                    # BQ Storage Read → Arrow
│   │   ├── parquet_store.py
│   │   ├── symbols.py               # symbol-master, ISIN identity
│   │   ├── calendar_nse.py
│   │   ├── corporate_actions.py
│   │   ├── features.py
│   │   ├── splits.py                # walk-forward + embargo
│   │   ├── leakage_tests.py         # invoked by CI
│   │   └── datasets.py              # IterableDataset
│   ├── models/
│   │   ├── backbone.py              # patch transformer
│   │   ├── tokenizer.py             # patching + RevIN
│   │   ├── heads.py                 # quantile / ranking / direction / vol
│   │   ├── orderbook.py             # OB encoder + features
│   │   ├── fusion.py                # gated cross-attention
│   │   ├── adapters.py              # LoRA
│   │   └── conformal.py             # split-conformal calibration
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── posttrain.py
│   │   ├── finetune.py
│   │   ├── losses.py
│   │   ├── schedulers.py
│   │   └── callbacks.py
│   ├── eval/
│   │   ├── metrics.py               # IC, NDCG, CRPS, calibration
│   │   ├── backtest_vec.py
│   │   ├── backtest_event.py
│   │   ├── frictions_nse.py
│   │   ├── capacity.py
│   │   └── tearsheet.py
│   ├── strategy/
│   │   ├── base.py                  # shared interface for backtest + paper
│   │   ├── intraday_xs.py
│   │   └── swing_xs.py
│   ├── serve/
│   │   ├── infer.py                 # batch inference
│   │   ├── paper_trade.py
│   │   ├── monitor.py               # live IC, drift detection
│   │   └── broker_stub.py           # interface, no real broker in v1
│   └── utils/
├── scripts/                         # thin CLI entry points
├── notebooks/                       # EDA, ablations, postmortems
└── tests/
    ├── unit/
    ├── leakage/                     # gating tests
    └── smoke/                       # tiny end-to-end run
```

### 5.2 Stack

PyTorch 2.x + Lightning. Hydra configs. `peft` for LoRA. Polars for feature engineering (faster than pandas, handles nulls properly). DuckDB + Parquet for the cache. `google-cloud-bigquery-storage` for ingest. W&B for tracking. `uv` for env management. `ruff` + `mypy` for hygiene.

### 5.3 Reproducibility

Seeds at every entry point, but more importantly: every artifact produced (cache slice, model checkpoint, backtest run) is tagged with the git SHA, the resolved Hydra config hash, and the data window (`as_of`). The same (SHA, config-hash, as_of) triple must produce the same result up to GPU non-determinism. This is checked in CI on the smoke run.

### 5.4 CI gates

- Unit tests on data transforms.
- Leakage tests on the feature pipeline (described in §2.5).
- A smoke training run: tiny model, tiny window, tiny universe, must complete end-to-end (pretrain → posttrain → backtest) in under 5 min on CPU.
- Lint and type checks.

### 5.5 Tracking

W&B for training runs. A `runs/` registry on local disk for backtests (one JSON per run with metrics + config-hash). The backtest tearsheet is a generated HTML file checked into a `reports/` directory.

---

## 6. Risk and Monitoring

For paper-trading we still build all the controls we'd want live, because a paper-trader without circuits is useless as a fidelity test:

- **Pre-trade**: position-cap check, ADV-fraction check, sector-cap check, SLB eligibility for shorts, halt/circuit guard, stale-data guard (no order if latest bar is more than 2× the bar-period old).
- **Intraday**: mark-to-market every minute, drawdown circuit at 1% intraday loss, model-drift circuit (predicted-vs-realized rank-IC over rolling 5 days flips below 0 → stand down).
- **Daily**: end-of-day reconciliation, IC report, calibration report, position diff vs. expected, capacity utilization log.
- **Weekly**: a generated review notebook summarizing live vs. backtest performance for the same period, flagging any regime shift, and surfacing whether the conformal bands are still calibrated.

---

## 7. Implementation Phases

Each phase ends with a runnable artifact and an acceptance criterion. We don't move past the gate without passing.

| Phase | Deliverable | Acceptance criterion | Effort |
|---|---|---|---|
| 0. Bootstrap | Repo skeleton, BQ auth, schema dumps of both tables, sanity-check notebook on a sample | Schema confirmed, OB symbol-day count locked, adjustment/survivorship status known | 1–2 d |
| 1. Data layer | BQ→Parquet ingest, NSE calendar, corp-action handling, feature engine, walk-forward splits, leakage tests | Leakage CI green; full universe loadable; one EDA notebook committed | 3–4 d |
| 2. Pretraining | Backbone + multi-task SSL + multi-res tokenizer; full pretrain run on full history | Val masked-MSE plateau reached; pretrained checkpoint < 200 MB; reconstructions visually plausible on held-out symbols | 4–5 d |
| 3. Post-training | Supervised heads (quantile + ranking + direction + vol), conformal calibration, eval harness, baseline vs. XGBoost on engineered features | Val rank-IC > XGBoost baseline at 30-min and 5-day horizons; conformal bands calibrated within 5% of nominal | 3–4 d |
| 4. OB fine-tune | OB encoder + features + gated fusion + LoRA wiring; with-/without-OB ablation | With-OB beats without-OB at 30-min rank-IC by ≥ 0.005 on val; without-OB no worse than post-train baseline | 3–4 d |
| 5. Backtest | Vectorized + event-driven engines, NSE friction model, two strategies, risk overlays, capacity curves, tearsheets | Walk-forward Sharpe > 1.0 net at ₹10Cr AUM on at least one strategy; capacity curve produced | 3–4 d |
| 6. Paper trade | Inference service, paper-trader sharing strategy code path with backtest, monitoring hooks (live IC, drift, calibration) | One-week dry run completes with no critical alerts; live IC within bootstrap CI of backtest IC | 2–3 d |

**Total**: ~3–4 weeks of focused work. Each phase commits a notebook (`notebooks/phase_N_writeup.ipynb`) summarizing what was built, what the numbers were, and what surprised us.

---

## 8. Risks and Mitigations

The risks worth pre-committing mitigations for, in priority order:

1. **OB volume is smaller than expected.** If Phase 0 reveals < 50 symbol-days, fine-tuning is dropped; OB is repurposed as a slippage-model calibration sample and a microstructure eval overlay only. This saves Phase 4 effort and doesn't compromise the rest.
2. **OHLCV is unadjusted and corp-actions table is incomplete.** Swing labels become unreliable. Mitigation: build the adjustment table from NSE bhavcopy as part of Phase 1; if adjustments are still unreliable, restrict v1 to intraday only.
3. **Survivorship bias in OHLCV.** If delisted symbols are missing, swing performance overstated. Mitigation: in Phase 0, query distinct symbols by year and look for the expected delisting pattern; if missing, source delisted history from a corporate-actions feed before Phase 3.
4. **Backbone overfits.** Symptom: train MSE keeps falling, val MSE flat or rising. Mitigation: stronger weight decay, longer warmup, smaller `d_model`, and the cross-sectional contrastive loss tends to help here. We already start small (~20M).
5. **OB fine-tune silently degrades the core.** Mitigation: the with-/without-OB ablation gate at deployment time. If without-OB is worse than the pre-fine-tune model, we ship the pre-fine-tune model.
6. **Forecast horizons leak into features.** Mitigation: the leakage test in CI; no rolling-window feature can use a window longer than `(now - max_horizon)`.
7. **Capacity blow-up at scale.** Mitigation: every tearsheet includes the capacity curve. We don't quote a single Sharpe — we quote (AUM, Sharpe) tuples.
8. **Regime shift mid-test.** Mitigation: walk-forward refits quarterly; the realized-vol head is monitored as a regime detector; the kill-switch limits damage from any single misforecast.

---

## 9. Open Questions for You

These don't block design approval but will resolve in Phase 0 and may tighten the plan:

1. **Universe scope for v1.** F&O 200 (recommended; cleaner, capacity-friendly, OB more likely to exist) vs Nifty 500 vs full NSE cash. I default to F&O 200 unless you say otherwise.
2. **Earliest history we trust.** Default: full available, with regime-aware sampling. Override if you know specific years are dirty.
3. **Adjustment status of `curated_ohlcv`.** Pre-adjusted vs raw — if you already know, tell me; otherwise Phase 0 confirms.
4. **OB snapshot rate and depth.** Per-second / sub-second? Top 5 / top 20 levels?
5. **Tracking platform.** W&B (default) vs self-hosted MLflow. W&B is faster to start; switch is straightforward later.

---

## 10. What "approved" means

If you approve this design, I will start with **Phase 0** (repo skeleton, BQ auth verification, schema introspection on both tables, sample-rate sanity, OB symbol-day census). I will write the result to `notebooks/phase0_data_audit.ipynb` and report back the actual numbers before moving on to Phase 1. Any answers that contradict the assumptions in §2 will trigger a short design amendment before we proceed.

If you want changes — sizing, scope, deployment target, anything in §3 or §4 — flag them and I will revise this doc rather than start implementation against a stale plan.
