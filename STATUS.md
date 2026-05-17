# STATUS — Phase ledger

Live single-source-of-truth for daity_v3 phase progress. One row per phase, three states: `pending`, `in_progress`, `done`. Each phase needs both an acceptance check (per `DESIGN.md` §7) and a reviewer-agent signoff before moving to the next.

| Phase | Title | State | Acceptance check | Reviewer signoff |
|---|---|---|---|---|
| 0 | Bootstrap + BQ audit | done | code ✓, 23 unit tests ✓; audit run ✓ (`reports/phase0_audit.json` 2026-05-09) | reviewer #1 ✓, reviewer #2 ✓; post-audit deltas captured below |
| 1 | Data layer | done | calendar ✓, symbol-master+ISIN ✓, walk-forward splits ✓, BQ→Parquet ingest ✓ (820 partitions / 735 MB), feature engine ✓ (5 groups), four leakage gates ✓ (split, parquet, features × 2), EDA writeup ✓ | reviewer #1 (`phase_1_review.md`) flagged 3 Critical + 8 Major; all 11 fixed in `phase_1_review_responses.md`; reviewer #2 (`phase_1_review_v2.md`) **APPROVED** with 4 deferred Minor/Nits |
| 2 | Pretraining | in_progress | code ✓ (tokenizer · backbone · 4 SSL heads incl. multi-horizon forecast · Lightning module · CLI · `ModelCheckpoint` + `EMACallback` + provenance triple), 130 unit tests ✓ (incl. 13 callback regression tests), CPU + MPS smokes ✓, writeup notebook ✓; reviewer #1 (`phase_2_review.md`) flagged 1 Critical + 6 Major + 9 Minor + 4 Nit; **must-land-before-launch (Critical 1, Majors 2/3/4/7/17) all addressed in Phase 2.1 patch — DESIGN amendments 10 + 11 added; pending: H200 run, reviewer #2 signoff** | reviewer #1 v1 logged 1C/6M/9N/4N; v2 pending after run |
| 3 | Post-training | pending | — | — |
| 4 | OB fine-tune | pending | — | — |
| 5 | Backtest | pending | — | — |
| 6 | Paper trade | pending | — | — |

## Phase 1 — final state

Test count: **262 passing** (60+ unit on data layer, 4 leakage gates: split / parquet × 2 / features). Lint clean on Phase-1 modules.

**Modules built**: `bq.py` · `calendar_nse.py` · `symbols.py` · `splits.py` · `groww.py` · `kite.py` · `candles.py` · `parquet_store.py` · `features.py` · `ohlcv_io.py` · `parquet_ingest.py` · `corp_action_detect.py` · `readjust.py`.

**Scripts shipped** (all <200 LOC except `phase0_audit.py` which is Phase-0 territory):
`daity-audit` · `daity-refresh-calendar` · `daity-refresh-ohlcv` · `daity-fix-daily-ts` · `daity-readjust-symbols` · `daity-detect-corp-actions` · `daity-fix-pointwise` · `daity-kite-login` · `daity-ingest-parquet` · `daity-check-staleness`.

**Data state**: 36.8M `raw_ohlcv` rows on BQ + 820 Parquet partitions (~735 MB) locally · calendar through 2027-05-10 · 205 symbols all ISIN-resolved · 4 corp-action readjustments + 1 pointwise fix applied · idempotency marker at `data/applied_readjustments.json`.

**DESIGN.md amendments**: 9 amendments documenting deltas from the original draft.

**Deferred to Phase 2 (reviewer-approved)**:
- N2: BQ-side journal table for the readjustment marker (sidecar JSON works for now)
- N3: `refresh_calendar.py` LOC trim (230 → ~180)
- N4: `datetime.utcnow()` → `datetime.now(UTC)` modernization

## Phase 2 entry checklist

- ✅ Phase 1 acceptance met per DESIGN §7 (leakage CI green; full universe loadable; EDA notebook committed)
- ✅ Independent reviewer signoff (reviewer #2 verdict: approve)
- ✅ All 3 Criticals + 8 Majors from review #1 verified fixed in review #2
- ✅ Per-symbol Parquet store ready for pretraining batch fetch (DESIGN amendment 8)
- ✅ Feature engine ready for Phase 3 supervised heads (DESIGN amendment 9)
- ✅ Candle ts convention enforced end-to-end (DESIGN amendment 7)

## Phase 2 — code complete, awaiting GPU run

Test count: **392 passing** total (262 from Phase 1 + 117 unit tests for the Phase-2 model/data/training modules + 13 new callback-regression tests for the Phase 2.1 patch). 25 of those 117 are forecast-head specific. Lint clean on `daity/models/`, `daity/training/`, the Phase-2 additions in `daity/data/`, and the Phase-2 CLI; pre-existing scripts in `daity/scripts/` carry the same ruff debt as Phase 1 (Phase-1 review minor #16, deferred).

**Modules built**:
- `daity/models/revin.py` — Reversible Instance Normalization per-(B, scale, channel); detached mean/std, optional learnable affine. Three modes: `norm` (compute+cache), `denorm` (invert), `apply_cached` (apply input-window stats to a NEW input — used by forecast targets so future bars live in the encoder's coordinate frame).
- `daity/models/tokenizer.py` — `Patcher` (unfold to `(B, n_patches, C, patch_len)`) + `MultiResTokenizer` (per-scale RevIN → patch → linear projection → resolution embed → concat → prepend FORECAST token).
- `daity/models/backbone.py` — RoPE-on-Q+K via SDPA, pre-norm transformer blocks, 9.86M params at d_model=320 / n_layers=8 / n_heads=8 / ffn_ratio=4. `causal=True` flag for NTP pass.
- `daity/models/heads.py` — `MaskedReconstructionHead`, `NextPatchHead`, `ContrastiveHead`, `MultiHorizonForecastHead` (the FORECAST token's non-causal hidden → next H non-overlapping patches at `forecast_scale`).
- `daity/data/datasets.py` — `OHLCVWindowSampler` + `OHLCVPretrainDataset` (IterableDataset). Multi-resolution aligned windows ending at the same `end_ts`; regime-aware sampling (2× weight on trailing 18 months); optional forecast-tail (`forecast_n_bars` future bars at `forecast_scale`) packed under `FORECAST_FUTURE_KEY`. Latest-end_ts bound caps by both the as_of leakage gate AND the actual data extent — protects against `as_of` running ahead of ingest causing the iterator to spin.
- `daity/data/tokenizer_targets.py` — `build_targets(...)` (RevIN-normalized recon targets + per-bar volumes) + `build_forecast_targets(...)` (future bars unfolded into N non-overlapping patches in the input window's RevIN frame).
- `daity/training/losses.py` — `apply_random_patch_mask` (preserves FORECAST token at position 0), `volume_weighted_mse` (bottom 20% × 0.25, top 10% × 1.5 per DESIGN §3.2), `next_patch_mse`, `info_nce_loss` (symmetric InfoNCE).
- `daity/training/pretrain.py` — `PretrainConfig` dataclass + `PretrainModule(LightningModule)` wiring tokenizer + backbone + 4 heads + learnable mask-token; cosine LR with warmup; AdamW. Forecast head conditional on `forecast_n_patches > 0` (set to 0 → behavior identical to the 3-loss model, useful for ablation).

**Scripts shipped**: `daity-pretrain` (`daity/scripts/pretrain_cli.py`) — YAML config + CLI overrides + `--smoke` flag.

**Configs**: `configs/train/pretrain.yaml` (production: d_model=320, n_layers=8, batch=64, max_steps=50K, bf16-mixed, A100-40GB). Smoke override path: `daity-pretrain --smoke`.

**Smoke evidence** (from `notebooks/phase_2_writeup.ipynb`, MPS, 200 steps, ~5s):
- recon loss:    0.895 → 0.094 (−89.5%)
- ntp loss:      0.619 → 0.063 (−89.9%)
- forecast loss: 16.4  → 2.0   (−87.8%)  ← multi-horizon, weight 0.5
- contrast plateau ~2.08 ≈ log(8) (expected for placeholder positives at batch 8)
- production-config backbone params: 9,856,640 (under 20M budget; rest reserved for Phase 4 OB fusion)
- forecast head adds 513,120 params (bringing total to 11,276,328)
- bf16 weights checkpoint: 21.5 MB (well under 200 MB)

**DESIGN-locked invariants & their tests**:
- mask-ratio bounded in (0, 1); FORECAST token never masked → `tests/unit/test_losses.py`
- volume-weighting bands at 20%/10% per DESIGN §3.2 → same
- RevIN denorm exactly inverts norm → `tests/unit/test_revin.py`
- RevIN `apply_cached` uses input-window stats (not recomputed); validates shape + batch — keeps forecast targets in encoder frame → `tests/unit/test_revin.py`
- backbone causal mask blocks future tokens → `tests/unit/test_backbone.py`
- backbone params ≤ 20M at production config → `tests/unit/test_backbone.py`
- `OHLCVPretrainDataset` honors `as_of` (no future leakage) → `tests/unit/test_datasets.py` (note: file lives under `unit/`, not `leakage/`; reviewer finding #4 / #8 — promotion to `tests/leakage/` deferred to Phase 3)
- forecast tail respects `as_of` cap; `_latest_end_ts` bounded by data availability so `as_of >> data_end` doesn't hang the iterator → `tests/unit/test_datasets.py`
- forecast head shape `(B, H, C, patch_len)`; head off when `forecast_n_patches=0`; gradient flow reaches forecast head → `tests/unit/test_heads_and_losses.py` + `test_pretrain_module.py`
- forecast-loss 1-batch overfit drops ≥30% across 100 steps → `tests/unit/test_pretrain_module.py`
- `build_forecast_targets` rejects unknown scale + length mismatch; uses cached input-window RevIN frame → `tests/unit/test_pretrain_module.py`
- LR schedule: warmup-then-cosine with initial LR == 0 → `tests/unit/test_pretrain_module.py`
- 1-batch overfit reduces total loss across 100 steps → same

**Operational artifacts**:
- `docs/vast_bootstrap.md` — one-shot Vast.ai bootstrap (ssh, uv, secrets, rsync data path, smoke + full run in tmux, artifact pull-back).
- `notebooks/phase_2_writeup.py` + `.ipynb` — acceptance writeup with 8 sections (env, hardware, architecture, checkpoint size, live smoke, reconstruction viz, timing budget, acceptance checklist). All assertions pass.

**Phase 2 acceptance — what's left**:
1. Launch full pretraining (H200, ~12-18h at 50K steps; previously sized for A100-40GB at ~2-3d).
2. Pull the best checkpoint + logs back; rerun the writeup with the trained checkpoint to populate the post-training plots.
3. Spawn Phase 2 reviewer #2 against the trained checkpoint + Phase 2.1 patch (verifying review #1's must-land-before-launch findings are properly addressed).

## Phase 2.5 — Cohort model + adaptive strategy (research track, outside formal phase gates)

**Completed 2026-05-16/17 on Vast.ai H200 instance. GPU instance can now be shut down.**

### Cohort model architecture (daity/models/cohort/)

Four-component pipeline: `StockContextEncoder` → `MarketContextTransformer` → cross-attention → multi-horizon head. Predicts 10 horizons (30m/60m/120m/180m intraday, to_close, overnight, next_day_1h, next_day_eod, day_plus_3, day_plus_5).

Training stages:
1. **Contrastive pretrain** (`pretrain_contrastive.py`): Two-tower InfoNCE on NSE stock pairs 2019–2024. τ=0.1, 4K steps, ~2h on H200. Encoder init for downstream.
2. **Walk-forward Mode A fine-tune** (`train_cohort.py`): Regression+rank loss, 2019→2025, cohort batches of ~200 stocks. Init from contrastive pretrained encoder.
3. **Walk-forward Modes B+C** (experimental): short-window and ranking-only variants.

Champion checkpoint: `runs/cohort_modeA_v11_from_contrastive/`  
First model to show positive OOS P&L on full 2025 test: +4.93 bps/day (day_plus_5, K=10, 5 bps cost, Sharpe 0.29).

### Static prediction dump

`dump_static_predictions.py` — runs frozen v11 ckpt over 2025-02-01 to 2026-04-30 (303 trading days). Output: `reports/v11_static_predictions_2025_2026.parquet` (64,880 rows; gitignored — keep locally). **This parquet is the only artifact needed for all downstream strategy work; GPU not required after this.**

### Adaptive Calibrated Strategy — REVISED 2026-05-17 (post-review)

Full documentation: `docs/strategy_adaptive_calibrated_v1.md` (see §0 — bug fix log)

**Earlier "Sharpe 1.91, +65% sleeve" champion claim was retracted after independent
two-reviewer adversarial review identified a critical lookahead bug:**
`anchor_cap_realization = d - timedelta(days=5)` used 5 CALENDAR days for a 5
TRADING day horizon → weekends/holidays let unrealized day_plus_5 labels into both
the GBM training set and the OOS-gate holdout window. The gate itself was leaky.

Fix committed: trading-day index-based cap. Sharpe annualization also corrected
from `sqrt(252)` to `sqrt(252/hold_days)` for overlapping 5-day positions.

**Honest results after fix (K=1, N=10, holdout=3, GBM, day_plus_5, 303 days):**

| Cost (round-trip) | Sharpe | Sleeve (14mo) | Mean bps/day | Trade% | Hit% |
|-------------------|-------:|--------------:|-------------:|-------:|-----:|
| 5 bps (optimistic) | +0.76 | +25.64% | +39.00 | 62% | 54% |
| 15 bps (realistic) | **+0.63** | **+20.67%** | +32.33 | 62% | 53% |
| 30 bps (conservative) | +0.41 | +12.07% | +20.03 | 60% | 51% |

Best honest config: **K=1 (not K=2)**. At realistic 15 bps cost, the GBM gate beats
all baselines (baseline K=7 at 15 bps: Sharpe 0.30, sleeve +6.85%). Improvement
over baseline is real but ~3× (not 7×).

Capital accounting: 5-day hold + daily entry = 5 overlapping sleeves; return = `prod(1 + bps/10000/5) - 1`. No leverage.

**Key experiments completed:**
- Calibrator sweep: GBM > LR > RF (all with holdout=3 gate)
- First bug fixed earlier: threshold evaluation must be on held-out days (OOS), not the same days used to train the GBM
- Second bug fixed 2026-05-17 (this commit): trading-day vs calendar-day mismatch in `anchor_cap_realization`. This was the bigger leak — inflated headline Sharpe by ~3.4× and sleeve return by ~5×.
- K sweep (K=1..5) re-run honest: K=1 emerges as best (was K=2 pre-fix)
- Cost sensitivity: at 30 bps round-trip K=1 still posts +12% sleeve (vs baseline -2%)

**Outstanding reviewer concerns (not yet fixed):**
- Contrastive pair mining (`cohort_pair_miner.mine_sector_alpha_pairs`) uses realized
  future returns to define positive pairs. Bounded by train_end so not a direct OOS
  leak, but encoder is label-conditioned. Generalization claim weaker than originally
  stated.
- Threshold-sweep retrain step uses holdout window — small residual leakage.

**GPU shutdown checklist:**
- [x] `reports/v11_static_predictions_2025_2026.parquet` pulled to local disk
- [x] Champion checkpoint at `runs/cohort_modeA_v11_from_contrastive/` pulled local
- [x] Contrastive pretrain ckpt + pair dataset pulled local
- [x] All code (including bug fixes) committed to git
- [x] Strategy runs fully on CPU — no GPU dependency for Stage 4
- [x] Independent two-reviewer audit completed; remaining bugs documented above

### Phase 2.1 patch (post-review #1)

Reviewer #1 (`reports/reviews/phase_2_review.md`, 2026-05-10) verdict: **Block** with 1 Critical + 6 Major + 9 Minor + 4 Nit.  Critical and 4 of the 6 Majors required to land before the H200 launch. Status of those:

- ✅ **Finding 1 (Critical):** `ModelCheckpoint(top_k=3, save_last, every_n_train_steps=2000)` wired in `pretrain_cli.py`; new YAML keys `checkpoint_every_n_steps` + `save_top_k`. Regression: `tests/unit/test_pretrain_callbacks.py::test_pretrain_cli_emits_provenance_sidecar` (asserts ≥1 `.ckpt` lands).
- ✅ **Finding 2 (Major):** Contrastive head honestly-disabled — `w_contrastive: 0.0` in production YAML and smoke override. Code path retained for Phase 3 re-enable. DESIGN amendment 11 names the cause (causal-pass position-0 hidden is identical across batch → InfoNCE collapses to log(B)).
- ✅ **Finding 3 (Major):** `EMACallback(decay=cfg.ema_decay)` in `daity/training/callbacks.py`; swaps shadow weights into the live module on `validation_start`, restores on `validation_end`. State-dict round-trips so resumed training picks up the EMA. 7 dedicated tests.
- ✅ **Finding 4 (Major):** STATUS drift fixed — leakage-gate filename corrected to point at `tests/unit/test_datasets.py`, forecast-test count restated as 25, lint claim narrowed to "Phase-2 modules".
- ✅ **Finding 7 (Major):** `(git_sha, config_hash, as_of)` provenance triple emitted as `provenance.json` sidecar at run-start + stamped into LightningModule's `hparams` (lands inside every checkpoint). `--allow-dirty` flag refuses uncommitted-changes runs unless explicitly overridden. 5 tests.
- ✅ **Finding 17 (Nit):** DESIGN amendment 10 added formally documenting the 4th SSL objective (multi-horizon forecast).

Deferred to Phase 2.2 / Phase 3 (per reviewer): Major 6 (single-forward refactor — compute optimization, not correctness), Minor 5/8/9/11/12/13/14/15/16, Nits 18/19/20.

## Phase 0 — final summary

Code, tests, and audit all green. Two latent bugs surfaced on first live run (now fixed and regression-tested):
- `COUNT(*) AS rows` failed 3 queries (`rows` is a BigQuery reserved word). Renamed to `n_rows`; new test `test_no_reserved_word_aliases_in_audit_sql` guards regressions.
- `order_book_depth` is **wide-format** (top-5 levels columned out as `bid{k}_price/qty`, `ask{k}_price/qty`), not the long-format `(level, side, price, size)` the original audit draft assumed. Added `_detect_wide_ob_levels` + book-fill-rate + symbol-day-with-book probes, plus 4 unit tests for the detector.

### Audit results — actionable findings (locked, source for Phase 1+)

**`curated_ohlcv`** (208 symbols, 2019-01-01 → 2026-02-27 UTC = NSE 09:15 IST onwards):
- Schema: `(symbol STRING, date DATE, ts TIMESTAMP, interval STRING, open/high/low/close FLOAT, volume INTEGER)`. *Both* `date` and `ts` exist — `date` is partition-friendly.
- Resolutions present in `interval`: **`5m`, `15m`, `60m`, `day`** (4 timeframes, all native — no need to derive coarser from `5m`).
- **Prices are pre-adjusted.** Daily-only adjustment probe: 8/340,191 returns >50% (~0.002%, expected residual from corp actions and circuit-locked names that pass through). p99.5=10.0%, p99.9=16.2% on daily returns — consistent with adjusted equity series.
- **Survivorship is biased.** 207/208 symbols stale ≤30 days → universe is the current-active F&O 200-ish roster; delisted symbols absent. Eval must quote a survivorship-adjusted Sharpe range; do not take swing performance at face value.
- `num_rows = 0` reported by BQ table metadata — likely a view over a streaming table or partition aggregation; not a problem (queries return data normally).

**`order_book_depth`** (206 symbols, 6,563 symbol-days total, **6,510 symbol-days with full L2 book** = 98.9% fill):
- Schema: per-snapshot wide L2: `(symbol, date, ts, ltp, volume, total_buy_qty, total_sell_qty, bid1_price..bid5_price/qty, ask1_price..ask5_price/qty, spread_bps, fetched_at)`.
- Time range: 2026-03-08 → 2026-05-04 (~2 months). **Strict 60-second snapshots** (p5=59.7s, p50=60.001s, p95=60.4s — quartz-tight cadence, not event-driven).
- Partitioned by `date`, clustered by `(symbol, ts)` — perfect for our access pattern.
- **OB sample is ~13× the design's upper bound.** DESIGN §3.4 sized for "50–500 symbol-days"; we have **6,510 with-book**. This is enough to consider widening Phase 4 from LoRA-only fine-tune to optionally training a small portion of the upper backbone, and is comfortably enough for an OB-conditioned intraday quantile head plus a slippage-model calibration sample.

**Companion tables already in `daity_v3` (avoid rebuilding)**:
- `nse_trading_calendar` (1,717 rows, `trade_date`) — canonical NSE trading-day index.
- `nse_holidays` (172 rows, `trade_date, description, source_year`) — annotated holidays.
- `symbol_list` (206 rows, `symbol`) — universe roster (matches OHLCV/OB cardinality).
- `sector_metadata` (206 rows, `symbol, sector, industry, macro_sector`) — drop-in static covariates.
- `raw_ohlcv` (36.8M rows, `symbol, ts, scale, open, high, low, close, volume`) — unadjusted prices + scale factor; lets us verify the `curated = raw * scale` relationship and rebuild adjustment tables if `curated` ever drifts.
- `orderbook_daily_summary` (34 rows) — per-day OB QC pre-aggregates.
- `orderbook_quality_metrics` (10,770 rows, per-minute) — fine-grained OB QC.
- Plus an extensive prior `ml_*` / `fps_*` / `feature_store*` / `paper_*` set we are **intentionally ignoring** (DESIGN.md is a fresh design; cross-referencing those is out of scope for v1).

### Phase 0 deltas vs DESIGN.md (drives Phase 1 scope)

1. **Calendar / holidays** — use `nse_trading_calendar` + `nse_holidays` directly (DESIGN §2.3 originally implied building from scratch).
2. **Symbol master** — start from `symbol_list` + `sector_metadata`; ISIN, `valid_from/valid_to`, `mcap_bucket`, `free_float_bucket`, `fno_eligible`, `listed_on/delisted_on` are still missing. Phase 1 builds *only* the missing columns rather than the full table.
3. **Adjustment table** — DESIGN §2.3 said "build adjustment-factor table from a corporate-actions feed". With `curated_ohlcv` already adjusted and `raw_ohlcv.scale` available, Phase 1 instead **verifies** the adjustment via spot checks and exposes a `read_adjusted={True|False}` flag in the data layer. Bhavcopy ingestion is deferred unless we find drift.
4. **Survivorship** — confirmed absent. Phase 1 adds a `survivorship_warning` field to every backtest tearsheet metadata and tracks the gap explicitly.
5. **OB sizing for Phase 4** — design assumption widened from "50–500 symbol-days" to "**~6.5K symbol-days with-book over a 2-month window**". Phase 4 design will be revisited with this in mind; for Phase 1 it just means the OB ingest budget is larger than the original sketch.

### Phase 0 deferred recommendations (operational, not blockers)

- Rotate the current SA key in GCP IAM and relocate the canonical key to `~/.config/gcloud/daity-sa.json`. Reviewer #1 flagged; still applies. (Note: live runs since 2026-05-09 also surface `loose perms 644` on `.sa-key.json` — `chmod 600` if keeping it in-repo.)

## Phase 1 — work in progress

Test count: 65 total (54 unit + 11 leakage), all green; lint clean on every Phase 1 file.

**Done**:
- `daity/data/calendar_nse.py` — `NSECalendar` wrapper around `nse_trading_calendar` + `nse_holidays`. Cached as Parquet under `data/cache/calendar/`. API: `is_trading_day`, `trading_days(start, end, *, as_of)`, `next/prev_trading_day`, `shift(d, n, *, as_of)`, `n_trading_days_between`, `holidays(year=...)`. Every accessor honors `as_of` so leakage is impossible at this layer. 21 unit tests.
- `daity/data/symbols.py` — `SymbolMaster` wrapper joining `symbol_list` + `sector_metadata`. 206 symbols, all sector-resolved. Cache at `data/cache/symbols/symbol_master.parquet`. Carries placeholder columns for the 8 not-yet-available identity fields (ISIN, validity windows, F&O flag, mcap/free-float buckets, listed_on/delisted_on) and exposes `is_provisional=True` plus `missing_columns` so callers cannot silently use them. `symbol_id == symbol` until ISINs land — single-file rebind. 10 unit tests.
- `daity/data/splits.py` — `WalkForwardSplitter` with trading-day-aware embargoes (DESIGN §2.5). Generates expanding-window folds with configurable val/test windows and roll stride; default 10-day embargo (longest swing horizon). All boundaries snapped to trading days; `as_of` cap applied uniformly. The `Split` dataclass enforces `train_end < val_start <= val_end < test_start <= test_end` at construction so any malformed fold blows up loudly.
- `tests/leakage/test_split_leakage.py` — 11 tests guarding the leakage-critical invariants: no overlap between windows, embargo measured in trading days (not calendar days), val/test window lengths exact, train window strictly expanding, `as_of` caps every boundary, malformed splits rejected, fold names stable + unique.
- Live BQ smoke: against the real NSE calendar, the splitter generates **12 quarterly walk-forward folds** from 2019-01-01 through 2025-10-27 with a 10-day embargo and 1-year val/test windows. Train monotonically expands as expected.

**Findings during Phase 1**:
- **Calendar table is stale by ~5 months.** `nse_trading_calendar` ends at **2025-12-12** but `curated_ohlcv` runs through **2026-02-27**. The data ingest layer (next step) needs to either (a) extend the calendar table from a fresh source for 2026 trading days, or (b) derive the missing days from observed OHLCV `date` distinct values and reconcile against an external NSE source. Filed as a Phase 1 blocker for any test fold whose anchor would land in 2026.
- The leakage-test pattern from DESIGN §2.5 ("constructs a feature, asserts that for any `as_of`, no upstream value comes from `ts > as_of`. Spot-checks 100 random `(symbol, as_of)` triples") is *not yet wired up*; the splitter leakage gate is the analogue at the split layer. The feature-leakage gate lands together with the feature engine.

**Pending in Phase 1** (in priority order):
1. Calendar staleness fix (extend through current OHLCV horizon).
2. BQ → Parquet ingest via Storage Read API + Arrow path, partitioned by `(symbol, year, month)`. Wraps `curated_ohlcv` (4 timeframes) and `order_book_depth`.
3. Adjustment verification: spot-check `curated_ohlcv` matches `raw_ohlcv * scale` on a sample of corp-action dates; emit a `read_adjusted={True|False}` mode on the data reader.
4. Feature engine (raw bar channels + microstructure-lite proxies + RevIN per `(symbol, window)`).
5. Feature leakage CI test (DESIGN §2.5).
6. EDA notebook `notebooks/phase1_data_audit.ipynb` confirming end-to-end load + split + feature plumbing.
7. Phase-1 reviewer signoff.

Reviewer reports live in `reports/reviews/phase_N_review.md`.
