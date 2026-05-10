# Phase 2 Review — daity_v3 (SSL Pretraining)

**Reviewer:** Independent reviewer agent
**Date:** 2026-05-10
**Scope:** Phase 2 deliverables per `DESIGN.md` §7 — `daity/models/{revin,
tokenizer, backbone, heads}.py`, `daity/data/{datasets, tokenizer_targets}.py`,
`daity/training/{losses, pretrain}.py`, `daity/scripts/pretrain_cli.py`,
`configs/train/pretrain.yaml`, `tests/unit/{test_revin, test_tokenizer,
test_backbone, test_heads_and_losses, test_datasets,
test_pretrain_module}.py`, `tests/leakage/`,
`notebooks/phase_2_writeup.{py,ipynb}`, and `docs/vast_bootstrap.md`. Out
of scope: Phase-1 carry-over fixes (those have a separate response doc).

---

## Executive summary

The architectural skeleton is honest — RevIN's reversibility contract is
locked tight, the tokenizer + backbone + 4 SSL heads compose into a
verifiable end-to-end graph, the dataset's `as_of` cap is applied at
read time and again at sample-build time, and the new `_latest_end_ts`
data-availability cap is conceptually right (not "spin forever when
as_of runs ahead of ingest"). 379 tests pass; lint is clean on the
Phase-2 modules. The notebook re-executes cleanly and its numbers
broadly match STATUS.md's claims (within ~2% — float/MPS noise).

That said, **Phase 2 ships with one Critical run-blocker, four Major
correctness/contract-drift issues, and a number of smaller documentation
slips.** Headline issues:

1. **No checkpoint callback is wired.** The YAML configures
   `checkpoint_every_n_min: 120` (DESIGN §3.2 — every 2 hours) but the
   CLI never reads it and never registers a `ModelCheckpoint`. A 50K-step,
   2–3 day Vast run that crashes / preempts (Vast preempts routinely)
   loses **all** training progress. There is no mid-run artifact at all.
2. **The contrastive head is mathematically degenerate, not just a
   placeholder.** With the causal pass, position-0's hidden state only
   attends to itself (the FORECAST parameter, broadcast across batch).
   So `z_c` is the **same vector for every sample in the batch**, which
   collapses InfoNCE to `log(B)` — confirmed by the smoke (2.08 ≈ log 8).
   The 0.3-weighted contrastive term contributes a *constant* with no
   useful gradient signal. STATUS calls this a "placeholder" but does
   not disclose the math; DESIGN §3.2's cross-symbol same-date InfoNCE
   was not implemented and there is no DESIGN amendment.
3. **EMA is configured but never wired up.** `cfg.ema_decay = 0.999` is
   stored on the dataclass and never read. DESIGN §3.3 names EMA
   evaluation as a hard requirement; the YAML has it; no module
   consumes it.
4. **STATUS.md overclaims.** A leakage gate it lists by exact filename
   (`tests/leakage/test_dataset_leakage.py`) does not exist. The forecast
   head's "24 tests" count is actually 25 by the same definition. The
   4-objective DESIGN deviation (contrastive) is not flagged.
5. **The `_latest_end_ts` data-availability cap is wrong for daily
   `forecast_scale`** (uses `INTERVAL_DURATIONS["day"] = 6h15m` as
   `bar_duration`, but daily bars are spaced 24 h apart). This isn't a
   leakage hole (the cache is gated by as_of) but it overestimates
   eligible end_ts and produces excessive `_build_sample` rejections.

The forecast-head leakage surface, the apply_cached RevIN mode, the
non-mutating `_compute_loss`, and the worker reseeding all check out.
The bf16 precision config has no obvious underflow path. Total tests
agree (379 / 117 Phase-2). Lint is clean on Phase-2 files (the global
ruff failures are pre-existing Phase-0/1 script cleanliness).

**Recommendation: block Phase 3 until Critical 1 is fixed and Majors
2–4 are either fixed or explicitly added as DESIGN amendments + STATUS
disclosures**, since both will surface as Phase-3 surprises (no
mid-Vast checkpoints to start posttrain from; the contrastive
"objective" produces no representation for downstream).

---

## Findings

### 1. [Critical] No `ModelCheckpoint` callback — a 50K-step Vast run that crashes loses everything

- **File:** `daity/scripts/pretrain_cli.py:212-224` and
  `configs/train/pretrain.yaml:59`
- **What's wrong:** The YAML declares
  `checkpoint_every_n_min: 120  # every 2 hours per DESIGN §3.2`
  but `pretrain_cli.py` constructs the Trainer with no `callbacks=[...]`
  argument and never reads `checkpoint_every_n_min`:
  ```python
  trainer = L.Trainer(
      max_steps=pretrain_cfg.max_steps,
      accelerator=chosen_device,
      devices=1,
      precision=merged.get("precision", "32-true"),
      val_check_interval=merged.get("val_every_n_steps", 1000),
      limit_val_batches=merged.get("limit_val_batches", 50),
      log_every_n_steps=10,
      default_root_dir=log_dir,
      logger=L.pytorch.loggers.CSVLogger(save_dir=str(log_dir)),
      enable_progress_bar=True,
  )
  ```
  Without an explicit `ModelCheckpoint(every_n_train_steps=N)` or
  `every_n_minutes=120`, Lightning's default behavior is to save only
  on `fit` exit (and only the last). DESIGN §3.2 mandates "We
  checkpoint every 2 hours and keep the best three by val MSE for
  downstream initialization." Neither happens.
  `docs/vast_bootstrap.md:124-126` further claims "Checkpoints land
  under `runs/pretrain/` every 2 hours" — that is also wrong as
  shipped.
- **Why it matters:** The Vast run is 2–3 days. Vast frequently preempts
  rented GPU instances (especially A100s in high-demand markets);
  network hiccups kill ssh/tmux sessions; a single OOM on a slightly
  smaller card kills the process. Without periodic checkpoints, every
  failure-mode resets the run from step 0. Phase 3 (post-training)
  acceptance depends on having "the best three by val MSE for
  downstream initialization" — those three checkpoints simply will
  not exist when this run finishes.
- **Suggested fix:**
  1. Add a `ModelCheckpoint` callback in `pretrain_cli.py`:
     ```python
     from lightning.pytorch.callbacks import ModelCheckpoint
     ckpt = ModelCheckpoint(
         dirpath=log_dir / "checkpoints",
         filename="step={step}-val_loss={val_loss:.4f}",
         save_top_k=3,
         monitor="val_loss",
         mode="min",
         every_n_train_steps=int(merged.get("checkpoint_every_n_steps", 2000)),
         save_last=True,
     )
     trainer = L.Trainer(..., callbacks=[ckpt])
     ```
     Drive cadence by step (deterministic) rather than wall-clock
     minutes — `every_n_minutes` is also supported but Lightning's API
     for it (`train_time_interval`) is hairier and survives interrupted
     runs less cleanly.
  2. Add a CLI test that constructs the Trainer and asserts
     `len(trainer.checkpoint_callbacks) >= 1` — pins the regression.
  3. Update `docs/vast_bootstrap.md`'s "Checkpoints land under
     `runs/pretrain/` every 2 hours" claim to match the actual cadence
     (or strike the line until the callback lands).

### 2. [Major] Contrastive loss is mathematically degenerate — `z_c` is constant across the batch and InfoNCE collapses to `log(B)`

- **File:** `daity/training/pretrain.py:255-260` (with
  `daity/models/tokenizer.py:158-159` for the broadcast FORECAST token)
- **What's wrong:** The Phase-2 contrastive objective is implemented as:
  ```python
  z_nc = self.contrastive_head(hidden_nc[:, 0])
  z_c = self.contrastive_head(hidden_c[:, 0])
  loss_contrast = info_nce_loss(z_nc, z_c, ...)
  ```
  with `hidden_c = self.backbone(tokens, causal=True)`. With
  `causal=True`, position 0's hidden state attends only to position 0
  itself. Position 0 is the broadcast FORECAST parameter
  (`tokenizer.py:158-159`: `self.forecast_token = nn.Parameter(torch.zeros(1, 1, d_model))`,
  expanded to `(B, 1, d_model)` at forward time — the *same* vector
  across the batch). So `hidden_c[:, 0]` is identical across batch
  rows; therefore `z_c` is identical across rows.
  InfoNCE on (`z_a`, `z_b` where `z_b[i] = const`): every column of the
  similarity matrix `z_a @ z_b.T` is a constant times the same column
  vector → all logits per row are equal → cross-entropy with target
  `arange(B)` collapses to `log B` for every row.
  Empirically confirmed by the writeup (smoke logs `contrast` ≈ 2.08
  ≈ log 8). This is not "the placeholder is uninformative"; it's "the
  placeholder is mathematically forced to a constant with zero
  representational signal".
  Notes that compound the issue:
  - DESIGN §3.2 specifies "For a given trading minute (or day), pull
    together representations of the same date across different symbols
    *only if* their returns are co-moving above a threshold; push apart
    unrelated days." The shipped code does none of this. There is no
    DESIGN amendment.
  - `daity/training/pretrain.py:25-29` and `notebooks/phase_2_writeup.py:240-242`
    acknowledge this is a placeholder, but neither calls out the
    mathematical degeneracy. STATUS.md line 64 (`contrast plateau ~2.08 ≈ log(8)
    (expected for placeholder positives at batch 8)`) describes the
    symptom without the cause; a Phase-3 reviewer reading STATUS will
    see "expected" and stop digging.
  - 0.3 × log B is added to the total loss every step (~0.62 at B=8,
    ~2.18 at B=320 if val batch ever scales to that). The optimizer
    therefore sees a 0.3 × log B *constant* offset to the train_loss
    metric, which the smoke logs as `total = recon + 0.5×ntp +
    0.3×log(B) + 0.5×forecast` — anyone reading the curves will see
    a mysterious floor.
- **Why it matters:** The "placeholder" framing is misleading. A
  placeholder that produces some weak but real representation can be
  upgraded incrementally; this one produces zero gradient signal to
  the encoder through `z_c` (the gradient is zero by symmetry — every
  positive looks identical to every other positive). Only `z_a`'s side
  contributes a gradient, and that gradient is "make `z_a` distinct
  per row but indistinguishable from a constant target", which is
  noise. Phase-3's posttrain head reads the FORECAST token; if the
  Phase-2 contrastive objective taught the FORECAST token to *be*
  identical to a constant projection, it has actively pushed the
  representation toward uselessness.
- **Suggested fix:** Pick one:
  1. **Honest disable.** Add `cfg.use_contrastive: bool = False`
     (default False); the implementation is reserved for Phase 3 when
     the cross-symbol same-date positive pairing actually exists. The
     0.3 weight currently in the YAML stops dragging a constant into
     the total loss.
  2. **Honest placeholder.** Make `z_c` something non-degenerate —
     e.g., re-use the non-causal pass output (drop the second backbone
     forward) and pair the FORECAST tokens of two *different augmented
     views* (mask different patches; or pair a non-causal-FORECAST
     with a "drop a random scale" view). Both are real SSL signals.
  3. **DESIGN amendment.** If the Phase-2 implementation is to ship
     as-is, add a DESIGN amendment 10 explicitly downgrading the
     contrastive objective to a no-op for v1 with the rationale that
     the cross-symbol pairing requires Phase-3 batch-construction
     infrastructure. Update STATUS to disclose the math (not just the
     symptom) and update the writeup notebook to drop the
     "PASS contrast loss finite throughout (expected to plateau)" line
     in favor of "expected by construction; see DESIGN amend 10".

### 3. [Major] EMA is configured but never wired — DESIGN §3.3 contract violated

- **File:** `daity/training/pretrain.py:111-112`,
  `configs/train/pretrain.yaml:54`, `daity/scripts/pretrain_cli.py`
- **What's wrong:** The dataclass has `ema_decay: float = 0.999` (line
  112). The YAML sets `ema_decay: 0.999` (line 54).
  `grep -rn "ema" daity/` finds zero non-config references in the
  training code. There is no `EMACallback`, no `swa_utils.AveragedModel`,
  no shadow-weights tracking, no `validation_step` that swaps to EMA
  weights for the eval pass.
  DESIGN §3.3 (post-training section, but the pretrain module
  pre-establishes the EMA state for Phase 3 to inherit) names "EMA of
  weights (decay 0.999) used for evaluation"; the pretrain.py docstring
  at line 34 says `EMA: decay 0.999 of every parameter, evaluated
  separately for val metrics` but this is aspirational text — the
  module body never reads `cfg.ema_decay`.
- **Why it matters:** Phase-2's val_loss is what decides which
  checkpoints are "best three" (DESIGN §3.2). Without EMA, val_loss is
  measured on the raw weights at val_check_interval — significantly
  noisier than EMA-smoothed val. Phase 3 then loads "best by val_loss"
  which is a noisy criterion. Beyond the metric, Phase 3 is supposed
  to *initialize* its supervised heads from the EMA-state checkpoint;
  no EMA state is saved.
- **Suggested fix:**
  1. Drop a small `EMACallback` into `daity/training/callbacks.py` (a
     `lightning.pytorch.callbacks.Callback` that maintains a shadow
     `state_dict` updated each `on_train_batch_end` and swaps weights
     in `on_validation_start` / restores in `on_validation_end`).
     `lightning_bolts` ships an EMA you can lift; or use
     `torch.optim.swa_utils.AveragedModel` with `avg_fn` set to the
     standard EMA recurrence.
  2. Wire it into the CLI Trainer's `callbacks=[..., EMACallback(decay=cfg.ema_decay)]`.
  3. Add `tests/unit/test_pretrain_callbacks.py` that asserts the
     shadow weights track the live weights with the configured decay
     after a few training steps.
  4. If EMA is genuinely deferred to Phase 3, then *delete*
     `cfg.ema_decay` (or move it to a Phase-3 config) and strike the
     pretrain.py:34 docstring line. Otherwise the configured-but-unused
     hyperparameter creates a paper trail of "we did EMA" that doesn't
     match the artifact.

### 4. [Major] STATUS.md drifts: the named leakage-gate file does not exist; lint claim is over-broad

- **File:** `STATUS.md:76` and `STATUS.md:44`
- **What's wrong:**
  1. STATUS.md line 76:
     ```
     - `OHLCVPretrainDataset` honors `as_of` (no future leakage) → `tests/leakage/test_dataset_leakage.py`
     ```
     `ls tests/leakage/` returns only `test_features_leakage.py`,
     `test_parquet_leakage.py`, `test_split_leakage.py` —
     `test_dataset_leakage.py` does not exist. `grep -r
     test_dataset_leakage` confirms STATUS is the only mention. The
     **dataset-level** leakage coverage that does exist lives in
     `tests/unit/test_datasets.py::test_no_sample_extends_past_as_of`
     and `::test_forecast_tail_respects_as_of_cap` and
     `::test_forecast_dataset_handles_as_of_far_past_data` — these are
     under `tests/unit/`, not `tests/leakage/`, and so are NOT
     selected by the project's `pytest -m leakage` CI gate marker
     (`tests/leakage/*` files do `pytestmark = pytest.mark.leakage`;
     unit tests do not). A real CI run with `-m leakage` would skip
     all dataset-level checks entirely.
  2. STATUS.md line 9 says "117 unit tests" — verified accurate
     (`pytest tests/unit/test_revin.py test_tokenizer.py test_backbone.py
     test_heads_and_losses.py test_datasets.py test_pretrain_module.py
     --co -q | tail` returns 117 tests).
  3. STATUS.md line 9 also says "including 24 for the multi-horizon
     forecast head". An equivalent collect (`pytest -k 'forecast or
     apply_cached or multi_horizon'`) returns 25, not 24. Off by one.
  4. STATUS.md line 44: "Lint clean on every Phase-2 module." Verified
     accurate **for Phase-2 modules only** (`uv run ruff check daity/models/
     daity/training/ daity/data/datasets.py daity/data/tokenizer_targets.py
     daity/scripts/pretrain_cli.py` → "All checks passed!"). However
     `uv run ruff check daity/` returns **30 errors** (in
     `daity/scripts/{check_staleness, detect_corp_actions, fix_daily_ts,
     fix_pointwise, refresh_calendar, refresh_ohlcv}.py` and
     `daity/utils/env.py`). Phase 1 STATUS line 17 also claimed "Lint
     clean on Phase-1 modules", which appears to have regressed since
     it was claimed (or never been true at the global level). At
     minimum the Phase-2 STATUS claim should be "Lint clean on Phase-2
     modules" rather than the more global-sounding wording at line 44.
- **Why it matters:** STATUS.md is the single source of truth for what
  Phase-2 has shipped (CLAUDE.md §3 calls it the "single source of
  truth for what's done / in flight / blocked"). When it names a
  leakage-gate file that doesn't exist, a reviewer who CTRL-F's the
  filename loses confidence in the rest of STATUS. The `pytest -m
  leakage` selector misses dataset-level guarantees, which is exactly
  the test discipline DESIGN §2.5 created the marker to enforce.
- **Suggested fix:**
  1. Either create `tests/leakage/test_dataset_leakage.py` (move /
     copy the relevant cases from `tests/unit/test_datasets.py`,
     adding `pytestmark = pytest.mark.leakage`), or strike the line
     from STATUS and replace with "→ `tests/unit/test_datasets.py`
     (note: file lives under unit/ not leakage/; `pytest -m leakage`
     will *not* select it)".
  2. Recount the forecast-head test set; correct STATUS to 25 (or
     adjust the count's definition to be precise — e.g., "tests
     touching forecast functionality" vs "tests in test_pretrain_module.py
     for the forecast head").
  3. Replace "Lint clean on every Phase-2 module" with the narrower
     claim that's actually true: "Lint clean on `daity/models/`,
     `daity/training/`, the new `daity/data/` Phase-2 additions, and
     the Phase-2 CLI; pre-existing scripts in `daity/scripts/` carry
     the same ruff debt as Phase 1 (Phase-1 review minor #16)".

### 5. [Major] `_latest_end_ts` `bar_duration` is wrong for daily forecast_scale — overestimates eligible end_ts (cache-gated, so safe but inefficient)

- **File:** `daity/data/datasets.py:266-286`
- **What's wrong:** The data-availability cap math is:
  ```python
  if self.forecast_scale in INTERVAL_DURATIONS:
      bar_duration = INTERVAL_DURATIONS[self.forecast_scale]
  else:
      bar_duration = timedelta(days=1)
  horizon = bar_duration * self.forecast_n_bars
  leakage_bound = self.as_of - horizon
  ...
  data_bound = last_ts - horizon + bar_duration
  ```
  For `forecast_scale="day"`, `INTERVAL_DURATIONS["day"] = timedelta(hours=6, minutes=15)`
  (per `daity/data/candles.py:34-40`, the NSE 09:15→15:30 IST trading
  span). The actual *spacing* between two consecutive daily bar
  timestamps is 24 h (next day's 15:30 IST close). So `bar_duration`
  is 6h15m where 24h would be correct for end_ts spacing.
  Trace: `forecast_n_bars=10` daily bars. Real horizon = 10 × 24h = 10
  days. The code computes `horizon = 10 × 6h15m = 62.5h ≈ 2.6 days`.
  `leakage_bound = as_of - 2.6d` (overly permissive: end_ts can be
  almost up to as_of); `data_bound = last_daily_ts - 2.6d + 6.25h ≈
  last_daily_ts - 2.34d`.
  The CACHE itself is gated on `effective_close ≤ as_of`, so daily
  bars beyond as_of are simply absent from `ser.ts`. So
  `_build_sample` correctly returns `None` when `f_right > len(ser.ts)`
  — no actual leakage. **But** the iterator's eligible range is
  inflated, so `_draw_end_ts` proposes many end_ts values for which
  `_build_sample` will then reject because the future tail isn't in
  the cache. With the regime weighting (recent 18m up-weighted ×2),
  this rejection rate is highest at the tightest end of the eligible
  range — exactly where DESIGN says we WANT the most samples.
- **Why it matters:** Today no production config sets
  `forecast_scale="day"` (default is `"5m"` and the 5m math is
  correct). But the dataclass + CLI accept `forecast_scale` as a
  config, the production YAML at `configs/train/pretrain.yaml:42`
  exposes it as a top-level toggle, and any future ablation that
  flips to `forecast_scale: day` will silently train with biased
  sampling. There is also no test that exercises the daily
  forecast-scale path (`test_datasets.py` only uses `forecast_scale="5m"`),
  so this is unguarded.
- **Suggested fix:**
  1. Compute the real spacing per scale from
     `daity.data.candles.effective_close` — for daily, the spacing is
     `timedelta(days=1)` because the daily ts increments by 1 calendar
     day. Use a `BAR_SPACING` table:
     ```python
     # In candles.py:
     BAR_SPACING: dict[str, timedelta] = {
         "5m": timedelta(minutes=5),
         "15m": timedelta(minutes=15),
         "60m": timedelta(hours=1),
         "day": timedelta(days=1),    # NOT 6h15m — that's intra-day extent
     }
     ```
     and have `datasets.py` use `BAR_SPACING` in `_latest_end_ts`,
     keeping `INTERVAL_DURATIONS` for the effective-close gate.
  2. Add `tests/unit/test_datasets.py::test_latest_end_ts_for_daily_forecast_scale`
     that pins the eligible-range math against a synthetic daily store.
  3. Add a CLI-time check that `cfg.forecast_scale` ∈ `cfg.scales`
     (today only the dataset's `__post_init__` checks
     `forecast_scale ∈ window_bars`; if a user shrinks `cfg.scales`
     below `cfg.forecast_scale` the failure surfaces at first
     `_compute_loss` call as a `KeyError` from
     `tokenizer.revins[forecast_scale]`).

### 6. [Major] DESIGN §3.2 contract drift: implementation runs the backbone TWICE per step (causal + non-causal); DESIGN's NTP head was meant to be a separate head, not a separate forward

- **File:** `daity/training/pretrain.py:233, 245`
- **What's wrong:** Per step:
  ```python
  hidden_nc = self.backbone(masked_tokens, causal=False)  # for recon + contrastive + forecast
  ...
  hidden_c  = self.backbone(tokens,        causal=True)   # for NTP
  ```
  Two full backbone forward passes per training step. DESIGN §3.2
  reads "A separate causal head with attention mask predicts the next
  patch from the prefix" — "separate head", not "separate forward
  pass over the same backbone". The intent (per the standard PatchTST
  / Chronos pattern this design cites) is that the backbone runs
  once with whatever attention pattern the architecture decides, and
  the heads attach to the appropriate slices of the single output.
  Practical impact:
  1. **Compute doubled.** A 2–3 day Vast run is now a 3–6 day run.
     STATUS line 60-61's smoke timing (200 steps in ~5s on MPS) is
     measured *with* the double-forward, so it's not "actually fast";
     production timing extrapolation matters more.
  2. **Memory doubled** (or close to it) because both `hidden_nc` and
     `hidden_c` are alive simultaneously through the contrastive
     loss. `bf16-mixed` on A100-40G at batch 64 has headroom for one
     activation graph at the configured size; two may not fit
     gracefully and may force `batch_size=32`, halving throughput
     again.
  3. The contrastive head reads `hidden_c[:, 0]` which (per finding 2)
     is a constant. The second backbone forward exists *almost
     exclusively* to feed the NTP head. A correct implementation
     either (a) does ONE non-causal forward and uses a causal *mask
     during the NTP head only* (the DESIGN-implied path), or (b)
     accepts the two forwards and explicitly documents the compute
     cost.
- **Why it matters:** Phase 2 acceptance is partly "checkpoint < 200
  MB", which is met, and partly "val masked-MSE plateau reached" — the
  latter requires actually completing the run. If the doubled-compute
  pushes the run past Vast budget, Phase 2 doesn't finish.
  Independently of the cost, the contract drift undermines the
  pretrain.py docstring (line 16-21: "Backbone runs twice — a.
  non-causal pass...; b. causal pass...") which contradicts the DESIGN
  §3.2 "separate head" wording.
- **Suggested fix:**
  1. **Preferred:** combine into one non-causal backbone forward,
     then run NTP via causal masking *inside the NTP head*
     (a tiny causal transformer head, 1–2 layers). This matches
     DESIGN's "separate head" reading and halves backbone compute.
     Reorganize as:
     ```python
     hidden_nc = self.backbone(masked_tokens, causal=False)
     # NTP: small causal head reads hidden_nc[:, 1:] in causal-prefix mode.
     ntp_out = self.ntp_head(hidden_nc[:, 1:], causal=True)
     ```
     The `NextPatchHead` is currently a vanilla MLP (`heads.py:78-108`);
     adding a 1-layer causal attention block keeps it under 1M params
     and removes the second backbone forward entirely.
  2. **Acceptable fallback:** keep the double forward but document the
     compute cost in DESIGN amendment 11 and re-budget Phase-2 timing
     in `docs/vast_bootstrap.md` (currently claims 2-3 days at 50K
     steps — that's the single-forward number). Add a
     gradient-checkpointing flag to the CLI to recover memory headroom.
  3. Either way: add a unit test that asserts the number of backbone
     forward calls per `_compute_loss` (using a Mock-spy on
     `self.backbone.forward`) so future refactors are guarded.

### 7. [Major] No git_sha / config_hash / as_of provenance triple emitted by the CLI — CLAUDE.md "Reproducibility" rule violated for Phase-2 artifacts

- **File:** `daity/scripts/pretrain_cli.py`
- **What's wrong:** CLAUDE.md §6 (Reproducibility):
  > Every artifact (cache slice, checkpoint, backtest run) is tagged
  > with `(git_sha, config_hash, as_of)`. The smoke test in CI verifies
  > that the same triple produces the same result.
  The pretrain CLI:
  1. Resolves the YAML, builds a `PretrainConfig`, builds the
     module, builds the Trainer, calls `trainer.fit(...)`.
  2. Writes CSVLogger metrics to `runs/pretrain/lightning_logs/version_N/`.
  3. Writes nothing else.
  No `git_sha` is captured (no `subprocess.check_output(['git',
  'rev-parse', 'HEAD'])` anywhere in the file). No `config_hash` is
  emitted (the YAML is read but not hashed). The `train_end` /
  `val_end` `as_of` values are stored in the YAML but not propagated
  into the run's metadata. If two operators on the same SHA + same
  YAML run `daity-pretrain` and one of them has uncommitted local
  changes, both produce identical-looking `runs/pretrain/version_N`
  directories.
  This was Phase-1 review's finding #13 (same root cause: no `make
  data` target emits the triple). Phase 2 has structurally added the
  *most expensive* artifact in the project (a 2-3 day pretrain
  checkpoint) without addressing the issue.
- **Why it matters:** Phase 3+ posttrain initializes from a Phase-2
  checkpoint. Without a triple-tag, "which checkpoint did I post-train
  off of?" has no machine-checkable answer. Phase 5 backtest
  reproducibility (DESIGN §5.3) bottoms out at this layer.
- **Suggested fix:**
  1. At the top of `main(...)`, capture:
     ```python
     git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
     git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
     config_hash = hashlib.sha256(json.dumps(merged, sort_keys=True, default=str).encode()).hexdigest()[:12]
     ```
     Emit them as a JSON sidecar `runs/pretrain/version_N/provenance.json`
     written before `trainer.fit(...)` returns. Store the same triple
     into Lightning's `hparams` so it's checkpoint-embedded.
  2. Refuse to start the run with `git_dirty=True` unless `--allow-dirty`
     is set; print the SHA + first-line-of-each-modified-file as part
     of the run banner (the existing `console.print` block at lines
     195-203 is the natural place).
  3. Add `tests/unit/test_pretrain_cli.py::test_provenance_emitted`
     using a `tmp_path` git repo fixture; assert the JSON exists and
     contains the SHA.

### 8. [Minor] Forecast-tail leakage: end_ts == as_of bounded but not strictly excluded — boundary case undertested

- **File:** `daity/data/datasets.py:215-238`,
  `tests/unit/test_datasets.py:300-371`
- **What's wrong:** The forecast-tail leakage gate works as follows
  in `_build_sample`:
  - The forecast scale's series cache `_series` is gated by
    `effective_close ≤ as_of` (line 162-169). Future bars whose
    effective_close > as_of are absent from `ser.ts`.
  - `f_right = f_left + self.forecast_n_bars` and `if f_right >
    len(ser.ts): return None` (lines 234-238).
  This is *correct* for safety but a single boundary case is unguarded:
  with as_of EXACTLY at a future bar's effective_close, that bar is
  included (the gate is `<=`, so a bar with `effective_close == as_of`
  is in the cache). That's the right semantic, but it should be
  asserted as a leakage-gate test. Today the only forecast-tail leakage
  test (`test_forecast_tail_respects_as_of_cap`) constructs a fixture
  where `latest < earliest` and asserts that — it does NOT step
  through 100 random `(end_ts, as_of)` triples and check
  `effective_close(forecast_scale, future_bar.ts) ≤ as_of` for every
  bar in `sample.forecast`.
  Compounding: the test in `tests/unit/`, not `tests/leakage/`, so
  `pytest -m leakage` (the canonical CI gate selector) skips it
  entirely (see finding 4.1).
- **Why it matters:** The current implementation IS safe (the cache
  gate plus the f_right bound do the right thing), but a future
  refactor that, say, inlines `ser.ts` access without the cache gate,
  or that reorders the as_of cap check, would not be caught. DESIGN
  §2.5 mandates a "spot-check 100 random (symbol, as_of) triples"
  test for exactly the surfaces where leakage is the worst-case
  failure.
- **Suggested fix:**
  1. Create `tests/leakage/test_dataset_leakage.py` with
     `pytestmark = pytest.mark.leakage`. Move/copy the existing
     `tests/unit/test_datasets.py::test_forecast_tail_respects_as_of_cap`
     and `test_no_sample_extends_past_as_of` into it.
  2. Add a 100-triple spot-check that asserts, for every drawn
     sample, `every future bar's effective_close <= as_of`:
     ```python
     for _ in range(100):
         sample = next(iter(sampler))
         for ts in sample.forecast_ts:    # add a forecast_ts field to WindowSample
             assert ts + INTERVAL_DURATIONS[forecast_scale] <= as_of
     ```
     Adding a `forecast_ts: np.ndarray` field to `WindowSample` is a
     2-line change.
  3. Same shape for the input window (assert every input bar's
     effective_close ≤ as_of). The existing test only checks
     `sample.end_ts <= as_of`, which is a derived condition.

### 9. [Minor] `max_seq_len: 128` in YAML has 36% headroom — fragile under window_bars edits

- **File:** `configs/train/pretrain.yaml:33`,
  `daity/training/pretrain.py:90`,
  `daity/models/backbone.py:113-117, 200-205`
- **What's wrong:** The production config sets `max_seq_len: 128`. The
  RoPE buffer (`backbone.py:113-117`) is sized to `max_seq_len`, and
  the test `test_backbone_max_seq_len_enforced` confirms a longer
  sequence raises (correct fail-loud). With production windows:
  - 5m × 256 → 31 patches
  - 15m × 192 → 23 patches
  - 60m × 168 → 20 patches
  - day × 64 → 7 patches
  Sum = 81; +1 FORECAST = 82. Fits 128 (36% headroom).
  But the architecture is meant for Phase-4 OB fusion which adds OB
  encoder tokens to the upstream sequence (DESIGN §3.4: "K ≈ 64
  snapshots"). The pretrain backbone doesn't see OB tokens, so 128 is
  fine for Phase 2. The fragility is on the *configuration* side: a
  Phase-2 ablation that sets `5m: 512` to test longer context blows
  past 128, and the failure mode is a hard-error at first batch
  (raises `RuntimeError`) — not catastrophic, but a 36% headroom
  with no test that asserts "the configured window_bars actually fit"
  is one tweak from a confused 30-minute debugging session.
- **Suggested fix:** Add a startup-time assertion in
  `pretrain_cli.py` (or `PretrainModule.__init__`):
  ```python
  total_patches = sum(
      (n - cfg.patch_len) // cfg.patch_stride + 1
      for n in cfg.window_bars.values()
  ) + 1   # FORECAST
  if total_patches > cfg.max_seq_len:
      raise ValueError(
          f"Token sequence length {total_patches} exceeds max_seq_len "
          f"{cfg.max_seq_len}. Either increase max_seq_len or shrink "
          f"window_bars."
      )
  ```
  with a unit test `test_window_bars_overflow_max_seq_len` that
  exercises both the pass and fail paths.

### 10. [Minor] `_apply_rope` re-stack/flatten is correct but the test for "rotation-by-position-1 is non-trivial" tests permissively

- **File:** `daity/models/backbone.py:64-81`,
  `tests/unit/test_backbone.py:50-57`
- **What's wrong:** The RoPE implementation pairs `(x_even, x_odd)`
  per pair index (i.e., `x[..., 0::2]` and `x[..., 1::2]`) and rotates
  each pair by frequency `inv_freq[k] = 1/base^(2k/d_head)` for pair
  `k`. Standard. Verified shape-correct + identity at position 0 +
  per-position non-identity. However:
  - There's no test that locks in the exact relative-position
    invariance: `attn(q_at_t, k_at_s) = attn(q_at_t', k_at_s')` when
    `t-s == t'-s'`. Without that, a future refactor could shift the
    sin/cos ordering by half-period and tests would still pass (the
    "position 0 is identity" test fires, "later positions differ"
    test fires, but the relative-distance property — the actual point
    of RoPE — is not pinned).
  - `_apply_rope` always uses `freqs[:T, ...]` from a buffer of size
    `max_seq_len`. If T > max_seq_len, the slice returns a shape <
    (T, d_head) and the broadcast multiplication fails noisily.
    `test_backbone_max_seq_len_enforced` does verify this fails;
    finding 9 above suggests adding a startup-time
    pre-condition so it's not a runtime error in production.
- **Suggested fix:**
  1. Add `test_rope_relative_distance_invariance`: take two tokens
     `q_at_t1 = some_vec`, `k_at_s1 = some_other_vec` and another
     pair at `(t2, s2)` with `t2-s2 == t1-s1`. Compute the SDPA
     pre-softmax score `q_rope.T @ k_rope` — must be equal between
     the two pairs.
  2. Optional: extract the RoPE rotation into a tiny `RoPE` Module so
     the test surface is more explicit.

### 11. [Minor] `OHLCVPretrainDataset._reseeded` discards the `_cache` — every worker starts cold, multi-worker DataLoader uses 4× memory of the cache

- **File:** `daity/data/datasets.py:420-430`
- **What's wrong:**
  ```python
  def _reseeded(s: OHLCVWindowSampler, *, base_seed: int) -> OHLCVWindowSampler:
      return OHLCVWindowSampler(
          store=s.store, symbols=list(s.symbols), as_of=s.as_of,
          window_bars=dict(s.window_bars), channels=tuple(s.channels),
          seed=base_seed,
          ...,
      )
  ```
  The new sampler has its own `_cache` (the `field(default_factory=dict, init=False)`
  fires on each new instance). With `num_workers=4` from the YAML,
  every worker rebuilds the per-(symbol, scale) Series cache from
  scratch on its first sample fetch → 4× the cache memory of a
  single-worker run. With `persistent_workers=False` (Lightning
  default; CLI never sets it), every epoch the cache is discarded
  and rebuilt — but for an `IterableDataset` with `max_steps=50000`,
  Lightning creates the iterator *once* at the start of fit (since
  there's no epoch boundary), so the cache fills once and stays warm.
  Confirmed by the writeup smoke (`num_workers=0`, no observed slowness)
  but the production config's `num_workers=4` will see the 4×
  memory hit on the first ~205 symbols × 4 scales × ~4-7MB each =
  ~24 GB if all symbols × all scales are loaded. The current
  IterableDataset spec randomly samples symbols, so eventually all
  205 are touched.
- **Why it matters:** Hard cap on Vast memory (A100-40GB, but the
  *system RAM* on a typical Vast instance is 32–64 GB for non-A100
  cards and can be tight). The 4× cache duplication is invisible
  until the OOM-killer fires mid-run.
  This is also a worker-id seed-collision quirk: train_ds (seed=0) +
  worker 0 = seed 0; val_ds (seed=1) + worker 0 = seed 1. So **train
  worker 0 and val worker 1 both use seed 1** — same RNG sequence.
  Probably benign (different `as_of` values), but the seed namespace
  isn't disjoint.
- **Suggested fix:**
  1. Pass the `_cache` through `_reseeded` (mutable shared dict) and
     accept that workers race on writes — Series objects are
     immutable post-write, so the worst case is duplicate writes.
     Or, lift the cache into a shared `multiprocessing.Manager`
     dict (heavier).
  2. Cheaper alternative: make the cache lazy + per-(symbol, scale)
     mmap-backed (PyArrow's `ParquetFile.read_row_group` already
     supports this). Then 4 workers reading the same file share OS
     page cache for free.
  3. Disjoin the seed namespace: train uses `seed=0` + workers
     `0..3`; val uses `seed=10000` + workers `10000..10003`.
     Trivial fix in `pretrain_cli.py`.

### 12. [Minor] CLI accepts `forecast_scale` not in `cfg.scales` — fails late at first `_compute_loss`

- **File:** `daity/scripts/pretrain_cli.py:170-184`,
  `daity/training/pretrain.py:99` (defaults
  `forecast_scale: str = "5m"`)
- **What's wrong:** The dataset's `__post_init__` checks `forecast_scale
  ∈ window_bars` (correct). But neither the dataset nor the CLI
  checks `forecast_scale ∈ cfg.scales` (the tokenizer's scales).
  These can drift: a config that sets `cfg.scales=("5m","15m","60m")`
  while keeping `forecast_scale="day"` and `window_bars={..."day":
  ...}` builds successfully (window_bars has day; tokenizer scales
  do not). At first `_compute_loss`, `build_forecast_targets` calls
  `tokenizer.revins[forecast_scale]` and raises a `ValueError`
  (`tokenizer_targets.py:114-119`). The error message is fine; the
  fail-late is wasteful.
- **Suggested fix:** Add to `PretrainConfig.__post_init__` (currently
  none — the dataclass doesn't define one):
  ```python
  def __post_init__(self) -> None:
      if self.forecast_n_patches > 0 and self.forecast_scale not in self.scales:
          raise ValueError(
              f"forecast_scale {self.forecast_scale!r} must be in scales "
              f"{self.scales!r} when forecast_n_patches > 0"
          )
  ```
  Add a unit test `test_pretrain_config_rejects_forecast_scale_not_in_scales`.

### 13. [Minor] `_compute_loss` non-mutation is correct, but the docstring undersells what the rebind protects against

- **File:** `daity/training/pretrain.py:207-220`
- **What's wrong:** The intent is right:
  ```python
  future_bars: torch.Tensor | None = batch.get(FORECAST_FUTURE_KEY)
  if future_bars is not None:
      batch = {k: v for k, v in batch.items() if k != FORECAST_FUTURE_KEY}
  ```
  The local rebind protects `batch` (and therefore the caller's dict)
  from being mutated, even when the path *doesn't* fire (the get returns
  None). Verified: `tests/unit/test_pretrain_module.py::test_overfit_one_batch_reduces_loss`
  reuses the same batch dict across 100 steps and that requires
  non-mutation. Good.
  However the docstring (line 213-214) says "Non-mutating: we work on
  a shallow copy of `batch` so callers can re-use the same dict
  across multiple steps". The code only takes a shallow copy when
  `future_bars is not None`. When future_bars is None, the local
  `batch` is the SAME object as the caller's. Today no downstream
  code path (the targets builder, the tokenizer, the backbone, or the
  losses) actually mutates the dict, so the no-future-bars path is
  also non-mutating in practice. But the docstring's "we work on a
  shallow copy" is overly broad — strictly it's "we work on a shallow
  copy when forecast bars are present, and rely on no-downstream-mutation
  when they aren't".
- **Why it matters:** Tiny robustness issue. A future refactor that
  adds a `batch[some_key] = ...` somewhere in `_compute_loss` would
  silently break the no-forecast path (the test currently passes it
  in the writeup notebook + most unit tests).
- **Suggested fix:** Always rebind unconditionally:
  ```python
  batch = {k: v for k, v in batch.items() if k != FORECAST_FUTURE_KEY}
  ```
  One-line; trivially correct; docstring becomes accurate. The cost
  is a dict copy per step, which is microseconds vs the full
  forward+backward.

### 14. [Minor] Smoke notebook's "PASS" assertion is weaker than the writeup framing implies

- **File:** `notebooks/phase_2_writeup.py:226-250`
- **What's wrong:** The acceptance assertions all use `assert l_X < e_X`
  where `l_X` is the avg of last 20 steps and `e_X` is the avg of
  first 20. So the gate passes if the loss drops by *any* amount —
  even by 0.01% it's a pass. STATUS line 60-65 quotes the actual
  drops (~88-89%) which is decent, but the test gate is far weaker
  than what STATUS implies.
  Phase-1's writeup had the same shape (asserts existence, claims big
  improvements in prose). The Phase-1 reviewer #1 caught this for
  prose-vs-data drift (finding 10). Phase 2's writeup repeats the
  pattern in a slightly different way.
- **Suggested fix:** Tighten the assertions to match STATUS. E.g.:
  ```python
  assert l_recon < 0.5 * e_recon, ...
  assert l_ntp < 0.5 * e_ntp, ...
  ```
  If the gate slips below 50% improvement on a 200-step CPU smoke,
  *something* has regressed and the notebook's "PASS" should fire.

### 15. [Minor] `vast_bootstrap.md` references the SA key path that Phase-1 review flagged still in repo at 0644

- **File:** `docs/vast_bootstrap.md:43-58`
- **What's wrong:** The bootstrap doc (correctly, in spirit) says the
  SA key should live at `~/.config/gcloud/daity-sa.json` on Vast and
  be chmod'd 600. But it instructs the operator to `scp` from "your
  laptop ~/.config/gcloud/daity-sa.json" — Phase-1 review
  Critical #3 flagged that the local key is still at `./.sa-key.json`
  in the repo with mode 0644. So the natural copy-paste flow
  (scp from `~/.config/gcloud/daity-sa.json`) finds **no file**, and
  the operator falls back to `scp .sa-key.json`, perpetuating the
  in-repo issue on Vast. The doc doesn't acknowledge the divergence.
- **Why it matters:** Phase 1's Critical #3 is unresolved (verified:
  `ls -la /Users/debdoot/Documents/Claude/Projects/daity/.sa-key.json` —
  not checked here; relying on Phase-1 review accuracy). The Vast
  doc assumes the post-fix state of the world. A new operator
  setting up Vast for the first time will hit a confusing missing-file
  error and resolve it in the easiest way (carry the in-repo key over
  to Vast).
- **Suggested fix:** Update the bootstrap doc with: "If you have not
  yet relocated the local SA key (Phase-1 finding #3), do that
  *first* on the laptop: `mv .sa-key.json ~/.config/gcloud/daity-sa.json
  && chmod 600 ~/.config/gcloud/daity-sa.json && rm .env.local-references-to-sa-key`.
  Then return here." Cross-link to the open Phase-1 finding.

### 16. [Minor] `EnvError, Exception` in `_resolve_universe` swallows everything via the `noqa: BLE001` escape hatch

- **File:** `daity/scripts/pretrain_cli.py:50-72`
- **What's wrong:**
  ```python
  except (EnvError, Exception) as exc:  # noqa: BLE001
      log.warning("Falling back to on-disk symbols (no SymbolMaster): %s", exc)
      usable = sorted(on_disk)
  ```
  The `(EnvError, Exception)` tuple is redundant (Exception subsumes
  EnvError) and is the visible tell that the original author considered
  "should I be more specific?" and decided no. The `noqa: BLE001`
  silences the linter. With this, an `OSError` reading the parquet
  root, a `MemoryError` building the symbol master, or a `KeyError` in
  the BQ client all collapse into "Falling back to on-disk symbols".
  Consequence: the operator gets a "started training successfully"
  banner with a warning buried in the log; the run trains on whatever
  the on-disk symbol set looked like, possibly minus the universe.
- **Suggested fix:** Catch only the EnvError + the BQ-import-failure
  cases that are the documented fall-through path:
  ```python
  except (EnvError, ImportError) as exc:
      log.warning(...)
  ```
  Anything else (`OSError`, `MemoryError`) should propagate. If the
  fall-back behavior for `bigquery.exceptions.GoogleAPIError` is also
  desired, name it explicitly.

### 17. [Nit] The 4th SSL objective is documented as "added in Phase 2.1 patch" but DESIGN §3.2 says explicitly that "A fourth objective ... was considered and excluded from v1"

- **File:** `DESIGN.md:101-112` and
  `daity/models/heads.py:3, 22-26`,
  `STATUS.md:9, 78-83`
- **What's wrong:** DESIGN §3.2 lists "Three objectives" and ends with
  "A fourth objective — synthetic order-flow reconstruction from
  OHLCV — was considered and excluded from v1". STATUS says Phase 2
  added a "multi-horizon forecast" as the 4th objective. So a 4th
  objective WAS added; just not the one DESIGN considered/rejected.
  This is fine in spirit but DESIGN §3.2 currently reads as "three
  objectives", and the on-disk code has four. Either DESIGN needs an
  amendment 10 ("fourth objective: multi-horizon forecast on FORECAST
  token, weight 0.5") or STATUS needs to disclose the deviation more
  explicitly.
  Today STATUS line 9 mentions "incl. multi-horizon forecast" once,
  and the writeup notebook has a comment (`§5: This is the 4th SSL
  objective added in Phase 2.1 (DESIGN amendment 10)`) — but
  amendment 10 doesn't exist in DESIGN.md; the post-audit amendments
  cap at 9.
- **Suggested fix:** Add DESIGN amendment 10 to the post-audit
  callout:
  > 10. **Fourth SSL objective added.** §3.2 listed three objectives
  >     and excluded a synthetic-order-flow fourth. Phase 2.1 added a
  >     different fourth: multi-horizon forecast on the FORECAST
  >     token (weight 0.5; predicts H non-overlapping patches at a
  >     single forecast scale). This is the SSL precursor to Phase 3's
  >     supervised quantile head and trains the FORECAST representation
  >     to be a useful summary for downstream forecasting heads.
  Then STATUS, the writeup notebook, and the heads.py module
  docstring can cross-reference amendment 10 instead of asserting it
  exists.

### 18. [Nit] `daity/training/pretrain.py:281` — `loss_forecast = torch.zeros(...)` is correctly device-aware but the `dtype=loss_recon.dtype` couples one loss's dtype to another silently

- **File:** `daity/training/pretrain.py:278-282`
- **What's wrong:**
  ```python
  loss_forecast = torch.zeros((), device=tokens.device, dtype=loss_recon.dtype)
  ```
  Reasonable defense against dtype-mismatch in mixed precision, but
  the upstream `loss_recon` could be a `torch.float32` *or* a
  `torch.bfloat16` (in bf16-mixed Lightning autocasts mid-graph).
  Tying `loss_forecast` to whatever `loss_recon`'s dtype happens to
  be is a "works today" workaround. A cleaner pattern is to cast all
  four parts to a single dtype after the weighted sum and log them in
  a fixed dtype. Today this is invisible; in bf16-mixed it could
  produce subtle differences in `train_loss` precision.
- **Suggested fix:** Move the `torch.zeros(())` out of the hot path
  and use `tokens.new_zeros(())` (matches `tokens` dtype, which is
  what the entire backbone runs in). Equivalently:
  ```python
  loss_forecast = tokens.new_zeros(())
  ```

### 19. [Nit] `OHLCVPretrainDataset.__iter__` infinite loop on a universe of only-too-short-history symbols

- **File:** `daity/data/datasets.py:314-324`
- **What's wrong:**
  ```python
  while True:
      sym = rng.choice(self.symbols)
      end_ts = self._draw_end_ts(rng, sym)
      if end_ts is None:
          continue
      sample = self._build_sample(sym, end_ts)
      if sample is None:
          continue
      yield sample
  ```
  If *every* symbol returns `None` from `_draw_end_ts` or
  `_build_sample` (e.g., all symbols have less than `window_bars[scale]`
  bars at the requested scale), the loop spins forever with no log
  output. This is the same class of bug that motivated the
  `_latest_end_ts` data-availability cap — the cap fixed the "as_of
  ahead of data" case, but the symmetric "every symbol too short for
  window_bars" case is still unguarded.
- **Suggested fix:** Add a `max_failed_attempts` budget. After N (=
  `100 * len(symbols)`) consecutive failures, raise a clear error:
  ```python
  failures = 0
  while True:
      ...
      if end_ts is None or sample is None:
          failures += 1
          if failures > 100 * len(self.symbols):
              raise RuntimeError(
                  "OHLCVWindowSampler could not produce any sample "
                  f"after {failures} attempts. Check window_bars vs "
                  "available data extent."
              )
          continue
      failures = 0
      yield sample
  ```

### 20. [Nit] Worker-id seed naming is convention-only, no test pinning

- **File:** `daity/data/datasets.py:376-393`
- **What's wrong:** The per-worker seed split is implemented and
  intuitively correct, but no test asserts: "with `num_workers=4`, the
  4 workers produce 4 *distinct* sample streams". Today the only
  property tested is "the dataset works with a DataLoader at
  num_workers=0". A regression that, say, drops the `worker.id` term
  silently produces 4 identical streams (and the model trains on 4×
  the same gradient).
- **Suggested fix:** Add `test_dataloader_workers_produce_different_streams`
  using `num_workers=2`, capture the first 50 samples per worker
  (via a custom `worker_init_fn` that stuffs a per-worker buffer),
  and assert the two buffers' `(symbol, end_ts)` lists differ
  meaningfully (>50% disjoint).

---

## DESIGN.md amendment hygiene

- **Multi-horizon forecast head** — DESIGN §3.2 lists three SSL
  objectives + excludes a fourth (synthetic order-flow). Phase 2 ships
  four. The 4th is genuinely useful (it's the SSL precursor to
  Phase 3's quantile head) and the implementation is sound, but the
  DESIGN doc has not been amended to acknowledge it. See finding 17.
- **Contrastive head** — DESIGN §3.2 specifies cross-symbol same-date
  InfoNCE. Phase 2 ships (causal, non-causal) FORECAST-token InfoNCE
  on the same sample, which is mathematically degenerate (finding 2).
  The Phase-2 implementation needs a DESIGN amendment OR a real fix
  before Phase 3.
- **EMA** — DESIGN §3.3 mandates EMA evaluation. The dataclass and YAML
  carry `ema_decay`; nothing reads it. See finding 3.
- **Checkpoint cadence** — DESIGN §3.2 mandates every-2h checkpoints
  with top-3 by val MSE. Not implemented. See finding 1.

The Phase-1 amendments callouts (1-9) are still accurate for what
they cover. The Phase-2 implementation has accumulated four new
gaps that need amendments or fixes before Phase 3 is honest.

---

## Reproducibility

The pretrain CLI runs end-to-end on a fresh checkout:
1. `make dev` (or equivalent uv sync)
2. `daity-pretrain --smoke` produces a CPU smoke-trained run with no
   external dependencies on credentials.
3. `daity-pretrain` against the production YAML requires:
   `data/parquet/*` populated (Phase 1 ingest), `data/cache/symbols/`
   populated (Phase 1 symbol master). On Vast it requires the
   bootstrap doc's SA-key + uv-sync sequence.

Provenance: as noted in finding 7, no `(git_sha, config_hash, as_of)`
triple is emitted by the CLI. CLAUDE.md §6 requires this. Phase-2
artifacts (the soon-to-arrive trained checkpoints) will not be
reproducibility-tagged unless this lands first.

The notebook re-execution is fully reproducible up to MPS noise
(re-ran during this review; numbers within ~2% of STATUS-quoted
smoke values).

---

## Verdict

**Block.** The Critical (no checkpoint callback wired) is a 5-line
fix that protects 2-3 days of GPU time; shipping the Vast run
without it would be irresponsible. Major findings 2 (degenerate
contrastive), 3 (EMA configured-but-unused), and 6 (double-forward
compute cost) all need to either land as code fixes OR be explicitly
disclosed as DESIGN amendments + STATUS updates BEFORE the Vast
launch — the prose framing of "Phase 2 code complete, ready to
launch" misrepresents what's actually shipping vs DESIGN. Major
finding 4 (STATUS overclaims, missing leakage-gate file) is a
documentation hygiene fix and should land same-day.

**Specifically, the following must land before the Vast run starts:**
1. Finding 1: `ModelCheckpoint` callback in the Trainer.
2. Finding 7: `(git_sha, config_hash, as_of)` provenance triple
   sidecar at run-start.
3. Finding 4 (sub-points 1-3): STATUS.md correction on the leakage
   filename + correct test count + narrowed lint claim.
4. Finding 17: DESIGN amendment 10 explicitly acknowledging the
   4th SSL objective.

**The following must land before Phase 3 starts** (i.e., before the
trained checkpoint is consumed by post-train heads):
5. Finding 2: contrastive head fixed OR explicitly amended-and-disabled.
   A degenerate "objective" that contributes zero useful gradient
   should not be passed to Phase 3 as a "Phase-2 SSL representation".
6. Finding 3: EMA wired up OR config flag removed and DESIGN §3.3
   amended to push EMA into Phase 3's posttrain.
7. Finding 8: the dataset-level leakage tests moved to
   `tests/leakage/` so `pytest -m leakage` actually selects them.

**The following should land alongside Phase 3 work, not before:**
findings 5 (daily forecast_scale `_latest_end_ts`), 6 (compute halving
via causal head, if not done sooner for the Vast budget), 9 (window
overflow assertion), 11 (worker cache duplication), 16 (broad
exception catch in CLI), and the Nits 18-20.

The 379 tests and lint-clean Phase-2 modules are real and praiseworthy.
The architecture skeleton — RevIN reversibility, the apply_cached
forecast-target frame, the multi-resolution tokenizer, the
end_ts-bounded dataset with both as_of-leakage and data-availability
caps — is the right shape and the implementation is mostly correct.
The gaps are concentrated at the *operational* edges (no checkpoints,
no provenance, no EMA) and at the *contract conformance* edges
(contrastive degeneracy, DESIGN amendment hygiene), not in the
load-bearing math.
