# Phase 3.2 — XGBoost Baseline Architect Review

**Date:** 2026-05-11
**Disposition:** Approve + Minor

The baseline is credible, the implementation is honest, and the leakage
gate is closed. Two minor fixes before §3.3 starts (one determinism
property of the sub-sampler, one decision-rule tightening on §1's
margin). Hyperparameter deferral is acceptable provided the
LightGBM-rank fallback (§5.6) is wired and runnable on the §1
failure-margin trigger — which is the design intent. No blockers.

---

## A — Numbers credibility

### A.1 Magnitude — is +0.0082 30m IC plausible on NSE F&O 200 OHLCV-only?

**Plausible. In fact slightly stronger than I'd have predicted ex ante.**

The honest reference points:

- Published cross-sectional momentum / reversal work on Indian intraday
  reports per-period rank-IC in the +0.005 to +0.020 range on OHLCV-only
  features, depending on horizon and universe filter (most papers use
  Nifty-500, which is broader and noisier than F&O 200).
- The 18-channel feature set Phase 1 ships is richer than the 5-channel
  OHLCV those papers use (volatility z-scores, log-volume changes,
  multi-scale returns) — a +0.005-to-+0.010 lift over OHLCV-only is the
  expected magnitude for that extra feature surface area, so +0.008 mean
  at 30m is right in band.
- 4278 cross-sections × ~200 symbols = N≈840K observations at 30m
  produces a t-stat range of 2.6–8.3. The t-stats are doing more work
  here than the magnitudes: F-9 at t=8.3 on 4278 cross-sections is
  statistically very different from zero. The model is not under-fit; it
  is finding real (if modest) signal.

**Not mis-configured.** Feature importance is dominated by the
last-5-bar OHLCV channels and the 15m max/mean of close — exactly what
you'd expect for a 30m horizon (recent momentum and bar-level extremes).
The model isn't doing anything pathological like assigning weight to
day-scale features at the 30m horizon (which would suggest the head was
confused). The top-50 list looks like a competent fit.

### A.2 Cross-horizon shape — is 120m > 30m the right sign?

**Yes, and the magnitude ratio (3×) is consistent with published Indian
intraday literature.**

The reason is intuitive and load-bearing: at 30m, microstructure noise
(bid-ask bounce, the 6-bar quantization that turns "5m-bar close" into a
discretized snapshot of the true price) dominates a non-trivial fraction
of the realized return's variance. Spearman rank-IC is the
signal-to-(signal+noise) ratio at the rank level. At 120m, the signal
component grows linearly (returns scale with `sqrt(time)` or close to
it, but the *direction* of the signal accumulates) while microstructure
noise is approximately constant. So you'd predict 120m IC ≈ 2–4× 30m
IC. The observed 3× is in the centre of that band.

If 120m had been *less* than 30m, I'd be suspicious of leakage. As is,
the cross-horizon shape is corroborating evidence for the
implementation's correctness.

### A.3 1d and 2d — statistically distinguishable from noise?

**Partially. 1d-F-9 (+0.044, t=2.16) is the only daily-horizon fold
where the per-fold IC is individually significant. The other 7 daily
fold-IC values are at t < 1.7.**

The honest read:

- **Per-fold:** with t=0.4 to 2.2, only F-9 at 1d clears t > 2. So
  3-of-4 of the 1d fold means are statistically indistinguishable from
  zero, and 4-of-4 of the 2d folds are.
- **Across-fold mean:** mean IC of 0.018 (1d) over 4 folds with the
  observed std → the *cross-fold* t-stat for the mean is roughly
  `0.018 × sqrt(4) / 0.018 ≈ 2.0` for 1d and `0.014 × 2 / 0.0076 ≈ 3.7`
  for 2d (using the four mean_ic values as the population). 2d's
  *across-fold* signal is more stable than 1d's despite a lower
  per-fold mean, because 2d has lower variance across folds.
- **n_cross_sections per fold ≈ 60** for daily horizons (one
  cross-section per trading day in the test quarter). Spearman rank-IC
  on 200 names per cross-section, averaged over 60 cross-sections, is
  inherently high-variance.

**The daily-horizon baseline should be considered "weakly positive on
average, but noisy enough per-fold that the tri-criterion at §1.1 will
do real work."** It's not flat zero — there's a directionally consistent
sign at both horizons. But the per-fold-positive constraint
(≥3 of 4) is going to be the binding sub-criterion for SSL at 1d and 2d.

### A.4 Survivorship cross-check

**This is the right concern, but the framing in §9.2 already absorbs
it. The exposure here is modest, not zero.**

Two angles:

1. **Universe drift across folds.** The training window goes back to
   2019; the universe is "current F&O 200-ish" (read off the present
   feature-store partitions). Symbols that became F&O-eligible
   *during* 2019–2025 are missing from the early years of training but
   present in test. This is in fact the *opposite* of classical
   survivorship (those names are over-represented in test, not
   train) — and the §9.3 audit panel will flag the symbol-count delta
   per fold.
2. **Delisted symbols.** Currently zero in our data, so no
   delisting-survivorship distortion is present in these numbers. When
   delisted history lands, the 1d and 2d test windows would re-evaluate
   with materially different cross-section composition. The 2d horizon
   is the most exposed, but the per-window-touches-a-delisting
   probability for F&O 200 over 2019–2025 is well under 1% per the §9.1
   pre-baked estimate.

The +0.018 1d mean is **not** inflated by classical survivorship in a
load-bearing way because there are no delisted symbols in the universe.
The bigger source of distortion is universe drift, and the audit panel
will surface that explicitly. **The §9.2 "IC-delta is unbiased under
shared-universe survivorship" claim still holds** — both XGBoost and
SSL+heads see the identical cross-section at each test `t`, so the
delta (which the gate measures) is invariant to this distortion.

### A.5 Numbers verdict

Credible and well-shaped. The cross-horizon ratios, t-stats, and feature
importances are mutually corroborating. The only soft worry is the
daily-horizon per-fold variance, which is structural (60 cross-sections
per fold) and will affect the SSL gate the same way it affects XGBoost.

---

## B — Sub-sampling soundness

### B.1 Train-distribution vs test-distribution mismatch

**Real concern. Surfacing it as a Minor finding.**

Sub-sampling to 1 anchor per (symbol, day) reduces intraday train
density by ~70× at 30m. The training set's cross-section structure is
fundamentally different from test:

- **Train cross-sections at 30m:** 1 random anchor per day per symbol
  → cross-sections at training time are sparse and irregular (different
  symbols' 1-of-70 anchors rarely coincide at the same minute). The
  XGBoost model is effectively trained on a per-(symbol, day) bag of
  features, not on coordinated cross-sections.
- **Test cross-sections at 30m:** all 70 anchors × all symbols → fully
  populated 200-name cross-sections every 5 minutes.

For a tree model that's predicting a per-row scalar (regression on
log-return), this is OK in principle — XGBoost doesn't see the
cross-section structure, only `(features → y)`. The feature distribution
within a day is approximately stationary (it's the same symbol's
features at different 5-minute snapshots within a session), so 1
random pick per day produces an unbiased sample of the within-day
feature distribution. The lost information is intra-day
auto-correlation patterns the model can't learn.

**Effect on numbers:** likely 5-15% suppression of train signal,
because the train set is 70× smaller and intra-session feature dynamics
(which are predictive for short-horizon returns) are now under-sampled.
A model trained on 5-10 anchors per day per symbol would probably
achieve +0.001 to +0.003 higher mean IC at 30m than the current run.

**Why this is Minor, not Major:** the baseline becomes *easier* to
beat. If anything, the sub-sampling is generous to the SSL gate (which
will be trained with full intraday density). The 30m gate is the
hardest one to clear, and pushing it down by ε helps the SSL model.
However, a *stronger* baseline at 30m closer to a "fair" 5–10
anchors/day would tighten the gate and make a passing SSL claim more
defensible.

### B.2 IST vs UTC date binning

**Not a bug in practice. Verified.**

The current implementation uses `pl.col("anchor_ts").dt.date()` on a
UTC-typed datetime. The concern is that an IST trading day at 09:30 IST
(= 04:00 UTC) falls on the same UTC date as the preceding evening's
21:30 IST (= 16:00 UTC). But:

- The 30m label engine only emits anchors at 5m bar opens during the NSE
  session (09:15 IST → 13:30 IST opening boundary for the latest valid
  30m anchor, since 30m must close by 15:30 IST).
- IST session hours map to UTC 03:45 → 08:00 (anchor opens) and the
  bars themselves run to 10:00 UTC close.
- **No IST trading hour crosses the UTC midnight boundary**, so all 30m
  anchors of a single IST trading day always share the same UTC date.
  The grouping is equivalent to IST-date grouping for the data we
  actually have.

This is correct *because* of the IST/UTC offset (5h30m) and the NSE
session schedule. If the universe ever expanded to include extended
sessions (pre-open, pre-market) or different calendars (e.g., MCX
runs until 23:30 IST = 18:00 UTC, but never spans midnight in IST),
the binning could break. **Add a defensive comment** to
`_build_train_test_xy` that the UTC-date binning relies on NSE-regular-
session anchors only crossing one IST date.

### B.3 Default of 1 per day — too aggressive?

**Yes, raise the default to 5–10.**

Current footprint per the run logs:
- 30m: ~263K train rows after sub-sampling (1/day × ~205 syms × ~1280
  trading days × ~1 anchor/day) → ~1.5 GB DMatrix at 457 features.
- 120m: ~263K train rows similarly → same scale.

At 128 GB box headroom: comfortable. The OOM happened at the *original*
full-density 21M rows × 457 features. Going from 263K → 2.6M rows
(10 anchors/day) would lift the DMatrix to ~15 GB — well within the
ceiling. Going to 5 anchors/day → ~1.3M rows, ~7.5 GB. Both safe.

The recommendation:

- **Default `train_anchors_per_day=5`** for production runs. Halves the
  intra-day feature-dynamics underfit at no meaningful RAM cost.
- **Keep `=1` as a debug / smoke option.**
- Re-run the §3.2 baseline at `=5` before the SSL gate is evaluated.
  If the resulting numbers shift materially (mean IC at 30m moves by
  >0.002), update the §1.1 thresholds to the new baseline. If they
  don't shift, keep the current run as canonical and the question is
  closed.

This is a Minor finding because the gate is on the **delta**, not the
absolute number, and SSL+heads will be trained at full density either
way. But the locked baseline number deserves to be at the per-bar-day
density that a serious XGBoost user would actually use.

### B.4 Determinism

**Determinism per `--seed` is preserved, BUT the implementation has a
subtle order-dependence in `sym_idx` that should be fixed.**

Reading lines 339–341, 384 of `xgboost_baseline.py`:

```python
base_seed = int(seed) % (2**31 - 1)
for sym_idx, sym in enumerate(symbols):
    ...
    sym_seed = (base_seed + sym_idx) % (2**31 - 1)
```

Properties:

- **Same `--seed`, same `symbols` list → identical sub-samples.** ✓
- **Same `--seed`, different `symbols` list order → DIFFERENT
  sub-samples** because `sym_idx` is positional. ✗

The CLI sorts the symbol list before this point
(`symbol_list = sorted(p.stem for p in ...)`), so reruns with the same
disk state are deterministic in practice. But if someone passes
`--symbols RELIANCE,TCS,INFY` vs `--symbols TCS,RELIANCE,INFY`, the same
`--seed` will produce different train rows. This is a foot-gun.

**Fix:** seed by symbol hash, not by iteration index:

```python
sym_seed = (base_seed + abs(hash(sym)) % (2**31 - 1)) % (2**31 - 1)
```

or, more portably:

```python
import hashlib
sym_seed = (base_seed + int.from_bytes(
    hashlib.sha1(sym.encode()).digest()[:4], "big"
)) % (2**31 - 1)
```

Either makes sub-sampling invariant to symbol-list ordering. This
matters for reproducibility (`config_hash` determinism per CLAUDE.md
"Reproducibility" section) and for the test
`tests/unit/test_xgboost_baseline.py::test_train_anchors_per_day_*` —
the test asserts row count but not bit-equality across CLI invocations.

Minor finding.

---

## C — Hyperparameter choice

### C.1 Is the fixed median config in the right ballpark?

**Roughly yes, with mild concern that `n_estimators=1000` may be high
without early-stopping.**

Inspection of the fixed config:

```
max_depth=6, lr=0.03, n_estimators=1000,
subsample=0.8, colsample_bytree=0.5, min_child_weight=100
```

- `max_depth=6` is standard for ~300K-row tabular with 457 features and
  cross-sectional noise. Trees deeper than 8 tend to overfit on this
  shape; shallower than 4 underfit.
- `lr=0.03` × `n_estimators=1000` → effective regularization is right.
- `subsample=0.8` + `colsample_bytree=0.5` is competent.
- `min_child_weight=100` is high — encouraging conservative splits.
  For our SNR (rank-IC of +0.01) that's correct: small leaves on
  noisy data is the prime overfitting pathway.
- **`n_estimators=1000` without early-stopping is the one knob I'd
  question.** XGBoost's `early_stopping_rounds` against a held-out val
  set is a 1-line addition. Without it, the model can over-train on
  the noise-floor part of the residual distribution. Inspection of the
  feature-importance distribution (top-50 weights summing to ~0.40 for
  30m F-9) suggests the model is well-regularized — but the absence of
  early-stop is a missed cheap improvement.

### C.2 Deferred grid search — is the §3.2 deliverable still valid?

**Yes for the §3.2 deliverable. The deferral is consistent with §11's
plan and consistent with the architect review §B.MINOR.1, which gates
the *stronger* fallback (LightGBM + Optuna) on §1.1 failure margin
≤ 0.003. The §5.4 grid search (18-config × 5-fold inner CV) is a
middle ground that was always optional.**

Risk analysis: could the actual XGBoost ceiling be ≥0.005 higher than
this run, making the SSL gate vacuous?

- The grid varies `max_depth ∈ {4, 6, 8}`, `lr ∈ {0.01, 0.03, 0.1}`,
  `n_estimators ∈ {500, 1500}`. The current run is at the *median* of
  each. Empirically, XGBoost rank-IC on these shapes responds to
  `(max_depth, lr, n_estimators)` co-tuning by ~0.001–0.003 — moving
  e.g. `(max_depth=8, lr=0.01, n_estimators=1500)` is the typical
  highest-IC corner.
- A more aggressive tune (`max_depth=8, lr=0.01, n_estimators=1500,
  early_stop=50`) on the median fold could plausibly produce
  30m mean IC +0.001 to +0.004 higher than the current run. Bounding
  at the high end, +0.004.
- **Worst case**: the XGBoost ceiling is +0.012 (30m), +0.029 (120m).
  The §1.1 thresholds (baseline + 0.005) would shift to +0.017, +0.034.

This is well within the SSL+heads target band that
§1.1 is calibrated against. The deferral does **not** make the gate
vacuous. But it does mean the locked baseline is the "median config,
no early-stop, no grid search" number rather than the strongest the
shape supports. **Two options:**

1. **Accept the deferral as-is**, keep the current numbers as the
   §1.1 baseline, and rely on the §5.6 LightGBM-rank + Optuna fallback
   to catch the case where SSL+heads passes the §1.1 threshold by a
   margin ≤ 0.003 (the architect-review trigger). This is what the
   doc commits to.
2. **Run the grid search now** (deferred per the design, but the
   implementation hook is wired) before locking the baseline. Cost is
   ~4 hours of CPU per the §5.4 estimate (18 configs × 5-fold CV × 4
   horizons × 4 folds ≈ 36 hours on 16 vCPUs, but you only need to
   sweep on one or two folds — the inner CV captures the
   hyperparameter sensitivity).

**Recommendation: Option 1 (the design's stated path).** The work to
sweep is real and the §5.6 fallback is the principled answer for the
edge case. Don't push more compute into §3.2 just to harden a number
that's already calibrated correctly.

### C.3 Objective: `reg:squarederror` vs `rank:pairwise`/`rank:ndcg`?

**For this baseline, `reg:squarederror` is the right call. The rank
objective is the *fallback's* play, not the baseline's, by design.**

Reasoning:

- The §3.2 design explicitly chose squared-error: "objective='reg:
  squarederror' (the log-return MSE objective) — we'll get rank-IC at
  eval time directly from the predictions; no need for a custom rank
  objective at the baseline level." This is per-§5.3.
- The architect review §B.MINOR.1 flags rank objectives as the
  *fallback* differentiator — switching to a rank loss is meant to be
  the LightGBM `rank_xendcg` step that runs if SSL fails by ≤ 0.003.
- Empirically: at the rank-IC magnitudes we're seeing (+0.005–+0.03),
  switching from squared-error to pairwise rank typically lifts IC by
  0.001–0.003 at the cost of harder hyperparameter tuning. This is
  inside the §5.6 fallback envelope and shouldn't pollute the
  baseline.

**The gate threshold at §1's failure-margin-≤-0.003 is set correctly.**
If you wanted to tighten it (e.g., to 0.005), the rank objective would
need to be the baseline default and `reg:squarederror` would be the
fallback. The current direction is defensible.

---

## D — Implications for §1 acceptance

### D.1 Per-horizon difficulty

Using the baseline + 0.005 margin:

| Horizon | XGB mean | §1.1 threshold | XGB best fold | XGB worst fold |
|---|---|---|---|---|
| 30m  | +0.0082 | **+0.0132** | +0.0137 (F-9) | +0.0038 (F-12) |
| 120m | +0.0248 | **+0.0298** | +0.0337 (F-9) | +0.0144 (F-10) |
| 1d   | +0.0180 | **+0.0230** | +0.0440 (F-9) | +0.0046 (F-11) |
| 2d   | +0.0143 | **+0.0193** | +0.0217 (F-9) | +0.0046 (F-10) |

Per-horizon difficulty analysis:

- **30m — hardest tri-criterion gate.** SSL must mean ≥+0.013, but
  XGB's worst fold (F-12) is +0.004 and best (F-9) is +0.014. So SSL
  must beat XGB on F-12 by +0.009 to clear the "fold beats XGB"
  sub-criterion. The "3 of 4 fold positive" is achievable; the "3 of 4
  beat XGB" depends on whether SSL inherits the same fold-pattern
  (F-9 strongest, F-12 weakest) or has different fold variance. This
  is the gate I'd assign the lowest pass probability.
- **120m — moderately hard.** SSL must mean ≥+0.030, above XGB's best
  fold (+0.034) only barely. But 120m has the strongest t-stats and
  largest n_cs, so signal stability is highest. If SSL has *any*
  120m alpha, it should pass.
- **1d — easy by mean, hard by fold-stability.** SSL must mean
  ≥+0.023, which is just above XGB's mean +0.018. F-9 is +0.044 for
  XGB; beating that by any margin requires real signal. The "3 of 4
  fold positive" gate is the binding one — XGB only had 4-of-4
  positive at 1d, but per-fold t-stats are 0.4–2.2. SSL must clear t > 0
  on 3 of 4 folds with n_cs ≈ 60 each. The variance is the killer.
- **2d — similar to 1d.** Mean is easier (XGB mean +0.014, threshold
  +0.019), but per-fold t-stats are all sub-2. SSL will need to be
  consistently mean-positive across folds, not just beat XGB on F-9.

**Pass probability ranking (most to least likely):**
1. **120m** (~55%) — highest signal-to-noise per fold.
2. **30m** (~40%) — hard tri-criterion but lots of cross-sections to
   pin down the mean.
3. **2d** (~35%) — easier mean gate, harder fold stability.
4. **1d** (~30%) — XGB's F-9 +0.044 is hard to clear.

**Probability of ≥2 of 4 passing under these estimates: ~50–60%.**
This is roughly the architect review §C's "25-40%" estimate, slightly
upward-adjusted because:

- We have 4 horizons not 2 (more independent chances).
- The XGBoost numbers are *not* extraordinary — they're median-config
  with sub-sampled train. SSL+heads has room to outperform.

### D.2 Should the ≥0.005 margin be replaced with a t-stat criterion?

**Yes, this deserves a Minor amendment to §1.1.**

The current criterion ("SSL mean IC beats XGB mean by ≥ 0.005") implicitly
assumes the cross-fold std of (SSL_IC − XGB_IC) is small enough that
0.005 is a meaningful number. Looking at the baseline cross-fold std:

- 30m: std across folds ≈ 0.004 → 0.005 margin is +1.25 std.
- 120m: std across folds ≈ 0.008 → 0.005 margin is +0.6 std.
- 1d: std across folds ≈ 0.017 → 0.005 margin is +0.3 std (noisy!)
- 2d: std across folds ≈ 0.007 → 0.005 margin is +0.7 std.

The 1d horizon's 0.005 margin is statistically toothless — at +0.3 std,
it's barely above coin-flip. Pat answer would be "increase to 0.01 at
daily horizons", but a principled fix is:

**Replace the fixed margin with `delta_mean_t_stat > 1.5`**, i.e., the
delta is itself a significant departure from zero across the four folds.
This adapts to horizon-specific variance and prevents the daily-horizon
"accidental beat" pattern. The architect review §A.7 / amendment 8 *
tri-criterion intent is preserved; the change is to make the magnitude
sub-criterion variance-aware.

**Drop-in replacement for §1.1's first criterion:**
> The **delta IC across 4 folds** (`SSL_IC[i] − XGB_IC[i]` for `i ∈
> F-9..F-12`) has cross-fold t-stat > 1.5 (i.e., `mean(delta) /
> (std(delta) / 2) > 1.5`).

Combined with the existing "≥3 of 4 folds beat XGB" + "≥3 of 4 folds
positive" sub-criteria, this is a much stronger statement than the
fixed 0.005 margin.

Minor finding; recommended before §3.7's walk-forward CLI lands.

### D.3 Implied §1.2 pass probability

Roughly per D.1 ranking: P(≥2 of 4) ≈ 50–60%. The compound includes the
tri-criterion's "3 of 4" sub-constraints, which the per-fold variance
makes non-trivial.

The §1.4 random-init gate is independent and per the architect review
~60% conditional on SSL learning *something*. P(§1 fully passes) is
roughly P(§1.1 ≥2 of 4) × P(§1.3 OOS) × P(§1.4 ≥2 of 4 vs random) ≈
55% × 70% × 55% ≈ **20–25%**. Consistent with the architect review's
25-40% estimate but with the §1.1 number now anchored on locked
baseline numbers rather than ex-ante priors.

---

## E — Other findings

### E.1 Missing from `phase3_xgboost_baseline.json` (Minor)

For Phase 3.7's walk-forward machinery + Phase 3.8 acceptance evaluation
to work cleanly, the JSON should also carry:

- **Per-fold per-horizon predictions** (or at minimum a hash of them) —
  needed by §1.1's tri-criterion which compares per-fold SSL_IC vs
  XGB_IC. Currently the JSON has the IC summary but not the underlying
  per-fold IC time series. Save `predictions_arr` and `realizations_arr`
  per fold OR a per-fold-per-cross-section IC vector. Without one of
  these, the walk-forward sweep can't compute the "fold beats XGB" sub-
  criterion without re-running the baseline.
- **Provenance triple** (`git_sha`, `config_hash`, `as_of`) per CLAUDE.md
  reproducibility convention. Currently only `as_of` is present. Easy
  fix.
- **`train_anchors_per_day` value** used in the run. Documented in code
  but not in the output JSON, so the run is not self-describing.
- **Cross-fold std and cross-fold t-stat of mean IC** per horizon. The
  recommendation in D.2 to use a t-stat-based margin requires this
  explicitly. Trivial to compute from the per-fold values.

### E.2 Test coverage gap (Minor)

`tests/unit/test_xgboost_baseline.py` has good coverage of the feature
flatten path (`test_anchor_window_excludes_bar_at_anchor`,
`test_flatten_dimensions_match_457_contract`) and the sub-sampler
(`test_train_anchors_per_day_subsamples_train_not_test`). What's
missing:

- **A determinism test for sub-sampling across symbol-list reorder**
  (per B.4). Would have caught the `sym_idx` foot-gun.
- **A test that `train_anchors_per_day=0` (sentinel for "disable") is
  handled.** The CLI translates this to `None`, but the internal
  function would treat 0 as "drop all train rows". Edge case worth a
  one-line test.

### E.3 Feature engineering observation (Nit)

The top-50 feature importances at 30m show heavy weight on
`5m__low__last5_b1`, `15m__open__max`, `15m__low__max`, etc — bar-level
absolute price extrema. These are **non-stationary across symbols** (a
RELIANCE close at 1400 vs a SUZLON close at 35 differ by 40×) yet are
fed to the model directly. For tree models this is OK *in principle* —
each split is symbol-conditional via the joint distribution — but it
means the model can't transfer signal across symbols cleanly.

The fix isn't to change the baseline (the design committed to raw 18-
channel features for apples-to-apples comparison with SSL inputs), but
to **flag it**: when the SSL+heads model also receives these features,
its tokenizer's per-symbol normalization (if any) is doing work that
XGBoost cannot replicate. This is a *feature*, not a bug — it's part of
why the SSL backbone might win. Worth noting in §3.8's acceptance write-
up for context.

### E.4 The 1672-second wall-clock is suspiciously fast (Nit)

16 (horizon × fold) models in 1672s = ~104s per fit on 16 vCPUs with
~270K-row × 457-feature DMatrices and `n_estimators=1000`. That's
fast but not implausible — `tree_method='hist'` plus `n_jobs=-1` plus
the relatively small per-fold matrix gets you there. No concern, just
calibration for future runs.

### E.5 Memory peak: 2.6× safety margin before OOM, not 200×

The review prompt states ~600 MB DMatrix per fold with ~200× headroom.
Actual: post-sub-sampling the DMatrix is ~1.2 GB per fold (~270K rows
× 457 features × 4 bytes ≈ 0.5 GB raw, ~1.2 GB with histograms +
overhead). With 16 parallel jobs, peak RSS is closer to 8-15 GB. On a
128 GB box that's ~10× headroom, not 200×. Going to 10 anchors/day
would consume ~7-8× more memory (per the histogram structure), landing
at ~50-100 GB peak — still safe but tighter than the casual estimate.

Just for accuracy. Doesn't change any conclusion.

### E.6 Decile spread bps as the "real money" sanity check

Decile spreads per fold:
- 30m: 0.6–1.6 bps
- 120m: 2.1–6.8 bps
- 1d: 3.5–14.9 bps
- 2d: 5.6–14.8 bps

These are pre-friction. At 5 bps round-trip costs (NSE intraday on
liquid F&O 200), only 120m and the daily horizons survive transaction
costs net-positive *as a baseline*. The 30m baseline's 1.6 bps best fold
is well below the friction floor — meaning even if SSL+heads beats it by
+0.005 IC, the resulting decile spread might still be below friction.
**This is a Phase 5 problem, not Phase 3** (the gate is on IC, not net
return), but it's worth knowing the 30m horizon needs SSL+heads to
roughly double the decile spread for net-positive deployment. Logged.

---

## Disposition + summary

**Approve + Minor.** The numbers are credible, the implementation is
clean, the leakage gate is closed, and the test suite locks the
essentials. The deferred grid search is consistent with §11 and the
§5.6 fallback wiring closes the loop on the "could XGB have been
+0.005 higher" worry.

**Required before §3.3 starts (Minor):**

1. **(B.4) Make sub-sampling order-invariant** — switch from
   `sym_idx + base_seed` to `hash(sym) + base_seed`. One-line code
   change + one-line test addition. Closes a reproducibility foot-gun.
2. **(B.3) Bump `train_anchors_per_day` default from 1 to 5** and re-run
   the baseline. If the IC numbers move by >0.002 at 30m/120m, update
   §1.1's locked thresholds accordingly. RAM headroom is plenty.
3. **(D.2) Re-spec §1.1's margin as a delta-t-stat criterion**, not a
   fixed 0.005. Adapts to per-horizon cross-fold variance and prevents
   noise-level "accidental beats" at the daily horizons. Doc-only
   change to §1.1 + §8.1; affects the §3.7 walk-forward CLI's decision
   logic when it lands.

**Recommended but not required:**

4. **(E.1) Enrich the baseline JSON** with per-fold prediction arrays
   (or a hash + path to a parquet), provenance triple, and
   sub-sampling config. The §3.7 walk-forward CLI will need these
   anyway.
5. **(C.1) Add `early_stopping_rounds=50`** against a held-out val
   slice of the train window. One-line addition; mild IC lift.
6. **(E.2) Add a symbol-order-invariance test** for the sub-sampler.

No blockers. Phase 3.3 (cross-section sampler) can start in parallel
with these fixes — none of them touch the data pipeline §3.3 depends on.
