# Phase 3 Design — Supervised Post-Training

**Status**: Draft v1 — pending architect-review signoff before any code lands.
**Date**: 2026-05-10.
**Parent docs**: `DESIGN.md` (project source of truth), `STATUS.md` (phase ledger), `reports/reviews/phase_2_architect_review.md` (architect-review feedback that reshaped Phase 3 scope).

> This document is the Phase 3 contract. It supersedes the one-paragraph
> sketch in `DESIGN.md` §3.3 + §7 and the bullet list in `STATUS.md`'s
> Phase 3 row. Where this doc and `DESIGN.md` disagree, `DESIGN.md` wins on
> overall direction; this doc wins on Phase-3 specifics.
>
> **Philosophy of this doc.** Phase 2 ran the experiment "does an
> SSL forecast head produce a meaningfully predictive 8-hour-ahead
> representation?" and the empirical answer was *no*: across five runs
> (v1 11M, v2 36M, v2_long, v3_lr, v3_features, v3_xattn) the forecast
> plateau held flat in the 4.4–4.8 log-return-MSE band, including under
> the architect-recommended cross-attention head fix. The Phase 2
> architect review (§5–§6) explicitly redirects Phase 2's acceptance
> criterion: **Phase 2 is done when Phase 3 beats XGBoost on rank-IC at
> 30-min and 5-day horizons**, not when val_forecast plateau breaks.
>
> Phase 3 is therefore the project's first real test of "does this
> system have alpha?". Everything in this design is shaped by that.

> **Post-architect-review amendments (2026-05-10).** Independent ML architect
> reviewed v1 of this doc (`reports/reviews/phase_3_architect_review.md`) and
> issued **Approve with Major edits** with 3 Major + 8 Minor + 4 Nit findings
> plus answers to all 10 open questions in §13. **Where this callout disagrees
> with text further down, the callout wins.**
>
> 1. **§1.3 conformal gate is now out-of-sample.** v1 measured coverage on the
>    calibration (val) set itself, which split conformal guarantees by
>    construction — a vacuous gate. §1.3 now requires coverage on the held-out
>    **test fold** of each walk-forward fold, averaged across the 4 folds, in
>    `[0.75, 0.85]` per horizon at the top level. Per-vol-bucket coverage is
>    reported but not gated (small per-cell test sample sizes). The in-sample
>    coverage check is kept as a sanity assertion in §7.5, not an acceptance gate.
> 2. **§4.3 — separate-stream sampling for 30m and 5d.** v1 aligned 30m and 5d
>    on the same `t` and downweighted 5d by `1/75 × 0.2` per-step. This created
>    gradient pathologies (75 near-duplicate 5d labels per day with effective
>    sample correlation ρ ≫ 0; loss-balance arithmetic was on per-day magnitude,
>    not per-day variance; 5d head saw "75 epochs of single-label data" within
>    each day). Replaced with two iterators (`iter_30m` yielding 75 cross-sections
>    per trading day, `iter_5d` yielding 1) and alternating gradient steps at
>    natural cadence (75:1). The `0.2×` factor in §6.2 and the `1/75` factor in
>    §4.3 are dropped.
> 3. **§1.4 random-init ablation is now hard-specified.** v1 said "must
>    underperform" with no margin, no compute spec, no seed handling. §1.4 now
>    requires SSL+heads to beat random-init by ≥0.005 rank-IC at both horizons,
>    with the random-init baseline trained as **full fine-tune** of the random
>    backbone (LoRA on random init is degenerate). 3 random seeds, take best as
>    comparison. Hard gate, not advisory.
> 4. **Direction head dropped from v1 (per A.2).** It was auxiliary, the rank
>    head already provides within-batch sign discrimination, and saving the
>    `w_dir_*` tuning surface keeps the loss-balance fight smaller. May come
>    back in Phase 5 behind a config flag if sizing-by-direction-confidence
>    proves useful. §6.1 / §6.2 / §6.9 updated; loss equation is now 3 terms.
> 5. **Two LoRA adapter banks (per A.3).** v1 had one LoRA bank shared across
>    horizons. Replaced with `lora_30m` and `lora_5d` via `peft`'s named-adapter
>    API, ~1.4M params each (still <8% of backbone). 30m head reads
>    post-`lora_30m`; 5d head reads post-`lora_5d`. Removes 30m/5d gradient
>    interference; makes 30m model independently shippable for Phase 4.
> 6. **Survivorship κ shrinkage dropped (per A.8).** The fixed `(0.02, 0.10)`
>    multiplicative shrinkage was a heuristic dressed as a derivation. Replaced
>    with: (a) per-fold survivorship audit panel in the tearsheet quantifying
>    universe size shift (e.g., "2019-Q1 had N=148 symbols; 2026 has 205"),
>    (b) raw rank-IC + raw IC-delta vs baseline as the reported numbers, (c)
>    prose caveat. The IC-delta vs XGBoost is unbiased under shared-universe
>    survivorship, which is the correct claim. §9 rewritten.
> 7. **Hparam compatibility check is now in Phase 3 (per A.10).**
>    `posttrain_cli.py` extracts backbone hparams from the v3_xattn checkpoint
>    and asserts they match the live Hydra config; raises
>    `IncompatibleCheckpointError` on mismatch. No silent coercion. New §6.10.
> 8. **Walk-forward decision rule is now tri-criterion (per A.7).** "Mean
>    rank-IC over 4 folds beats XGBoost mean by ≥0.005" plus "≥3 of 4 folds
>    individually positive" plus "≥3 of 4 folds individually beat XGBoost".
>    Single-fold dominance and missing-monotonicity blind spots closed. §1.1 /
>    §1.2 / §8.1 updated.
> 9. **Conformal calibration runs both CQR + vanilla split (per A.5).** Per
>    `(horizon, vol_bucket)` cell, pick whichever score function gives empirical
>    coverage closer to `1-α`. CQR's vol-modulation isn't a sure win when the
>    quantile head is calibration-shaky on a noisy 8h-ahead target; vanilla is
>    the safety net. §7.1 / §7.2 updated.
> 10. **Inference contract section added (§6.11).** Phase 3 ships a model that
>     emits predictions at strategy time. The `(input → output)` contract,
>     latency budget, and conformal lookup mechanism are now spec'd here so
>     Phase 5 doesn't inherit a vague handoff.
> 11. **Loss weights via magnitude-balance heuristic, not hardcoded (per
>     B.MINOR.2).** v1's `(1.0, 1.0, 0.3)` (or `0.3, 0.3` with direction) were
>     guesses. §6.2 now sets weights from a one-batch step-0 measurement so
>     each `w_h × L_h ≈ 1.0`. Test asserts within-2× balance at init.
> 12. **LoRA-only fallback escape hatch (per A.1).** If random-init ablation
>     beats LoRA-only by ≥0.002 on val rank-IC at 30m, switch to
>     `unfreeze_top_n_layers=2 + lr_backbone=lr_lora * 0.1`, no LoRA. One-line
>     config flag. §6.7.
> 13. **Walk-forward HTML tearsheet deferred to Phase 5 (per C.4).** Phase 3
>     writes JSON only; Phase 5's `tearsheet.py` renders it. Saves 0.5 days
>     and matches DESIGN §5.1's `eval/tearsheet.py` placement. §8.4.
> 14. **Effort estimates revised (per B.MINOR.7).** Total: ~11 days, ~50
>     H200-hours. §11 table updated. Sub-phase 3.4 → 3 days; 3.5 → 1.5 days;
>     3.7 → 2.5 days.
> 15. **Minor edits applied inline.** Stronger-fallback XGBoost / LightGBM
>     baseline (§5), `as_of_inferred_end` definition (§3.4), VWAP-vs-execution
>     coordinate-frame reason (§3.3), K=5 vol-bucket cell-size sanity (§6.4),
>     `tradable` mask handling at inference (§6.11 / §8), W&B tracking section
>     (§6.12), per-symbol IC diagnostic (§8.2), provenance triple extension
>     to Phase 3 artifacts (§6.10 / §8), `test_xgboost_features_leakage.py`
>     leakage gate (§5.6 / §10).
>
> 16. **Horizon set expanded to `{30m, 120m, 1d, 2d}`** (post-amendment 16,
>     2026-05-10). v1 had two horizons: 30m (intraday) and 5d (swing). The
>     new horizon set drops 5d and adds 120m, 1d, and 2d. **Rationale:**
>     - **5d dropped.** Architect review §A.8 + §C.3 flagged 5d as the
>       most survivorship-bias-exposed horizon, the smallest effective
>       sample size (~17K samples / fold), and the most marginal P(pass)
>       in §C.1 (30%). §C.3 explicitly listed it as cut-first if budget
>       tight. Rather than ship-and-then-cut, drop it now.
>     - **120m, 1d, 2d added.** These fill the "intraday-to-short-overnight"
>       band where NSE F&O 200 alpha is more concentrated empirically. 120m
>       (24 forward 5m bars) is a natural intraday horizon between 30m and
>       end-of-day. 1d (close-to-close) and 2d (close-to-close-2-days-out)
>       are short-enough swing horizons that survivorship distortion is
>       small (delisted-during-window names are rare at this horizon).
>     - **Acceptance gate restructured to "≥2 of 4 horizons pass
>       tri-criterion vs XGBoost"** (§1.1 / §1.2 unified, §1.4 random-init
>       likewise). With 4 partially-correlated bets, the probability of
>       at-least-2-passing is meaningfully higher than v1's compound 30m∧5d
>       gate, while keeping the bar that *some* form of edge must show up.
>       Per-horizon results all reported in the tearsheet for downstream
>       strategy-design intelligence.
>     - **Sampler now has FOUR iterators** (`iter_30m`, `iter_120m`,
>       `iter_1d`, `iter_2d`) with weighted-random alternation at natural
>       cadences (~70 : 50 : 1 : 1 per trading day). §4.3 updated.
>     - **Heads × LoRA banks now 4× instead of 2×.** Three heads × four
>       horizons = 12 head instances (~1.2M params). Four LoRA banks
>       (`lora_30m`, `lora_120m`, `lora_1d`, `lora_2d`) at ~184K each =
>       ~736K. Total trainable ~2M, still <6% of frozen backbone.
>     - **Effort impact.** §11 sub-phases 3.4 / 3.5 / 3.7 each bumped
>       0.5 d for the additional plumbing. New total: ~12 days, ~60–90
>       H200-hours.
>
> 17. **§1.1 / §1.4 magnitude sub-criterion is now a delta-t-stat,
>     not a fixed margin** (post-amendment 17, 2026-05-11). Per the
>     XGBoost baseline review §D.2: cross-fold delta-IC std is
>     0.004–0.017 across horizons; a fixed `≥0.005` margin was only
>     +0.3 to +0.7 sigma at the daily horizons (1d/2d) — statistically
>     toothless where the gate mattered most. Replaced with
>     `cross_fold_t_stat(SSL_IC - XGB_IC) > 1.5` (same for §1.4 vs
>     random-init). Adapts to per-horizon variance. The other two
>     sub-criteria (≥3 of 4 folds positive, ≥3 of 4 folds beat
>     baseline) are unchanged.
>
> 18. **XGBoost baseline run #2 used `train_anchors_per_day=5`**
>     (post-amendment 18). Run #1 at `=1` was too aggressive — review
>     B.3 found ~5-15% suppression of train signal at 30m/120m vs.
>     5/day, with plenty of RAM headroom. The locked baseline numbers
>     in `reports/phase3_xgboost_baseline.json` reflect the `=5` run.
>     Also fixed: hash-based sub-sample seed (review B.4 — was
>     order-dependent on `--symbols`), early-stopping
>     (`early_stopping_rounds=50`) per review C.1, JSON enrichment
>     with provenance triple + per-cross-section IC arrays + cross-fold
>     stats per review E.1.

---

---

## 1. North Star and Acceptance Criteria

The deliverable is a **calibrated cross-sectional ranker** that, given
`(symbol, as_of)` for every symbol in the F&O 200-ish universe,
emits:

- A predicted distribution over forward log-return at horizons
  **`H ∈ {30m, 120m, 1d, 2d}`** (post-amendment 16; an 80% predicted
  band plus the median per horizon).
- A cross-sectional rank score per symbol per horizon.
- A predicted next-horizon realized vol (for Phase-5 sizing).
- A calibration certificate: per-horizon, per-vol-bucket conformal
  residual quantiles guaranteeing nominal coverage on the val window.

**Acceptance gate (locked, post-amendment 16):** the Phase-3 head,
trained from the `v3_xattn` SSL checkpoint, must satisfy ALL of:

1. **Per-horizon tri-criterion vs XGBoost.** For each horizon
   `H ∈ {30m, 120m, 1d, 2d}` independently, define the
   **tri-criterion-pass** as ALL THREE of:
   - **Cross-fold delta-IC t-stat > 1.5** (post-amendment 17 /
     baseline review D.2). For the 4 walk-forward folds, compute
     `delta_i = SSL_IC_i - XGB_IC_i` for `i ∈ {F-9..F-12}` per-fold
     mean ICs. Require `mean(delta) / (std(delta) / sqrt(4)) > 1.5`.
     This replaces the prior fixed `≥0.005` margin (review D.2: at
     1d/2d horizons the cross-fold IC std is 0.007–0.017, so a fixed
     0.005 margin was only +0.3 to +0.7 sigma — statistically toothless
     where it mattered most). The t-stat formulation adapts to
     per-horizon variance.
   - **≥3 of 4 folds** individually positive (per-fold mean rank-IC > 0).
   - **≥3 of 4 folds** individually beat XGBoost (per-fold delta > 0).

   The horizon-set tri-criterion-pass count is reported per horizon.
2. **At least 2 of 4 horizons pass the tri-criterion.** Phase 3
   acceptance is met if `count(horizons where (1) is satisfied) ≥ 2`.
   Rationale: with 4 partially-correlated horizon bets, requiring all
   4 is too tight a bar (compound P ≪ marginal P); requiring just 1
   is too loose. ≥2-of-4 is the right balance — it forces the SSL
   representation to be useful at *some* recurring scale, not just a
   lucky cherry-pick. The horizons that pass determine which
   downstream strategies survive (see Failure-mode budget below).
3. **Conformal calibration within 5% of nominal — out-of-sample.**
   Empirical coverage of the predicted 80% band on the **held-out test
   fold** of each walk-forward fold (NOT the val set used to compute
   conformal `s*`), averaged across the 4 folds, is in `[0.75, 0.85]`
   per **horizon that passed (1)**. Horizons that fail (1) are not
   subject to (3) — calibrating bands for a head that doesn't have
   alpha is meaningless. Per-vol-bucket coverage is reported in the
   tearsheet but not gated. (Architect review B.MAJOR.1 / amendment 1.)
4. **At least 2 of 4 horizons beat random-init — hard gate.** Same
   **cross-fold delta-IC t-stat > 1.5** criterion as §1.1 (post-
   amendment 17), but with the random-init full-fine-tune baseline in
   place of XGBoost. Random-init baseline: identical heads + full
   fine-tune of all backbone weights at `lr_backbone=3e-4` (no LoRA —
   LoRA on random init is degenerate; LoRA's whole premise is a useful
   base). Run **3 random seeds**; take the **best** as the comparison
   number. The set of horizons satisfying §1.4 must overlap with the
   set satisfying §1.1's tri-criterion — "the SSL backbone helps at
   the same horizons it produces alpha at". If 0 or 1 horizons clear
   random-init at delta-t > 1.5, the SSL representation is not
   contributing and Phase 2 needs revisiting. Hard gate (architect
   review B.MAJOR.3 / amendment 3), not advisory.

**Failure-mode budget** (post-amendment 16 — branches by which
horizons pass):
- **Gates 1+2 partial pass — 30m and/or 120m pass but 1d/2d fail**:
  intraday-focused strategy continues; Phase 4 OB fine-tune is high
  priority (OB conditioning sharpens the intraday horizons specifically).
  Drop the short-overnight strategy from Phase 5; ship intraday-only.
- **Gates 1+2 partial pass — 1d and/or 2d pass but 30m/120m fail**: a
  surprising but legible outcome (the SSL backbone produces useful
  representation at close-to-close horizons but not intraday). Phase 4
  OB fine-tune is downscoped (OB primarily helps intraday); Phase 5
  ships short-overnight cross-sectional swing-style strategy on the
  passing horizons.
- **Gates 1+2 fully fail (0 or 1 horizons pass)**: failure-mode-budget
  trigger — run the stronger-fallback baseline (§5.6 — LightGBM-rank
  + interactions + Optuna sweep) before declaring failure. If the
  failure margin against the stronger fallback is ≤ 0.003 on any 2
  horizons, accept the borderline pass and continue to Phase 4 with
  appropriate caution. If still 0 or 1 horizons pass: the project's
  go/no-go for Phase 4. We proceed to Phase 4 with intraday focus
  (OB sample is 13× DESIGN sizing — Phase 4 is the natural next
  lever even if Phase 3 didn't establish strong baseline alpha).
- **Gate 3 fails** (out-of-sample conformal coverage outside [0.75,
  0.85] for any horizon that passed (1)): conformal pipeline gets a
  second pass — the §7.2 picker already chooses CQR vs vanilla per
  cell; if the out-of-sample gap is structural rather than per-bucket,
  manually force vanilla globally (more robust under model
  miscalibration). If still failing, that horizon's sizing in Phase 5
  falls back to vol-targeted heuristic and we ship that horizon
  without the full quantile distribution guarantee. Other horizons
  that passed §1.3 ship normally.
- **Gate 4 fails** (SSL backbone doesn't beat random-init at any
  horizon, or only 1 of 4): Phase 2's acceptance was "Phase 3 beats
  XGBoost AND beats random-init" (architect review §C); a Phase 4
  with a useless backbone is dubious. We pause Phase 4 and retry
  Phase 2 with one of: (a) a different SSL objective family
  (predict-direction + rank, drop forecast entirely), (b) a smaller
  backbone (16M-param) more appropriate to the 2.4M sample-day pool,
  or (c) shipping Phase 4 with a feature-engineering-only initial
  backbone and treating the foundation-model framing as a Phase-7
  retrofit.

**Non-goals.** No ensembling across SSL checkpoints (we have one,
v3_xattn). No multi-task heads beyond {quantile, rank, direction, vol}
from DESIGN §3.3. No early-stop tuning per-symbol (cross-sectional
training only). No alternative SSL backbones (Phase 2 is closed).

---

## 2. Pipeline Overview

```
   data/parquet/{scale}/{symbol}.parquet      (5-channel raw OHLCV)
   data/features_parquet/{scale}/{symbol}.parquet  (18-channel features)
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Phase 3.1 Labels                                       │
   │  data/labels/{horizon}/{symbol}.parquet                 │
   │   forward log-returns at {30m, 120m, 1d, 2d}             │
   └─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
   ┌─────────────────────┐       ┌─────────────────────────┐
   │  Phase 3.2 XGBoost  │       │  Phase 3.3 Cross-symbol │
   │  baseline (no SSL)  │       │  batch sampler — TWO    │
   │  on 18-ch features  │       │  iterators: iter_30m +  │
   │  + LightGBM-rank    │       │  iter_5d, alternating   │
   │  fallback (§5.6)    │       │  steps at 75:1 cadence  │
   └─────────────────────┘       └────────────┬────────────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
              ┌─────────────────────────┐            ┌──────────────────────────┐
              │  Phase 3.4-ssl  Quantile│            │  Phase 3.4-rand  random- │
              │  + rank + vol heads on  │            │  init backbone, FULL     │
              │  FORECAST token. TWO    │            │  fine-tune + identical   │
              │  LoRA banks (lora_30m,  │            │  heads. 3 seeds; best as │
              │  lora_5d). Init from    │            │  comparison for §1.4.    │
              │  v3_xattn checkpoint.   │            │                          │
              └────────────┬────────────┘            └────────────┬─────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │  Phase 3.5 Split        │
                                │  conformal calibration  │
                                │  on val residuals       │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │  Phase 3.6 Walk-forward │
                                │  rank-IC eval vs XGB    │
                                │  + survivorship-adj     │
                                └─────────────────────────┘
```

Each box is one PR with its own tests. Acceptance is run end-to-end
only after §3.6 lands.

---

## 3. Labels — `daity/data/labels.py`

### 3.1 Definition

For each `(symbol, t)` where `t` is an `as_of` timestamp on the trading
clock, we emit a **forward log-return** at horizon `H`:

$$
y_H(\text{sym}, t) = \log\!\big( P_{\text{close}}(\text{sym},\, t + H) / P_{\text{close}}(\text{sym},\, t)\big)
$$

with **four concrete configurations** (post-amendment 16):

- **`H = 30m`** — intraday short. `t` aligned to 5m bar opens;
  horizon-end `t + 30m` is the close of the 6th forward 5m bar.
  Latest valid `t` is when the 6th forward bar's close is ≤ 15:30
  (session close); typical valid count ~70 anchors per trading day.
  Strictly intraday — never spans overnight. If the 6th forward bar
  lies past 15:30, the label is dropped.
- **`H = 120m`** — intraday medium. `t` aligned to 5m bar opens;
  horizon-end `t + 120m` is the close of the 24th forward 5m bar.
  Latest valid `t` opens at 13:30 (24 forward bars closing at 15:30);
  ~52 valid anchors per trading day. Strictly intraday.
- **`H = 1d`** — overnight close-to-close. `t` is the 15:30 IST close
  of a trading day; `t + 1d` is the 15:30 IST close of the **next
  trading day** (NSE calendar). 1 valid label per trading day per
  symbol. Spans overnight (close → next-open gap → next-close).
- **`H = 2d`** — short-swing. `t` is the 15:30 IST close of a trading
  day; `t + 2d` is the close of the **2nd trading day** forward.
  1 valid label per trading day per symbol.

Plus the cross-sectional rank-normalized form per horizon:
- **`H_xs ∈ {30m_xs, 120m_xs, 1d_xs, 2d_xs}`**: same labels but
  cross-sectionally rank-normalized within each `as_of` to uniform
  `[0, 1]`. Used by the rank head. Storing the ranked form avoids
  recomputing per batch; ranks are stable given a fixed universe at
  `as_of`.

**Why this horizon set.** 30m + 120m together span the intraday band
(half-hour to two-hour); the strategy layer in Phase 5 chooses
which intraday timescale to act on per regime. 1d + 2d span the
short-overnight band where survivorship distortion is small (only
delisted-during-window names create bias, and that fraction is much
smaller for 1-2 day windows than it was for 5d). Together the four
horizons cover every timescale the v1 strategy will plausibly trade,
without overlapping with the swing-week regime that Phase 2 evidence
suggests is signal-limited.

### 3.2 Why log-returns and not raw

DESIGN §2.4 + Phase 2 amendment "log-return form for inputs and
targets" already forced log-returns at the SSL stage. Phase 3 keeps
the same target form so the v3_xattn checkpoint's hidden state is in
its trained coordinate frame; switching to absolute returns at
post-train would force the head to learn an extra log-transform.

### 3.3 Why close-to-close, not VWAP

Three reasons. (1) `curated_ohlcv` doesn't have VWAP for all bars;
trade-count is also missing on a meaningful fraction. Close-to-close
is the lowest-common-denominator label that's reproducible from the
same data the SSL backbone already saw. (2) The Phase-5 backtest will
re-derive its own implementation-aware label (next-bar-open execution,
slippage-adjusted) — Phase-3 labels are about training signal, not
exec realism. (3) The Phase-5 strategy executes at next-bar-open or
via a slippage model, NOT at VWAP — predicting close-to-close keeps
prediction and execution in compatible coordinate frames. Predicting
VWAP labels and trading at next-bar-open would create a calibration
mismatch where the conformal band's "true" coverage in deployment
diverges from its calibration-set coverage. (Architect review B.NIT.3.)

### 3.4 Point-in-time correctness

`labels.py` operates on the same `as_of` contract as every Phase-1
reader: a row with `t = as_of` may be emitted only if the FULL forward
window's effective_close is also `≤ as_of_inferred_end`, where
**`as_of_inferred_end` is the calendar-aware end-of-trading-day
timestamp on the day of `as_of`** — for `as_of = 2025-12-31 15:25 IST`
that's `2025-12-31 23:59:59 UTC`. Encoded in
`daity.data.candles.effective_close`. (Architect review B.NIT.4.)
This means training labels at `as_of = 2025-12-31 15:25 IST` must use
forward data with `effective_close ≤ 2025-12-31 23:59:59 UTC` (or
whatever the ingestion cutoff is). The leakage gate is at
`tests/leakage/test_label_leakage.py`:

- For a random `(sym, t)`, the label uses no row with
  `effective_close > t + H + ε` where `ε` accounts for the bar
  containing the close-tick.
- Labels can never reference a bar whose effective_close is in the
  future relative to *the training query's* `as_of` — this guards
  against the subtle bug where the label producer has its own broader
  `as_of` than the training-time query.
- 100 random `(sym, t)` triples are bit-equal between full-history and
  truncated-history runs of `LabelEngine`.

### 3.5 Halts, circuits, and the `tradable` mask

Per DESIGN §2.3, halts and limit-locked bars are tagged. The label
engine emits a parallel `valid_label` mask: it's `False` if any of the
forward bars in the horizon window is a halt/circuit-locked bar.
Training drops invalid rows; they're not predicted on at inference
either.

### 3.6 Storage

Per-horizon, per-symbol Parquet files:
`data/labels/{horizon}/{symbol}.parquet` for each
`horizon ∈ {30m, 120m, 1d, 2d}`. Schema:
`(symbol str, t timestamp, t_horizon timestamp, y_log_return float32, y_rank float32, valid_label bool, halt_count int8)`.
Sorted by `t`. Mirrors `data/features_parquet/` layout so the existing
`ParquetStore.read(symbol, scale=horizon, as_of=...)` pattern works
with one trivial extension (`scale` is a free-form string at the
ParquetStore layer).

Storage cost: 4 horizons × 205 symbols × ~75K rows (intraday) or
~1.7K rows (daily) × ~40 bytes ≈ 250 MB total — trivial.

### 3.7 CLI

`daity-build-labels --horizon {30m|120m|1d|2d} --as-of 2026-04-30
--workers 8`, parallel via `ProcessPoolExecutor` over symbols (the
precompute pattern from Phase 1's feature engine). A
`--horizon all` shorthand runs all four horizons sequentially.

### 3.8 Tests

- `tests/unit/test_labels.py`: 30m / 120m labels align to bar
  boundaries; 1d / 2d labels use NSE trading calendar correctly
  (including across holidays — `t + 2d` skips weekends and
  exchange holidays); intraday horizons drop rows whose forward
  window crosses session close; halts produce `valid_label=False`;
  cross-sectional rank uniform-distributed per `as_of`. Per-horizon
  parametrized.
- `tests/leakage/test_label_leakage.py`: the three guarantees above
  hold for all four horizons.

---

## 4. Cross-Symbol Batch Construction — `daity/data/cross_section.py`

### 4.1 Why a new sampler

The Phase-2 `OHLCVPretrainDataset` is a `(symbol, end_ts)`
*per-sample* IterableDataset — every batch is N independent
`(sym, t)` pairs sampled mostly across different `t`s. That's wrong
for Phase 3 because:

1. **Cross-sectional rank head needs all symbols at the same `t`.**
   ListMLE / Spearman is computed over the cross-section.
2. **Conformal calibration is per-`as_of` exchangeable** — residuals
   from samples at the same `as_of` are not exchangeable with
   residuals from a different `as_of`, so we need to control the per-
   batch composition.
3. **Cross-sectional contrastive (DESIGN §3.2, deferred from Phase 2
   per amendment 11)** requires same-`as_of` symbols-per-batch.

So Phase 3 introduces a `CrossSectionalSampler` whose unit of
iteration is **one `t`, one full cross-section of N symbols, plus
labels at all configured horizons**.

### 4.2 Interface

```python
@dataclass
class CrossSectionalSampler:
    parquet_store: ParquetStore           # for inputs (18-ch features)
    label_store: ParquetStore             # for labels (per-horizon)
    universe: list[str]                   # ~205 symbols
    window_bars: dict[str, int]           # same as Phase 2
    as_of: datetime
    min_end_ts: datetime | None = None    # for val/test
    seed: int = 0
    drop_invalid_threshold: float = 0.5

    # FOUR horizon-specific iterators (post-amendment 16). Each yields
    # one cross-section per call at its natural cadence.
    def iter_30m(self) -> Iterator[CrossSection]: ...     # ~70 / day
    def iter_120m(self) -> Iterator[CrossSection]: ...    # ~52 / day
    def iter_1d(self) -> Iterator[CrossSection]: ...      # 1   / day
    def iter_2d(self) -> Iterator[CrossSection]: ...      # 1   / day
```

`CrossSection` carries (per-horizon, post-amendments 2 + 16 — one
horizon per emit, not a dict):
- `t: datetime`
- `horizon: str` — one of `"30m"`, `"120m"`, `"1d"`, `"2d"`
- `symbols: list[str]` — the in-universe-at-`t` subset (some symbols
  may have insufficient history)
- `inputs: dict[scale, Tensor(N, window_bars[scale], 18)]`
- `labels: Tensor(N,)` log-returns at this horizon
- `ranks: Tensor(N,)` cross-sectional rank in `[0,1]`
- `valid_mask: Tensor(N,)` of bools
- `vol_bucket: Tensor(N,)` of int8 — horizon-appropriate lookback
  (20 trading days for 1d/2d; 78 trailing 5m bars ≈ 1 session for
  30m; 156 trailing 5m bars ≈ 2 sessions for 120m); see §6.4

### 4.3 What "one cross-section" means concretely — four separate streams per horizon

**Architecture (post-amendments 2 + 16; supersedes v1's
same-`t`-with-1/75-downweight design AND the 2-horizon variant).**
Each of the 4 horizons gets its own iterator at its natural cadence;
the training loop alternates gradient steps via **weighted-random
sampling** with weights proportional to natural cadence.

```python
class CrossSectionalSampler:
    def iter_30m(self):
        # ~70 cross-sections per trading day:
        # t ∈ 5m-bar opens within 09:15..14:55 (last anchor with valid
        # 6-bar forward window). Each cross-section: all in-universe-at-t
        # symbols, 18-ch inputs, 30m forward log-return label, rank,
        # vol_bucket (78-bar lookback).
        ...

    def iter_120m(self):
        # ~52 cross-sections per trading day:
        # t ∈ 5m-bar opens within 09:15..13:30. Each: 24-bar forward
        # log-return label, rank, vol_bucket (156-bar lookback).
        ...

    def iter_1d(self):
        # 1 cross-section per trading day:
        # t = 15:30 IST close. Label: log-return to next trading day's
        # 15:30 close. vol_bucket: 20-trading-day lookback.
        ...

    def iter_2d(self):
        # 1 cross-section per trading day:
        # t = 15:30 IST close. Label: log-return to close 2 trading days
        # forward. vol_bucket: 20-trading-day lookback.
        ...
```

The training loop in `posttrain.py`:

```python
# Natural per-trading-day cadence ratios (~70 : ~52 : 1 : 1).
# Tunable via `cadence_weights` in YAML for ablation.
WEIGHTS = {"30m": 70, "120m": 52, "1d": 1, "2d": 1}
total = sum(WEIGHTS.values())  # 124

iters = {
    "30m":  sampler.iter_30m(),
    "120m": sampler.iter_120m(),
    "1d":   sampler.iter_1d(),
    "2d":   sampler.iter_2d(),
}
rng = random.Random(seed)
horizons = list(WEIGHTS.keys())
probs = [WEIGHTS[h] / total for h in horizons]

for step in range(max_steps):
    h = rng.choices(horizons, weights=probs, k=1)[0]
    cs = next(iters[h])                       # one cross-section
    loss = compute_loss(cs, horizon=h)        # picks the right LoRA bank
    accumulate_and_step(loss)
```

**Why weighted-random and not strict round-robin.** Strict rotation
("70 30m steps then 52 120m steps then 1 1d step then 1 2d step,
repeat") leaves the 1d/2d heads idle for long stretches and creates
LR-schedule artifacts (the 1d head sees its first gradient at step
122 of every "day"). Weighted-random gives every horizon a non-zero
probability per step, smoothing the gradient trajectory across
horizons. In expectation over many steps, each horizon sees its
natural-cadence share of compute.

**Why this and not the v1 single-iterator design.** v1 (then v1.5
post-architect-review) aligned horizons on the same `t` (originally
both, then 30m + 5d separately). The 4-horizon set magnifies the v1
problem: `floor_to_day(t)` is the same for 1d and 2d labels at every
intraday `t` within a day, so a same-`t` design would produce
~70 + ~52 = 122 near-duplicate 1d labels per day and 122 near-
duplicate 2d labels per day. The architect-review B.MAJOR.2 reasoning
applies even more strongly to the 4-horizon version. Separate streams
is the only architecture that respects each horizon's natural label
exchangeability.

**Eval.** At eval time, all four iterators run sequentially (not
alternating) over the test window. Each test fold produces four
independent IC tables — no mixed-horizon cross-section needed for
eval.

**Implementation note.** `CrossSection` carries a single horizon's
labels per emit (per amendment 2, unchanged). The `vol_bucket` is
horizon-specific (lookback per §6.4); same struct shape, different
contents per horizon.

### 4.4 Sub-sampling for memory

Full N=205 cross-section × 4 scales × 18 channels × max-window-bars
is 205 × (256+192+168+64) × 18 × 4 bytes = 10.3 MB per cross-section
input tensor — fits comfortably. But forward + backward through the
36M backbone on 205 samples at once needs ~6 GB activation memory at
bf16. On the H200 (143 GB) that's fine; on the local RTX-4060 (8 GB)
we'd need gradient accumulation. The implementation supports
**`max_symbols_per_step` sub-sampling**: if set, we sub-sample N
symbols out of the universe with replacement for each step (training
only; eval uses full cross-section). Default `None` (full
cross-section) on H200.

### 4.5 The "missing symbol at as_of" problem

A few symbols have late listing dates (post-2019) and won't have
enough history at every `t`. The sampler emits the
**in-universe-at-t** subset, with `valid_mask` zeroing out rows that
can't satisfy `window_bars` lookback. Cross-sections with fewer than
`drop_invalid_threshold * |universe|` valid symbols are skipped (a
loud `INFO` log so we can tune the threshold). Default 0.5: any
cross-section where < 50% of symbols are sampleable is dropped.

### 4.6 Tests

- `tests/unit/test_cross_section.py`: same `t` per cross-section;
  invalid symbols masked, not raised; rank within `[0, 1]`; vol_bucket
  in `[0, K-1]`; iteration order deterministic given seed.
- `tests/leakage/test_cross_section_leakage.py`: no input bar's
  effective_close > t; no label uses any row > `t + H`.

---

## 5. XGBoost Baseline — `daity/eval/xgboost_baseline.py`

### 5.1 Why this is the bar to beat

DESIGN §7 names XGBoost-on-engineered-features as the Phase-3
acceptance baseline. The architect review §6 reaffirms it's the
"actual question we care about". Reasons:

- It's the **strongest cheap baseline** for tabular/time-series
  cross-sectional prediction.
- It's **zero-deep-learning**, so any rank-IC delta is direct
  evidence that the SSL backbone added value (vs. "deep model on
  features beats deep model on features" which is uninformative).
- It **shares its features with our model's inputs** (the 18-channel
  feature set), so the comparison is apples-to-apples on the
  *information* axis; the only difference is what does the
  representation learning.

### 5.2 Feature construction for XGBoost

XGBoost can't ingest a 4-scale × 18-channel × variable-window
tensor. We flatten to a per-`(symbol, t)` tabular row by **summary
statistics over the lookback window** at each scale:

For each scale × each channel:
- **Last value** (the most recent bar at the scale).
- **Mean over the window**.
- **Std over the window**.
- **Min, max over the window**.
- **Linear-trend slope over the window** (regression of channel value
  vs. bar index, the slope coefficient).
- **Last 5 values flattened** (only for the 5m scale's bar_channels —
  the most recent 5 minutes of intraday context).

Total feature count: 4 scales × 18 channels × 6 stats + 5 × 5 last-5m
features = 432 + 25 = **457 features**. XGBoost handles this
comfortably; multicollinearity is irrelevant for tree models.

### 5.3 Labels

Same `y_log_return` per horizon as §3. Two separate XGBoost models,
one per horizon. Both with `objective='reg:squarederror'` (the
log-return MSE objective) — we'll get rank-IC at eval time directly
from the predictions; no need for a custom rank objective at the
baseline level.

### 5.4 Training

- 5-fold time-series-split CV (sklearn `TimeSeriesSplit`) within
  the train window for hyperparam search (max_depth ∈ {4, 6, 8},
  learning_rate ∈ {0.01, 0.03, 0.1}, n_estimators ∈ {500, 1500},
  subsample = 0.8, colsample_bytree = 0.5, min_child_weight = 100).
- Final model retrained on full train window with best hyperparams.
- One model per (horizon, walk-forward fold) — 12 folds × 2 horizons
  = 24 models, ~10 min each on CPU = 4 hours wall-clock. Trivial.

### 5.5 Output

`reports/phase3_xgboost_baseline.json` with per-fold per-horizon
rank-IC, average-and-CI, and feature importance top-20. This is the
**locked baseline number** the SSL+head model has to beat.

### 5.6 Stronger-fallback baseline (LightGBM + Optuna)

The 457-feature XGBoost is the **minimum competent baseline**. To
guard against the embarrassing failure where a 1-day-better baseline
would have changed the gate decision, the failure-mode budget §1
defines the acceptance gate as "SSL+heads beats *the better of*
baseline-XGBoost and stronger-fallback". If §1.1 / §1.2 fail by a
margin ≤ 0.003 against XGBoost, run the stronger fallback before
declaring failure:

- **LightGBM with `objective='rank_xendcg'`** as the rank head (more
  recent than XGBoost's rank objective; sometimes outperforms).
- **Interaction features** between top-decile importance pairs after
  a first-pass fit: explicit ratios like `vol_z_5m_last /
  vol_z_60m_last`, `ret_5_5m_mean / ret_60_60m_mean`. Adds 1-2% IC on
  cross-sectional tasks empirically.
- **Optuna sweep, 50 trials**, over `(num_leaves, learning_rate,
  feature_fraction, bagging_fraction, min_child_samples,
  lambda_l1, lambda_l2)`. ~4 hours CPU.

The stronger fallback is implemented but not run by default — it's
gated on §1's failure-margin trigger to keep the default path lean.

### 5.7 Leakage gate

`tests/leakage/test_xgboost_features_leakage.py`. The 457 flattened
features are derived from Phase-1's 18-channel feature engine, so
they inherit those leakage gates — BUT the per-window aggregation
(`mean`, `std`, `min`, `max`, `slope`, `last`) introduces a new
leakage opportunity. Specifically: "linear-trend slope over the
window" computed across `(t - window, t]` includes the bar at `t`
(open == start of bar at t); computed across `[t - window, t)`
excludes it. The difference is one bar — and one bar is the
difference between leakage and not-leakage when the bar at `t`
hasn't closed yet. The gate tests that all 6 aggregations operate
on `[t - window, t)` strictly, with a 100-triple `(sym, t)` random
spot-check. (Architect review C.5 / B.MINOR.1.)

### 5.8 Tests

`tests/unit/test_xgboost_baseline.py`: train on synthetic data
(linear y with noise) and assert rank-IC > 0.5; deterministic given
seed; feature flattening dimensions match.

---

## 6. The Supervised Heads + LoRA — `daity/training/posttrain.py`

### 6.1 What the heads attach to

The v3_xattn checkpoint has a backbone producing
`encoder_hidden ∈ ℝ^(B, 82, 480)` (FORECAST token at position 0, then
81 patch tokens). Phase 3 attaches **three new heads** (post-amendment 4
— direction head dropped from v1), all reading from the FORECAST
token's hidden:

- **Quantile head** — `Linear(480, 480) → GELU → Linear(480, 5)`
  emitting `(q_0.1, q_0.25, q_0.5, q_0.75, q_0.9)` log-return
  quantiles. Pinball loss across all five quantiles.
- **Rank head** — same shape ending at `Linear(480, 1)`, emitting a
  scalar rank score per `(sym, t)`. ListMLE loss within the
  cross-section batch.
- **Realized-vol head** — `Linear(480, 1)`, predicts
  `log(sqrt(realized_var))` over the horizon window. MSE. Used by
  Phase 5's sizing layer; also a known regularizer for return-
  prediction heads.

All three read from a *single* FORECAST token's hidden — no
cross-attention forecast head this time, because Phase 2 confirmed
the bottleneck wasn't head architecture. The architect review §3
recommended pinball at Phase 2 for alignment; Phase 3 honors that
at the *natural* place for it.

**Why drop the direction head from v1.** The architect review §A.2
called the trade-off: 3-class direction CE adds gradient pressure on
the FORECAST token toward sign discrimination, but the rank head
*already* provides this implicitly via within-batch ordering. The
direction head adds two more loss-weight knobs (`w_dir_30m`,
`w_dir_5d`) without a downstream consumer Phase 5 needs that
rank+quantile doesn't already cover. May come back behind a config
flag in Phase 5 when sizing-by-direction-confidence is on the
critical path.

**Per-horizon multi-tasking with FOUR LoRA banks (post-amendments 5
+ 16).** Each head has FOUR instances, one per horizon
`{30m, 120m, 1d, 2d}`, sharing the FORECAST token but separate
parameters. The H-horizon head reads from the FORECAST hidden
produced under `lora_H`. Heads are tiny (~1.2M params total across
12 instances = 3 heads × 4 horizons) — the LoRA + head trainable-
param count is dominated by the four LoRA banks (~736K params total
at `r=8` on `(q_proj, v_proj)` × 12 layers × 4 banks).
Combined trainable surface: ~2M params, <6% of frozen backbone.

### 6.2 Loss

Post-amendments 2 + 4 + 11 + 16 — direction term dropped, 0.2× factor
removed (each horizon trains at its own natural cadence per §4.3),
weights set by magnitude-balance heuristic at step 0, four horizons:

```python
# Per gradient step, ONE horizon is active (weighted-random per §4.3):
# h ∈ {"30m", "120m", "1d", "2d"}
L = (
    w_q[h]  * pinball_loss(quantile_head[h](z[h]),    y[h])
  + w_r[h]  * listmle_loss(rank_head[h](z[h]),       ranks[h])
  + w_v[h]  * mse_loss(vol_head[h](z[h]),         vol_target[h])
)
# z[h] = FORECAST_hidden produced under lora_H
```

**Initial weights via magnitude-balance, not hardcoded.** v1 wrote
`(1.0, 1.0, 0.3)` from intuition. Post-amendment 11 (architect review
B.MINOR.2): at step 0 on a single batch per horizon, compute
per-head loss values and pick `w_h` such that `w_h × L_h(step=0) ≈
1.0` for each head within the horizon. Test
`tests/unit/test_posttrain.py::test_loss_weights_balance_at_init`
asserts each head's `w_h × L_h(step=0)` is within 2× of the others
(a regression catch if heads change later). Weights are written
into `posttrain_v1.yaml` after the first measurement and locked for
reproducibility (`config_hash` determinism).

**Why both quantile AND rank heads.** DESIGN §3.3 lists both. The
quantile head is what Phase 5's sizing layer consumes (need a
distribution); the rank head is what the cross-sectional strategy
*directly* trades on. Computing rank from the median quantile would
work, but ListMLE on a dedicated scalar gives a cleaner gradient for
the ranking objective (the median quantile is also pulled by the
pinball loss toward the center of the residual distribution, not
toward a discriminative ordering). Cost is one extra `Linear(480,1)`
per horizon — trivially cheap. (Architect review §A.2 confirmed.)

### 6.3 LoRA on the backbone — two adapter banks

DESIGN §3.4 has LoRA only at the OB fine-tune stage; we **promote**
LoRA to Phase 3 because:

1. The architect review §1 (Phase-2 review) flags the 36M backbone as
   data-efficient-but-overfittable on our 2.4M sample-day pool. Full
   fine-tuning of the backbone is the most overfit-prone option.
2. `peft` LoRA on `(W_q, W_v)` of every backbone attention layer at
   `r=8, alpha=16` adds **~184K trainable params per adapter bank**
   on the 12-layer d_model=480 backbone (`2 projections × 12 layers
   × (480 × 8 + 8 × 480) × 2 ranks` for `r=8`, plus a similar number
   for `lora_B`; total ~184K). **Four banks (one per horizon) ≈
   736K** (post-amendment 16). Combined with heads' ~1.2M, total
   trainable ≈ ~2M — about 5–6% of the backbone's 36M frozen params.
3. LoRA at Phase 3 prepares the codepath for Phase 4 (where LoRA is
   mandatory per DESIGN). One implementation, two phases.

LoRA config (locked, post-amendments 5 + 16 — FOUR banks via
`peft`'s named-adapter API; B.MINOR.4 reconciliation done — narrative
and config now agree):

- `target_modules = ['q_proj', 'v_proj']` — only the Q and V
  projections per the LoRA paper's strongest recipe (not all 4
  attention projections; see B.MINOR.4 reconciliation). Smaller
  trainable surface, lower overfit risk for our scale.
- `r = 8, alpha = 16, lora_dropout = 0.05`.
- **Four banks**: `lora_30m`, `lora_120m`, `lora_1d`, `lora_2d`,
  switched per horizon via
  `peft.set_adapter('30m' | '120m' | '1d' | '2d')` between gradient
  steps.
- Backbone params frozen (`requires_grad=False`); four LoRA banks
  + three heads × four horizons (= 12 head instances) trainable.

**Why `(q, v)` only.** The LoRA paper's strongest recipe at scale.
`(q, k, v, o)` doubles the trainable surface for marginal gain on
small downstream tasks. B.MINOR.3 / B.MINOR.4 flagged this; the
sensitivity to `r` and `target_modules` is logged as a Phase 3.5
fallback ablation (run `r ∈ {4, 16}` if §1.1 fails by ≤ 0.003).

**Sensitivity.** `r=8` chosen by analogy to the LoRA paper. Not
swept. If §1.1 fails by a small margin, the cheapest escape is a
`r ∈ {4, 16, 32}` sweep. Logged as a fallback in §11.

### 6.4 Vol bucket

Per DESIGN §3.3 + §6.5, conformal calibration is **stratified by
realized-vol bucket** so the predicted band adapts to regime. We
pre-compute per-`(sym, t)` realized vol over the **horizon-appropriate
lookback**:

| Horizon | Vol-bucket lookback |
|---|---|
| 30m  | 78 trailing 5m bars (~1 session) |
| 120m | 156 trailing 5m bars (~2 sessions) |
| 1d   | 20 trading days |
| 2d   | 20 trading days |

Bin into K=5 quintiles. The bucket index is carried in the batch,
used at conformal time only (not at training).

**Cell-size sanity (post-amendments 15 + 16 / B.MINOR.5).**
Per-horizon × per-bucket cell sizes on the 4-month val window:

| Horizon | Independent samples | Per-bucket (K=5) |
|---|---|---|
| 30m  | ~84 days × ~70 t/day × 205 ≈ 1.2M | ~240K |
| 120m | ~84 days × ~52 t/day × 205 ≈ 0.9M | ~180K |
| 1d   | ~84 × 205 ≈ 17K | ~3.4K |
| 2d   | ~83 × 205 ≈ 17K | ~3.4K |

Intraday horizons are comfortably above the 1000-sample threshold;
1d/2d are above but not by orders of magnitude. The conformal
calibration code logs per-cell sample counts and **warns if any cell
has <1000 effective independent samples**. If the 1d or 2d × top-
vol-bucket cell falls below 1000, fall back to K=3 buckets for that
horizon. Logged in W&B as a calibration-time metric.

### 6.5 EMA

Phase 2's `EMACallback(decay=0.999)` is reused. EMA is swapped in for
val and test eval. Phase 3 uses 0.999 by default; if val rank-IC
oscillates step-to-step (a known LoRA + small-head pathology — small
trainable param count + 30K-step budget can produce a noisy moving
average), bump to 0.9995 as a one-line YAML change. Logged in W&B and
tracked. (Architect review B.MINOR.8.)

The architect review B.MINOR.8 noted Phase 2's default may not port:
Phase 2 is 50K steps × heavy regularization; Phase 3 is 30K steps ×
~3M trainable params. The "EMA averages over the last ~1000 steps"
window represents the last 3% of Phase 3 training (vs ~2% in
Phase 2). Probably doesn't matter; the canary metric (`ema_delta`
in W&B per §6.12) makes it visible if it does.

### 6.6 Optimization

- AdamW, `lr_lora = 3e-4, lr_head = 1e-3, lr_backbone = 0` (frozen).
- Cosine schedule with 5% warmup, `lr_min_ratio = 0.05`.
- `weight_decay = 1e-3` on heads, `0` on LoRA (per `peft` recipe).
- `gradient_clip_val = 1.0` (Phase 2 lesson — heavy-tailed targets).
- bf16-mixed.
- Batch unit: one cross-section. Effective batch size = N_symbols ×
  number-of-cross-sections-per-step. With `max_symbols_per_step=128`
  and 4 cross-sections per step (gradient accumulation), effective
  batch ≈ 512 samples — same as Phase 2's `pretrain.yaml`.
- `max_steps = 30K` (Phase 3 trains a HEAD; doesn't need 150K). On
  H200 with frozen backbone forward, ~6 hours wall-clock.

### 6.7 Schedule + LoRA-only fallback

DESIGN §3.3: "backbone frozen for the first epoch, then unfrozen with
a 10× lower LR than heads". Phase 3 amends this to **backbone always
frozen; LoRA trainable from step 0**. Reasoning: with LoRA, "unfreeze
the backbone" is equivalent to "raise LoRA LR", which we don't need
because the LoRA is small enough that it doesn't blow up the
representation. The DESIGN-prescribed warmup-then-unfreeze schedule
is for FULL fine-tuning; LoRA is more stable.

**LoRA-only fallback escape hatch (post-amendment 12 / architect
review §A.1).** If the random-init ablation (§1.4) outperforms
LoRA-only on val rank-IC by ≥0.002 averaged across the 4 horizons
on fold F-9, switch to `unfreeze_top_n_layers=2` with
`lr_backbone=lr_lora * 0.1`, no LoRA. This is a one-line config flag
in `posttrain_v1.yaml` (`finetune_mode: "lora" | "partial_unfreeze"`).
The trigger is sample-efficient: it costs one extra fold's worth of
compute (~6 H200-hours) and the corner-cut would otherwise be
invisible until Phase 5.

### 6.8 Dropout

DESIGN §3.4 calls for fusion dropout 0.3 in Phase 4. For Phase 3 (no
OB), dropout 0.1 on head inputs (FORECAST hidden) only. Backbone
and LoRA modules use their default dropout (0 in Phase 2 config).

### 6.9 Tests

- `tests/unit/test_posttrain.py`: pinball loss correctness; ListMLE
  gradient sign (predictions move toward the correct order); LoRA-
  wrapped backbone has LoRA params + frozen base; named-adapter
  switch (`peft.set_adapter('30m' | '120m' | '1d' | '2d')`) is
  correctly honored (post-amendments 5 + 16); backbone state-dict
  is round-trip safe; 1-batch overfit reduces all three loss terms
  across 100 steps for all 4 horizons; loss-magnitude-balance
  assertion at init (post-amendment 11); hparam-compat refuses on
  mismatched checkpoint (post-amendment 7); weighted-random
  alternation hits cadence ratio in expectation (post-amendments
  2 + 16).
- `tests/leakage/test_posttrain_leakage.py`: 100 random `(sym, t)`
  triples — the Phase-3 module's forward + loss never reads
  `effective_close > t + max_horizon`.

### 6.10 Hparam compatibility check

Post-amendment 7 / architect review §A.10. The v3_xattn checkpoint
encodes the backbone hparams it was trained with —
`d_model`, `n_layers`, `n_heads`, `ffn_ratio`, `revin_affine` (must
be `False` per DESIGN amendment 12), `feature_channels=18`, `scales`,
`window_bars`, `patch_len`, `patch_stride`, `forecast_head_type`
(must be `cross_attention` for v3_xattn). Posttrain MUST match these
for `state_dict` loading to work, AND for the encoded representation
to be in the same coordinate frame the heads will train against.

**`posttrain_cli.py` extracts `backbone_hparams` from the checkpoint's
stored hparams and asserts they match the live Hydra config. On
mismatch: raise `IncompatibleCheckpointError` listing the deltas. No
silent coercion; no `strict=False` state-dict load.** Test:
`tests/unit/test_posttrain.py::test_hparam_mismatch_refuses` (creates
a synthetic checkpoint with one hparam off, asserts the error is
raised with the offending key in the message).

**Provenance triple extension.** Phase 2's
`(git_sha, config_hash, as_of)` triple extends to Phase 3 outputs:
the conformal artifacts (§7.4) and walk-forward artifacts (§8.4) are
written with a `provenance.json` sidecar carrying the triple plus
the `parent_checkpoint_hash` (sha256 of the v3_xattn checkpoint
file). This makes "which checkpoint was Phase 3 run from" auditable.
(Architect review C.5.)

### 6.11 Inference contract

The Phase 3 deliverable is a model that emits predictions at strategy
time. Phase 5 consumes this; making the contract explicit here saves
Phase 5 plumbing-debug. (Architect review C.5.)

**Input** (per `(symbol, t)` query at inference):
- 18-channel feature window at every scale, ending at `effective_close
  ≤ t`. Same shape as training.
- `as_of = t` enforced by the data layer.
- `horizon ∈ {"30m", "120m", "1d", "2d"}` selects which LoRA bank
  (`lora_30m | lora_120m | lora_1d | lora_2d`) is active.
- `vol_bucket: int8` computed from realized vol over the horizon-
  appropriate lookback (§6.4) — used to look up the conformal `s*`.

**Output**:
- `q: Tensor(5,)` — five quantile predictions
  `(q_0.1, q_0.25, q_0.5, q_0.75, q_0.9)` of forward log-return.
- `r: float` — rank score (cross-sectional ranking value).
- `vol: float` — predicted log-realized-vol over horizon.
- `band_lo: float`, `band_hi: float` — calibrated 80% prediction band,
  computed from the conformal `s*` table at `(horizon, vol_bucket)`.
- `valid_input: bool` — `False` if the input window contains a
  halt/circuit (strategy filters on this); model still emits
  predictions but the strategy ignores them.

**Latency budget**:
- Single-symbol inference: target ≤ 50 ms on CPU (head forward + LoRA-
  adapted backbone forward, no batching). Acceptable for 30-min
  rebalance cadence.
- Cross-section inference: 205 symbols at one `t` should target
  ≤ 5 s on CPU at the strategy machine, ≤ 0.5 s on GPU. Met
  trivially by batching the cross-section.

**Conformal lookup**: `conformal_table.json` is loaded once at startup
into a dict `{(horizon, vol_bucket): {"s_star": float, "score_fn":
"cqr" | "vanilla"}}`. Lookup at inference is O(1).

**Versioning**: the model artifact bundles
`(checkpoint.ckpt, conformal_table.json, provenance.json,
posttrain_v1.yaml)` into a single tarball
`runs/phase3_{config_hash}.tar`. Phase 5 loads the tarball; no
loose-file dependencies.

### 6.12 W&B tracking

Phase 2 invested in W&B; Phase 3 continues it. Logged metrics
(architect review C.5):

All metrics emitted per horizon `H ∈ {30m, 120m, 1d, 2d}`
(post-amendment 16):

**Per-step (every gradient step):**
- `train_loss_pinball_{H}` — pinball loss per horizon.
- `train_loss_listmle_{H}` — rank loss per horizon.
- `train_loss_vol_{H}` — vol-MSE per horizon.
- `step_horizon` — which horizon was sampled this step (for
  cadence-debugging of the weighted-random sampler).
- `grad_norm_{lora_30m, lora_120m, lora_1d, lora_2d, heads}` —
  gradient norms by trainable group; flags cross-horizon gradient
  imbalance if it shows up (e.g., 1d/2d LoRA banks under-trained
  due to natural cadence weighting).
- `lora_weight_norm_{H}` — per-bank LoRA parameter L2 norm trajectory.
- `revin_gamma_norm` — Phase 2 architect review §9 canary; sanity
  check that the SSL-trained gamma is still where it was at
  checkpoint load.

**Per-validation (every 500 steps):**
- `val_rank_ic_{H}` — Spearman rank-IC on val cross-sections.
- `val_pinball_{H}` — held-out pinball.
- `ema_delta` — EMA-vs-live weight difference; flags if EMA is
  oscillating (B.MINOR.8 trigger).

**Per-epoch:**
- `per_symbol_ic_{H}` — diagnostic table (per architect review
  C.5.1) showing which symbols the model wins/loses on, per horizon.
  Surfaced as a W&B table artifact, not a single scalar.

---

## 7. Conformal Calibration — `daity/models/conformal.py`

### 7.1 Method — both CQR and vanilla, picked per cell

Post-amendment 9 / architect review §A.5: we implement BOTH CQR and
vanilla split-conformal, and at calibration time **pick whichever
gives empirical coverage closer to `1-α`** per
`(horizon, vol_bucket)` cell. CQR's adaptivity is only better when
the quantile head is well-calibrated; on a noisy 8h-ahead target
with a quantile head trained for ~30K steps on a possibly-uninformative
backbone, this isn't a sure thing. Implementing both costs ~30 lines
and the picker is the empirical safety net.

**Split conformal regression** (Vovk 2005, Romano-Patterson-Candès
2019). The val window is the calibration set; we don't touch the
test window. For each horizon × vol-bucket cell, we compute TWO
conformal scores per residual:

**CQR score** (heteroscedastic; Romano-Patterson-Candès Eq. 6):

$$
s^{\text{cqr}}_i = \max\!\left(\frac{\hat{q}_{0.1}(x_i) - y_i}{\hat{q}_{0.5}(x_i) - \hat{q}_{0.1}(x_i)},\ \frac{y_i - \hat{q}_{0.9}(x_i)}{\hat{q}_{0.9}(x_i) - \hat{q}_{0.5}(x_i)}\right)
$$

**Vanilla score** (constant-width; absolute residual):

$$
s^{\text{van}}_i = |y_i - \hat{q}_{0.5}(x_i)|
$$

The empirical `(1-α)`-quantile of `{s_i}` per cell × per score is
computed; the cell's selected score is whichever yields an empirical
val-set coverage in `[1-α-ε, 1-α+ε]` first (smallest ε wins). The
choice is stored in the conformal JSON.

At inference, the predicted 80% band for a new `(sym, t)` depends on
the picker:

- If CQR was selected for the cell:
  ```
  band_lo = q_0.5(x) - (1 + s*) * (q_0.5(x) - q_0.1(x))
  band_hi = q_0.5(x) + (1 + s*) * (q_0.9(x) - q_0.5(x))
  ```
- If vanilla was selected:
  ```
  band_lo = q_0.5(x) - s*
  band_hi = q_0.5(x) + s*
  ```

with `s*` and the score function looked up by
`(horizon, vol_bucket(x))`.

### 7.2 When CQR vs vanilla wins

Vanilla split-conformal gives a single `s*` per horizon-cell — the
band is locally constant-width regardless of how confident the model
is at a given `(sym, t)`. For an asset that's in a low-vol regime,
this overstates uncertainty; in a high-vol regime, it understates.
**CQR uses the model's own quantiles to *modulate* the band by
predicted volatility.** This is the right inductive bias for finance
and matches how MOIRAI / Chronos report their bands — *when the
quantile head is well-calibrated*.

When the quantile head is poorly calibrated (e.g., consistently
over-narrow `(q_0.1, q_0.9)` band on the val set), CQR amplifies the
miscalibration: it scales an already-bad estimate. Vanilla is more
robust under that pathology. The picker per §7.1 routes around the
problem at calibration time.

### 7.3 Calibration set sizing

Per-cell calibration set ≥ 1000 samples for the empirical `s*` to be
stable to within ±5% relative error (binomial CI). Our val window is
4 months × 75 cross-sections/day × 205 symbols ≈ 5M samples; per
(horizon, vol_bucket=K=5) cell that's ~500K. More than enough.

### 7.4 Storage

`reports/conformal/{horizon}_{vol_bucket}.json` —
`{"alpha": 0.2, "s_star": 1.234, "n_calibration": 432123, "ci": [...]}`.
Loaded at inference time.

### 7.5 Coverage check — out-of-sample is the gate

Empirical coverage on val (the calibration set itself, in-sample) is
≥ `1-α-ε` by construction (split conformal's marginal coverage
guarantee — true for both CQR and vanilla scores). The in-sample
check is therefore a **sanity assertion** (catches code bugs in the
quantile lookup or score computation), not an acceptance gate.

The acceptance gate §1.3 (post-amendment 1) requires coverage on the
**held-out test fold** of each walk-forward fold, averaged across
the 4 folds, in `[0.75, 0.85]` per horizon at the top level
(per-vol-bucket reported but not gated). This is where calibration
can actually fail (distribution shift between val and test windows;
per-bucket sample-size insufficiency exposing a stratification
boundary). Architect review B.MAJOR.1: the v1 doc had this in
§7.5 only as a "stronger test deferred to Phase 5"; the deferral was
the bug — the §1.3 gate without it was vacuous.

### 7.6 Tests

- `tests/unit/test_conformal.py`: synthetic data with known noise
  distribution; conformal coverage matches nominal within 5% on
  calibration set; bands grow with predicted vol; CQR score
  computation matches Romano-Patterson-Candès Eq. 6.

---

## 8. Walk-Forward Evaluation — `daity/eval/walk_forward.py`

### 8.1 Setup

The Phase-1 `WalkForwardSplitter` already produces 12 quarterly
folds 2019–2025 with 10-day embargo. Phase 3 evaluates on **the most
recent 4 folds** (covering the last year of training data plus 4
months of val):

| Fold | Train | Val | Test |
|---|---|---|---|
| F-9 | 2019-Q1 → 2024-Q3 | 2024-Q4 | 2025-Q1 |
| F-10 | 2019-Q1 → 2024-Q4 | 2025-Q1 | 2025-Q2 |
| F-11 | 2019-Q1 → 2025-Q1 | 2025-Q2 | 2025-Q3 |
| F-12 | 2019-Q1 → 2025-Q2 | 2025-Q3 | 2025-Q4 |

**Why only 4 folds and not 12.** Each fold = full Phase-3 train (LoRA
+ heads) + conformal calibration + eval. ~6 H200-hours per fold per
horizon = 48 hours total. Doable but heavy. The 4 most recent folds
cover the regime that matters for go-live; older folds (2019–2023)
are out of regime for the v1 deployment.

**Decision rule — per-horizon tri-criterion + ≥2-of-4-horizons gate
(post-amendments 8 + 16).** §1.1 / §1.2's gate operates per horizon
`H ∈ {30m, 120m, 1d, 2d}`:

For each horizon, the **tri-criterion-pass** requires ALL THREE of:

1. **Mean** rank-IC across 4 fold test windows beats XGBoost mean by
   ≥0.005.
2. **At least 3 of 4 folds** are individually positive on rank-IC.
3. **At least 3 of 4 folds** individually beat XGBoost on rank-IC.

Phase 3 acceptance §1.1 requires `count(horizons that pass tri-
criterion) ≥ 2`.

The "mean of 4" alone has two pathologies the tri-criterion closes:

- **Single-fold dominance.** Walk-forward folds in volatile periods
  (2025-Q2 if there was a regime shift) have IC variance ~5× higher
  than calm periods. A single lucky fold can flip the mean while
  underlying signal is unstable. Phase 5 deployment is into the
  *next* regime, which is one fold's worth of variance.
- **No monotonicity check.** F-9 wins by 0.02, F-10/F-11/F-12 lose by
  0.005 each → the mean still beats XGBoost, but the live trajectory
  looks like F-12.

(Architect review §A.7 + amendment 16.)

### 8.2 Metrics

Per fold × per horizon:
- **Spearman rank-IC** of predicted-median vs realized log-return,
  averaged over `t` within the test window.
- **Top-bottom decile spread** in basis points, with t-stat.
- **CRPS** on the predicted quantile distribution.
- **Calibration**: empirical coverage of 80% band, per vol-bucket
  (reported, not gated, per amendment 1).
- **Per-symbol IC table** (post-amendment 15 / architect review C.5.1):
  rank-IC computed *per symbol* over the test window, sortable by
  IC. Surfaces "the model is great at large-cap PSU banks and bad at
  small-cap pharma" findings that the strategy layer will use to
  filter / weight the universe in Phase 5. Bottom-10 and top-10
  symbols by IC are highlighted.

**Invalid-row handling** (post-amendment 15 / architect review
B.MINOR.6). Two masks:
- `valid_label`: false if any forward-horizon bar was a halt/circuit
  (§3.5). Rows with `valid_label=False` are *excluded* from rank-IC
  and CRPS metrics.
- `valid_input`: false if any input-window bar was a halt/circuit.
  The model still emits predictions for these rows (the input is
  technically present), but the strategy layer in Phase 5 filters
  them out. Phase 3 reports rank-IC under both
  "all valid_label rows" and "valid_label AND valid_input rows" —
  the latter is the deployment-aligned number.

Survivorship handling: see §9 (post-amendment 6 — no κ shrinkage,
just an audit panel).

### 8.3 Comparison

For each fold × horizon × metric: SSL+heads vs XGBoost vs
random-init ablation. Three columns. Acceptance §1 reads off the
tri-criterion across the 4 folds. The random-init column is the
hard gate per §1.4 (post-amendment 3).

### 8.4 Walk-forward output — JSON only (HTML rendering deferred to Phase 5)

Post-amendment 13 / architect review C.4. v1 had Phase 3 generate
both JSON AND an HTML tearsheet; this is a phase-bundling violation.
Phase 3 writes a structured JSON; Phase 5's `tearsheet.py` (per
DESIGN §5.1's `eval/tearsheet.py` placement) renders the HTML.

`reports/phase3_walk_forward.json` schema (post-amendment 16 — 4
horizons):
```jsonc
{
  "provenance": { "git_sha": "...", "config_hash": "...",
                  "as_of": "...", "parent_checkpoint_hash": "..." },
  "folds": [
    {
      "fold_id": "F-9", "horizon": "30m",
      "ssl_heads": { "rank_ic": 0.011, "crps": ..., "decile_spread_bps": ...,
                     "coverage_80": 0.81, "per_symbol_ic": { ... } },
      "xgboost":   { ... },
      "random_init": { ... }
    },
    // F-9..F-12 × {30m, 120m, 1d, 2d} → 16 entries total
  ],
  "tri_criterion": {
    "30m":  { "mean_delta": 0.007, "folds_positive": 4,
              "folds_beat_xgb": 3, "passed": true },
    "120m": { ... },
    "1d":   { ... },
    "2d":   { "mean_delta": 0.001, "folds_positive": 2,
              "folds_beat_xgb": 1, "passed": false }
  },
  "horizons_passed": ["30m", "120m", "1d"],     // §1.1 outcome
  "n_horizons_passed": 3,
  "phase3_gate_1_passed": true,                 // ≥2 of 4
  "phase3_gate_3_passed": true,                 // conformal OOS
  "phase3_gate_4_passed": true,                 // beats random-init at ≥2
  "survivorship_audit": [
    { "fold_id": "F-9", "n_symbols_at_train_start": 148,
      "n_symbols_at_test_end": 205,
      "missing_at_train_start": ["SYM_A", ...] }, ...
  ]
}
```

### 8.5 Tests

- `tests/unit/test_walk_forward.py`: rank-IC computation matches
  scipy.stats.spearmanr; CRPS implementation matches
  properscoring.crps_ensemble; tri-criterion decision rule correctly
  identifies pass/fail edge cases (3-of-4-positive but mean-fails,
  4-of-4-positive but mean-fails, etc.); JSON schema round-trips.

---

## 9. Survivorship Handling

### 9.1 Why this is Phase 3, not Phase 5

DESIGN §2.3 + amendment 4 commit to "survivorship-adjusted Sharpe
range" in the **Phase-5 backtest tearsheet**. The Phase-2 architect
review §10 + Phase-3 architect review §A.8 both want similar
honesty in Phase 3 IC metrics.

**Post-amendment 16, the survivorship surface area shrinks.** v1's
horizon set included 5d swing, which was the most survivorship-
exposed (multi-day windows had non-trivial probability of touching
a delisted name during 2019-2025). The post-amendment-16 horizon
set caps at 2d, where the per-window probability of intersecting a
delisting is small (under 1% historically for NSE F&O 200). The
audit panel below remains valuable but the materiality of
survivorship-bias on the 4-horizon set is markedly lower than on
the v1 30m+5d set.

### 9.2 No multiplicative shrinkage — audit panel only (post-amendment 6)

v1 of this doc proposed a fixed multiplicative shrinkage
`(κ_30m = 0.02, κ_5d = 0.10)`. **Dropped per architect review §A.8.**
Two reasons:

1. The `(0.02, 0.10)` numbers were not derived from anything specific
   to NSE F&O 200 over 2019–2025; they were "industry rough
   estimates". The actual survivorship pattern in our universe is
   bidirectional: symbols that became F&O-eligible *during* our
   2019–2025 window are also missing from early-window training
   (since they weren't F&O in 2019). A single multiplicative κ
   doesn't capture the IPO direction.
2. Survivorship doesn't shrink rank-IC linearly. It changes the
   distribution of which symbols are in the cross-section, and the
   IC under a different cross-section is *not* `(1 - κ) × original
   IC`. It can be more, less, or sign-flipped.

**The honest claim, lean into it:** the IC-**delta** vs XGBoost is
unbiased under shared-universe survivorship. Both sides see the same
distortion; the delta is the statistic of interest, and it's the one
acceptance §1.1 / §1.2 actually gate on. We report raw IC and raw
delta. We do NOT report a "survivorship-adjusted IC".

### 9.3 What we DO report — survivorship audit panel

Each fold's tearsheet entry carries:

- **Universe size at train start** (e.g., F-9 train begins 2019-01-01:
  N=148 symbols traded; vs N=205 in 2026-04-30).
- **Names missing at train start** that are present in the test
  window (the IPO/F&O-listing symbols).
- **Names missing at test end** that were present at train start
  (the delisted symbols — currently always 0 since we have no
  delisted history; this row is a placeholder for when delisted
  data lands).
- **Per-fold change in cross-section composition.**

This is a fact, not a heuristic. The reader can apply their own
discount.

### 9.4 Replacement when delisted history lands

If we backfill delisted symbols in a future phase, the audit panel
gets richer (delisted-during-test column populates) and the prose
caveat shrinks. The IC-delta interpretation doesn't change.
Tracked as Phase-1 tech debt.

---

## 10. Repo Layout

New files (no existing files modified materially in Phase 3 beyond
adding entries to `daity/training/__init__.py` and similar):

```
daity/
├── data/
│   ├── labels.py                  # NEW: §3
│   └── cross_section.py           # NEW: §4 (two iterators per amend. 2)
├── models/
│   └── conformal.py               # NEW: §7 (CQR + vanilla per amend. 9)
├── training/
│   └── posttrain.py               # NEW: §6 (two LoRA banks per amend. 5)
├── eval/
│   ├── metrics.py                 # EXTEND: rank-IC, CRPS, calibration
│   │                              #         (file may exist from prior phases)
│   ├── xgboost_baseline.py        # NEW: §5
│   ├── lightgbm_fallback.py       # NEW: §5.6 stronger fallback
│   └── walk_forward.py            # NEW: §8 (JSON output; HTML in Phase 5)
├── scripts/
│   ├── build_labels.py            # NEW: CLI for §3
│   ├── xgboost_baseline_cli.py    # NEW: CLI for §5
│   ├── posttrain_cli.py           # NEW: CLI for §6 (hparam-compat amend. 7)
│   └── walk_forward_cli.py        # NEW: CLI for §8
configs/
└── train/
    ├── posttrain_v1.yaml          # NEW: §6 config (two-LoRA-bank schema)
    └── walk_forward_v1.yaml       # NEW: §8 fold config
tests/
├── unit/
│   ├── test_labels.py             # NEW
│   ├── test_cross_section.py      # NEW (separate-stream sampling)
│   ├── test_posttrain.py          # NEW (incl. hparam-compat,
│   │                              #      loss-magnitude balance)
│   ├── test_conformal.py          # NEW (CQR + vanilla pickers)
│   ├── test_xgboost_baseline.py   # NEW
│   └── test_walk_forward.py       # NEW (incl. tri-criterion edge cases)
└── leakage/
    ├── test_label_leakage.py      # NEW
    ├── test_cross_section_leakage.py     # NEW
    ├── test_posttrain_leakage.py         # NEW
    └── test_xgboost_features_leakage.py  # NEW (per amendment 15)
reports/
├── phase3_xgboost_baseline.json   # generated by §5
├── phase3_walk_forward.json       # generated by §8 (HTML deferred to Phase 5)
└── conformal/                     # generated by §7
    └── {horizon}_{vol_bucket}.json
```

LOC budget: each file < 200 LOC per `CLAUDE.md` convention; CLIs
< 200 LOC each delegating to library code.

---

## 11. Implementation Phases (Sub-phases)

Effort revised per amendment 14 / architect review B.MINOR.7. v1's
9-day estimate was optimistic; sub-phases 3.4 / 3.5 / 3.7 needed
buffer.

| Sub-phase | Deliverable | Acceptance | Effort |
|---|---|---|---|
| 3.1 | `labels.py` + `build_labels.py` + leakage tests for **all 4 horizons** | leakage CI green; labels for {30m, 120m, 1d, 2d} generated for full universe + window | **1.5 d** (was 1; +0.5 for 4 horizons × per-horizon edge cases) |
| 3.2 | XGBoost baseline + LightGBM fallback + leakage gate; **4 baseline models** (one per horizon) | `phase3_xgboost_baseline.json` committed with per-horizon per-fold IC; `test_xgboost_features_leakage.py` passes | **1.5 d** (was 1; +0.5 for 4 horizons) |
| 3.3 | `cross_section.py` four-iterator sampler + weighted-random alternation + tests | leakage CI green; `iter_{30m,120m,1d,2d}` yield well-formed batches; weighted-random sampler hits cadence ratio in expectation | **1.5 d** (was 1; +0.5 for 4 iterators + alternation logic) |
| 3.4 | `posttrain.py` four-LoRA-banks + 3 heads × 4 horizons + magnitude-balance + hparam-compat + 1-batch overfit + smoke | smoke on CPU completes; train_loss drops monotonically across 100 steps for **all 4 horizons**; hparam-compat refuses on synthetic mismatch | **3.5 d** (was 3; +0.5 for 12 head instances + 4 LoRA banks via `peft` named adapters) |
| 3.5 | Single-fold full Phase-3 train (LoRA + random-init ablation) on H200 | LoRA + heads converge for **all 4 horizons**; random-init full-fine-tune baseline runs to completion (3 seeds); basic rank-IC table printed | **1.5 d** (incl. H200 wall-clock + debug roundtrip) |
| 3.6 | `conformal.py` (CQR + vanilla pickers) + tests; per-horizon × per-bucket cells | per-cell picker logic correct; in-sample sanity passes; out-of-sample test on synthetic data passes | 0.5 d |
| 3.7 | `walk_forward.py` + 4-fold sweep on H200 (SSL + random-init × 4 folds × **4 horizons of eval**) | `phase3_walk_forward.json` populated; per-horizon tri-criterion + ≥2-of-4 decision rule emits pass/fail | **3 d** (was 2.5; +0.5 for evaluating 4 horizons per fold + plotting) |
| 3.8 | Acceptance gate run + reviewer signoff | acceptance §1 (≥2-of-4 + §1.3 + §1.4 ≥2-of-4 random-init) explicitly evaluated; decision tree per §1 failure-mode budget invoked if needed | 0.5 d |

**Total**: ~12.5 days of focused work, ~60–90 H200-hours
(post-amendment 16 — 4 horizons compounds the random-init compute).

**Compute footnote.** §1.4 random-init × 3 seeds × 4 folds = 12 H200
runs at ~7h each (4 horizons trained per run, ~1h each) = ~84 hours
on top of the SSL 4-fold sweep (~28 hours). If GPU budget is tight,
the failure-mode-budget option (§C.3 of the architect review) is to
drop random-init to 1 seed (lose statistical robustness on the §1.4
gate; gain 56 hours).

---

## 12. Risks and Mitigations

In priority order:

1. **No edge over XGBoost at 30m (signal-limit confirmation).** This is
   the architect-review §5 hypothesis #5 worst case. Mitigation:
   acceptance §1's failure-mode budget (proceed to Phase 4 OB; drop
   swing strategy from v1). Also, the random-init ablation §1.4 tells
   us *whether the SSL helped at all* — even a small positive delta is
   informative for Phase 4 (saying "the backbone added ε; OB
   conditioning needs to add 10ε").

2. **Conformal coverage misses calibration on val.** Mitigation: the
   coverage gap usually comes from heteroscedasticity that the vol-
   bucket stratification didn't capture. First fallback: increase
   K=5 → K=10 buckets. Second: switch from CQR to a different
   nonconformity score (raw residual, signed residual). Third: ship
   without conformal, sizing falls back to vol-targeted (per §1.3
   failure-mode budget).

3. **Cross-section sampler is too slow** (full N=205 per step on every
   step). Mitigation: `max_symbols_per_step` sub-sampling at training,
   full N at eval. Profiling first; if iter speed is < 1 cross-
   section/sec we redesign. Worst case fallback: per-`(sym, t)`
   sampling like Phase 2, with the rank head computed only when
   batch size ≥ a quorum and skipped otherwise.

4. **LoRA-only fine-tuning underperforms full fine-tune.** Mitigation:
   if LoRA val rank-IC is meaningfully below random-init or just
   trivially above XGBoost, fall back to full fine-tune of the upper-2
   backbone layers + heads. This is the DESIGN §3.3 default plan that
   §6.7 amended away. Switching back is a 1-line config change.

5. **Quantile head dominates / drowns out rank head.** The pinball loss
   on log-returns is on a different scale than ListMLE on rank logits.
   Mitigation: monitor per-head loss in W&B; rebalance `w_*`.

6. **Per-horizon gradient noise differs across {30m, 120m, 1d, 2d}.**
   Longer-horizon labels (1d, 2d) are higher-variance per-sample than
   30m/120m. Post-amendment 16, each horizon trains under its own
   LoRA bank at its own natural cadence (no cross-horizon weight
   coupling), so per-horizon noise affects only its own bank.
   Mitigation: monitor `grad_norm_lora_{H}` and `lora_weight_norm_{H}`
   per-horizon in W&B; if one bank is destabilizing, reduce
   `cadence_weights[H]` (downsample that horizon's training steps)
   or apply per-horizon `gradient_clip_val`.

7. **Universe shifts during val window** (a symbol delisted between
   train_end and val_end). Mitigation: the `valid_label` mask catches
   this at label time; the cross-section sampler drops the row. We
   note any such occurrences in the tearsheet caveat panel.

8. **The H200 box becomes unavailable.** Mitigation: posttrain's
   max_steps=30K is small enough to fit on a smaller GPU
   (RTX-4090 24GB) with reduced batch size; the LoRA+heads forward
   memory is ~4 GB. Fallback to local training if needed.

---

## 13. Open Questions — answered by architect review

The v1 of this doc had 10 open questions. The architect review
(`reports/reviews/phase_3_architect_review.md` §A) answered all 10
explicitly; the answers are folded into this doc as the
post-architect-review amendments callout at the top + the inline
edits per section. Cross-reference table:

| Q | Topic | Answer location |
|---|---|---|
| 1 | LoRA-only vs partial unfreeze | §6.7 + amendment 12 (LoRA-only default; one-flag fallback to partial unfreeze if random-init beats LoRA by ≥0.002) |
| 2 | Quantile + rank head simultaneously | §6.1 + amendment 4 (keep both; drop direction; rank head pays for itself) |
| 3 | Per-horizon multi-task vs separate model | §6.3 + amendment 5 (shared FORECAST + two LoRA banks per horizon — best of both) |
| 4 | Aligning 30m and 5d on same `t` | §4.3 + amendment 2 (DROP — separate streams alternating at 75:1) |
| 5 | CQR vs simpler conformal | §7.1 / §7.2 + amendment 9 (run BOTH, pick per cell) |
| 6 | Pull cross-sectional contrastive into Phase 3 | NO — architect review §A.6 disagrees with predecessor review's recommendation. Rank head IS the supervised analogue. Defer (or kill) for v1. |
| 7 | 4 walk-forward folds enough | YES — but tri-criterion decision rule, not "mean of 4". §1 + §8.1 + amendment 8. |
| 8 | Survivorship adjustment via fixed κ | DROP — amendment 6 + §9 rewrite (raw IC + audit panel; the IC-delta is unbiased) |
| 9 | Cap on 30m at 6 bars vs session-time-conditional | OBSOLETED by amendment 16 — horizon set expanded to {30m, 120m, 1d, 2d}, which spans the relevant session-time bands without per-`t` conditioning. Strategy layer in Phase 5 still chooses *which* `t`s to act on per regime. |
| 10 | Pretrain → posttrain hparam compat check | YES, refuse on mismatch — amendment 7 + new §6.10 |

---

## 14. What "approved" means for this doc

**Status (2026-05-10):** Architect reviewer issued **Approve with
Major edits** (`reports/reviews/phase_3_architect_review.md`). All 3
Major findings + all 4 strongly-recommended changes + all 8 Minor
findings + all 4 Nit findings have been incorporated as amendments
1–15 in the post-review callout at the top of this doc + the
corresponding inline edits in §1 / §2 / §3 / §4 / §5 / §6 / §7 / §8 /
§9 / §10 / §11. The doc is ready for implementation.

Implementation will start with **Phase 3.1 (labels)** per §11. The
first PR will land `labels.py`, `build_labels.py`, the leakage tests,
and a smoke run printing rank distributions to confirm well-formed
labels — before any modeling work begins.

If a finding requires going back to Phase 2 (e.g., "you need a
different SSL checkpoint to make this work"), Phase 3 stalls until
Phase 2 produces it — same gating discipline DESIGN §7 applies
between phases. The §1 failure-mode budget covers this case
(criterion 4 fail → revisit Phase 2).
