# Phase 2 Strategic / Architect Review — daity_v3

**Reviewer:** Independent ML architect (strategic review, not code-correctness)
**Date:** 2026-05-10
**Scope:** Architecture choices, SSL objective design, the empirical
v1 → v2 → v2_long → v3_lr → v3_features experimental chain, the forecast-loss
plateau, phase-split / downstream readiness, and budget/opportunity cost.

This review operates one level above
`reports/reviews/phase_{1,2}_review.md` (those are line-level code-correctness
audits and have done their job — 392 tests, lint-clean Phase-2 modules,
provenance triple, EMA, ModelCheckpoint all landed). My job here is to ask
whether the strategy that those correct pieces of code are *implementing* is
the right strategy. The honest answer is: **the foundation-model layer is
mostly the right shape, but the central technical bet of Phase 2 — that an
SSL forecast head will produce a meaningfully predictive 8-hour-ahead
representation on NSE 5m bars — is the highest-risk part of the project, and
the empirical evidence so far is consistent with that bet not paying off.**

The strategic move now is not "iterate harder on Phase 2"; it is to
**get to Phase 3 with a known-decent SSL checkpoint and let the supervised
quantile head tell us whether Phase 2 actually learned anything useful**.
Details below.

---

## 1. Backbone family — patch transformer is fine, but it isn't the leverage

**Verdict:** PatchTST/Chronos-style channel-independent multi-resolution patch
transformer is a defensible choice for Phase 2 of *this* project; it is not
where the marginal bet should be placed.

**Reasoning.** PatchTST, MOIRAI, Chronos, TimesFM, Lag-Llama: the Time-FM
literature has converged on patch transformers because they (a) handle
arbitrary numerical channel counts, (b) play nicely with channel dropout
(needed for Phase-4 OB conditioning), (c) admit RoPE/relative-position
attention that helps cross-resolution, and (d) the published scaling laws
exist mostly for this family. There are alternatives — S4/Mamba state-space,
TimeMixer, iTransformer (channel-as-token), retrieval-augmented (e.g.
Time-LLM/TEMPO) — but none has dominantly outperformed PatchTST on the kind
of multi-resolution OHLCV regression we care about, and the "best in class
for finance" papers (Kronos, FinPT) all use patch transformers.

The architectural choice that *does* matter, and is currently underweighted,
is **how the FORECAST head is wired**. The current head reads a SINGLE
`d_model=480` vector (the FORECAST token's hidden) through a 2-layer MLP and
emits 6 patches × 5 channels × 16 bars = 480 numbers (or 1,728 at 18-channel).
A single-vector → multi-step direct head is the same shape MOIRAI calls
"single-channel forecast head", and it is known to underperform a
**cross-attention forecast head** that lets each future-bar query attend over
the encoder's full token sequence (PatchTST 2024, MOIRAI-Base, TimesFM, and
Chronos all use cross-attention on the decoder side). Backbone family is not
the leverage; **head architecture is the leverage**, and finding 3 below
quantifies the cost.

The 36M-param size is also too small for serious Time-FM SSL: Chronos-Base is
200M, MOIRAI-Base is 14–311M, TimesFM is 200M, Lag-Llama is 24M (and it is the
weakest of the four on the benchmarks). 36M sits in the awkward middle —
big enough to overfit our 11.5K trading days × 205 symbols ≈ 2.4M
(symbol, day) pairs, small enough that the forecast head's representation
ceiling is plausibly under-capacity for 8-hour ahead prediction.

**Recommendation (ranked by leverage):**

1. **Replace the FORECAST MLP head with a 2-layer cross-attention head**
   (queries: H learnable forecast-position tokens; keys/values: the full
   encoder output `out[:, 1:]`). This is the single change most likely to
   move val_forecast — `daity/models/heads.py:145-198` (`MultiHorizonForecastHead`)
   should be the first thing to swap. Cost: ~1-2 days; head adds maybe
   500K-1M params; the rest of the system is untouched. **High leverage.**
2. **Hold off on scaling to 100M+ params** until the head architecture is
   the right one. Scaling a poorly-shaped head is what we've already done
   twice (v1 11M → v2 36M, no improvement); a third doubling is unlikely
   to break the pattern.
3. **Defer the alternative-architecture conversation** (Mamba, mixers,
   retrieval) to a Phase 7 / next-design-cycle decision. They are not the
   bottleneck today.

---

## 2. The multi-resolution sequence layout — sound, but not where the value is

**Verdict:** Concatenating per-scale patches plus a FORECAST token is a
reasonable layout; the multi-resolution part is mildly under-motivated for
the downstream task.

**Reasoning.** The current layout is `[FORECAST, 5m_patches, 15m_patches,
60m_patches, day_patches]` — 81 patch tokens + 1 FORECAST = 82 tokens, all
concatenated, with a per-scale resolution embedding. This is the simplest
honest implementation of multi-resolution and works fine.

However, the *value* of multi-resolution depends on whether the forecast
target genuinely needs cross-scale context. For an 8-hour-ahead 5m forecast,
day bars are nearly useless (1 day bar covers the whole forecast horizon —
it's a label, not context); the 60m bars are weakly useful; the 15m and 5m
bars carry essentially all the relevant information. The cost of paying for
day patches is that the model spends some capacity learning to ignore them
(or worse, learns spurious patterns from 7 daily tokens × 18 channels × 16
bars = 2,016 daily features per sample). Published patch-transformers that
explicitly do multi-resolution (Time-MoE, MOIRAI's "any frequency" mode)
either drop the irrelevant scales or use frequency-aware attention masks.

Separately, the **`day` candle convention** — `ts` = close at 15:30 IST per
DESIGN amendment 7 — interacts with multi-resolution in a subtle way: the
last day-token's close is *temporally past* the most recent intraday tokens
(intraday `ts` = open). Whether the encoder learns this via positional
encoding alone is unclear; a publication-quality treatment would use
explicit time-of-bar embeddings rather than relying on RoPE-on-token-index
to encode physical time.

**Recommendation:**

1. **Run a cheap ablation: drop `day` from `cfg.scales`.** If val_forecast
   doesn't move, you've removed 7 patch tokens × 18 channels of confounding
   input for free. This is a 30-line config change and you can run it
   alongside v3_features. **Cheap, informative.**
2. **Defer "frequency-aware attention" / relative-time embeddings** to
   Phase 4. They are publishable improvements but not Phase-2-blocking.
3. **Document why the FORECAST token is at position 0 and not the
   trailing position.** Right now it reads from a non-contextualized
   parameter and the encoder pushes information *into* it via attention.
   That's a defensible pattern (matches BERT's CLS) but it's worth pinning
   the choice with one sentence in DESIGN — moved to position-last, the
   FORECAST token would have RoPE position == "end of input", which is
   arguably the right inductive bias for a forecast token. Not
   blocking. Worth a 15-minute sweep when head architecture is in flight.

---

## 3. SSL objective design — recon + NTP saturate cleanly; forecast does the wrong thing; contrastive is honestly off

**Verdict:** The recon + NTP pair is doing useful work; the multi-horizon
forecast objective is shaped wrong (single-vector MLP head over 480-element
target); contrastive is correctly disabled. The combination as a whole has
two real problems beyond the disabled contrastive.

**Reasoning.**

**The recon + NTP free-lunch problem (real but not fatal).** With `patch_len=16`
and `stride=8`, two adjacent patches overlap on 8 bars. So in the NTP head,
50% of patch t+1 is *literally* present in patch t — the model can copy bars
8-15 of patch t into bars 0-7 of patch t+1 and get the first half right for
zero learning cost. This is why the v2_long NTP loss saturates at ~0.20 in 2K
steps (`runs/h200_v2_long/lightning_logs/version_0/metrics.csv`): half the
target is trivially reproducible, and the other half is hard. Cross-scale
makes this worse: 15m, 60m, and day bars are aggregations of 5m bars, so the
model sees the same information at four different granularities.

This isn't fatal — it's structurally true of any patch transformer with
overlap, and PatchTST / MOIRAI ship the same setup — but it means **NTP loss
saturating fast doesn't mean the model has learned anything useful for
forecasting**. The empirical pattern across all five runs (NTP plateaus
quickly, recon plateaus quickly, forecast plateaus immediately) is exactly
what this account predicts.

**The forecast head is single-bottleneck (the central problem).** Per
`daity/models/heads.py:145-198`, the forecast head is:

```
forecast_hidden ∈ ℝ^480  →  Linear(480, 960) GELU Linear(960, 1728)  →  reshape (6, 18, 16)
```

The encoder's full output is `(B, 82, 480)` ≈ 39,360 numbers per sample, and
the forecast head only sees 480 of them. Every relevant signal for predicting
the next 6 patches must squeeze through that 480-dim bottleneck — which is the
same encoder dimension that also has to encode "what kind of regime is this",
"which symbol is this", "what's the recent volatility", etc. A
**cross-attention head** would let each of the 6 forecast-position queries
attend independently over the full encoder output, getting 39,360 numbers per
position instead of 480 shared. This is the architectural move every
production Time-FM uses and is the second-highest-leverage change after
fixing the head architecture itself.

**Quantile / pinball loss should arrive in Phase 2, not Phase 3.** Currently
the forecast head emits MSE point predictions of bar values. MSE on 8h-ahead
log-returns is dominated by the noise floor — even a perfect forecaster gets
a tiny R². A pinball loss across `{0.1, 0.5, 0.9}` quantiles would be a much
more honest objective and would directly precede Phase 3's quantile head;
adding it now means the SSL task and the supervised task share an output
space. DESIGN §3.3 already specifies pinball for the supervised head — pulling
it forward to Phase 2 is not a contract violation, it's an alignment.

**Realized-vol prediction would be a cheap, well-shaped auxiliary.**
Predicting next-window realized vol from the FORECAST token is (a) a much
easier target than predicting bar values (vol is autocorrelated and partially
deterministic from recent bars), (b) directly useful for Phase 5's strategy
sizing layer, and (c) a known regularizer for return-prediction heads
(MOIRAI, TimesFM both include it). Adding it is a 50-line patch.

**Sign-of-return classification and cross-sectional rank prediction**:
both add real signal for the downstream task. Cross-sectional rank,
specifically, requires the cross-symbol same-date batch construction that
DESIGN §3.2 already describes and amendment 11 defers — this is the *one*
SSL objective whose rationale is genuinely tied to the downstream
strategy (DESIGN §4.3's strategies consume rankings). Folding it into
Phase 3 as part of post-train, rather than re-attempting in Phase 2 SSL,
is the right call given Phase 2's track record.

**Recommendation (ranked by leverage):**

1. **Cross-attention forecast head** (see §1.1) — single biggest lever for
   moving val_forecast. **High leverage.**
2. **Add a pinball-loss quantile head as the primary forecast objective**,
   with MSE point as auxiliary. The current MSE-only head is optimizing the
   wrong thing for a noisy 8h-ahead target. **Medium-high leverage.**
3. **Add a realized-vol head** with weight ~0.2. Cheap, well-shaped, and
   directly useful for Phase 5. **Medium leverage.**
4. **Don't re-enable contrastive in Phase 2.** The cross-symbol same-date
   InfoNCE belongs in Phase 3 where the batch-construction infrastructure
   actually exists. The current placeholder is correctly disabled
   (DESIGN amendment 11). Don't put it back until Phase 3.

---

## 4. The empirical experimental chain — disciplined but slow

**Verdict:** The change-one-variable-at-a-time discipline is correct in
principle but is yielding ~2-bit decisions per run, which is too slow for the
plateau problem we have.

**Reasoning.** The chain is:

| Run | Δ | Result | Bits learned |
|-----|---|--------|------|
| v1 (11M, 50K, broken val) | baseline | val_forecast 9.21 → 7.83 | very little — broken metric |
| v2 (36M, 50K, broken val) | 3.3× scale | val_forecast 9.04 → 8.02 | ~1 bit: scaling alone doesn't fix it |
| v2_long (36M, 30K of 150K, fixed val) | val fix + batch×8 + lr×2.8 | val_forecast 22.83 → 22.82 (flat) | 1 bit: capacity isn't the issue at this objective |
| v3_lr (36M, 8K of 150K, log-return targets) | target form | val_forecast plateau ≈ 4.4 (different units) | 0 bits: same plateau, different scale |
| v3_features (38M, running) | 18-channel input | TBD | 1 bit if it converges meaningfully below 4.4; 0 bits otherwise |

So in ~30 H200-hours we've moved from "the metric was broken" to "the metric
is real and the loss plateaus". That's progress, but it's three runs to
establish what one well-designed predict-the-mean baseline would have given
us in 30 minutes: **what is the val-forecast loss of a constant predictor
that just emits the empirical mean of the target distribution?** If v2_long
plateaus at 22.8 and a mean-predictor gets 22.5, the model is learning
nothing — and we'd know that before running v3_lr. If a mean-predictor gets
80 and we're at 22.8, the model is learning a lot and the plateau just means
we've hit a different ceiling. **You can't interpret a plateau without a
naive baseline. This is the most important diagnostic experiment that has
not been run.**

The "kill v2_long at 30K of 150K because it looked flat" call was
**defensible but risky**. The v2_long val_forecast trajectory across 16 val
checks (every 2K steps) varies between 22.69 and 22.96 — a 1.2% range that
is at the noise level of a 50-batch val set. Killing it was rational under
the prior "if it hasn't moved by 30K, it won't move by 150K"; the prior is
plausible but unverified. **A publication-quality treatment runs at least
one config to its full max_steps so we can plot loss vs steps on log-log and
fit an actual scaling exponent**. We don't have that data.

The discipline of changing one variable per run is correct; the **selection of
which variable to change** is questionable. v2 → v2_long changed the val
split (correctness fix, mandatory), batch size (8×), and LR (2.8×) — that's
three changes wearing one hat. v3_lr changed input form, target form, LR
(1.4e-3 → 7e-4 after the explosion), and grad clip (0 → 1.0) — that's four.
v3_features changes input channel count (5 → 18). Each of those bundles is
defensible individually; the **aggregate** is that there is no single run
in this chain that's a clean A/B isolation of one architectural variable.

**Recommendation (ranked by leverage):**

1. **Run a predict-the-mean (and predict-the-last-bar) baseline immediately**,
   on CPU if needed. Total cost: <1 hour. This is the missing reference
   that makes every other val_forecast number interpretable. **Highest
   leverage diagnostic.**
2. **Stop iterating Phase 2 hyperparameters until baseline numbers exist.**
   Every additional run without a baseline is a bit-decision in the dark.
3. **For the future-pivot decision (head architecture), do NOT change two
   variables at once.** If you change the head to cross-attention, hold the
   target form, the channel count, and the LR fixed at v3_lr's known-stable
   settings. One variable, one run.
4. **Adopt a "min run length to interpret" rule**: a run that hasn't
   completed 50K steps doesn't get to vote on plateaus. Kill v2_long-style
   early-stops only on instability, not flatness.

---

## 5. The forecast-plateau hypotheses — ranked

**Verdict:** The two highest-likelihood causes are **(2) architecture
bottleneck** and **(5) fundamental signal limit**, in that order. (1)
capacity, (3) input signal, (4) objective mismatch, and (6) training-budget
mismatch are real but secondary.

**Reasoning, ranked.**

**(2) Architecture bottleneck — single FORECAST token compressed to d_model
isn't expressive enough. Probability: 50%.** This is the hypothesis the
current architecture has not tested. A 480-dim vector being asked to encode
a full 8-hour predictive distribution for a stochastic asset return process
is shape-mismatched against the published Time-FM literature, all of which
uses cross-attention from forecast queries to encoder hidden states. **The
fact that NTP and recon — both of which read PER-PATCH hidden states, not a
single bottleneck — saturate cleanly while forecast plateaus is direct
evidence for this hypothesis.** The recon and NTP heads have access to
`(B, 81, 480) ≈ 39K dims of conditioning per sample`; the forecast head has
access to 480. Run the cross-attention head experiment (§3.1) before
concluding anything else.

**(5) Fundamental signal limit — 8h-ahead 5m NSE bars are mostly random
walk. Probability: 30%.** This is the worst-case version of the project's
central thesis. It's the right hypothesis to take seriously because: (a) the
same plateau holds across two target representations (raw price and
log-returns) and across 3.3× model scale; (b) Indian equity intraday
returns have a known low-Sharpe ceiling, with most of the alpha
concentrated in the first 30 min and last 60 min of the session, not in
random 8h windows; (c) no pure-OHLCV forecasting paper claims a meaningful
R² at 8h horizons on 5m bars without external information. **If the
cross-attention head doesn't move val_forecast meaningfully, this becomes
the leading hypothesis and it kills the strategic premise of Phase 2 as
designed.**

The mitigation is obvious and is in fact what the project is supposed to
do anyway: Phase 3 supervised heads, calibrated against rank-IC at 30-min
horizons (DESIGN §4.1), are how you get usable signal. **Phase 2's goal
should be to produce a backbone that is "not noise" — i.e., better than
random init on Phase 3's downstream head — and not necessarily a backbone
that beats a baseline at the auxiliary 8h forecast objective.** This is a
critical reframe.

**(1) Capacity bottleneck — too small. Probability: 10%.** The 11M → 36M
result already lower-bounds this: 3.3× more capacity moved val_forecast by
~2% (7.83 → 8.02 on the broken metric, where v2 was actually slightly
worse). A 10× scale to 360M is unlikely to break the plateau under the
current head architecture, and the project doesn't have the compute budget
to test that hypothesis at scale (one 360M run on 200K steps = 30+ H200-hours).

**(3) Input signal — needs microstructure features. Probability: 5%
incremental.** v3_features tests this and it might add a small signal, but
microstructure-from-OHLCV (the Phase 1 deferred features: Lee-Ready,
Roll's spread, Amihud) are themselves noisy proxies; the *real*
microstructure signal lives in the order book (Phase 4). Even if
v3_features moves val_forecast from 4.4 to 4.0, that's 9% improvement, not
the 50%+ that would constitute "the model is now meaningfully predictive".

**(4) Objective mismatch — should predict distributions/rank/sign.
Probability: ~5% as a Phase 2 fix; high probability as a Phase 3 fix.**
Pinball / quantile loss would change the loss values without necessarily
moving the underlying forecast skill. It's worth doing for alignment with
Phase 3 (§3.2) but isn't going to be the thing that "fixes" Phase 2.

**(6) Training-budget mismatch — 50-150K steps too few. Probability: ~5%
under current architecture; would rise to ~20% under the cross-attention
head.** Chronos is trained for >1M steps, but Chronos is also 200M params
on hundreds of thousands of time-series. At 36M params on 205 symbols, more
steps is unlikely to break the plateau because the gradient signal ceiling
is set by the architecture, not the optimization budget. (Same architecture,
50K vs 150K of v2_long: zero progress over 30K steps tested.)

**Recommendation:**

1. **Test (2) by switching to cross-attention head.** This is the single
   experiment that distinguishes architecture from signal. Two outcomes: it
   moves the plateau substantially (architecture wins, scale further) or it
   doesn't (signal wins, accept Phase 2's role as "build a non-noise
   backbone, not a predictive forecaster", and move to Phase 3).
2. **Run baseline (mean predictor and last-bar predictor) NOW** so (5) can
   be evaluated honestly.
3. **Be willing to declare Phase 2 "good enough" at val_forecast plateau ≈
   v3_lr's 4.4** if (1) cross-attention doesn't help and (2) Phase 3 shows
   the SSL backbone outperforms random init on rank-IC at 30-min horizons.
   Phase 2's job is not to be a great forecaster; its job is to produce a
   useful representation.

**The biggest strategic risk is doubling down on Phase 2 plateau-breaking
when the right answer is "the plateau is real, accept it, move to Phase 3
and let the supervised head answer the actual question".**

---

## 6. Phase split + downstream readiness — Phase 3 should start now, in parallel

**Verdict:** The 7-phase split is still sound; do not collapse Phase 2/3.
But Phase 3 work should start in parallel with v3_features completing.

**Reasoning.** The phase split is correct because Phase 2 (SSL) and Phase 3
(supervised) optimize different things and need different infra (Phase 3
needs cross-symbol batch construction, conformal calibration code,
walk-forward eval; Phase 2 doesn't). Collapsing them into "supervised
pretraining" loses the regime-aware sampling weight (DESIGN §3.2), the
universe-coverage benefit of unlabeled data, and the contrastive cross-
sectional objective DESIGN §3.2 actually wants to do.

But Phase 3 doesn't need a "perfect" Phase 2 checkpoint. The standard
recipe is: take *any* checkpoint that's better than random init, run Phase
3 against it, and use the rank-IC delta vs random init as the actual
quality metric for Phase 2. **The v3_lr-fixed checkpoint at val_forecast=4.4
is plenty for this; it's the best one we have, and it has the right target
form for Phase 3 (log-returns aligns with Phase 3's quantile head)**.

The opportunity cost of waiting for v3_features is: 8.5 hours of GPU + 1-2
days of analysis, during which Phase 3 work (XGBoost baseline, quantile head
implementation, conformal split, eval metrics) could be code-complete. Phase 3
acceptance per DESIGN §7 is "val rank-IC > XGBoost baseline at 30-min and
5-day horizons" — that's a genuinely independent bar that doesn't depend on
which Phase 2 checkpoint you start with.

**Recommendation (ranked by leverage):**

1. **Start Phase 3 NOW with the v3_lr-fixed checkpoint.** Don't wait for
   v3_features. The Phase 3 acceptance criterion (rank-IC > XGBoost) is
   the actual question we care about; let it answer. **High leverage —
   moves the whole project forward.**
2. **In parallel, run the cross-attention forecast head experiment**
   (one H200-day) and the naive baseline (one CPU-hour). These give
   us the architecture-vs-signal verdict that informs whether to swap
   in a better Phase 2 checkpoint later.
3. **Don't kill v3_features mid-run** — it's already running, the H200
   time is sunk, and the result is a useful data point. But do not block
   Phase 3 on it.
4. **Adopt a "Phase 2 done when Phase 3 beats XGBoost" rule** instead of
   "Phase 2 done when val_forecast plateau breaks". The latter has been
   moving for 30 H200-hours; the former is the actual goal.

---

## 7. Budget + opportunity cost — the marginal experiment value is declining

**Verdict:** Yes, the marginal value of additional Phase-2 hyperparameter
experiments is declining. Stop running them after v3_features completes.

**Reasoning.** The chain has:

- v1 → v2: ~1 H200-hour, learned 1 bit (capacity alone doesn't help)
- v2 → v2_long: ~5 H200-hours, learned 1 bit (val metric is real, plateau
  exists; broken metric was hiding 22.8 → 7.8 difference)
- v2_long → v3_lr: ~10 H200-hours (incl. the explosion + restart), learned
  ~1 bit (plateau is target-form-invariant, so it's structural)
- v3_lr → v3_features: 8.5 H200-hours, will learn at most 1 bit (input
  signal contributes either marginally or not at all)

Each additional iteration is constant cost (~1 day end-to-end including
SSH, wait, pull-back, analyze, re-config, re-launch) and yields decreasing
information per dollar. The **shape of the next experiment** (decision
rule embedded in v3_features's config block) is "if val_forecast clearly <
4.4, win; if ≈ 4.4, move on to Phase 3" — the decision is correctly framed
but the answer space is also half "go to Phase 3 anyway", which means the
experiment is partly burning compute to confirm a fait accompli.

The real opportunity cost is the **compounding delay on Phase 3-6**, where
the actual project value lives:

- Phase 3 supervised heads: this is where rank-IC, the quantile head, and
  conformal calibration land. It's where we find out if the model has any
  alpha at all.
- Phase 4 OB fine-tune: the OB sample is 13× larger than DESIGN sized for
  (per amendment 1) — this is the highest-edge, lowest-explored part of
  the project, and Phase 4 is gated on Phase 3.
- Phase 5 backtest: the friction model + capacity curves + tearsheet
  pipeline is the deliverable that determines whether the project
  succeeds at its DESIGN §1 success criterion ("Sharpe > 1.0 net at
  ₹10Cr AUM").

Every day spent iterating Phase 2 is a day not spent learning whether
Phase 3-5 deliver. Given the project's central uncertainty is "does this
model produce a working trading system" rather than "does this SSL
checkpoint converge slightly better", **opportunity cost dominates the
marginal-experiment cost**.

**Recommendation:**

1. **Cap Phase-2 hyperparameter experiments at v3_features.** Whatever
   it produces, the next move is Phase 3.
2. **Prioritize the cross-attention head experiment**, but as a "swap in
   later if it works" thing, not a Phase-3 blocker.
3. **Treat Phase 3's XGBoost-baseline comparison as the real Phase 2
   acceptance criterion.** Codify in STATUS.md.

---

## 8. Loss weights, channel ordering, training tricks — minor calibration

**Verdict:** Loss weights are fine; channel ordering is fine; a couple of
standard tricks are missing but none is load-bearing.

**Reasoning.**

**Loss weights** `(w_masked=1.0, w_ntp=0.5, w_contrastive=0, w_forecast=0.5)`:
the relative magnitudes match DESIGN §3.2's initial weights and are
defensible. The forecast weight at 0.5 is arguably *too low* given that
the forecast objective is the only one whose representation Phase 3 will
actually consume — `daity/training/pretrain.py:131-135`. Bumping
`w_forecast` to 1.0 (matching `w_masked`) would put the FORECAST token's
training pressure on equal footing with patch reconstruction, and the
loss-curve evidence (`val_loss_recon` ≈ 0.12 vs `val_loss_forecast` ≈ 4-22
across runs) shows the recon term dominates the gradient at current weights.

**Channel ordering** `(open, high, low, close, volume)` for the 5-channel
case, then `(...5 raw, 4 ret_lags, 4 wicks, 1 vol_z, 4 time_covariates)`
for the 18-channel case — the ordering is fine but the **18-channel
indexing is positional, not named**, and any future channel insertion will
silently break downstream consumers. `daity/data/feature_precompute.py:8-30`
hardcodes the order. A `CHANNEL_ORDER_18` named tuple + assertion in the
dataset constructor would be cheap insurance. Not blocking.

**Standard tricks present:** RevIN ✓, mixed precision (bf16-mixed) ✓,
gradient clipping (added in v3_lr) ✓, EMA (Phase 2.1 patch) ✓,
LR warmup + cosine ✓, ModelCheckpoint top-k ✓, regime-aware sampling ✓,
provenance triple ✓.

**Standard tricks missing:**

1. **No FlashAttention specifics check.** `F.scaled_dot_product_attention`
   with bf16-mixed should pick FlashAttention-2 on H200, but there's no
   assertion; on H200 it should be ~2-3× faster than the math fallback.
   Current throughput (~10 it/s on v1, ~5.5 on v2_long) is reasonable but
   not verified against FA2's expected number.
2. **No gradient checkpointing** despite the dual backbone forward pass
   (the Phase-2 review's finding 6 — deferred to Phase 2.2). On a 36M model
   at batch 512, memory is comfortable per the v2_long config comments
   (11.7 GB / 143 GB). On a 100M+ scaled model, this would matter.
3. **No EMA decay sweep.** 0.999 is the default; for 50K-step SSL,
   a higher decay (0.9995 or 0.9999) is sometimes meaningfully better
   per the Time-FM literature. This is a hyperparameter chosen by analogy.
4. **The contrastive temperature 0.07 and proj_dim 128** were chosen by
   analogy to SimCLR and never revisited; with the head disabled this is
   moot, but flag it for Phase 3 when contrastive comes back.
5. **No `torch.compile`** — Lightning + bf16-mixed + SDPA usually composes
   with compile cleanly, and 1.5-2× wall-clock speedup is plausible. Not
   tried. Worth one config flag.

**Recommendation:**

1. **Bump `w_forecast` to 1.0** in the next config. The downstream consumer
   is the FORECAST token; weight it accordingly. Cheap experiment.
2. **Verify FlashAttention-2 is being used on H200** with a one-line
   `torch.backends.cuda.flash_sdp_enabled()` check or a kernel profile.
3. **Add a named `CHANNEL_ORDER_18`** and assert it in the dataset
   constructor. 5-line change; permanent insurance.
4. **Defer compile/EMA-sweep/gradient-checkpoint experiments to Phase 4**
   (when scale matters more).

---

## 9. RevIN affine OFF — DESIGN amendment 12 was correct, and it's an instance of a class

**Verdict:** Amendment 12 (`revin_affine=False`) is correct. The pattern it
exposes — "self-supervised loss on a target the model also normalizes can
collapse" — is worth flagging as a class.

**Reasoning.** Amendment 12 caught a textbook SSL failure: gamma → 0 with
target on the RevIN output drives MSE → 0 trivially. The fix is right and
the regression test is real (`tests/unit/test_pretrain_module.py`). Two
broader observations:

1. The same class of issue applies anywhere the model can rescale its own
   target: any future affine layer between the encoder and the
   reconstruction target needs the same scrutiny. The `MaskedReconstructionHead`
   itself produces values that are compared against RevIN-normalized
   targets — that's fine because the head's output is not the target's
   normalizer. But if anyone introduces a "learnable per-channel target
   scale" anywhere in the forecast/recon pipeline, it has the same
   collapse pathology.
2. A canary metric should track this. The amendment caught it via "gamma
   collapsed to ~0.04 of init"; a one-line metric `train_revin_gamma_norm`
   would surface this immediately on any future regression. Currently the
   only way to catch it is a 30K-step run that nukes itself.

**Recommendation:**

1. **Add a `_log_revin_health` step-level metric** that logs the L2 norm
   of every RevIN gamma parameter and asserts > 0.5 × init at run end.
   Cheap, permanent insurance. (`daity/training/pretrain.py` `training_step`,
   ~10 lines.)
2. **Document the class of issue in DESIGN's modeling section** — one
   paragraph along the lines of "any self-normalizing layer whose output
   is a loss target risks gamma → 0 collapse; default to non-affine
   normalization in SSL contexts and verify with a gamma-norm canary".

---

## 10. Survivorship bias is not addressed in Phase 2's training data

**Verdict:** Not Phase-2-blocking, but flagged for tearsheet honesty in
Phase 5.

**Reasoning.** DESIGN amendment 4 confirms the universe is the current-active
F&O 200-ish roster (delisted symbols absent). Phase 2's pretraining sees
only this surviving universe; the model learns "what survives" rather than
"what the population of NSE F&O underlyings looks like". For SSL on bar
dynamics, this is mostly OK — the bar dynamics of survivors are not
qualitatively different from delisteds for the regime they were both
trading in. Where it bites is Phase 5 backtests on swing horizons; DESIGN
acknowledges this and §2.3 mandates the survivorship-adjusted Sharpe
range.

For Phase 2 specifically, the only thing to flag is: if Phase 3 starts
showing positive rank-IC, **the swing-horizon (5d) IC is biased upward
from this**. The intraday (30m) IC is much less affected. The
recommendation in DESIGN §2.3 to quote "survivorship-adjusted Sharpe range"
should extend to "survivorship-adjusted rank-IC range" in the Phase 3
tearsheet — currently DESIGN §4.1 doesn't mention it for IC.

**Recommendation:** Phase 3-time fix; not Phase 2. Note it in the Phase 3
plan when it lands.

---

## Final verdict + roadmap

### Confidence in DESIGN §1's Sharpe > 1.0 net at ₹10Cr success criterion

**Confidence: 35-50%** that the project produces a working paper-trading
system with verifiable Sharpe > 1.0 net at ₹10Cr AUM by Phase 6. This range
reflects:

- **Upside (50% scenario):** Phase 3 supervised heads beat XGBoost at 30-min
  rank-IC; Phase 4 OB fine-tune adds the expected microstructure edge given
  the 6.5K symbol-day OB sample (13× DESIGN's sizing); the friction model and
  capacity curves are rigorous and the strategy survives them.
- **Downside (50% scenario):** Phase 2 plateau is signal-limited (hypothesis
  5); Phase 3 rank-IC is meaningfully positive but capacity-adjusted
  intraday Sharpe at ₹10Cr collapses below 1.0 due to NSE friction (this is
  the historical pattern for naive intraday cross-sectional on F&O 200).
  Long-only swing might still get there but is more survivorship-biased.

### Biggest risk to that outcome

**The forecast plateau is signal-limited, not architecture-limited.** If
8h-ahead 5m NSE bars are mostly random walk + noise, *no* SSL setup will
break the plateau, and Phase 3 will inherit a representation that is "not
much better than features-on-XGBoost". The project might still hit Sharpe
> 1.0 via Phase 4 OB conditioning + the friction-aware backtest, but the
margin will be much thinner than the foundation-model framing suggests.

### Single most important change to make NOW

**Replace the FORECAST token's MLP head with a cross-attention head, AND
start Phase 3 in parallel against the v3_lr-fixed checkpoint.** These are
two changes, but they share an answer: they together resolve the central
strategic question (is Phase 2's plateau architectural or fundamental?) at
the earliest possible time, in parallel rather than serial. The
cross-attention experiment costs ~1 H200-day; Phase 3 boilerplate costs
~3-4 days of CPU-side work and unblocks the rank-IC vs XGBoost comparison
that is the real Phase 2 acceptance criterion.

If forced to pick ONE: **start Phase 3.** Cross-attention is a Phase-2
optimization; Phase 3 is the answer to "does this project work".

### What to cut from the roadmap if budget is tight

In priority order of "cut first":

1. **Cut the contrastive head entirely from v1.** It was disabled in
   amendment 11 and is only useful with the cross-symbol same-date batch
   construction. If Phase 3 lands without it and works, leave it cut.
2. **Cut `day` scale from pretraining inputs.** Empirically test it
   first; if val_forecast doesn't move, save 7 daily patch tokens × 18
   channels of input from every batch.
3. **Cut Phase 2 hyperparameter iteration after v3_features.** Move on.
4. **Consider cutting the swing strategy** if Phase 4 OB fine-tune (the
   intraday-only edge) is showing meaningful capacity-adjusted Sharpe. The
   swing strategy is the most survivorship-bias-exposed part of the
   roadmap and has the smallest known edge in NSE; intraday with OB
   conditioning is the project's natural unique advantage.
5. **Cut any further model-scale experiments** beyond 36M until the
   architecture (cross-attention head) and the downstream signal (Phase 3
   rank-IC) are both confirmed. Scaling without architecture is a known
   negative.

What to **keep** under all budget constraints: the friction model, the
capacity curves, the conformal calibration, the with/without-OB ablation
gate, and the leakage CI tests. These are the parts of the project that
make a Sharpe number honest, and they are non-negotiable.
