# Phase 3 Design — Architect Review

**Reviewer:** Independent ML architect
**Date:** 2026-05-10
**Scope:** Strategic + design correctness review of `PHASE3.md` (v1 draft) — the supervised post-training phase that turns the v3_xattn SSL checkpoint into a calibrated cross-sectional ranker. This is the project's first real test of "does this system have alpha".
**Disposition:** **Approve with Major edits.** Three Major findings must be addressed in a post-review amendments callout before any Phase-3 code lands; everything else is Minor / Nit and may be merged inline. No Critical findings — the design is sound in shape, but a few decisions are likely to bite if they ship as written.

This review operates one level above `phase_2_review.md` (line-level code correctness). My job is to challenge whether the strategy, the architectural choices, the loss design, the eval methodology, and the phase split are the right ones — and what to change before implementation begins.

The short version: PHASE3.md is a competent formalization of the predecessor architect review's §6 recommendations. The acceptance gate is the right one. The four-headed multi-task design is mostly the right shape. The two changes I am flagging Major are: (a) acceptance §1.3 conformal coverage measured on the calibration set is a circularity that needs an honest second pass, (b) the "same `t` for 30m and 5d with 1/75 downweight" alignment in §4.3 has worse gradient pathologies than the design admits and a separate-stream alternative is cleaner. The third Major is about the random-init ablation gate's sample efficiency — the way it's currently written, it can pass while saying nothing.

---

## Section A — Open Questions Answered

The author flagged 10 questions in §13 as genuinely uncertain and asked for a call. Here are the calls.

### A.1 LoRA-only vs partial unfreeze (§6.7)

**Recommendation: Keep LoRA-only as the default; add a single-flag fallback to "unfreeze upper-2 backbone layers + heads, no LoRA" if LoRA val rank-IC at fold F-9 is within 0.002 of random-init.**

**Reasoning.** Three pressures push against full or partial fine-tune at Phase 3:

1. The Phase-2 checkpoint is `val_forecast=4.7956` — i.e., it converged to the same plateau as v3_lr. The architect-recommended cross-attention head fix did not break the plateau. There is moderate evidence the backbone is *barely* better than random init for the forecast objective. If we full-fine-tune at Phase 3, we are training on a labeled set of ~4 months × 75 t/day × 205 symbols ≈ 4.6M (sym, t) labels for the 30m head and ~84 trading days × 205 ≈ 17K labels for the 5d head. The 5d head will overfit a 36M backbone trivially.
2. LoRA at `r=8, alpha=16` on 4 attention projections × 12 layers ≈ 1.4M trainable params is the right capacity for ~17K 5d labels.
3. Promoting LoRA to Phase 3 prepares the codepath for Phase 4 (where LoRA is mandatory). One implementation, two phases.

The author's worry — "am I cutting a corner that bites at signal time" — is real but second-order. The corner-cut would be visible immediately: if LoRA val rank-IC ≈ random-init val rank-IC, swap to partial-unfreeze. The risk is sample-efficient (one extra fold's worth of compute), not strategic.

**Suggested edit.** Add to §6.7: "If random-init ablation (§1.4) outperforms LoRA-only by ≥0.002 on val rank-IC at 30m, switch to `unfreeze_top_n_layers=2` with `lr_backbone=lr_lora * 0.1`, no LoRA. This is a `posttrain_v1.yaml` flag."

### A.2 Quantile + rank head simultaneously (§6.1, §6.2)

**Recommendation: Keep both heads. Drop the direction head from v1.**

**Reasoning.** The author asked the right question: is the added complexity worth the inductive-bias gain?

- **Quantile head.** Pinball loss across `{0.1, 0.25, 0.5, 0.75, 0.9}` is the right shape for Phase 5's sizing layer (which needs a distribution) and for the conformal calibration (CQR explicitly needs the q_0.1 and q_0.9 outputs). This is non-negotiable — drop it and you lose §1.3.
- **Rank head.** The cross-sectional rank score is what the strategy *trades on*. The author's reasoning — "the median quantile is pulled by pinball toward the center of the residual distribution, not toward a discriminative ordering" — is correct. ListMLE's gradient is signed by within-batch order, which is the cheaper objective for a rank-discriminative head. The cost is one extra `Linear(480, 1)` per horizon. **Cheap and well-shaped.**
- **Direction head.** This is the one that doesn't pay for itself. The author admits in §6.1 it's "auxiliary, mainly a regularizer". Three-class classification on a noisy 8h-ahead target with class-balanced CE adds gradient pressure on the FORECAST token toward sign discrimination — which is what the rank head is *already* doing implicitly (the within-batch ordering carries sign info). The direction head adds two more loss-weight knobs to tune (`w_dir_30m`, `w_dir_5d`) without a cleaner downstream consumer than rank+quantile.

**Suggested edit.** Drop `w_dir = 0.3` from the §6.2 loss equation; remove the direction head from §6.1; remove the class-balanced CE test from §6.9. Saves about 0.5 days of implementation and one fewer hyperparameter.

If the author wants to keep direction "as a regularizer", add it back behind a config flag with default off — re-enable in Phase 5 when sizing-by-direction-confidence may matter.

### A.3 Per-horizon multi-task vs separate model per horizon (§6.1)

**Recommendation: Per-horizon multi-task on shared FORECAST hidden, but train two SEPARATE LoRA adapter sets — one for 30m, one for 5d.**

**Reasoning.** The current design ties 30m and 5d through the *same* LoRA adapters on the *same* backbone. The 5d objective is much noisier than 30m (5d log-returns are ~5× more variance than 30m log-returns), so the 5d gradient through LoRA will dominate the LoRA's parameter trajectory if the loss weights aren't perfectly tuned — and the author is candid in risk #5/#6 that this is a real concern. The 0.2× factor compensates for sample-count, not gradient variance.

The cleaner architectural move is: same FORECAST token, same backbone (frozen), but **two LoRA adapter banks** (`peft` supports multi-adapter natively via `set_adapter('30m')` / `set_adapter('5d')`). Forward passes alternate which adapter is active. Each adapter is ~1.4M params; total 2.8M is still under 3% of backbone.

The author's worry that this "doubles the trainable param count" is fine — it's still well under 3M, and it removes the gradient-interference problem entirely. It also makes the 30m model independently shippable for Phase 4 (which only consumes the intraday head per DESIGN §3.4) without 5d gradients having ever touched it.

**Suggested edit.** §6.3 LoRA section: "Two LoRA adapter banks via `peft`'s named-adapter API: `lora_30m` and `lora_5d`. Forward passes use the horizon-appropriate adapter. Heads share FORECAST hidden but read post-adapter."

If the author rejects this for simplicity reasons, the fallback is: keep one LoRA bank, but **monitor adapter weight-norm trajectory and gradient-norm-per-loss-term in W&B**, and fall back to two adapters if 5d swamps 30m.

### A.4 Aligning 30m and 5d on the same `t` (§4.3)

**Recommendation: Separate batch streams for 30m and 5d, alternating gradient steps. The 1/75 downweight is an unforced error.**

This is one of the Major findings (B.2 below). Short version here:

The "same `t`" alignment forces the 5d label to be repeated 75× per trading day (one per intraday `t ∈ {09:15, …, 15:25}`). The 0.2× per-term weight + 1/75 effective per-day weight is meant to balance per-day gradient magnitudes — but **the 5d label at `t = 09:20` and `t = 09:25` differs only in the input window (the 5d label itself is identical because `floor_to_day(t)` is the same)**. So the 5d head sees 75 nearly-identical labels per day, with only the input-context distinguishing them. This:

1. Inflates the effective sample size of 5d gradients in a way the 1/75 doesn't fully correct, because 75 noisy gradients with same target ≠ 1 gradient with that target.
2. Encourages the 5d head to learn "what intraday context is consistent with this day's 5d outcome" — which is regression toward the mean conditional on intraday-features, not 5d signal.
3. Makes conformal calibration of 5d residuals dependent on 75 highly-correlated samples per day — calibration set size is ~75× smaller in effective independent samples than §7.3 claims.

The cleaner alternative — separate batch streams, alternate gradient steps — is exactly the standard PEFT recipe for multi-task. It costs a few extra lines in `posttrain.py` to interleave two `iter_cross_sections()` calls, and it kills the 1/75 nonsense. Each horizon gets its own training distribution at its own natural cadence.

**Suggested edit.** §4.3 rewritten: "We maintain **two iterators**: `iter_30m` (yields 75 cross-sections per trading day) and `iter_5d` (yields 1 per trading day). The training loop alternates `intraday_steps_per_5d_step` 30m steps for every 5d step (default ratio 75:1, matching natural cadence). The gradient accumulation buffer holds steps from both streams; effective batch matches DESIGN §3.3." Drop the 0.2× factor in §6.2; both heads see their natural label cadence.

### A.5 CQR vs simpler conformal (§7.2)

**Recommendation: Implement BOTH (CQR + vanilla split-conformal on raw residuals). At calibration time, pick whichever gives better empirical coverage on val per (horizon, vol_bucket) cell. This is cheap and the author already flagged it.**

**Reasoning.** The author's worry is correct: CQR's adaptivity is only better than vanilla if the quantile head is well-calibrated. On a noisy 8h-ahead target with a quantile head trained for ~30K steps on a possibly-uninformative backbone, this is not a sure thing. Vanilla split-conformal gives a constant `s*` per cell; CQR gives a vol-modulated one. The "right" choice is empirical.

Implementation cost: ~30 lines. Both methods share the calibration set; the only difference is the score function and the band-construction formula. Pick per cell based on empirical coverage closeness to `1-α`.

**Suggested edit.** §7.1 rewrite: "We compute both the CQR score (Romano-Patterson-Candès) and the absolute-residual score (vanilla split-conformal). At calibration time we select per `(horizon, vol_bucket)` cell whichever achieves empirical coverage closer to `1-α`. The choice is logged in the conformal JSON."

### A.6 Cross-sectional contrastive (DESIGN §3.2) — pull into Phase 3?

**Recommendation: No. Defer to Phase 5 or kill it from v1 entirely.**

**Reasoning.** The architect review §3 said "fold it into Phase 3". I now disagree with my predecessor's recommendation, on three grounds:

1. **Phase 3's acceptance gate is "beat XGBoost on rank-IC".** The contrastive head adds no labeled signal — it's still SSL. If it helps, it helps the *backbone representation*, which Phase 2 already had its chance at. Adding it in Phase 3 means re-running pretraining-style optimization with new infra, on top of supervised optimization. That's a Phase 2 redo masquerading as Phase 3 work.
2. **The cross-sectional rank head IS the supervised version of cross-sectional contrastive.** ListMLE on rank labels does what InfoNCE-on-co-moving-pairs is trying to do, except with actual return-direction labels instead of a heuristic threshold. If the rank head works, contrastive's contribution is redundant; if the rank head doesn't work, contrastive isn't going to save it.
3. **Phase 3 is already 9 days. Adding contrastive is +1-2 days of plumbing for a speculative gain.**

**Suggested edit.** Add a one-line rejection in §13.6: "Decision: contrastive deferred to a possible Phase 7 retrofit. Rank head is the supervised analogue and is the right v1 lever."

### A.7 Are the 4 most recent walk-forward folds enough (§8.1)?

**Recommendation: 4 folds is the right compute budget. But change the decision rule from "mean rank-IC over 4 folds > XGBoost mean" to "mean rank-IC over 4 folds > XGBoost mean by ≥0.005 AND at least 3 of 4 folds individually positive AND at least 3 of 4 folds individually beat XGBoost".**

**Reasoning.** The "mean of 4" rule has two pathologies:

- **Single-fold dominance.** Walk-forward folds in volatile periods (e.g., 2025-Q2 if there was a regime shift) have IC variance ~5× higher than calm periods. A single lucky fold can flip the mean while the underlying signal is unstable. Phase 5 deployment is not into "the average regime"; it's into the next regime, which is one fold's worth of variance.
- **No monotonicity check.** If the model wins F-9 by 0.02 and loses F-10/F-11/F-12 by 0.005 each, the mean still beats XGBoost — but the live performance will look like F-12, the worst.

Requiring "3 of 4 individually positive AND 3 of 4 individually beat XGBoost" plus the mean delta is cheap and gives a much more defensible "yes" decision. The math is the same Bonferroni-style intuition that gates §1.3's coverage check across vol buckets.

**Suggested edit.** §8.1 decision rule rewritten as above. §1.1 / §1.2 acceptance gates updated to reference the new tri-criterion.

### A.8 Survivorship adjustment via fixed κ (§9.2)

**Recommendation: Drop the multiplicative shrinkage. Report raw rank-IC + a textual caveat panel in the tearsheet that shows the fraction of universe that delisted/IPO'd within each fold's window. Don't pretend κ is a derivation when it's a heuristic.**

**Reasoning.** The fixed κ is two problems pretending to be one:

1. The `(0.02, 0.10)` numbers are not derived from anything. They are "industry rough estimates" — but the survivorship pattern in NSE F&O 200 over 2019-2025 is specific. The user's universe is the *currently active* roster as of 2026-05; that means symbols that became F&O-eligible *during* the window are also missing from the early-window training (since they weren't in F&O in 2019). Both directions of survivorship matter, and a single multiplicative κ doesn't capture the IPO direction.
2. Any single κ applied to rank-IC is mathematically dubious. Survivorship doesn't shrink rank-IC linearly; it changes the distribution of which symbols are in the cross-section, and the IC under a different cross-section is *not* `(1-κ)` × original IC. It might be more, less, or sign-flipped.

The honest move: report raw IC, plus a panel that quantifies the sample bias. The bias is: "in 2019-Q1, the universe had N=148 symbols; today it has 205. 57 names that are in our 2026 universe wouldn't have traded in 2019." That's the survivorship statement. The reader can decide how to discount.

The author's §9.2 already says "they're NOT applied to acceptance §1's gate (raw IC vs raw XGB IC) — both sides see the same survivorship distortion, so the *delta* is unbiased". This is correct and is the single most important sentence in §9. Lean into it: the delta-IC is the statistic of interest, and it's unbiased. Don't muddy the reporting with a heuristic that doesn't actually correct anything.

**Suggested edit.** §9.2 rewritten to drop the κ formula. §9 becomes a tearsheet honesty section that reports (a) survivorship audit table per fold, (b) raw IC and IC-delta vs baseline, (c) prose caveat. Saves time and is more honest.

### A.9 Cap on 30m horizon at 6 bars (§3.1)

**Recommendation: Keep uniform 30m horizon for v1. Add an experiment-tracking todo to test session-time-conditional horizons in Phase 5 / strategy layer.**

**Reasoning.** The architect review §5 mentioned "most of the alpha concentrated in the first 30 min and last 60 min of the session" — that's a strategy-layer observation about *when* to trade, not a forecasting-horizon observation. The forecasting model should output predictions at every `t`; the strategy layer decides which `t`s to act on.

A session-time-conditional horizon (e.g., 30m at 09:30, 60m at 14:30) would be a different model output schema for different `t`s, complicating the eval (you can't directly compare rank-IC across `t`s with different horizons), the conformal calibration (each horizon needs its own calibration set), and the strategy layer (which now consumes a heterogeneous prediction stream).

Keep v1 simple: uniform 30m, all `t`. Phase 5's strategy can choose to only consume `t ∈ {09:15-09:45, 14:30-15:25}` predictions if alpha is concentrated there.

**Suggested edit.** §3.1 unchanged. Add to §13.9: "Decision: uniform 30m for v1. Session-time-conditional horizons logged as a Phase 5 strategy-layer experiment."

### A.10 Pretrain → posttrain compatibility test

**Recommendation: Yes, add it. Behavior on mismatch: refuse with a clear error message.**

**Reasoning.** The hparams the v3_xattn checkpoint was trained with — `forecast_head_type`, `forecast_num_channels`, `revin_affine=False` (per amendment 12), `d_model=480`, `n_layers=12`, etc. — are encoded in the Lightning hparams that ride inside the checkpoint. Posttrain MUST match these for `state_dict` loading to work, AND for the encoded representation to be in the same coordinate frame the heads will train against.

The failure mode if you don't check: state_dict load silently succeeds with `strict=False`, the backbone's RevIN normalization is misaligned with what the FORECAST token expects, and you train the heads on a subtly broken representation. This will look like "Phase 3 underperforms" without a clear cause.

**The right behavior on mismatch is refuse**, not coerce or warn. Config drift between phases is a class of bug the project's `(git_sha, config_hash, as_of)` provenance triple is meant to catch — extending the check to "loaded checkpoint's hparams must match this run's backbone-hparams subset" is a 30-line addition.

**Suggested edit.** Add §6.10 (new): "Hparam compatibility. `posttrain_cli.py` extracts `backbone_hparams` from the checkpoint's stored hparams (`d_model`, `n_layers`, `n_heads`, `ffn_ratio`, `revin_affine`, `feature_channels`, `scales`, `window_bars`, `patch_len`, `patch_stride`) and asserts they match the live Hydra config. On mismatch: raise `IncompatibleCheckpointError` listing the deltas. No silent coercion. Test: `tests/unit/test_posttrain.py::test_hparam_mismatch_refuses`."

---

## Section B — Critical / Major / Minor / Nit Findings

No Critical findings. Three Major, eight Minor, four Nit. Full list below.

### B.MAJOR.1 — Acceptance §1.3 conformal coverage measured on the calibration set is circular

**Severity:** Major
**Location:** §1.3 + §7.5
**Issue.** The acceptance gate says "empirical coverage of the predicted 80% band on val is in `[0.75, 0.85]`, per horizon and per realized-vol bucket". §7.5 explicitly states "Empirical coverage on val (the calibration set itself, in-sample) is ≥ `1-α-ε` by construction (split conformal's marginal coverage guarantee)". So the "acceptance gate" is testing a property the algorithm guarantees by construction. **It cannot fail.** It's not a real gate — it's a smoke test that the conformal code didn't have a bug.

**Reasoning.** Split conformal's coverage guarantee is *marginal* over the calibration set. So checking coverage on the same set is checking "is the empirical quantile equal to the theoretical quantile of the same distribution it was estimated from", which is true by construction modulo finite-sample noise. The §1.3 gate at `[0.75, 0.85]` (10% wide window, ~5σ on a 500K-sample binomial) basically tests "is N > 0".

The acceptance gate the project actually wants is **out-of-sample coverage** — empirical coverage on the *test* fold, not the calibration (val) set. That's where calibration can fail (distribution shift between val and test, or per-bucket sample-size insufficiency exposing a bucket boundary). The author actually nodded to this in §7.5: "A stronger test, deferred to Phase 5: coverage on a held-out month out-of-sample." That deferral is the actual bug — Phase 3's acceptance has no honest calibration test.

**Why Major.** Three of the four acceptance gates are direct (§1.1, §1.2, §1.4). §1.3 is a no-op as written. Shipping with a vacuous gate means we approve Phase 3 without ever testing whether the conformal pipeline actually does what it claims. If §1.3 fails in Phase 5 we have to redo Phase 3 — the gate's job was to catch this.

**Recommendation.** Rewrite §1.3 to: "Conformal calibration within 5% of nominal on the **test fold** (not val): empirical coverage of the predicted 80% band on the held-out test fold of each walk-forward fold is in `[0.75, 0.85]` per horizon and per top-level vol bucket, averaged across the 4 folds. Per-bucket coverage is reported in the tearsheet but not gated (small per-cell test sample sizes make per-bucket gates unreliable)." This makes §1.3 the actual property the design wants. Keep §7.5's in-sample check as a sanity assertion, not an acceptance gate.

The 5% margin on top-level (not per-bucket) is right; per-bucket on a few hundred test samples will fail by chance.

### B.MAJOR.2 — The "same `t` for 30m and 5d with 1/75 downweight" in §4.3 has gradient pathologies the design doesn't acknowledge

**Severity:** Major
**Location:** §4.3 + §6.2
**Issue.** Already detailed in A.4 above. The 5d label is repeated 75× per trading day with only intraday context distinguishing the inputs; the 1/75 downweight does not correct for the gradient-correlation pathology this creates.

**Reasoning.** Three concrete pathologies:

1. **Effective independent samples.** §7.3 claims val has ~5M samples → ~500K per (horizon, vol_bucket) cell. For 5d, the actual independent-day count is `4 months × 21 days/month ≈ 84 days × 205 symbols ≈ 17K samples`. The other 75× factor is repeated correlated samples. Conformal's coverage guarantee is per-exchangeable-sample; with 75× correlation, the effective N for the 5d conformal calibration is 17K, not 1.27M. This is enough but barely.
2. **Loss-balance arithmetic is wrong.** The author writes "the 0.2 factor on 5d-horizon terms compensates for the 1/75 down-weighting from §4.3 (so per-day 5d gradient ≈ per-day 30m gradient)". That's true for *per-day total magnitude* of the weighted gradient, but **not** for *per-day variance*. Variance of a sum of 75 correlated noisy signals downweighted by 1/75 is `Var(single signal) × (1 + 74ρ)/75 ≈ ρ × Var` for moderate ρ — so the 5d gradient variance is ρ× the 30m's, not equal.
3. **Curriculum effect.** Across a single training epoch, the 5d head sees the same labels 75× before the day rolls forward. This is closer to "75 epochs of single-label data" than "1 epoch of 75-label data" for the 5d head. SGD on near-duplicate samples is known to converge to a different optimum than SGD on independent samples (it's effectively low noise in the gradient, more aggressive descent — overfitting accelerator).

**Why Major.** This is the core of the cross-section sampler's training contract. If it's wrong, the 5d head trains on a malformed target distribution, and the §1.2 5d-rank-IC gate will be unreliably correlated with deployment performance.

**Recommendation.** Implement the separate-stream alternative from A.4. Two iterators, alternating gradient steps at natural cadence (1 5d step per 75 30m steps). Drop the 0.2× and 1/75 factors entirely. The implementation cost is ~10 lines in the training loop; the conceptual win is large.

If the author rejects this: at minimum, fold the 5d label de-duplication into the sampler — emit 5d labels once per (sym, day) and zero the 5d gradient on intraday-`t`s where the day's 5d has already contributed to a prior step. But that's hacky; the clean fix is two streams.

### B.MAJOR.3 — Random-init ablation gate (§1.4) is under-specified

**Severity:** Major
**Location:** §1.4
**Issue.** The acceptance gate says "a model trained identically but with the SSL-pretrained backbone replaced by random weights must underperform". As written, this can pass while saying nothing useful — there's no margin specified, no significance test, no statement of "by how much".

**Reasoning.** Three failure modes:

1. **Margin under-specified.** "Must underperform" — by 0.001? 0.0001? Random-init can be 0.0001 worse and the gate passes, while the SSL backbone has effectively done nothing. The §1.1 / §1.2 gates have explicit ≥0.005 margins; §1.4 should too.
2. **Compute budget for random-init.** Training "identically" means same `max_steps=30K`, same batch, same LoRA. But random-init has *more* to learn than LoRA-from-pretrained (it's training the backbone from scratch via LoRA, which doesn't actually work — LoRA assumes a useful base). The honest random-init baseline is **full fine-tune of the random-init backbone**, not LoRA on it. Otherwise the comparison is "useful pretrain + LoRA vs useless pretrain + LoRA which has 1.4M trainable params and 36M frozen random params" — that's a stacked deck.
3. **Single-seed comparison.** Random init is by definition variance-prone. A single random-init run might be 0.01 worse or 0.005 better than SSL by chance. At minimum, run 3 random seeds and use the best for the comparison.

**Why Major.** The §1.4 gate is the project's diagnostic for "did Phase 2 actually do anything useful". If the gate is loose, we ship Phase 3 not knowing if the SSL backbone matters — which means Phase 4's cost-benefit (more SSL? more OB? more eval?) is being decided in the dark.

**Recommendation.** Rewrite §1.4 as: "Beats random-init at rank-IC by ≥0.005 (same margin as XGBoost gate) at both horizons. Random-init baseline: identical heads + full fine-tune of all backbone weights at `lr_backbone=3e-4` (no LoRA, since LoRA-on-random-init is degenerate). Run 3 random seeds; take the best as the comparison number. If LoRA-on-pretrained doesn't beat random-init-best by ≥0.005, the SSL representation is not contributing and Phase 2 needs revisiting. This is a hard gate, not advisory."

### B.MINOR.1 — XGBoost baseline could be stronger

**Severity:** Minor
**Location:** §5.2, §5.4
**Issue.** The 457-feature flattening is reasonable but is the *minimum* a competent XGBoost user would build. To be a serious bar:

- Add **interaction features** between top-decile importance pairs after a first-pass fit. XGBoost handles them implicitly but explicit ratio features (e.g., `vol_z_5m_last / vol_z_60m_last`, `ret_5_5m_mean / ret_60_60m_mean`) often add 1-2% IC on cross-sectional tasks.
- Use **LightGBM with `objective='rank_xendcg'`** as the rank head. XGBoost's rank objective is also fine but LightGBM's is more recent and sometimes outperforms.
- The current hyperparameter grid is sparse (`max_depth ∈ {4, 6, 8}`, etc.) — for 4 hours of CPU compute, a Bayesian optimization sweep with 50 trials is more thorough.

**Recommendation.** §5 unchanged in shape but add a paragraph: "If §1.1 / §1.2 fail at the 0.005 margin, run a stronger baseline (LightGBM-rank + interaction features + 50-trial Optuna sweep) before declaring failure. The decision rule is: SSL+heads must beat *the better of* baseline-XGBoost and stronger-LightGBM."

This protects against the embarrassing failure mode where a 1-day better baseline would have passed, and the author already commits to baseline-strength as the reference number.

### B.MINOR.2 — Loss weights `(1.0, 1.0, 0.3, 0.3)` are not justified beyond "tune on val"

**Severity:** Minor
**Location:** §6.2
**Issue.** The four weights aren't anchored. The author admits this in risk #5 ("Quantile head dominates / drowns out rank head"). DESIGN §3.2 prescribes weights via "loss-magnitude balance at step 0" — i.e., compute initial loss values for one batch, set weights so each loss term contributes equally, then tune from there.

**Recommendation.** §6.2: "Initial weights are set by the loss-magnitude-balance heuristic: at step 0 on a single batch, compute per-head loss values and pick weights such that `w_h × L_h ≈ 1.0` for each head. Tune from there on val with the W&B per-head loss curves." Drop the hardcoded `(1.0, 1.0, 0.3, 0.3)` — they're a guess.

If the author rejects this for reproducibility (config-hash determinism), keep the hardcoded weights but add a one-line `tests/unit/test_posttrain.py::test_loss_weights_balance_at_init` that asserts each head's `w_h × L_h(step=0)` is within 2× of the others. Catches drift if someone changes the heads later without rebalancing.

### B.MINOR.3 — `r=8` LoRA rank not justified

**Severity:** Minor
**Location:** §6.3
**Issue.** Just stated as "locked". `r=8, alpha=16` is the LoRA paper's first-canonical recipe, but the right rank depends on the downstream task complexity and the base model size. For a 480-dim 12-layer transformer being adapted to 8 supervised heads, `r=8` is plausible but not derived.

**Recommendation.** Add to §6.3: "Sensitivity to `r` was not tested; `r=8` chosen by analogy to the LoRA paper. If §1.1 fails by a small margin, a `r ∈ {4, 16, 32}` sweep is the cheapest experiment to escape — log as a Phase 3.5 fallback in §11."

### B.MINOR.4 — `target_modules` is `['q_proj', 'k_proj', 'v_proj', 'o_proj']` but the design narrative says `(W_q, W_v)`

**Severity:** Minor
**Location:** §6.3
**Issue.** The narrative on line 463 says "`peft` LoRA on `(W_q, W_v)` of every backbone attention layer at `r=8, alpha=16` adds ~1.4M trainable params". The locked config on line 473 says all 4 attention projections (`q, k, v, o`). These don't agree — `(q, v)` is 2 projections × 12 layers × 480 × 8 × 2 ≈ 184K params; `(q, k, v, o)` is 4 projections ≈ 368K. (Both are well under 1.4M; the 1.4M figure is for both directions of LoRA which is also a discrepancy.)

**Recommendation.** Reconcile. The right answer is probably `(q, v)` per the LoRA paper's strongest recipe; `(q, k, v, o)` is the more aggressive PEFT default. Pick one, math the param count correctly, and update both the narrative and the config to agree. The numerical claim "~1.4M trainable params" should also be sanity-checked.

### B.MINOR.5 — K=5 vol buckets may be insufficient for fine-grained calibration

**Severity:** Minor
**Location:** §6.4 + §7.3
**Issue.** §7.3 says "per (horizon, vol_bucket=K=5) cell that's ~500K samples". But after the de-correlation in B.MAJOR.2 (5d cell is 17K independent days, not 1.27M), the 5d × bucket-4 (top quintile) cell may have only ~3K independent days. This is on the edge of conformal's stable-quantile regime (the rule of thumb is ≥1000 for ±5% empirical coverage, which 3K satisfies — but barely, before any further stratification).

**Recommendation.** §6.4: "K=5 is the default but the calibration code logs per-cell sample count and warns if any cell has <1000 effective independent samples. If the 5d × top-vol-bucket cell falls below 1000, fall back to K=3."

### B.MINOR.6 — No mention of the `tradable` mask at inference / strategy time

**Severity:** Minor
**Location:** §3.5 (and missing from §6, §8)
**Issue.** §3.5 correctly says training drops invalid (halt/circuit) rows. But what does the model emit at inference time when it's asked to predict on a halted symbol? The pipeline diagram in §2 shows §3.6 walk-forward IC eval; that eval needs a clean answer for "what happens to symbol X at time t when symbol X is halted".

**Recommendation.** Add to §6.1 or §8.2: "Inference-time invalid-symbol handling: if `valid_label=False` at `(sym, t)`, the model still emits predictions (the input bars are valid even if the forward window was halted), but the eval and strategy layers ignore those `(sym, t)` cells. The `valid_input` mask (no halt in the *input* window) is also tracked and is what the strategy layer ultimately filters on."

### B.MINOR.7 — Effort estimate of 9 days is optimistic

**Severity:** Minor
**Location:** §11
**Issue.** 3.4 (`posttrain.py` LoRA + heads + 1-batch overfit + smoke) is budgeted at 2 days. This is the single most complex sub-phase: it integrates `peft`, 8 head instances, 4 loss terms, EMA from Phase 2.1, gradient accumulation across cross-sections, and the 1-batch overfit test. 2 days with no buffer is tight.

3.5 (single-fold full Phase-3 train on H200, "1 d incl. H200 wall-clock") is probably 1.5 days because the H200 run alone is ~6 hours and any debug roundtrip eats half a day.

**Recommendation.** Bump 3.4 to 3 days, 3.5 to 1.5 days, 3.7 to 2.5 days (4 × 6h H200 runs + per-fold debug). New total: ~11 days, ~50 H200-hours. Don't over-promise; the project hasn't yet shipped a sub-phase under its first-listed budget.

### B.MINOR.8 — Phase 3 doesn't articulate what to do with the EMA decay

**Severity:** Minor
**Location:** §6.5
**Issue.** "Phase 4 may sweep decay; Phase 3 uses the default." The default 0.999 is from Phase 2 SSL pretraining, where 50K steps × heavy regularization is the regime. Phase 3 has 30K steps with much smaller trainable param count (LoRA + heads, ~3M). EMA decay 0.999 means the EMA averages over the last ~1000 steps — which at Phase 3's 30K-step budget is the last 3% of training. That's a different regime than Phase 2's "EMA averages over the last 2% of pretraining". May not matter, but defaults are not always portable across phases.

**Recommendation.** Drop "Phase 4 may sweep decay"; replace with "Phase 3 uses 0.999 by default; if val rank-IC oscillates step-to-step (a known LoRA + small-head pathology), bump to 0.9995 as a one-line config change. Logged in W&B and tracked."

### B.NIT.1 — Section §10's repo layout shows `daity/eval/metrics.py` as NEW, but DESIGN §5.1 already lists `eval/metrics.py`

**Severity:** Nit
**Location:** §10
**Issue.** Cosmetic. `metrics.py` exists in DESIGN §5.1 layout — Phase 3 is presumably extending it (rank-IC, CRPS, calibration), not creating it. Worth a one-word change.

**Recommendation.** Change "NEW: rank-IC, CRPS, calibration" to "EXTEND: add rank-IC + CRPS + calibration metrics; existing file may already exist from Phase 1/2". Saves the future implementer a `git status` confusion.

### B.NIT.2 — Pipeline diagram (§2) is missing the random-init ablation branch

**Severity:** Nit
**Location:** §2 ASCII art
**Issue.** The diagram shows the SSL-pretrained backbone path but not the random-init ablation that §1.4 demands. Diagrams should show all required deliverables.

**Recommendation.** Add a parallel box "Phase 3.4-rand: random-init ablation" branching from "Phase 3.3 sampler" — same heads, same loss, random-init backbone, full fine-tune. Helps anyone reading the doc see that §1.4 is a separate train run.

### B.NIT.3 — §3.3 "Why close-to-close, not VWAP" is fine but should mention the other reason

**Severity:** Nit
**Location:** §3.3
**Issue.** §3.3 names two reasons. There's a third: VWAP labels make the conformal calibration's predicted-band correspond to VWAP-execution returns, while the strategy layer in Phase 5 will likely execute at next-bar-open or with a slippage model. Predicting VWAP and trading at next-bar-open is a calibration mismatch.

**Recommendation.** Add one line to §3.3: "(3) The Phase-5 strategy executes at next-bar-open or via a slippage model, not at VWAP — predicting close-to-close keeps prediction and execution in compatible coordinate frames."

### B.NIT.4 — `as_of_inferred_end` in §3.4 is undefined

**Severity:** Nit
**Location:** §3.4
**Issue.** "`effective_close ≤ as_of_inferred_end`" — the reader has to guess what `as_of_inferred_end` is. Presumably it's the calendar-aware end-of-day-corresponding-to-as_of, but it's not defined anywhere in PHASE3.md or referenced from DESIGN.md.

**Recommendation.** One-line definition in §3.4: "`as_of_inferred_end` is the calendar-aware end-of-trading-day timestamp on the day of `as_of` — for `as_of = 2025-12-31 15:25 IST`, it's `2025-12-31 23:59:59 UTC`. Encoded in `daity.data.candles.effective_close`."

---

## Section C — Strategic Verdict

### C.1 Per-gate probability of acceptance

Bottom line: **35-45% confidence that all four §1 gates pass on first try, 60-70% that gates 1, 3, and 4 pass and gate 2 (5d rank-IC) is the failure point.**

| Gate | Description | P(pass) | Reasoning |
|---|---|---|---|
| §1.1 | 30m rank-IC > XGBoost by ≥0.005 | **55%** | The architect-review reframed Phase 2's job: SSL just needs to give a representation marginally better than 18-channel features. XGBoost on 457 flattened features is a strong baseline; the SSL+heads model has cross-attention forecast head's representation but that head was trained on a noise-floor-bound objective. Even-money bet. |
| §1.2 | 5d rank-IC > XGBoost by ≥0.005 | **30%** | 5d at 17K independent samples × ~205 symbols is small for a 36M backbone. Survivorship bias affects 5d most. The cross-section "same `t`" pathology (B.MAJOR.2) creates training-time correlation that hurts 5d most. Even with the recommended fix, 5d is the hardest gate. |
| §1.3 | Conformal coverage in [0.75, 0.85] | **95% as written, 70% under the recommended out-of-sample rewrite** | As written it's vacuous (B.MAJOR.1). With the fix to test on the held-out test fold, coverage drift between val and test is real, but conformal is robust enough that 70% marginal coverage is achievable. |
| §1.4 | Beats random-init by ≥0.005 | **60% under the recommended margin spec** | This is the diagnostic for "did SSL do anything". The H200 v3_xattn run hit val_loss 4.7956 — meaningfully below random-init's expected 5.5+ on the SSL forecast objective, so the SSL did learn *something*. Whether that something transfers to 30m rank-IC depends on representation quality. Slightly better than coin flip. |

The compound event "all 4 pass" is ~55% × ~30% × ~70% × ~60% with positive correlation between gates 1 and 4 (if SSL helps, it helps both vs XGB and vs random) and between 1 and 2 (similar mechanism). Net: 25-40%.

### C.2 Single most important change

**Implement B.MAJOR.2's separate-stream sampler.** The "same `t` for both horizons with 1/75 downweight" alignment is the design's central architectural decision in §4.3, and it has gradient pathologies that make the 5d gate (the hardest gate) even harder. The fix is ~10 lines of training-loop code and removes a source of unrelenting noise from the 5d head's training.

Second-place: **B.MAJOR.1's conformal acceptance rewrite.** Without it, §1.3 is a smoke test, not a gate.

### C.3 What to cut from Phase 3 if budget is tight

In priority order of cut-first:

1. **Cut the direction head** (per A.2). Saves 0.5 days, simplifies §6.2 loss balance, doesn't lose downstream consumer.
2. **Cut the survivorship-shrinkage section** (per A.8). Saves nothing in compute but saves a week of "is 0.10 the right κ" debate downstream. The raw-IC + caveat-panel approach is more honest.
3. **Cut the random-init full-fine-tune ablation if compute is the binding constraint.** Replace with random-init LoRA-only (cheaper and known to be a bad random-init baseline, but it tells you "is the gate's spirit met"). Note this risks a vacuous §1.4 — only do it if the compute budget genuinely doesn't fit 4 SSL folds + 4 random-init folds.
4. **Cut 5d horizon entirely from v1.** This is the strategic option I'd consider if 30m rank-IC at fold F-9 is borderline. If the model can't even do 30m, 5d is wishful. Drop §1.2, drop the 5d head, drop the swing strategy from Phase 5, ship intraday-only. The architect review's Section 7 already recommends this as a Phase-5 cut; pulling it forward to Phase 3 saves 30% of Phase 3 compute.
5. **Cut the contrastive head considerations entirely** (per A.6). Already done implicitly; just make it explicit in the doc.
6. **Cut Phase 3 fold count from 4 to 2** (F-11 and F-12 only). Saves 50% compute. Requires accepting lower confidence on the acceptance decision.

Don't cut: the leakage tests, the conformal pipeline, the XGBoost baseline (even a shittier version), the random-init ablation in *some* form. These are the parts that make Phase 3's "yes" or "no" decision honest.

### C.4 Phase-bundling issues — what should move between phases

**Things PHASE3.md is currently doing that could wait until Phase 4 / 5:**

- **Realized-vol head** (§6.1, last bullet). It's auxiliary in Phase 3 and the actual consumer (Phase 5 sizing) doesn't yet exist. Moving it to "Phase 4 fine-tune" means we don't pay for the multi-task complexity until the consumer materializes. Light cost saving but cleaner phase contract.
- **Tearsheet HTML rendering** (§8.4). This is Phase 5 territory (DESIGN §5.1 puts `tearsheet.py` in `eval/`). Phase 3's `walk_forward.py` should write a JSON; Phase 5's tearsheet renders the JSON into HTML. Cleaner separation; saves 0.5 days from §11's 3.7.
- **Vol bucket precomputation** (§6.4). Used only by the conformal calibrator; doesn't need to be in the cross-section batch. Move to a post-train artifact that the conformal step computes from val residuals.

**Things PHASE3.md is deferring that should arrive in Phase 3:**

- **Coverage-on-test-fold conformal check** (B.MAJOR.1). The author defers it to Phase 5. It needs to be in Phase 3.
- **Hparam compatibility check** (A.10). The author leaves it as an open question. It needs to be in Phase 3 from day one — the cost of catching a phase-skew bug at Phase 5 time is one phase of redo.

### C.5 Missing pieces

1. **Per-symbol bias diagnostics in the IC eval.** §8.2 lists 4 metrics but doesn't surface which symbols the model wins/loses on. The strategy layer needs this — "the model is good at large-cap PSU banks and bad at small-cap pharma" is the kind of finding that changes Phase 5 strategy design. Add a per-symbol IC table to the tearsheet.
2. **Determinism / reproducibility statement.** Phase 2's `(git_sha, config_hash, as_of)` provenance triple needs to extend to Phase 3 outputs. Mentioned in CLAUDE.md but not in PHASE3.md. The conformal artifacts (§7.4) and the walk-forward artifacts (§8.4) need provenance.
3. **W&B tracking section.** Phase 2 invested heavily in W&B (per Phase 2.1 patch). PHASE3.md doesn't mention W&B once. Add a section: which run-level and step-level metrics are logged (per-head loss, per-horizon rank-IC at val time, EMA delta, RevIN gamma norm canary from `phase_2_architect_review.md` §9, gradient-norm-per-loss-term).
4. **A "what does inference look like" section.** §1 is clear about training acceptance, but the deliverable is a model that emits predictions at strategy time. What's the `(input → output)` contract? What's the latency budget? How does conformal lookup work at inference (pickled `s*` table per cell)? This becomes Phase 5 plumbing's headache without it; including a §6.10 ("Inference contract") makes Phase 5 cheaper.
5. **Failure-mode automation.** §1's failure-mode budget describes what to do if each gate fails, but it's prose. Add a "what triggers each fallback" decision tree, ideally as part of `walk_forward_cli.py`'s output: "If 30m fails by ≤ 0.003, run `--fallback=stronger_xgboost`; if fails by > 0.003, run `--fallback=larger_lora_r`; if both fail, declare Phase 4 entry conditional on §1.4 passing." This is the kind of structure that turns a vague "if it fails, do something" into a script.
6. **The leakage gate for the XGBoost baseline.** §5 doesn't mention a leakage test for the baseline. The 457 features are derived from the same 18-channel feature engine, so it inherits Phase 1's leakage gates — but the per-window aggregation (`mean`, `std`, etc.) introduces a new opportunity to leak (e.g., if "linear-trend slope over the window" is computed across `(t - window, t]` vs `[t - window, t)`, the difference is one bar and one bar is the difference between leakage and not-leakage). Add `tests/leakage/test_xgboost_features_leakage.py`.

---

## Section D — Final Disposition

**Disposition: Approve with Major edits.**

The design is sound in shape and the acceptance gate is the right one. The four-headed multi-task setup with LoRA + frozen backbone + cross-section batch + conformal calibration + walk-forward is a defensible architecture that closes the loop on "does the SSL backbone produce useful representations". The author has internalized the predecessor architect review's redirect ("Phase 2 is done when Phase 3 beats XGBoost") and is shipping the design that operationalizes it.

The three Major findings — vacuous §1.3 conformal gate, gradient pathologies in the §4.3 same-`t` alignment, under-specified §1.4 random-init ablation — are all fixable in a post-review amendments callout (style: same as `DESIGN.md`'s amendments) without touching the bulk of the doc. Fix those, address the eight Minor findings inline (most are 1-3 line edits), accept or reject the Nits at author discretion, and Phase 3 implementation can start.

**Required before any Phase-3 code lands:**

1. B.MAJOR.1: Rewrite §1.3 to test on test fold, not calibration set.
2. B.MAJOR.2: Switch §4.3 to separate-stream sampling for 30m and 5d. Drop the 0.2× and 1/75 factors in §6.2.
3. B.MAJOR.3: Specify the random-init ablation with margin (≥0.005), full fine-tune of random backbone, and 3-seed best-of comparison.

**Strongly recommended before Phase-3 code lands:**

4. Drop the direction head (A.2) — saves 0.5 days, simplifies loss balance.
5. Two-LoRA-bank architecture (A.3) — eliminates 30m/5d gradient interference.
6. Hparam compatibility check (A.10) — catches a phase-skew bug class.
7. Drop the survivorship κ (A.8) — replace with the raw-IC + caveat-panel approach.

**Minor edits (do inline, don't block on):**

8. B.MINOR.1-8 — small architectural / documentation improvements.
9. B.NIT.1-4 — cosmetic.

### Confidence summary

- Probability all four §1 gates pass on first try: **25-40%**.
- Probability of "30m gate passes, 5d gate fails" (the most likely partial outcome): **40%**.
- Probability of "no gate passes, Phase 2 needs revisiting": **15-20%**. This is the worst case but not the modal case.

The biggest strategic risk to the project is no longer Phase 2 (closed) but **Phase 3's 5d gate**. Even after addressing Major findings, 5d's combination of high noise, small effective sample size, and survivorship bias makes it a genuinely uncertain bet. The author's failure-mode budget already plans for this — drop swing from v1 if §1.2 fails — which is the right hedge.

The biggest tactical risk is **shipping Phase 3 with a vacuous conformal acceptance gate (B.MAJOR.1)**. Without that fix, we approve a deliverable whose calibration claim cannot be falsified at acceptance time, deferring the actual test to Phase 5 where a calibration failure is a Phase-3-redo cost. Don't approve the doc until §1.3 is rewritten.

Once those are addressed, ship it. Phase 3 is the project's first real test of "does this system have alpha?", and getting to that test cleanly is more important than getting it perfect.
