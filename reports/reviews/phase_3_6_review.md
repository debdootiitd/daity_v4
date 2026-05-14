# Phase 3.6 Review — Conformal calibration (CQR + vanilla, per-cell picker)

**Reviewer:** Independent code reviewer
**Date:** 2026-05-11
**Disposition:** Approve with minor changes (0 Critical, 2 Major)

Scope: `daity/models/conformal.py` (~310 LOC),
`tests/unit/test_conformal.py` (~294 LOC, 14 tests — not 17 as the
review prompt stated; recount: score-fn (5) + calibration (3) +
predict_band (4) + coverage (3) + persistence (2) = 17 if you count
the index-alignment test; either way the coverage is broad).
Cross-referenced PHASE3.md §7 (amendment 9), §6.4 (K=5 → K=3
fallback), §1.3 (out-of-sample gate), §6.11 (inference contract).

The module is small, freestanding, and reads cleanly. All score
formulas are correct, the picker logic is sound for the documented
intent, and persistence round-trips reliably. The two Major findings
are both about contract / spec-conformance gaps rather than
algorithmic bugs: (a) `min_cell_size` default of 100 is an order of
magnitude below the PHASE3.md §6.4 / §7.3 spec of 1000, and (b) the
module silently consumes pathological quantile predictions
(`q_lo > q_hi`) without warning. Everything else is Minor or below.

§3.7 will own out-of-sample coverage on a held-out test fold (PHASE3
§1.3) — correctly NOT in §3.6's scope.

---

## Section A — Conformal correctness

### A.1 CQR score formula (scrutiny A.1) — correct, with one nuance

Per Romano-Patterson-Candès 2019 Eq. 6:

```
s_i = max( (q_lo - y_i) / (q_med - q_lo),  (y_i - q_hi) / (q_hi - q_med) )
```

The implementation at `conformal.py:86-88` adds `_EPS = 1e-9` to each
denominator. The published formula does NOT use an epsilon — and yes,
this introduces a small DOWNWARD bias on the score for any cell where
either denominator is on the order of 1.

For typical log-return scales (`|q_med - q_lo| ~ 0.005` for 30m bars):
the relative bias is `1e-9 / 5e-3 = 2e-7`, negligible. For a
well-trained quantile head, `_EPS` never matters.

For a pathologically-uncalibrated head where `q_med ≈ q_lo` (e.g., the
quantile head collapsed all quantiles to the same value, possible
after a divergent training run), `_EPS = 1e-9` makes the ratio finite
but artificially small. The score then under-estimates the true
nonconformity. In split conformal, a single under-estimated score is
absorbed into the empirical quantile, so the effect is bounded —
unless ~all denominators are degenerate, in which case `s_star` is
~0/`_EPS` = ~0, the band collapses to `q_med ± 0`, and the picker's
in-sample coverage will be ~0.1 (the `y == q_med` mass). The picker
will then favor vanilla — *if* vanilla's score function happens to be
better-behaved. So the failure mode is detected by the picker.

**Recommendation:** Add a docstring sentence stating that `_EPS` is a
defensive guard against degenerate quantile-head outputs and that the
bias it introduces is `O(_EPS / (q_med - q_lo))` — operationally
zero for any non-degenerate model. See **Nit-1**.

A defensible alternative is to compute the ratio WITHOUT epsilon and
mask `np.inf` / `np.nan` scores out of the empirical quantile (treat
degenerate residuals as "score = +inf — definitely outside the band"
for picker purposes). That's a Minor design discussion, not a bug.

### A.2 Empirical quantile formula (scrutiny A.2) — correct

`_empirical_quantile` at `conformal.py:116-127`:

```python
k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
k = min(max(k, 0), n - 1)
return float(np.sort(scores)[k])
```

Per Romano 2019 Eq. 1, the split-conformal `(1-α)`-quantile is the
`⌈(n+1)(1-α)⌉ / n` empirical quantile, which equals the
`⌈(n+1)(1-α)⌉`-th order statistic in 1-indexed land, i.e.
`⌈(n+1)(1-α)⌉ - 1` in 0-indexed land. **Matches.** ✓

**Boundary cases:**

- `n = 0`: returns `inf` (correct: no calibration data → band must
  not cover; conservative).
- `n = 1, alpha = 0.2`: `k = ⌈2 × 0.8⌉ - 1 = 1`, clamped to `0`. The
  single score becomes the "quantile". Bounded-above by `n-1` is the
  right behavior — when `(n+1)(1-α) > n`, no finite-sample quantile
  attains the nominal coverage, and the standard convention is to
  emit `+∞`. The current code emits the max score instead. **This is
  a deviation from the strict split-conformal guarantee when `n` is
  smaller than `⌈1/α⌉ - 1 = 4` for `α = 0.2`.** With val cells of
  ~3.4K rows (§6.4 1d/2d × K=5), unreachable in production, but a
  small-N unit test (`n=10` say) should pin down this behavior. See
  **Minor-3**.

### A.3 Picker bias (scrutiny A.3) — implementation matches spec

`calibrate_one_cell` at `conformal.py:159`:

```python
score_fn = "cqr" if dist_cqr <= dist_van else "vanilla"
```

PHASE3.md §7.1 says "pick whichever yields an empirical val-set
coverage in `[1-α-ε, 1-α+ε]` first (smallest ε wins)". The code's
`abs(cov - target)` formulation captures "smallest ε". Ties → CQR by
convention. **Matches spec.** ✓

The reviewer is correct that both functions' in-sample coverage is
bounded below by `1-α` (the split-conformal guarantee), so the picker
favors whichever score has LOWER (tighter) coverage. This is the
"closer-to-target" choice the spec mandates. The inductive bias is
"prefer the score whose finite-sample guarantee is least slack" —
which is a defensible reading of "closest to `1-α`". §7.2's narrative
about "CQR is the right inductive bias when calibrated" supports the
tie-break direction.

There's a subtle wrinkle worth noting: with `n = 1000, α = 0.2`, the
finite-sample correction yields `k / n = ⌈(1001 × 0.8)⌉ / 1000 =
801 / 1000 = 0.801` — so both in-sample coverages will be ≥ 0.801,
and both will be in `[0.801, 0.801 + 1/1000]` for any reasonable
score distribution. The picker is essentially measuring noise. With
n ~5K-240K (§6.4 estimates), the picker has more signal, but on the
small 1d/2d × top-bucket cells (~3.4K), the picker's choice is
nearly stochastic. The spec accepts this. See **Minor-4** (worth a
warning).

### A.4 Picker doesn't penalize wider bands (scrutiny A.4) — design question

The picker is indifferent between two score functions that achieve
the same in-sample coverage with different band widths. PHASE3.md §7.1
doesn't mention width; §7.2's narrative argues CQR is "the right
inductive bias" (i.e., width should adapt with predicted vol), but
the picker's actual selection rule ignores width entirely.

**Two failure modes this admits:**

1. CQR with badly-calibrated `(q_lo, q_hi)` produces a wider band
   than vanilla but achieves identical in-sample coverage. The picker
   picks CQR (or vanilla, on ties, depending on float arithmetic).
   Operator gets a wider band than necessary on this cell. PnL hit:
   wider bands → looser strategy thresholds → noisier execution.
2. Vanilla with no vol-modulation has the *same* width across vol
   buckets, but achieves the same coverage as CQR (which DOES modulate).
   The picker prefers whichever wins the tiebreak — the operator
   loses the cross-bucket adaptivity CQR was supposed to bring.

Recommendation: surface as a design question for the §3.7 reviewer.
Adding an `interval_width` tiebreaker is a 3-line change:

```python
width_cqr = float(s_star_cqr * np.mean((q_pred[:, IDX_Q_HI] - q_pred[:, IDX_Q_LO])))
width_van = float(2 * s_star_van)
if dist_cqr == dist_van:
    score_fn = "cqr" if width_cqr <= width_van else "vanilla"
```

But this changes the spec. Defer to a §3.6 amendment. See **Minor-1**.

### A.5 Quantile-monotonicity NOT enforced (scrutiny A.5) — silent corruption

If the quantile head emits `q_lo > q_med` or `q_med > q_hi`, the CQR
denominators go negative. The formula still computes a value, but:

- `(q_lo - y) / (q_med - q_lo)`: denominator is negative, so a `y`
  ABOVE `q_lo` (which is the "good" case) yields a NEGATIVE numerator
  divided by a NEGATIVE denominator → POSITIVE score. The score now
  fires when `y` is COVERED, exactly inverted from the intended
  semantics.
- The `_EPS` guard does NOT fix this — it's only a divide-by-zero
  guard, not a sign-correction guard.

PHASE3.md §6.1 doesn't specify a quantile-monotonicity constraint in
the posttrain loss; pinball loss does not enforce ordering. The
quantile-crossing problem is well-known in quantile regression
(Chernozhukov 2009, "Quantile rearrangement"). Empirically with
30K-step posttrain on log-return targets, the head's output may
occasionally cross — especially in low-data cells (1d/2d) or early
in training.

The conformal module is the natural place to detect this. Three
defensive options, in increasing strength:

1. **Log + count.** `n_crossing = sum(q_lo > q_med) + sum(q_med >
   q_hi)`. Log warn if `n_crossing > 0`. Cheap; surfaces the problem
   to the operator without breaking the pipeline.
2. **Rearrange.** Sort the quantiles before computing scores
   (`np.sort(q_pred, axis=1)`). One-line fix; matches the published
   "quantile rearrangement" trick.
3. **Refuse + fall back to vanilla.** If any row has crossing
   quantiles, raise on that cell, force the picker to vanilla.

Option 1 is the right §3.6 fix; option 2 might be the right §3.7 fix
once the operator has data on how often crossings happen in production.
Currently option 0 ("silently corrupt the score function") is what
ships. **Major-2.**

---

## Section B — Pooling logic

### B.1 `min_cell_size = 100` default (scrutiny B.6) — spec mismatch

PHASE3.md §6.4 + §7.3:

> **warns if any cell has <1000 effective independent samples**. If the
> 1d or 2d × top-vol-bucket cell falls below 1000, fall back to K=3
> buckets for that horizon.

The code default is `min_cell_size = 100` (`conformal.py:179`), an
order of magnitude smaller than the spec's 1000. Three implications:

1. **The K=5 → K=3 fallback is the SAMPLER's job, not the conformal
   module's**, per the spec ("the conformal module logs per-cell
   sample counts and warns; if 1d or 2d × top-bucket cell falls below
   1000, fall back to K=3"). The conformal module doesn't currently
   warn — it pools at 100, silently. The spec's behavior is "log
   warning, then operator changes K and re-runs".
2. The pooling rule is also different: spec says "K=5 → K=3 for the
   whole horizon", code says "pool just the small cells into
   `bucket=-1` and leave the rest as K=5". Both are defensible, but
   the spec's K=3 fallback preserves stratification on the larger
   buckets in a different way — the §3.7 reviewer will need to know
   which one the production CLI uses.
3. The unit test default `min_cell_size = 100` (`test_conformal.py:154`)
   passes with synthetic data and locks in the wrong constant.

**Recommendation:**
- Default `min_cell_size = 1000` to match §7.3.
- Emit a `log.warning` (not `log.info`) when pooling happens, listing
  the affected buckets and their sample counts.
- Document the divergence from the spec's "K=5 → K=3 for the horizon"
  pooling rule, OR implement the spec's rule (pool ALL buckets into
  the new K=3 stratification when ANY bucket falls below threshold,
  not just the small ones).

See **Major-1.**

### B.2 Pooling under-estimates s_star (scrutiny B.7) — yes, conservative

Pooling heterogeneous small buckets into `bucket=-1` produces a
fatter score distribution than each bucket would individually, so
`s_star_pooled ≥ max(s_star_per_bucket)` in expectation. This means
small-cell predictions get LARGER bands than the per-bucket
calibration would.

This IS the desired conservative behavior — split conformal's
marginal coverage guarantee REQUIRES the calibration set to be
exchangeable with the test set. When the bucket has too few samples
to estimate its own quantile reliably, pooling is the only way to
preserve the exchangeability assumption. Conservative bands are the
price. Note in docstring. See **Nit-2.**

### B.3 No warning when buckets are degenerate (scrutiny B.8) — yes, fine

If `vol_bucket` has only one unique value (e.g., production day where
all symbols are in the same regime — unlikely but possible),
`np.unique` returns `[v]`, no pooling, one cell. `predict_band`
queries with `vol_bucket=v` will find the cell. Operator's downstream
sanity check (per-bucket coverage table in the tearsheet) will surface
the degenerate input. Acceptable; not a bug. See **Nit-3** for a
log.info ("calibrate_per_cell: only 1 unique vol_bucket; K = 1").

---

## Section C — Predict band

### C.1 Vanilla band ignores quantile predictions (scrutiny C.9) — yes, by design

The vanilla band `q_med ± s_star` doesn't use `q_lo` or `q_hi`. This
is correct vanilla split-conformal: a single global `s_star` per cell
gives a homoscedastic band. The reviewer's framing — that CQR vs
vanilla is a representational choice, not just a calibration knob —
is exactly right. The picker, by selecting per-cell, effectively
chooses heteroscedastic vs homoscedastic on a cell-by-cell basis.

This should be documented in the docstring (currently only the
`if cell.score_fn == "cqr"` branch is described; the architectural
implication is implicit). See **Nit-4.**

A subtle consequence: in a regime where the quantile head is
well-calibrated GLOBALLY but mis-calibrated per-bucket (e.g.,
high-vol bucket has overconfident quantiles), CQR loses on the
high-vol bucket only, and the picker correctly routes that ONE bucket
to vanilla. The operator's deployed system then has a band that's
CQR on 4 buckets and vanilla on 1. This is the design; verify the
§3.7 walk-forward report exposes which bucket got which score
function (per-cell `score_fn` JSON field is already there ✓).

### C.2 Python-level loop in predict_band (scrutiny C.10) — acceptable

`for i in range(n)` at `conformal.py:246`. At inference:
- 205 symbols × 75 timestamps × 4 horizons = ~61.5K predictions/day.
- Python-loop overhead: ~50ns per iteration (per modern numpy
  benchmarks). 61.5K × 50ns = 3 ms/day. Negligible.

But: in a backtest sweep where the conformal is called per
cross-section across 4 folds × 1 year = ~4M predictions, that's
~200 ms. Still fine.

If §3.7 finds it's a hot spot, vectorize:

```python
# Build per-row s_star and score_fn flag.
buckets = np.array([int(b) for b in vol_bucket])
s_star_arr = np.full(n, np.nan)
is_cqr = np.zeros(n, dtype=bool)
for b, cell in cells.items():
    mask = buckets == b
    if not mask.any():
        continue
    s_star_arr[mask] = cell.s_star
    is_cqr[mask] = (cell.score_fn == "cqr")
# Fill fallback
unfilled = np.isnan(s_star_arr)
if fallback is not None and unfilled.any():
    s_star_arr[unfilled] = fallback.s_star
    is_cqr[unfilled] = (fallback.score_fn == "cqr")
# Compute both branches, blend
band_lo_cqr = q_med - (1.0 + s_star_arr) * (q_med - q_lo)
band_lo_van = q_med - s_star_arr
band_lo = np.where(is_cqr, band_lo_cqr, band_lo_van)
# ... similar for band_hi
```

Acceptable to defer. See **Nit-5.**

### C.3 `cells.get(b) or fallback` truthiness (scrutiny C.11) — safe

`@dataclass`-decorated classes are truthy by default; only `__bool__`
or `__len__` (when 0) make them falsy, neither of which `ConformalCell`
implements. So `cells.get(b) or fallback` correctly returns the
fetched cell when present, fallback otherwise. ✓

A stylistic concern: `cells.get(b, fallback)` is the idiom for this
pattern and would be unambiguous. The `or` form invites a future
refactor that adds `__bool__` (e.g., "consider an uncalibrated cell
falsy") and silently breaks lookup. See **Nit-6.**

---

## Section D — Persistence

### D.1 Filename parser handles digits and negatives (scrutiny D.12) — yes

`stem.rsplit("_", 1)` splits on the LAST underscore:

- `30m_5.json` → `("30m", "5")` ✓
- `120m_-1.json` → `("120m", "-1")` → `int("-1") = -1` ✓
- `30m_-1.json` → `("30m", "-1")` → `int("-1") = -1` ✓

The test `test_load_cells_handles_horizon_with_digits` covers `120m_5`
but doesn't cover the negative case (which is the production fallback
bucket). Add a test: `test_load_cells_handles_negative_bucket_id` with
`bucket=-1`. See **Minor-2.**

What about `30m_abc.json` (corrupt filename)? `int("abc")` raises
`ValueError`, which `load_cells` propagates. Loud is correct.

What about `30m_5_extra.json`? `rsplit("_", 1) → ("30m_5", "extra")`,
`int("extra")` raises. Correct.

What about `30m.json` (no underscore at all)? `rsplit("_", 1)` returns
a 1-element list, `horizon, bucket_str = ...` raises `ValueError:
not enough values to unpack`. The `*_*.json` glob filter prevents
this, but a defensive check wouldn't hurt:

```python
parts = stem.rsplit("_", 1)
if len(parts) != 2:
    log.warning(f"load_cells: skipping {path.name} (unexpected filename)")
    continue
```

See **Nit-7.**

### D.2 JSON schema stability (scrutiny D.13) — should version

`asdict(cell)` writes whatever fields `ConformalCell` has. If Phase 5
adds a `confidence_interval: tuple[float, float]` field (e.g., a
bootstrap CI on `s_star`), loading an OLD JSON into a NEW
`ConformalCell` raises `TypeError: __init__() missing 1 required
positional argument: 'confidence_interval'`.

The fix is conventional:

1. Make new fields `Optional[T] = None`.
2. Add a `schema_version: int = 1` field to `ConformalCell`.
3. In `load_cells`, read `data.get("schema_version", 0)` and migrate
   if `< current`.

The current schema is small (7 fields, all primitives). The risk of a
field addition is real (e.g., width-tiebreaker from A.4 would add
`interval_width: float`). See **Minor-5.**

A lighter alternative: load via `ConformalCell(**{k: v for k, v in
data.items() if k in ConformalCell.__dataclass_fields__})` — silently
drops unknown fields, but still fails on missing required ones.

Recommend: add `schema_version: int = 1` now (one-line change), make
all FUTURE fields optional with defaults, document the contract.

---

## Section E — Test coverage gaps

### E.1 What's tested — strong coverage of the implemented surface

- Score formulas: CQR sign in 3 cases, vanilla absolute residual. ✓
- Empirical quantile: implicit via the in-sample coverage assertion. ✓
- Picker selects a function, in-sample coverage near nominal. ✓
- Cell pooling: small buckets merged into `bucket=-1`. ✓
- predict_band: CQR width formula, vanilla constant width, fallback,
  missing-cell raise. ✓
- empirical_coverage: perfect and zero cases. ✓
- End-to-end: synth → calibrate → predict → measure, coverage in
  `[0.78, 0.82]` for n=2000. ✓
- Persistence round-trip: positive bucket, fallback bucket, horizon
  with digits. ✓

### E.2 What's NOT tested — high-value gaps

In risk-order:

a. **Quantile crossing (q_lo > q_med).** No test verifies the module's
   behavior on pathological quantile predictions. Per §A.5 above,
   the current code SILENTLY produces meaningless scores. A test that
   forces `q_lo > q_med` and asserts either (i) `log.warning` fires,
   or (ii) scores are computed under monotonicity rearrangement,
   would lock in defensive behavior. Critical for §3.6 robustness.
   Tied to **Major-2.**

b. **Out-of-sample coverage on a held-out fold.** PHASE3.md §1.3 is
   the actual acceptance gate. §3.6 correctly defers this to §3.7,
   but a unit test that splits the synthetic universe (cal n=1000,
   test n=1000) and asserts test-coverage ∈ [0.75, 0.85] would
   provide an internal regression catch without waiting for §3.7's
   end-to-end. ~30 LOC. **Recommended.**

c. **Picker on data where CQR vs vanilla diverges meaningfully.**
   `_synth_quantiles_aligned_to_y` is symmetric and homoscedastic, so
   CQR and vanilla converge to the same answer (which is why the test
   asserts `score_fn in ("cqr", "vanilla")` rather than picking one).
   Add a test with heteroscedastic Y (`sigma = sigma_0 * (1 +
   x/q_hi)`) where CQR should win; and a test with badly miscalibrated
   `(q_lo, q_hi)` where vanilla should win. Validates the picker's
   discriminative ability. See **Minor-6.**

d. **Negative s_star** (CQR can produce these when the inner band
   already over-covers). When the calibration set has `s_max < 0`,
   the empirical quantile is negative, `s_star < 0`. Then
   `(1 + s_star)` can be in `(0, 1)`, shrinking the band INSIDE the
   `(q_lo, q_hi)` interval. Geometrically valid (over-covering inner
   band gets shrunk to nominal), but the test suite doesn't exercise
   it. Add a test:

   ```python
   def test_predict_band_negative_s_star_shrinks_band() -> None:
       cell = ConformalCell(..., score_fn="cqr", s_star=-0.5, ...)
       # q_lo = -1, q_hi = +1, s_star = -0.5
       # band_lo = 0 - 0.5 * 1 = -0.5
       # band_hi = 0 + 0.5 * 1 = 0.5
       # Band is INSIDE the original (q_lo, q_hi) interval.
   ```

   See **Minor-7.**

e. **Empty cells / empty `np.unique`** — `vol_bucket` is empty array.
   Should `calibrate_per_cell` return `{}` or raise? Current behavior:
   loop body never runs, returns `{}`. Add a test that pins this.
   See **Nit-8.**

f. **Very small calibration sets (n=1, 4, 10)** to lock in
   `_empirical_quantile`'s boundary clamp behavior. Per §A.2, n < 4
   is the regime where the strict guarantee doesn't hold. See
   **Minor-3.**

g. **Per-bucket out-of-sample coverage holds.** The picker's
   per-cell decision is supposed to give per-cell coverage near `1-α`
   on a held-out fold. No test verifies this. Highest-value addition,
   defer to §3.7. See **Recommendation.**

h. **Cell-size warning fires.** Per §B.1, the spec says "warn if any
   cell has <1000". No test on the warning. See **Minor-1.**

---

## Section F — Integration with §3.5 / §3.7

### F.1 Wiring (scrutiny F.16) — module contract not documented

The conformal module takes numpy arrays — `q_pred (N, 5)`,
`y (N,)`, `vol_bucket (N,)` — and emits JSON. The §3.7 walk-forward
CLI must:

1. Build a cal-set sampler over the val window.
2. Forward all val cross-sections through the trained module, collect
   `q_pred` (the quantile head's output) and `vol_bucket`.
3. Build per-horizon arrays.
4. Call `calibrate_per_cell` per horizon.
5. Save cells under `reports/conformal/{horizon}_{bucket}.json`.
6. On the test fold: load, run `predict_band`, compute
   `empirical_coverage` per `(horizon, bucket)` and overall.

None of this is in §3.6, correctly. But the module's docstring should
spell out this consumer contract — what shape arrays it expects,
what the calling pattern is, what the output JSON layout looks like.
Currently the docstring lists the four public functions but doesn't
narrate the §3.7 lifecycle. ~10 lines added. See **Minor-8.**

### F.2 Vol-bucket alignment (scrutiny F.17) — contract should be explicit

`CrossSection.vol_bucket: Tensor(N,) int8 ∈ [0, K-1]` per
`cross_section.py:26-30`. The conformal module accepts arbitrary
integers and treats `-1` as the special fallback bucket.

**Contract that should be documented:**

- The sampler emits `vol_bucket ∈ [0, K-1]`.
- The conformal module reserves `-1` for the pooled fallback cell.
- Negative bucket ids OTHER than `-1` should never occur; either the
  module raises, or it pools them with `-1`.
- The module accepts `int`-like input (`int8`, `int32`, `int64`, plain
  Python `int`). `predict_band` converts via `int(vol_bucket[i])`. ✓

Defensive: `calibrate_per_cell` could `assert (vol_bucket >= 0).all()`
to enforce the sampler's invariant. See **Minor-9.**

The `int8` type is the sampler's choice. If the sampler ever emits
buckets outside `[-128, 127]` (e.g., K > 127, not in scope), `int8`
overflows. K=5 with `-1` fallback well within range. ✓

---

## Section G — Other findings

### G.1 `compute_cqr_scores` validates only `q_pred.shape[1] >= 5`

`conformal.py:75-77`: validates `q_pred` has at least 5 columns. But
the function hard-codes indices `IDX_Q_LO=0, IDX_Q_MED=2, IDX_Q_HI=4`,
implicitly assuming the input is in the order `(q_0.1, q_0.25, q_0.5,
q_0.75, q_0.9)`. If a caller passes the quantiles in a different
order (e.g., `(q_0.5, q_0.1, q_0.25, q_0.75, q_0.9)`), the scores are
garbage and there's no detection.

The test `test_quantile_indices_match_phase3_quantiles` locks the
indices to `PHASE3_QUANTILES`, which is good — but it tests the
indices, not that the caller's `q_pred` is in this order. The §3.7
caller will pass `outputs.q` from the quantile head, which IS in this
order per `losses.py:214`. But the contract is implicit.

Strengthen: add an assertion or a `q_levels: tuple[float, ...]`
parameter to the score functions and validate against `(0.1, 0.5,
0.9)` at the indices. See **Nit-10.**

### G.2 `compute_vanilla_scores` accepts wider `q_pred`

The vanilla score only uses `q_med = q_pred[:, IDX_Q_MED]`. The
validation `q_pred.shape[1] >= 5` is over-strict — vanilla could work
with `q_pred.shape[1] >= 3` (just q_med). This is fine for the v3
pipeline but if a future caller wanted to do quantile-free vanilla
conformal (e.g., for a point-prediction backbone), they'd have to
pass a fake quantile array. Defer; doesn't matter for §3.6. **No
action.**

### G.3 `alpha = 0.2` is a function default, not a config knob

`alpha=0.2` is hard-defaulted in `calibrate_one_cell`,
`calibrate_per_cell`. PHASE3.md §1.3 specifies `α = 0.2` (80% band) as
the gate. If §3.7's walk-forward wants to ablate `α ∈ {0.1, 0.2}`, the
default is fine; the CLI should pass `alpha` explicitly. Document
that `α` is a global Hydra-config parameter and not a per-cell knob.
**Nit-11.**

### G.4 `coverage_in_sample` stored but unused at inference

`ConformalCell.coverage_in_sample` is stored in the JSON for sanity
purposes (per the docstring at `conformal.py:104-106`) but `predict_band`
doesn't read it. That's correct — it's a diagnostic, not a runtime
input. But §3.7 should surface this in the walk-forward report so the
operator can see per-cell calibration-set coverage at a glance. **Just
a note for §3.7.** No action in §3.6.

### G.5 Style consistency with §3.5

The module's style is clean, dataclass-driven, type-hinted, and uses
the project's `get_logger`. No `print` statements. Matches the
project's conventions per the §3.5 review. ✓

### G.6 Numerical safety

`_empirical_quantile`'s `np.sort(scores)[k]` is O(n log n) per cell;
production cells are <250K rows so sort is ~5ms. `np.partition`
would be O(n) and worth O(20%) at scale, but the saving is invisible
in §3.7's wall-clock. **No action.**

`compute_cqr_scores` produces `np.inf` and `np.nan` if `_EPS` is too
small for catastrophic cancellation. Production scale (`|q|` up to
~0.2 for 2d log-return) is comfortably finite. No action.

---

## Critical / Major / Minor / Nit findings

### Critical

(None.)

### Major

**Major-1** — `min_cell_size = 100` default is an order of magnitude
below the PHASE3.md §6.4 + §7.3 spec (1000) and the warn-and-fallback
pooling behavior the spec mandates (K=5 → K=3 for the WHOLE horizon
when any cell falls below threshold) is replaced with a different
behavior (pool just the small cells into `bucket=-1`, leave the rest
at K=5). Three changes:

a. Default `min_cell_size = 1000`.
b. `log.warning` (not `log.info`) when pooling happens; structured
   payload with bucket id + size.
c. Document the divergence from the spec's "K=5 → K=3 for the
   horizon" rule, OR implement the spec's rule. Recommend implementing
   the spec's rule and surfacing it via a new `min_cell_size_strategy:
   Literal["pool_small", "k_fallback"]` parameter (the test's existing
   "pool_small" behavior is reasonable for the sampler-level case).

The unit test default also needs to change OR an explicit
`min_cell_size=100` should be passed everywhere it's needed for
the small synth dataset. Either way, lock in the production default
via a smoke test that asserts the warning fires.

**Major-2** — Quantile-crossing (`q_lo > q_med` or `q_med > q_hi`)
silently produces meaningless CQR scores. The denominators flip sign,
inverting the score's semantics. Pinball loss does not enforce
monotonicity; in low-data cells (1d/2d × small bucket) it's
empirically plausible. The conformal module is the right place to
detect — either log+count and proceed (option 1 above) or rearrange
quantiles before scoring (option 2). Add a test that injects a
crossing row and asserts defensive behavior.

### Minor

**Minor-1** — Picker doesn't penalize wider bands. If CQR and vanilla
tie on in-sample coverage, the implementation picks CQR on ties; a
width tiebreaker would prefer the tighter band. PHASE3.md §7.1
doesn't explicitly mandate this. Surface as a §3.7 design question;
the 3-line fix is documented above (§A.4).

**Minor-2** — `load_cells` test coverage misses the negative-bucket
case. Add `test_load_cells_handles_negative_bucket_id` with
`bucket=-1`.

**Minor-3** — `_empirical_quantile`'s boundary clamp `min(max(k, 0),
n - 1)` deviates from the strict split-conformal guarantee when
`n < ⌈1/α⌉ - 1`. With `α = 0.2` that's `n < 4`. Production unreachable
(cells >100). But add a small-N test to pin behavior:
`test_empirical_quantile_n_lt_4_clamps_to_max_score`.

**Minor-4** — Picker is noisy on small cells (n < ~5K). The
`abs(cov_cqr - target)` differences are ~`1/n` even when both
functions are equally well-calibrated. Add a `log.info` line that
emits both `(cov_cqr, cov_van)` for operator transparency:

```python
log.info(
    "calibrate_one_cell %s bucket=%d: cqr cov=%.4f, van cov=%.4f, picked=%s",
    horizon, vol_bucket, cov_cqr, cov_van, score_fn,
)
```

**Minor-5** — JSON schema versioning. Add `schema_version: int = 1`
to `ConformalCell`. Future field additions become backward-compatible
via `Optional[T] = None`. Mitigates a Phase 5 reload break.

**Minor-6** — Tests don't exercise picker discrimination. Add a
heteroscedastic-Y test where CQR should provably win, and a
narrow-band-quantile test where vanilla should provably win. Locks in
the picker's discriminative power.

**Minor-7** — Test coverage gap: `s_star < 0` (CQR over-covers).
Geometrically valid but unexercised. ~10-line test.

**Minor-8** — Module docstring doesn't narrate the §3.7 consumer
contract. Add a "Lifecycle" section: (1) val-set forward to get
`q_pred + y + vol_bucket`, (2) `calibrate_per_cell`, (3)
`save_cells`, (4) test-fold forward, (5) `load_cells + predict_band +
empirical_coverage`. Helps the §3.7 implementer.

**Minor-9** — `calibrate_per_cell` should assert
`(vol_bucket >= 0).all()` to enforce the sampler's invariant
(`int8 ∈ [0, K-1]`), with a clear error message pointing at the
sampler.

### Nit

**Nit-1** — `_EPS = 1e-9` is a defensive guard; the bias it
introduces should be documented in the `compute_cqr_scores` docstring.

**Nit-2** — Add a docstring sentence to `calibrate_per_cell`
explaining that pooled cells produce wider bands by construction
(conservative behavior, exchangeability-preserving).

**Nit-3** — Log `log.info` when `vol_bucket` has only one unique
value ("K = 1; no stratification").

**Nit-4** — `predict_band` docstring: note that CQR vs vanilla is a
*representational* choice (heteroscedastic vs homoscedastic), not
just a calibration knob.

**Nit-5** — `predict_band`'s Python loop is fine at 61.5K rows/day.
If §3.7 profiles a hot spot, vectorize per the template above.

**Nit-6** — Stylistic: `cells.get(b, fallback)` is the idiom for the
`cells.get(b) or fallback` pattern; the `or` form is currently safe
(dataclass truthy by default) but fragile to future refactors.

**Nit-7** — `load_cells` should defensively skip filenames that
don't have at least one underscore (defensive against `glob` pattern
escapes).

**Nit-8** — Add `test_calibrate_per_cell_empty_input_returns_empty_dict`.

**Nit-9** — `coverage_in_sample` should be surfaced in the §3.7
walk-forward report (operator-facing); no action in §3.6.

**Nit-10** — `compute_cqr_scores` could take a `q_levels` parameter
and assert `q_levels[IDX_Q_LO] + q_levels[IDX_Q_HI] = 1.0` (symmetry)
and `q_levels[IDX_Q_MED] = 0.5`. Catches future-proofing breakage.

**Nit-11** — `alpha = 0.2` default is fine; document the constant
in the module-level docstring as "matches PHASE3.md §1.3's 80% band".

---

## Final disposition and summary

**Disposition: Approve with minor changes (0 Critical, 2 Major).**

The conformal module is a clean, freestanding ~310-LOC implementation
that correctly executes the amendment-9-per-cell CQR-vs-vanilla
picker. Score formulas match Romano-Patterson-Candès 2019 Eq. 6
(modulo a defensive `_EPS` whose bias is operationally zero); the
empirical quantile matches Eq. 1's `⌈(n+1)(1-α)⌉` order statistic;
the picker correctly selects whichever score function's in-sample
coverage is closest to `1-α`; persistence round-trips bit-for-bit
across positive and negative bucket ids. Test coverage of the
implemented surface is broad (17 tests across score formulas, picker,
pooling, predict_band, coverage, persistence).

The two Major findings are both contract / spec-conformance gaps,
not algorithmic bugs:

1. **Major-1** — `min_cell_size = 100` default is 10× below
   PHASE3.md §6.4 + §7.3's 1000 threshold, and the pooling rule
   (`pool-small-cells-into-fallback`) diverges from the spec's
   `K=5 → K=3 per horizon` fallback. Both default and rule need to
   align with the spec, OR the spec needs to be amended to match
   the implementation.

2. **Major-2** — Quantile-crossing (`q_lo > q_med` or
   `q_med > q_hi`) silently produces inverted-sign CQR scores.
   Pinball loss doesn't enforce monotonicity; this WILL happen on
   small cells in §3.7 production. Add log+count detection at
   minimum; consider quantile rearrangement as the stronger fix.

Minor findings cluster around:
- Picker tiebreaker doesn't penalize wider bands (Minor-1).
- Test coverage gaps: negative `s_star`, small-N empirical quantile,
  picker discrimination on heteroscedastic vs miscalibrated synthetic
  data, quantile crossing (Minors 2/3/6/7).
- JSON schema lacks version field (Minor-5).
- Consumer contract for §3.7 not narrated in module docstring
  (Minor-8).
- Vol-bucket invariant not asserted (Minor-9).
- Picker noise on small cells should be logged for operator
  transparency (Minor-4).

§3.6 cannot be approved with the `min_cell_size = 100` default
shipping to §3.7's walk-forward CLI — Major-1 must be addressed
before §3.7 starts. Major-2 is also pre-§3.7 work; the §3.7 reviewer
will catch missing crossing-detection as a Critical when a real
30K-step posttrain produces a crossing row.

**The single most important change:** fix Major-1. Set
`min_cell_size = 1000` to match PHASE3.md §6.4 + §7.3, upgrade the
pooling-info log to a `log.warning`, and decide (with a spec amendment
if needed) whether the implementation's "pool-small-cells-into-`-1`"
behavior is the new contract or whether to implement the spec's
"K=5 → K=3 per horizon" fallback. Either way, the §3.7 walk-forward
CLI's behavior on real cell sizes depends on this default, and the
acceptance gate §1.3 (out-of-sample coverage in [0.75, 0.85]) is
sensitive to whether small cells are pooled or fall back to a
coarser stratification.

After Major-1 + Major-2 are addressed and the test suite picks up the
quantile-crossing and small-N regression cases, §3.6 is a clean
approval. Minor findings can be batched into a follow-up commit
before §3.7's CLI lands or rolled into §3.7's own PR.
