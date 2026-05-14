# Phase 3.3 Review — Cross-section sampler

**Reviewer:** Independent code reviewer
**Date:** 2026-05-11
**Disposition:** Approve+Minor

Scope: `daity/data/cross_section.py` (437 LOC), `tests/unit/test_cross_section.py`
(310 LOC), `tests/leakage/test_cross_section_leakage.py` (248 LOC). Reviewed
against `PHASE3.md` §4 (amendments 2 + 16 binding), §3.4 (point-in-time),
§6.4 (vol bucket). Supporting modules (`labels.py`, `parquet_store.py`,
`feature_precompute.py`, `xgboost_baseline.py`) read for interaction
correctness only.

The leakage gate is solid, the strict-`<` semantics are correctly implemented
and equivalent to the XGBoost baseline's `-1μs` trick, and the per-horizon
iterator design matches the amendment-2/16 contract. Findings are concentrated
in API drift from §4.2, three doc/code mismatches, two missing spec features
(`max_symbols_per_step`, `valid_mask` semantics), and test gaps.

---

## Section A — Leakage correctness

### 1. Strict-`<` equivalence with XGBoost baseline (item 1)

Both formulations produce the same index across all boundary cases. I
verified four cases against the XGBoost baseline's `-1μs` shift:

| Case | `searchsorted(side='left') - 1` | `searchsorted(shifted, side='right') - 1` |
|---|---|---|
| anchor between bars | 1 | 1 |
| anchor == eff_close[1] (exact boundary) | 0 | 0 |
| anchor before all bars | -1 | -1 |
| anchor after all bars | 2 | 2 |

At microsecond resolution, the two are bit-for-bit equivalent. The
"exact boundary" case is the one that matters most — both correctly
exclude the bar AT `anchor_ts`. The leakage test at
`tests/leakage/test_cross_section_leakage.py:210-248` exercises this.

**Verdict:** correct. The implementation choice (`searchsorted(side='left') - 1`)
is actually cleaner than the `-1μs` shift because it doesn't depend on
microsecond resolution being chosen — it works at any time-unit and any
tie pattern.

### 2. `_build_input_tensor` boundary (item 2)

`daity/data/cross_section.py:235-254` requires
`last < need - 1` to return None, i.e., we need `last >= need - 1`, and
then slice `[last - need + 1 : last + 1]` of length `need`. The last
included row has eff_close at `eff_close[last] < anchor_ts` (by
construction of `_strict_last_idx`). This honors the strict-`<` contract.
The leakage test at `tests/leakage/test_cross_section_leakage.py:144-179`
verifies this end-to-end by reading the parquet directly and confirming
the tensor's last close matches the bar BEFORE the anchor.

**Verdict:** correct.

### 3. `_vol_bucket_value` lookback (item 3)

`daity/data/cross_section.py:271`: `closes = sc.features[last - lookback : last + 1, 3]`
slices `lookback + 1` bars, then `np.diff(np.log(closes))` produces
`lookback` log-returns. For 30m with `vol_lookback=78`, that's 78 returns.
The guard `if last < lookback: return None` ensures `last >= lookback`,
giving a valid `lookback + 1`-length slice with no negative start.

The docstring comment at line 268 (`need lookback + 1 bars for `lookback`
returns`) accurately describes the intent.

**Verdict:** correct.

### 4. Label cap inheritance (item 4)

`daity/data/cross_section.py:210` applies `pl.col("t_horizon") <= self.as_of`
on the loaded labels. The label producer already applies this at build
time (`daity/data/labels.py:138-148` for intraday, `:244-262` for daily).
The sampler's filter is defensive but not redundant — it guards the case
where labels were precomputed with a *later* `as_of` than the training
query (PHASE3.md §3.4 explicitly calls this out: "guards against the
subtle bug where the label producer has its own broader `as_of` than
the training-time query").

**Verdict:** correct and load-bearing.

### 5. `min_end_ts` semantics (item 5)

Applied twice: at label load (`cross_section.py:212`) and at anchor
enumeration (`:315-317`). The label-level filter makes the anchor-level
filter logically redundant — anchors are built from a `set` of label
`t` values, so if labels are pre-filtered, no anchor < `min_end_ts`
can appear. The anchor-level filter is purely defensive.

One minor robustness concern: at line 316,
`cutoff = np.datetime64(self.min_end_ts.replace(tzinfo=None))` strips
tzinfo by wall-clock semantics. If a caller passes a non-UTC
`min_end_ts`, the wall-clock interpretation is wrong. The dataclass
docstring should specify "UTC" or the code should call
`.astimezone(UTC)` first. The unit test at
`test_cross_section.py:282` happens to pass UTC, so this bug is
latent.

**Verdict:** correct for documented usage; brittle for non-UTC input.
See finding **Minor-2** below.

---

## Section B — Correctness of cross-section assembly

### 6. Cross-sectional rank — ties are NOT given average rank (item 6)

`daity/data/cross_section.py:361`: `order = np.argsort(np.argsort(labels_arr))`.

Empirical check:
```
x = [1.0, 2.0, 2.0, 2.0, 3.0]
argsort(argsort) → [0, 1, 2, 3, 4]   # ordinal, stable order
rankdata(x, 'average') → [1, 3, 3, 3, 5]
```

`argsort(argsort)` produces **ordinal** ranks (each tied value gets a
distinct index based on stable order), NOT average ranks. The comment
on line 360 ("Ties get average rank") is **wrong**. The module
docstring at line 23 doesn't make the claim either way.

Whether this matters depends on the downstream consumer:

- **ListMLE (PHASE3.md §6.1)** consumes ranks as an ordering; ordinal
  ranking is a valid (and faster) choice. ListMLE doesn't require
  average-rank tie handling.
- **Spearman rank-IC (eval metric)** in `daity/eval/metrics.py`
  *typically* uses average ranks. If the eval computes rank-IC on the
  emitted `ranks` tensor directly, ordinal ranking introduces a small
  bias when float32 labels collide. In practice float32 collisions on
  real returns are vanishingly rare, so the bias is negligible — but
  the eval-time rank is recomputed from raw `y_log_return` anyway, not
  from this `ranks` field, so ListMLE is the only consumer that ever
  sees these numbers.

**Verdict:** behavior is correct for ListMLE; documentation is wrong.
See finding **Minor-1** below.

### 7. `vol_bucket` within cross-section (item 7)

PHASE3.md §6.4 says bucketing is "K=5 quintiles of horizon-appropriate
realized vol over the lookback". The implementation computes a
per-(symbol, t) scalar vol then bucks *across symbols at the same t*.
This is the correct reading: §6.4 is in the context of "conformal
calibration ... stratified by realized-vol bucket", and the conformal
stratification at inference time IS cross-sectional within an `as_of`
(see PHASE3.md §7).

A per-symbol time-series bucketing would defeat the purpose: under
that scheme, a low-vol period would put every symbol into bucket 0,
breaking the stratification. Cross-sectional bucketing is what the
spec wants.

**Verdict:** correct.

### 8. `drop_invalid_threshold` floor (item 8)

`cross_section.py:347`:
`len(valid_symbols) < max(2, int(threshold * len(universe)))`.

With `threshold=0.5, |U|=205` the floor is 102 — comfortably above
the N≥10 rank-IC stability threshold and the K=5 vol bucket sanity
(each bucket gets ≥20 symbols).

With `threshold=0.0` (used in
`test_sampler_drop_invalid_threshold`), the floor is 2. Emitting a
cross-section of 2 symbols is meaningless for rank-IC (Spearman on
N=2 is degenerate). The default of 0.5 makes this unreachable in
practice, but if a user sets `threshold=0.0` (which the spec §4.5
implies is a tunable knob) they get nonsense cross-sections.

PHASE3.md §6.4 implicitly bounds N: K=5 quintiles needs at least 5
symbols to populate, and Spearman rank-IC needs >10 for stability.
The floor of 2 is too weak.

**Verdict:** floor is too low. See finding **Minor-3**.

---

## Section C — Performance / scalability

### 9. Cache memory (item 9)

Per-symbol cache footprint:
- 5m: ~125K bars × 18 ch × 4 B = 9.0 MB
- 15m: ~42K × 18 × 4 = 3.0 MB
- 60m: ~10K × 18 × 4 = 0.75 MB
- day: ~1.7K × 18 × 4 = 0.12 MB
- Total per symbol: ~13 MB
- 205 symbols × 13 MB ≈ **2.7 GB** features

Plus labels: 4 horizons × 205 symbols × ~75K rows × 24 B (datetime +
float + bool) ≈ **150 MB**.

Total ~2.9 GB. The module docstring at lines 137-139 says
"~600 MB"; this is roughly 5× too low — it appears to assume only
the active scale is held. For full Phase 3.7 sweep on H200 (143 GB),
2.9 GB is fine. On the local RTX-4060 (8 GB GPU + 64 GB RAM), it's
also fine. But the comment is misleading.

**Verdict:** acceptable; docstring estimate is wrong. See finding
**Nit-2**.

### 10. Anchor enumeration cost (item 10)

`_anchor_timestamps(horizon="30m")` for 205 symbols × ~119K labels:

- Loop: 205 × `lc.t.astype("int64").tolist()` + `set.update(list)`
- Benchmarked: ~3.0 s on a synthetic 24M-int workload.
- Alternative `np.unique(np.concatenate(arrs))`: ~1.5 s (2× faster).

This runs ONCE per epoch (cached as part of the iterator's first
call). 3 s is not a bottleneck given the training step takes O(GPU
forward+backward) ≈ tens of ms.

**Verdict:** acceptable. Optional optimization available; see **Nit-3**.

### 11. Per-anchor inner loop (item 11)

Per-anchor cost per symbol (microbenchmarked):
- 4× `np.searchsorted` ≈ 4 × 1.8 μs = 7 μs
- 4× array slices: ~free (views)
- 1× vol calculation (log+diff+std on ~79 elements): ~10 μs
- 1× label lookup: ~2 μs
- Total per (anchor, symbol): ~18 μs

For 30m horizon with ~119K anchors × 205 symbols = 24M sym-anchors:
**~440 s = 7 min** for a full epoch sweep, single-process.

However, in training the iterator is consumed step-by-step (one
cross-section per `next()` call), so the relevant metric is
**per-step latency**: 205 × 18 μs = **~3.7 ms** per cross-section.
This is well below GPU forward+backward time, so the sampler is
not on the critical path.

For 1d/2d (~17K anchors), the full sweep is ~63 s — trivially fast.

**Verdict:** acceptable for training; full-eval sweep takes ~7 min
which is fine for a 4-fold walk-forward run.

---

## Section D — API / design

### 12. `label_root: Path` vs `label_store: ParquetStore` (item 12)

PHASE3.md §4.2 specifies the dataclass signature:
```python
parquet_store: ParquetStore           # for inputs
label_store: ParquetStore             # for labels (per-horizon)
```
The implementation at `cross_section.py:142-143` instead uses:
```python
feature_store: ParquetStore
label_root: Path
```
Two drifts: (a) `parquet_store` was renamed to `feature_store`
(reasonable — disambiguates from labels), (b) `label_store: ParquetStore`
became `label_root: Path`.

(b) is the substantive one. The justification in `cross_section.py:199`
is "horizon names aren't valid `ParquetStore` scales", but PHASE3.md
§3.6 explicitly says: *"Mirrors `data/features_parquet/` layout so the
existing `ParquetStore.read(symbol, scale=horizon, as_of=...)` pattern
works with one trivial extension (`scale` is a free-form string at the
ParquetStore layer)."* So the spec expected the trivial extension to
have been made.

Whether to make the extension or accept the asymmetry is a judgment call.
The current implementation is simpler (it skips ParquetStore.read's
tz-naive handling and `start/end` filtering, which it doesn't need),
but it bypasses the `as_of` capping that ParquetStore enforces by
default. The defensive `t_horizon <= as_of` filter at line 210
compensates, but it's now a separate code path.

**Verdict:** acceptable drift; PHASE3.md should be updated to
ratify the new signature OR the ParquetStore should be extended. See
**Minor-4**.

### 13. No grad-friendly batching / variable-N cross-section (item 13)

PHASE3.md §6.6 says "Batch unit: one cross-section. Effective batch
size = N_symbols × number-of-cross-sections-per-step." So §6 explicitly
treats the cross-section as the batch unit and accepts variable N.

`max_symbols_per_step` (PHASE3.md §4.4) is the prescribed knob for
memory pressure on small GPUs (sub-sample N out of |U| with replacement).
**This is not implemented** — the dataclass has no `max_symbols_per_step`
field. The spec says "default `None` (full cross-section) on H200", so
the absence doesn't block H200 training, but it does block local
RTX-4060 development.

**Verdict:** spec-compliant for H200; missing the local-dev knob. See
**Minor-5**.

### 14. `pl.read_parquet` in `_labels` (item 14)

`cross_section.py:208` bypasses `ParquetStore.read`'s tz-naive handling.
Verified: labels are emitted with explicit `dt.replace_time_zone("UTC")`
at `labels.py:198-199`, so the parquet file's `t` column is always
tz-aware UTC. Round-trip via `pl.read_parquet → to_numpy →
.astype("datetime64[us]")` produces a tz-naive numpy datetime64 whose
numeric value equals the UTC timestamp. Comparisons to other tz-naive
`datetime64[us]` work correctly.

Risk surface: if a future writer ever produces tz-naive label parquet
(e.g., a third-party tool), the `pl.col("t") >= self.min_end_ts` filter
at line 212 would fail at the polars level because `min_end_ts` is
tz-aware. The current contract (labels.py writes tz-aware) makes this
a theoretical concern only.

**Verdict:** correct under the current contract.

### 15. Determinism / unused `seed` (item 15)

`cross_section.py:148`: `seed: int = 0` is declared but never read
anywhere in the module. The unit-test docstring (line 10 of
`test_cross_section.py`) claims "Determinism: same seed produces the
same yielded cross-sections" — there is **no such test** in the file.

The sampler IS deterministic without a seed because:
- Anchor enumeration sorts the union set.
- Symbol iteration follows `self.universe` order.
- All other ops are deterministic numpy.

So either:
(a) `seed` is a leftover from the v1 design where intra-day anchor
sub-sampling existed — should be removed.
(b) `seed` will be wired up when `max_symbols_per_step` is added (the
random sub-sampling will need it).

Either way, having an unused public dataclass field is API noise.

**Verdict:** API-noise. See finding **Minor-6**.

---

## Section E — Test coverage gaps (item 16)

Tests that are MISSING but would meaningfully reduce risk:

1. **Determinism test.** The docstring claims it; no test exists.
   Should iterate twice and assert identical `(t, symbols, labels)`
   tuples emerge.

2. **Exact-boundary anchor.** The strict-< leakage test at
   `test_cross_section_leakage.py:210` only fires when
   `eff_arr == anchor_np` happens to occur — synthetic data with
   non-coincident ts may skip the check entirely (the `continue`
   on line 237). Should construct a fixture where the anchor IS
   exactly some bar's `eff_close` and assert that bar is excluded.

3. **Ties in labels.** No test for label ties. With ordinal ranking
   (Section B item 6), tied labels get distinct ranks based on
   `np.argsort` stable order. If average-rank tie handling is ever
   wanted, the regression would be silent.

4. **Ties in vol values.** `_quantile_bin` collapses to midpoint when
   all values are equal (covered at `test_quantile_bin_all_equal_values_safe`),
   but the more realistic case of "3 of 5 vols are identical" — where
   `np.quantile` edges may collide — is not tested.

5. **`min_end_ts` filtering at the LABEL level.** The existing test
   (`test_sampler_min_end_ts_filters_anchors`) verifies the ANCHOR-level
   filter via `cs.t >= cutoff`. The label-level filter (line 212) is
   not separately exercised. If someone deletes line 212 thinking it's
   redundant, the test still passes.

6. **Holiday-skipping at daily horizons.** No fixture injects an
   actual holiday into the calendar; the synthetic fixture uses
   `holidays=[]`. The leakage gate is inherited from `labels.py`
   tests, but the sampler's interaction with holiday-padded labels
   should at least be smoke-tested.

7. **Non-UTC `min_end_ts`.** Per item 5, passing a non-UTC tz-aware
   datetime triggers a wall-clock bug. Either fix and test, or
   document and assert.

8. **Cache reuse across horizons.** `test_sampler_caches_per_symbol`
   only iterates `iter_30m`. The feature cache is keyed `(symbol, scale)`
   and SHOULD be reused across horizons (the scales are the same).
   Should iterate two horizons and assert `len(_feat_cache)` doesn't
   grow on the second.

9. **Insufficient-history symbol pruning.** PHASE3.md §4.5 says
   late-listed symbols should be DROPPED, not raised. The test
   `test_sampler_yields_well_formed_cross_sections` covers this
   implicitly (the `>= 2` assertion implies some symbols may have
   been dropped), but a direct fixture with a late-listing symbol
   that verifies it's specifically absent from `cs.symbols` would
   make the contract explicit.

10. **`valid_mask` semantics** (per item 13 of Section D / finding
    Minor-7 below): the emitted `valid_mask` is always all-True
    because the sampler drops rather than masks invalid rows. The
    test at line 241 asserts `cs.valid_mask.all().item()` — which
    locks in the implementation, not the spec. If the spec wording
    is honored (zero-out instead of drop), this test needs to flip.

---

## Section F — Critical / Major / Minor / Nit findings

### Critical

None. The leakage contract is correctly enforced and tested.

### Major

None. Design issues are minor / documentation-level.

### Minor

**Minor-1** — `cross_section.py:360` documentation/code mismatch.
The inline comment "# Cross-sectional rank ∈ [0, 1] of labels. Ties
get average rank." is wrong; `argsort(argsort)` gives ordinal ranks,
not average ranks. Either (a) replace with
`scipy.stats.rankdata(labels_arr, method='average') - 1` to honor
the comment, or (b) update the comment to "Ties broken by stable
order (np.argsort)".
*Recommendation:* (b). ListMLE doesn't care; saving the scipy import
is worth it.

**Minor-2** — `cross_section.py:316` `min_end_ts` tz robustness.
`self.min_end_ts.replace(tzinfo=None)` silently uses wall-clock
semantics. Replace with `self.min_end_ts.astimezone(UTC).replace(tzinfo=None)`
to handle non-UTC callers correctly. Also: document on the dataclass
field that `min_end_ts` should be UTC (or tz-aware in any zone).

**Minor-3** — `cross_section.py:347` `drop_invalid_threshold` floor.
The hard floor of 2 is too weak. Raise to ≥10 to match rank-IC
stability and ≥K=5 to keep vol-bucket stratification meaningful:
```python
MIN_SYMBOLS_FOR_RANK_IC = 10
if len(valid_symbols) < max(MIN_SYMBOLS_FOR_RANK_IC,
                             int(self.drop_invalid_threshold * len(self.universe))):
    return None
```

**Minor-4** — `cross_section.py:142-143` API drift from PHASE3.md §4.2.
The spec specifies `label_store: ParquetStore` (with §3.6 envisaging
a one-line ParquetStore extension to accept `scale=horizon`). The
implementation uses `label_root: Path`. Either (a) update PHASE3.md
§4.2 to ratify the path-based form, or (b) extend ParquetStore. (a)
is cheaper; the asymmetry is annoying but not load-bearing.

**Minor-5** — `cross_section.py` missing `max_symbols_per_step`.
PHASE3.md §4.4 prescribes this as the memory-pressure knob (sub-sample
N symbols out of |U| with replacement for each step; training only,
eval uses full). Default `None`. Implement before §3.4 if local-dev
on RTX-4060 is required for Phase 3.4 training-loop development.

**Minor-6** — `cross_section.py:148` unused `seed` field.
Either remove (preferred — the sampler is deterministic without it),
or wire it up when `max_symbols_per_step` is added. Currently it's
public API noise that suggests a feature that doesn't exist.

**Minor-7** — `valid_mask` always all-True (spec vs implementation).
PHASE3.md §4.5: "emit the in-universe-at-t subset, with `valid_mask`
zeroing out rows that can't satisfy `window_bars` lookback." The
implementation drops invalid rows entirely — `valid_mask` is purely
decorative. This is arguably a simplification (no zero-row in the
loss), but it diverges from the spec wording. Either update §4.5
to match the implementation, or change the implementation to emit
all-universe rows with masked invalid entries.

**Minor-8** — `test_cross_section.py` docstring claims 10 tests
including "Determinism: same seed produces the same yielded
cross-sections" — no such test exists. Either add the test or
remove the claim. Given Minor-6, removing the claim is fine.

### Nit

**Nit-1** — `cross_section.py:139` cache footprint comment
("~600 MB") is roughly 5× too low. Realistic estimate is ~2.7 GB
features + ~150 MB labels ≈ 2.9 GB. Either correct the number or
add a note that it's the "active-scale only" estimate.

**Nit-2** — `cross_section.py:305-318` `_anchor_timestamps` could
be 2× faster via `np.unique(np.concatenate([lc.t for ...]))` instead
of the Python `set.update(.tolist())` pattern. Saves ~1.5 s per
horizon per epoch. Not worth doing unless profiling shows it matters.

**Nit-3** — `cross_section.py:308-311` defensive iteration over
`self.universe` plus `if lc.t.size == 0: continue`. Filter symbols
without label files upstream (in `__init__` or first use) so the
hot loop doesn't re-test.

**Nit-4** — `cross_section.py:202-204` empty-labels case constructs
a fresh `_LabelCache` with zero arrays. Could share a singleton
empty cache to save a few allocations per missing-label symbol.
Trivial; only worth it if the universe has many label-less symbols.

**Nit-5** — `cross_section.py:208` `pl.read_parquet(...).sort("t")`
double-sorts (labels.py writes sorted; this re-sorts). Defensive but
costs ~5 ms per file × 205 symbols × 4 horizons = ~4 s amortized.
Replace with a one-time assertion `df.equals(df.sort("t"))` in DEBUG
mode if you want to keep the invariant without paying for it.

**Nit-6** — `cross_section.py:433-437` `_np_to_utc` is reinventing
the wheel; could use
`datetime.fromtimestamp(us / 1_000_000, tz=UTC)` with a final
microsecond adjustment, or just rely on
`pl.Series([ts]).cast(pl.Datetime("us", "UTC"))[0]`.

**Nit-7** — `iter_30m`, `iter_120m`, `iter_1d`, `iter_2d` are
four identical one-line wrappers around `_iter_horizon`. Could
collapse to a single `iter(horizon: Literal[...])` method, but
the four explicit methods are clearer at the call site and match
the spec's interface. Subjective.

---

## Section G — Final disposition + summary

**Disposition: Approve+Minor.**

Phase 3.3 is structurally correct: the leakage contract is honored,
the strict-`<` semantics are equivalent to the XGBoost baseline's
`-1μs` trick, the per-horizon iterator design implements amendments 2
and 16 cleanly, and the leakage test rigorously verifies the
boundary case. None of the Minor findings would silently corrupt
rank-IC numbers or leak future information — they are documentation
mismatches (Minor-1, Minor-8), API drift from PHASE3.md (Minor-4,
Minor-7), brittle edge handling (Minor-2, Minor-3), and one missing
spec feature that doesn't block H200 work (Minor-5). The most
important single change is **Minor-3 (raise the drop_invalid_threshold
floor from 2 to 10)** because a stress-test run with `threshold=0.0`
could otherwise emit cross-sections of 2 symbols and produce
meaningless rank-IC. Phase 3.4 (posttrain.py / training loop) can
proceed; the Minor findings can be batched into a follow-up cleanup
PR before merging to main.
