# Phase 3.7 Review — Walk-forward sweep aggregator + tri-criterion + acceptance gate + CLI

**Reviewer:** Independent code reviewer
**Date:** 2026-05-11
**Disposition:** Approve with major changes (0 Critical, 4 Major, 8 Minor, 6 Nit)

Scope: `daity/eval/walk_forward.py` (~380 LOC),
`daity/scripts/walk_forward_cli.py` (~200 LOC), and
`tests/unit/test_walk_forward.py` (18 tests across tri-criterion edge
cases, random-init best-of-seeds, the ≥2-of-4-horizons rule, conformal
in/out-of-band, JSON emit + schema, and a threshold magic-number lock).
Cross-referenced PHASE3.md §1 + §8 (amendments 8 + 16 are most
binding), §6.10 (provenance triple), §11 (compute footnote).

The aggregator is a small, freestanding decision-rule engine in the
spirit the spec calls for: the §1 acceptance gate compresses
4 horizons × 4 folds × (SSL, XGB, random_init × 3 seeds) +
out-of-sample coverage into a single yes/no answer, and the
implementation reads cleanly — tri-criterion is one function per
gate, random-init best-of-seeds is one `max(...)` per fold and one
mean across folds, the CLI is intentionally a thin orchestrator.
Test coverage of the implemented surface is broad and the
threshold-locking test (`test_thresholds_are_locked`) is the right
defensive ratchet for §1's magic numbers.

The four Major findings are split between contract / spec-conformance
gaps (Major-1 / Major-2 / Major-3) and one design footgun (Major-4):

1. **The `folds` key in the output JSON does NOT carry the 16
   per-(fold, horizon) result rows the §8.4 schema mandates** — the
   CLI populates it with `_default_folds()` (4 fold time-window
   metadata dicts).
2. **`provenance` drops `config_hash` and `as_of`** from the
   PHASE3.md §6.10 triple — the artifact violates the
   `(git_sha, config_hash, as_of)` reproducibility contract.
3. **`daity.eval.walk_forward` transitively imports `torch`** via
   `from daity.training.posttrain import HORIZONS` — a pure-Python
   aggregator that can't be loaded without torch is a fragility
   the §3.8 acceptance step will trip on if run from a CPU-only box.
4. **CLI exits `0` on gate 1 only** — Phase 3 acceptance requires
   gates 1 + 3 + 4 all pass; an exit code of 0 on gate-1-pass with
   gate-3 or gate-4 failed is silently wrong.

Minor and Nit findings cluster around fold-mismatch handling, missing
test coverage for the `_load_fhr_json` wrapper path, conformal gate
semantics (per-horizon vs all-horizons), the `RandomInitGateResult`
discarding seed-by-seed and fold-by-fold detail that the failure-mode
budget needs, and the `n_random_seeds` CLI flag being decorative.

---

## Section A — Tri-criterion correctness

### A.1 Mean delta threshold (scrutiny 1) — `>=` matches the spec ✓

PHASE3.md §1.1 says "beats XGBoost mean by ≥0.005". The
implementation at `walk_forward.py:124` is:

```python
passed_a = mean_delta >= delta_threshold
```

`test_tri_criterion_mean_delta_at_threshold` at line 102-108 exercises
`mean_delta == 0.005` exactly (SSL all 0.005, XGB all 0.000) and
asserts `passed_mean_delta`. ✓

**Note** (file as Nit-1): floating-point summation of four 0.005
values then subtracting four 0.000 values is exact in IEEE-754
binary64 (0.005 isn't exact, but the same rounding applies to both
sides and cancels). The test's `abs(res.mean_delta - 0.005) < 1e-9`
tolerance is sound, but a pathological input where SSL = `(0.0049999,
0.0050001, 0.005, 0.005)` would land at the boundary with the inexact
mean being either `0.005` ± a few ULPs. The current `>= 0.005`
inclusive check is the right semantic — calling out so a future
reader doesn't tighten to `>` and break the at-threshold case.

### A.2 4-folds × 3-of-4 boundary (scrutiny 2) — test does exercise N=4 with one negative ✓

`test_tri_criterion_3_of_4_positive_passes_b_and_c` at lines 67-85
uses four folds with SSL = `(0.020, 0.015, 0.010, -0.005)` and
XGB all zero. `folds_ssl_positive == 3`, `folds_ssl_beat_xgb == 3`,
both pass `≥ MIN_FOLDS_POSITIVE` and `≥ MIN_FOLDS_BEAT_XGB`. ✓

But there's a subtle test-coverage gap (Minor-1): the test asserts
`passed_folds_positive` and `passed_folds_beat_xgb` but **does NOT
assert `res.passed`**. With `mean_delta = (0.02 + 0.015 + 0.01 -
0.005) / 4 = 0.01`, criterion (a) passes too, so `res.passed` should
be `True`. Add the assert. Trivially defensive — if a future refactor
swaps `or` for `and` at `walk_forward.py:134`, this test wouldn't
catch it because the per-criterion flags are still correct
individually.

### A.3 Per-fold delta sign — ties NOT counted as beating (scrutiny 3)

At `walk_forward.py:122`:

```python
folds_beat = sum(1 for d in deltas if d > 0)
```

A fold where SSL == XGB (delta = 0) is NOT counted as beating.
PHASE3.md §1.1 says "≥3 of 4 folds individually beat XGBoost"; "beat"
is naturally strict. The implementation matches the natural reading.
✓

That said: tied folds are very rare in practice (real rank-IC numbers
are noisy floats at the 4th decimal). The semantic question — should
a tie count as "neither beat nor lost"? — is decided correctly by
strict `>` here. Architect-review §A.7's intent ("monotonicity check")
is to penalize folds where SSL underperforms, and ties don't help the
monotonicity argument.

**Note** (Nit-2): for symmetry, `folds_positive` at line 121 uses
`> 0` (strict), and `passed_mean_delta` at line 124 uses `>=
delta_threshold` (inclusive). The asymmetry is correct (per-fold sign
is a strict comparison, but the mean-delta threshold is an inclusive
margin), but worth a sentence in the function's docstring so a future
reader doesn't "normalize" them.

### A.4 Fold-pairing logic — silently drops mismatched IDs (scrutiny 4)

At `walk_forward.py:103`:

```python
fold_ids = sorted(set(ssl_by_fold) & set(xgb_by_fold))
```

If `ssl_by_fold` has `{F-9, F-10, F-11, F-12}` and `xgb_by_fold` has
`{F-9, F-10, F-11}` (one XGB fold missing — e.g., a crashed XGB run),
the code silently evaluates the tri-criterion on 3 folds. `n_folds`
in the result becomes 3, and `MIN_FOLDS_POSITIVE = 3` against a
3-fold population is a **different criterion** — "all 3 must be
positive" rather than "3 of 4 positive". This is **Major-3**.

The walk-forward `phase3_gate_1_passed` becomes harder to satisfy,
not easier, but the user has no way to know the gate was evaluated
on the wrong population. The expected behavior is to **raise** when
fold IDs mismatch and force the operator to either re-run the
missing fold or explicitly accept a smaller-N evaluation via a
config flag.

**Recommendation:** raise `ValueError` if
`set(ssl_by_fold) != set(xgb_by_fold)` (and likewise for
random-init). Add an `allow_partial: bool = False` parameter for
the rare debugging case. See **Major-3**.

### A.5 Single-fold-dominance pathology (scrutiny 5) — test is faithful ✓

`test_tri_criterion_lucky_fold_dominates_mean_but_fails_monotonicity`
at lines 119-138 uses SSL = `(0.085, -0.001, -0.002, -0.002)` against
XGB = `(0, 0, 0, 0)`. Mean delta = 0.02 (passes (a)) but only 1 of 4
folds positive (fails (b) and (c)). The test's docstring explicitly
calls out "the architect-review §A.7 single-fold-dominance pathology
the tri-criterion was DESIGNED to catch" — the right level of
intentionality.

The test also docs "this is exactly what amendment 8 was designed to
prevent" — the intent ratchet is in place. ✓

(Nit-3): the `RANK_IC_DELTA_THRESHOLD` import is unused in this test
file but the test_thresholds_are_locked test imports it via the
top-level `from daity.eval.walk_forward import (...)`. Fine; just
noting that the constant is exercised once.

---

## Section B — §1.4 random-init gate

### B.1 Best-of-seeds per fold, then mean (scrutiny 6) — implementation matches spec ✓

`evaluate_random_init_gate` at `walk_forward.py:157-186`:

```python
rand_by_fold[r.fold_id].append(r.rank_ic_mean)
...
rand_ics = [max(rand_by_fold[f]) for f in fold_ids]
mean_rand = sum(rand_ics) / len(fold_ids)
```

This is the correct order of operations per PHASE3.md amendment 3
("3 random seeds; take best as comparison"): **best across seeds per
fold**, then **mean across folds**. The alternative (best overall,
then mean) would mix folds' seed identities — a subtle bias because
the same seed has IC variance across folds.

`test_random_init_takes_best_of_seeds` at line 150 sets up 3 seeds
× 4 folds with seed-2 winning each fold (0.005, 0.008, 0.009 per fold)
and asserts the per-fold best of 0.009. SSL mean = 0.015, delta =
0.006 → passes. ✓

(Minor-2): the test exercises the "best is the same seed at every
fold" case (seed 2 wins everywhere). A more discriminative test
would have **seed 0 winning at fold F-9, seed 1 at F-10, seed 2 at
F-11/F-12** — that's the test that catches a bug where someone
takes `max([r.rank_ic_mean for r in random_init_results])` (best
overall) instead of `max(rand_by_fold[f])` (best per fold). The
current test passes under both implementations.

### B.2 Missing folds in random_init (scrutiny 7)

Same problem as A.4: if random-init has only 2 of 4 folds completed
(e.g., one seed × fold combo crashed), `fold_ids =
sorted(set(rand_by_fold) & set(ssl_by_fold))` silently shrinks to 2.
The mean over fewer folds is skewed and the gate evaluation is
silently on the wrong population. Same fix as Major-3.

### B.3 §1.4 wording "≥2 of 4 horizons beat random-init" (scrutiny 8) — matches ✓

`evaluate_walk_forward` at line 232:

```python
rand_passed = sum(1 for h in hs if rand[h].passed)
return WalkForwardOutcome(
    ...
    phase3_gate_4_passed=rand_passed >= min_horizons_passed,
)
```

`min_horizons_passed` defaults to `MIN_HORIZONS_PASSED = 2`. ✓

**But there's a subtle PHASE3.md §1.4 spec wrinkle the code does
NOT enforce**: §1.4 says "The set of horizons satisfying §1.4 must
overlap with the set satisfying §1.1's tri-criterion — i.e., 'the
SSL backbone helps at the same horizons it produces alpha at'." The
current implementation evaluates §1.4 independently — a configuration
where 30m + 120m pass §1.1 but 1d + 2d pass §1.4 would set
`phase3_gate_1_passed = True` and `phase3_gate_4_passed = True`,
even though no single horizon both produces alpha AND clears
random-init. This is **Major-4**.

The fix: `phase3_gate_4_passed` should require ≥`min_horizons_passed`
in `set(horizons_passing_§1.1) ∩ set(horizons_passing_§1.4)`, not
just `set(horizons_passing_§1.4)`. The intersection semantic is what
the spec text says.

(Recommendation: also surface `horizons_passing_random_init` and
`horizons_passing_both` in the output JSON for operator clarity.)

---

## Section C — Conformal §1.3 gate

### C.1 `[0.75, 0.85]` bounds — both inclusive (scrutiny 9) ✓

At `walk_forward.py:275`:

```python
in_acceptance_band=(lower_bound <= mean_cov <= upper_bound),
```

PHASE3.md §1.3 says "in `[0.75, 0.85]`" which is naturally inclusive
of both endpoints. Code matches. ✓

(Nit-4): no test exercises the exact-boundary cases (`mean_cov ==
0.75` or `mean_cov == 0.85`). Add
`test_conformal_gate_exact_boundary_passes` for symmetry with the
mean-delta-at-threshold test.

### C.2 Only horizons passing §1.1 are gated (scrutiny 10) — implementation matches spec ✓

At `walk_forward.py:350`:

```python
for h in outcome.horizons_passed:
    conformal_gates[h] = evaluate_conformal_gate(...)
```

PHASE3.md §1.3 says "Horizons that fail (1) are not subject to (3)
— calibrating bands for a head that doesn't have alpha is
meaningless." The code only iterates `outcome.horizons_passed` (i.e.,
horizons that passed the tri-criterion). ✓

**However**, this has a downstream observability gap (Minor-3): the
output `conformal_gates` dict will be **empty** when §1.1 fails (no
horizons pass), and `phase3_gate_3_passed` falls back to `False` via
the `else False` branch at line 356. An operator reading the JSON
might mistakenly interpret "gate 3 failed" as a coverage problem,
when in reality gate 3 was never evaluated because gate 1 failed
upstream. Emit `phase3_gate_3_passed = None` (or a sentinel
`"not_evaluated"`) in that case, or document the convention in the
emit docstring.

### C.3 Gate 3 logic — ALL passing horizons must have in-band coverage (scrutiny 11)

At `walk_forward.py:354-357`:

```python
gate_3 = (
    all(g.in_acceptance_band for g in conformal_gates.values())
    if conformal_gates else False
)
```

PHASE3.md §1.3's literal text: "Empirical coverage of the predicted
80% band on the held-out test fold of each walk-forward fold, ...
averaged across the 4 folds, is in [0.75, 0.85] **per horizon that
passed (1)**." The "per horizon" suggests per-horizon gates, and the
spec at §1.3 doesn't say "≥X of Y horizons" — it says each passing
horizon must individually clear coverage. The implementation's
`all(...)` matches the literal "per horizon that passed (1)" reading.
✓

That said, the symmetry with gates 1+4 (≥2 of 4 horizons) is broken —
a strict reader could argue gate 3 should also be "≥2 of 4 horizons
with passing tri-criterion AND in-band coverage". The current
"every passing horizon must be in band" is stricter, which is
defensible (a passing horizon with bad coverage is worse than no
horizon at all — you'd ship miscalibrated quantiles). I'll mark this
as **Nit-5**: the choice should be documented in the
`emit_walk_forward_json` docstring, and `PHASE3.md §1.3 + failure-mode
budget` should explicitly clarify "ALL passing horizons must clear"
to ratify the implementation's reading.

**Test coverage** (`test_emit_json_gate_3_requires_all_passing_horizons_to_be_in_band`,
lines 285-301): exercises the case where 30m is in band but 120m is
out of band — `phase3_gate_3_passed` is False. Good. ✓

(Minor-4): no test for the case where `conformal_coverage=None`
(the all-horizons-fail-§1.1 case or the operator-forgot-the-flag
case). In both cases, `gate_3 = False` — but the semantics differ.
Add `test_emit_json_gate_3_when_no_horizons_pass_phase1`.

---

## Section D — JSON schema

### D.1 Schema completeness (scrutiny 12)

PHASE3.md §8.4's schema specifies these top-level keys:

| Key | Spec §8.4 | Code |
|---|---|---|
| `provenance` | yes (object with git_sha, config_hash, as_of, parent_checkpoint_hash) | partial — missing `config_hash` + `as_of` |
| `folds` | yes (list of per-(fold, horizon) result rows, 16 entries) | **wrong** — populated with 4 fold-time-window dicts |
| `tri_criterion` | yes (per-horizon) | yes ✓ |
| `horizons_passed` | yes (list) | yes ✓ |
| `n_horizons_passed` | yes | yes ✓ |
| `phase3_gate_1_passed` | yes | yes ✓ |
| `phase3_gate_3_passed` | yes | yes ✓ |
| `phase3_gate_4_passed` | yes | yes ✓ |
| `survivorship_audit` | yes (list of fold audits) | yes ✓ (passthrough) |
| `schema_version` | not in §8.4, sensible add | yes ✓ |
| `generated_at` | not in §8.4, sensible add | yes ✓ |
| `random_init_gates` | not in §8.4, sensible add | yes ✓ |
| `conformal_gates` | not in §8.4, sensible add | yes ✓ |

**Major-1** is the `folds` key mismatch. The spec's `folds` is the
data-bearing portion of the file (the 16 per-(fold, horizon) result
rows with `ssl_heads`, `xgboost`, `random_init` numbers, per-symbol
IC, decile spread, CRPS, coverage_80, etc.). The implementation's
`folds` key is filled from the `fold_metadata` parameter, which the
CLI sets to `_default_folds()` — a 4-entry list of
`{fold_id, train_start, train_end, val_end, test_start, test_end}`.

The downstream consumer (Phase 5's `tearsheet.py`) reads `folds[*]`
expecting per-horizon rank-IC and decile spread; instead it'll see
4 dicts of fold time windows. **This will break Phase 5 plumbing.**

**Recommendation:** `emit_walk_forward_json` should build the
`folds` list from `(ssl_results, xgb_results, random_init_results)`
internally — 16 entries (4 folds × 4 horizons), each with
`{fold_id, horizon, ssl_heads: {rank_ic, ...}, xgboost: {...},
random_init: {...}}`. Keep `fold_metadata` as a SEPARATE key
(e.g., `fold_windows`) if the time-window info is useful (it is —
the §8.1 fold table needs it).

### D.2 Provenance missing config_hash + as_of (scrutiny 13) — Major-2

PHASE3.md §6.10 explicitly extends the
`(git_sha, config_hash, as_of)` triple to Phase 3 artifacts:

> the conformal artifacts (§7.4) and walk-forward artifacts (§8.4)
> are written with a `provenance.json` sidecar carrying the triple
> plus the `parent_checkpoint_hash` (sha256 of the v3_xattn
> checkpoint file). This makes "which checkpoint was Phase 3 run
> from" auditable.

The current implementation at `walk_forward.py:362-365`:

```python
provenance={
    "git_sha": _git_sha(),
    "parent_checkpoint_hash": _checkpoint_hash(parent_checkpoint),
},
```

Misses `config_hash` and `as_of`. Both are load-bearing:
- `config_hash` is the deterministic hash of the Hydra config (the
  smoke-test reproducibility contract — CLAUDE.md "the same
  `(git_sha, config_hash, as_of)` produces the same result").
- `as_of` is the data-cutoff timestamp; a walk-forward run on data
  ingested at different `as_of`s is NOT comparable, and Phase 5 has
  to be able to detect this.

**Recommendation:** add both. `config_hash` is computed from the
serialized Hydra config dict (same hash function as Phase 2 used,
e.g., `hashlib.sha256(json.dumps(cfg.to_container(), sort_keys=True).encode()).hexdigest()[:16]`).
`as_of` is the latest `effective_close` across all input data, or
the CLI's `--as-of` flag (which doesn't currently exist — add it,
default to "now").

### D.3 Schema version bump cadence (scrutiny 14)

`schema_version: 1` is hardcoded at `walk_forward.py:310`. There's
no documented bump cadence — when an amendment 17 adds a field,
the implementer needs to know "bump to 2 and add a loader
back-compat shim".

**Recommendation** (Minor-5): add a module-level docstring section
"Schema version bump policy: bump major when removing/renaming a
required key; bump on additive changes too — Phase 5's loader pins
the version it understands. Each version's diff is documented in
`PHASE3.md §8.4` history." The `schema_version` field is necessary
ratchet for forward compatibility.

---

## Section E — CLI design

### E.1 No inline orchestration — explicit refusal (scrutiny 15) — defensible UX

The CLI at `walk_forward_cli.py:134-141`:

```python
if not ssl_results or not xgb_results:
    raise click.UsageError(
        "Inline orchestration not implemented (cost: 60-90 H200-hours). "
        ...
    )
```

The rationale ("user can't accidentally launch on the wrong
machine") is sound for a 60-90 H200-hour workload. The current UX
forces the operator to pre-run sub-phases and then aggregate, which
is the safe default.

That said: there's no `--inline-orchestrate` opt-in flag. If a
future operator legitimately needs to do an end-to-end run, they
either edit the CLI or write a wrapper. **The simpler fix** (Nit-6):
add a `--allow-inline-orchestrate / --no-inline-orchestrate` flag
defaulting to `--no-inline-orchestrate`. When set, the CLI orchestrates
the `run_single_fold` × 16 + `fit_predict_one_fold` × 16 + conformal
loop. Until that's wired, document in the docstring "inline
orchestration is intentionally deferred — see the H200 workflow
script in `scripts/`" (which doesn't exist yet — add a stub).

Marking Nit-6 not Minor: the current "raise UsageError" is the
right default and the workaround (run sub-phases separately) is
documented in the CLI's docstring (lines 22-30).

### E.2 `--report-path` exit code — Major-4 (also flagged in summary)

At `walk_forward_cli.py:185`:

```python
sys.exit(0 if payload.phase3_gate_1_passed else 1)
```

PHASE3.md §1 acceptance is **all four gates** (§1.1 + §1.2 + §1.3 +
§1.4). The CLI's exit code only reflects gate 1. A user running
`daity-walk-forward` in CI will see exit 0 even when:
- Gate 3 fails (conformal coverage out of band).
- Gate 4 fails (SSL doesn't beat random-init at any 2 horizons).

This silently passes a failing Phase 3 acceptance. The fix:

```python
acceptance = (
    payload.phase3_gate_1_passed
    and payload.phase3_gate_3_passed
    and payload.phase3_gate_4_passed
)
sys.exit(0 if acceptance else 1)
```

The console output already prints all three gates (lines 173-181),
so the UX impact is just the exit code. **This is Major-4.**

(Note: the failure-mode budget in §1 explicitly allows "partial
pass — proceed to Phase 4 with intraday focus". An operator who
wants to continue past a failing gate-3 or gate-4 should do so
EXPLICITLY, e.g., via a `--ignore-gate=3` flag, not via a silently
0-exit CLI.)

### E.3 Compute footprint comment (scrutiny 17) — matches §11

CLI docstring lines 19-20 say "60-90 H200-hours". PHASE3.md §11
table footnote at line 1466: "~60–90 H200-hours". ✓ Match.

Also matches §11's compute footnote at line 1469-1474 (random-init
× 3 seeds × 4 folds = 12 H200 runs at ~7h each = ~84 hours; SSL
4-fold sweep ~28 hours).

(Nit-7): the CLI docstring at line 19 says "30K SSL steps × 4 folds
+ 30K × 12 random-init" — the random-init multiplier is correctly
12 (= 3 seeds × 4 folds). Just confirming arithmetic is right.

### E.4 `--n-random-seeds` is decorative (Minor-6)

CLI flag at line 62 takes `--n-random-seeds=3`. But the CLI's
orchestration path is `raise UsageError` (E.1), so this flag is
never consumed — the random-init JSON is plumbed in pre-computed.
Drop the flag or document that it's a placeholder for the
future inline-orchestrate path.

### E.5 `--max-symbols` and `--max-steps` likewise unused (Minor-7)

Same problem as E.4 — these are placeholders for the unwired
inline-orchestrate path. Either gate them behind the eventual
`--allow-inline-orchestrate` flag (E.1) or drop until needed.

### E.6 `_load_fhr_json` handles both raw list and `{"results": [...]}` wrapper

At `walk_forward_cli.py:188-195`:

```python
def _load_fhr_json(path: Path | None) -> list[FoldHorizonResult]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return [FoldHorizonResult(**row) for row in data]
```

This is good defensive parsing — `daity-xgboost-baseline` emits
`{"per_fold_per_horizon": [...]}` (or similar) while a quick-and-dirty
hand-written aggregator might emit just `[...]`. But the wrapper key
is hardcoded to `"results"` — `xgboost_baseline.py`'s actual emit
key might differ.

(Minor-8): verify the upstream emit format and align. If
`xgboost_baseline_cli.py` writes `{"results": [...]}`, document it
in `_load_fhr_json`'s docstring. If it writes a different key, the
loader will fail at runtime when reading the wrapper file. Worth a
quick `tests/unit/test_walk_forward_cli.py` smoke (which doesn't
exist yet — see F).

---

## Section F — Test coverage

### F.1 What's tested (scrutiny 18) ✓

18 tests covering:
- Tri-criterion: 4-of-4 passes, 3-of-4 boundary, 2-of-4 fails, mean
  at threshold, mean below threshold, lucky-fold-dominance,
  empty input.
- Random-init: best-of-seeds (insufficient discriminative power —
  see B.1 Minor-2), delta-below-threshold fails.
- Horizon-set rule: 2-of-4 passes gate 1, 1-of-4 fails, all 4 pass.
- Conformal: in-band, below-band, empty.
- JSON: full-schema-keys present, gate-3 requires all-passing in-band.
- Thresholds locked: `RANK_IC_DELTA_THRESHOLD == 0.005`,
  `MIN_FOLDS_POSITIVE == 3`, `MIN_FOLDS_BEAT_XGB == 3`.

### F.2 What's NOT tested (scrutiny 19)

Each of these is a gap worth filling before the H200 run lands:

- **Provenance fields populated** (Minor-9). No test asserts that
  the output JSON has a non-empty `provenance.git_sha` and
  `provenance.parent_checkpoint_hash`. The `_git_sha` fallback to
  `"unknown"` means a malformed git invocation silently produces a
  legal-looking JSON with `"git_sha": "unknown"`. Add
  `test_emit_json_provenance_populated`.

- **`_load_fhr_json` wrapper-and-list paths** (Minor-8). No test
  exercises the loader. Add `test_load_fhr_json_handles_raw_list`
  and `test_load_fhr_json_handles_results_wrapper`. Use a tmp_path
  fixture.

- **Mismatched fold_ids between SSL and XGB** (Major-3). The current
  silent-drop behavior has no test. Once Major-3 lands (raise on
  mismatch), add `test_evaluate_horizon_gate_raises_on_fold_mismatch`.

- **`phase3_gate_3_passed` when `conformal_coverage` is empty / None /
  no horizons pass §1.1** (Minor-4). Three distinct cases, each
  should produce a documented value (`False` or `None` per C.2).

- **Survivorship audit list passthrough** (Nit-8). The 18th test
  asserts `survivorship_audit` is in the JSON keys but doesn't
  check the list content survives the round-trip. Trivial to add:
  `assert data["survivorship_audit"] == [{"fold_id": "F-9", ...}]`.

- **`schema_version == 1`** (Nit-9). Lock the version constant the
  same way the thresholds are locked.

- **Sub-phases that depend on horizons cardinality**. `MIN_HORIZONS_PASSED
  = 2` assumes 4 horizons; if amendment 17 adds a 5th, `2/5` is
  weaker than `2/4`. Document the dependency in a comment.

- **`evaluate_walk_forward` with `horizons=None`** (Nit-10). The
  default pulls from `daity.training.posttrain.HORIZONS`. No test
  exercises this default. Add `test_evaluate_walk_forward_default_horizons`.

### F.3 Module import dependency on torch — Major-3 (re-emphasized)

`walk_forward.py:32`:

```python
from daity.training.posttrain import HORIZONS
```

`daity.training.posttrain` imports torch (line 46). The aggregator
module — which is otherwise a pure-Python decision-rule engine —
cannot be imported in a CI-light environment without the full torch
+ peft stack installed.

This bit the reviewer's own attempt to run the unit tests:
`ModuleNotFoundError: No module named 'torch'`. CI may have torch
installed (it does, per pyproject.toml), but:

1. The §3.8 acceptance-check operator on a fresh CPU box, just
   wanting to re-aggregate JSONs, has to install torch.
2. The module's intent ("decision-rule + aggregation engine") is
   violated by a torch dependency.
3. Phase 5's HTML tearsheet, which will consume the JSON output,
   shouldn't need to import torch to RE-emit a JSON.

**Recommendation:** move the `HORIZONS` tuple to a small shared
constants module (e.g., `daity/constants.py` or
`daity/eval/__init__.py`) and import from there. Or inline the
default `("30m", "120m", "1d", "2d")` into `walk_forward.py`
directly (it's already locked-in by PHASE3.md amendment 16). The
trade-off: inlining adds one more place to update if a future
amendment 18 changes the horizon set; a shared constants module
is the cleaner fix.

This is **Major-3** elevated because the test suite can't even
collect without torch — a structural fragility.

---

## Section G — Phase 3 closure

### G.1 Phase 3 acceptance gate operational? (scrutiny 20)

**Mostly yes, with caveats:**

Given populated `ssl_results.json`, `xgb_results.json`,
`random_init_results.json`, and `conformal_coverage.json`, the CLI
runs `emit_walk_forward_json` → produces a yes/no on each of gates
1, 3, and 4. Gate 2 (the ≥2-of-4 horizons rule) is folded into
gate 1's `phase3_gate_1_passed`.

The caveats:
1. **Major-4** (exit code only reflects gate 1). A user reading
   the CLI's exit code as "Phase 3 accepted" gets the wrong
   answer if gate 3 or gate 4 failed.
2. **Major-1** (`folds` key in JSON misses the 16-row data
   payload). The operator gets a yes/no but the supporting
   per-(fold, horizon) numbers aren't in the JSON — Phase 5's
   tearsheet has to derive them from `tri_criterion` summary or
   re-load the upstream JSONs. The single-source-of-truth
   property the schema was supposed to provide is lost.
3. **Major-4** (gate 4 doesn't intersect with gate 1). The "SSL
   helps where it produces alpha" semantic from PHASE3.md §1.4
   isn't enforced.

After Majors 1 + 2 + 4 land, the answer to G.1 is unambiguously
yes.

### G.2 Phase 3 → Phase 4 handoff (scrutiny 21)

PHASE3.md §1's failure-mode budget specifies four branches:
- **30m and/or 120m pass but 1d/2d fail** → intraday-only Phase 4
  with high-priority OB fine-tune.
- **1d and/or 2d pass but 30m/120m fail** → swing-style Phase 5,
  downscoped OB.
- **0 or 1 horizons pass** → run stronger fallback (LightGBM); if
  still failing, Phase 4 with intraday focus + warning.
- **Gate 3 fails per-horizon** → that horizon's sizing falls back
  to vol-targeted heuristic.

The output JSON has `horizons_passed: list[str]` and `tri_criterion`
per horizon, which is enough for Phase 4's strategy chooser to read
"which timescale strategies survive". But:

- **No `random_init_gates[h].passed` exposed as a list of horizons
  passing §1.4** — easy to compute from `random_init_gates` dict
  but a Phase-4 reader has to do it.

- **No `horizons_failing_only_conformal`** — the failure-mode bullet
  "Gate 3 fails for any horizon that passed (1)" requires the
  Phase 4 strategy to know which horizons cleared alpha but failed
  calibration. Easy to derive from `conformal_gates` but worth a
  dedicated field for handoff clarity.

- **No `is_borderline` flag** — PHASE3.md §1's
  failure-mode-budget says "if failure margin against the stronger
  fallback is ≤ 0.003 on any 2 horizons, accept the borderline pass".
  No code path emits this — the operator has to eyeball the
  `mean_delta` values manually. Add it.

These are documented as **Minor-10** (provide a Phase-4 handoff
summary field). Not blocking, but the §3.8 acceptance step will
appreciate the operator-facing summary.

---

## Severity-grouped findings

### Major (0 Critical, 4 Major)

**Major-1 — `folds` key in JSON does not carry the 16-row per-(fold,
horizon) data payload.** The CLI populates `folds=_default_folds()`
(4 fold-time-window dicts), but PHASE3.md §8.4 specifies the
`folds` list as 16 entries, each with `ssl_heads`, `xgboost`,
`random_init` per-(fold, horizon) rank-IC + CRPS + decile spread +
coverage_80 + per_symbol_ic. Phase 5's `tearsheet.py` will read
the missing data when rendering. Build the `folds` list from the
result inputs inside `emit_walk_forward_json`; move fold time
windows to a separate `fold_windows` key.

**Major-2 — Provenance is missing `config_hash` and `as_of`.**
PHASE3.md §6.10 extends Phase 2's `(git_sha, config_hash, as_of)`
triple to Phase 3 artifacts. The current `provenance` dict only
carries `git_sha` and `parent_checkpoint_hash`. Add `config_hash`
(deterministic hash of the Hydra config) and `as_of` (data cutoff
timestamp). Without them, the smoke-test reproducibility contract
in CLAUDE.md is violated.

**Major-3 — `walk_forward` module imports torch transitively;
fold-ID mismatches silently drop folds.** Two related fragilities:
(a) `from daity.training.posttrain import HORIZONS` pulls in torch,
making the pure-Python aggregator unrunnable on a CPU-only box;
move `HORIZONS` to a shared constants module or inline. (b)
`evaluate_horizon_gate` and `evaluate_random_init_gate` silently
take the intersection of fold IDs when SSL/XGB/random-init have
mismatched fold sets — the tri-criterion is then evaluated against
a different population than the 4-fold spec, with no warning.
Raise on mismatch; add `allow_partial: bool = False` escape hatch.

**Major-4 — CLI exit code reflects gate 1 only; gate 4 doesn't
intersect with gate 1.** Two semantic gaps in the acceptance
computation: (a) The CLI's `sys.exit(0 if gate_1_passed else 1)`
silently passes when gates 3 or 4 fail — should exit 0 only on
gates 1 ∧ 3 ∧ 4. (b) PHASE3.md §1.4 requires the random-init-passing
horizon set to **overlap** with the tri-criterion-passing horizon
set; `evaluate_walk_forward` computes the two sets independently
and sets `phase3_gate_4_passed = len(rand_passed) >= 2` without
checking overlap. Add the intersection check.

### Minor (8)

**Minor-1** — `test_tri_criterion_3_of_4_positive_passes_b_and_c`
doesn't assert `res.passed` (only the per-criterion flags). Add the
assertion to ratchet against future `and` → `or` regressions at
`walk_forward.py:134`.

**Minor-2** — `test_random_init_takes_best_of_seeds` uses
seed-2-wins-every-fold; doesn't discriminate "best per fold" from
"best overall". Add a variant where the winning seed differs by
fold.

**Minor-3** — `phase3_gate_3_passed = False` when no horizons pass
gate 1 is ambiguous (was gate 3 evaluated and failed, or never
evaluated?). Emit `None` or a sentinel for "not evaluated"; document
in the JSON schema.

**Minor-4** — Three test cases missing for `phase3_gate_3_passed`:
(a) `conformal_coverage is None`, (b) `conformal_coverage = {}`,
(c) no horizons pass gate 1.

**Minor-5** — `schema_version` bump cadence not documented.
Add a module-level docstring section.

**Minor-6** — `--n-random-seeds` CLI flag is decorative (the
inline-orchestration path is `raise UsageError`). Drop or
document as a placeholder.

**Minor-7** — Same for `--max-symbols`, `--max-steps`,
`--feature-root`, `--label-root` — the aggregation-only mode
doesn't consume any of them. Drop or gate behind a future
`--allow-inline-orchestrate` flag.

**Minor-8** — `_load_fhr_json` has no test; the
`{"results": [...]}` wrapper key is hardcoded but the upstream
JSONs may emit a different key. Verify alignment with
`daity-xgboost-baseline` and `daity-posttrain` emit formats; add
tests for both raw-list and wrapper-dict paths.

**Minor-9** — No test asserts `provenance.git_sha` is non-empty
in the output JSON (the `_git_sha` fallback to `"unknown"` masks
silent failures). Add `test_emit_json_provenance_populated`.

**Minor-10** — No Phase-4-handoff summary fields in the JSON:
`horizons_passing_§1.4`, `horizons_failing_only_conformal`,
`is_borderline_per_horizon` (re: failure-mode-budget's "≤0.003
against stronger fallback" rule). Easy to derive but worth
explicit fields for handoff clarity.

### Nit (6)

**Nit-1** — `0.005` mean-delta threshold can land at IEEE-754
boundary for pathological-but-legal fold sums. Document that
`>= 0.005` (inclusive) is the locked semantic.

**Nit-2** — `folds_positive` uses strict `> 0`; `passed_mean_delta`
uses inclusive `>= 0.005`. The asymmetry is correct but worth a
docstring sentence.

**Nit-3** — `RANK_IC_DELTA_THRESHOLD` import in tests is only
exercised once (in `test_thresholds_are_locked`). Fine; noting
for review.

**Nit-4** — `test_conformal_gate_in_band` doesn't exercise the
exact boundary cases (`mean_cov == 0.75` and `== 0.85`). Add for
symmetry with the mean-delta-at-threshold test.

**Nit-5** — Gate-3-as-`all(in_band)` (every passing horizon must
clear) breaks symmetry with gates 1+4 (which are `>= 2 of 4`).
The choice is defensible (shipping a miscalibrated band is worse
than not shipping the horizon at all) but should be documented in
PHASE3.md §1.3 to ratify the implementation.

**Nit-6** — Add `--allow-inline-orchestrate` opt-in flag for the
case where an operator legitimately needs end-to-end. Default
`--no-inline-orchestrate` so the safe default holds.

**Nit-7** — Docstring arithmetic ("30K × 12 random-init") is
correct (= 3 seeds × 4 folds). Confirming.

**Nit-8** — `test_emit_walk_forward_json_writes_full_schema`
asserts `survivorship_audit` key is present but doesn't check
list content. Add a content assertion.

**Nit-9** — Lock `schema_version == 1` in the test suite the same
way thresholds are locked.

**Nit-10** — `evaluate_walk_forward` with `horizons=None` (default
pulls `HORIZONS`) is untested. Add one test.

---

## Final disposition and summary

**Disposition: Approve with major changes (0 Critical, 4 Major, 8
Minor, 6 Nit).**

The walk-forward aggregator is the right shape and the right size
for what §3.7 is supposed to do: a freestanding decision-rule engine
that compresses 4 horizons × 4 folds × 4 result populations into a
yes/no Phase 3 acceptance, with the CLI as a thin orchestrator that
forces the user to pre-run sub-phases. Tri-criterion correctness is
sound (the `>= 0.005` inclusive boundary matches the spec, the
3-of-4 / 4-of-4 boundary is tested, and the architect-review §A.7
single-fold-dominance test is faithful). The §1.4 random-init
best-of-seeds-per-fold semantic is correct. The §1.3 conformal
"per-horizon coverage in [0.75, 0.85]" gate matches the spec, and
the "only horizons that passed §1.1 are gated" predicate is
correctly threaded.

The four Major findings are all contract / spec-conformance gaps:
- Major-1 — `folds` JSON key carries fold time windows instead of
  the 16 per-(fold, horizon) data rows the §8.4 schema mandates.
- Major-2 — `provenance` is missing `config_hash` and `as_of` per
  the §6.10 triple.
- Major-3 — `walk_forward` imports torch transitively (via
  `HORIZONS`), and the gate functions silently drop mismatched
  fold IDs.
- Major-4 — CLI exit code reflects gate 1 only (gates 3 and 4 can
  silently fail), and gate 4 doesn't enforce the §1.4
  "SSL-helps-where-alpha-is" intersection with gate 1.

The Minor findings cluster around test-coverage gaps (provenance
field assertions, `_load_fhr_json` paths, fold-ID mismatch
behavior, gate-3 corner cases) and CLI flags that are decorative
in the aggregation-only mode.

**The single most important change**: fix Major-1. The `folds`
JSON key is supposed to be the data-bearing portion of the file —
the 16 per-(fold, horizon) rows with rank-IC, CRPS, decile spread,
coverage_80, per-symbol IC, plus the SSL/XGB/random-init triples.
Phase 5's `tearsheet.py` will consume this directly. The current
implementation stuffs the key with 4 fold-time-window dicts (from
`_default_folds()`), which is BOTH the wrong shape AND the wrong
cardinality. Build the 16-row payload inside
`emit_walk_forward_json` from the inputs; keep `fold_metadata`
(time windows) as a separate `fold_windows` key. This is what
unblocks Phase 5's downstream rendering AND closes the schema
contract in PHASE3.md §8.4.

After Majors 1 + 2 + 3 + 4 are addressed and the test suite picks
up provenance-populated, fold-ID-mismatch, gate-3-corner-case, and
`_load_fhr_json` cases, §3.7 is a clean approval. Minor and Nit
findings can be batched into a follow-up commit.

---

**Phase 3 overall — go/no-go readiness:** With Majors 1, 2, 3, and 4
addressed, the project will be operationally ready to take Phase 3's
go/no-go decision once the 60-90 H200-hour walk-forward sweep lands —
the decision-rule engine is correct in substance, the CLI is the right
plumbing shape, and the failure-mode budget in PHASE3.md §1 is well
enough specified that the Phase-4 strategy choice follows mechanically
from `horizons_passed` plus the gate flags; the four Major findings
are all fixable in a single afternoon's work before the H200 run, not
structural rewrites.
