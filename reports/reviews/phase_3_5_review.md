# Phase 3.5 Review — Posttrain training loop + random-init ablation + CLI

**Reviewer:** Independent code reviewer
**Date:** 2026-05-11
**Disposition:** Request changes (1 Critical, 4 Major)

Scope: `daity/training/posttrain_loop.py` (491 LOC),
`daity/scripts/posttrain_cli.py` (175 LOC),
`tests/unit/test_posttrain_loop.py` (3 smoke tests, 262 LOC).
Cross-referenced `posttrain.py` for the new `_use_lora` gating.
Phase 3.4 review's two Major findings cross-checked for disposition.

All 3 smoke tests pass, but they don't exercise the production code
path: the train/val split is silently broken, random-init correctness
relies on a non-persistent attribute, and no smoke asserts convergence
or leakage. The load-bearing §3.5 acceptance claim (LoRA + heads beat
random-init by ≥0.005 rank-IC at ≥2-of-4 horizons) cannot be tested
by the current suite, and a full-scale H200 run as currently coded
would train on val labels.

---

## Section A — Random-init ablation correctness (§1.4 hard gate)

### 1. Major-2 carryover: `vol_target` plumbing NOT done

Phase 3.4 review flagged `vol_target = |y|.log()` as a §3.4-only proxy,
with explicit commitment to plumb `forward_realized` through
`CrossSection` in §3.5. Status:

- `posttrain.py:117` still defaults `vol_target_mode="abs_y_log"`.
- `cross_section.py:101-112` — `CrossSection` is
  `@dataclass(slots=True)` with no `vol_target` field. Setting it
  raises `AttributeError`.
- `posttrain.py:455` reads `getattr(cs, "vol_target", None)` —
  unreachable on the slotted class; returns `None` and the
  `forward_realized` branch raises.
- `RunConfig`/CLI don't expose `vol_target_mode`. No operator knob.
- `labels.py` was not extended to emit per-horizon realized vol.

The full-scale random-init run on H200 will train the vol head
against the §3.4-flagged biased proxy. Magnitude-balance will down-
weight it silently; both LoRA and random_init use the same broken
proxy, so the comparison is OK for the rank-IC numbers but the vol
head output is uninterpretable. The §3.4 reviewer was explicit that
this is the §3.5 prerequisite. **Unresolved.** See **Major-1**.

### 2. `_build_module` random_init scope (`posttrain_loop.py:152-181`)

**LayerNorm NOT reset.** The re-init loop touches only `nn.Linear`.
The backbone has 25 LayerNorms (12 × `norm1` + 12 × `norm2` +
`final_norm`, `backbone.py:152, 156, 206`). They retain SSL-trained
values. For a "true random init" per §1.4 ("full fine-tune of all
backbone weights"), LayerNorms count as backbone weights. Empirical
impact small (O(d_model) each), but it's not the comparison §1.4
specifies. See **Major-2** below.

**Loop ordering is load-bearing but uncommented.** Line 167-168
unfreezes ALL backbone params, which (because PEFT inserts LoRA
in-place under `self.backbone.qkv.lora_*`) flips LoRA to
`requires_grad=True`. The next loop (174-176) re-freezes LoRA.
Net result correct; ordering is fragile. If anyone reorders, random-
init silently becomes "random-init + LoRA" — exactly what §1.4
forbids. See **Minor-1**.

**`disable_adapter_layers()` is idempotent in PEFT 0.19.1.** Verified
empirically; second call is a no-op. With `_use_lora=False`, the
module's `forward` skips `set_adapter`, so the disabled state
persists. ✓

### 3. Tokenizer NOT reset in random_init

Lines 152-158: "tokenizer stays at SSL-trained init (just RevIN +
patch projection; not a representational learner)". This is a
defensible interpretation but NOT what §1.4 specifies ("full fine-
tune of all backbone weights at lr_backbone=3e-4").

Tokenizer at `d_model=480` is ~556K params (4 scale_projections +
resolution_embed + forecast_token). The SSL-trained patch projection
**is** the representational asset of Phase 2. Keeping it warm-started
biases §1.4 in the SSL's favor by exactly the tokenizer's share of
the total represented function. The principled fix is to also reset
+ unfreeze the tokenizer in random_init mode (the optimizer's
`backbone_params` bucket would pick it up automatically since
tokenizer names don't match `lora_*` or `_heads*`). At minimum,
document in PHASE3.md §1.4 that "backbone weights" excludes the
tokenizer and acknowledge the bias. See **Major-2**.

### 4. `_use_lora` flag — fragile, non-persistent

Plain attribute set at `posttrain.py:357` (default `True`), flipped
to `False` in `posttrain_loop.py:177`. `nn.Module.state_dict()` does
NOT serialize plain attributes — `_use_lora` does not survive
save/load.

Phase 3.7 builds fresh modules per fold so in-process is safe. But
Phase 3.6 conformal and Phase 5 inference will load saved checkpoints;
`_use_lora` defaults to `True`, silently routing a random-init
checkpoint through LoRA banks. The `disable_adapter_layers` state
also doesn't survive a fresh `__init__`.

Fix: register `_use_lora` as a buffer (`register_buffer("_use_lora_flag",
torch.tensor(1.0))`), OR move the mode into `PostTrainConfig`. See
**Major-3**.

---

## Section B — Training-loop correctness

### 5. `_HorizonStream.next()` retry semantics

Lines 249-276. Walkthrough on a permanently-empty horizon: `tried`
grows by at most 1 per outer iter (the `if h in tried: continue`
skip doesn't bump it; the StopIteration branch does, then
`continue`s). With `len(tried) == 4` the outer while exits returning
`None`. The restart-and-retry pattern wastes ~2 `iter()`
constructions per known-empty draw — bounded at `len(horizons)`
total. **No infinite loop; correct.** Minor optimization at **Nit-1**.

### 6. CRITICAL — train sampler unbounded by `train_end`

`run_single_fold` lines 376-391:

```python
train_sampler = CrossSectionalSampler(
    ..., as_of=cfg.as_of,
    min_end_ts=None,   # train: anchors up to train_end       ← LIE
)
val_sampler = CrossSectionalSampler(
    ..., as_of=cfg.as_of,
    min_end_ts=cfg.val_start,
)
```

The comment claims "anchors up to train_end" but **`train_end` is
never passed to the sampler**. The sampler emits anchors with `t ≤
as_of`. `cfg.as_of` defaults via CLI to `datetime.now(UTC) + 1d`
(line 113). So for a typical operator invocation:

```
daity-posttrain --train-start 2019-01-01 --train-end 2024-12-31 \
                --val-start  2025-01-01 --val-end   2025-03-31
```

the train sampler emits anchors at every valid `t` from feature-cache
earliest through 2026-05-12, **including the val window 2025-01-01
to 2025-03-31**. The val sampler emits from 2025-01-01 through
today. They overlap on `[val_start, as_of]`. The training loop draws
cross-sections from val anchors, then `evaluate()` scores on those
same labels.

`train_start`, `train_end`, `val_end` are declared in `RunConfig`
(lines 77-80) and NEVER read by `run_single_fold`. All three are
dead. The "anchors up to train_end" comment is the surface symptom.

**Effect.** Final rank-IC under §1.1 tri-criterion and §1.4 vs
random-init are arbitrarily inflated. The acceptance gate cannot be
trusted to measure what it claims to measure.

**Fix:**

```python
train_sampler = CrossSectionalSampler(
    ..., as_of=cfg.train_end,            # cap at train_end
    min_end_ts=cfg.train_start,          # NEW lower bound
)
val_sampler = CrossSectionalSampler(
    ..., as_of=cfg.val_end,              # cap at val_end
    min_end_ts=cfg.val_start,
)
```

Plus a leakage test under `tests/leakage/test_posttrain_loop_leakage.py`:
`for cs in train_sampler.iter_30m(): assert cs.t < val_start`.

See **Critical-1**.

### 7. `_build_optimizer` parameter groups

Verified empirically on a minimal PEFT-wrapped module: `nn.Module.
named_parameters()` deduplicates against `self.backbone` and
`self.peft_backbone` (they share storage; `get_peft_model` mutates
`self.backbone` in-place to insert LoRA). Names look like
`backbone.qkv.lora_A.30m.weight` (LoRA) and `backbone.norm1.weight`
(plain backbone).

- LoRA mode: backbone frozen → `backbone_params` empty. Two groups
  `(lora, heads)`. base_lrs = `[lr_lora, lr_head]`. ✓
- random_init mode: LoRA frozen → `lora_params` empty. Two groups
  `(heads, backbone)`. base_lrs = `[lr_head, lr_backbone]`. ✓

`base_lrs = [g["lr"] for g in optimizer.param_groups]` (line 424)
captures the right values; the schedule zip at line 433 preserves
them. Correct.

But the step-log line 470-474 references `param_groups[0]["lr"]` as
`lr_lora` — **misleading in random_init mode where group 0 is heads**.
See **Nit-2**.

### 8. Gradient clipping

`module.trainable_parameters()` (`posttrain.py:520`) filters by
`requires_grad`. In LoRA mode: LoRA + heads. In random_init: backbone
+ heads. Correct. ✓

### 9. LR schedule edge case

`_cosine_with_warmup_lr` with smoke's `warmup_steps_pct=0.0,
lr_min_ratio=1.0`: `warmup_steps=0`, branch `if step < 0 and 0 > 0`
is False (short-circuit), returns `base_lr * (1.0 + 0.0 * cos_factor)
= base_lr` at every step. Constant LR. ✓

Production with `warmup_steps_pct=0.05`: at step 0,
`warmup_steps=1500`; returns `base_lr * 1/1500` — tiny but non-zero.
Standard HuggingFace-style warmup; reasonable.

### 10. Magnitude-balance robustness

Lines 397-419. Pulls one cross-section per horizon DIRECTLY via
`next(iter(...))`. If a horizon is empty, `continue`. If <4 horizons
covered: log WARNING, skip calibration entirely.

In production this means: any partial-coverage scenario silently
proceeds with `loss_weights = {h: {q: 1, r: 1, v: 1}}`, and the
§3.4-proxy vol head then explodes vol-MSE on quiet bars. The
warning is too quiet — partial coverage almost always indicates a
config bug (window too short, threshold too high). Default should
be raise; opt-out via `RunConfig.allow_partial_calibration: bool =
False`. See **Minor-2**.

---

## Section C — CLI design

### 11. `--checkpoint` required in `random_init` mode

Lines 86-96 unconditionally raise without a checkpoint. The cause is
`PostTrainModule.__init__` calls `_load_ssl_checkpoint(cfg)` at line
275 unconditionally. Cleaner design: pass `mode` into
`PostTrainConfig`, skip checkpoint load when `mode=random_init`,
use `cfg.d_model` etc. directly to build the module. See **Minor-3**.

### 12. Default `--max-steps 30000` runtime

Sampler ~3.7 ms/step (per §3.3 review C.11); H200 forward ~20 ms;
backward+step ~10 ms → ~33 ms/step × 30K = ~16 minutes/fold. Multiply
by 3 seeds × random-init = ~50 minutes; ~16 minutes for the SSL
sweep. §11's "30K-step ~6 hours" estimate assumed batch-size-512
with gradient accumulation, which §3.5 does NOT implement — single
cross-section per step likely under-utilizes H200. See **Minor-4** /
**Major-4**.

For §3.7's 4-fold × (SSL + 3 random-init) = 16 runs: ~4-5 GPU hours.
Comfortable.

### 13. No `--device` flag

`run_single_fold` line 373 picks `cuda` if available. No operator
override. `CUDA_VISIBLE_DEVICES` workaround. See **Nit-4**.

### 14. `--report-path` JSON schema impoverished

Payload at lines 145-160 keeps `mean_ic` but drops `std_ic, t_stat,
n_observations` from `RankICResult`. No `git_sha`, `config_hash`,
`parent_checkpoint_hash` (the §6.10 provenance triple). §3.7 needs
all of these to aggregate folds. Persist the full `RankICResult` +
provenance triple + checkpoint sha256. See **Minor-5**.

---

## Section D — Test coverage

### 15. What's tested

Three `tmp_path` synthetic-universe smokes:
1. `lora` mode completes; IC table well-formed in `[-1, 1]`.
2. `random_init` mode unfreezes backbone, freezes LoRA.
3. At least 3 of 4 horizons reach `final_loss < inf`.

Wiring + freeze invariants only. No convergence, leakage, or
acceptance-property checks.

### 16. What's NOT tested — HIGH-VALUE GAPS

In risk-reduction order:

a. **Train/val anchor disjointness.** No test asserts the train
   stream emits no cross-section with `t ≥ val_start`. Would have
   caught **Critical-1** at PR time. Trivial: `for cs in
   train_sampler.iter_30m(): assert cs.t < val_start`. **Critical add.**

b. **Loss DECREASES in lora mode.** `test_lora_mode_loss_decreases`'s
   docstring even admits "we don't enforce strict decrease". §3.4
   reviewer noted the predecessor asserts ≥30% drop over 100 steps;
   §3.5 is a regression. Restore the strict assertion.

c. **Magnitude-balance ACTUALLY balances.** Smoke calls calibration
   but no assertion checks `module.loss_weights ≠ 1.0`.

d. **`_use_lora` persistence under module reload.** Round-trip
   `save → load`, assert `_use_lora=False` survives. With current
   code this would FAIL — see **Major-3**.

e. **Determinism.** Two runs with `seed=42` produce identical
   `final_loss_by_horizon` and `per_horizon_rank_ic`. §3.7's 3-seed
   sweep relies on reproducibility.

f. **Random-init weights actually differ from SSL init.** After
   the re-init pass, `backbone.qkv.base_layer.weight` should not
   equal the checkpoint value. Catches future refactors that reorder
   the load vs init steps.

g. **LR schedule non-zero at step 0** — regression catch.

---

## Section E — Performance

### 17. Cadence weights `70:52:1:1` (scrutiny 16)

In a 30K-step run: 30m gets ~17K cross-sections, 120m ~12.6K, **1d
and 2d each ~240**. ListMLE rank head on 240 cross-sections is
borderline — each cross-section is one ListMLE evaluation, 240
gradient steps for 1d's rank head.

PHASE3.md §6.6 doesn't specify per-horizon cadence absolutes; the
70:52:1:1 is "natural cadence" (anchors-per-trading-day proportional).
If §1.4 fails on 1d/2d specifically, the cheapest escape is to bump
to e.g. 70:52:5:5. Recommend logging per-horizon `n_cross_sections`
at evaluate time and flagging horizons below 1000. See **Minor-6**.

### 18. Single-process sampler (scrutiny 17)

~3.7 ms/step on the main thread. With H200 forward ~20 ms, sampler
is ~16% of wall-clock. Multi-worker would require pickling
`ParquetStore` + caches — fragile. Defer to §3.7 with profiling.
See **Nit-5**.

---

## Section F — Other findings

### 19. Scope drift from PHASE3.md §6.6

- **bf16 NOT implemented** (§6.6 spec'd). `loss.backward()` runs in
  fp32. ~2× slower + ~2× memory on H200. 5-LOC fix
  (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`). See
  **Major-4**.
- **EMA NOT implemented** (§6.6 + Phase 2 precedent). `evaluate()`
  runs on live weights. See **Minor-8**.
- **Gradient accumulation NOT implemented** (§6.6 spec's "effective
  batch ~512"). One cross-section per step. See **Minor-4**.
- **W&B NOT initialized** (§6.12 spec'd full per-horizon metric
  emission). See **Minor-10**.
- **Mid-training val NOT called.** `val_every_n_steps: int = 2000`
  is declared (line 104) and read nowhere. Final val only.
  See **Minor-11**.

### 20. JSON `Infinity` bug

`final_loss_by_horizon` initialized to `float("inf")` (line 429). If
a horizon is never sampled, the JSON report contains `"1d":
Infinity` — non-standard JSON, breaks some parsers. Convert to
`None`. See **Minor-7**.

### 21. Carryover from Phase 3.4

- **Major-1 (silent SSL load).** Resolved. `posttrain.py:298-322`
  now raises on empty `tok_sd`/`bb_sd` OR non-empty `missing` keys.
  ✓
- **Major-2 (vol_target proxy).** **NOT resolved.** See Section A
  item 1 + **Major-1** below.
- Other Minors: Lightning wrapper (Minor-4) not adopted, surfacing
  as bf16/EMA/grad-accum gaps in §3.5.

### 22. Simplification opportunities

- Device-move construction duplicated 3 times (lines 314-322,
  404-411, 447-454). Factor out `_cs_to(cs, device)`.
- `evaluate()` tz-strip (lines 327-329) — add comment that it's
  safe because `_np_to_utc` guarantees UTC.

---

## Critical / Major / Minor / Nit findings

### Critical

**Critical-1** — Train/val data leakage in `run_single_fold`. Train
sampler uses `as_of=cfg.as_of, min_end_ts=None`; with CLI default
`as_of = today+1d`, train anchors are drawn from the full timeline,
overlapping the val window on `[val_start, as_of]`. `train_start`,
`train_end`, `val_end` are declared in `RunConfig` and never read.
The "anchors up to train_end" comment at line 382 is wrong. Final
§1.1 + §1.4 acceptance numbers will be invalid. **Fix:**
`as_of=cfg.train_end` on train sampler, `as_of=cfg.val_end` on val
sampler, plus `min_end_ts=cfg.train_start` on train. Add a leakage
gate. Blocks §3.5 acceptance + §3.7.

### Major

**Major-1** — Phase 3.4's Major-2 (`vol_target` plumbing through
`CrossSection`) is UNRESOLVED. Defaults still `"abs_y_log"`;
`CrossSection` is slotted with no `vol_target` field; `RunConfig`/CLI
don't expose `vol_target_mode`; `labels.py` not extended.
Magnitude-balance silently masks the biased proxy. Predecessor
explicitly flagged this as the §3.5 prerequisite. Either
(a) plumb end-to-end (labels → `_LabelCache` → `CrossSection.vol_target`
slot → module), or (b) ratify the proxy in PHASE3.md and remove
`forward_realized` as dead code.

**Major-2** — Tokenizer NOT reset in random_init mode. ~556K
SSL-trained params (the patch projection — Phase 2's actual
representational asset) remain warm-started. Biases §1.4 in the SSL's
favor. Also LayerNorms (~30K params, 25 instances) are skipped by
the kaiming re-init loop. Either reset+unfreeze the tokenizer (and
include `nn.LayerNorm` in the re-init pass) OR document in PHASE3.md
§1.4 that "backbone weights" excludes the tokenizer and acknowledge
the bias.

**Major-3** — `_use_lora` flag doesn't persist through state_dict.
Plain attribute at `posttrain.py:357`; not serialized. A reload of
a random-init-trained checkpoint into a fresh module silently
defaults `_use_lora=True`, routing through (frozen) LoRA banks; the
`disable_adapter_layers` state also doesn't survive `__init__`.
Phase 3.6 conformal + Phase 5 inference will silently use the wrong
forward path. Fix: register as buffer or move into `PostTrainConfig`.
Add a save/load round-trip regression test.

**Major-4** — bf16-mixed precision NOT implemented despite §6.6
specifying it. `loss.backward()` runs in fp32. ~2× slower and ~2×
memory on H200. 5-LOC fix:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    out = module(cs_dev, horizon=h)
    loss = module.compute_loss(out, cs_dev, horizon=h)
```

No `GradScaler` needed (bf16 has fp32 dynamic range).

### Minor

**Minor-1** — `_build_module` random_init loop ordering is
load-bearing (unfreeze-all then re-freeze-LoRA) but uncommented.
Reordering silently flips random-init into "random-init + LoRA",
which §1.4 forbids. Comment OR rewrite to filter LoRA out of the
unfreeze pass in one go.

**Minor-2** — Magnitude-balance partial-coverage silently proceeds
with weights=1.0 (lines 415-419). In production this almost
certainly indicates a config bug. Default: raise. Opt-out:
`RunConfig.allow_partial_calibration`.

**Minor-3** — `--checkpoint` required in `random_init` mode is poor
UX. Operator must supply an SSL checkpoint they're explicitly
ablating. Skip `_load_ssl_checkpoint` when mode=random_init; use
`cfg.d_model` etc. directly to build the module.

**Minor-4** — No gradient accumulation. §6.6 spec'd effective batch
~512 via 4-CS accumulation. §3.5 is 1-CS/step. Combined with Major-4,
the §3.5 throughput on H200 likely falls short of the §11 budget
assumption.

**Minor-5** — `--report-path` JSON drops `std_ic, t_stat,
n_observations` from `RankICResult` and the `(git_sha, config_hash,
parent_checkpoint_hash)` provenance triple. §3.7 needs both to
aggregate folds.

**Minor-6** — `DEFAULT_CADENCE_WEIGHTS = {30m:70, 120m:52, 1d:1, 2d:1}`
allocates ~240 cross-sections each to 1d/2d over 30K steps. ListMLE
on 240 is borderline. Either bump (e.g. 70:52:5:5), add a CLI flag
for ablation, or log+flag horizons with `n_cross_sections < 1000`.

**Minor-7** — `final_loss_by_horizon = float("inf")` (line 429)
serializes to non-standard JSON `Infinity`. Convert to `None` before
emit.

**Minor-8** — No EMA. §6.6 spec'd `EMACallback(decay=0.999)` per
Phase 2 precedent. `evaluate()` uses live weights → noisier val
rank-IC.

**Minor-9** — bf16 + grad-accum + EMA + W&B all missing → §3.4
review's Minor-4 (Lightning wrapper) is now costing §3.5 four
distinct §6.6 amenities. Reconsider.

**Minor-10** — No W&B. §6.12 spec'd per-step + per-eval per-horizon
metrics. Add `--wandb-project / --wandb-entity` flags or wire into
the existing logger.

**Minor-11** — Mid-training val plumbed but never called.
`val_every_n_steps: int = 2000` (line 104) is read nowhere. Either
implement periodic eval or remove the field.

**Minor-12** — `train_start`, `train_end`, `val_end` are dead fields
on `RunConfig`. Symptom of Critical-1. Add `__post_init__` validation
`train_start < train_end ≤ val_start < val_end ≤ as_of` to catch the
gap at config build time.

**Minor-13** — `limit_val_cross_sections=50` (line 105) caps
evaluate at 50 CS/horizon. On a 4-month val window that's ~0.85%
of available; std error ~14%. Default should be `None`; smoke sets
`50` explicitly.

**Minor-14** — Tests for cross-horizon gradient isolation (§3.4
review Minor-8) and prefix-mismatch fail-loud (§3.4 review Minor-9)
still missing.

### Nit

**Nit-1** — `_HorizonStream` restart-and-retry burns ~2 `iter()`
constructions per known-empty draw. Bounded waste; `_try_next(h)`
helper would clarify.

**Nit-2** — Step-log line 470-474 references `param_groups[0]["lr"]`
as `lr_lora`. In random_init mode group 0 is heads. Use `g["_name"]`.

**Nit-3** — 8-line device-move duplicated 3 times. Factor out
`_cs_to(cs, device)`.

**Nit-4** — No `--device` CLI flag. Operators use
`CUDA_VISIBLE_DEVICES`; document in `--help`.

**Nit-5** — Single-process sampler at ~16% of step wall-clock. Defer
multi-worker to §3.7 with profiling.

**Nit-6** — `torch.manual_seed` called twice (line 367 in
`run_single_fold`, line 159 in `_build_module`). Harmless but
redundant.

**Nit-7** — `evaluate()` line 327-329 `cs.t.replace(tzinfo=None)` is
safe because `_np_to_utc` guarantees UTC. Add comment.

**Nit-8** — `RunResult.final_step = cfg.max_steps` (line 485) is
misleading when the loop short-circuits via `if res is None: break`.
Track actual exit step.

**Nit-9** — Smoke fixture rebuilds the synthetic universe per test
(~3s × 3 = 9s overhead). `pytest fixture scope="module"` would
amortize.

**Nit-10** — `self._use_lora: bool = True` at `posttrain.py:357` is
buried in a long `__init__`. Move to the top so the gating semantics
are visible at a glance.

---

## Final disposition and summary

**Disposition: Request changes (1 Critical, 4 Major).**

§3.5 lands a coherent single-fold trainer with a sensible mode
selector, weighted-random horizon stream with correct empty-iterator
handling, and three smokes that demonstrate the pipeline runs. The
random-init `_build_module` correctly unfreezes the backbone,
disables PEFT adapter routing, and freezes LoRA params — modulo two
soft issues (tokenizer + LayerNorms not reset).

The blockers are not in the algorithmic logic but in the data-window
plumbing and the leftover §3.4 prerequisite:

1. **Critical-1** — train sampler reads anchors through `cfg.as_of`
   (default = today+1d) instead of being bounded at `train_end`.
   `train_start`/`train_end`/`val_end` are dead fields. The H200
   run will train on val labels; §1.1/§1.4 acceptance numbers will
   be invalid.
2. **Major-1** — Phase 3.4's Major-2 (vol_target plumbing) carried
   forward, NOT addressed. Production uses the §3.4-flagged biased
   proxy; magnitude-balance silently masks the failure.
3. **Major-2/3** — random-init correctness gaps (tokenizer +
   LayerNorms not reset; `_use_lora` non-persistent through
   state_dict).
4. **Major-4** — bf16 not implemented despite §6.6 specifying it.

Minor findings cluster around (a) PHASE3.md §6.6 features not
implemented (bf16, EMA, grad-accum, W&B, periodic val); (b) test
coverage gaps (no leakage gate, no loss-decrease assertion, no
determinism check, no `_use_lora` persistence test); (c) report
schema impoverishment for §3.7's needs.

**The single most important change:** fix Critical-1. Cap the train
sampler at `train_end` (`as_of=cfg.train_end`) and the val sampler
at `val_end` (`as_of=cfg.val_end`); pass `min_end_ts=cfg.train_start`
on train. Add a leakage gate
(`tests/leakage/test_posttrain_loop_leakage.py`) asserting
`for cs in train_sampler.iter_30m(): assert cs.t < val_start`.
Without this fix, §1.4's hard gate could pass under leakage — the
project's go/no-go decision for Phase 4 would rest on an unsound
measurement.

Major-1 (vol_target) is the second priority and should land before
the §3.5 H200 run; the §3.4 reviewer explicitly flagged it as the
§3.5 prerequisite and it has not been done.

§3.5 cannot be approved as-is. After Critical-1 is fixed and at least
Major-1 is addressed (or explicitly downgraded in PHASE3.md), §3.5
is a small-edit-distance approval. Major-2/3/4 can be batched into
a follow-up commit before §3.7 starts.
