# Phase 3.4 Review — Posttrain module

**Reviewer:** Independent code reviewer
**Date:** 2026-05-11
**Disposition:** Approve+Major

Scope: `daity/training/posttrain.py` (466 LOC), `daity/training/losses.py`
(Phase-3 additions, lines 192-278), `tests/unit/test_posttrain.py`
(14 tests), `tests/leakage/test_posttrain_leakage.py` (3 gates). Reviewed
against PHASE3.md §6 (amendments 2 + 4 + 5 + 7 + 11 + 16 binding).
Supporting modules (`cross_section.py`, `backbone.py`, `tokenizer.py`,
`pretrain.py`) read for interaction correctness only.

All 17 tests pass on `peft==0.19.1`. Loss kernels (pinball, ListMLE)
numerically verified; named-adapter LoRA routing is empirically correct
under PEFT 0.19; leakage gates exercise the right invariants. **The
single Major blocker** is a silent-load-failure path in
`_load_ssl_checkpoint` / `_strip_prefix` that produces a randomly-
initialized backbone for plausible checkpoint prefix patterns
(`torch.compile`'d `tokenizer._orig_mod.X`, DDP-wrapped `module.tokenizer.X`)
without raising — defeating the entire posttrain premise. The Phase-3
loss-decrease acceptance criterion can pass on a random backbone, so
the failure is undetected by CI.

---

## Section A — Architectural correctness

### 1. PEFT named-adapter routing

Empirically verified on `peft==0.19.1`. With 2 LoRA banks `{'a', 'b'}`,
both with `requires_grad=True`, `set_adapter('a')`, then forward+backward:
`lora_*.a.weight.grad` is nonzero; `lora_*.b.weight.grad` is `None`. The
inactive bank simply doesn't appear in the autograd graph, so
`requires_grad=True` is harmless on inactive banks. `forward()`'s pattern
(`set_adapter(horizon)` then `_enable_all_lora_grads()`) is therefore
correct.

**Caveat:** `peft.add_adapter(name)` emits a warning if `name in 'lora_'`
(substring check). `'30m'`, `'120m'`, `'1d'`, `'2d'` don't match, so the
warning won't fire in practice; would fire if anyone renames to
`'lora_q'`.

**Verdict:** correct.

### 2. Checkpoint loading hparam-shape variance

`_load_ssl_checkpoint` (lines 182-211) handles three shapes:
(1) Lightning flat `{"state_dict", "hyper_parameters": {...}}`,
(2) Lightning `cfg` wrapper `hyper_parameters["cfg"] = namespace`,
(3) plain `state_dict` with no hparams.

Phase-2's `PretrainModule` at `pretrain.py:178-180` saves hparams as a
**flat dict**. So shape (1) is what daity's own v3_xattn ckpt writes;
the `cfg` unwrap path (lines 202-207) is **dead code** for daity ckpts,
and no test exercises it. The two-step unwrap covers `Namespace` and
plain `dict`, but not OmegaConf `DictConfig` — for that case the unwrap
falls through and the compat check sees `d_model = None`, which would
flag every key as a delta. Loud failure, fine — just not what the
operator expects.

**Verdict:** acceptable for the flat-dict shape; `cfg` branch is untested
and brittle. See **Minor-1**.

### 3. State-dict prefix stripping — silent failure (the main concern)

`_strip_prefix(sd, "tokenizer")` matches `"tokenizer."` exactly. Several
real-world prefix patterns produce keys this DOES NOT strip:

- `torch.compile`'d saves → `tokenizer._orig_mod.X`
- DDP/FSDP-wrapped → `tokenizer.module.X`
- Lightning subclass aliasing → `_tokenizer.X`

Empirically verified: with a synthetic ckpt whose keys are
`tokenizer._orig_mod.X`:

```
WARNING PostTrainModule: missing tokenizer keys: [forecast_token, ...]
WARNING PostTrainModule: missing backbone keys: [layers.0.norm1.weight, ...]
# Module is constructed; backbone.layers[0].attn.qkv weights are random init,
# NOT the 0.42 value I set in the source backbone. Load silently failed.
```

The Phase-3 acceptance criterion ("loss decreases for 100 steps") still
passes on a randomly-initialized backbone — so a busy operator or CI
won't catch it. We'd burn H200 hours on §3.5 training a random model
and only notice at eval when rank-IC ≈ random baseline.

Three fixes:
1. **Fail loud on missing/unexpected.** Currently `unexpected` (lines
   274, 278) is computed and discarded. If either list is non-empty,
   raise `IncompatibleCheckpointError`.
2. **Try multiple candidate prefixes** (`tokenizer.`,
   `_orig_mod.tokenizer.`, `module.tokenizer.`, …) until one yields a
   strict-clean load.
3. **At minimum: raise on `len(missing) > 0.1 * len(self.tokenizer.state_dict())`.**

**Verdict:** Major. See **Major-1**.

### 4. Tokenizer frozen

PHASE3.md §6.7 specifies "backbone always frozen; LoRA trainable from
step 0" — doesn't mention the tokenizer. The implementation freezes
everything in `self.tokenizer.parameters()` (~556K params at d=480):
4 scale_projections, resolution_embed, forecast_token. RevIN has no
affine params so nothing there. The conservative choice; spec doesn't
require otherwise.

Docstring at lines 287-288 says "LoRA-only fine-tunes the attention
path only" — imprecise. LoRA only touches `qkv` (Minor-5), not FFN, not
`out`, not LayerNorms. See **Nit-1**.

**Verdict:** correct.

### 5. `vol_target = |y|.log()` fallback

Line 403: `vol_target = y.abs().clamp(min=1e-6).log()`. The `CrossSection`
carrier has `vol_bucket` (int8 for conformal stratification) but no
per-symbol vol target; this fallback substitutes.

Issues with the proxy:

1. `|y|` is NOT realized vol. For 30m: realized vol = std of forward
   5min returns (6 bars). `|y| = |log(close_{t+30m}/close_t)|` has a
   point mass at zero (no-move bars), different distribution.
2. `clamp(min=1e-6)` maps zero-return rows to `log(1e-6) ≈ -13.8`. Real
   30m equity log-vol is roughly `[-9, -5]`. Quiet bars produce extreme
   negative targets → MSE penalizes inordinately → vol gradient swamps
   pinball + listmle.
3. Label producer at `daity/data/labels.py` doesn't compute realized
   vol either; plumbing requires a new column per-horizon, not a
   trivial CrossSection extension.

Fine for 3.4 smoke; the magnitude-balance heuristic will down-weight a
huge vol-MSE so loss decreases. But this MUST NOT survive into 3.5 —
the magnitude-balance silently masks the real failure mode (vol head
trains on a biased proxy). The TODO at 397-402 acknowledges but
doesn't mark blocking. See **Major-2** (3.5 prerequisite).

**Verdict:** acceptable as a 3.4 placeholder; blocking for 3.5.

---

## Section B — Loss correctness

### 6. Pinball loss formula

`losses.py:226-228`: `loss = torch.maximum(τ * diff, (τ-1) * diff)` with
`diff = y - q`. Matches `ρ_τ(u) = max(τ·u, (τ-1)·u)`. At τ=0.5:
`max(0.5u, -0.5u) = 0.5|u|`.

Verified:
- `y=[1,-2,0.5], q=[[0]]*3, τ=[0.5]` → impl returns 0.5833... = expected
  `0.5 * 3.5 / 3`.
- `y=0, q=+1, τ=0.1` → 0.9 (overshoot penalty); `q=-1` → 0.1 (undershoot).

**Verdict:** correct.

### 7. ListMLE numerics

`losses.py:267-278`. Stable suffix-LSE via `flip → logcumsumexp → flip`.

Reversed extreme `scores=[-10, 10], targets=[1, -1]`: sorted by
descending target → `[-10, 10]`. Suffix LSE = `[lse(-10,10), lse(10)] =
[10, 10]`. Per-step = `[10-(-10), 10-10] = [20, 0]`. Mean = **10.0**.

Aligned extreme `scores=[10, -10]`: sorted → `[10, -10]`. Suffix LSE =
`[10, -10]`. Per-step = `[0, 0]`. Mean = **0**.

(Prompt expected ~20 for reversed — that would be true if summed.
Implementation uses `.mean()`, so 10 is correct. `.mean()` is the
N-independent convention, more standard for cross-section ranking.)

**Verdict:** correct.

### 8. Quantile ordering not enforced

The quantile head emits 5 independent scalars. At random init,
monotonicity is violated with high probability. Pinball pressure pushes
toward monotonicity *softly* (asymmetric penalty at τ=0.1 vs τ=0.9), but
nothing enforces `q_0.1 ≤ q_0.25 ≤ ... ≤ q_0.9` at every `(sym, t)`.
Quantile crossings are a known pathology of unconstrained multi-quantile
regression.

For 3.6 CQR consumption (`band_lo = q_0.1 - s*, band_hi = q_0.9 + s*`),
crossings yield `band_lo > band_hi` (nonsense). PHASE3.md §7.2 doesn't
address this.

Fixes (any one):
- Monotonic head (Cannon, 2018): `cumsum(softplus(deltas))` over 5
  output scalars.
- Post-hoc sort at inference: `q, _ = torch.sort(q, dim=-1)`.
- Crossing penalty: `λ * Σ ReLU(q_i - q_{i+1})` added to pinball.

See **Minor-2**.

---

## Section C — Hparam compat coverage

### 9. What's checked, what's not

Checked: `d_model, n_layers, n_heads, ffn_ratio, patch_len, patch_stride,
num_channels, revin_affine, scales`.

Not checked, but in `PretrainConfig`: `max_seq_len, attn_dropout,
ffn_dropout, forecast_head_type, forecast_n_patches,
forecast_num_channels, contrastive_proj_dim, mask_ratio, ...`.

Of the unchecked architecture items:
- `max_seq_len`: only affects the precomputed RoPE buffer (non-persistent,
  not in state_dict). RoPE values are position-by-position identical
  regardless of `max_seq_len`. No breakage.
- `attn_dropout`, `ffn_dropout`: no params. Posttrain hardcodes 0.0,
  which matches v3_xattn (also 0.0) by accident. Silent override; only
  affects training stochasticity, not inference.
- `forecast_head_type`, `forecast_n_patches`, `forecast_num_channels`:
  control the SSL forecast HEAD, which posttrain doesn't load
  (top-level module not under `tokenizer.` or `backbone.`). Not load-
  bearing.

The checked list is adequate. A comment explaining what's deliberately
*not* checked would be nice.

### 10. Scales-order check

`_check_hparam_compat` normalizes to `list(...)`. Order matters because
`MultiResTokenizer.resolution_embed` is `nn.Parameter(len(scales), d_model)`
indexed by scale position. Swapped order → resolution-embed row for
`"5m"` would be applied to `"day"` patches. Inline comment at lines
165-166 documents this.

**Verdict:** correct; rationale documented.

---

## Section D — Test coverage

### 11. What's tested

14 unit + 3 leakage tests. Covers: pinball + listmle correctness;
freeze invariants; LoRA adapter switch sets `active_adapter`; forward
shapes; loss is finite + positive; 1-batch overfit reduces loss ≥30%
across 100 steps for all 4 horizons; `calibrate_loss_weights` produces
w*L in [0.5, 2.0]; hparam mismatch raises; plain state_dict (no
hparams) loads. Leakage: forward independent of out-of-carrier data;
label perturbation changes loss only; forward doesn't mutate params.

### 12. What's MISSING but probably should be

In risk-reduction order:

- **Cross-horizon gradient isolation.** No test verifies that
  `forward(cs, '30m'); backward()` produces zero grad on
  `lora_120m/1d/2d` and on `quantile_heads['120m']` etc. This is the
  load-bearing claim of the multi-bank design. I verified empirically
  (item 1) but a regression test would lock it. **Critical add.**
- **Prefix-mismatch fail-loud.** Per Major-1: build a ckpt with
  `tokenizer._orig_mod.X` keys, assert construction RAISES.
- **`_load_ssl_checkpoint`'s `cfg`-unwrap branch.** Dead code in the
  test suite; either remove or write a test that uses a
  `Namespace(cfg=Namespace(d_model=64, ...))` blob.
- **Degenerate cross-section.** `compute_loss` with `labels =
  torch.zeros(12)` (all-tied ranks, degenerate pinball) should not NaN.
- **LoRA bank weights NOT bit-equal across banks at init.** Verified
  empirically that PEFT 0.19 reseeds Kaiming init per-bank, so 4 banks
  have 4 different starts. Trivial test:
  `(lora_A['30m'].weight - lora_A['120m'].weight).abs().sum() > 0`.
- **`calibrate_loss_weights` with dropout > 0.** Current test uses
  `head_dropout=0.0` so `L_before == L_after` and `w*L = 1.0` trivially.
  Tolerance `[0.5, 2.0]` is never exercised. Run with `head_dropout=0.1`.
- **Test-warning cleanup.** `test_module_calibrate_loss_weights`
  triggers `UserWarning: Converting a tensor with requires_grad=True to
  a scalar`. Wrap in `torch.no_grad()` or call `.detach().float()`.

### 13. Leakage test scope

The 3 leakage gates verify architectural invariants for the posttrain
*layer*. They don't verify the "bit-equality under truncated history"
pattern from `test_label_leakage.py`. That pattern more naturally lives
at the `CrossSection` sampler level (and is partly covered by
`tests/leakage/test_cross_section_leakage.py`'s strict-`<` boundary).
Posttrain is downstream; the right gate at this layer is "forward only
reads carrier tensors", which the 3 tests cover.

**Verdict:** acceptable.

---

## Section E — Performance / scalability

### 14. `_enable_all_lora_grads()` called inside every `forward()`

Cost: ~50 μs per call × 30K steps ≈ 1.5 s total. Negligible.

The init call (line 309, before optimizer construction) is mandatory.
The per-forward call (line 369) is only required *after* a horizon
switch. Cheap guard:

```python
def forward(self, cs, horizon):
    if self.peft_backbone.active_adapter != horizon:
        self.peft_backbone.set_adapter(horizon)
        self._enable_all_lora_grads()
    ...
```

Also avoids redundant `set_adapter` calls on consecutive same-horizon
forwards (which happens in the per-horizon overfit test). See **Nit-2**.

### 15. Module size at real config

At `d_model=480, n_layers=12, lora_r=8, head_hidden=None→480`:

**LoRA:** per layer per bank = `2 * (8*480 + 1440*8) = 15,360`. ×12
layers ×4 banks = **737,280** params. Matches PHASE3.md §6.3 estimate
"~736K" exactly.

**Heads** (`Linear(480, 480) → GELU → Dropout → Linear(480, K)`):
- Quantile (K=5): `230,880 + 2,405 = 233,285`
- Rank/Vol (K=1): `230,880 + 481 = 231,361`
- Per horizon (q+r+v): `696,007`
- ×4 horizons: **2,784,028** total.

**Combined trainable: ~3.52M params.**

PHASE3.md §6.3 claims "~2M total" / "heads ~1.2M total". This is **~75%
low** — the head cost is dominated by the Linear(480, 480) which the
spec's estimate implies `head_hidden ≈ 200` (not 480). The implementation
matches §6.1's explicit shape spec; §6.3's estimate is internally
inconsistent with §6.1's shape. Either drop `head_hidden` default to
~200 OR update §6.3 to ~3.5M. See **Minor-3**.

---

## Section F — Design choices

### 16. No Lightning wrapper

`PostTrainModule: nn.Module`; Phase 2's `PretrainModule: L.LightningModule`.
PHASE3.md §6.6 specifies bf16-mixed + grad clip 1.0 + EMA + cosine LR +
AdamW per-group LR/wd + W&B. All one-line LightningModule configs;
re-implementing in §3.5 by hand will be ~200 LOC of training-loop
boilerplate that `pretrain.py` already solved.

Recommend adding `PostTrainLightning` wrapper in 3.4 (~30 LOC: forward
→ `training_step`, `validation_step`, `configure_optimizers`,
`on_train_start` hook for magnitude-balance). Reduces 3.5 surface area
materially. Not blocking 3.4 acceptance. See **Minor-4**.

### 17. Magnitude-balance frequency

`calibrate_loss_weights` is called once, locked thereafter. Matches
PHASE3.md §6.2 ("weights written into `posttrain_v1.yaml` after the
first measurement and locked for reproducibility"). Recalibrating
periodically would break `config_hash` determinism. The downside (head
magnitudes can diverge during training) is a known soft-balance
limitation, not a correctness bug.

**Verdict:** correct.

---

## Section G — Other findings

### 18. LoRA target deviates from PHASE3.md §6.3

PHASE3.md §6.3: `target_modules = ['q_proj', 'v_proj']` (Q + V only,
"smaller trainable surface, lower overfit risk").

Implementation: `target_modules = ['qkv']` on `Backbone`'s fused
`Linear(d_model, 3*d_model)`. LoRA on the fused projection produces a
rank-r update that projects onto **all of Q, K, V — including K**.

Inline comment at lines 99-103 claims "Equivalent to `(q_proj, v_proj)`
LoRA on an unfused architecture". This is **not equivalent in
adaptation surface** (the fused-qkv target adapts K too, which (q, v)
does not). Param count is identical (LoRA matrices are the same size
either way: `r·(d_in + d_out)`).

Two reconciliations:
1. Split `RoPESelfAttention.qkv` into three linears (`q_proj`, `k_proj`,
   `v_proj`); target `['q_proj', 'v_proj']`. Faithful but architectural.
2. Update PHASE3.md §6.3 to ratify the fused-`qkv` target and
   acknowledge K-adaptation.

See **Minor-5**.

### 19. `nn.ModuleDict` per-horizon

`quantile_heads: nn.ModuleDict({h: ...})` correctly registers heads as
submodules (visible in `state_dict()` and `parameters()`). Good design.

### 20. `_phase3_quantiles` buffer

Registered as buffer (lines 332-335) for device tracking. Correct.

### 21. Defensive copy of `cfg.initial_weights`

`self.loss_weights[h] = dict(cfg.initial_weights[h])` (line 328) copies
the inner dict to avoid `cfg` mutation. Good.

### 22. `_HparamDelta` error formatting

Uses `repr()` for both values (lines 257-265). Readable; includes
checkpoint path and offending key per spec §6.10.

### 23. Test fixture `argsort(argsort)` matches sampler

`_fake_cross_section` builds ranks via `torch.argsort(torch.argsort(...))`,
matching the sampler at `cross_section.py:360` (ordinal, not average,
per the Phase 3.3 review Minor-1).

### 24. Duplicate optimizer construction in test

`test_module_one_batch_overfit_reduces_loss_for_all_horizons` constructs
`optim` twice (lines 242 + 244); the first is overwritten. Clarity
nit; no functional impact.

---

## Critical / Major / Minor / Nit findings (grouped)

### Critical

None.

### Major

**Major-1** — Silent SSL-weight load failure for compile/wrapped prefixes.
`_strip_prefix(state, "tokenizer")` only matches exact `tokenizer.X`.
For `torch.compile`'d ckpts (`tokenizer._orig_mod.X`) or DDP-wrapped
(`module.tokenizer.X`), strip yields garbage keys; `load_state_dict
(strict=False)` silently leaves the backbone randomly initialized and
only logs WARNING. The Phase-3 acceptance criterion (loss decreases over
100 steps) passes on a random backbone, so the failure is invisible to
CI. **Fix:** raise on any non-empty `missing` OR `unexpected` from
`load_state_dict` (currently `unexpected` at lines 274, 278 is computed
and discarded), AND/OR try multiple candidate prefixes
(`tokenizer.`, `_orig_mod.tokenizer.`, `module.tokenizer.`) until one
yields a strict-clean load. Empirically reproduced; see Section A item 3.

**Major-2** — `vol_target = |y|.log()` fallback (line 403) is a biased
proxy. `|y|` has a zero-return point mass that the `clamp(min=1e-6)` maps
to `log(1e-6) ≈ -13.8`, far below the real log-vol range `[-9, -5]`.
On quiet bars the vol-MSE explodes, swamping pinball + listmle gradient.
The magnitude-balance heuristic silently down-weights the vol head,
masking the failure mode. Fine for 3.4 smoke; must be replaced in 3.5
with a proper realized-vol target plumbed through
`daity/data/labels.py` → `CrossSection.vol_target`. The TODO at lines
397-402 acknowledges but doesn't mark blocking.

### Minor

**Minor-1** — `_load_ssl_checkpoint`'s `cfg`-unwrap branch (lines
202-207) is dead code for daity-produced checkpoints (Phase-2's
`PretrainModule` saves hparams as a flat dict, `pretrain.py:178-180`).
No test exercises it. The branch covers `Namespace` and `dict` cases
but not OmegaConf `DictConfig`. Either delete the branch or write a
test that uses a non-daity ckpt with the wrapped shape.

**Minor-2** — Quantile crossings unaddressed. The quantile head emits 5
independent scalars; nothing enforces `q_0.1 ≤ ... ≤ q_0.9`. At random
init they're scrambled; pinball pressure pushes softly toward
monotonicity but doesn't enforce. For 3.6 CQR consumption (`band_lo =
q_0.1 - s*, band_hi = q_0.9 + s*`), crossings produce `band_lo > band_hi`.
Recommend one of: monotonic head (cumsum-softplus), post-hoc sort at
inference, or `λ * Σ ReLU(q_i - q_{i+1})` crossing penalty. PHASE3.md
§6.1 should specify.

**Minor-3** — Trainable param count doesn't match PHASE3.md §6.3 claim.
At the real config (d_model=480, head_hidden=None→480) the
implementation produces ~3.52M trainable params (737K LoRA + 2.78M
heads); §6.3 claims "~2M total" / "heads ~1.2M". The implementation
matches §6.1's shape spec; §6.3's estimate implies `head_hidden ≈ 200`.
Either drop `head_hidden` default to ~200 OR update §6.3 to ~3.5M.

**Minor-4** — No Lightning wrapper. §3.5 will need to re-implement
bf16-mixed + grad clip + EMA + cosine LR + AdamW per-group LR/wd + W&B
logging by hand (~200 LOC) when a `PostTrainLightning` wrapper in 3.4
(~30 LOC) would reduce it to ~15 LOC of `Trainer` setup. Phase-2's
`PretrainModule` is the precedent.

**Minor-5** — LoRA target deviates from PHASE3.md §6.3. Spec says
`['q_proj', 'v_proj']`; implementation uses `['qkv']` on the fused
projection, which adapts Q, K, AND V. Same param count, different
adaptation surface. Comment at lines 99-103 incorrectly claims
equivalence. Either split `RoPESelfAttention.qkv` into three linears
(architectural) or update PHASE3.md §6.3 to ratify fused-`qkv` and
acknowledge K-adaptation.

**Minor-6** — `unexpected` from `load_state_dict` discarded (lines
274, 278). If the SSL ckpt has extra keys under `tokenizer.` (e.g., an
accidentally-saved aux head), no signal. Log both `missing` AND
`unexpected`, or raise on either (preferred — see Major-1).

**Minor-7** — `calibrate_loss_weights` test is degenerate. Uses
`head_dropout=0.0` so `L_before == L_after` and `w*L = 1.0` trivially;
the `[0.5, 2.0]` tolerance is never exercised. Run with `head_dropout=0.1`.

**Minor-8** — Missing test: cross-horizon gradient isolation. Verify
that `forward(cs, '30m'); loss.backward()` produces nonzero grad on
`lora_30m` params AND `grad is None`/zero on `lora_120m/1d/2d`. The
load-bearing claim of the multi-bank design is currently locked only
by empirical inspection, not regression test.

**Minor-9** — Missing test: prefix-mismatch fail-loud. After Major-1
fix, add a test that builds a ckpt with `tokenizer._orig_mod.X` keys
and asserts construction RAISES.

**Minor-10** — Missing test: degenerate cross-section. `compute_loss`
with `labels = torch.zeros(12)` (all-tied ranks, degenerate pinball)
should not NaN.

### Nit

**Nit-1** — Docstring at lines 287-288 ("LoRA-only fine-tunes the
attention path only") is imprecise. LoRA only touches `qkv` (per
Minor-5), not FFN, not `out`, not LayerNorms. Tokenizer also fully
frozen. Clearer: "Backbone + tokenizer fully frozen; only LoRA adapters
on `qkv` + the 12 head MLPs are trainable."

**Nit-2** — `_enable_all_lora_grads()` called on every forward. Cost
negligible but the call is only required after a horizon switch. Guard
with `if self.peft_backbone.active_adapter != horizon`.

**Nit-3** — `per_head_losses` returns tensors with `requires_grad=True`.
`test_module_calibrate_loss_weights` triggers
`UserWarning: Converting a tensor with requires_grad=True to a scalar`.
Wrap in `torch.no_grad()` or call `.detach().float()`. Cleanup.

**Nit-4** — `forward` return type `dict[str, torch.Tensor]` — could be
a `TypedDict` or dataclass to make typos static-check errors.

**Nit-5** — `optim` constructed twice in
`test_module_one_batch_overfit_reduces_loss_for_all_horizons`
(lines 242 + 244); first is overwritten. Clarity.

**Nit-6** — Successful checkpoint load could log INFO with path for
traceability; currently silent on success.

**Nit-7** — `HORIZONS`/`SCALES` module-level tuples could be annotated
`Final[tuple[str, ...]]` for typing clarity.

---

## Final disposition + summary

**Disposition: Approve+Major.**

Phase 3.4 is structurally solid: loss kernels are numerically correct,
PEFT multi-bank routing works as designed on PEFT 0.19, leakage gates
exercise the right invariants, all 17 tests pass. The single Major
blocker (Major-1) is the silent-load-failure in
`_strip_prefix`/`_load_ssl_checkpoint`: with a `torch.compile`'d or
DDP-wrapped ckpt prefix pattern, the SSL weights silently fail to load
and the module trains a randomly-initialized backbone, with only a
WARNING log. The Phase-3 loss-decrease acceptance criterion passes on a
random backbone, so CI doesn't catch it. Fix is small (raise on
`missing` or `unexpected` non-empty after the load).

Major-2 (vol-target proxy) is a 3.5 prerequisite, not a 3.4 blocker —
the current `|y|.log()` proxy is correct enough for the 3.4 smoke test,
but the magnitude-balance heuristic silently down-weights an exploding
vol-MSE on quiet bars, hiding the real failure. 3.5's first PR should
plumb a proper realized-vol target through.

Minor findings cluster around (a) test gaps that are 5-line adds
(cross-horizon isolation, prefix-mismatch fail-loud, degenerate
cross-section); (b) PHASE3.md doc inconsistencies (param count §6.3,
LoRA target spec §6.3); (c) one architectural choice (no Lightning
wrapper) that pushes complexity into §3.5 without strong justification.

**The single most important change:** fix `_load_ssl_checkpoint` /
`_strip_prefix` to fail loud on prefix mismatch (Major-1). Either raise
on any `missing`/`unexpected` from `load_state_dict`, or try multiple
candidate prefixes (`tokenizer.`, `_orig_mod.tokenizer.`,
`module.tokenizer.`) until one yields a strict-clean load. Without
this, §3.5 could spend H200 hours training a random model and only
catch it at eval-time when rank-IC matches random baseline.

3.4 can be approved with Major-1 fixed in a follow-up commit before
3.5 starts. Major-2 (vol target) deferred to 3.5's first PR. All
Minor findings batchable into a cleanup commit.
