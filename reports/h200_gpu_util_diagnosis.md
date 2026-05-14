# H200 GPU under-utilization diagnosis — Phase 3 posttrain

**TL;DR.** You are not data-bound, you are not compute-bound, you are **CPU-launch-overhead bound**. At N=205, K=82 tokens, d=480, every step issues 100+ tiny CUDA kernels whose individual work fits in <1 ms on H200 but whose Python+CUDA dispatch overhead is 5-15 µs each. That floors per-step time around 30-50 ms of pure launch latency, before any data wait. The 70-100 ms you observe is launch-overhead plus the synchronous-H2D + GIL/IPC contention from the data path. The H200 is **wildly over-provisioned for this batch size** — until you raise effective batch (multi-CS micro-batching) or amortize launches (CUDA Graphs / torch.compile done right), GPU util cannot rise.

The empirical clue everyone misses: GPU memory used is 5 GB / 143 GB = 3.5%. That is the smoking gun. You have ~40× headroom to grow batch before HBM is the constraint.

## 1. Root causes, ranked by impact

### RC-1 (≈40-50 ms / step) — CPU-side kernel-launch overhead dominates GPU compute.

The forward pass at the actual posttrain batch (`B=205, T=82, d=480`, 12 layers) is **~30 GFLOPs total**. On H200 sustained bf16 (≈ 1.5 PFLOPs realistic for small matmuls), pure compute is ~20 µs. But each transformer block dispatches:

`norm1 → qkv (LoRA: base_linear + lora_A + lora_B + scale + add) → SDPA → out → norm2 → ffn0 → GELU → ffn1` ≈ 11 kernels/block × 12 blocks = **132 kernels for backbone**, plus tokenizer (patcher unfold + revin mean/std/sub/div + scale_projection + add_embed) × 4 scales ≈ 25 kernels, plus FORECAST cat, plus 3 heads × (Linear+GELU+Dropout+Linear) = 12 kernels, plus the **loss path** (`pinball`: broadcast sub, mul, max, mean; `listmle`: argsort, gather, flip, logcumsumexp, flip, sub, mean; `vol_mse`: abs, clamp, log, mse_loss; weighted sum). Loss ≈ 30 kernels.

**Forward+loss = ~190 kernels**. Backward roughly 2× = ~380. AdamW step over ~130 trainable tensors with `foreach`: ~5-10 fused kernels. **Grand total ≈ 600 kernels/step**.

PyTorch eager-mode dispatch cost: ~5-15 µs/kernel from Python (function call → dispatcher → device dispatch → cudaLaunchKernel). 600 × 10 µs = **6 ms minimum just for launch latency**. But each kernel's H2D dispatch *also* waits for previous launches to finish queueing — so the GPU's queue depth isn't enough to hide it. Empirically, eager-mode transformer training at this scale (B=200, d=512, 12 layers) sees 30-60 ms/step purely from launch overhead. This is **the dominant cost**.

Evidence in your code:
- `daity/models/backbone.py:119-135` — `RoPESelfAttention.forward` does 7 separate ops (linear, view, unbind, 3× transpose, 2× `_apply_rope`, SDPA, transpose, contiguous, view, linear) — every one a kernel + dispatch.
- `daity/models/backbone.py:64-81` — `_apply_rope` does 6 elementwise ops (`mul, mul, sub, mul, mul, add, stack, flatten`) per attention. Across 12 layers × 2 (Q+K) = 24 RoPE calls. **144 elementwise kernels per forward just for RoPE.**
- `daity/training/posttrain.py:467-487` — `per_head_losses` does `abs → clamp → log` (3 kernels) + pinball's `unsqueeze, sub, mul, mul, sub, max, mean` (7 kernels) + listmle's `argsort, index, flip, logcumsumexp, flip, sub, mean` (7 kernels) + `mse_loss` (3 kernels). That's ~20 kernels of sub-microsecond GPU work each.

This is also why **fused AdamW made it WORSE** (8 sps vs 14 sps) — fused AdamW reduces the *number* of param updates per step, which is irrelevant when launch overhead is on the *forward+backward*, not the optimizer. The "fused" version has a higher constant launch overhead (the foreach setup) that you pay even though it would save time at a different batch size.

### RC-2 (≈10-20 ms / step) — `peft.set_adapter()` + `_enable_all_lora_grads()` on every step.

Empirical measurement on a CPU-only stand-in matching your 12-layer × `qkv`-LoRA × 4-adapter shape (see debug script §5 to reproduce on H200):

| call | wall-clock |
|---|---|
| `peft_backbone.set_adapter("30m")` | ~380 µs |
| `_enable_all_lora_grads()` (`named_parameters()` walk + 96 `requires_grad_(True)` calls) | ~185 µs |
| Direct alternative: just set `layer._active_adapter = h` on each LoRA layer | **6.8 µs** |

That's CPU time, but it's CPU time on the **main thread between consecutive GPU ops** — so the GPU sits idle while it happens. The 565 µs combined is ~1% of a 70 ms step on its own, but the bigger sin is what it does inside:

`set_adapter` walks `model.modules()` (peft model has **~317 modules** for your backbone), checks `isinstance(BaseTunerLayer)` on each, and for each of the 12 LoRA layers walks `adapter_layer_names = ('lora_A','lora_B','lora_embedding_A','lora_embedding_B')` × 4 adapters × params per adapter, calling `_set_layer_requires_grad` on each. See `/Users/debdoot/Documents/Claude/Projects/daity/.venv/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:2076-2103` and `:1530-1552`. That's **~12 × 4 × 4 × 2 = 384 `requires_grad_()` mutations per step**. Each mutation is a Python attribute write + autograd state check.

Then `_enable_all_lora_grads()` (`daity/training/posttrain.py:403-413`) does **another** full `named_parameters()` walk and re-enables every `lora_*` param — **completely undoing** the freeze that `set_adapter` just did. The two calls are working at cross purposes, with `_enable_all_lora_grads` being the one that "wins" — and that's the one you actually want.

**You don't need `set_adapter` at all.** The LoRA forward kernel (`peft/tuners/lora/layer.py:941-982`) routes through `self.active_adapters` — that's `[self._active_adapter]`. If you just set `lora_layer._active_adapter = h` on each of the 12 LoRA layers, forward routes correctly. Requires-grad for all 4 banks stays True (which is what you want for the single-optimizer-over-all-banks design).

Note: in `random_init` mode `_use_lora=False` so this entire path is skipped. The 10 sps vs 14 sps gap between random_init and LoRA mode is consistent with this: random_init **skips ~1-3 ms of per-step peft housekeeping** that the LoRA mode pays, *but* random_init unfreezes the backbone (36M params) so its **backward and optimizer step are heavier** — net result is similar wall-clock. The 4 sps gap (29% slower) is what you'd expect from training 12× more params per step.

### RC-3 (≈5-15 ms / step) — `.to(device, non_blocking=True)` on unpinned memory is a silent **synchronous** copy.

`daity/training/posttrain_loop.py:646-666` — `_AsyncHorizonStream._to_device` issues `.to(self.device, non_blocking=True)` from producer threads on **unpinned** CPU tensors. From the PyTorch docs and source: when the source is pageable (unpinned), `non_blocking=True` is silently downgraded to synchronous because cudaMemcpyAsync requires a pinned source. **The producer thread blocks for the full H2D duration** holding the CPython GIL except briefly inside the C++ call.

Per step the H2D moves 6 tensors (4 scales × N × L × C float32 + labels + ranks + mask + bucket) ≈ 10 MB. At ~12 GB/s pageable H2D on PCIe 5 (realistic for unpinned) that's ~1 ms of actual transfer, but the call to `cudaMemcpy` includes a sync, so all four producer threads serialize on the CUDA default stream and on the GIL. With queue depth of 4 you mostly hide this — but only for the *first* drawn CS each opt step. Then the steady-state pattern is "producer drains queue trying to fill, main pulls one and runs forward, producer immediately wants the GIL back to refill queue."

**Why pin_memory made it WORSE (your test)**: the pin allocator on `torch.from_numpy(...).pin_memory()` has to allocate from a separate pool, and at our cross-section build rate (every ~25 ms) the pin allocator is contended across 4 producer threads. The fix isn't `pin_memory()` per tensor — it's either (a) pre-allocate a persistent pinned buffer per scale and copy into it (no per-step alloc), or (b) skip the producer-thread H2D entirely and let the main thread do it from a CUDA stream pool with proper overlap.

### RC-4 (≈2-5 ms / step) — the loss path is a stack of tiny Python-dispatched kernels with a sync point in ListMLE.

`pinball_loss` and `vol_mse` are cheap-ish (~5 kernels each), but **`listmle_loss` has hidden costs**:

`daity/training/losses.py:269-278`:
```python
sorted_idx = torch.argsort(targets, descending=True)
sorted_scores = scores[sorted_idx]
flipped = torch.flip(sorted_scores, dims=[0])
rev_lse = torch.flip(torch.logcumsumexp(flipped, dim=0), dims=[0])
per_step = rev_lse - sorted_scores
return per_step.mean()
```

`argsort` on float32 of size 205 is a small radix-sort kernel + a host-side wait (PyTorch's argsort on small CUDA tensors can fall back to a path that requires a sync). `logcumsumexp` is a *scan* — these are notoriously latency-bound. Two `flip` calls + index + sub + mean. ~7-10 kernels for ListMLE alone, including a likely sync inside argsort that stalls the whole stream.

Combined: pinball (~5 kernels) + vol MSE w/ `abs.clamp.log` (~5 kernels) + listmle (~10 kernels including a sync) ≈ **20 kernels at ~100 µs of stream-serial overhead** = ~2 ms per step. Compute is trivial.

### RC-5 (≈1-3 ms / step) — `clip_grad_norm_` walks `module.trainable_parameters()` every step, which iterates all `named_parameters()`.

`daity/training/posttrain.py:540-541`:
```python
def trainable_parameters(self):
    return [(n, p) for n, p in self.named_parameters() if p.requires_grad]
```

Called every step in `posttrain_loop.py:1011`. Walks all ~230 param tensors in LoRA mode, ~700+ in random_init. Python list-comprehension overhead is ~180 µs (measured). The clip itself is a single `foreach` kernel — cheap. But the list construction is on the main thread.

Aggregated **Python-side** overhead per step:
- `set_adapter` + `enable_grads`: ~600 µs (LoRA mode)
- `trainable_parameters()` for clip: ~200 µs
- `_cs_to_device` dict-comp in main thread (for MP path): ~50 µs
- `autocast_ctx()` context-manager entry+exit: ~30 µs
- `optimizer.zero_grad(set_to_none=True)`: ~150 µs (200 params)
- LR schedule update over param groups: ~20 µs
- `compute_loss` / `per_head_losses` Python overhead: ~200 µs

Sum: **~1.2-1.5 ms / step of pure-Python overhead in your hot loop, plus all the dispatch overheads above.** That's not the dominant cost, but it's where the easy wins live.

## 2. Why your attempts didn't work

| Attempt | What actually happened |
|---|---|
| MP DataLoader (16 workers, spawn) | Each cross-section pickled across IPC ≈ 10 MB × 2× pickle/unpickle overhead = 20-40 ms IPC per CS. Plus `_cs_to_device` in main thread on unpinned tensors = another ~5 ms sync H2D. Net: data path now ~25-45 ms/CS instead of ~25 ms via threads. **MP didn't help because data wasn't the bottleneck.** Throughput was always launch-overhead-floored at ~10-14 sps. |
| `torch.compile(peft_backbone, dynamic=True)` | The peft `LoraLayer.forward` has Python-level branches on `self.disable_adapters`, `self.merged`, and iterates `self.active_adapters` from a Python list. `dynamic=True` makes the compiler keep these as Python-level decisions. PT 2.6's inductor handles this by spawning **32 background compile workers per process** which fight the data path for the same vCPUs. Even if it had compiled cleanly, `fullgraph=False` means it would have given you partial graphs broken at each peft branch — minimal speedup. |
| `fused=True` AdamW | Fused AdamW's setup cost (foreach-init, dtype check across all groups) is higher than non-fused at small param counts. At 3M params split into 2-3 groups with ~130 tensors total, the savings from a single CUDA kernel vs `foreach`-default's ~5 kernels is < setup overhead. **And it doesn't touch the forward/backward, which IS where you're spending time.** |
| `pin_memory()` per-tensor | Allocator contention between 4 producer threads + the pin-memory ring buffer running through small ~2 MB allocations. The allocations themselves take ~1-2 ms each. With 6 tensors × 4 threads = 24 pin allocations every 25 ms, you spent more on `cudaHostAlloc` than you saved on the (now-truly-async) copy. |
| `cudnn.benchmark + tf32` | Backbone uses **SDPA**, not cudnn convs — `cudnn.benchmark` is a no-op for matmul/attention paths. TF32 doesn't apply when you're in `bf16` autocast (which you are). These are no-ops at your config, not regressions. |
| `grad_accum_steps=4` | Theoretically helps RC-2 and RC-5 (amortize peft + clip overhead 4×), but each `accum` step still pays full forward+backward kernel-launch overhead (RC-1). Worse: with `accum=4` you're now serially executing 4 cross-sections worth of `set_adapter` (one per iteration if horizons differ — typical given the 70:52:1:1 cadence). Net win: maybe 10%. Not the breakthrough you'd hope for. |

## 3. Recommended fixes, ranked

### Fix-1 (highest impact, low risk) — **Skip `set_adapter()` and `_enable_all_lora_grads()` after the first step. Use direct `_active_adapter` assignment.**

**Mechanism.** As shown in RC-2, `set_adapter` does a `model.modules()` walk and 384+ requires_grad mutations per step that you immediately undo. The LoRA forward only reads `layer.active_adapters` which is `[layer._active_adapter]`. Set this directly on each LoRA layer (12 of them) — that's ~7 µs and is mathematically identical for the forward pass. Call `_enable_all_lora_grads()` ONCE at construction time only.

**Patch sketch** (`daity/training/posttrain.py`):

```python
# In __init__, after building peft_backbone:
from peft.tuners.lora.layer import LoraLayer
self._lora_layers: list[LoraLayer] = [
    m for m in self.peft_backbone.modules() if isinstance(m, LoraLayer)
]
self._enable_all_lora_grads()  # once, not per-step

# In forward():
if self._use_lora:
    # Direct assignment instead of peft_backbone.set_adapter(horizon)
    for layer in self._lora_layers:
        layer._active_adapter = [horizon]  # the property unwraps str→[str]
    # NB: do NOT call _enable_all_lora_grads here. Grads were enabled in __init__.
```

**Estimated speedup**: shaves 1-2 ms/step in pure-Python overhead. Standalone: ~5%. The bigger win is removing the GIL-held block between forward and the previous data wait, which can unblock GPU prefetch better. Combined effect: **8-12% throughput gain** → ~15-16 sps from 14.

**Risk to training**: zero. Identical forward math, identical gradient flow. The only behavioral diff is that the optimizer continues to see all 4 LoRA banks with `requires_grad=True` every step — which is what your `_enable_all_lora_grads` already enforces. No new code paths exercised by inactive banks because forward only routes through the active one.

### Fix-2 (highest impact, moderate risk) — **Eliminate the per-step launch tax via `torch.compile(mode="reduce-overhead")` or CUDA Graphs on the FROZEN backbone forward.**

**Mechanism.** The 600 kernel launches per step (RC-1) are *fixed in number* — you do the exact same sequence of ops on every step for a given horizon. The shapes are also fixed per horizon (cadence weights are per-horizon; you stream in horizon-batched chunks). This is the textbook case for **CUDA Graphs** which capture a stream of kernels into a single submit-once primitive, reducing the per-step CPU overhead from ~10 ms to ~10 µs.

PT 2.6 exposes this via `torch.compile(model, mode="reduce-overhead")` which uses CUDA Graphs under the hood and handles the bf16 + autocast + dynamic input-pad issues for you. The blocker for naive `torch.compile` was the peft `LoraLayer.forward` Python branches — but `mode="reduce-overhead"` + `fullgraph=False` will give you graph capture *inside* the matmul-heavy regions and Python fallback only at the branches.

Better alternative for full control: **manually capture a CUDA graph** of the static forward+backward+optimizer.step for each horizon, replay on each step. PT has a `torch.cuda.graph_pool_handle()` + `torch.cuda.CUDAGraph()` API. Capture once per horizon (4 graphs), match input shape via padding to the max N across train (= |universe|=205).

**Patch sketch** (a `torch.compile` MVP first, as it's lower-effort):

```python
# In _build_module after assembling the module:
if cfg.compile_forward and device.type == "cuda":
    # Compile the heavy parts separately so peft's set_adapter
    # boundary doesn't break the graph. Backbone is the win.
    module.peft_backbone = torch.compile(
        module.peft_backbone,
        mode="reduce-overhead",   # CUDA Graphs underneath
        dynamic=False,            # shapes ARE static once you pad N
        fullgraph=False,          # tolerate the peft Python branches
    )
    module.tokenizer = torch.compile(module.tokenizer, mode="reduce-overhead", dynamic=False, fullgraph=True)
```

You'll need to **pad N to a fixed value** (max universe size, 205) — pad with zeros + use the `valid_mask` you already track to mask the loss. Tokenizer + backbone produce a fixed-shape output regardless of valid_mask; losses already handle a per-row reduction.

Critically: **kill the `dynamic=True`** you tried before. Dynamic shapes prevent CUDA Graph capture which is the entire point of `reduce-overhead`. Pad and go static.

Also **disable the compile worker process spam**: set `TORCH_LOGS=` and `TORCHINDUCTOR_COMPILE_THREADS=1` env vars before launching. The "32 workers" you saw is `torch._inductor.config.compile_threads` — pin it.

**Estimated speedup**: 2-3× on per-step compute. At 70 ms/step → 25-35 ms/step. Realistic throughput target: **30-40 sps** in LoRA mode, **25-35 sps** in random_init.

**Risk to training**: moderate. CUDA Graph capture is sensitive to anything that allocates GPU memory inside the captured region — including some autograd machinery. First capture pass may crash and require code tweaks. The graph is also frozen to shape and dtype, so changes (e.g., evaluation with different N) need a separate graph or a fallback path. Also: `torch.compile` and peft have known interaction issues — your fullgraph=False saved you last time, keep it that way.

### Fix-3 (high impact, high risk) — **Multi-cross-section micro-batching: batch 4-16 CrossSections per forward.**

**Mechanism.** This is the *real* answer for an H200. At N=205 you are wasting >95% of the GPU. A 12-layer transformer at B=205 takes the same wall-clock as B=2048 (kernel-launch-bound, not compute-bound). Stack 8 CrossSections per forward: effective B=1640. Still bf16 memory ~5 GB (well under 143 GB). Now each kernel launch does ~8× more work, kernel-launch overhead amortizes 8×, **and you can drop bf16 for fp32 if it matters for stability**.

The challenge: you're currently architected around one-horizon-per-step (because of the peft `set_adapter` requirement). Two options:

1. **Same-horizon micro-batch.** Stack 8 CSes of the same horizon. Pad each to N=205. Sample weights by cadence_weights from the queue, but draw 8 consecutive same-horizon CSes (drain the horizon's queue more aggressively). Simple impl.

2. **Mixed-horizon micro-batch via `peft._mixed_batch_forward`.** Peft supports per-row adapter selection via `adapter_names=[...]` kwarg. See `peft/tuners/lora/layer.py:712-756`. Stack 8 CSes of any horizons; pass `adapter_names = ['30m']*205 + ['120m']*205 + ...`. peft does the right thing internally. Slightly less efficient (loops adapters inside each layer), but eliminates the horizon-sequencing constraint.

**Patch sketch (option 1, the safer one)**: refactor `_HorizonStream.next()` to return a list of K same-horizon CSes (drawn from one chosen horizon), and have `module.forward` accept a list and stack along dim 0. The losses already average over dim 0 — no change needed there. Pinball is `mean()` over (N, K); listmle needs to be per-CS (rank within the cross-section, not across the stacked dim). Fix: compute listmle per-row-block, then average.

**Estimated speedup**: 3-5×. **Realistic target with Fix-1 + Fix-2 + Fix-3: 50-80 sps**. GPU util should hit 70%+.

**Risk to training**: moderate-high. Effective batch is now 8× larger → LR may need 0.5-1× cosine adjustment (use linear-scaling rule: lr * sqrt(K) is a reasonable starting heuristic). LoRA's expected behavior at higher effective batch is still well-understood (it just gets less noisy). ListMLE within-CS computation is the only mathematical change and is straightforward. Magnitude-balance calibration (`calibrate_loss_weights_at_init`) still runs at K=1.

## 4. What NOT to try

1. **Don't add `pin_memory()` per-tensor again.** The allocator contention will repeat. If you want pinned memory, pre-allocate per-scale pinned buffers in `_AsyncHorizonStream.__init__` (size = max_N × max_L × 18 × 4 bytes ≈ 9 MB per scale) and `.copy_()` numpy data into them, then `.to(device, non_blocking=True)` the pinned buffer. ONE allocation, reused forever.

2. **Don't try MP DataLoader workers again.** The 10 MB pickle round-trip per CS at your build rate (~25 ms/CS) wastes the time you save on parallelism. Multi-process is only worth the IPC if per-worker work is >100 ms.

3. **Don't enable `fused=True` AdamW.** Confirmed regression. Will be worth revisiting only after Fix-3 raises effective batch — at that point the AdamW step on accumulated grads might benefit.

4. **Don't `torch.compile` with `dynamic=True` and `fullgraph=False`.** That's the config that gave you no improvement *and* spawned 32 compile-worker processes. The path forward is `mode="reduce-overhead", dynamic=False` with **padded fixed-shape inputs**.

5. **Don't tune `attention_dropout`/`ffn_dropout` or pretend GradScaler is needed for bf16.** bf16 doesn't need GradScaler — your code correctly omits it. Don't add one "just in case."

6. **Don't profile with `nvprof` / `nsys` until you've applied Fix-1.** Removing the per-step peft housekeeping makes the actual hotspots visible. Profiling current code will show a long Python tail that hides the kernel-launch-bound pattern.

7. **Don't switch to `nn.MultiheadAttention`.** Your hand-rolled `RoPESelfAttention` with `F.scaled_dot_product_attention` IS the optimal kernel path on H200 (FlashAttention-2 backend). Don't regress.

## 5. Debug script to prove the root cause on H200

Run this on the H200 box before applying any fix. It will time each suspected hotspot in isolation and produce a side-by-side breakdown. Save as `daity/scripts/h200_diagnose.py`:

```python
"""Phase 3 H200 diagnosis — per-step hotspot timing.

Runs 200 warmup + 500 measured steps and prints a breakdown:
  - data-wait (queue.get block)
  - cs_to_device (H2D copy)
  - set_adapter + enable_all_lora_grads
  - forward (tokenizer + backbone + heads)
  - per_head_losses
  - backward
  - clip_grad_norm_ (incl. trainable_parameters walk)
  - optimizer.step + zero_grad

If forward dominates (~30 ms) and data-wait is small (~1 ms), you are
launch-overhead bound (Fix-2/3 applies). If data-wait dominates,
Fix-1's pinned-buffer rework is the target.
"""
from __future__ import annotations
import time, statistics, torch
from contextlib import contextmanager
# Adjust path/import to your project layout:
from daity.training.posttrain_loop import (
    RunConfig, _build_module, _build_optimizer, _AsyncHorizonStream,
)
from daity.data.cross_section import CrossSectionalSampler
from daity.data.parquet_store import ParquetStore

@contextmanager
def cuda_timer(timings: dict, name: str):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    timings.setdefault(name, []).append((time.perf_counter() - t0) * 1000)

def main(cfg: RunConfig) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    module = _build_module(cfg).to(device)
    optimizer = _build_optimizer(module, cfg)
    sampler = CrossSectionalSampler(
        feature_store=ParquetStore(cfg.feature_root), label_root=cfg.label_root,
        universe=cfg.universe, as_of=cfg.train_end,
        min_end_ts=cfg.train_start, window_bars=cfg.window_bars,
        drop_invalid_threshold=cfg.drop_invalid_threshold,
    )
    stream = _AsyncHorizonStream(sampler, cfg.cadence_weights, seed=0, device=device, queue_size=4)
    module.train()

    timings: dict[str, list[float]] = {}
    # Warmup
    for _ in range(50):
        h, cs = stream.next()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = module(cs, horizon=h)
            loss = module.compute_loss(out, cs, horizon=h)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Measured
    for _ in range(500):
        # data-wait + cs_to_device are inside stream.next()
        t0 = time.perf_counter()
        res = stream.next()
        torch.cuda.synchronize()
        timings.setdefault("stream.next() (data wait + H2D)", []).append(
            (time.perf_counter() - t0) * 1000)
        h, cs = res

        with cuda_timer(timings, "set_adapter + enable_grads"):
            module.peft_backbone.set_adapter(h)
            module._enable_all_lora_grads()

        with cuda_timer(timings, "forward (tokenizer+backbone+heads)"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = module(cs, horizon=h)

        with cuda_timer(timings, "per_head_losses"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = module.compute_loss(out, cs, horizon=h)

        with cuda_timer(timings, "backward"):
            loss.backward()

        with cuda_timer(timings, "clip_grad_norm_ (incl. trainable_parameters)"):
            if cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in module.trainable_parameters()],
                    cfg.gradient_clip_val,
                )

        with cuda_timer(timings, "optimizer.step + zero_grad"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    print(f"\n{'Phase':<48} {'p50 (ms)':>10} {'p90 (ms)':>10} {'count':>8}")
    print("-" * 80)
    total = 0.0
    for name, t in timings.items():
        p50 = statistics.median(t); p90 = sorted(t)[int(0.9*len(t))]
        total += p50
        print(f"{name:<48} {p50:>10.2f} {p90:>10.2f} {len(t):>8}")
    print("-" * 80)
    print(f"{'sum of p50s':<48} {total:>10.2f}")
    print(f"{'observed sps if no overlap':<48} {1000/total:>10.1f}")

if __name__ == "__main__":
    # Build a small fold config and call main(cfg). Pull from your existing
    # smoke-test fixture so this reproduces in CI on a CUDA box.
    raise SystemExit("Wire up cfg from your fold config and call main(cfg)")
```

**How to interpret.** If forward+backward sum to >40 ms with everything else small, RC-1 (launch overhead) is confirmed and Fix-2 (compile/CUDA graph) is the right target. If `stream.next()` dominates with >20 ms, RC-3 (sync H2D) is in play and the pinned-buffer rework is the right target. If `set_adapter` shows >2 ms, Fix-1 is clearly right.

Bonus pass — use `torch.profiler` to see kernel-launch overhead directly:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True, with_stack=True,
    schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs/h200"),
) as prof:
    for _ in range(50):
        # ... step ...
        prof.step()
```

Look at the trace: if you see **gaps between kernel completions on the CUDA stream**, that's the GPU waiting for the next launch. Those gaps are RC-1.

## 6. Expected end-state with all three fixes

| Stage | sps (LoRA) | GPU util | notes |
|---|---:|---:|---|
| Baseline (today) | 14 | 25% | RC-1 + RC-3 + RC-2 + RC-5 |
| + Fix-1 (direct `_active_adapter`) | 15-16 | 28% | -2 ms/step Python overhead |
| + Fix-2 (`compile mode=reduce-overhead`, padded shapes) | 30-40 | 50-65% | launch overhead absorbed |
| + Fix-3 (8-CS micro-batch) | **80-120** | **75-90%** | actual H200 saturation |

If Fix-3 lands cleanly, you should also reconsider whether you need bf16 autocast — at this throughput fp32 is fine for the small backbone, and you remove autocast Python overhead. Test both.

## 7. Open questions worth answering after Fix-2

- Is **AdamW with `foreach=True`** the right choice at higher effective batch, or should you switch to `foreach=False` once tensors are fewer? (At K=8 micro-batch with same param count, foreach is still better.)
- Is **mixed-horizon batching via `_mixed_batch_forward`** worth the per-layer adapter loop? Worth a 50-step ablation once Fix-3 is in.
- Does **dropping the vol head's `abs.clamp.log` preprocessing** (move to data layer) save the 3 kernels? Trivial change, ~50 µs/step. Worth doing while you're already touching the loss path.
- Should the **forecast-token slice** (`hidden[:, 0]`) be moved BEFORE `final_norm` so you skip the LayerNorm over 81 unused tokens? Tiny win, but easy.

## 8. The one-line summary you can paste in PHASE3.md

> H200 utilization at ~25% is **not** a data-pipeline or framework-tuning problem; it's a structural mismatch between batch size (N=205) and GPU capability (designed for B≥1k). Throughput is gated by per-kernel launch overhead in eager PyTorch (~600 launches/step × ~10 µs each). Fix priorities: (1) replace `peft.set_adapter()` with direct `_active_adapter` assignment to cut Python overhead, (2) `torch.compile(mode="reduce-overhead", dynamic=False)` with shape-padded inputs to capture CUDA Graphs and amortize launches, (3) micro-batch 4-8 cross-sections per opt step to raise effective B and reach 50%+ GPU util. Memory at 5/143 GB confirms massive headroom.
