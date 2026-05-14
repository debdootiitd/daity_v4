---
name: Maximize GPU utilization
description: Always try to keep the GPU busy. Run sweep variants in parallel when memory headroom allows. Never let the GPU sit idle when there are useful experiments queued.
type: preference
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
When running ML training sweeps on vast.ai GPU (or any GPU):

**Default to parallel execution when feasible:**
- Check GPU memory headroom (`nvidia-smi --query-gpu=memory.used,memory.total`). For training that uses < 30% of VRAM, 2-3 processes in parallel typically gives 1.4-1.7× throughput.
- Each parallel process uses its own CUDA context — just launch separate `nohup bash` wrappers with distinct W&B run names.
- Even at 70% per-process GPU util, two processes overlap compute usefully because each has bursty patterns (forward → backward → data load).

**Auto-queue next sweep when one finishes:**
- Don't wait for me to ask — if the current sweep has converged or finished, kick off the next planned iteration automatically.
- Use a wait-and-launch wrapper script: watch the sweep log for the "complete" marker, then start the next sweep with `nohup`.
- Iterate continuously on the experimentation plan; only stop when results plateau or I explicitly stop the loop.

**Don't be conservative about compute usage:**
- vast.ai is paid per hour — idle GPU time is wasted money.
- If unsure whether a config will help, just run it. The marginal cost of one more sweep is small compared to the cost of waiting for me.
- Prefer "fire-and-forget" sweeps with intermediate eval over carefully-staged confirmations.

**When to break this rule:**
- If a sweep is clearly diverging or broken, kill it before queuing more.
- If memory could OOM with parallel runs, drop to serial.
- If timing/wall-clock comparisons matter (rare), run serially.
