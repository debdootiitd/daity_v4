---
name: Sweep v1 auto-loop pipeline
description: Where the auto-loop sweep system lives, what it produces, how the trading-analyst agent should access per-config insights
type: reference
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
The Phase 5 sweep system runs an automated label-mine → train → backtest loop across config matrices. Lives at `daity/scripts/sweep_runner.py`. Configs in `reports/sweep_v1/configs.json` (hot-reloadable — workers re-read each iteration).

**Persistent state on vast.ai (`/root/daity/reports/sweep_v1/`):**
- `configs.json` — the queue of configs to run (id, label barriers, train hparams, backtest barriers). Append to this file to add new experiments — workers pick them up automatically.
- `claimed/<cfg_id>` — atomic per-config lock (mkdir-based). Worker creates on claim, leaves on completion. Stale claims after crash → remove `claimed/<id>` to allow re-processing.
- `done/<cfg_id>` — completion marker. Worker creates after full backtest grid + insight written. Workers skip configs with this marker.
- `results.csv` — flat per-backtest results. One row per (config, backtest_variant, cost_variant). Streams live.
- `insights.md` — **structured per-config insight blocks** with: label/backtest barrier params, R:R, train metrics, full backtest table, and auto-inferred observations (e.g., "X cells positive with N≥10", "win<50% + R:R<1.2 = structurally negative").
- `worker_{A,B,C}.log` — per-worker stdout.

**For the trading-analyst agent on each loop iteration:**
1. SCP `insights.md` from vast.ai to local: `scp -P 19497 root@175.155.64.226:/root/daity/reports/sweep_v1/insights.md /tmp/insights.md`
2. Read it locally and EMBED full contents in the analyst's prompt (the Plan subagent has Read but no SSH)
3. Also pass `results.csv` (or fetched leaderboard) for numerical comparison
4. Analyst proposes new configs in the same JSON schema; append to `configs.json` on vast.ai

**Schema for a config entry in configs.json:**
```json
{"id": "X1_descriptive",
 "label_tgt_pct": 0.01, "label_stop_pct": 0.005,
 "label_entry_offset_min": 5, "label_h_max_min": 120, "label_cost_bps": 19,
 "train_lr": 1e-4, "train_max_steps": 12000,
 "backtest_target_pct": 0.01, "backtest_stop_pct": 0.005, "backtest_h_max_min": 120}
```
Note: label barriers can differ from backtest barriers — that's the "train tight / deploy wide" lever (and inverse).

**Workers are idempotent:** mine_labels checks if parquets exist (skip), train_model checks if `model.state.best.state.pt` exists (skip), run_backtest checks if report JSON exists (skip). Safe to restart anytime.

**Add more workers anytime** by launching `nohup .venv/bin/python scripts/sweep_runner.py worker_X &`. Memory headroom on RTX 6000 Ada (49 GB) handles 4-6 parallel trainings comfortably.
