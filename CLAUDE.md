# CLAUDE.md — daity_v3 conventions

This file orients future Claude sessions. Keep it short and current.

## Project state

- Design doc: `DESIGN.md` — approved 2026-05-09. Treat as source of truth for architecture decisions.
- Phase ledger: `STATUS.md` — single source of truth for what's done / in flight / blocked.
- Historical plan: `PLAN.md` — superseded by `DESIGN.md`. Do not edit; kept for context.

## Workflow rules

1. **Phase gate.** Each phase has explicit acceptance criteria in `DESIGN.md` §7. Do not start phase N+1 until phase N's acceptance is met *and* an independent reviewer agent has signed off (see `reports/reviews/`).
2. **No live broker integration in v1.** Paper-trading only. The `serve/broker_stub.py` interface is the only contract that future live adapters implement.
3. **Point-in-time correctness.** Every data reader takes an `as_of: datetime`. No row whose `ts > as_of` ever leaves the data layer. CI enforces this via `tests/leakage/`.
   - **Candle ts convention** (load-bearing for as-of joins): intraday intervals (`5m`/`15m`/`60m`) store `ts` as the candle's *open* time; `day` stores `ts` as the *close* time at **15:30 IST = 10:00 UTC**. Use `daity.data.candles.effective_close(interval, ts)` whenever joining intervals on `as_of` — never reason about open vs close per call site.
4. **Symbol identity is ISIN-based.** Never join on raw ticker. Use `symbol_id` from the symbol-master table.

## Code conventions

- Python 3.11. `uv` for env. `ruff` for lint+format. `mypy` for types.
- Polars is the default dataframe; pandas only at library boundaries that demand it.
- Config via Hydra (`configs/`). Never hard-code paths; resolve from config or env.
- All entry points are CLI scripts under `daity/scripts/` or `scripts/`. Each is < 200 LOC and delegates to library code.
- Tests: `tests/unit/` for pure logic, `tests/leakage/` for data-pipeline gates, `tests/smoke/` for tiny end-to-end.

## Secrets

- `GOOGLE_APPLICATION_CREDENTIALS` points at a service-account JSON. **Never commit.**
- `.gitignore` blocks `*sa-key*.json`, `*credentials*.json`, `.env`. Verify before adding any secret-shaped file.

## Reproducibility

- Every artifact (cache slice, checkpoint, backtest run) is tagged with `(git_sha, config_hash, as_of)`. The smoke test in CI verifies that the same triple produces the same result.

## What lives where

```
daity/data/        BQ ingest, Parquet cache, calendar, corp-actions, features, splits
daity/models/      backbone, tokenizer, heads, OB encoder, fusion, LoRA adapters, conformal
daity/training/    pretrain / posttrain / finetune loops, losses, schedulers
daity/eval/        metrics (IC/CRPS/calibration), backtest engines, frictions, capacity, tearsheet
daity/strategy/    cross-sectional intraday + swing strategies (shared by backtest and paper-trade)
daity/serve/       inference, paper-trade, monitor, broker_stub
```
