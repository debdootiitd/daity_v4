---
name: Strategies must be LONG-only
description: All trading strategies, configs, and backtest variants must be long-only. Do not propose short-only or both-side configs.
type: feedback
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
All trading strategies in this project are LONG-only. Do not propose or queue configs with `backtest_side: short_only` or `both`. Do not include short-side analysis in trading-analyst loops.

**Why:** User stated explicitly during sweep iteration 3 (2026-05-13) after I queued a `K2_short_only_tightTgt` config; they removed it and said "we should only have Long only strategies." Likely a regulatory/operational constraint (Indian retail short-selling restrictions, intraday-only short permitted, or a personal trading mandate) — but in any case, the directive is unambiguous.

**How to apply:**
- When the trading-analyst Plan agent runs, the brief MUST exclude any "test the SHORT side" / "directional asymmetry" / "short_only mirror" proposals.
- Never set `backtest_side` in any sweep config; default `long_only` is the only valid value.
- When analyzing why win rates are low or AvgW < AvgL, do NOT recommend testing the short side as a diagnostic.
- The encoder still emits `short_score` heads as part of the ranking loss (it's a multi-head model), but downstream deployment and backtests are long-only.
