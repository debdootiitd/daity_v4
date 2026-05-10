# Phase 1 Review — Response to Findings

**Author:** Implementer
**Date:** 2026-05-10
**Source:** `phase_1_review.md` (3 Critical + 8 Major + 8 Minor + 4 Nit findings)

This document records the disposition of each Phase-1 review finding. The
3 Criticals are blockers per the reviewer's verdict; the 8 Majors were
optional-pre-Phase-2 but I addressed all of them. Minors and Nits are
listed at the end with disposition.

**Headline:** all 3 Criticals + all 8 Majors fixed. 262 tests pass (was
245). Lint applied. Re-review request follows.

---

## Critical findings — all fixed

### #1 — `readjust_symbols.py` not idempotent

- **Fix:** added a marker file `data/applied_readjustments.json` recording
  every applied `(symbol, cutoff_date_ist, ratio)` triple. New helpers in
  `daity/data/readjust.py`: `marker_key`, `load_marker`, `filter_already_applied`,
  `record_applied`. CLI guard: refuses to re-apply marked entries unless
  `--allow-rerun` is passed.
- **Backfill:** `data/applied_readjustments.json` ships with the 3 Phase-1
  readjustments (HDFCAMC, KOTAKBANK, IRB) so a fresh checkout doesn't
  double-apply them on a system that already had them.
- **Tests:** 8 new unit tests in `test_readjust_symbols.py` covering marker
  key stability, cache round-trip, atomic write, end-to-end double-apply
  blocking, plus a default-config sanity check that the 3 Phase-1 entries
  are present.
- **Live verification:** `uv run daity-readjust-symbols` now prints
  "ALREADY APPLIED" for HDFCAMC / KOTAKBANK / IRB and exits without
  touching prod.

### #2 — `ParquetStore.read_cross_section` skipped effective_close gate for intraday

- **Fix:** rewrote the gate to `effective_close(scale, ts) > cap` instead of
  `ts > cap`. For intraday scales this catches the in-progress bar; for
  daily it's identical to the prior check (since `ts` already encodes the
  close).
- **Combined fix for #5:** parameterized the DuckDB query (`?` placeholders)
  so neither the path glob, the ts, nor the cap can become an injection
  vector. The glob itself is built from a `_check_scale`-validated name +
  the project's cache root — user-controlled input never reaches the SQL.
- **Tests:** 2 new leakage tests in `test_parquet_leakage.py` (one
  deterministic boundary case, one 100-triple random sweep on intraday).
  Both would catch the original bug.

### #3 — SA key at mode 0644 + Phase 0 recommendation unaddressed

- **Fix:** `chmod 600 .sa-key.json` applied. README updated with the
  recommended `~/.config/gcloud/daity-sa.json` relocation (with a one-liner
  `mv` + `sed` to update `.env`).

---

## Major findings — all fixed

### #4 — `SymbolMaster.is_provisional` lying after partial backfill

- **Fix:** `is_provisional` is now a `@property` derived from
  `bool(missing_columns)`. The invariant `is_provisional == bool(missing_columns)`
  cannot drift. Added `has_column(name)` for per-column gates so callers
  prefer the fine-grained API over the coarse flag.
- The flag now stays True until ALL identity columns (ISIN, valid_from,
  valid_to, mcap_bucket, free_float_bucket, fno_eligible, listed_on,
  delisted_on) are populated.
- **Tests:** updated `test_isin_backfill_drops_isin_from_missing_columns`
  + new `test_is_provisional_only_false_when_all_columns_populated`,
  `test_has_column_returns_false_when_in_missing_columns`,
  `test_cache_round_trip_preserves_missing_columns`.

### #5 — SQL injection in `read_cross_section`

Combined with #2 above. Parameterized via `con.execute(sql, [glob, ts, cap])`.

### #6 — Wrong predicate-pushdown comment in `parquet_store.read`

- **Fix:** switched from `pl.read_parquet(path).filter(...)` to
  `pl.scan_parquet(path).filter(...).collect()` — actual lazy predicate
  pushdown. Polars now skips row groups whose ts range is outside
  `[start, cap]`, so per-symbol time-range queries on 5m × 7y partitions
  scan ~1-3 row groups instead of all ~80.
- Updated comment to be accurate; added a defensive tz-handling note
  explaining the BQ-EXPORT-naive-vs-old-aware schema variance and how the
  filter works in both cases.

### #7 — 5 scripts violate <200 LOC convention

- **Fix:** extracted data-layer logic into 3 new lib modules under
  `daity/data/`:
  - `ohlcv_io.py` — shared BQ ops (stage / merge / candle frame conversion /
    spot-check / max-ts read) used by `refresh_ohlcv` + `fix_pointwise`.
  - `parquet_ingest.py` — GCS export / download / per-symbol split logic
    used by `ingest_parquet`.
  - `corp_action_detect.py` — segmentation + proposal builder used by
    `detect_corp_actions`.
  - `readjust.py` — config / marker / WHERE-clause helpers used by
    `readjust_symbols`.
- **LOC after refactor (vs limit 200):**
  - `refresh_ohlcv.py`: 430 → 195 ✓
  - `fix_pointwise.py`: 263 → 167 ✓
  - `ingest_parquet.py`: 313 → 185 ✓
  - `detect_corp_actions.py`: 263 → 142 ✓
  - `readjust_symbols.py`: 371 → 189 ✓
- **Tests:** existing tests updated to import from the new lib paths;
  zero behavior change, 262 tests still pass.
- **Note:** `phase0_audit.py` (673 LOC) deliberately untouched — flagged
  by reviewer's broader sweep but explicitly out of Phase-1 scope per the
  review's "out of scope" header. Will revisit in a Phase-0 cleanup pass.

### #8 — `kite._to_candles` defaults naive datetime to UTC, not IST

- **Fix:** changed `tzinfo=timezone.utc` → `tzinfo=IST_TZ` for both the
  datetime and date branches in `_to_candles`. Kite's native zone is IST;
  if the SDK ever returns a naive value, treating it as IST is the only
  correct default.
- **Test:** renamed `test_to_candles_handles_naive_datetime_as_utc` →
  `_as_ist`, updated assertions to use IST. The new test fails if the
  default ever drifts back to UTC (which would silently shift every bar
  by 5h30m).

### #9 — Calendar staleness has no scheduled refresh

- **Fix:** new `daity-check-staleness` CLI exits non-zero if:
  - `nse_trading_calendar.last_day` is less than `--calendar-min-lead`
    days past today (default 30), OR
  - any scale's `raw_ohlcv` max-ts is more than `--ohlcv-max-staleness`
    *trading* days behind today (default 5).
- README + script docstring document a recommended cron schedule
  (Mon-Fri 13:00 UTC = 18:30 IST, after market close).
- **Live test:** currently reports ALL FRESH (calendar 365d lead, all 4
  scales 0 trading days stale).

### #10 — 12 detected corp-action drifts >10% magnitude

- **Triage:** investigated each entry. 100% false positives:
  - **PIIND (3 entries, Jan 2019)** — known data-entry alternating-bar
    bug, already documented. Not a corp action.
  - **NMDC, COCHINSHIP, KPITTECH, BPCL, PSUBNKBEES (8 entries, all 2020-03)**
    — paired reciprocal ratios on adjacent days during COVID circuit-
    breaker volatility. Not real corp actions; web-confirmed NMDC had no
    2020-03 corp action (next bonus was Dec 2024).
  - **ALPL30IETF (1 entry, 2023-01-27)** — ETF NAV reset / dividend
    distribution, not a multiplicative corp action.
- **Detector hardening:** added `--min-segment-days` filter (default 3) to
  `proposed_readjustments`. Rejects transitions where either adjacent
  segment is shorter than N days — suppresses single-day vol-spike false
  positives. All 12 original drifts would be filtered by default now.
  2 new tests lock in the filter behavior.

### #11 — Feature engine reads entire partition, not bounded window

- **Fix:** when `window_start` is provided, `FeatureEngine.compute` now
  passes `read_start = window_start - WARMUP_CALENDAR_DAYS[scale]` to
  `ParquetStore.read`, so the lazy Parquet reader's row-group pruning
  kicks in.
- **Per-scale warmup table:** 5m=7, 15m=7, 60m=21, day=100 calendar days
  for the 60-bar lookback. Sized to overshoot for safety; row-group
  pruning makes the cost of overshooting negligible.
- When `window_start is None` we still read the whole partition (no
  usable lower bound).
- **Tests:** 2 new tests using a `monkeypatch` spy on `ParquetStore.read`:
  one verifies the bounded read uses the per-scale warmup, one verifies
  the unbounded case still reads everything.

---

## Minor + Nit findings — disposition

| # | Finding | Disposition |
|---|---|---|
| 12 | `effective_close` not used inside `ParquetStore.read` | Kept as-is (per-symbol reads use `ts <= as_of`; the effective-close gate is applied at consumer time by `FeatureEngine._filter_by_effective_close`). Comment in `parquet_store.py` clarifies this split now. |
| 13 | No `make ingest` / `make refresh` targets | Deferred to Phase 2 — the bootstrap sequence is short (kite-login, refresh-calendar, refresh-ohlcv, ingest-parquet) and the per-script `--help` covers it. Will revisit when Phase 2 needs a one-command "fresh checkout" demo. |
| 14 | `survivorship_warning` not attached to anything queryable | Deferred — adding it now would over-design before Phase 5 strategy code knows what it wants. STATUS.md / DESIGN.md still mention it. |
| 15 | `_validate_sa_key` warns instead of failing on loose perms | Kept as warn (the `chmod 600` happened in fix #3). Failing here would block users on first run; a warn-then-recommend is better DX. |
| 16 | `_to_jsonable` skips numpy scalars | Carried from Phase 0; not exercised in Phase 1 paths. Deferred. |
| 17 | Mixed naive/aware datetime conventions in some helpers | Reduced via the broader normalization in fixes #6 and #8. Not exhaustive; will follow up if any leak surfaces. |
| 18 | `_seeded_random()` uses fixed seed | Intentional — deterministic failures are easier to triage. Could change to date-based later. |
| 19 | No `tests/integration/` BQ smoke | Out of scope; added `daity-check-staleness` as a lighter-weight production canary in #9. |
| 20 | PIIND intraday Jan-2019 data left in production | Documented as a known caveat in `phase_1_writeup.py` §10. Both prod and Kite have the same broken alternating-bar pattern; cannot fix from current sources. |
| 21 | Notebook re-fetches symbol-master + calendar with `force_refresh=True` | Kept — the notebook is a writeup-and-canary, deliberately reflecting current BQ state. |
| 22 | `Holiday.source_year` typing is `int \| None` but no test exercises None | Trivial; will add when next touching `calendar_nse.py`. |
| 23 | Notebook's "stale symbols" assertion uses hardcoded `TODAY = 2026-05-09` | Acknowledged — will fix at next notebook refresh. Treats writeup as a snapshot rather than a live monitor. |

---

## Test count

- Before fixes: 245 passing
- After fixes: **262 passing** (17 new tests covering the regressions)
- Lint: 56 warnings remain (mostly long-line style nits in older files);
  130 fixable issues auto-fixed.
