# Phase 1 Review — daity_v3 (Data Layer)

**Reviewer:** Independent reviewer agent
**Date:** 2026-05-10
**Scope:** Phase 1 deliverables per `DESIGN.md` §7 — `daity/data/*`,
`daity/scripts/*` (data-layer flavored), `tests/unit/`, `tests/leakage/`,
`configs/data/corp_actions.yaml`, the executed write-up notebook, and the
local 820-partition Parquet cache. Out of scope: Phase 0 audit script
(reviewer #1 + #2 already signed off).

---

## Executive summary

The data layer is in materially better shape than its volume of code might
suggest: the candle-ts convention (DESIGN amendment 7) is honored end-to-end
on the real ingested universe, the three leakage CI gates exist and exercise
the core invariants, the per-symbol Parquet layout (DESIGN amendment 8) is
sized and laid out as designed, and the feature engine (DESIGN amendment 9)
ships with bit-equality-on-truncation as one of its leakage tests — that's
the right test to write. 245 unit + leakage tests pass.

That said, Phase 1 has shipped with **two Critical correctness issues that
silently corrupt prod data on re-runs**, **one Critical that's the same SA-key
operational risk Phase 0 flagged and is still not addressed**, and a handful of
Major correctness / leakage-surface issues. The headline: `readjust_symbols.py`
is *not* idempotent despite a docstring claiming otherwise, the cross-section
read at the strategy layer's boundary bypasses the effective-close gate that
the feature engine applies, and the `is_provisional` flag on `SymbolMaster`
quietly flips False on ISIN backfill while seven other identity columns are
still missing — meaning a downstream caller checking `is_provisional` will
believe the master is complete when it is not. The third Critical (SA key on
disk in the working directory at `0644`) is unchanged from Phase 0 and is
still actionable.

**Recommendation: fix Critical items 1, 2, 3 first, then proceed to Phase
2.** Major items 4–11 should be addressed alongside Phase-2 work; none of
them blocks pretraining starting, but several will become harder to fix
once a checkpoint depends on the current behavior.

---

## Findings

### 1. [Critical] `readjust_symbols.py` is not idempotent — re-running double-applies the ratio

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/readjust_symbols.py:134-185`
- **What's wrong:** The `_apply` and `_build_apply_script` functions emit:
  ```sql
  UPDATE raw_ohlcv
     SET open=open*0.5, high=high*0.5, low=low*0.5, close=close*0.5,
         volume=CAST(volume / 0.5 AS INT64)
   WHERE symbol = 'HDFCAMC'
     AND DATE(ts, 'Asia/Kolkata') < DATE '2025-11-03'
  ```
  The `WHERE` clause is `(symbol, cutoff_date)` — purely date-based, with no
  guard on the price range. Running the same config twice multiplies all
  pre-cutoff prices by `0.5 * 0.5 = 0.25` and divides volume by `0.25`. The
  docstring at line 142–147 explicitly claims the script is idempotent
  ("each subsequent run finds 0 rows that match the post-update state, since
  the price ranges shift") — this is **flatly incorrect**. The WHERE clause
  is symbol/date-based; price-range conditions never enter it. The only
  reason a careful operator might not blow up prod is that the dry-run
  preview shows `n_rows` immediately, but a `--apply` re-run is one
  arrow-up-and-enter away from corrupting 558K rows.
- **Why it matters:** A 5-second mistake silently corrupts 558K rows of
  prod data. The data layer's whole acceptance gate (and Phase 2/3
  reproducibility) depends on `raw_ohlcv` being trustworthy. Any operator
  who re-runs the readjust to "make sure the IRB entry took" gets a 4×
  KOTAKBANK, a 4× HDFCAMC, and a 4× IRB — none of which trip a CI test.
- **Suggested fix (do all three):**
  1. **Add a journal table.** Before applying, INSERT a row into a new
     `raw_ohlcv_readjustments_journal` table: `(run_id, applied_at,
     symbol, cutoff_date_ist, ratio, n_rows_affected, sha_of_config)`.
     The script's `_apply` should fail loudly if there is already a row for
     `(symbol, cutoff_date_ist, ratio)` — that is the
     "already-applied" check.
  2. **Make the WHERE clause data-aware.** Augment with `AND open BETWEEN
     <pre_event_min> AND <pre_event_max>` derived from the staged Kite
     comparison snapshot. If a row's price is already in the post-event
     range, the WHERE clause excludes it.
  3. **Fix the docstring.** Strike the "idempotent" claim and replace
     with: "Re-running this script over rows that have already been
     readjusted will double-apply the ratio. Always inspect the journal
     table before --apply." (Or add the journal and *make* it idempotent —
     option 1 is the right path.)
  4. **Provide an UNDO path.** `_apply` should also emit the inverse
     UPDATE statement to a checkpoint file (`reports/readjust_<runid>.undo.sql`)
     so a fat-fingered run is recoverable in 5 lines of SQL rather than a
     re-ingest from Kite.

### 2. [Critical] `ParquetStore.read_cross_section` does not apply the effective-close gate — same-bar leakage path at the strategy layer

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/parquet_store.py:220-252`
- **What's wrong:** The cross-section reader filters only `ts == ts_norm`
  AND `ts <= cap`. For intraday scales it does not filter
  `effective_close(scale, ts) <= as_of`. Concretely: at `as_of = 09:30 UTC`
  asking `read_cross_section(scale='5m', ts=09:30 UTC, as_of=09:30 UTC)`
  returns the bar opening at 09:30 UTC — but that bar's `effective_close`
  is 09:35 UTC, so it is still in progress at 09:30 UTC. The strategy
  layer (DESIGN §4.3 — "intraday cross-sectional rebalance every 30 min")
  is the named consumer of this function in the docstring. If a strategy
  evaluates "rebalance at 15:00 IST" by calling `read_cross_section(scale='5m',
  ts=09:30 UTC, as_of=09:30 UTC)`, every symbol's open/high/low/close at
  the in-progress bar leaks into the rebalance signal.

  The feature engine (`features.py:_filter_by_effective_close`) gets this
  right by branching on `CLOSE_TIME_INTERVALS`. The leakage tests
  (`tests/leakage/test_features_leakage.py:test_features_never_emit_in_progress_intraday_bars`)
  prove the *feature* layer is safe — but bypass the cross-section layer
  entirely. Worse, the cross-section leakage test
  (`tests/leakage/test_parquet_leakage.py:test_cross_section_read_never_returns_future_rows`)
  uses `scale='day'` only (where the ts==close convention makes the bug
  invisible). No cross-section test exercises an intraday scale where the
  effective-close gate would fire.

- **Why it matters:** The doctring on `read_cross_section` advertises it
  as the path "used by ranking / strategy code that needs the full universe
  slice at a single bar boundary". Any Phase 5 backtest that consumes
  this directly (the design says it will) gets a same-bar lookahead leak
  per symbol per rebalance. This is exactly the "worst class of bug"
  CLAUDE.md and DESIGN.md §2.5 are written to prevent.
- **Suggested fix:**
  1. Add an `effective_close <= as_of` filter inside `read_cross_section`
     — branch on `scale in CLOSE_TIME_INTERVALS` exactly like
     `_filter_by_effective_close`. The intraday branch becomes:
     ```python
     # ts is open; effective_close = ts + duration. Bar is in progress at
     # as_of unless ts + duration <= as_of.
     if scale not in CLOSE_TIME_INTERVALS:
         duration = INTERVAL_DURATIONS[scale]
         if ts_norm + duration > cap:
             return _empty_frame()
     ```
  2. Add a test
     `test_cross_section_read_excludes_in_progress_intraday_bars` to
     `tests/leakage/test_parquet_leakage.py` that drives 50 random `(sym,
     scale='5m', ts, as_of)` triples where `as_of` lands strictly between
     `ts` and `ts + 5m`, and asserts the returned frame is empty.
  3. The single-symbol `read` has the same gap (also no
     effective-close filter); fix it the same way OR document loudly that
     callers must apply the gate. The current behavior is "consumer must
     filter" — fine, but the advertisement on `read_cross_section`
     ("strategy at rebalance time uses this") makes consumer-side filtering
     a foot-gun.

### 3. [Critical] SA key still resident in repo at `0644`, contradicting Phase 0 acceptance

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/.sa-key.json`
- **What's wrong:** Phase 0 review #1 (Critical #1) demanded rotation +
  relocation of the SA key to `~/.config/gcloud/daity-sa.json`. STATUS.md
  Phase 0 deferred recommendations ack the issue ("Rotate the current SA
  key in GCP IAM and relocate the canonical key …  still applies. Note:
  live runs since 2026-05-09 also surface `loose perms 644` …`chmod 600`
  if keeping it in-repo."). The key is still in the repo with mode `0644`
  on 2026-05-10. `.env.example` was correctly updated to recommend
  `$HOME/.config/gcloud/daity-sa.json` but the actual file in `.env` is
  presumably still pointing at `./.sa-key.json` based on the persistence
  of the file. The `_validate_sa_key` warning fires on every `load_env`
  call and is being ignored.
- **Why it matters:** Same reasons as Phase 0 review — service-account
  private keys grant full BigQuery + GCS access to
  `atomic-vault-476315-m3.daity_v3`. With Phase 1 now also using the GCS
  staging bucket `daity-raw-data-476315`, the blast radius has grown.
  Every agent or tool reading the working directory pulls the key. Phase
  1 added 6 new scripts that rely on the key path, so the temptation to
  "just leave it where it is" is now structural.
- **Suggested fix:** Same as Phase 0 review's Critical #1 (rotate, relocate,
  add a pre-commit hook). Do not call this "addressed" until: (a) `ls
  .sa-key.json` returns "no such file"; (b) `.env`'s
  `GOOGLE_APPLICATION_CREDENTIALS` points at a path outside the repo; (c) a
  pre-commit hook fails on `*-key.json`. Until then, every Phase 1 ingest
  run is operating on borrowed time.

### 4. [Major] `SymbolMaster.is_provisional=False` after ISIN backfill is misleading — 7 identity columns still null

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/symbols.py:151-203, 213-238`
- **What's wrong:** `SymbolMaster._build_frame` (and `_try_from_cache`)
  flip `is_provisional=False` if every row has a non-null `isin`. But
  `valid_from`, `valid_to`, `mcap_bucket`, `free_float_bucket`,
  `fno_eligible`, `listed_on`, `delisted_on` remain entirely null in the
  cached parquet — confirmed: `pl.read_parquet('data/cache/symbols/symbol_master.parquet')`
  shows all 7 columns 100% null across all 205 rows. The phase 1 writeup
  asserts "After Phase 1 the master is non-provisional" (line 88-91 of
  `notebooks/phase_1_writeup.py`), reinforcing the false impression that
  the master is complete.

  The class still tracks the gap correctly via `missing_columns` (which
  drops `'isin'` from the placeholder set when `is_provisional` flips
  False), so a careful caller can detect it. But the *primary* signal —
  the flag named "is provisional" — now lies. The original semantics
  (`is_provisional == "this master is incomplete; do not trust it"`) is
  silently re-defined to ("ISIN identity has been wired up").
- **Why it matters:** Phase 3+ static-covariate code will join on
  `mcap_bucket`/`fno_eligible`/etc. The natural pattern is
  `if not master.is_provisional: <use covariates>` — that branch will
  fire as soon as Groww ISIN backfill succeeded, but the covariates are
  all None. A strategy that depends on F&O eligibility (DESIGN §4.3
  "long-only variant for SLB-restricted names") gets `None`/silently-skips
  every check.

- **Suggested fix:** Either:
  - **(A) Tighten the flag.** `is_provisional = bool(missing_columns)` —
    flips False only when *all* placeholder columns are populated. Until
    then, the flag stays True even with ISINs done. Then add an
    `isin_resolved: bool` accessor for the narrower question.
  - **(B) Rename for clarity.** Drop `is_provisional`, replace with
    explicit `isin_resolved` and `static_covariates_resolved` booleans
    derived from `missing_columns`. The notebook write-up's "non-provisional"
    assertion becomes "ISIN-resolved" and is no longer load-bearing for the
    covariate question.

  Either way, fix the writeup line "After Phase 1 the master is
  non-provisional" — it currently misleads the reviewer.

### 5. [Major] `read_cross_section` SQL injection via `ts.isoformat()` — and DuckDB `read_parquet(glob)` bypasses the safe-symbol check

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/parquet_store.py:238-252`
- **What's wrong:** Two issues:
  1. `glob = self.root / scale / "*.parquet"` is interpolated into the SQL
     as `read_parquet('{glob}')`. If `self.root` ever contained an
     apostrophe (or backslash, or anything DuckDB parses), the SQL parser
     either errors or — worse on a malicious path — opens a file outside
     the cache root. `tmp_path` doesn't surface this in tests, but a
     user-installed cache root like
     `~/My Drive/data/parquet` (Mac iCloud) or `~/Project's data/parquet`
     could.
  2. `ts_norm.isoformat()` and `cap.isoformat()` are interpolated as
     literals into the `WHERE` clause. Both come from `_ensure_utc`, so
     they are well-formed datetimes — but the pattern is "format-strings
     becoming SQL", which is the same mistake the project's own
     guidance (CLAUDE.md, "no hardcoded paths; resolve from config or env")
     tries to avoid.
- **Why it matters:** Defense-in-depth. `SAFE_SYMBOL_RE` exists exactly to
  defend against exfiltration via path-traversal in symbol names; the
  `glob` path bypasses that gate by joining the cache root with `*.parquet`
  literally and DuckDB-globbing it. A malicious symbol named `*' OR 1=1; --`
  can't reach this codepath today, but the pattern is one refactor away
  from biting (e.g., if someone adds `read_one_symbol(symbol)` using the
  same DuckDB path).
- **Suggested fix:** Use DuckDB's parameterized query API:
  ```python
  files = sorted(str(p) for p in (self.root / scale).glob("*.parquet"))
  if not files:
      return _empty_frame()
  con.execute(
      "SELECT * FROM read_parquet(?) WHERE ts = ? AND ts <= ?",
      [files, ts_norm, cap],
  )
  ```
  DuckDB accepts a Python list of paths as the first parameter to
  `read_parquet` and will quote them safely; tz-aware datetimes are
  passed as native parameters and bind to TIMESTAMPTZ.

### 6. [Major] `pl.read_parquet` does not push down predicates — claim in `parquet_store.py:205-206` is inaccurate

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/parquet_store.py:205-213`
- **What's wrong:** The comment in `read` claims "Parquet's predicate
  pushdown on row group ts stats means range filters are effectively
  free." `pl.read_parquet(path)` reads the entire file into memory, then
  `df.filter(...)` runs in Polars memory. Predicate pushdown only
  happens via lazy frames (`pl.scan_parquet(...).filter(...).collect()`)
  or PyArrow with `filters=...`. For tiny per-symbol files (~6 MB each,
  per the `du -sh data/parquet/` output: 735 MB / 820 files ≈ 900 KB avg),
  this is a non-issue *today*. But:
  - The notebook's Section 8 liquidity heatmap loops over 205 symbols and
    reads each fully — an O(N) full scan when row-group pruning would have
    cut it to one row group of 24 rows for a 6-month window.
  - The `partition_meta` path (which DOES use the row-group stats
    correctly) is the right pattern to imitate in `read`.
- **Why it matters:** Phase 2 pretraining will hit this layer in a tight
  random-window loop. Adding a 200ms-per-read dataloader is the
  difference between a 4-hour and a 24-hour pretraining run.
- **Suggested fix:** Replace `pl.read_parquet(path)` with
  `pl.scan_parquet(path).filter(...).collect()` so the polars optimizer
  pushes the ts predicate down to the parquet reader. The
  `df[TS_COLUMN].dtype.time_zone is None` defense becomes a `with_columns`
  expression on the lazy frame. Confirm the fix with a targeted benchmark
  test that reads a 1-month window from a 7-year file and asserts wall-time
  is ≪ "read whole file" wall-time.

### 7. [Major] `read_ohlcv.py`, `phase0_audit.py`, `ingest_parquet.py`, `detect_corp_actions.py`, `fix_pointwise.py` all exceed the 200-LOC convention

- **Files:**
  - `daity/scripts/refresh_ohlcv.py` — 430 LOC
  - `daity/scripts/phase0_audit.py` — 673 LOC (out of scope but flagged)
  - `daity/scripts/ingest_parquet.py` — 313 LOC
  - `daity/scripts/detect_corp_actions.py` — 263 LOC
  - `daity/scripts/fix_pointwise.py` — 263 LOC
  - `daity/scripts/readjust_symbols.py` — 256 LOC
  - `daity/scripts/refresh_calendar.py` — 230 LOC
- **What's wrong:** CLAUDE.md mandates "Each is < 200 LOC and delegates to
  library code." Six of the eight Phase-1 scripts are over the limit;
  `refresh_ohlcv.py` is 2.2× the limit. The right pattern is for the
  script to do CLI parsing + orchestration only, with the actual
  staging/diff/merge/fetching factored out into `daity/data/`. Today,
  `_stage`, `_spot_check`, `_apply_merge`, `_candles_to_frame`,
  `_fetch_one`, `_build_gap_keys` all live in scripts and are not reused
  across the suite, even though the patterns are nearly identical between
  `refresh_ohlcv.py`, `fix_pointwise.py`, and `ingest_parquet.py`.
- **Why it matters:** The convention exists so library logic gets unit
  tests (which CLI scripts struggle to). Today the staging/diff/merge
  helpers in `refresh_ohlcv.py` are tested only via
  `test_refresh_ohlcv_frame.py` which covers `_candles_to_frame` only.
  `_spot_check` (which decides whether to abort the merge) and
  `_apply_merge` are untested in their own right; their failures will
  surface only on a live run.
- **Suggested fix:** Lift the shared staging/MERGE primitives into
  `daity/data/bq_staging.py` (or similar): `stage_ohlcv_frame(bq, frame,
  *, run_id) -> str` and `merge_staged_into(bq, staging, *, mode='insert'|
  'upsert')`. `refresh_ohlcv`, `fix_pointwise`, and `ingest_parquet` all
  call it, the per-script body shrinks under 200 LOC, and the new module
  gets unit tests with a mocked `bigquery.Client`.

### 8. [Major] `kite._to_candles` falls back to `tzinfo=timezone.utc` for naive datetimes — but Kite returns IST

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/kite.py:348-353`
- **What's wrong:** `if isinstance(d, datetime): ts_dt = d if d.tzinfo
  is not None else d.replace(tzinfo=timezone.utc)`. Kite's `historical_data`
  returns tz-aware `datetime` with `pytz.timezone('Asia/Kolkata')`, so the
  `tzinfo is not None` branch is what fires today. But:
  - The `elif isinstance(d, date)` branch on line 352 also assigns
    `ts_epoch = int(datetime(d.year, ..., tzinfo=timezone.utc).timestamp())`.
    That's wrong: Kite's daily DATEs are IST-naive midnights.
  - If the `kiteconnect` library ever changes to emit naive datetimes
    (e.g., a future version, or if the SDK changes timezone handling),
    every Phase 1 candle is silently shifted by 5h30m. There is no test
    that locks in the assumption "Kite returns tz-aware".

  Compounding: `_apply_daily_ts_shift` (which fires only for `interval
  in CLOSE_TIME_INTERVALS`) adds another `+15:30` shift on top. So a
  hypothetical naive Kite daily ts would produce a `ts_epoch` that is
  IST-treated-as-UTC, then shifted +15:30, ending up 5h30m too late.
- **Why it matters:** All ingest depends on `_to_candles` being correct.
  This is the leaf where every other tz invariant in the data layer
  bottoms out. A silent 5h30m shift on intraday is the kind of bug whose
  symptom is "the model overfits to a deterministic timing artifact" —
  exactly the failure mode Phase 2 risk-mitigation can't catch.
- **Suggested fix:** Treat naive `date`/`datetime` as IST (Kite's native
  zone), not UTC:
  ```python
  if isinstance(d, datetime):
      ts_dt = d if d.tzinfo is not None else d.replace(tzinfo=IST_TZ)
      ts_epoch = int(ts_dt.timestamp())
  elif isinstance(d, date):
      # Kite daily date is IST midnight (00:00 IST same day).
      ts_epoch = int(datetime(d.year, d.month, d.day,
                              tzinfo=IST_TZ).timestamp())
  ```
  Add a test in `test_kite_client.py` that exercises both branches with a
  naive datetime and asserts the resulting epoch matches IST-interpretation.

### 9. [Major] Calendar staleness "fix" is one-shot — there's no scheduled refresh, and `cal.last_day > today` will silently start failing

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/refresh_calendar.py:160-227`
- **What's wrong:** The `refresh_calendar.py` script takes `--forward-days`
  (default 365) and writes to BQ. The notebook (Section 3) asserts
  `cal.last_day > TODAY.date()`. This is a tripwire — once the calendar
  expires, every Phase 1 reader that calls `WalkForwardSplitter.splits(...,
  as_of=...)` near the calendar's edge silently produces fewer folds
  (folds with `test_end > cal.last_day` are dropped — see
  `splits.py:_build_split` returning `None` from `cal.shift(...)`).

  Today, with the refresh having been run, `cal.last_day` is somewhere
  in 2027 (forward-days=365 from May 2026). Once that date passes, the
  splitter starts dropping folds without warning. There's no monitoring,
  no scheduled job, and no CI test that asserts `cal.last_day > today +
  90 days` — only the notebook's assertion, which is run when a human
  re-executes it. The phase ledger lists "Calendar staleness fix" as
  done, but really only "do it once" is done.
- **Why it matters:** Phase 5+ is a year of running this system on a
  rolling basis. A silent "the splitter drops the most recent fold"
  failure mode is exactly the kind of issue that surfaces only when a
  research finding is two weeks old.
- **Suggested fix:**
  1. Add a CI test that asserts `NSECalendar.from_cache(...).last_day >
     date.today() + timedelta(days=180)` and is run *daily* (cron / GH
     Actions / cowork scheduled task). When it fails, the alert is
     "calendar refresh overdue".
  2. Have `WalkForwardSplitter.splits(...)` log a warning (not just
     drop) when `last_test_end > cal.last_day`.
  3. The `refresh_calendar.py` Makefile target ought to be `make
     refresh-calendar` (currently absent — see finding 13 below).

### 10. [Major] Group-B "dividend-like" drifts deferred without triage — 12 of 98 entries exceed 10% magnitude

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/configs/data/corp_actions.detected.yaml`
- **What's wrong:** The notebook (Section 10, "Known data-quality caveats
  (deferred)") describes the deferred items as "Group-B 'dividend-like'
  small drifts (~5-7% ratio across ~20 symbols near 2020-03 COVID era)".
  Inventory of the actual file disagrees:
  - 98 entries across 43 symbols (not ~20).
  - 12 of 98 entries have `|ratio - 1.0| >= 10%` — explicitly outside
    the "small drift" band described in the writeup. Examples:
    `ALPL30IETF` 1.158, `BPCL` 1.111, `COCHINSHIP` 0.870, etc.
  - These are the kind of magnitudes that signal real corp actions
    (special dividends, stock splits, name changes), not the
    "Kite-applies-dividends-prod-doesn't" drift the writeup hand-waves
    about.

  The detector script has correctly surfaced these. The decision to defer
  triage is fine in principle, but the writeup's *characterization* of
  the deferred items understates the magnitude and scope. A reviewer
  reading only the notebook will believe ~20 symbols × tiny dividends are
  out of scope; the reality is ~43 symbols, including a dozen with
  >10% deltas that are almost certainly missed corp actions.
- **Why it matters:** Phase 3 swing labels and Phase 5 backtests
  multiply small adjustment errors across the universe and across years.
  An undeclared 5% mis-adjustment on 43 symbols is a non-trivial source
  of Sharpe inflation. Phase 1's acceptance gate is "data layer is
  trustworthy"; if the writeup's accounting of remaining mis-adjustments
  is itself wrong, the gate isn't crossed honestly.
- **Suggested fix:**
  1. Update the writeup Section 10 with the accurate stats (98 entries /
     43 symbols / 12 large), and split the discussion into
     "small drifts (likely dividend-convention)" vs. "large drifts
     (probable missed corp actions, marked TODO)".
  2. Triage the 12 large entries before Phase 5 (cross-check NSE bhavcopy
     for the boundary date). The 86 small entries can stay deferred but
     should be tagged in the YAML with a `category: dividend_like` /
     `category: probable_corp_action` field so the deferral decision is
     itself auditable.
  3. Add a leakage-style CI test that asserts no detected entry with
     `|ratio - 1.0| >= 0.05` is *both* unannotated *and* unapplied — i.e.,
     forces the implementer to mark each large entry as one of
     `applied`/`deferred_dividend`/`deferred_for_phase5_triage` before
     the test passes.

### 11. [Major] `MAX_LOOKBACK_BARS = 60` is conservative for some scales but the engine reads the **whole** partition anyway

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/features.py:78, 144-151`
- **What's wrong:** The module documents `MAX_LOOKBACK_BARS = 60` as the
  warm-up needed for rolling features (max of `LOG_RETURN_LAGS` and
  `VOLUME_Z_WINDOW`). The comment at line 144-147 then reads "We
  deliberately read the whole partition (no `start` lower bound) so
  rolling features always have full warm-up. Per-symbol Parquet files
  are small (≤8 MB even for 5m × 7y), so the extra IO is negligible".
  Two issues:
  1. The "small files" claim contradicts `pq.ParquetFile('data/parquet/5m/RELIANCE.parquet').metadata.num_rows = 136,038`
     (verified). For a small 1-day window the engine reads 136K rows to
     emit ~75. The "negligible IO" claim is only true relative to the
     alternative; it's still 1000× the necessary work.
  2. The actual fix would be `start = window_start - MAX_LOOKBACK_BARS *
     bar_period`, but `bar_period` for `day` is *24 hours of clock time*
     even though the bar represents a trading day (~6h15m). The engine
     dodges the bar-period question by reading everything; the tradeoff
     is real, but the comment misrepresents it.

  Compounding with finding 6 (no predicate pushdown), every feature
  computation is O(whole partition) per call.
- **Why it matters:** Phase 2 pretraining + Phase 3 supervised heads will
  call `engine.compute(...)` once per (symbol × split × epoch). At 205
  symbols × 12 folds × 100 epochs that's ~250K calls. Even at "1 second
  per call" that's 70 hours of pure feature compute, dwarfing the actual
  GPU work.
- **Suggested fix:**
  1. Pass `start = window_start - MAX_LOOKBACK_BARS * <calendar-aware-step>`
     into the `store.read(...)` call. For daily, use the
     `NSECalendar.shift(window_start, -MAX_LOOKBACK_BARS)` to get the
     trading-day-correct lower bound. For intraday, use
     `MAX_LOOKBACK_BARS * INTERVAL_DURATIONS[scale]` and accept the
     overshoot across weekends (still a 100× win over reading
     everything).
  2. Combined with the lazy-scan fix in finding 6, the warm-up read
     becomes O(MAX_LOOKBACK_BARS + window_size) bars rather than
     O(whole-partition).
  3. Add a benchmark test under `tests/unit/` that times
     `engine.compute(window_start=t, window_end=t + 1 day, scale='5m')`
     and asserts wall-time is < 100ms; this anchors the optimization so
     Phase 2's dataloader has a real perf SLA to live up to.

### 12. [Minor] `effective_close` is intentionally not used inside `ParquetStore.read` — but the docstring and module split makes that opaque

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/parquet_store.py:181-218`
- **What's wrong:** `read` advertises "no row with `ts > as_of` ever escapes"
  but does not promise effective_close semantics. Callers reading the
  docstring + the prominent `daity/data/candles.py` module would
  reasonably assume `read(scale='5m', as_of=t)` returns only bars
  fully closed at `t`. It does not — it returns bars whose `ts <= t`,
  including the in-progress bar opening at `t`. This is consistent with
  finding 2 but worth flagging separately because the *single-symbol*
  `read` is the more common consumer.
- **Suggested fix:** Either fix the underlying behavior (apply
  `effective_close` here) or add a sentence to the docstring: "**Note:**
  this method enforces only `ts <= as_of`. For the stricter
  `effective_close <= as_of` (no in-progress intraday bars), use
  `FeatureEngine.compute` or filter via
  `daity.data.candles.effective_close` at the call site." Linking the
  two modules in the docstring closes the cognitive gap.

### 13. [Minor] No `make ingest` / `make refresh` / `make readjust` targets — bootstrap is undocumented

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/Makefile`
- **What's wrong:** Phase 1 added 6 new console-scripts (`daity-refresh-calendar`,
  `daity-kite-login`, `daity-refresh-ohlcv`, `daity-fix-daily-ts`,
  `daity-readjust-symbols`, `daity-detect-corp-actions`,
  `daity-fix-pointwise`, `daity-ingest-parquet`). None of them have a
  Makefile target. CLAUDE.md says "every artifact … is tagged with `(git_sha,
  config_hash, as_of)`" and Phase 1 acceptance is partially "full universe
  loadable". A new contributor / agent on a fresh checkout has to read 8
  scripts, the writeup notebook, and STATUS.md to figure out the bootstrap
  order. The right answer is one `make data` (or `make bootstrap-data`)
  target that runs:
  ```
  daity-refresh-calendar --apply
  daity-kite-login   # interactive
  daity-refresh-ohlcv --apply
  daity-fix-daily-ts --apply
  daity-readjust-symbols --apply
  daity-ingest-parquet
  ```
- **Why it matters:** Reproducibility (DESIGN §5.3) and onboarding.
  Without a single command, the dependency graph between scripts is
  tribal knowledge. The `make data` should also touch a `data/.bootstrap.json`
  with `{git_sha, ran_at, n_partitions}` so the as-of triple Phase 5
  needs is tracked from day one.
- **Suggested fix:** Add `data:` and `data-refresh:` targets that chain
  the scripts, with progressive `--apply` for the destructive steps. The
  `kite-login` step is interactive and should print a clear "press
  enter when done" prompt rather than silently expecting a paste.

### 14. [Minor] `survivorship_warning` not yet attached to anything queryable

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/symbols.py`
  (and DESIGN amendment 4)
- **What's wrong:** STATUS.md claims Phase 1 will "add a `survivorship_warning`
  field to every backtest tearsheet metadata and tracks the gap explicitly".
  No code in `daity/` mentions survivorship outside the audit script's
  staleness probe. The `SymbolMaster` would be the natural carrier
  (e.g., `SymbolMaster.is_survivorship_biased: bool = True`), but it
  currently has no such field. The notebook's Section 9 prose acknowledges
  the gap, but prose is not load-bearing.
- **Why it matters:** Phase 5 acceptance is "Walk-forward Sharpe > 1.0
  net at ₹10Cr AUM on at least one strategy"; a survivorship-flag-on-the-master
  ensures the tearsheet generator can't silently quote a single Sharpe
  without the disclaimer DESIGN §2.3 mandates.
- **Suggested fix:** Add `is_survivorship_biased: bool = True` to
  `SymbolMaster` (defaulting True until the universe is augmented with
  delisted history). Defer the tearsheet wiring to Phase 5 — the field
  just needs to exist now so it can be threaded through.

### 15. [Minor] `_validate_sa_key` warning is non-fatal — Phase 0 review suggested it should fail, not warn

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/utils/env.py:115-118`
- **What's wrong:** The `if mode & 0o077: log.warning(...)` permits
  loose-permed SA keys to be used. The Phase 0 review#2 verification
  ack'd this as "warn-not-fail is acceptable" but the persistence of the
  loose-permed key in this repo (finding 3) suggests warnings are
  ignored in practice. A `--strict-perms` config option (default True
  in CI, False locally) would let CI block the bad-perm path while
  allowing local dev to keep working.
- **Why it matters:** Keys live and breathe through the warning level.
  Either escalate to error in strict mode, or remove the warning since
  it isn't being acted on.
- **Suggested fix:** Add `DAITY_STRICT_SA_PERMS=true` env var (default
  False locally, set True in `make ci-test` or whatever CI target
  Phase 5 needs). When True, `EnvError` instead of warn.

### 16. [Minor] `_to_jsonable` skips numpy bool/int/float

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/bq.py:184`
- **What's wrong:** Phase 0 review#2's note N4 — `isinstance(v, (bool,
  int, float, str))` returns False for `numpy.bool_` / `numpy.int64`.
  Those flow into the final `return str(v)` rather than being preserved
  as primitives. Phase 0 says "not a regression for Phase 0 since BQ
  returns Python primitives, but worth knowing for Phase 1." Phase 1
  has now landed; the issue is unaddressed and will surface the moment
  any `_to_jsonable` consumer is given a polars/numpy intermediate.
- **Suggested fix:** Test for numpy scalars explicitly:
  ```python
  import numpy as np
  if isinstance(v, np.generic):
      return v.item()
  ```
  Add a unit test in `test_bq_jsonable.py` that pins this contract.

### 17. [Minor] `read_max_ts_per_symbol_interval` returns naive→UTC, but downstream code mixes naive UTC and tz-aware UTC

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/refresh_ohlcv.py:96-104`
- **What's wrong:** The function checks `if ts.tzinfo is None: ts =
  ts.replace(tzinfo=timezone.utc)` to normalize. But other paths
  (e.g., `_candles_to_frame` line 193) build tz-aware UTC from
  `datetime.fromtimestamp(c.ts_epoch, tz=timezone.utc)`. The mixing has
  been caught by `parquet_store.write_partition`'s `dt.replace_time_zone("UTC")`
  defense (line 157-160). But the defensive-coding sprawl is symptomatic:
  there are at least three places (parquet_store, refresh_ohlcv,
  partition_meta) that all individually re-normalize tz-naive→UTC.
  One central helper would prevent drift.
- **Suggested fix:** Promote `parquet_store._ensure_utc` to a
  module-level helper in `daity/data/__init__.py` (or `daity/utils/time.py`)
  and have every layer call it once at its boundary. Delete the
  parallel `if ts.tzinfo is None` checks.

### 18. [Minor] `_seeded_random()` in leakage tests uses a date-based seed but the date is arbitrary

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/tests/leakage/test_parquet_leakage.py:25-27`
  and `tests/leakage/test_features_leakage.py:32-33`
- **What's wrong:** `random.Random(20260510)` is a fixed-but-arbitrary
  seed. CI re-runs the same 100 triples deterministically, which means
  a real leak that *only* fires on a different draw never gets caught.
  DESIGN §2.5 ("spot-check 100 random triples") implies *new* triples
  per CI run; the stable seed makes it 100 *fixed* triples plus the
  illusion of randomness.
- **Why it matters:** A bug that survives the seed-20260510 draw
  survives forever. Adversarial leakage tests should rotate the seed
  per CI run (e.g., from `os.environ.get("CI_RUN_ID")` or
  `int(time.time())`), and surface the seed in the failure message so
  the failure is reproducible.
- **Suggested fix:**
  ```python
  def _seeded_random() -> tuple[random.Random, int]:
      seed = int(os.environ.get("LEAKAGE_TEST_SEED", str(int(time.time()))))
      return random.Random(seed), seed
  ```
  The first failing assertion includes the seed in its message; CI logs
  the seed on success. Determinism is preserved when needed (set the
  env var); coverage is broadened when not.

### 19. [Minor] No `tests/integration/` BQ smoke despite Phase 0 review #1 finding 13

- **File:** `tests/integration/` (does not exist)
- **What's wrong:** Phase 0 review #1 finding 13 said "Add a
  `tests/integration/test_bq_smoke.py` (skipped if
  `GOOGLE_APPLICATION_CREDENTIALS` is unset) that does a single
  `list_tables` call". Phase 1 added six BQ-touching scripts; none has
  an integration test. The `make test` filter `not integration and not
  slow` is therefore still a no-op against an empty filter set.
- **Suggested fix:** Add `tests/integration/test_bq_smoke.py` with one
  test that calls `bq.list_tables()` and asserts the canonical Phase 1
  tables exist (`raw_ohlcv`, `nse_trading_calendar`, `nse_holidays`,
  `symbol_list`, `sector_metadata`). Mark `@pytest.mark.integration`.
  Same idea for Kite — one test that calls `kc.lookup('RELIANCE')`,
  skipped if `KITE_ACCESS_TOKEN` unset.

### 20. [Nit] PIIND intraday Jan-2019 alternating bars left in production

- **File:** `notebooks/phase_1_writeup.py:367-371`
- **What's wrong:** "PIIND intraday 2018-12-31 .. 2019-01-09 — alternating
  ~860/~95 close pattern in *both* prod and Kite … Daily was fixed pointwise.
  Intraday left as-is; if it bites in Phase 2 SSL we'll mask those bars."
  This is a reasonable triage decision, but the masking mechanism
  doesn't exist yet. Phase 2 SSL will hit those bars unless someone
  remembers to mask. No code in `daity/data/` is currently aware of
  PIIND's bad-bar window.
- **Suggested fix:** Add a `data/known_bad_bars.yaml` with explicit
  `(symbol, scale, start_ts, end_ts)` entries; thread a
  `bad_bar_mask` field through the feature engine output (or just drop
  matched bars at read time). PIIND/5m/2018-12-31..2019-01-09 is the
  first entry. Mark it as a Phase-2 follow-up; just having the file
  prevents the "wait, what was that issue again" recovery.

### 21. [Nit] `notebooks/phase_1_writeup.py` re-fetches symbol-master + calendar from BQ with `force_refresh=True`

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/notebooks/phase_1_writeup.py:79-81, 112`
- **What's wrong:** Every notebook execution forces a BQ query for both
  the symbol_master and calendar — even though the parquet caches under
  `data/cache/` are deterministic and already populated. Cost is small
  (~$0.001) per run but the BQ-or-bust pattern means the notebook can't
  be re-executed offline (e.g., during a conf demo on flaky wifi).
- **Suggested fix:** Drop `force_refresh=True` and add a separate
  `daity-refresh-symbol-master` console script for explicit refreshes.
  The notebook becomes a pure analysis artifact again.

### 22. [Nit] `Holiday.source_year` typing is `int | None` but no test exercises None

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/calendar_nse.py:101`
- **What's wrong:** `source_year=(int(r["source_year"]) if r.get("source_year") is not None else None)`
  guards against missing source_year, but `tests/unit/test_calendar_nse.py`
  doesn't cover the None branch. A future schema change to
  `nse_holidays` that drops the column would silently land None
  everywhere with no test failure.
- **Suggested fix:** Add a 1-line test
  `test_holiday_with_null_source_year_passes_through` to
  `test_calendar_nse.py`. Cheap.

### 23. [Nit] The notebook's "stale symbols" assertion uses `< TODAY - 90d` but TODAY is hardcoded to 2026-05-09

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/notebooks/phase_1_writeup.py:67, 355-360`
- **What's wrong:** `TODAY = datetime(2026, 5, 9, ...)` is the
  notebook's anchor for *all* "fresh" assertions. After 2026-08-07 the
  90-day staleness check passes only if the data layer has been refreshed
  ≥ daily — but TODAY doesn't roll forward. The notebook assertion
  `n_stale == 0` will become a false-positive ("looks fine") even when
  prod is 6 months out of date.
- **Suggested fix:** Either replace `TODAY = datetime(2026, 5, 9, ...)`
  with `TODAY = datetime.now(tz=timezone.utc)`, OR derive TODAY from
  the max ts in the parquet cache (so the notebook's "today" is always
  "freshest data you have"). Either makes the staleness check adaptive.

---

## Things done well

- **The candle-ts convention is enforced cleanly.** `daity/data/candles.py`
  is the right shape — small, documented, single source of truth, with
  `effective_close` as the join helper. The intraday-vs-daily asymmetry is
  the right call (storing daily at the open would be a foot-gun); the
  module docstring explains it well. Confirmed end-to-end on real data:
  RELIANCE day bars at 10:00 UTC = 15:30 IST, RELIANCE 5m bars on the
  09:15-IST grid.
- **The leakage gates are pointed at the right invariants.**
  `test_features_leakage.py:test_rolling_features_no_future_peek` is the
  *right* test to write — bit-equality between full-history and
  truncated-history feature values pins the no-future-peek invariant
  more strongly than any "no row > as_of" test could. Three gates × ~12
  tests is proportional to the surface area.
- **The Parquet layout decision (DESIGN amendment 8) is well-justified
  and matches reality.** 820 files / 735 MB / monthly row groups is
  what the design called for; per-symbol mmap-friendliness is the right
  optimization for the dominant Phase 2 access pattern. `partition_meta`
  uses Parquet footer stats correctly (the one place row-group pruning
  actually fires today).
- **`SAFE_SYMBOL_RE`** is appropriately conservative; the test
  `test_partition_path_rejects_unsafe_symbol` covers `../etc`,
  `FOO/BAR`, etc. The `M&M` / `BAJAJ-AUTO` allow-list test is the
  right counter-balance.
- **The `WalkForwardSplitter`** has the right contract: `Split.__post_init__`
  enforces ordering at construction so a malformed fold blows up loudly
  rather than silently producing leakage. Trading-day-aware embargoes
  (not calendar-day) is the correct design and the test
  `test_embargo_is_in_trading_days_not_calendar_days` pins it.
- **The notebook is real.** `phase_1_writeup.ipynb` exercises every
  headline piece on the actual ingested universe (not synthetic
  fixtures), surfaces the deferred caveats, and the 7 PASS markers /
  10 asserts all fire on real data. That's the right thing for a
  Phase write-up.
- **Defensive ts-handling in ParquetStore.** The `if df[TS_COLUMN].dtype.time_zone
  is None: dt.replace_time_zone("UTC")` defense in `read` is the right
  pattern given the BQ-EXPORT-vs-per-query history (BQ EXPORT writes
  tz-naive UTC; per-query reads return tz-aware UTC — the codebase
  has been bitten by this and recovered).
- **The corp-action detector + write-config-then-apply split** is the
  right architectural pattern. Detection is auto, application is manual,
  and the YAML schema is a stable interface between them. The 98
  auto-detected entries weren't auto-applied — that's a feature, not
  a bug.
- **`SymbolMaster._try_from_cache` reconstructing `is_provisional` from
  data, not dataclass defaults.** The bug-discovery anecdote ("a cache
  written after ISIN backfill would round-trip to is_provisional=True")
  is exactly the kind of latent-trap-found-by-notebook DESIGN §7's
  notebook acceptance criterion is supposed to catch. (Modulo the
  `is_provisional` flag's broader semantic problem — see finding 4.)
- **The fix-daily-ts script's collision pre-check.** Pre-flighting
  `_verify_no_target_collision` before applying the +15:30 shift is the
  right paranoia level for a destructive operation. The fact that the
  script is dry-run by default + has the collision guard means a
  re-run can't double-apply (the second run finds 0 rogue rows).

---

## DESIGN.md amendment hygiene

- **Amendment 7 (candle-ts convention):** Accurate. `effective_close` is
  used in the feature engine and tests; intraday-vs-daily semantics
  match the on-disk reality. Two minor gaps: (a) `read_cross_section`
  doesn't apply it (finding 2), (b) consumer-of-`read` could trip on
  it if the docstring isn't expanded (finding 12). The amendment text
  itself is fine.
- **Amendment 8 (Parquet cache layout):** Accurate. 820 files / per-(scale,
  symbol) / monthly row groups, all confirmed on disk. The "200-500ms
  cross-sectional read" claim is currently optimistic — without the
  predicate-pushdown fix in finding 6, cross-section reads will be
  slower in practice; needs benchmarking.
- **Amendment 9 (5 feature groups):** Accurate. The five groups
  (bar_channels / log_returns / wicks / volume_z / time_covariates) are
  all present and tested. The "RevIN moves to tokenizer" deferral is
  the right call. Microstructure-lite, static covariates, days-to-expiry
  remain genuinely deferred; the `_add_*` plug-in pattern in `features.py`
  makes the deferral cheap to undo.

Original DESIGN §2.5 ("the leakage CI test … spot-checks 100 random
triples") is implemented as three gates rather than one, with appropriate
specialization. That's a fair interpretation, not a silent invalidation.

DESIGN §2.4 ("RevIN per (symbol, window)") is now contradicted by amendment
9 ("RevIN moves into the tokenizer"); §2.4 should be cross-referenced or
struck through. Cosmetic.

---

## Reproducibility

A fresh checkout cannot rebuild the data layer with one command. The
implicit chain is:

```
make dev
# (manually) populate .env, .grow_cred.env, .kite_cred.env
daity-refresh-calendar --apply         # (phase ledger says done; no make target)
daity-kite-login                        # interactive
daity-refresh-ohlcv --apply             # ~5 min
daity-fix-daily-ts --apply              # one-shot data-shape fix
daity-readjust-symbols --apply          # corp actions
daity-ingest-parquet                    # BQ → GCS → local Parquet
```

CLAUDE.md mandates `(git_sha, config_hash, as_of)` tagging on every
artifact. None of the above scripts emit a `(triple)` to a registry, so
"the same triple produces the same result" is unverifiable for Phase 1
artifacts. This is a Phase-5+ concern but tagging *should* be in scope
for Phase 1 since the artifacts are what Phase 2+ consumes — see finding
13's suggested `data/.bootstrap.json`.

---

## Verdict

**Recommendation: fix Critical items 1, 2, 3 before proceeding to Phase
2.**

Critical 1 (`readjust_symbols.py` non-idempotency) and Critical 2
(`read_cross_section` effective-close gap) are correctness bugs that will
silently corrupt the inputs to Phases 2-5. Critical 3 (SA key in repo) is
the same operational risk Phase 0 flagged that has not been remediated;
with Phase 1 having added more credential-using scripts, the "fix it
later" debt is compounding.

Major items 4–11 should land alongside Phase 2 work; finding 4
(`is_provisional` semantics) and finding 6 (no predicate pushdown) in
particular will become much harder once Phase 2 dataloaders depend on
the current behavior.

Minor items 12–19 and Nits 20–23 can land opportunistically.

Headline numbers — 245 tests passing (54 unit + 11 leakage on the data
layer + the Phase 0 carry-overs), full-universe Parquet cache (820 / 735 MB)
populated, calendar/symbol-master/splits all wired and tested — *are*
real and the Phase 1 acceptance gate ("leakage CI green; full universe
loadable; one EDA notebook committed") is technically met. The Critical
items are not in the acceptance criteria as written, but they are in the
spirit of what the gate is trying to enforce: a data layer that is
trustworthy under re-runs, free of leakage in its advertised consumer
paths, and operationally safe.
