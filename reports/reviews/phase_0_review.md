# Phase 0 Review — daity_v3 (Bootstrap + BigQuery audit)

**Reviewer:** Independent reviewer agent
**Date:** 2026-05-09
**Scope:** Phase 0 deliverables per `DESIGN.md` §7 — code, layout, audit-script
correctness. Out of scope: runtime audit results (audit not yet executed).

---

## Executive summary

The repo skeleton matches the design's `§5.1` layout, the env/credential
plumbing is clean, and the column-resolution heuristics in
`daity/scripts/phase0_audit.py` are sensibly factored and tested. However, a
**Critical** finding exists: a real Google service-account private key is
already on disk at `./.sa-key.json` despite `STATUS.md` claiming the SA key drop
is pending. While `.gitignore` does block this filename, the repository is not
yet a git repo, the key is sitting in a directory tree the user's tooling
copies into Cowork sessions, and the project_id and client_email are now
exposed to any tool/agent reading the working directory. There is also one
**Major** correctness issue (timezone handling in the OB symbol-day census, the
single headline number Phase 0 must lock) and a handful of Major SQL
robustness issues. Recommendation: **fix Major items first, then proceed to
Phase 1.** The Critical secret-handling issue is operational, not in the code
itself, but must be remediated before any further work because the locked
acceptance number for OB sizing depends on a key that is now potentially
exposed.

---

## Findings

### 1. [Critical] Live SA private key is committed-in-waiting at `./.sa-key.json`

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/.sa-key.json`
- **What's wrong:** The file exists and contains a real RSA private key for
  `daity-scraper@atomic-vault-476315-m3.iam.gserviceaccount.com` (project_id
  `atomic-vault-476315-m3`, key id `1c9dad04025eaa86ccdc72f02a45a620ef59a5c0`).
  `STATUS.md` line 23 states "*pending SA key drop*", and the review brief
  itself says the key has not been placed — both are now stale. The repo is
  not yet a git repo (no `.git/`), so `.gitignore` does not protect the file
  *yet*; the moment `git init && git add .` runs without first removing the
  file, it will be committed even if the ignore patterns are correct, because
  `git add .` shows a warning but adds files only if they aren't already
  ignored — and the patterns *do* ignore this exact name, so the immediate
  commit risk is low. However:
  1. The key is being copied into agent/tooling sessions on every read of the
     working directory (it has already been copied into this review session).
  2. Ad-hoc backups, IDE syncs, Time Machine, and Spotlight indexing pick it
     up.
  3. If a user ever runs `git add -f .sa-key.json`, the gitignore won't stop
     them, and there is no pre-commit hook or `git secrets` install to catch
     it.
- **Why it matters:** Service-account private keys grant full BigQuery access
  to `atomic-vault-476315-m3.daity_v3`. Leakage = data exfil + billing risk.
  This is the single highest-impact failure mode of the entire project's
  bootstrap.
- **Suggested fix (do all four):**
  1. **Rotate the key now.** Treat it as compromised. In GCP IAM, delete key
     id `1c9dad04025eaa86ccdc72f02a45a620ef59a5c0` for
     `daity-scraper@…`, generate a new one, place it on disk in a path
     *outside* the repo tree (e.g. `~/.config/gcloud/daity-sa.json` with
     mode `0600`), and update `.env` to point `GOOGLE_APPLICATION_CREDENTIALS`
     at the absolute external path.
  2. **Move the canonical SA-key location out of the repo** in
     `.env.example` and `README.md`. Replace
     `GOOGLE_APPLICATION_CREDENTIALS=./.sa-key.json` with
     `GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/daity-sa.json`.
     This eliminates an entire class of "agent copies the working directory"
     leaks.
  3. **Add a pre-commit hook** that fails on any file matching
     `*-key.json`, `*.pem`, or detects `BEGIN PRIVATE KEY` blocks. The
     `detect-secrets` pre-commit, or a 5-line shell hook, suffices.
  4. **Update `STATUS.md`** to either reflect that the key has been placed
     (after rotation + relocation), or remove the misleading "pending SA key
     drop" line.

### 2. [Critical] `.sa-key.json` is checked by `_pick`/heuristics-blind audit but the audit script trusts the file's *contents* without parsing

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/utils/env.py:80-90`
- **What's wrong (defense-in-depth):** `load_env` only checks
  `cred_path.exists()`. If a non-key file ends up at that path (e.g. an
  empty placeholder, a partial download, an attacker's substituted file),
  `BQClient(...)` will fail later with an opaque google-auth error and the
  audit reports "Audit failed: <traceback>" without telling the operator why.
  More importantly, there is no check that the file is at restrictive perms
  (`0600`) — on macOS a SA key written at the default `umask 022` is
  world-readable.
- **Why it matters:** This is the second leg of the SA-key story. Even after
  rotation, restrictive perms + early validation block silent
  misconfiguration and reduce blast radius.
- **Suggested fix:** Add to `load_env` after the `cred_path.exists()` check:
  ```python
  if cred_path.exists():
      mode = cred_path.stat().st_mode & 0o777
      if mode & 0o077:
          log.warning("SA key %s has loose perms %o; chmod 600 recommended",
                      cred_path, mode)
      try:
          payload = json.loads(cred_path.read_text())
          if payload.get("type") != "service_account":
              raise EnvError(f"{cred_path} is not a service_account key file")
      except json.JSONDecodeError as exc:
          raise EnvError(f"{cred_path} is not valid JSON: {exc}") from exc
  ```
  (Promote to Critical only because of pairing with finding 1; in isolation
  this would be Major.)

### 3. [Major] OB symbol-day census uses `DATE(ts)` with no timezone — likely off-by-one across NSE trading days

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:206-217, 235-254`
- **What's wrong:** `DATE(ts)` in BigQuery returns the UTC date when `ts` is
  a TIMESTAMP. NSE trades 09:15–15:30 IST (i.e. 03:45–10:00 UTC), so the
  trading day fits inside one UTC date — *unless* the snapshot rate produces
  pre-open auction or post-close ticks, OR the timestamps are stored in IST
  with a TIMESTAMP type (which BQ stores as UTC and `DATE(ts)` would shift
  back). If `ts` is DATETIME stored in IST naive, `DATE(ts)` returns IST
  date — fine. If TIMESTAMP, the resulting date may straddle a
  03:30 UTC crossover for any pre-open or post-close ticks.
  The **symbol-day count is the headline Phase 0 acceptance criterion** — it
  decides whether OB fine-tuning happens at all (`§3.4` requires 50–500
  symbol-days; fewer triggers risk-mitigation `§8.1`). An off-by-one across
  the day boundary would inflate or deflate this count by up to 2× depending
  on the snapshot rate.
- **Why it matters:** This is the number Phase 0 exists to lock. Getting it
  wrong by even a 10–20% factor changes the Phase-4 plan.
- **Suggested fix:** Use `DATE(ts, 'Asia/Kolkata')` everywhere a "trading
  day" is computed. Two call sites:
  ```python
  # symbol_day_count
  out["symbol_day_count"] = bq.query_scalar(
      f"SELECT COUNT(DISTINCT CONCAT(CAST({sym_col} AS STRING), '|', "
      f"CAST(DATE({ts_col}, 'Asia/Kolkata') AS STRING))) FROM `{fq}`"
  )
  # daily_symbol_count
  ... DATE({ts_col}, 'Asia/Kolkata') AS date ...
  # snapshot_rate_seconds — partitioning
  ... PARTITION BY {sym_col}, DATE({ts_col}, 'Asia/Kolkata') ORDER BY {ts_col} ...
  ```
  Note: `DATE(timestamp, tz)` works only for TIMESTAMP columns. If `ts_col`
  is DATETIME, `DATE(ts_col)` is correct as-is *if* the producer stores IST.
  Recommend resolving the type from `meta.schema` and branching:
  ```python
  ts_field = next(c for c in cols if c["name"] == ts_col)
  if ts_field["type"] == "TIMESTAMP":
      day_expr = f"DATE({ts_col}, 'Asia/Kolkata')"
  else:  # DATETIME / DATE
      day_expr = f"DATE({ts_col})"
  ```
  …and reuse `day_expr` in all three queries. This also de-risks finding 4.

### 4. [Major] `TIMESTAMP_DIFF` and `DATE(...)` will fail on a DATETIME column

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:112-127, 235-254`
- **What's wrong:** BigQuery's `TIMESTAMP_DIFF` requires both arguments to be
  TIMESTAMP. If `curated_ohlcv.ts` (or `order_book_depth.ts`) is stored as
  DATETIME (a common choice for naive IST bars), the staleness query and the
  snapshot-rate query both raise
  `Argument type DATETIME does not match TIMESTAMP`. The script would catch
  this for the adjustment-hint query (try/except) but **not** for the
  staleness query or snapshot-rate query — those propagate up and abort the
  whole audit, leaving the operator with the headline output unproduced.
  The OB sample-row inspection shows `DATETIME` for the OB `ts` column in
  one of the candidate datasets, so this is non-hypothetical.
- **Why it matters:** A type error here means the audit run fails on the
  first attempt and yields nothing. `STATUS.md`'s "schema confirmed" gate is
  not crossed.
- **Suggested fix:** Resolve the column type once and emit the right diff
  function:
  ```python
  ts_field = next(c for c in cols if c["name"] == ts_col)
  diff_fn = "TIMESTAMP_DIFF" if ts_field["type"] == "TIMESTAMP" else "DATETIME_DIFF"
  ```
  Then format `diff_fn` into the staleness and snapshot queries. Same for
  the `MAX({ts_col})` scalar subquery (no change there — both types support
  MAX). Wrap each individual aggregate query in its own try/except so a
  single failure doesn't abort the whole audit (the adjustment-hint query
  already does this; do it for staleness, snapshot-rate, and rows-by-year as
  well).

### 5. [Major] Adjustment hint silently returns zero counts when timeframe filter matches nothing

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:134-156`
- **What's wrong:** When `tf_col` is detected, the script restricts to
  `'1d','d','day','daily','eod','1day'`. If `curated_ohlcv` stores
  daily bars under, say, `'D1'`, `'1D'` (uppercase before lower-cast handles
  it, fine), `'EOD_NSE'`, `'eq_day'`, or any other label, the WHERE matches
  nothing, the CTE returns empty, and the SELECT returns zeros / NULLs. The
  audit then prints
  `adjustment hint (large-return frequency): {... n_returns: 0 ...}` which
  *looks* like "no big returns => prices are adjusted" — a false positive
  for the most important Phase 0 question (DESIGN.md §2.3 "Corporate
  actions").
  Additionally:
  - `LAG(close)` orders by `{ts_col}` only, not `(symbol, ts_col)`. The
    `PARTITION BY {sym_col}` is correct so the order is per-symbol — fine.
  - When `tf_col` is *absent*, the LAG mixes 1m / 5m / 15m / 1d into one
    series, and the "ABS(ret) > 0.30" count becomes meaningless (intraday
    returns of 30%+ are real on circuit-locked names).
  - No guard against `close = 0` halts producing 100% drops on the next
    valid bar (numerator = 0 / non-zero = 0 → ret = -1).
- **Why it matters:** This is the Phase 0 answer to "is OHLCV adjusted?" — a
  precondition for trustworthy swing labels. A silent zero is worse than a
  failure.
- **Suggested fix:** (a) detect the densest timeframe per the actual values
  (use `rows_by_timeframe` and pick the longest-bar one — or the one with
  the median row count if heuristics confuse you); (b) fall back to "all
  rows" only with a warning attached; (c) filter out non-positive close.
  ```python
  daily_filter = ""
  if tf_col and out.get("rows_by_timeframe"):
      # Pick the modal label that looks daily, else fall back to all rows
      daily_labels = {"1d","d","day","daily","eod","1day","1day_eq",
                      "day_eq","eq_day","eod_nse"}
      candidates = [r["timeframe"] for r in out["rows_by_timeframe"]
                    if str(r["timeframe"]).lower() in daily_labels]
      if candidates:
          quoted = ",".join(f"'{c}'" for c in candidates)
          daily_filter = f"WHERE CAST({tf_col} AS STRING) IN ({quoted})"
      else:
          out["warnings"].append(
              f"No daily-looking timeframe in {[r['timeframe'] for r in out['rows_by_timeframe']]}; "
              "adjustment hint computed across all timeframes (treat with skepticism)."
          )
  ```
  And in the CTE itself: `... FROM ` + `` `{fq}` ` `` + ` {daily_filter} ` +
  `WHERE close > 0`. Combine the filters properly (the current `f-string`
  concatenation will produce double-WHERE if both fire — fix that too).

### 6. [Major] Audit cannot survive missing `close` column without aborting

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:145`
- **What's wrong:** The adjustment-hint query hardcodes the column name
  `close`. If `curated_ohlcv` uses `c`, `close_price`, `Close`, or
  `adj_close`, the query errors. The try/except catches it and adds a
  warning — that's fine, but the script's "must answer adjustment status"
  acceptance criterion silently fails.
- **Why it matters:** Phase 0 acceptance demands "adjustment/survivorship
  status known". A graceful failure is not an answer.
- **Suggested fix:** Add `CLOSE_CANDIDATES = ("close", "close_price", "c",
  "adj_close", "Close")` next to the other candidate tuples and resolve at
  the top of `audit_ohlcv`. Skip the hint with a clear warning if no
  close-like column is found, rather than relying on the broad except.

### 7. [Major] `bq.sample` falls back on broad `Exception` which masks credential / network failures

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/bq.py:121-130`
- **What's wrong:** `TABLESAMPLE SYSTEM (1 PERCENT)` requires the source
  table to be larger than ~1 partition (~1GB) and to NOT be a clustered
  table without a partition filter. NSE OHLCV / OB data may well be both
  large and clustered. When TABLESAMPLE fails the script silently falls back
  to `LIMIT N` which yields the *first physical N rows* — typically all the
  same symbol or all the same date, which makes the "sample for visual
  inspection" misleading. Worse, the bare `except Exception` also catches
  auth errors, billing errors, and transient 5xx errors, swallowing them as
  "TABLESAMPLE failed".
- **Why it matters:** The sample is one of the few human-eye outputs of the
  audit; a non-representative sample (all rows from one symbol) makes
  schema sanity-checking unreliable. Auth/network errors that masquerade
  as TABLESAMPLE failures send a false "things mostly worked" signal.
- **Suggested fix:**
  ```python
  from google.api_core.exceptions import BadRequest, NotFound
  def sample(self, table: str, n: int = 100) -> list[dict[str, Any]]:
      sql = f"SELECT * FROM `{self.cfg.fq_table(table)}` TABLESAMPLE SYSTEM (1 PERCENT) LIMIT {int(n)}"
      try:
          return self.query_rows(sql)
      except BadRequest as exc:
          log.warning("TABLESAMPLE not supported (%s); falling back to ORDER BY RAND() LIMIT %d", exc, n)
          sql_fb = (
              f"SELECT * FROM `{self.cfg.fq_table(table)}` "
              f"WHERE RAND() < 0.001 LIMIT {int(n)}"
          )
          return self.query_rows(sql_fb)
  ```
  This narrows the catch and uses a probabilistic random sample as a more
  honest fallback.

### 8. [Major] `_to_jsonable` lossily floats large NSE volumes

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/bq.py:133-143`
- **What's wrong:** Any object with `__float__` that is not int/float/bool
  is float-cast. `decimal.Decimal` from BIGNUMERIC columns and large
  integer-typed volumes (BQ INT64 returns Python `int` so it skips this
  branch — fine), but for NUMERIC volumes (e.g. trade counts up to 1e10)
  cast through `float` loses precision past 2^53 (≈9.007e15). For NSE
  intraday volume aggregates this is unlikely to bite in *Phase 0*, but the
  helper is a foundation that gets reused.
- **Why it matters:** Reusable utility with an actual data-loss bug. Phase
  1 BigQuery ingest will lean on the same wrapper for spot checks; getting
  it right now costs nothing.
- **Suggested fix:** Special-case `Decimal` to pass through as `str` (JSON
  has no native decimal; string preserves precision) instead of float-cast:
  ```python
  from decimal import Decimal
  def _to_jsonable(v: Any) -> Any:
      if v is None or isinstance(v, (int, float, bool, str)):
          return v
      if hasattr(v, "isoformat"):
          return v.isoformat()
      if isinstance(v, Decimal):
          return str(v)
      if isinstance(v, (bytes, bytearray)):
          return v.hex()
      return str(v)  # fallback: never explode the JSON dump
  ```

### 9. [Major] BQ default location `US` will fail if the `daity_v3` dataset is `asia-south1`

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/.env.example:10`
  and `daity/utils/env.py:62`
- **What's wrong:** Default `DAITY_BQ_LOCATION=US`. Indian-data BigQuery
  datasets are commonly created in `asia-south1`. If the actual dataset is
  in `asia-south1`, `bigquery.Client(location="US")` will return
  `404 Not found` for both `list_tables` and `get_table`, confusing the
  operator into thinking the SA key has the wrong perms.
- **Why it matters:** First-attempt-fail experience for the entire team.
- **Suggested fix:** In `BQClient.__init__`, after constructing the client,
  fetch the dataset's actual location and warn-on-mismatch:
  ```python
  ds = self._client.get_dataset(self.cfg.fq_dataset)
  if ds.location.lower() != cfg.location.lower():
      log.warning(
          "Configured DAITY_BQ_LOCATION=%s but dataset is in %s; using dataset's location",
          cfg.location, ds.location,
      )
      self._client = bigquery.Client(project=cfg.project, location=ds.location)
  ```
  Also: change `.env.example` default to leave `DAITY_BQ_LOCATION` *unset*
  with a comment "(auto-detected from dataset; override only if needed)".

### 10. [Minor] `query_scalar` over a billions-of-rows table is a billing risk

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:86-88, 197-199`
- **What's wrong:** `SELECT COUNT(DISTINCT {sym_col}) FROM ...` over the
  full `curated_ohlcv` (potentially billions of rows) runs without a
  partition filter and bills for the full distinct-aggregation. This is
  cheap relative to a SELECT * but not free; if you run the audit many
  times during iteration, the bill adds up.
- **Suggested fix:** Use `APPROX_COUNT_DISTINCT` for the audit (off-by-1%
  is fine here), or use `SELECT COUNT(*) FROM (SELECT DISTINCT {sym_col} ...)`
  with `LIMIT 50000` to cap. Better: query
  `INFORMATION_SCHEMA.PARTITIONS` for size hints first and gate the scan.
  At minimum, set
  `bigquery.QueryJobConfig(maximum_bytes_billed=10 * 2**30)` (10 GiB) on
  the BQClient default to fail noisy-and-loud rather than silently
  expensive.

### 11. [Minor] `run_audit` raises `SystemExit` for "tables missing" instead of routing through the script's error handler

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:274-278`
- **What's wrong:** `raise SystemExit(msg)` from inside a library function
  conflates "library found wrong shape of dataset" with "CLI is asking the
  process to exit". The CLI handler in `main` only catches `EnvError` and
  bare `Exception`, so SystemExit slips out as a zero-context one-liner.
- **Suggested fix:** Define a richer exception class
  (e.g. `AuditError(RuntimeError)`) for "expected tables not present" and
  let `main` print a useful message including which tables WERE found and
  what the dataset configuration was.

### 12. [Minor] `_to_jsonable` does not normalize `bytes` / `bytearray`

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/data/bq.py:133-143`
- **What's wrong:** A BYTES column in a sample row would slip into
  `json.dumps` and raise `TypeError`. Even though BYTES is rare in OHLCV /
  OB, the audit is generic "show all sample rows".
- **Suggested fix:** Covered in finding 8's snippet.

### 13. [Minor] No `slow` / `integration` markers actually applied

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/pyproject.toml:77-81`
  and `tests/`
- **What's wrong:** The markers are declared but no test in the tree uses
  them. `make test` filters on `not integration and not slow`, which is
  effectively a no-op right now. Fine while there are no integration tests,
  but the marker definitions create dead-code that might mask a real
  filter mistake later.
- **Suggested fix:** Add a `tests/integration/test_bq_smoke.py` (skipped if
  `GOOGLE_APPLICATION_CREDENTIALS` is unset) that does a single `list_tables`
  call and asserts the expected tables exist. Mark it `@pytest.mark.integration`.

### 14. [Minor] `tests/__init__.py` and four other init files are empty zero-byte files

- **What's wrong:** Per pytest conventions, package-style test folders
  don't need `__init__.py` (pytest's rootdir collection picks them up via
  `testpaths`). Empty `__init__.py` here is harmless but creates noise; if
  you do want them, add a brief docstring for consistency.
- **Suggested fix:** Either remove or add a one-line docstring so they
  match the convention used in `daity/utils/__init__.py`.

### 15. [Minor] `audit_ohlcv` uses `f-string` interpolation of column names — brittle if a column ever has odd characters

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py` (multiple)
- **What's wrong:** Column names are pulled from BQ schema and interpolated
  unquoted (`SELECT {sym_col} ...`). NSE column names are unlikely to
  contain spaces or reserved words, but `timestamp` is a reserved keyword
  in BigQuery and would need backticks (`` `timestamp` ``).
- **Suggested fix:** Wrap each interpolation in backticks:
  `` SELECT `{sym_col}` AS symbol ... ``. Also escape backticks defensively
  (column names can't contain backticks per BQ spec, so that's belt-and-
  suspenders).

### 16. [Minor] No coverage of `BQClient.query_rows` / `_to_jsonable` in unit tests

- **What's wrong:** `tests/unit/` covers the env loader and the column-
  resolution heuristics, but `_to_jsonable` (which has actual
  type-coercion logic worth testing) and `query_scalar`'s "expected one
  column" guard are uncovered.
- **Suggested fix:** Add a small unit test that mocks `bigquery.Client` (or
  uses a fake row object) and exercises `_to_jsonable` for datetime,
  Decimal, bytes, int, float, bool, None — pin the contract before Phase 1
  consumes it.

### 17. [Nit] Logging configuration mutability

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/utils/logging.py:15-16`
- **What's wrong:** `_CONFIGURED: Final[dict[str, bool]]` is a sentinel —
  fine, but `Final` on a `dict` only marks the *binding* as final, not the
  contents (which `_configure_root` mutates). Common pattern; reads as a
  small inconsistency.
- **Suggested fix:** Use a module-level boolean with `global`, or a
  `threading.Event`. Cosmetic.

### 18. [Nit] `daily_filter` SQL concatenation is brittle

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/daity/scripts/phase0_audit.py:146`
- **What's wrong:** `FROM ``{fq}`` {daily_filter}` produces
  `FROM `proj.ds.t` WHERE …` correctly, but if you ever need to add a
  second WHERE clause (see finding 5) you'll get `WHERE … WHERE …`. Use a
  list of conditions and join with `AND`.
- **Suggested fix:** Build conditions as `clauses: list[str]` and emit
  `("WHERE " + " AND ".join(clauses)) if clauses else ""`.

### 19. [Nit] `STATUS.md` is out of sync with reality

- **File:** `/Users/debdoot/Documents/Claude/Projects/daity/STATUS.md:23`
- **What's wrong:** "*pending SA key drop*" is no longer true — the key
  file exists. (See finding 1.)
- **Suggested fix:** Update once the SA key is rotated and relocated.

---

## Things done well

- **Repo layout matches `DESIGN.md` §5.1 verbatim.** All package
  directories present with explanatory `__init__.py` docstrings.
- **Column-resolution heuristic is well-factored and well-tested.**
  `_pick` + tuple-of-candidates pattern is the right shape; case-insensitive
  match plus first-wins ordering is the right policy; tests pin the
  behavior cleanly.
- **`load_env` does the right things** in the right order: load `.env`,
  validate, resolve to absolute path, push back into the process env so
  google libs see it. The `strict=False` mode for tests is well thought out.
- **`.gitignore` is paranoid in the right way.** Multiple overlapping
  patterns (`sa-key.json`, `.sa-key.json`, `*sa-key*.json`,
  `*service-account*.json`, `*credentials*.json`, `*-key.json`, `.env`,
  `.env.*` with a `!.env.example` allowlist). This *is* the canonical
  pattern; I cannot find a plausible filename it would miss.
- **`pyproject.toml` is appropriately conservative** — Python 3.11 pin,
  numpy upper-bound (avoids the `numpy>=2.2` interop issues with
  pandas/polars), torch and W&B in an optional `ml` extra so the audit
  can run without GPU stack.
- **`CLAUDE.md`** captures the right invariants (phase gate, ISIN identity,
  PIT correctness) for future sessions.
- **The console summary** uses `rich.Table` for the schema, which makes the
  most important Phase 0 output (the schema) actually readable. Good
  attention to operator UX.

---

## Out of scope (will revisit in later phases)

- **Symbol-master / ISIN identity table** — not yet written; that's a
  Phase 1 deliverable per `DESIGN.md` §2.2. Phase 0 only needs to confirm
  the column names. Note for Phase 1 review: the audit script's
  `SYMBOL_CANDIDATES` includes `'symbol'`, `'ticker'`, `'symbol_id'`,
  `'instrument'`, `'scrip'`, `'tradingsymbol'` — when Phase 1 builds the
  symbol-master, the actual physical column should drive the ISIN join key
  selection, not the heuristic order here.
- **Leakage tests / `tests/leakage/`** — empty placeholders. Phase 1
  acceptance per `DESIGN.md` §2.5 is "leakage CI green"; Phase 0 doesn't
  need them.
- **Storage Read API path** — explicitly deferred to Phase 1 per the
  docstring of `bq.py`. Confirmed scope. The `query_rows` warning "Use
  only for small results" is appropriate.
- **Reproducibility tagging** (git_sha + config_hash + as_of triple) is a
  Phase 1+ concern; Phase 0 audit doesn't need it but the audit JSON
  *could* embed git_sha cheaply for traceability — recommend adding when
  Phase 1 begins.
- **`adjustment_hint` interpretation logic** — the audit produces the
  numbers but doesn't classify "looks adjusted" / "looks unadjusted". That
  classification + the "trigger Phase 1 corp-actions table build"
  decision can live in Phase 1's analysis notebook; Phase 0 just needs the
  numbers correct (see findings 5, 6).

---

## Verdict

**Recommendation: fix Major items first, then proceed to Phase 1.**

The Critical issues (1, 2) are operational and require rotating the SA key
and relocating it before any further development. They do not block code
review of the audit script, but they block running the audit safely.

Major items (3, 4, 5, 6, 7, 8, 9) directly affect whether the audit's
output answers Phase 0's acceptance question correctly. Items 3–6 in
particular determine whether the OB symbol-day census number that gates
Phase 4 is trustworthy. These should be fixed before the audit is run for
the official Phase 0 result, not after.

Minor / Nit items can land alongside Phase 1 work without holding the
gate.

---

## Reviewer #2 — verification of fixes (2026-05-09)

### Executive summary

I verified the implementer's fixes for Critical #2 and Major #3 through #9
against the changed files (`daity/utils/env.py`, `daity/data/bq.py`,
`daity/scripts/phase0_audit.py`, `.env.example`, and the new
`tests/unit/test_bq_jsonable.py`). All eight Major-or-Critical items the
implementer set out to address are properly fixed, and several flagged
Minor items (#10 billing budget, #11 AuditError, #15 backticks, #18 clauses
list) were also picked up. The fixes are surgical, type-aware, and the new
unit tests pin the JSON-coercion contract before Phase 1 consumes it. I
found one Low-severity edge case (a `None`-slice in the SA-key identity log
line) and one Low-severity dead variable (`day_expr` is computed but unused
in `audit_ohlcv`); neither is a blocker. Critical #1 (SA key on disk) is
acknowledged as an operational decision and not re-flagged. Recommend
**approve to proceed to audit run + Phase 1**.

### Per-finding verification

**#2 SA-key validation — addressed.** `_validate_sa_key` in
`daity/utils/env.py:45-80` performs the perm-warning, JSON parse, and
`type == 'service_account'` checks the original finding asked for. It also
logs the resolved identity (client_email + truncated key_id) without
exposing the private key, which improves on the suggested fix. Strict and
non-strict paths are wired correctly (`load_env` lines 130-135). New unit
tests `test_load_env_strict_with_creds`, `_rejects_bad_json`,
`_rejects_non_service_account` cover the contract. Minor edge case noted
under "new findings".

**#3 IST-aware day expression — addressed.** `_ts_helpers` in
`phase0_audit.py:87-103` returns `DATE(ts, 'Asia/Kolkata')` for TIMESTAMP
columns, and the resulting `day_expr` is used in `symbol_day_count`
(`audit_order_book` line 334), `daily_symbol_count` (line 343), and the
snapshot-rate `LAG(...) PARTITION BY {qsym}, {day_expr}` (line 383). NSE
trading-day boundaries no longer slip across UTC midnight.

**#4 TIMESTAMP/DATETIME branching — addressed.** `_ts_helpers` returns the
correct `diff_fn` for TIMESTAMP/DATETIME/DATE; staleness (`audit_ohlcv`
lines 191-210) and snapshot rate (`audit_order_book` lines 377-400) now use
`{diff_fn}`. Crucially, every aggregate query is wrapped in `_safe`
(`phase0_audit.py:105-112`), so a single type mismatch no longer aborts
the whole audit — exactly the behavior the original finding called for.

**#5 Adjustment hint silent-zero — addressed.** The hint now
(`audit_ohlcv` lines 213-269): (a) is skipped entirely with a warning if no
close-like column resolves, (b) restricts to daily timeframe labels *only
when such labels actually exist* in `rows_by_timeframe` and otherwise
records an explicit "treat with skepticism" warning, (c) adds `close > 0`
to the WHERE clauses, and (d) builds the WHERE via a `clauses: list[str]`
joined with `AND` so multiple conditions compose cleanly. The false-positive
"prices are adjusted" risk is gone.

**#6 close column resolution — addressed.** `CLOSE_CANDIDATES` is defined
at module scope (`phase0_audit.py:45`), resolved via `_pick`, and a missing
close column produces an explicit warning and skips the hint
(`audit_ohlcv` lines 217-221). The acceptance criterion now fails loudly
rather than silently.

**#7 `bq.sample` exception narrowing — addressed.** `BQClient.sample`
(`bq.py:156-175`) catches only `gcp_exceptions.BadRequest`; auth, 5xx, and
NotFound errors propagate. The fallback uses `WHERE RAND() < 0.001 LIMIT N`
which is a probabilistic random sample rather than the misleading
"first physical N rows". The implementer dropped `NotFound` from the catch
list relative to the suggested fix; that is a defensible call (a missing
table genuinely shouldn't be swallowed) and I agree.

**#8 `_to_jsonable` Decimal/bytes — addressed.** `_to_jsonable`
(`bq.py:178-197`) now special-cases `Decimal` to `str()` (precision-
preserving), `bytes`/`bytearray` to `hex()`, and recurses into `list`,
`tuple`, and `dict`. The new `tests/unit/test_bq_jsonable.py` covers
primitives, datetime, Decimal precision round-trip, bytes hexing, nested
collections, and the `__str__` fallback — exactly the contract the
original finding asked for. Note: the type-check ordering puts the
`isinstance(v, (bool, int, float, str))` clause first, so a numpy bool/
int would still flow into `str()` via the final fallback rather than
preserving its primitive form; not a regression for Phase 0 since BQ
returns Python primitives, but worth knowing for Phase 1.

**#9 BQ location auto-detect — addressed.** `BQClient.__init__`
(`bq.py:50-85`) builds a client at the configured location, calls
`get_dataset`, and rebinds the client to the dataset's actual location
when they differ, with a warning. `BQConfig.location` is now `str | None`
and `load_env` maps an empty/missing var to `None`
(`env.py:106`). `.env.example` comments `DAITY_BQ_LOCATION` out by default
with the auto-detect rationale. The `NotFound` path raises a clear
`RuntimeError` rather than the original silent 404. New unit test
`test_load_env_location_unset_means_none` pins the env contract.

### New findings introduced by the fixes

**N1. [Low] `_validate_sa_key` may TypeError on a key whose `private_key_id`
is JSON `null`.** `env.py:79` does
`payload.get("private_key_id", "<unknown>")[:12] + "..."`. The default only
fires when the key is absent; if `private_key_id: null`, `.get` returns
`None` and the slice crashes. Mitigation: `(payload.get("private_key_id") or "<unknown>")[:12]`.
Real SA keys always carry a non-null `private_key_id`, so this is an
edge-case robustness nit, not a P0 blocker.

**N2. [Low] Dead variable `day_expr` in `audit_ohlcv`.**
`phase0_audit.py:150` assigns `day_expr, diff_fn = _ts_helpers(...)` but
`day_expr` is never read in the OHLCV path (only `diff_fn` is used). Cosmetic;
remove with `_, diff_fn = _ts_helpers(...)` or suppress with a leading
underscore. Will trip a strict ruff rule once `F841` is enabled.

**N3. [Low] `_ts_helpers` returns `DATETIME_DIFF` for STRING/unknown ts
types, then formats it with `MILLISECOND` granularity in
`snapshot_rate_seconds`.** If a producer ever stores `ts` as STRING, the
snapshot-rate query will fail at the diff call. It fails *gracefully* via
`_safe`, so the audit still completes — the symbol-day census still
produces, and the operator gets a clear warning. Acceptable for Phase 0;
worth tightening in Phase 1 to detect STRING ts up-front and skip the
snapshot-rate query with a targeted warning.

**N4. [Low] Test coverage gap: `BQClient` location auto-detect is not
unit-tested.** `tests/unit/test_env.py` covers `BQConfig` and the env
contract, and `test_bq_jsonable.py` covers the JSON helper, but the new
`__init__` rebinding logic in `BQClient` has no mock-based test. The
location-mismatch warn-and-rebind path will only be exercised by an
integration test or live audit run. Phase 0 acceptance can ride on the
audit run itself; Phase 1 should add a `bigquery.Client`-mocked unit test.

### Final verdict

**Approve to proceed to audit run + Phase 1.** All Major findings #3–#9
and Critical #2 are properly addressed; the implementer also picked up
Minor #10 (`maximum_bytes_billed` budget), #11 (`AuditError` instead of
`SystemExit`), #15 (backtick-quoted identifiers via `_bq_quote`), and #18
(clauses list). The new findings (N1–N4) are all Low severity and can land
with Phase 1 cleanup. Run the audit; if the OB symbol-day census comes
back in the 50–500 range with confidence, the Phase 0 gate is met.
