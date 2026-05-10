"""Phase 0 BigQuery data audit.

Introspects `curated_ohlcv` and `order_book_depth` to confirm schemas, row
counts, coverage, adjustment hints, survivorship hints, and (critically) the
order-book symbol-day census which sizes the OB fine-tune stage.

Run:
    python -m daity.scripts.phase0_audit
    # or, if installed as a package:
    daity-audit

Writes:
    reports/phase0_audit.json    — machine-readable
    stdout                       — human summary
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient, TableMeta
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

# Heuristics for mapping logical → physical column names.
TS_CANDIDATES = ("ts", "timestamp", "event_time", "bar_time", "time", "datetime", "dt")
SYMBOL_CANDIDATES = ("symbol", "ticker", "symbol_id", "instrument", "scrip", "tradingsymbol")
TIMEFRAME_CANDIDATES = ("timeframe", "interval", "resolution", "freq", "frequency", "bar_size")
LEVEL_CANDIDATES = ("level", "depth_level", "lvl", "rank")
SIDE_CANDIDATES = ("side", "book_side", "type")
PRICE_CANDIDATES = ("price",)
SIZE_CANDIDATES = ("size", "qty", "quantity", "volume")
CLOSE_CANDIDATES = ("close", "close_price", "c", "adj_close", "Close", "px_close")

# Lower-cased timeframe labels we treat as "daily" for the adjustment-hint probe.
DAILY_TIMEFRAME_LABELS: frozenset[str] = frozenset(
    {"1d", "d", "day", "daily", "eod", "1day", "1day_eq", "day_eq", "eq_day", "eod_nse"}
)

# IST timezone — NSE trading day boundaries.
IST_TZ = "Asia/Kolkata"

# Companion tables inside `daity_v3` that already exist (discovered in Phase 0
# audit). We capture their schemas so Phase 1 can decide whether to reuse vs
# rebuild calendar/symbol-master/etc.
COMPANION_TABLES: tuple[str, ...] = (
    "nse_trading_calendar",
    "nse_holidays",
    "symbol_list",
    "sector_metadata",
    "raw_ohlcv",
    "orderbook_daily_summary",
    "orderbook_quality_metrics",
)

# Wide-format OB detection: if `bid1_price` and `ask1_price` are present we
# treat the table as a per-snapshot wide layout (top-K levels per row) rather
# than the long format the original draft assumed.
WIDE_OB_BID_PREFIX = "bid"
WIDE_OB_ASK_PREFIX = "ask"
WIDE_OB_PRICE_SUFFIX = "_price"
WIDE_OB_QTY_SUFFIX = "_qty"


class AuditError(RuntimeError):
    """Raised when the audit can't complete (vs `EnvError` for env problems)."""


def _pick(schema: list[dict[str, Any]], candidates: tuple[str, ...]) -> str | None:
    """Find the first column whose name (case-insensitive) is in `candidates`."""
    by_name = {c["name"].lower(): c["name"] for c in schema}
    for cand in candidates:
        if cand.lower() in by_name:
            return by_name[cand.lower()]
    return None


def _all_columns(schema: list[dict[str, Any]]) -> list[str]:
    return [c["name"] for c in schema]


def _column_type(schema: list[dict[str, Any]], name: str) -> str:
    """Return the BQ type of `name` (case-insensitive lookup)."""
    for c in schema:
        if c["name"].lower() == name.lower():
            return str(c["type"]).upper()
    return ""


def _bq_quote(name: str) -> str:
    """Backtick-quote a BigQuery identifier (defends against reserved words)."""
    # Column names cannot contain backticks per BQ spec; belt-and-suspenders strip.
    return f"`{name.replace('`', '')}`"


def _ts_helpers(ts_col: str, ts_type: str) -> tuple[str, str]:
    """Return (day_expr, diff_fn) appropriate for a TIMESTAMP or DATETIME column.

    `day_expr` evaluates the IST trading-day for `ts_col`. `diff_fn` is
    `TIMESTAMP_DIFF` or `DATETIME_DIFF` to match the column type.
    """
    qts = _bq_quote(ts_col)
    if ts_type == "TIMESTAMP":
        return f"DATE({qts}, '{IST_TZ}')", "TIMESTAMP_DIFF"
    if ts_type == "DATETIME":
        # Assume the producer stored IST naive — the most common NSE choice.
        return f"DATE({qts})", "DATETIME_DIFF"
    if ts_type == "DATE":
        return qts, "DATE_DIFF"
    # Unknown / STRING — fall back to permissive form; caller may warn.
    return f"DATE({qts})", "DATETIME_DIFF"


def _detect_wide_ob_levels(schema: list[dict[str, Any]]) -> list[int]:
    """Return the sorted list of `k` for which both bid{k}_price and ask{k}_price exist.

    A non-empty result indicates wide-format L2 (top-K levels per row).
    """
    cols = {c["name"].lower() for c in schema}
    levels: set[int] = set()
    for col in cols:
        if not col.startswith(WIDE_OB_BID_PREFIX) or not col.endswith(WIDE_OB_PRICE_SUFFIX):
            continue
        mid = col[len(WIDE_OB_BID_PREFIX) : -len(WIDE_OB_PRICE_SUFFIX)]
        if not mid.isdigit():
            continue
        k = int(mid)
        ask_col = f"{WIDE_OB_ASK_PREFIX}{k}{WIDE_OB_PRICE_SUFFIX}"
        if ask_col in cols:
            levels.add(k)
    return sorted(levels)


def _safe(name: str, fn: Any, warnings: list[str]) -> Any:
    """Run `fn()`, capture any exception as a warning string, and return None on failure."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{name}: {type(exc).__name__}: {exc}")
        log.warning("%s failed: %s", name, exc)
        return None


def audit_ohlcv(bq: BQClient, table: str, meta: TableMeta) -> dict[str, Any]:
    """Run OHLCV-specific aggregates."""
    cols = meta.schema
    ts_col = _pick(cols, TS_CANDIDATES)
    sym_col = _pick(cols, SYMBOL_CANDIDATES)
    tf_col = _pick(cols, TIMEFRAME_CANDIDATES)
    close_col = _pick(cols, CLOSE_CANDIDATES)

    out: dict[str, Any] = {
        "resolved_columns": {
            "ts": ts_col,
            "symbol": sym_col,
            "timeframe": tf_col,
            "close": close_col,
        },
        "ts_type": _column_type(cols, ts_col) if ts_col else None,
        "all_columns": _all_columns(cols),
        "warnings": [],
    }

    fq = bq.cfg.fq_table(table)
    qfq = f"`{fq}`"

    # Sample rows for visual inspection (always attempted; failures are loud).
    out["sample_rows"] = _safe("sample_rows", lambda: bq.sample(table, n=20), out["warnings"]) or []

    if not (ts_col and sym_col):
        out["warnings"].append(
            "Could not auto-resolve ts/symbol columns; skipping aggregates. "
            "Inspect 'all_columns' and update TS_CANDIDATES / SYMBOL_CANDIDATES."
        )
        return out

    qts = _bq_quote(ts_col)
    qsym = _bq_quote(sym_col)
    # OHLCV audit only needs diff_fn for the staleness probe; day_expr is OB-specific.
    _, diff_fn = _ts_helpers(ts_col, out["ts_type"] or "")

    # Coverage
    out["distinct_symbols"] = _safe(
        "distinct_symbols",
        lambda: bq.query_scalar(f"SELECT APPROX_COUNT_DISTINCT({qsym}) FROM {qfq}"),
        out["warnings"],
    )
    rng = _safe(
        "ts_range",
        lambda: bq.query_rows(f"SELECT MIN({qts}) AS ts_min, MAX({qts}) AS ts_max FROM {qfq}"),
        out["warnings"],
    )
    out["ts_range"] = rng[0] if rng else {}

    # Timeframe distribution. NB: `rows` is a reserved word in BigQuery (window-
    # frame syntax), so we alias to `n_rows`.
    if tf_col:
        qtf = _bq_quote(tf_col)
        out["rows_by_timeframe"] = _safe(
            "rows_by_timeframe",
            lambda: bq.query_rows(
                f"SELECT {qtf} AS timeframe, COUNT(*) AS n_rows "
                f"FROM {qfq} GROUP BY 1 ORDER BY 2 DESC"
            ),
            out["warnings"],
        )
    else:
        out["warnings"].append("No timeframe column detected; treating as single-resolution.")
        out["rows_by_timeframe"] = None

    # Rows by year (for sanity + regime context)
    out["rows_by_year"] = _safe(
        "rows_by_year",
        lambda: bq.query_rows(
            f"SELECT EXTRACT(YEAR FROM {qts}) AS year, COUNT(*) AS n_rows "
            f"FROM {qfq} GROUP BY 1 ORDER BY 1"
        ),
        out["warnings"],
    )

    # Survivorship probe (uses type-aware diff_fn)
    out["staleness_buckets"] = _safe(
        "staleness_buckets",
        lambda: bq.query_rows(
            f"""
            WITH last AS (
              SELECT {qsym} AS symbol, MAX({qts}) AS last_ts
              FROM {qfq} GROUP BY 1
            ),
            ref AS (SELECT MAX({qts}) AS overall_max FROM {qfq})
            SELECT
              COUNTIF({diff_fn}((SELECT overall_max FROM ref), last_ts, DAY) > 30)  AS stale_30d,
              COUNTIF({diff_fn}((SELECT overall_max FROM ref), last_ts, DAY) > 90)  AS stale_90d,
              COUNTIF({diff_fn}((SELECT overall_max FROM ref), last_ts, DAY) > 180) AS stale_180d,
              COUNTIF({diff_fn}((SELECT overall_max FROM ref), last_ts, DAY) > 365) AS stale_365d,
              COUNT(*) AS total_symbols
            FROM last
            """
        ),
        out["warnings"],
    )

    # --- Adjustment hint ---
    # Pick a daily timeframe label *only* if it actually exists in the data.
    # Otherwise warn and skip; do not silently produce zero counts (which would
    # falsely suggest "prices are adjusted").
    out["adjustment_hint"] = None
    if not close_col:
        out["warnings"].append(
            "No close-like column detected; skipping adjustment hint. "
            "Inspect 'all_columns' and update CLOSE_CANDIDATES."
        )
    else:
        qclose = _bq_quote(close_col)
        clauses: list[str] = [f"{qclose} > 0"]
        if tf_col and out.get("rows_by_timeframe"):
            qtf = _bq_quote(tf_col)
            present_labels = {
                str(r["timeframe"]).lower(): r["timeframe"]
                for r in out["rows_by_timeframe"]
                if r.get("timeframe") is not None
            }
            matches = [
                lbl for low, lbl in present_labels.items() if low in DAILY_TIMEFRAME_LABELS
            ]
            if matches:
                quoted = ",".join(
                    "'" + str(m).replace("'", "''") + "'" for m in matches
                )
                clauses.append(f"CAST({qtf} AS STRING) IN ({quoted})")
            else:
                out["warnings"].append(
                    "No daily-looking timeframe label present; adjustment hint computed "
                    f"across all timeframes (treat with skepticism). Labels seen: "
                    f"{[r['timeframe'] for r in out['rows_by_timeframe']]}"
                )
        where_sql = "WHERE " + " AND ".join(clauses)
        out["adjustment_hint"] = _safe(
            "adjustment_hint",
            lambda: bq.query_rows(
                f"""
                WITH base AS (
                  SELECT {qsym} AS symbol, {qts} AS ts,
                         SAFE_DIVIDE(
                           {qclose},
                           LAG({qclose}) OVER (PARTITION BY {qsym} ORDER BY {qts})
                         ) - 1 AS ret
                  FROM {qfq} {where_sql}
                )
                SELECT
                  COUNTIF(ABS(ret) > 0.30) AS n_abs_ret_gt_30pct,
                  COUNTIF(ABS(ret) > 0.50) AS n_abs_ret_gt_50pct,
                  APPROX_QUANTILES(ABS(ret), 1000)[OFFSET(995)] AS abs_ret_p99_5,
                  APPROX_QUANTILES(ABS(ret), 1000)[OFFSET(999)] AS abs_ret_p99_9,
                  COUNT(*) AS n_returns
                FROM base WHERE ret IS NOT NULL
                """
            ),
            out["warnings"],
        )

    return out


def audit_order_book(bq: BQClient, table: str, meta: TableMeta) -> dict[str, Any]:
    """Run order-book-specific aggregates.

    The symbol-day census is the headline number; it sizes the Phase 4 OB
    fine-tune. We use IST-aware day expressions so a TIMESTAMP column doesn't
    silently bucket across UTC midnight.
    """
    cols = meta.schema
    ts_col = _pick(cols, TS_CANDIDATES)
    sym_col = _pick(cols, SYMBOL_CANDIDATES)
    lvl_col = _pick(cols, LEVEL_CANDIDATES)
    side_col = _pick(cols, SIDE_CANDIDATES)
    price_col = _pick(cols, PRICE_CANDIDATES)
    size_col = _pick(cols, SIZE_CANDIDATES)

    out: dict[str, Any] = {
        "resolved_columns": {
            "ts": ts_col,
            "symbol": sym_col,
            "level": lvl_col,
            "side": side_col,
            "price": price_col,
            "size": size_col,
        },
        "ts_type": _column_type(cols, ts_col) if ts_col else None,
        "all_columns": _all_columns(cols),
        "warnings": [],
    }

    fq = bq.cfg.fq_table(table)
    qfq = f"`{fq}`"

    out["sample_rows"] = _safe("sample_rows", lambda: bq.sample(table, n=20), out["warnings"]) or []

    if not (ts_col and sym_col):
        out["warnings"].append(
            "Could not auto-resolve ts/symbol on order_book_depth; skipping aggregates."
        )
        return out

    qts = _bq_quote(ts_col)
    qsym = _bq_quote(sym_col)
    day_expr, diff_fn = _ts_helpers(ts_col, out["ts_type"] or "")

    out["distinct_symbols"] = _safe(
        "distinct_symbols",
        lambda: bq.query_scalar(f"SELECT APPROX_COUNT_DISTINCT({qsym}) FROM {qfq}"),
        out["warnings"],
    )
    rng = _safe(
        "ts_range",
        lambda: bq.query_rows(f"SELECT MIN({qts}) AS ts_min, MAX({qts}) AS ts_max FROM {qfq}"),
        out["warnings"],
    )
    out["ts_range"] = rng[0] if rng else {}

    # Symbol-day census — the headline Phase 0 number. Uses IST-aware day_expr.
    out["symbol_day_count"] = _safe(
        "symbol_day_count",
        lambda: bq.query_scalar(
            f"SELECT COUNT(DISTINCT CONCAT(CAST({qsym} AS STRING), '|', "
            f"CAST({day_expr} AS STRING))) FROM {qfq}"
        ),
        out["warnings"],
    )
    # Daily distribution. `rows` is reserved in BigQuery — alias to `n_rows`.
    out["daily_symbol_count"] = _safe(
        "daily_symbol_count",
        lambda: bq.query_rows(
            f"SELECT {day_expr} AS date, "
            f"APPROX_COUNT_DISTINCT({qsym}) AS n_symbols, "
            f"COUNT(*) AS n_rows "
            f"FROM {qfq} GROUP BY 1 ORDER BY 1"
        ),
        out["warnings"],
    )

    # --- Wide-format detection ---
    # The original draft assumed a long layout (level/side rows). The actual
    # `order_book_depth` table is per-snapshot wide: top-K levels columned out
    # as bid{k}_price/bid{k}_qty/ask{k}_price/ask{k}_qty. Detect and probe both.
    wide_levels = _detect_wide_ob_levels(cols)
    out["layout"] = "wide" if wide_levels else ("long" if (lvl_col and side_col) else "unknown")
    out["wide_levels_present"] = wide_levels
    out["max_level"] = max(wide_levels) if wide_levels else None

    if wide_levels:
        # Book-fill rate at the top level — what fraction of snapshots actually
        # carry quotes vs being LTP-only rows. This sizes the *usable* OB
        # sample for fine-tuning. The Phase 0 sample showed many bid/ask NULLs.
        qbid1 = _bq_quote(f"{WIDE_OB_BID_PREFIX}1{WIDE_OB_PRICE_SUFFIX}")
        qask1 = _bq_quote(f"{WIDE_OB_ASK_PREFIX}1{WIDE_OB_PRICE_SUFFIX}")
        out["book_fill_rate"] = _safe(
            "book_fill_rate",
            lambda: bq.query_rows(
                f"""
                SELECT
                  COUNTIF({qbid1} IS NOT NULL AND {qask1} IS NOT NULL) AS n_with_book,
                  COUNTIF({qbid1} IS NULL OR {qask1} IS NULL) AS n_without_book,
                  COUNT(*) AS n_total
                FROM {qfq}
                """
            ),
            out["warnings"],
        )
        # Symbol-day count restricted to rows where the book is actually filled.
        # This is the headline number for sizing Phase 4 OB fine-tune.
        out["symbol_day_count_with_book"] = _safe(
            "symbol_day_count_with_book",
            lambda: bq.query_scalar(
                f"SELECT COUNT(DISTINCT CONCAT(CAST({qsym} AS STRING), '|', "
                f"CAST({day_expr} AS STRING))) FROM {qfq} "
                f"WHERE {qbid1} IS NOT NULL AND {qask1} IS NOT NULL"
            ),
            out["warnings"],
        )
    elif lvl_col:
        qlvl = _bq_quote(lvl_col)
        out["levels"] = _safe(
            "levels",
            lambda: bq.query_rows(
                f"SELECT {qlvl} AS level, COUNT(*) AS n_rows FROM {qfq} GROUP BY 1 ORDER BY 1"
            ),
            out["warnings"],
        )

    if side_col and not wide_levels:
        qside = _bq_quote(side_col)
        out["sides"] = _safe(
            "sides",
            lambda: bq.query_rows(
                f"SELECT {qside} AS side, COUNT(*) AS n_rows FROM {qfq} GROUP BY 1 ORDER BY 1"
            ),
            out["warnings"],
        )

    # Snapshot rate (median delta_t). Uses IST-aware day partition + type-aware diff.
    out["snapshot_rate_seconds"] = _safe(
        "snapshot_rate_seconds",
        lambda: bq.query_rows(
            f"""
            WITH ordered AS (
              SELECT {qsym} AS symbol, {qts} AS ts,
                     LAG({qts}) OVER (PARTITION BY {qsym}, {day_expr} ORDER BY {qts}) AS prev_ts
              FROM {qfq}
            ),
            deltas AS (
              SELECT {diff_fn}(ts, prev_ts, MILLISECOND) / 1000.0 AS dt_sec
              FROM ordered
              WHERE prev_ts IS NOT NULL
            )
            SELECT
              APPROX_QUANTILES(dt_sec, 1000)[OFFSET(50)]   AS dt_p0_05,
              APPROX_QUANTILES(dt_sec, 1000)[OFFSET(500)]  AS dt_p50,
              APPROX_QUANTILES(dt_sec, 1000)[OFFSET(950)]  AS dt_p95,
              COUNT(*) AS n_deltas
            FROM deltas
            """
        ),
        out["warnings"],
    )

    return out


def run_audit(env_file: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    """Execute the full audit and return the report dict."""
    cfg = load_env(env_file=env_file, strict=True)
    bq = BQClient(cfg)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "project": cfg.project,
        "dataset": cfg.dataset,
        "location": bq.location,
        "tables_in_dataset": bq.list_tables(),
        "tables": {},
    }

    expected = {"ohlcv": "curated_ohlcv", "order_book": "order_book_depth"}
    missing = [n for n in expected.values() if n not in report["tables_in_dataset"]]
    if missing:
        msg = (
            f"Expected tables not found in dataset {cfg.fq_dataset}: {missing}. "
            f"Found: {report['tables_in_dataset']}"
        )
        log.error(msg)
        raise AuditError(msg)

    for kind, table in expected.items():
        log.info("Auditing %s (%s)", kind, table)
        meta = bq.table_info(table)
        block: dict[str, Any] = {"meta": asdict(meta)}
        if kind == "ohlcv":
            block.update(audit_ohlcv(bq, table, meta))
        elif kind == "order_book":
            block.update(audit_order_book(bq, table, meta))
        report["tables"][table] = block

    # Companion-table inventory: schema + a tiny sample for any pre-existing
    # asset that Phase 1 might want to reuse (calendar, holidays, symbol master,
    # sector metadata, raw OHLCV for adjustment comparison, OB summaries). We
    # only capture metadata + 5 sample rows — no aggregates — so this stays cheap.
    report["companion_tables"] = {}
    for tbl in COMPANION_TABLES:
        if tbl not in report["tables_in_dataset"]:
            continue
        log.info("Auditing companion table %s", tbl)
        warnings: list[str] = []
        meta = _safe(f"{tbl}.table_info", lambda t=tbl: bq.table_info(t), warnings)
        sample = _safe(f"{tbl}.sample", lambda t=tbl: bq.sample(t, n=5), warnings)
        report["companion_tables"][tbl] = {
            "meta": asdict(meta) if meta else None,
            "sample_rows": sample or [],
            "warnings": warnings,
        }

    output = output or Path("reports/phase0_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", output)
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Pretty console summary of the audit."""
    console.rule("[bold]Phase 0 — BigQuery audit summary")
    console.print(
        f"Project: [cyan]{report['project']}[/cyan]   "
        f"Dataset: [cyan]{report['dataset']}[/cyan]   "
        f"Location: [cyan]{report['location']}[/cyan]"
    )
    console.print(f"Tables present in dataset: {report['tables_in_dataset']}")

    for table, block in report["tables"].items():
        meta = block["meta"]
        console.rule(f"[bold yellow]{table}")
        console.print(
            f"rows=[bold]{meta['num_rows']:,}[/bold]  size=[bold]{meta['num_bytes']/1e9:.2f} GB[/bold]  "
            f"created={meta['created']}  modified={meta['modified']}"
        )
        # Schema
        schema_tbl = Table(title="Schema", show_lines=False)
        schema_tbl.add_column("name")
        schema_tbl.add_column("type")
        schema_tbl.add_column("mode")
        for col in meta["schema"]:
            schema_tbl.add_row(col["name"], col["type"], col["mode"] or "")
        console.print(schema_tbl)

        if "resolved_columns" in block:
            console.print(f"resolved columns: {block['resolved_columns']}")
        if "ts_type" in block:
            console.print(f"ts column type: {block['ts_type']}")

        if table == "curated_ohlcv":
            console.print(
                f"distinct symbols (approx): [bold]{block.get('distinct_symbols')}[/bold]   "
                f"ts range: {block.get('ts_range')}"
            )
            if block.get("rows_by_timeframe"):
                tf_tbl = Table(title="Rows by timeframe")
                tf_tbl.add_column("timeframe")
                tf_tbl.add_column("rows")
                for r in block["rows_by_timeframe"]:
                    tf_tbl.add_row(str(r.get("timeframe")), f"{int(r.get('rows', 0)):,}")
                console.print(tf_tbl)
            if block.get("staleness_buckets"):
                console.print(
                    f"survivorship probe (stale-symbol counts): {block['staleness_buckets'][0]}"
                )
            if block.get("adjustment_hint"):
                console.print(
                    f"adjustment hint (large-return frequency): {block['adjustment_hint'][0]}"
                )

        if table == "order_book_depth":
            console.print(
                f"distinct symbols (approx): [bold]{block.get('distinct_symbols')}[/bold]   "
                f"symbol-day count (any row): [bold]{block.get('symbol_day_count')}[/bold]   "
                f"symbol-day count (with book): [bold red]{block.get('symbol_day_count_with_book')}[/bold red]   "
                f"ts range: {block.get('ts_range')}"
            )
            console.print(
                f"layout: {block.get('layout')}   "
                f"levels present: {block.get('wide_levels_present')}   "
                f"max depth: {block.get('max_level')}"
            )
            if block.get("book_fill_rate"):
                bfr = block["book_fill_rate"][0]
                with_book = bfr.get("n_with_book") or 0
                total = bfr.get("n_total") or 1
                pct = 100.0 * with_book / total if total else 0.0
                console.print(
                    f"book fill rate: [bold]{pct:.1f}%[/bold]  "
                    f"(with={with_book:,} / total={total:,})"
                )
            if block.get("snapshot_rate_seconds"):
                r = block["snapshot_rate_seconds"][0]
                console.print(
                    f"snapshot dt seconds  p5={r.get('dt_p0_05')}  "
                    f"p50={r.get('dt_p50')}  p95={r.get('dt_p95')}  n={r.get('n_deltas')}"
                )

        if block.get("warnings"):
            for w in block["warnings"]:
                console.print(f"[yellow]warning:[/yellow] {w}")

    if report.get("companion_tables"):
        console.rule("[bold cyan]Companion tables (pre-existing assets)")
        for tbl, comp in report["companion_tables"].items():
            meta = comp.get("meta") or {}
            cols = [c["name"] for c in meta.get("schema", [])]
            console.print(
                f"[bold]{tbl}[/bold]  rows={meta.get('num_rows', 0):,}  "
                f"size={(meta.get('num_bytes', 0) or 0) / 1e6:.1f} MB  cols={cols}"
            )
            for w in comp.get("warnings", []):
                console.print(f"  [yellow]warning:[/yellow] {w}")

    console.rule()


@click.command()
@click.option(
    "--env-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to .env (defaults to ./.env).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("reports/phase0_audit.json"),
    help="Where to write the JSON report.",
)
def main(env_file: Path | None, output: Path) -> None:
    """Run the Phase 0 BigQuery audit."""
    try:
        report = run_audit(env_file=env_file, output=output)
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        console.print("Set up [bold]./.env[/bold] (see .env.example) and place your SA key.")
        sys.exit(2)
    except AuditError as exc:
        console.print(f"[red]Audit error:[/red] {exc}")
        sys.exit(3)
    except Exception as exc:  # noqa: BLE001
        log.exception("Audit failed")
        console.print(f"[red]Audit failed unexpectedly:[/red] {exc}")
        sys.exit(1)
    print_summary(report)
    console.print(f"\n[green]OK[/green] — wrote [bold]{output}[/bold]")


if __name__ == "__main__":
    main()
