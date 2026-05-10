"""Local Parquet cache for OHLCV (Phase 1.5 — the offline-first read path).

Layout (per `(scale, symbol)`, with monthly row groups inside each file):

    data/parquet/
    ├── 5m/
    │   ├── RELIANCE.parquet     # ~7y of 5m bars sorted by ts; row groups ≈ 1 month
    │   ├── TCS.parquet
    │   └── ... (one per symbol)
    ├── 15m/
    ├── 60m/
    └── day/

Why per-(scale, symbol) not per-(symbol, year, month) (DESIGN.md original choice):
- Pretraining is the heaviest read pattern: random window per random symbol,
  millions of times. Per-symbol files are mmap-friendly — open once, slice many.
- Cross-sectional reads (strategy at rebalance time) cost ~1 file open per
  symbol; with row group ts-stat pruning + DuckDB's parallel scan that's
  ~200-500ms which is fine for the 30-min rebalance loop. If it ever bites
  in Phase 5 we add a derived `(scale, year, month)` materialized view.

Why monthly row groups:
- Predicate pushdown on ts: a query for "RELIANCE 5m bars 2024-06-01..30"
  hits exactly 1 row group. Without row groups Parquet would scan the whole
  ~6 MB file for every range query, which is fine but wasteful.

`as_of` is mandatory on every read — that's the leakage gate. No row whose
`ts > as_of` ever escapes this layer. CI enforces this in
`tests/leakage/test_parquet_leakage.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq

from daity.data.candles import effective_close
from daity.utils.logging import get_logger

log = get_logger(__name__)

# raw_ohlcv calls the timeframe column `scale`; we keep the same name for
# any frame that lands in the Parquet store.
SCALE_COLUMN = "scale"
TS_COLUMN = "ts"
SYMBOL_COLUMN = "symbol"

VALID_SCALES: tuple[str, ...] = ("5m", "15m", "60m", "day")

# Per-scale row group size, chosen so each row group ≈ 1 month of bars.
# The numbers are upper bounds — actual group size adapts to the data shape.
ROW_GROUP_SIZES: dict[str, int] = {
    "5m":  1750,   # ~75 bars/day × ~22 trading days
    "15m": 600,    # ~25 × 22
    "60m": 160,    # ~7 × 22
    "day": 24,     # ~22
}

# A symbol may only contain alphanum + a few separators NSE actually uses
# (e.g. "M&M", "BAJAJ-AUTO"). Any other character is rejected at write time
# to defend against directory-traversal and filesystem-unsafe names.
SAFE_SYMBOL_RE = re.compile(r"^[A-Z0-9_&\-]+$")


@dataclass(frozen=True, slots=True)
class PartitionMeta:
    """Lightweight metadata for one (scale, symbol) Parquet file."""

    scale: str
    symbol: str
    path: Path
    n_rows: int
    min_ts: datetime | None
    max_ts: datetime | None
    file_bytes: int


class ParquetStore:
    """Per-(scale, symbol) Parquet store with `as_of`-safe reads.

    Construct with the cache root; all operations resolve relative to it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._duckdb: duckdb.DuckDBPyConnection | None = None

    # ----- Path resolution -----

    def _check_scale(self, scale: str) -> None:
        if scale not in VALID_SCALES:
            msg = f"Unknown scale {scale!r}; expected one of {list(VALID_SCALES)}"
            raise ValueError(msg)

    def _check_symbol(self, symbol: str) -> None:
        if not SAFE_SYMBOL_RE.match(symbol):
            msg = f"Symbol {symbol!r} contains characters not allowed by SAFE_SYMBOL_RE."
            raise ValueError(msg)

    def partition_path(self, scale: str, symbol: str) -> Path:
        """Filesystem path for one (scale, symbol) partition."""
        self._check_scale(scale)
        self._check_symbol(symbol)
        return self.root / scale / f"{symbol}.parquet"

    def list_partitions(self, scale: str) -> list[Path]:
        """All partition files currently on disk for one scale."""
        self._check_scale(scale)
        d = self.root / scale
        if not d.exists():
            return []
        return sorted(d.glob("*.parquet"))

    # ----- Writes -----

    def write_partition(
        self,
        scale: str,
        symbol: str,
        frame: pl.DataFrame,
        *,
        row_group_size: int | None = None,
    ) -> Path:
        """Write `frame` as the Parquet partition for `(scale, symbol)`.

        - Frame must contain at least `(symbol, ts, scale, open, high, low,
          close, volume)`.
        - Sorted by `ts` before write (in-place sort produces a stable file).
        - All rows must have `symbol == symbol` and `scale == scale`; mismatched
          rows raise to prevent cross-partition contamination.
        - Replaces any existing file atomically (write to .tmp + rename).
        """
        path = self.partition_path(scale, symbol)
        if frame.height == 0:
            msg = f"Refusing to write empty frame for ({scale!r}, {symbol!r})"
            raise ValueError(msg)
        # Defensive: every row in this file must belong to this partition.
        bad_sym = frame.filter(pl.col(SYMBOL_COLUMN) != symbol).height
        bad_sc = frame.filter(pl.col(SCALE_COLUMN) != scale).height
        if bad_sym or bad_sc:
            msg = (
                f"Frame for partition ({scale!r}, {symbol!r}) contains "
                f"{bad_sym} rows with wrong symbol and {bad_sc} rows with wrong scale."
            )
            raise ValueError(msg)
        # Normalize ts: BQ EXPORT writes TIMESTAMP as tz-naive UTC; the old
        # per-query path produced tz-aware UTC. Force tz-aware here so every
        # partition file has the same schema regardless of who wrote it.
        sorted_frame = frame.sort(TS_COLUMN)
        if sorted_frame[TS_COLUMN].dtype.time_zone is None:
            sorted_frame = sorted_frame.with_columns(
                pl.col(TS_COLUMN).dt.replace_time_zone("UTC")
            )
        rgs = row_group_size if row_group_size is not None else ROW_GROUP_SIZES[scale]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # PyArrow gives finer row-group control than polars.write_parquet.
        table = sorted_frame.to_arrow()
        pq.write_table(
            table, tmp,
            row_group_size=rgs,
            compression="zstd",
            use_dictionary=True,
        )
        tmp.replace(path)
        log.info(
            "ParquetStore: wrote %s/%s.parquet (%d rows, row_group_size=%d, %d bytes)",
            scale, symbol, frame.height, rgs, path.stat().st_size,
        )
        return path

    # ----- Reads -----

    def read(
        self,
        *,
        symbol: str,
        scale: str,
        as_of: datetime,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Per-symbol read for `(scale, symbol)` with `as_of` cap.

        - `as_of` is **mandatory**: no row with `ts > as_of` ever escapes.
          Pass `datetime.now(tz=timezone.utc)` for a "current" read.
        - `start` / `end` further bound the time window; both inclusive.
        - Returns an empty frame (with the right schema) if the partition
          doesn't exist on disk — *not* a FileNotFoundError, because absence
          is a legitimate state (the symbol may not have data yet for that
          scale, especially during incremental ingest).
        """
        path = self.partition_path(scale, symbol)
        if not path.exists():
            return _empty_frame()

        cap = _ensure_utc(as_of)
        # Lazy scan + filter so polars pushes the ts predicate into the
        # Parquet reader: the row group's min/max stats let it skip groups
        # whose entire ts range is outside [start, cap]. For partitions with
        # multiple row groups (5m × 7y has ~80, sized to ~1 month each) this
        # is a real win — the alternative `pl.read_parquet(path).filter(...)`
        # materializes every row before filtering.
        #
        # Defensive tz handling: BQ-EXPORT-written partitions store ts as
        # tz-naive UTC (Parquet INT96 doesn't preserve tz), so we inspect the
        # column dtype before plumbing the filter. We compare tz-naive cap to
        # tz-naive ts, or tz-aware to tz-aware — never mix.
        scan = pl.scan_parquet(path)
        ts_dtype = scan.collect_schema()[TS_COLUMN]
        ts_is_naive = ts_dtype.time_zone is None
        cap_for_filter = cap.replace(tzinfo=None) if ts_is_naive else cap
        scan = scan.filter(pl.col(TS_COLUMN) <= cap_for_filter)
        if start is not None:
            s = _ensure_utc(start)
            s_for_filter = s.replace(tzinfo=None) if ts_is_naive else s
            scan = scan.filter(pl.col(TS_COLUMN) >= s_for_filter)
        if end is not None:
            e = _ensure_utc(end)
            e_for_filter = e.replace(tzinfo=None) if ts_is_naive else e
            scan = scan.filter(pl.col(TS_COLUMN) <= e_for_filter)
        df = scan.collect()
        # Promote tz-naive ts to tz-aware UTC AFTER filter so the returned
        # frame's schema is uniform regardless of which writer produced the file.
        if ts_is_naive:
            df = df.with_columns(pl.col(TS_COLUMN).dt.replace_time_zone("UTC"))
        return df

    def read_cross_section(
        self,
        *,
        scale: str,
        ts: datetime,
        as_of: datetime,
    ) -> pl.DataFrame:
        """All symbols at one `ts`, capped by `as_of`.

        Used by ranking / strategy code that needs the full universe slice at
        a single bar boundary. Implemented via DuckDB so we get parallel scan
        + row-group pruning across all per-symbol files in one query.

        **Leakage gate.** For intraday scales `(5m, 15m, 60m)` the bar with
        `ts == as_of` is *still in progress* (its effective close is
        `ts + interval_duration`), so we additionally require
        `effective_close(scale, ts) <= as_of`. For daily, `ts` already
        encodes the close, so the check is `ts <= as_of`.
        """
        self._check_scale(scale)
        cap = _ensure_utc(as_of)
        ts_norm = _ensure_utc(ts)
        # Same-bar leakage gate — must hold for intraday bars too. A bar
        # opening at `ts` doesn't carry final OHLCV until `ts + duration`.
        if effective_close(scale, ts_norm) > cap:
            return _empty_frame()
        # Glob is deterministic + scale-validated above, so directory is safe.
        scale_dir = self.root / scale
        if not scale_dir.exists() or not any(scale_dir.glob("*.parquet")):
            return _empty_frame()

        con = self._get_duckdb()
        # Parameterized query — DuckDB binds via `?` placeholders so neither
        # the path glob, the ts, nor the cap can become an injection vector.
        # The glob path is constructed from a validated scale name (see
        # `_check_scale`) and the project's cache root; user-controlled input
        # never reaches the SQL string.
        glob_str = str(scale_dir / "*.parquet")
        sql = (
            f"SELECT * FROM read_parquet(?) "
            f"WHERE {TS_COLUMN} = ? AND {TS_COLUMN} <= ?"
        )
        arrow_table = con.execute(sql, [glob_str, ts_norm, cap]).arrow()
        return pl.from_arrow(arrow_table)  # type: ignore[return-value]

    def partition_meta(self, scale: str, symbol: str) -> PartitionMeta | None:
        """Lightweight metadata read — uses Parquet footer, no row scan."""
        path = self.partition_path(scale, symbol)
        if not path.exists():
            return None
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        # Pull min/max ts from row group statistics.
        min_ts: datetime | None = None
        max_ts: datetime | None = None
        ts_col_idx = pf.schema_arrow.get_field_index(TS_COLUMN)
        if ts_col_idx >= 0:
            for rg in range(pf.metadata.num_row_groups):
                stats = pf.metadata.row_group(rg).column(ts_col_idx).statistics
                if stats is None:
                    continue
                # Parquet row-group stats are naive when the column dtype is
                # naive (e.g. files written via BQ EXPORT). Normalize so all
                # callers see tz-aware UTC, matching the read path.
                lo = _ensure_utc(stats.min) if isinstance(stats.min, datetime) else None
                hi = _ensure_utc(stats.max) if isinstance(stats.max, datetime) else None
                if lo is not None:
                    min_ts = lo if min_ts is None or lo < min_ts else min_ts
                if hi is not None:
                    max_ts = hi if max_ts is None or hi > max_ts else max_ts
        return PartitionMeta(
            scale=scale, symbol=symbol, path=path, n_rows=n_rows,
            min_ts=min_ts, max_ts=max_ts, file_bytes=path.stat().st_size,
        )

    # ----- Internals -----

    def _get_duckdb(self) -> duckdb.DuckDBPyConnection:
        if self._duckdb is None:
            self._duckdb = duckdb.connect(":memory:")
            # Pin to UTC so TIMESTAMPTZ literals + returned columns stay tz-stable
            # regardless of the user's local timezone (IST on a Mac surfaces as
            # 'Asia/Kolkata', which then trips polars dtype comparisons).
            self._duckdb.execute("SET TimeZone='UTC'")
        return self._duckdb


# ----- Module helpers -----


def _empty_frame() -> pl.DataFrame:
    """An empty frame with the canonical OHLCV schema."""
    return pl.DataFrame(
        schema={
            SYMBOL_COLUMN: pl.Utf8,
            TS_COLUMN: pl.Datetime("us", "UTC"),
            SCALE_COLUMN: pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )


def _ensure_utc(dt: datetime | date) -> datetime:
    """Coerce dates / naive datetimes to tz-aware UTC.

    Naive datetimes are treated as UTC, not local — this is the *only* place
    in the data layer where we make that assumption, because all upstream
    flows (Kite, BQ) hand us tz-aware timestamps already.
    """
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    msg = f"Expected datetime or date, got {type(dt).__name__}: {dt!r}"
    raise TypeError(msg)
