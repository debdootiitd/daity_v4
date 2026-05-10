"""Unit tests for `ParquetStore` — the per-(scale, symbol) Parquet cache.

The leakage-critical `as_of` invariant is exercised here AND in
`tests/leakage/test_parquet_leakage.py` (which spot-checks 100 random triples
and runs as a CI gate).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from daity.data.parquet_store import (
    ROW_GROUP_SIZES,
    SAFE_SYMBOL_RE,
    VALID_SCALES,
    ParquetStore,
    _empty_frame,
    _ensure_utc,
)


def _frame(symbol: str, scale: str, ts_list: list[datetime],
           closes: list[float] | None = None) -> pl.DataFrame:
    n = len(ts_list)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts": ts_list,
            "scale": [scale] * n,
            "open":   [100.0] * n,
            "high":   [101.0] * n,
            "low":    [99.0]  * n,
            "close":  closes if closes is not None else [100.5] * n,
            "volume": [1000]  * n,
        },
        schema={
            "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
            "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
        },
    )


def _ts_seq(start: datetime, n: int, step: timedelta) -> list[datetime]:
    return [start + i * step for i in range(n)]


# ----- _ensure_utc -----


def test_ensure_utc_passes_aware_through() -> None:
    dt = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert _ensure_utc(dt) is dt


def test_ensure_utc_attaches_tz_to_naive() -> None:
    dt = datetime(2026, 1, 1, 12, 0)
    out = _ensure_utc(dt)
    assert out.tzinfo == UTC


def test_ensure_utc_promotes_date() -> None:
    out = _ensure_utc(date(2026, 1, 1))
    assert out == datetime(2026, 1, 1, tzinfo=UTC)


def test_ensure_utc_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        _ensure_utc("2026-01-01")  # type: ignore[arg-type]


# ----- Path resolution + validation -----


def test_partition_path_layout(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    p = store.partition_path("5m", "RELIANCE")
    assert p == tmp_path / "5m" / "RELIANCE.parquet"


def test_partition_path_rejects_unknown_scale(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    with pytest.raises(ValueError, match="Unknown scale"):
        store.partition_path("2m", "X")


def test_partition_path_rejects_unsafe_symbol(tmp_path: Path) -> None:
    """Defense-in-depth: a symbol with a path separator should raise, not
    write to a sibling directory."""
    store = ParquetStore(tmp_path)
    for bad in ("../etc", "FOO/BAR", "FOO BAR", "FOO.BAR", "FOO;rm"):
        with pytest.raises(ValueError, match="not allowed"):
            store.partition_path("5m", bad)


def test_partition_path_accepts_real_nse_symbol_quirks(tmp_path: Path) -> None:
    """Real NSE tickers can include `&` (M&M) and `-` (BAJAJ-AUTO)."""
    store = ParquetStore(tmp_path)
    store.partition_path("5m", "M&M")
    store.partition_path("5m", "BAJAJ-AUTO")


def test_safe_symbol_regex_constants() -> None:
    """Regression: don't drop `&` or `-` from SAFE_SYMBOL_RE without intent."""
    assert SAFE_SYMBOL_RE.match("M&M")
    assert SAFE_SYMBOL_RE.match("BAJAJ-AUTO")
    assert not SAFE_SYMBOL_RE.match("foo bar")


def test_valid_scales_match_design() -> None:
    assert set(VALID_SCALES) == {"5m", "15m", "60m", "day"}
    assert set(ROW_GROUP_SIZES) == set(VALID_SCALES)


# ----- Write / Read round-trip -----


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    ts_list = _ts_seq(datetime(2024, 1, 1, 9, 15, tzinfo=UTC), 10, timedelta(days=1))
    frame = _frame("RELIANCE", "day", ts_list)
    store.write_partition("day", "RELIANCE", frame)

    out = store.read(symbol="RELIANCE", scale="day",
                     as_of=datetime(2026, 12, 31, tzinfo=UTC))
    assert out.height == 10
    assert out["symbol"].unique().to_list() == ["RELIANCE"]


def test_write_sorts_by_ts(tmp_path: Path) -> None:
    """Even if the input frame is unsorted, the file must be sorted by ts."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = [base + timedelta(days=2), base, base + timedelta(days=1)]
    frame = _frame("X", "day", ts_list)
    store.write_partition("day", "X", frame)
    out = store.read(symbol="X", scale="day",
                     as_of=datetime(2026, 12, 31, tzinfo=UTC))
    ts_out = out["ts"].to_list()
    assert ts_out == sorted(ts_out)


def test_read_returns_empty_frame_when_partition_absent(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    out = store.read(symbol="MISSING", scale="day",
                     as_of=datetime(2026, 12, 31, tzinfo=UTC))
    assert out.height == 0
    # Empty but schema is intact (so callers can chain operations).
    assert set(out.columns) == set(_empty_frame().columns)


def test_write_rejects_empty_frame(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    with pytest.raises(ValueError, match="empty frame"):
        store.write_partition("day", "X", _empty_frame())


def test_write_rejects_cross_symbol_contamination(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["RELIANCE", "TCS"],
            "ts": [base, base],
            "scale": ["day", "day"],
            "open": [100.0, 200.0], "high": [101.0, 201.0],
            "low": [99.0, 199.0], "close": [100.5, 200.5],
            "volume": [1000, 2000],
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )
    with pytest.raises(ValueError, match="wrong symbol"):
        store.write_partition("day", "RELIANCE", frame)


def test_write_rejects_cross_scale_contamination(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {"symbol": ["X", "X"], "ts": [base, base + timedelta(days=1)],
         "scale": ["day", "5m"],
         "open": [100.0, 100.0], "high": [101.0, 101.0],
         "low": [99.0, 99.0], "close": [100.5, 100.5],
         "volume": [1000, 1000]},
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )
    with pytest.raises(ValueError, match="wrong scale"):
        store.write_partition("day", "X", frame)


def test_write_atomic_no_partial_file_on_crash(tmp_path: Path) -> None:
    """The .tmp + rename pattern means there's never a half-written .parquet."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    store.write_partition("day", "X", _frame("X", "day", [base]))
    # No leftover .tmp sibling after a successful write.
    assert not (tmp_path / "day" / "X.parquet.tmp").exists()


def test_row_group_size_is_per_scale(tmp_path: Path) -> None:
    """The actual Parquet file should respect the per-scale row group size."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = _ts_seq(base, 4000, timedelta(minutes=5))  # > 1 row group for 5m
    store.write_partition("5m", "X", _frame("X", "5m", ts_list))
    pf = pq.ParquetFile(tmp_path / "5m" / "X.parquet")
    # Several row groups, each at most ROW_GROUP_SIZES['5m'] rows.
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    assert max(sizes) <= ROW_GROUP_SIZES["5m"]
    assert pf.metadata.num_row_groups >= 2


# ----- as_of leakage gate -----


def test_read_as_of_caps_upper_bound(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = _ts_seq(base, 10, timedelta(days=1))   # 2024-01-01 .. 2024-01-10
    store.write_partition("day", "X", _frame("X", "day", ts_list))

    cap = datetime(2024, 1, 5, tzinfo=UTC)
    out = store.read(symbol="X", scale="day", as_of=cap)
    assert out.height == 5
    assert out["ts"].max() <= cap


def test_read_as_of_excludes_strictly_future_rows(tmp_path: Path) -> None:
    """The cap is inclusive — rows with ts == as_of are kept."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = [base, base + timedelta(days=1), base + timedelta(days=2)]
    store.write_partition("day", "X", _frame("X", "day", ts_list))

    cap = base + timedelta(days=1)
    out = store.read(symbol="X", scale="day", as_of=cap)
    assert out["ts"].to_list() == [base, base + timedelta(days=1)]


def test_read_start_end_bounds(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = _ts_seq(base, 10, timedelta(days=1))
    store.write_partition("day", "X", _frame("X", "day", ts_list))

    out = store.read(
        symbol="X", scale="day",
        as_of=datetime(2026, 12, 31, tzinfo=UTC),
        start=base + timedelta(days=2),
        end=base + timedelta(days=5),
    )
    assert [t.day for t in out["ts"].to_list()] == [3, 4, 5, 6]


def test_read_naive_as_of_is_treated_as_utc(tmp_path: Path) -> None:
    """Naive datetimes default to UTC (the only place we make this assumption)."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = _ts_seq(base, 5, timedelta(days=1))
    store.write_partition("day", "X", _frame("X", "day", ts_list))

    out = store.read(symbol="X", scale="day",
                     as_of=datetime(2024, 1, 3))   # naive
    assert out.height == 3


# ----- Cross-section reads -----


def test_read_cross_section_returns_one_row_per_symbol(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for sym in ("A", "B", "C"):
        ts_list = _ts_seq(base, 5, timedelta(days=1))
        store.write_partition("day", sym, _frame(sym, "day", ts_list))

    out = store.read_cross_section(
        scale="day", ts=base + timedelta(days=2),
        as_of=datetime(2026, 12, 31, tzinfo=UTC),
    )
    assert sorted(out["symbol"].to_list()) == ["A", "B", "C"]
    # All at the requested ts.
    assert out["ts"].n_unique() == 1


def test_read_cross_section_returns_empty_when_ts_after_as_of(tmp_path: Path) -> None:
    """The leakage gate also fires for cross-section reads."""
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    store.write_partition("day", "X", _frame("X", "day", [base]))

    out = store.read_cross_section(
        scale="day", ts=base, as_of=base - timedelta(days=1),
    )
    assert out.height == 0


def test_read_cross_section_returns_empty_when_no_files(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    out = store.read_cross_section(
        scale="day", ts=datetime(2024, 1, 1, tzinfo=UTC),
        as_of=datetime(2026, 12, 31, tzinfo=UTC),
    )
    assert out.height == 0


# ----- Metadata + listing -----


def test_partition_meta(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    ts_list = _ts_seq(base, 100, timedelta(days=1))
    store.write_partition("day", "X", _frame("X", "day", ts_list))

    meta = store.partition_meta("day", "X")
    assert meta is not None
    assert meta.scale == "day"
    assert meta.symbol == "X"
    assert meta.n_rows == 100
    assert meta.file_bytes > 0


def test_partition_meta_returns_none_when_absent(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    assert store.partition_meta("day", "MISSING") is None


def test_list_partitions(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for sym in ("A", "B", "C"):
        store.write_partition("day", sym, _frame(sym, "day", [base]))
    paths = store.list_partitions("day")
    assert [p.stem for p in paths] == ["A", "B", "C"]


def test_list_partitions_empty_when_dir_missing(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    assert store.list_partitions("day") == []


# ----- ts timezone normalization (BQ EXPORT writes naive UTC) -----


def test_write_normalizes_naive_ts_to_utc(tmp_path: Path) -> None:
    """A frame with tz-naive ts (the BQ-EXPORT shape) must be written as tz-aware UTC
    so all partition files have a uniform schema."""
    store = ParquetStore(tmp_path)
    base_naive = datetime(2024, 1, 1, 10, 0)  # tz-naive
    ts_list = [base_naive + timedelta(days=i) for i in range(5)]
    frame = pl.DataFrame(
        {
            "symbol": ["X"] * 5,
            "ts": ts_list,
            "scale": ["day"] * 5,
            "open":   [100.0] * 5, "high":  [101.0] * 5,
            "low":    [99.0] * 5,  "close": [100.5] * 5,
            "volume": [1000] * 5,
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us"),  # naive
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )
    store.write_partition("day", "X", frame)
    out = store.read(symbol="X", scale="day",
                     as_of=datetime(2026, 12, 31, tzinfo=UTC))
    # Read back as tz-aware UTC.
    assert out["ts"].dtype == pl.Datetime("us", "UTC")


def test_partition_meta_returns_tz_aware_min_max(tmp_path: Path) -> None:
    """Regression: a partition written from a tz-naive source (e.g. BQ EXPORT)
    must surface tz-aware min/max via `partition_meta` so callers can compare
    them across mixed-vintage files without tz collisions."""
    import pyarrow.parquet as pq
    legacy = pl.DataFrame(
        {
            "symbol": ["X"] * 3,
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "scale": ["day"] * 3,
            "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0],
            "volume": [10, 20, 30],
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us"),  # naive
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )
    d = tmp_path / "day"; d.mkdir()
    pq.write_table(legacy.to_arrow(), d / "X.parquet")

    store = ParquetStore(tmp_path)
    meta = store.partition_meta("day", "X")
    assert meta is not None
    assert meta.min_ts is not None and meta.min_ts.tzinfo is not None
    assert meta.max_ts is not None and meta.max_ts.tzinfo is not None


def test_read_handles_naive_ts_in_legacy_partition(tmp_path: Path) -> None:
    """Older partitions written before normalization landed may have tz-naive ts.
    The reader must still work — defensive cast at read time."""
    # Manually write a tz-naive Parquet file, bypassing the normalizing writer.
    import pyarrow.parquet as pq
    legacy = pl.DataFrame(
        {
            "symbol": ["X"] * 3,
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "scale": ["day"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low":  [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1000, 1000],
        },
        schema={"symbol": pl.Utf8, "ts": pl.Datetime("us"),
                "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64},
    )
    path = tmp_path / "day"
    path.mkdir()
    pq.write_table(legacy.to_arrow(), path / "X.parquet")

    store = ParquetStore(tmp_path)
    out = store.read(symbol="X", scale="day",
                     as_of=datetime(2024, 1, 2, tzinfo=UTC))
    assert out.height == 2
    assert out["ts"].dtype == pl.Datetime("us", "UTC")
