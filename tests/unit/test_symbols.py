"""Unit tests for the symbol master.

The BQ pull path is exercised in a separate live integration test; here we
verify the join + placeholder + lookup logic against synthetic frames.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from daity.data.symbols import PLACEHOLDER_COLUMNS, SymbolMaster


def _list() -> pl.DataFrame:
    return pl.DataFrame({"symbol": ["RELIANCE", "TCS", "INFY", "ZOMATO"]})


def _sectors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["RELIANCE", "TCS", "INFY"],  # ZOMATO intentionally missing
            "sector": ["Energy", "IT", "IT"],
            "industry": ["Refining", "Software", "Software"],
            "macro_sector": ["Commodities", "Tech", "Tech"],
        }
    )


def test_build_frame_joins_and_inserts_placeholders() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert m.frame.height == 4
    # symbol_id == symbol while ISINs are absent
    assert m.frame["symbol_id"].to_list() == m.frame["symbol"].to_list()
    # Placeholder columns are present and entirely null.
    for c in PLACEHOLDER_COLUMNS:
        assert c in m.frame.columns
        assert m.frame[c].is_null().all()


def test_missing_sector_left_null() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    z = m.lookup("ZOMATO")
    assert z is not None
    assert z["sector"] is None
    assert z["industry"] is None


def test_provisional_flag_is_true_until_isins_land() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert m.is_provisional is True
    assert "isin" in m.missing_columns


def test_lookup_returns_none_for_unknown() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert m.lookup("NOT_A_SYMBOL") is None


def test_in_sector_case_insensitive() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert sorted(m.in_sector("it")) == ["INFY", "TCS"]
    assert sorted(m.in_sector("IT")) == ["INFY", "TCS"]
    assert m.in_sector("Energy") == ["RELIANCE"]
    assert m.in_sector("nonexistent") == []


def test_in_macro_sector() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert sorted(m.in_macro_sector("tech")) == ["INFY", "TCS"]
    assert m.in_macro_sector("commodities") == ["RELIANCE"]


def test_symbols_property_and_len() -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert len(m) == 4
    assert sorted(m.symbols) == ["INFY", "RELIANCE", "TCS", "ZOMATO"]


def test_cache_round_trip(tmp_path: Path) -> None:
    m = SymbolMaster._build_frame(_list(), _sectors())
    m._write_cache(tmp_path)
    loaded = SymbolMaster.from_cache(tmp_path)
    assert loaded.frame.equals(m.frame)


def test_from_cache_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SymbolMaster.from_cache(tmp_path)


def test_column_order_is_stable() -> None:
    """Cache + downstream consumers depend on a stable column order."""
    m = SymbolMaster._build_frame(_list(), _sectors())
    expected = ["symbol_id", "symbol", "sector", "industry", "macro_sector", *PLACEHOLDER_COLUMNS]
    assert m.frame.columns == expected


# ----- ISIN backfill path -----


def _isin_lookup(*pairs: tuple[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [s for s, _ in pairs], "isin": [i for _, i in pairs]},
        schema={"symbol": pl.Utf8, "isin": pl.Utf8},
    )


def test_isin_backfill_drops_isin_from_missing_columns() -> None:
    """When every symbol resolves to an ISIN, `isin` is removed from
    missing_columns. is_provisional STAYS True until ALL identity columns
    are populated — flipping it on partial backfill would be a lie."""
    isins = _isin_lookup(
        ("RELIANCE", "INE002A01018"),
        ("TCS",      "INE467B01029"),
        ("INFY",     "INE009A01021"),
        ("ZOMATO",   "INE758T01015"),
    )
    m = SymbolMaster._build_frame(_list(), _sectors(), isin_df=isins)
    assert "isin" not in m.missing_columns
    # Every symbol now has an ISIN.
    assert m.frame["isin"].is_null().sum() == 0
    rel = m.lookup("RELIANCE")
    assert rel is not None
    assert rel["isin"] == "INE002A01018"
    # But other identity columns are still null, so is_provisional is True.
    assert m.is_provisional is True
    assert m.has_column("isin") is True
    assert m.has_column("mcap_bucket") is False
    assert m.has_column("fno_eligible") is False


def test_isin_backfill_with_partial_misses_keeps_provisional() -> None:
    """If even one symbol is missing an ISIN, the master stays provisional."""
    isins = _isin_lookup(
        ("RELIANCE", "INE002A01018"),
        ("TCS",      "INE467B01029"),
        ("INFY",     None),         # missing
        ("ZOMATO",   "INE758T01015"),
    )
    m = SymbolMaster._build_frame(_list(), _sectors(), isin_df=isins)
    assert m.is_provisional is True
    assert "isin" in m.missing_columns


def test_isin_backfill_does_not_change_column_order() -> None:
    """Cache stability: the column order must be identical with and without
    ISIN backfill — only the values differ."""
    isins = _isin_lookup(
        ("RELIANCE", "INE002A01018"),
        ("TCS",      "INE467B01029"),
        ("INFY",     "INE009A01021"),
        ("ZOMATO",   "INE758T01015"),
    )
    m_with = SymbolMaster._build_frame(_list(), _sectors(), isin_df=isins)
    m_without = SymbolMaster._build_frame(_list(), _sectors())
    assert m_with.frame.columns == m_without.frame.columns


def test_no_isin_lookup_means_isin_column_still_present_but_null() -> None:
    """Backwards compat: callers that don't pass isin_df still get a frame
    with the `isin` column (so downstream readers don't need a try/except)."""
    m = SymbolMaster._build_frame(_list(), _sectors())
    assert "isin" in m.frame.columns
    assert m.frame["isin"].is_null().all()


def test_cache_round_trip_preserves_missing_columns(tmp_path: Path) -> None:
    """Regression: cache reload must reconstruct `missing_columns` from the
    actual frame contents — without this, downstream callers checking
    `m.has_column('isin')` after a reload would lie."""
    isins = _isin_lookup(
        ("RELIANCE", "INE002A01018"),
        ("TCS",      "INE467B01029"),
        ("INFY",     "INE009A01021"),
        ("ZOMATO",   "INE758T01015"),
    )
    m = SymbolMaster._build_frame(_list(), _sectors(), isin_df=isins)
    assert "isin" not in m.missing_columns
    m._write_cache(tmp_path)
    loaded = SymbolMaster.from_cache(tmp_path)
    assert "isin" not in loaded.missing_columns
    # The other 7 columns are still missing, so is_provisional remains True.
    assert loaded.is_provisional is True
    assert loaded.has_column("isin")


def test_cache_round_trip_keeps_provisional_when_isins_partial(tmp_path: Path) -> None:
    """Mirror: an unfilled ISIN column must round-trip with `isin` still in `missing_columns`."""
    m = SymbolMaster._build_frame(_list(), _sectors())  # no isin_df
    m._write_cache(tmp_path)
    loaded = SymbolMaster.from_cache(tmp_path)
    assert loaded.is_provisional is True
    assert "isin" in loaded.missing_columns


def test_is_provisional_only_false_when_all_columns_populated(tmp_path: Path) -> None:
    """The headline contract: is_provisional == bool(missing_columns).
    This invariant cannot drift because is_provisional is a property, not a field."""
    # Synthetic master with all columns populated.
    n = 4
    full_frame = pl.DataFrame(
        {
            "symbol_id": ["A", "B", "C", "D"],
            "symbol":    ["A", "B", "C", "D"],
            "sector":    ["X"] * n, "industry": ["X"] * n,
            "macro_sector": ["X"] * n,
            "isin": ["I"] * n, "valid_from": [date(2020, 1, 1)] * n,
            "valid_to": [date(2030, 1, 1)] * n,
            "mcap_bucket": ["L"] * n, "free_float_bucket": ["L"] * n,
            "fno_eligible": [True] * n,
            "listed_on": [date(2020, 1, 1)] * n,
            "delisted_on": [None] * n,
        }
    )
    # `delisted_on` is null by design (active symbols), so it's NOT a missing
    # column in the frame-content sense. We need a coarser definition: a column
    # is "missing" if it's never populated for ANY symbol.
    # For this test we just verify the (missing_columns=()) → is_provisional=False contract.
    sm = SymbolMaster(frame=full_frame, missing_columns=())
    assert sm.is_provisional is False
    sm2 = SymbolMaster(frame=full_frame, missing_columns=("mcap_bucket",))
    assert sm2.is_provisional is True


def test_has_column_returns_false_when_in_missing_columns() -> None:
    """`has_column` is the per-column gate callers should prefer over the
    coarse `is_provisional` flag."""
    m = SymbolMaster._build_frame(_list(), _sectors())  # no isin
    assert m.has_column("symbol") is True       # base column, always present
    assert m.has_column("isin") is False        # in missing_columns
    assert m.has_column("mcap_bucket") is False
    assert m.has_column("nonexistent_col") is False  # not in frame at all
