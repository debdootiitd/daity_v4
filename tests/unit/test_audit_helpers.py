"""Unit tests for the column-resolution heuristics in the audit script."""

from __future__ import annotations

import inspect

from daity.scripts import phase0_audit
from daity.scripts.phase0_audit import (
    LEVEL_CANDIDATES,
    SIDE_CANDIDATES,
    SYMBOL_CANDIDATES,
    TIMEFRAME_CANDIDATES,
    TS_CANDIDATES,
    _all_columns,
    _detect_wide_ob_levels,
    _pick,
)


def _schema(*names: str) -> list[dict[str, str]]:
    return [{"name": n, "type": "STRING", "mode": "NULLABLE", "description": None} for n in names]


def test_pick_finds_canonical_names() -> None:
    s = _schema("ts", "symbol", "open", "high", "low", "close", "volume", "timeframe")
    assert _pick(s, TS_CANDIDATES) == "ts"
    assert _pick(s, SYMBOL_CANDIDATES) == "symbol"
    assert _pick(s, TIMEFRAME_CANDIDATES) == "timeframe"


def test_pick_is_case_insensitive() -> None:
    s = _schema("Symbol", "TimeStamp")
    assert _pick(s, SYMBOL_CANDIDATES) == "Symbol"
    assert _pick(s, TS_CANDIDATES) == "TimeStamp"


def test_pick_returns_none_when_no_match() -> None:
    s = _schema("foo", "bar", "baz")
    assert _pick(s, TS_CANDIDATES) is None
    assert _pick(s, LEVEL_CANDIDATES) is None
    assert _pick(s, SIDE_CANDIDATES) is None


def test_pick_prefers_first_candidate() -> None:
    # Both 'ts' and 'event_time' present; should pick 'ts' (first in TS_CANDIDATES).
    s = _schema("event_time", "ts")
    assert _pick(s, TS_CANDIDATES) == "ts"


def test_all_columns_preserves_order() -> None:
    s = _schema("z", "a", "m")
    assert _all_columns(s) == ["z", "a", "m"]


def test_detect_wide_ob_levels_finds_paired_levels() -> None:
    # Top-5 wide layout — the actual `order_book_depth` schema.
    s = _schema(
        "symbol", "ts", "ltp",
        "bid1_price", "bid1_qty", "bid2_price", "bid2_qty",
        "bid3_price", "bid3_qty", "bid4_price", "bid4_qty",
        "bid5_price", "bid5_qty",
        "ask1_price", "ask1_qty", "ask2_price", "ask2_qty",
        "ask3_price", "ask3_qty", "ask4_price", "ask4_qty",
        "ask5_price", "ask5_qty",
    )
    assert _detect_wide_ob_levels(s) == [1, 2, 3, 4, 5]


def test_detect_wide_ob_levels_requires_both_sides() -> None:
    # Only bid columns present → not a usable wide L2.
    s = _schema("symbol", "ts", "bid1_price", "bid1_qty", "bid2_price", "bid2_qty")
    assert _detect_wide_ob_levels(s) == []


def test_detect_wide_ob_levels_skips_partial_pairs() -> None:
    # bid1 has matching ask1; bid2 has no ask2 → only level 1.
    s = _schema("bid1_price", "ask1_price", "bid2_price", "bid2_qty")
    assert _detect_wide_ob_levels(s) == [1]


def test_detect_wide_ob_levels_empty_for_long_format() -> None:
    s = _schema("symbol", "ts", "side", "level", "price", "size")
    assert _detect_wide_ob_levels(s) == []


def test_no_reserved_word_aliases_in_audit_sql() -> None:
    """Guard the BigQuery reserved-word bug: never alias any aggregate to `rows`.

    `rows` is reserved (window-frame syntax) and silently breaks any GROUP BY
    that uses it. Use `n_rows` instead.
    """
    src = inspect.getsource(phase0_audit)
    assert " AS rows " not in src and " AS rows\n" not in src, (
        "Found `AS rows` in audit SQL — `rows` is a BigQuery reserved word; "
        "use `AS n_rows` instead."
    )


def test_interval_is_backtick_quoted_in_refresh_sql() -> None:
    """`interval` is a BQ reserved word (used in `INTERVAL '1 day'` literals);
    every reference in our refresh-OHLCV SQL must be backtick-quoted."""
    from daity.scripts import refresh_ohlcv

    src = inspect.getsource(refresh_ohlcv)
    # Find unbacktick'd references inside the SQL string literals.
    # A loose heuristic: within any line containing ".interval " or "S.interval"
    # (without backtick), flag it.
    bad_patterns = [
        "T.interval ", "S.interval ", "P.interval ",
        ", interval,", "BY interval", " interval IN",
    ]
    for pat in bad_patterns:
        assert pat not in src, (
            f"Found unbacktick'd `{pat}` in refresh_ohlcv SQL — `interval` is a "
            f"BigQuery reserved word; use `\\`interval\\``."
        )
