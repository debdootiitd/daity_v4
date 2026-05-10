"""Unit tests for `_to_jsonable` and basic BQ helper behavior."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from daity.data.bq import _to_jsonable


def test_passthrough_primitives() -> None:
    for v in (None, True, False, 0, 1, -1, 1.5, "x", ""):
        assert _to_jsonable(v) is v or _to_jsonable(v) == v


def test_datetime_isoformat() -> None:
    d = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    assert _to_jsonable(d) == d.isoformat()
    da = dt.date(2024, 1, 2)
    assert _to_jsonable(da) == da.isoformat()


def test_decimal_preserves_precision_via_str() -> None:
    big = Decimal("12345678901234567890.123456789")
    out = _to_jsonable(big)
    assert isinstance(out, str)
    # Round-trip preserves precision.
    assert Decimal(out) == big


def test_bytes_become_hex() -> None:
    assert _to_jsonable(b"\xde\xad") == "dead"
    assert _to_jsonable(bytearray(b"\x00\x01")) == "0001"


def test_nested_collections() -> None:
    val = {"d": dt.date(2024, 1, 1), "n": Decimal("1.5"), "xs": [b"\x00", 2]}
    out = _to_jsonable(val)
    assert out == {"d": "2024-01-01", "n": "1.5", "xs": ["00", 2]}


def test_unknown_object_falls_back_to_str() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird"

    assert _to_jsonable(Weird()) == "weird"
