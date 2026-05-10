"""Unit tests for the re-adjustment config + WHERE-clause builder.

Live BQ work (preview, apply) is exercised by the integration smoke; here
we lock in the pure logic so config errors and SQL injection vectors fail
loudly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from daity.data.readjust import (
    Readjustment,
)
from daity.data.readjust import (
    filter_already_applied as _filter_already_applied,
)
from daity.data.readjust import (
    load_config as _load_config,
)
from daity.data.readjust import (
    marker_key as _marker_key,
)
from daity.data.readjust import (
    record_applied as _record_applied,
)
from daity.data.readjust import (
    where_clause as _where_clause,
)


def test_readjustment_rejects_zero_or_negative_ratio() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Readjustment(symbol="X", ratio=0, cutoff_date_ist=None, event="", notes="")
    with pytest.raises(ValueError, match="must be positive"):
        Readjustment(symbol="X", ratio=-0.5, cutoff_date_ist=None, event="", notes="")


def test_readjustment_rejects_noop_ratio() -> None:
    """ratio=1.0 would change nothing; almost certainly a config typo."""
    with pytest.raises(ValueError, match="no-op"):
        Readjustment(symbol="X", ratio=1.0, cutoff_date_ist=None, event="", notes="")


def test_where_clause_with_cutoff() -> None:
    r = Readjustment(symbol="HDFCAMC", ratio=0.5,
                     cutoff_date_ist=date(2025, 11, 3), event="", notes="")
    where = _where_clause(r)
    assert "symbol = 'HDFCAMC'" in where
    assert "DATE(ts, 'Asia/Kolkata') < DATE '2025-11-03'" in where


def test_where_clause_without_cutoff_means_all_rows() -> None:
    """A null cutoff means apply to ALL rows for the symbol (used for IRB:
    the bonus is post-prod-window, so every prod row needs the ratio)."""
    r = Readjustment(symbol="IRB", ratio=0.5, cutoff_date_ist=None, event="", notes="")
    where = _where_clause(r)
    assert where == "symbol = 'IRB'"
    assert "DATE(ts" not in where  # no date predicate


def test_where_clause_escapes_single_quote_in_symbol() -> None:
    """Defense-in-depth against a bad config that puts a quote in the symbol."""
    r = Readjustment(symbol="O'BAD", ratio=0.5, cutoff_date_ist=None, event="", notes="")
    where = _where_clause(r)
    # SQL-escaped: single quote doubled. No unescaped quote remains in the value.
    assert "symbol = 'O''BAD'" in where


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "x.yaml"
    cfg.write_text(
        """
readjustments:
  - symbol: A
    event: 1:1 bonus
    record_date: 2025-11-26
    cutoff_date_ist: 2025-11-03
    ratio: 0.5
    notes: foo
  - symbol: B
    event: 1:5 split
    record_date: 2026-01-14
    cutoff_date_ist: null
    ratio: 0.2
    notes: bar
"""
    )
    plan = _load_config(cfg)
    assert len(plan) == 2
    assert plan[0].symbol == "A"
    assert plan[0].cutoff_date_ist == date(2025, 11, 3)
    assert plan[0].ratio == 0.5
    assert plan[1].cutoff_date_ist is None
    assert plan[1].ratio == 0.2


def test_load_config_handles_empty_file(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    assert _load_config(cfg) == []


def test_load_config_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_config(tmp_path / "nope.yaml")


def test_default_config_loads_and_has_expected_symbols() -> None:
    """The shipped default config must parse and contain at least the three
    web-confirmed splits (HDFCAMC, KOTAKBANK, IRB)."""
    plan = _load_config(Path("configs/data/corp_actions.yaml"))
    syms = {r.symbol for r in plan}
    assert {"HDFCAMC", "KOTAKBANK", "IRB"} <= syms
    by_sym = {r.symbol: r for r in plan}
    assert by_sym["HDFCAMC"].ratio == 0.5
    assert by_sym["KOTAKBANK"].ratio == 0.2
    assert by_sym["IRB"].ratio == 0.5
    assert by_sym["IRB"].cutoff_date_ist is None  # the all-rows case


# ----- Idempotency marker (Critical 1 fix) -----


def _r(sym: str, ratio: float, cutoff: date | None = None) -> Readjustment:
    return Readjustment(symbol=sym, ratio=ratio, cutoff_date_ist=cutoff,
                        event="t", notes="t")


def test_marker_key_is_stable_across_equivalent_inputs() -> None:
    """Same (sym, cutoff, ratio) → same key, regardless of float repr quirks."""
    a = _r("HDFCAMC", 0.5, date(2025, 11, 3))
    b = _r("HDFCAMC", 0.50, date(2025, 11, 3))
    assert _marker_key(a) == _marker_key(b)


def test_marker_key_distinguishes_different_cutoffs() -> None:
    a = _r("X", 0.5, date(2024, 1, 1))
    b = _r("X", 0.5, date(2024, 1, 2))
    assert _marker_key(a) != _marker_key(b)


def test_marker_key_distinguishes_null_vs_dated_cutoff() -> None:
    """`cutoff=None` (apply-to-all) is a different operation than a dated cutoff."""
    a = _r("X", 0.5, None)
    b = _r("X", 0.5, date(2024, 1, 1))
    assert _marker_key(a) != _marker_key(b)


def test_filter_already_applied_skips_recorded(tmp_path: Path) -> None:
    """The headline guarantee: if marker has the entry, it's filtered out of `todo`."""
    marker = tmp_path / "applied.json"
    marker.write_text(
        '[{"symbol": "X", "cutoff_date_ist": "2024-01-01", "ratio": 0.5}]'
    )
    plan = [
        _r("X", 0.5, date(2024, 1, 1)),  # already applied
        _r("Y", 0.5, date(2024, 1, 1)),  # new
    ]
    todo, skipped = _filter_already_applied(plan, marker)
    assert [r.symbol for r in todo] == ["Y"]
    assert [r.symbol for r in skipped] == ["X"]


def test_filter_already_applied_with_no_marker_file(tmp_path: Path) -> None:
    """Missing marker → everything is to-do, nothing skipped."""
    plan = [_r("X", 0.5, date(2024, 1, 1))]
    todo, skipped = _filter_already_applied(plan, tmp_path / "absent.json")
    assert len(todo) == 1
    assert skipped == []


def test_record_applied_appends_atomically(tmp_path: Path) -> None:
    """Writing a new entry must (a) preserve existing entries and (b) leave no .tmp behind."""
    marker = tmp_path / "applied.json"
    marker.write_text(
        '[{"symbol": "X", "cutoff_date_ist": "2024-01-01", "ratio": 0.5,'
        ' "event": "old", "applied_at_utc": "2024-01-01T00:00:00+00:00",'
        ' "run_id": "old", "rows_affected": 100}]'
    )
    new_entries = [_r("Y", 0.2, date(2024, 6, 1))]
    _record_applied(marker, new_entries, run_id="r1",
                    n_rows_per_sym={"Y": 50})

    import json as _json
    data = _json.loads(marker.read_text())
    assert len(data) == 2  # old kept, new appended
    syms = sorted(e["symbol"] for e in data)
    assert syms == ["X", "Y"]
    new_y = next(e for e in data if e["symbol"] == "Y")
    assert new_y["ratio"] == 0.2
    assert new_y["rows_affected"] == 50
    # No leftover .tmp file.
    assert not (tmp_path / "applied.json.tmp").exists()


def test_record_applied_creates_marker_when_absent(tmp_path: Path) -> None:
    """If the marker doesn't exist yet, _record_applied creates it."""
    marker = tmp_path / "subdir" / "applied.json"
    new_entries = [_r("X", 0.5, date(2024, 1, 1))]
    _record_applied(marker, new_entries, run_id="r1", n_rows_per_sym={"X": 10})
    assert marker.exists()
    import json as _json
    data = _json.loads(marker.read_text())
    assert len(data) == 1
    assert data[0]["symbol"] == "X"


def test_marker_round_trip_blocks_double_apply(tmp_path: Path) -> None:
    """End-to-end: record an entry, then a second filter pass excludes it."""
    marker = tmp_path / "applied.json"
    plan = [_r("X", 0.5, date(2024, 1, 1))]
    _record_applied(marker, plan, run_id="r1", n_rows_per_sym={"X": 10})
    todo, skipped = _filter_already_applied(plan, marker)
    assert todo == []
    assert len(skipped) == 1


def test_shipped_marker_has_three_phase1_entries() -> None:
    """The marker file shipped with the repo records the 3 Phase-1 readjustments
    (HDFCAMC, KOTAKBANK, IRB). Without these, re-running the default config
    after a fresh checkout would double-apply on a system that already had them."""
    import json as _json
    p = Path("data/applied_readjustments.json")
    if not p.exists():
        pytest.skip("marker file not on disk in this environment")
    entries = _json.loads(p.read_text())
    syms = {e["symbol"] for e in entries}
    assert {"HDFCAMC", "KOTAKBANK", "IRB"} <= syms
