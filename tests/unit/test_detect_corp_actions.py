"""Unit tests for the detect-corp-actions segmentation + proposal logic.

The BQ-side query is exercised by integration smoke. Here we lock in:
  - segmentation correctly identifies step-changes in a ratio series
  - the proposal builder emits the right cutoff/ratio per transition
  - dividend-band filter keeps small drifts out of proposals
"""

from __future__ import annotations

from datetime import date, timedelta

from daity.data.corp_action_detect import (
    Segment,
)
from daity.data.corp_action_detect import (
    proposed_readjustments as _proposed_readjustments,
)
from daity.data.corp_action_detect import (
    segments_for_symbol as _segments_for_symbol,
)


def _row(d: date, ratio: float, sym: str = "X") -> dict:
    return {"symbol": sym, "dt": d, "ratio": ratio}


def _series(start: date, ratios: list[float], sym: str = "X") -> list[dict]:
    return [_row(start + timedelta(days=i), r, sym) for i, r in enumerate(ratios)]


# ----- segmentation -----


def test_constant_ratio_yields_one_segment() -> None:
    segs = _segments_for_symbol(_series(date(2020, 1, 1), [0.5, 0.5, 0.5]),
                                ratio_noise=0.005)
    assert len(segs) == 1
    assert segs[0].median_ratio == 0.5


def test_clean_step_change_yields_two_segments() -> None:
    segs = _segments_for_symbol(
        _series(date(2020, 1, 1), [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
        ratio_noise=0.005,
    )
    assert len(segs) == 2
    assert segs[0].median_ratio == 0.5
    assert segs[1].median_ratio == 1.0
    # Boundary date is the first day of the new segment.
    assert segs[1].start_date == date(2020, 1, 4)


def test_within_noise_stays_one_segment() -> None:
    """Float-precision residuals (e.g., 0.5001 vs 0.4998) must NOT split a segment."""
    segs = _segments_for_symbol(
        _series(date(2020, 1, 1), [0.4999, 0.5001, 0.5003, 0.4998]),
        ratio_noise=0.005,
    )
    assert len(segs) == 1


def test_two_step_changes_yields_three_segments() -> None:
    """A symbol with two missed splits over time should produce 3 segments."""
    segs = _segments_for_symbol(
        _series(date(2020, 1, 1), [0.1, 0.1, 0.5, 0.5, 1.0, 1.0]),
        ratio_noise=0.005,
    )
    assert len(segs) == 3
    assert [s.median_ratio for s in segs] == [0.1, 0.5, 1.0]


def test_empty_input_yields_no_segments() -> None:
    assert _segments_for_symbol([], ratio_noise=0.005) == []


def test_unsorted_input_is_sorted_internally() -> None:
    rows = [
        _row(date(2020, 1, 5), 1.0),
        _row(date(2020, 1, 1), 0.5),
        _row(date(2020, 1, 3), 0.5),
    ]
    segs = _segments_for_symbol(rows, ratio_noise=0.005)
    assert len(segs) == 2
    assert segs[0].start_date == date(2020, 1, 1)
    assert segs[0].median_ratio == 0.5
    assert segs[1].start_date == date(2020, 1, 5)
    assert segs[1].median_ratio == 1.0


# ----- proposal builder -----


def _seg(sym: str, start: date, end: date, ratio: float, n: int = 30) -> Segment:
    """Build a Segment for tests. Default `n_rows=30` so segments comfortably
    pass the `min_segment_days=3` filter; tests that specifically exercise the
    filter override `n` to a small value."""
    return Segment(symbol=sym, start_date=start, end_date=end,
                   n_rows=n, median_ratio=ratio)


def test_proposal_for_simple_split() -> None:
    """One missed 1:1 bonus → one proposal at the boundary, ratio = old/new = 0.5/1.0 = 0.5."""
    segs = [
        _seg("HDFCAMC", date(2024, 1, 1), date(2025, 11, 2), 0.5),
        _seg("HDFCAMC", date(2025, 11, 3), date(2026, 5, 1), 1.0),
    ]
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["symbol"] == "HDFCAMC"
    assert p["ratio"] == 0.5
    assert p["cutoff_date_ist"] == "2025-11-03"


def test_proposal_for_two_missed_splits_chains_correctly() -> None:
    """Two missed splits → two proposals, each with its own boundary + ratio."""
    segs = [
        _seg("X", date(2020, 1, 1), date(2022, 6, 14), 0.1),  # pre-A
        _seg("X", date(2022, 6, 15), date(2024, 9, 19), 0.2),  # post-A pre-B
        _seg("X", date(2024, 9, 20), date(2026, 5, 1), 1.0),   # post-B
    ]
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    assert len(proposals) == 2
    # Older-segment proposal: 0.1 / 0.2 = 0.5 (the relative jump at 2022-06-15)
    older = next(p for p in proposals if p["cutoff_date_ist"] == "2022-06-15")
    assert older["ratio"] == 0.5
    # Newer-segment proposal: 0.2 / 1.0 = 0.2 (the relative jump at 2024-09-20)
    newer = next(p for p in proposals if p["cutoff_date_ist"] == "2024-09-20")
    assert newer["ratio"] == 0.2


def test_proposal_skips_dividend_band_drifts() -> None:
    """Small ratio drifts (~0.97) within the dividend band must be skipped —
    they're convention differences (Kite includes dividends, prod doesn't),
    not bugs."""
    segs = [
        _seg("WIPRO", date(2024, 1, 1), date(2025, 6, 1), 0.97),
        _seg("WIPRO", date(2025, 6, 2), date(2026, 5, 1), 1.0),
    ]
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    assert proposals == []  # 0.97/1.0 = 0.97 → within ±0.05 → skipped


def test_proposal_keeps_clean_split_outside_dividend_band() -> None:
    """A real split (ratio outside the dividend band) is kept."""
    segs = [
        _seg("Y", date(2024, 1, 1), date(2025, 6, 1), 0.5),
        _seg("Y", date(2025, 6, 2), date(2026, 5, 1), 1.0),
    ]
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    assert len(proposals) == 1
    assert proposals[0]["ratio"] == 0.5


def test_no_proposals_for_single_segment() -> None:
    """If a symbol's ratio is constant across the whole window, there's no
    transition → no proposal."""
    segs = [_seg("Z", date(2020, 1, 1), date(2026, 5, 1), 1.0)]
    assert _proposed_readjustments(segs, dividend_band=0.05) == []


def test_no_proposals_when_recent_segment_already_one() -> None:
    """If older segment is at 1.0 and newer is at 1.0 (just noise across a
    boundary that shouldn't have triggered), still no proposal."""
    segs = [
        _seg("A", date(2020, 1, 1), date(2025, 1, 1), 1.0),
        _seg("A", date(2025, 1, 2), date(2026, 5, 1), 1.0),
    ]
    assert _proposed_readjustments(segs, dividend_band=0.05) == []


def test_proposal_filters_short_blip_segments() -> None:
    """Major-10 fix: reject 1-2 day "segments" caused by single-day vol spikes
    (e.g. 2020-03 COVID circuit breakers). Phase-1 reviewer triage of the
    original 12 detected drifts >10% found 100% false positives in 1-2 day
    segments — none were real corp actions.
    """
    # Pattern: long stable segment → 1-day vol spike → long stable segment.
    segs = [
        _seg("X", date(2020, 1, 1), date(2020, 3, 17), 1.0, n=50),  # long, ratio 1.0
        _seg("X", date(2020, 3, 18), date(2020, 3, 18), 1.15, n=1),  # 1-day blip
        _seg("X", date(2020, 3, 19), date(2026, 5, 1), 1.0, n=900), # long, ratio 1.0
    ]
    # Default min_segment_days=3 → both transitions rejected (the middle
    # segment is too short).
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    assert proposals == [], (
        f"min_segment_days filter must reject 1-day blips; got {len(proposals)} proposals"
    )


def test_proposal_keeps_long_segments_through_filter() -> None:
    """Sanity: a real long-vs-long transition is NOT rejected by min_segment_days."""
    segs = [
        _seg("X", date(2020, 1, 1), date(2024, 1, 1), 0.5, n=1000),  # 4 years pre-split
        _seg("X", date(2024, 1, 2), date(2026, 5, 1), 1.0, n=600),   # 2.5 years post
    ]
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    assert len(proposals) == 1
    assert proposals[0]["ratio"] == 0.5


def test_proposal_yaml_schema_matches_readjust_loader() -> None:
    """The keys in a proposal must exactly match what `_load_config` reads
    in readjust_symbols (otherwise the user's edited YAML wouldn't load)."""
    segs = [
        _seg("X", date(2020, 1, 1), date(2025, 6, 1), 0.5),
        _seg("X", date(2025, 6, 2), date(2026, 5, 1), 1.0),
    ]
    proposals = _proposed_readjustments(segs, dividend_band=0.05)
    p = proposals[0]
    # readjust_symbols._load_config reads these keys:
    for key in ("symbol", "ratio", "cutoff_date_ist"):
        assert key in p, f"proposal missing key {key!r} expected by readjust loader"
