"""Auto-detect missed corp-action adjustments by segmenting the Kite-vs-prod
close-ratio time series per symbol.

Premise: Kite applies adjustments retroactively, so the close-ratio
`kite_close / prod_close` for a (symbol, ts) pair tells us exactly how
prod's adjustment differs from Kite's at that ts. A segment of consecutive
rows with closely matching ratios identifies a stable adjustment regime;
the boundary between segments marks a corp-action ex-date.

`segments_for_symbol(rows, ratio_noise)` runs the segmentation;
`proposed_readjustments(segments, dividend_band)` converts segments into
the schema `daity-readjust-symbols` consumes.

The CLI (`daity-detect-corp-actions`) is just orchestration: pull from BQ,
call these helpers, write YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of consecutive trading days with a near-constant kite/prod ratio."""

    symbol: str
    start_date: date
    end_date: date
    n_rows: int
    median_ratio: float


def segments_for_symbol(rows: list[dict], *, ratio_noise: float) -> list[Segment]:
    """Greedy 1-D segmentation: each new row joins the current segment if its
    ratio is within `ratio_noise` of the segment's running median; otherwise
    starts a new segment. Robust to occasional outliers (median, not mean)."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["dt"])
    segments: list[Segment] = []
    cur_start: date = rows[0]["dt"]
    cur_ratios: list[float] = [rows[0]["ratio"]]
    cur_end: date = rows[0]["dt"]
    sym = rows[0]["symbol"]

    def flush() -> None:
        sorted_r = sorted(cur_ratios)
        med = sorted_r[len(sorted_r) // 2]
        segments.append(Segment(symbol=sym, start_date=cur_start,
                                end_date=cur_end, n_rows=len(cur_ratios),
                                median_ratio=med))

    for r in rows[1:]:
        sorted_r = sorted(cur_ratios)
        med = sorted_r[len(sorted_r) // 2]
        if abs(r["ratio"] - med) <= ratio_noise:
            cur_ratios.append(r["ratio"])
            cur_end = r["dt"]
        else:
            flush()
            cur_start = r["dt"]
            cur_end = r["dt"]
            cur_ratios = [r["ratio"]]
    flush()
    return segments


def proposed_readjustments(
    segments: list[Segment], *, dividend_band: float, min_segment_days: int = 3,
) -> list[dict]:
    """Convert segments into entries matching `corp_actions.yaml` schema.

    For a symbol with N segments [S0, S1, ..., S_{N-1}] ordered by date,
    we want each segment's prod data to match Kite. For each older segment,
    propose `ratio = (segment_median / next_segment_median)` to bring it
    onto the next segment's scale. Cutoff = `next_segment.start_date`.

    Skips a proposal when:
    - The ratio is within ±`dividend_band` of 1.0 (likely a dividend
      convention difference, not a missed corp action), or
    - Either the older or newer adjacent segment is shorter than
      `min_segment_days` (default 3). A real corp action transitions to a
      new persistent regime; spurious 1-2 day "segments" from extreme single-
      day volatility (e.g. 2020-03 COVID circuit breakers) are rejected by
      this filter. Phase-1 reviewer triage of the original 12 >10% drifts
      found 100% false positives in 1-2 day segments.
    """
    if len(segments) < 2:
        return []
    proposals: list[dict] = []
    for i in range(len(segments) - 1):
        s = segments[i]
        nxt = segments[i + 1]
        ratio = s.median_ratio / nxt.median_ratio if nxt.median_ratio else 0
        if abs(ratio - 1.0) <= dividend_band:
            continue
        # Suppress "blip" segments — single-day vol spikes that look like a
        # transition but resolve back to the prior regime within a day.
        if s.n_rows < min_segment_days or nxt.n_rows < min_segment_days:
            continue
        proposals.append({
            "symbol": s.symbol,
            "event": f"detected: ratio {s.median_ratio:.4f} → {nxt.median_ratio:.4f} "
                     f"(boundary {nxt.start_date.isoformat()})",
            "record_date": nxt.start_date.isoformat(),
            "cutoff_date_ist": nxt.start_date.isoformat(),
            "ratio": round(ratio, 6),
            "notes": "Auto-detected from Kite-vs-prod ratio segmentation. "
                     "Cross-check against NSE corp-actions before applying.",
        })
    return proposals
