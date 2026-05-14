"""`daity-loss-trajectory` — analyze a posttrain log's per-horizon loss
trajectory and emit a "plateaued / still descending" verdict.

For each horizon found in the log:
- bin steps into ~25 chunks
- compute mean loss per bin (smoothed via per-bin avg)
- compute slope on the LAST 25% of training (the "tail")
- compute the total descent (last_bin - first_bin) and % drop
- classify the trajectory:
    DESCENDING : tail-slope <= -5e-6 (model still learning)
    PLATEAUED  : -5e-6 < tail-slope <= 0 (flat-ish)
    DIVERGING  : tail-slope > 0 (loss rising — bad)

Output is both a per-horizon table and a per-horizon ASCII sparkline.

Recommendation logic:
- If ANY horizon is DESCENDING and overall-descent > 5%, suggest extending
  max_steps to 1.5x or 2x the current count.
- If ALL horizons are PLATEAUED, current step budget is sufficient.
- If ANY horizon is DIVERGING, flag training instability.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

STEP_PAT = re.compile(r"step=(\d+) h=(\S+) loss=([+-]?[0-9.eE+-]+)")
SPARK_CHARS = "▁▂▃▄▅▆▇█"

console = Console()


def _spark(values: list[float]) -> str:
    """Render `values` as a unicode sparkline."""
    if not values: return ""
    vmin, vmax = min(values), max(values)
    rng = max(vmax - vmin, 1e-9)
    out = []
    n_chars = len(SPARK_CHARS) - 1
    for v in values:
        idx = int(round((v - vmin) / rng * n_chars))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def _parse_log(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {horizon: [(step, loss), ...]} from a posttrain log."""
    by_h: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for ln in log_path.read_text(errors="replace").splitlines():
        m = STEP_PAT.search(ln)
        if not m: continue
        step, h, loss = int(m.group(1)), m.group(2), float(m.group(3))
        by_h[h].append((step, loss))
    for h in by_h:
        by_h[h].sort(key=lambda t: t[0])
    return by_h


def _bin_smooth(
    points: list[tuple[int, float]], n_bins: int = 25,
) -> list[tuple[float, float]]:
    """Bin points into `n_bins` equal-width step buckets, mean-loss per bin."""
    if not points: return []
    s0 = points[0][0]
    s1 = points[-1][0]
    if s1 <= s0: return [(float(s0), points[0][1])]
    width = (s1 - s0) / n_bins
    buckets: list[list[float]] = [[] for _ in range(n_bins)]
    centers: list[float] = []
    for s, l in points:
        b = min(int((s - s0) / width), n_bins - 1)
        buckets[b].append(l)
    out: list[tuple[float, float]] = []
    for i, vals in enumerate(buckets):
        if not vals: continue
        c = s0 + width * (i + 0.5)
        out.append((c, sum(vals) / len(vals)))
    return out


def _slope(xy: list[tuple[float, float]]) -> float:
    """OLS slope on (x, y) — units: dloss/dstep."""
    n = len(xy)
    if n < 2: return 0.0
    sx = sum(x for x, _ in xy)
    sy = sum(y for _, y in xy)
    sxx = sum(x*x for x, _ in xy)
    sxy = sum(x*y for x, y in xy)
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else 0.0


def _classify(
    tail_slope: float,
    last_bin_avg: float,
    tail_first_avg: float,
    plateau_thr_pct: float = 2.0,
) -> str:
    """Classify by relative descent in tail, not raw slope.

    `tail_first_avg` = mean loss in the first half of the tail.
    `last_bin_avg`   = mean loss in the last 1-2 bins of the tail.
    If last < tail_first by >`plateau_thr_pct`%, still descending.
    If last > tail_first by >`plateau_thr_pct`%, diverging.
    Otherwise plateaued.
    """
    if tail_first_avg <= 1e-9:
        return "PLATEAUED"
    rel = 100.0 * (last_bin_avg - tail_first_avg) / abs(tail_first_avg)
    if rel <= -plateau_thr_pct:
        return "DESCENDING"
    if rel >= plateau_thr_pct:
        return "DIVERGING"
    return "PLATEAUED"


@click.command()
@click.argument("log_path", type=click.Path(path_type=Path, exists=True))
@click.option("--n-bins", type=int, default=25, show_default=True,
              help="Number of step buckets for smoothing.")
@click.option("--tail-frac", type=float, default=0.25, show_default=True,
              help="Fraction of training to use for tail-slope estimate.")
@click.option("--plateau-thr-pct", type=float, default=2.0, show_default=True,
              help="% rel change in tail (last vs first half of tail) below "
                   "which we call it 'plateaued'. Default 2%.")
def main(log_path: Path, n_bins: int, tail_frac: float, plateau_thr_pct: float) -> None:
    """Analyze loss trajectory and recommend whether to extend training."""
    by_h = _parse_log(log_path)
    if not by_h:
        console.print(f"[red]no step= lines found in {log_path}[/red]")
        sys.exit(1)

    table = Table(title=f"Loss trajectory: {log_path.name}")
    table.add_column("horizon")
    table.add_column("n_steps")
    table.add_column("first_loss")
    table.add_column("last_loss")
    table.add_column("% descent")
    table.add_column("tail_slope")
    table.add_column("verdict")
    table.add_column("sparkline (smoothed)", width=30)

    overall = {"DESCENDING": 0, "PLATEAUED": 0, "DIVERGING": 0}
    max_descent = 0.0
    tails_still_falling = []

    for h, pts in sorted(by_h.items()):
        smoothed = _bin_smooth(pts, n_bins=n_bins)
        if not smoothed: continue
        first = smoothed[0][1]
        last = smoothed[-1][1]
        pct = 100.0 * (first - last) / max(abs(first), 1e-9)
        max_descent = max(max_descent, pct)

        k_tail = max(4, int(len(smoothed) * tail_frac))
        tail = smoothed[-k_tail:]
        slope = _slope(tail)
        half = max(2, len(tail) // 2)
        tail_first_avg = sum(y for _, y in tail[:half]) / half
        last_bin_avg = sum(y for _, y in tail[-max(2, half//2):]) / max(2, half//2)
        verdict = _classify(slope, last_bin_avg, tail_first_avg, plateau_thr_pct)
        overall[verdict] += 1
        if verdict == "DESCENDING":
            tails_still_falling.append((h, slope, pct))

        color = {"DESCENDING": "yellow", "PLATEAUED": "green", "DIVERGING": "red"}[verdict]
        table.add_row(
            h, f"{len(pts)}", f"{first:.3f}", f"{last:.3f}",
            f"{pct:+.1f}%", f"{slope:+.2e}",
            f"[{color}]{verdict}[/{color}]",
            _spark([y for _, y in smoothed]),
        )

    console.print(table)
    console.print()

    # Recommendation logic.
    if overall["DIVERGING"] > 0:
        console.print(
            "[red bold]WARNING:[/red bold] one or more horizons is DIVERGING. "
            "Training is unstable — DO NOT extend without lowering LR or "
            "investigating gradient explosions."
        )
    elif overall["DESCENDING"] > 0:
        h_summary = ", ".join(
            f"{h} (slope={s:+.1e}, descent={d:+.1f}%)"
            for h, s, d in tails_still_falling
        )
        new_steps = "1.5x-2x current max_steps"
        console.print(
            f"[yellow bold]RECOMMEND EXTEND.[/yellow bold] {overall['DESCENDING']} "
            f"of {sum(overall.values())} horizons still descending in the tail: "
            f"{h_summary}. Recommend running with [bold]max_steps={new_steps}[/bold]."
        )
    else:
        console.print(
            f"[green bold]CONVERGED.[/green bold] All {overall['PLATEAUED']} "
            f"horizons plateaued in the tail. Current max_steps is sufficient; "
            f"more steps unlikely to improve rank-IC."
        )


if __name__ == "__main__":
    main()
