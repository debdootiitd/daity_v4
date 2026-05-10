"""Per-symbol corp-action readjustment helpers.

The CLI (`daity-readjust-symbols`) is a thin orchestrator over these functions:
parse YAML config, consult the marker file for idempotency, build + run
UPDATE statements, then record the applied entries back to the marker.

The marker file (`data/applied_readjustments.json`) is the load-bearing
safety net — without it, the WHERE clause `(symbol, cutoff_date)` is *not*
idempotent, and a re-run would double-multiply the ratio. The marker stores
every applied `(symbol, cutoff_date_ist, ratio)` triple; before applying
again we filter out anything already in the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Readjustment:
    """One per-symbol corp-action adjustment."""

    symbol: str
    ratio: float
    cutoff_date_ist: date | None
    event: str
    notes: str

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            msg = f"Readjustment ratio must be positive, got {self.ratio} for {self.symbol}"
            raise ValueError(msg)
        if self.ratio == 1.0:
            msg = f"Readjustment ratio is 1.0 for {self.symbol} — would be a no-op."
            raise ValueError(msg)


def load_config(path: Path) -> list[Readjustment]:
    """Parse a corp-actions YAML into `Readjustment` records."""
    import yaml
    if not path.exists():
        msg = f"Config not found: {path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("readjustments", [])
    out: list[Readjustment] = []
    for e in entries:
        cd = e.get("cutoff_date_ist")
        # PyYAML auto-parses ISO dates → datetime.date; accept either form.
        if cd is None:
            cutoff: date | None = None
        elif isinstance(cd, date):
            cutoff = cd
        else:
            cutoff = date.fromisoformat(str(cd))
        out.append(Readjustment(
            symbol=str(e["symbol"]),
            ratio=float(e["ratio"]),
            cutoff_date_ist=cutoff,
            event=str(e.get("event", "")),
            notes=str(e.get("notes", "")),
        ))
    return out


def where_clause(r: Readjustment) -> str:
    """Build the SQL WHERE clause for one readjustment.

    `symbol = '<sym>'` AND (cutoff is None OR `DATE(ts, 'Asia/Kolkata') < <date>`).
    Symbol is single-quote-escaped defensively.
    """
    sym_lit = r.symbol.replace("'", "''")
    parts = [f"symbol = '{sym_lit}'"]
    if r.cutoff_date_ist is not None:
        parts.append(
            f"DATE(ts, 'Asia/Kolkata') < DATE '{r.cutoff_date_ist.isoformat()}'"
        )
    return " AND ".join(parts)


# ----- Idempotency marker -----


def marker_key(r: Readjustment) -> str:
    """Stable canonical string for a Readjustment — duplicate detection key."""
    cutoff = r.cutoff_date_ist.isoformat() if r.cutoff_date_ist else "ALL"
    return f"{r.symbol}|{cutoff}|{r.ratio:.10g}"


def _marker_key_from_dict(e: dict[str, Any]) -> str:
    cutoff = e.get("cutoff_date_ist") or "ALL"
    return f"{e['symbol']}|{cutoff}|{float(e['ratio']):.10g}"


def load_marker(path: Path) -> list[dict[str, Any]]:
    """Read the applied-readjustments marker. Returns [] if file is absent."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot read marker file {path}: {exc}"
        raise RuntimeError(msg) from exc
    if not isinstance(data, list):
        msg = f"Marker file {path} must contain a JSON list, got {type(data).__name__}"
        raise RuntimeError(msg)
    return data


def filter_already_applied(
    plan: list[Readjustment], marker_path: Path,
) -> tuple[list[Readjustment], list[Readjustment]]:
    """Split `plan` into (to-apply, already-applied) by consulting the marker."""
    applied = {_marker_key_from_dict(e) for e in load_marker(marker_path)}
    todo: list[Readjustment] = []
    skipped: list[Readjustment] = []
    for r in plan:
        if marker_key(r) in applied:
            skipped.append(r)
        else:
            todo.append(r)
    return todo, skipped


def record_applied(
    marker_path: Path, applied: list[Readjustment], *,
    run_id: str, n_rows_per_sym: dict[str, int],
) -> None:
    """Append entries for newly-applied readjustments. Atomic via .tmp + rename."""
    existing = load_marker(marker_path)
    now = datetime.now(UTC).isoformat()
    for r in applied:
        existing.append({
            "symbol": r.symbol,
            "cutoff_date_ist": r.cutoff_date_ist.isoformat() if r.cutoff_date_ist else None,
            "ratio": r.ratio,
            "event": r.event,
            "applied_at_utc": now,
            "run_id": run_id,
            "rows_affected": n_rows_per_sym.get(r.symbol, 0),
        })
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2, sort_keys=True))
    tmp.replace(marker_path)
