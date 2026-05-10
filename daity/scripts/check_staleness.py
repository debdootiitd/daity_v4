"""Health-check on data freshness — exits non-zero on a stale state.

Calendar refreshes (`daity-refresh-calendar --apply`) and OHLCV refreshes
(`daity-refresh-ohlcv --apply`) are one-shot scripts, not scheduled. If
nobody runs them, `nse_trading_calendar.last_day` and `raw_ohlcv` max-ts
silently fall behind today, and downstream code (splitter, features,
strategy) starts producing wrong / empty answers without anyone noticing.

This script is the canary. Run it from cron / a periodic task; it returns:
  - exit 0  if every check is within tolerance
  - exit 1  if any check is stale, with a per-check report on stdout

Run:
    daity-check-staleness                      # default thresholds
    daity-check-staleness --calendar-min-lead 90    # require at least 90 cal days
    daity-check-staleness --ohlcv-max-staleness 5   # raw_ohlcv must be ≤5d stale

Recommended cron (Mon-Fri 18:30 IST = 13:00 UTC, after market close):
    30 13 * * 1-5  cd /path/to/daity && uv run daity-check-staleness
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime

import click
from rich.console import Console
from rich.table import Table

from daity.data.bq import BQClient
from daity.utils.env import EnvError, load_env
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()


def _calendar_last_day(bq: BQClient) -> date | None:
    """Most recent trade_date in `nse_trading_calendar`."""
    fq = bq.cfg.fq_table("nse_trading_calendar")
    rows = bq.query_rows(f"SELECT MAX(trade_date) AS last_day FROM `{fq}`")
    if not rows or rows[0]["last_day"] is None:
        return None
    last = rows[0]["last_day"]
    return last if isinstance(last, date) else date.fromisoformat(str(last)[:10])


def _ohlcv_max_ts_per_scale(bq: BQClient) -> dict[str, datetime | None]:
    """Most recent ts per scale in `raw_ohlcv`."""
    fq = bq.cfg.fq_table("raw_ohlcv")
    sql = f"SELECT scale, MAX(ts) AS last_ts FROM `{fq}` GROUP BY scale"
    out: dict[str, datetime | None] = {}
    for r in bq.query_rows(sql):
        ts = r.get("last_ts")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        out[r["scale"]] = ts
    return out


def _trading_days_between(bq: BQClient, start: date, end: date) -> int:
    """Count trading days strictly after `start` and on/before `end`.

    Used for "how many trading days have happened since data was last refreshed".
    """
    fq = bq.cfg.fq_table("nse_trading_calendar")
    sql = (
        f"SELECT COUNT(*) AS n FROM `{fq}` "
        f"WHERE trade_date > DATE '{start.isoformat()}' "
        f"  AND trade_date <= DATE '{end.isoformat()}'"
    )
    rows = bq.query_rows(sql)
    return int(rows[0]["n"]) if rows else 0


@click.command()
@click.option("--calendar-min-lead", default=30, show_default=True,
              help="Calendar must extend ≥ N calendar days past today.")
@click.option("--ohlcv-max-staleness", default=5, show_default=True,
              help="raw_ohlcv max-ts must be ≤ N TRADING days stale per scale.")
def main(calendar_min_lead: int, ohlcv_max_staleness: int) -> None:
    """Verify data freshness; exit 1 on any stale state."""
    try:
        cfg = load_env()
    except EnvError as exc:
        console.print(f"[red]Environment error:[/red] {exc}")
        sys.exit(2)
    bq = BQClient(cfg)

    today = datetime.now(UTC).date()
    failures: list[str] = []
    tbl = Table(title=f"daity_v3 staleness — {today}")
    tbl.add_column("check"); tbl.add_column("got"); tbl.add_column("threshold"); tbl.add_column("status")

    # 1) Calendar lead.
    last_cal = _calendar_last_day(bq)
    if last_cal is None:
        failures.append("calendar: no rows in nse_trading_calendar")
        tbl.add_row("calendar last_day", "(empty)", f"≥ today + {calendar_min_lead}d", "FAIL")
    else:
        lead_days = (last_cal - today).days
        ok = lead_days >= calendar_min_lead
        tbl.add_row("calendar last_day",
                    f"{last_cal} (lead {lead_days}d)",
                    f"≥ today + {calendar_min_lead}d",
                    "OK" if ok else "STALE")
        if not ok:
            failures.append(
                f"calendar: lead is {lead_days} days, threshold {calendar_min_lead}; "
                f"run `daity-refresh-calendar --apply`"
            )

    # 2) OHLCV freshness per scale.
    last_per_scale = _ohlcv_max_ts_per_scale(bq)
    for scale in ("5m", "15m", "60m", "day"):
        ts = last_per_scale.get(scale)
        if ts is None:
            failures.append(f"raw_ohlcv {scale}: no data")
            tbl.add_row(f"raw_ohlcv {scale} max_ts", "(empty)",
                        f"≤ {ohlcv_max_staleness} trading days stale", "FAIL")
            continue
        staleness_trading_days = _trading_days_between(
            bq, ts.astimezone(UTC).date(), today,
        )
        ok = staleness_trading_days <= ohlcv_max_staleness
        tbl.add_row(
            f"raw_ohlcv {scale} max_ts",
            f"{ts.isoformat()} ({staleness_trading_days} trading-d stale)",
            f"≤ {ohlcv_max_staleness} trading days",
            "OK" if ok else "STALE",
        )
        if not ok:
            failures.append(
                f"raw_ohlcv {scale}: {staleness_trading_days} trading days stale, "
                f"threshold {ohlcv_max_staleness}; run `daity-refresh-ohlcv --apply`"
            )

    console.print(tbl)
    if failures:
        console.print(f"\n[red]STALE — {len(failures)} check(s) failed:[/red]")
        for f in failures:
            console.print(f"  • {f}")
        sys.exit(1)
    console.print("\n[green]ALL FRESH[/green]")


if __name__ == "__main__":
    main()
