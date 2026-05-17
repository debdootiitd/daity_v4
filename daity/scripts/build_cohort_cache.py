"""Build a disk-backed cache of cohort samples for ablation training.

Each worker process holds its own CohortAssembler (so the per-stock series
cache lives in process-local memory, no IPC overhead per anchor). The job
queue is a flat list of (date, anchor_ist) pairs; results are written to
disk as `<date>_<HHMM>.pt`. Already-existing files are skipped (idempotent
restart).

Usage:
  python -m daity.scripts.build_cohort_cache \
      --start 2024-01-01 --end 2026-04-30 \
      --anchors-ist 09:15-15:30:5m \
      --workers 4 \
      --out-dir cache/cohort_samples_v1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import polars as pl

from daity.data.cohort_cache import cache_filename, ist_to_utc, write_sample


# Worker-local state (initialized once per process)
_WORKER_STATE: dict = {}


def _parse_anchors(spec: str) -> list[dtime]:
    """09:15-15:30:5m -> [09:15, 09:20, ..., 15:30] or 09:30,12:00,15:25 -> list."""
    if "-" in spec and ":" in spec.split("-", 1)[1]:
        range_part, step_part = spec.rsplit(":", 1)
        start_s, end_s = range_part.split("-")
        if not step_part.endswith("m"):
            raise ValueError("step must end with 'm'")
        step_min = int(step_part[:-1])
        sh, sm = (int(x) for x in start_s.split(":"))
        eh, em = (int(x) for x in end_s.split(":"))
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        out = []
        cur = start_min
        while cur <= end_min:
            out.append(dtime(cur // 60, cur % 60))
            cur += step_min
        return out
    return [dtime(int(t.split(":")[0]), int(t.split(":")[1])) for t in spec.split(",") if t.strip()]


def _trading_days(start: date, end: date, calendar) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        if calendar.is_trading_day(cur):
            out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def _worker_init(cache_root: str, feature_root: str, universe_end: str,
                 universe_source: str, online_end: str):
    """Each worker process inits its own assembler (called once on pool startup)."""
    from daity.data.calendar_nse import NSECalendar
    from daity.data.cohort_dataset import CohortAssembler, DEFAULT_COHORT_CHANNELS
    from daity.data.parquet_store import ParquetStore
    from daity.data.symbols import SymbolMaster

    master = SymbolMaster.from_cache(Path(cache_root))
    all_syms = sorted(master.frame["symbol"].to_list())
    univ_end_dt = datetime.fromisoformat(universe_end).replace(tzinfo=UTC)
    t_start = datetime(2019, 1, 1, tzinfo=UTC)
    day_root = Path(feature_root) / "day"

    if universe_source == "all":
        universe = tuple(all_syms)
    else:
        alive = []
        for s in all_syms:
            try:
                df = pl.read_parquet(day_root / f"{s}.parquet", columns=["ts"])
            except Exception:
                continue
            if df.height == 0:
                continue
            ts = df["ts"]
            if (ts >= t_start).any() and (ts <= univ_end_dt).any():
                alive.append(s)
        universe = tuple(alive)

    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in master.frame["symbol"].to_list()
    })
    sec_to_id = {s: i for i, s in enumerate(all_sectors)}
    sec_by = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }

    calendar = NSECalendar.from_cache(Path(cache_root))
    store = ParquetStore(Path(feature_root))
    online_end_dt = datetime.fromisoformat(online_end).replace(tzinfo=UTC)
    as_of = online_end_dt + timedelta(days=14)
    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe, sector_id_by_symbol=sec_by, as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )
    _WORKER_STATE["assembler"] = assembler
    _WORKER_STATE["pid"] = os.getpid()


def _worker_assemble(args_tuple) -> tuple[str, str | None]:
    """Assemble one sample and write to disk. Returns (status, error_msg_or_None)."""
    d_iso, t_hhmm, out_dir = args_tuple
    out_path = Path(out_dir) / f"{d_iso}_{t_hhmm}.pt"
    if out_path.exists():
        return ("exists", None)
    try:
        d = date.fromisoformat(d_iso)
        t = dtime(int(t_hhmm[:2]), int(t_hhmm[2:]))
        anchor_utc = ist_to_utc(d, t)
        assembler = _WORKER_STATE["assembler"]
        sample = assembler.assemble(anchor_utc)
        if sample is None:
            return ("empty", None)
        write_sample(Path(out_dir), sample, anchor_utc)
        return ("ok", None)
    except Exception as e:
        return ("err", str(e))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root",   type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--universe-end", type=str, default="2024-12-31")
    ap.add_argument("--universe-source", choices=["all", "alive_in_range"], default="all")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end",   type=str, required=True)
    ap.add_argument("--anchors-ist", type=str, default="09:15-15:30:5m")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from daity.data.calendar_nse import NSECalendar
    calendar = NSECalendar.from_cache(args.cache_root)

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)
    days = _trading_days(start_d, end_d, calendar)
    anchors = _parse_anchors(args.anchors_ist)

    jobs = []
    for d in days:
        for t in anchors:
            jobs.append((d.isoformat(), f"{t.hour:02d}{t.minute:02d}", str(args.out_dir)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache build: {len(days)} days × {len(anchors)} anchors = {len(jobs)} samples",
          flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"out: {args.out_dir}", flush=True)

    init_args = (str(args.cache_root), str(args.feature_root),
                 args.universe_end, args.universe_source, args.end)

    t0 = time.time()
    counts = {"ok": 0, "exists": 0, "empty": 0, "err": 0}
    errors = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=init_args) as pool:
        for i, (status, err) in enumerate(pool.imap_unordered(_worker_assemble, jobs, chunksize=8)):
            counts[status] = counts.get(status, 0) + 1
            if err and len(errors) < 10:
                errors.append(err)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(jobs) - i - 1) / max(rate, 1e-6)
                print(f"  {i+1}/{len(jobs)} | "
                      f"ok={counts['ok']} exists={counts['exists']} "
                      f"empty={counts['empty']} err={counts['err']} | "
                      f"{rate:.1f} samp/s | ETA {eta/60:.1f} min",
                      flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. Counts: {counts}", flush=True)
    if errors:
        print(f"First {len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
