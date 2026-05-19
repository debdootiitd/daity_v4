"""Worker module for parallel cohort assembly in online_walk_forward.

Each worker process keeps its OWN assembler instance (with its own
internal series cache). The main process sends (date, anchor_ist_time)
tuples; workers return CohortSample objects (picklable).

Usage from main process:

    from concurrent.futures import ProcessPoolExecutor
    from daity.scripts._assembly_worker import init_worker, assemble_for

    executor = ProcessPoolExecutor(
        max_workers=N,
        initializer=init_worker,
        initargs=(feature_root, cache_root, universe_end, channels, as_of_iso),
    )
    for result in executor.map(assemble_for, jobs):
        ...
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, UTC
from pathlib import Path

# Module-level globals populated by init_worker(). Each worker process
# gets its own copy.
_assembler = None
_calendar = None


def init_worker(
    feature_root: str,
    cache_root: str,
    universe_end: str,
    channels: tuple[str, ...],
    as_of_iso: str,
) -> None:
    """Initialize per-worker state. Called once when a worker process starts."""
    global _assembler, _calendar

    # Local imports keep child startup fast
    import polars as pl
    from daity.data.symbols import SymbolMaster
    from daity.data.calendar_nse import NSECalendar
    from daity.data.cohort_dataset import CohortAssembler
    from daity.data.parquet_store import ParquetStore

    master = SymbolMaster.from_cache(Path(cache_root))
    all_syms = sorted(master.frame["symbol"].to_list())
    univ_end_dt = datetime.fromisoformat(universe_end).replace(tzinfo=UTC)
    t_start = datetime(2019, 1, 1, tzinfo=UTC)
    day_root = Path(feature_root) / "day"
    alive = []
    for s in all_syms:
        p = day_root / f"{s}.parquet"
        if not p.exists():
            continue
        try:
            df = pl.read_parquet(p, columns=["ts"])
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
        for s in all_syms
    })
    sec_to_id = {s: i for i, s in enumerate(all_sectors)}
    sec_by_sym = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }

    _calendar = NSECalendar.from_cache(Path(cache_root))
    store = ParquetStore(Path(feature_root))
    as_of = datetime.fromisoformat(as_of_iso)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    _assembler = CohortAssembler(
        store=store,
        calendar=_calendar,
        universe=universe,
        sector_id_by_symbol=sec_by_sym,
        as_of=as_of,
        channels=tuple(channels),
    )


def _ist_to_utc(d, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def assemble_for(args_tuple):
    """Assemble (pred, train) cohort samples for one (date, anchor_time) job.

    args_tuple format: (di, ai, date, t_anchor, do_predict,
                        train_anchor_utc_or_None)
    Returns: (di, ai, date, t_anchor, anchor_utc, train_anchor_utc, do_predict,
              pred_sample, train_sample)
    """
    di, ai, d, t_anchor, do_predict, train_anchor_utc = args_tuple
    anchor_utc = _ist_to_utc(d, t_anchor)
    pred_sample = _assembler.assemble(anchor_utc) if do_predict else None
    train_sample = _assembler.assemble(train_anchor_utc) if train_anchor_utc else None
    return (di, ai, d, t_anchor, anchor_utc, train_anchor_utc, do_predict,
            pred_sample, train_sample)
