"""Materialize positive-pair dataset for contrastive pretraining.

Pair types:
  - 'same_stock'   : same symbol at two different anchor times. Captures
                     temporal invariance (a stock should look "like itself"
                     across nearby anchors).
  - 'sector_alpha' : two different symbols in the same macro sector whose
                     realized 10-horizon log returns are highly cosine-
                     similar (>= threshold). Captures behavioral coherence
                     within a sector.

Materialization workflow:
  1. Iterate the configured (date, anchor_time_IST) grid over a training
     window (e.g. 2019-01 → 2024-11).
  2. For each anchor, look up each alive stock's realized 10-horizon log
     return vector via `CohortAssembler.assemble(anchor_utc).labels`.
  3. Save a sparse "fact table" of (stock, anchor_ts, sector_id, label_*)
     so pair mining can be done in pure numpy/polars.
  4. Mine pairs:
     - same_stock: for each stock, pair every consecutive K anchors (anchor
       gap parameterizable). Optionally cap pairs per stock.
     - sector_alpha: for each (sector, anchor), pairwise within the same
       (sector, anchor) cohort by cos(returns) > threshold. Optionally
       extend to (sector, different anchor) — same-sector with similar
       behavior at different times.
  5. Write the pair table to parquet:
       (stock_a, anchor_a_us, stock_b, anchor_b_us, pair_type, cosine,
        sector_id_a, sector_id_b)

Downstream the contrastive trainer (two-tower with shared encoder) reads
from this parquet, materializes the per-stock windows, and runs InfoNCE.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from daity.data.calendar_nse import NSECalendar
from daity.data.cohort_dataset import CohortAssembler, DEFAULT_COHORT_CHANNELS
from daity.data.intraday_horizons import N_HORIZONS, intraday_anchor_grid
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster


def _ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def build_anchor_grid(
    start: date,
    end: date,
    calendar: NSECalendar,
    anchor_times: list[dtime],
) -> list[tuple[date, dtime]]:
    out = []
    d = start
    while d <= end:
        if calendar.is_trading_day(d):
            for t in anchor_times:
                out.append((d, t))
        d = d + timedelta(days=1)
    return out


def materialize_facts(
    *,
    assembler: CohortAssembler,
    anchor_pairs: list[tuple[date, dtime]],
    sector_id_by_symbol: dict[str, int],
) -> pl.DataFrame:
    """For each (anchor, stock) pair: store realized 10-horizon log returns
    and per-horizon label_validity. Output schema:
        anchor_us (i64), date (date), stock (str), sector_id (i32),
        labels (list[f32]), label_valid (list[bool])
    """
    rows = []
    for i, (d, t) in enumerate(anchor_pairs):
        anchor_utc = _ist_to_utc(d, t)
        sample = assembler.assemble(anchor_utc)
        if sample is None:
            continue
        anchor_us = int(anchor_utc.timestamp() * 1_000_000)
        for sidx, sym in enumerate(sample.symbols):
            rows.append({
                "anchor_us": anchor_us,
                "date": d,
                "anchor_minute": t.hour * 60 + t.minute,
                "stock": sym,
                "sector_id": int(sample.sector_ids[sidx].item()),
                "labels": sample.labels[sidx].tolist(),
                "label_valid": sample.label_validity_per_stock[sidx].tolist(),
            })
        if (i + 1) % 50 == 0:
            print(f"  materialized facts for {i+1}/{len(anchor_pairs)} anchors "
                  f"({len(rows)} rows so far)", flush=True)
    return pl.DataFrame(rows)


def _norm_returns(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """L2-normalize the (n_rows, 10) labels per row, zeroing invalid entries."""
    masked = np.where(valid, labels, 0.0).astype(np.float32)
    norms = np.linalg.norm(masked, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    return masked / norms


def mine_same_stock_pairs(
    facts: pl.DataFrame,
    *,
    pairs_per_stock: int = 20,
    min_anchor_gap_days: int = 1,
    max_anchor_gap_days: int = 60,
    seed: int = 0,
) -> pl.DataFrame:
    """Each stock contributes up to `pairs_per_stock` (anchor_a, anchor_b)
    where the gap in trading days is within `[min_anchor_gap_days, max_anchor_gap_days]`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for stock, g in facts.group_by("stock"):
        sym = stock[0]
        anchors = sorted(g["anchor_us"].to_list())
        sec_id = int(g["sector_id"][0])
        if len(anchors) < 2:
            continue
        n_attempts = 0
        chosen = 0
        max_attempts = pairs_per_stock * 5
        while chosen < pairs_per_stock and n_attempts < max_attempts:
            n_attempts += 1
            i, j = sorted(rng.choice(len(anchors), size=2, replace=False))
            gap_us = anchors[j] - anchors[i]
            gap_days = gap_us / 1_000_000 / 86400
            if gap_days < min_anchor_gap_days or gap_days > max_anchor_gap_days:
                continue
            rows.append({
                "stock_a": sym, "anchor_a_us": anchors[i],
                "stock_b": sym, "anchor_b_us": anchors[j],
                "pair_type": "same_stock",
                "cosine": 1.0,  # by construction
                "sector_id_a": sec_id, "sector_id_b": sec_id,
            })
            chosen += 1
    return pl.DataFrame(rows)


def mine_sector_alpha_pairs(
    facts: pl.DataFrame,
    *,
    cos_threshold: float = 0.5,
    max_pairs_per_anchor: int = 50,
    seed: int = 0,
) -> pl.DataFrame:
    """Intra-anchor mining: at each (anchor, sector), find pairs of stocks
    whose realized 10-horizon return vectors have cosine > threshold.

    Same-sector + similar-returns = positive. Reports top `max_pairs_per_anchor`
    by cosine (deterministic).
    """
    rows = []
    rng = np.random.default_rng(seed)
    # Group by (anchor_us, sector_id) — within group, compute pairwise cosines.
    grouped = facts.group_by(["anchor_us", "sector_id"]).agg(
        [pl.col("stock"), pl.col("labels"), pl.col("label_valid")],
    )
    for row in grouped.iter_rows(named=True):
        stocks = row["stock"]
        if len(stocks) < 2:
            continue
        labels = np.asarray(row["labels"], dtype=np.float32)
        valid = np.asarray(row["label_valid"], dtype=bool)
        if labels.shape[0] != len(stocks):
            continue
        z = _norm_returns(labels, valid)                       # (M, 10)
        cos = z @ z.T                                          # (M, M)
        np.fill_diagonal(cos, -1.0)
        # Collect upper triangle pairs above threshold.
        ii, jj = np.where(np.triu(cos > cos_threshold, k=1))
        if ii.size == 0:
            continue
        # Sort by cosine descending and keep top max_pairs_per_anchor.
        scores = cos[ii, jj]
        order = np.argsort(-scores)
        keep = order[: max_pairs_per_anchor]
        for k in keep:
            i, j = int(ii[k]), int(jj[k])
            rows.append({
                "stock_a":     stocks[i],
                "anchor_a_us": row["anchor_us"],
                "stock_b":     stocks[j],
                "anchor_b_us": row["anchor_us"],
                "pair_type":   "sector_alpha",
                "cosine":      float(scores[k]),
                "sector_id_a": row["sector_id"],
                "sector_id_b": row["sector_id"],
            })
    return pl.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--train-start", type=str, default="2019-01-01")
    ap.add_argument("--train-end",   type=str, default="2024-11-30")
    ap.add_argument("--universe-end", type=str, default=None)
    # Anchor grid: trade-off coverage vs cost. Default = 1 anchor/day at 15:25 IST.
    ap.add_argument("--anchor-times", type=str, default="15:25",
                    help="Comma-separated IST anchor times HH:MM. Default 15:25 only.")
    # Pair mining
    ap.add_argument("--pairs-per-stock", type=int, default=20,
                    help="Max same-stock pairs per symbol.")
    ap.add_argument("--cos-threshold", type=float, default=0.5)
    ap.add_argument("--max-pairs-per-anchor", type=int, default=50,
                    help="Max sector_alpha pairs per (anchor, sector) group.")
    # IO
    ap.add_argument("--out-facts", type=Path, required=True,
                    help="Where to write the (anchor, stock, labels) facts parquet.")
    ap.add_argument("--out-pairs", type=Path, required=True,
                    help="Where to write the materialized pair table parquet.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    master = SymbolMaster.from_cache(args.cache_root)
    all_syms = sorted(master.frame["symbol"].to_list())
    if args.universe_end is not None:
        univ_end = datetime.fromisoformat(args.universe_end).replace(tzinfo=UTC)
    else:
        univ_end = datetime.fromisoformat(args.train_end).replace(tzinfo=UTC)
    t_start = datetime.fromisoformat(args.train_start).replace(tzinfo=UTC)
    day_root = args.feature_root / "day"
    alive = []
    for s in all_syms:
        try:
            df = pl.read_parquet(day_root / f"{s}.parquet", columns=["ts"])
        except Exception:
            continue
        if df.height == 0:
            continue
        ts = df["ts"]
        if (ts >= t_start).any() and (ts <= univ_end).any():
            alive.append(s)
    universe = tuple(alive)
    print(f"universe: {len(universe)} stocks", flush=True)

    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in all_syms
    })
    sec_to_id = {s: i for i, s in enumerate(all_sectors)}
    sec_by_sym = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }

    calendar = NSECalendar.from_cache(args.cache_root)
    store = ParquetStore(args.feature_root)
    as_of = (datetime.fromisoformat(args.train_end).replace(tzinfo=UTC)
             + timedelta(days=14))
    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe, sector_id_by_symbol=sec_by_sym, as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )

    anchor_times = []
    for s in args.anchor_times.split(","):
        h, m = s.strip().split(":")
        anchor_times.append(dtime(int(h), int(m)))
    anchor_pairs = build_anchor_grid(
        date.fromisoformat(args.train_start),
        date.fromisoformat(args.train_end),
        calendar, anchor_times,
    )
    print(f"anchor grid: {len(anchor_pairs)} (date, time) pairs", flush=True)

    args.out_facts.parent.mkdir(parents=True, exist_ok=True)
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    facts = materialize_facts(
        assembler=assembler,
        anchor_pairs=anchor_pairs,
        sector_id_by_symbol=sec_by_sym,
    )
    facts.write_parquet(args.out_facts)
    print(f"facts: {facts.shape[0]} (anchor, stock) rows → {args.out_facts}", flush=True)

    same_stock = mine_same_stock_pairs(
        facts, pairs_per_stock=args.pairs_per_stock, seed=args.seed,
    )
    sector_alpha = mine_sector_alpha_pairs(
        facts, cos_threshold=args.cos_threshold,
        max_pairs_per_anchor=args.max_pairs_per_anchor, seed=args.seed,
    )
    pairs = pl.concat([same_stock, sector_alpha], how="vertical")
    pairs.write_parquet(args.out_pairs)
    print(f"pairs: same_stock={same_stock.shape[0]}, "
          f"sector_alpha={sector_alpha.shape[0]} → {args.out_pairs}", flush=True)
    print(f"total pairs: {pairs.shape[0]}", flush=True)
    print(f"sample sector_alpha cosines stats:", flush=True)
    if sector_alpha.shape[0] > 0:
        c = sector_alpha["cosine"]
        print(f"  min={c.min():.3f}  max={c.max():.3f}  mean={c.mean():.3f}  "
              f"median={c.median():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
