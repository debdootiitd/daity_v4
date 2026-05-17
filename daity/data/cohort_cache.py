"""Disk-backed cache of assembled cohort samples for fast ablation training.

Workflow:
  1. `build_cache(...)` (CLI: daity-build-cohort-cache) walks (date, anchor)
     pairs in parallel via multiprocessing.Pool, assembles each sample, and
     writes `<date>_<HHMM>.pt` files to a cache directory.
  2. `CohortSampleCache(cache_dir)` is a PyTorch Dataset; `__getitem__`
     `torch.load`s the file. Used with DataLoader(num_workers=N) for fast
     pipelined training loops that bypass CPU-bound CohortAssembler.

Tradeoff: ~7 MB per sample. For 5-min anchors over 16 months ≈ 112 GB on
disk. Each ablation then runs at GPU-speed only (no CPU bottleneck).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


def ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def cache_filename(d: date, t: dtime) -> str:
    return f"{d.isoformat()}_{t.hour:02d}{t.minute:02d}.pt"


def parse_cache_filename(name: str) -> tuple[date, dtime] | None:
    try:
        stem = name.removesuffix(".pt")
        date_part, time_part = stem.split("_")
        d = date.fromisoformat(date_part)
        t = dtime(int(time_part[:2]), int(time_part[2:]))
        return d, t
    except Exception:
        return None


@dataclass
class CacheRecord:
    """Minimal serialized form of a cohort sample for fast load."""
    x_by_scale: dict[str, torch.Tensor]
    stock_ids: torch.Tensor
    sector_ids: torch.Tensor
    regime_feats: torch.Tensor
    labels: torch.Tensor
    label_validity_per_stock: torch.Tensor
    validity: torch.Tensor
    symbols: tuple[str, ...]
    anchor_us: int
    date_iso: str


def write_sample(cache_dir: Path, sample, anchor_utc: datetime) -> None:
    """Serialize an assembled sample to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rec = CacheRecord(
        x_by_scale={sc: v.cpu() for sc, v in sample.x_by_scale.items()},
        stock_ids=sample.stock_ids.cpu(),
        sector_ids=sample.sector_ids.cpu(),
        regime_feats=sample.regime_feats.cpu(),
        labels=sample.labels.cpu(),
        label_validity_per_stock=sample.label_validity_per_stock.cpu(),
        validity=sample.validity.cpu(),
        symbols=tuple(sample.symbols),
        anchor_us=int(anchor_utc.timestamp() * 1_000_000),
        date_iso=anchor_utc.date().isoformat(),
    )
    d = anchor_utc.date()
    t = anchor_utc.time()
    torch.save(rec, cache_dir / cache_filename(d, t))


class CohortSampleCache(Dataset):
    """torch.utils.data.Dataset that yields CacheRecord objects from disk.

    Use with DataLoader(num_workers=8, pin_memory=True, persistent_workers=True,
    prefetch_factor=4) for max throughput.
    """

    def __init__(self, cache_dir: Path,
                 start_date: date | None = None,
                 end_date: date | None = None):
        self.cache_dir = Path(cache_dir)
        self.files: list[Path] = []
        for f in sorted(self.cache_dir.glob("*.pt")):
            parsed = parse_cache_filename(f.name)
            if parsed is None:
                continue
            d, _ = parsed
            if start_date and d < start_date: continue
            if end_date and d > end_date: continue
            self.files.append(f)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> CacheRecord:
        return torch.load(self.files[idx], weights_only=False)

    def date_at(self, idx: int) -> date | None:
        parsed = parse_cache_filename(self.files[idx].name)
        return parsed[0] if parsed else None

    def anchor_at(self, idx: int) -> dtime | None:
        parsed = parse_cache_filename(self.files[idx].name)
        return parsed[1] if parsed else None


def record_to_batch(rec: CacheRecord, device: str) -> dict:
    """Convert a CacheRecord into a model-input batch (B=1)."""
    return {
        "x_by_scale": {sc: v.unsqueeze(0).to(device, non_blocking=True)
                       for sc, v in rec.x_by_scale.items()},
        "stock_ids":  rec.stock_ids.unsqueeze(0).to(device, non_blocking=True),
        "sector_ids": rec.sector_ids.unsqueeze(0).to(device, non_blocking=True),
        "anchor_ts":  torch.tensor([rec.anchor_us], dtype=torch.int64, device=device),
        "regime_feats": rec.regime_feats.unsqueeze(0).to(device, non_blocking=True),
    }


def records_to_batch(records: list[CacheRecord], device: str) -> dict | None:
    """Stack records into a single batch dict (B=len). Returns None if shapes mismatch."""
    if not records:
        return None
    n_first = records[0].stock_ids.shape[0]
    same_n = [r for r in records if r.stock_ids.shape[0] == n_first]
    if not same_n:
        return None
    return {
        "x_by_scale": {sc: torch.stack([r.x_by_scale[sc] for r in same_n], dim=0).to(device, non_blocking=True)
                       for sc in same_n[0].x_by_scale.keys()},
        "stock_ids":  torch.stack([r.stock_ids for r in same_n], dim=0).to(device, non_blocking=True),
        "sector_ids": torch.stack([r.sector_ids for r in same_n], dim=0).to(device, non_blocking=True),
        "anchor_ts":  torch.tensor([r.anchor_us for r in same_n], dtype=torch.int64, device=device),
        "regime_feats": torch.stack([r.regime_feats for r in same_n], dim=0).to(device, non_blocking=True),
    }, [r.labels for r in same_n], [r.label_validity_per_stock for r in same_n], [r.validity for r in same_n]
