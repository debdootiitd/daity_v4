"""PyTorch dataset that yields PAIRS of (stock, anchor) windows for two-tower
contrastive pretraining.

Each iteration emits a paired sample:
    {
        "a": batch_dict for tower A (single-sample format, B=1, N=1),
        "b": batch_dict for tower B (single-sample format, B=1, N=1),
        "pair_type": "same_stock" | "sector_alpha",
        "cosine":    float,
    }

The pair table is read from a parquet produced by `daity.data.cohort_pair_miner`.
Per-anchor window assembly uses the same `CohortAssembler` machinery as the
prediction trainer so the data semantics match.

For efficiency we cache the assembled per-(stock, anchor) record so the same
anchor is not assembled twice across passes through the loader.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset

from daity.data.cohort_dataset import CohortAssembler


class CohortPairDataset(IterableDataset):
    """`IterableDataset` yielding (a, b) paired single-stock samples.

    Construction:
        assembler:   already-built CohortAssembler.
        pairs_path:  parquet of pre-mined positive pairs (from
                     `daity.data.cohort_pair_miner`).
        shuffle:     shuffle pair order per epoch.
        seed:        rng seed.

    The dataset is "infinite" only in the sense that `__iter__` performs one
    full shuffled pass per epoch. The caller's training loop controls how many
    epochs to do.

    Per-stock single-sample assembly:
        For each anchor in the pair, we call `assembler.assemble(anchor_utc)`
        and then EXTRACT the row for the specific stock in the pair. The
        per-anchor cohort assembly is cached so the same anchor is not
        re-assembled across pairs.
    """

    def __init__(
        self,
        *,
        assembler: CohortAssembler,
        pairs_path: Path,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.assembler = assembler
        self.pairs = pl.read_parquet(pairs_path)
        self.shuffle = shuffle
        self.seed = seed
        self._cohort_cache: dict[int, object] = {}
        # Pre-compute (symbol → stock_id) for quick lookup at iteration time.
        self._stock_to_idx = assembler.stock_id_by_symbol
        # Per-stock fast index inside the cohort sample.
        self._sym_to_pos_cache: dict[int, dict[str, int]] = {}

    def _get_cohort(self, anchor_us: int):
        c = self._cohort_cache.get(anchor_us)
        if c is not None:
            return c
        anchor_utc = datetime.fromtimestamp(anchor_us / 1_000_000, tz=UTC)
        sample = self.assembler.assemble(anchor_utc)
        if sample is not None:
            # Build symbol→position map for fast row lookup.
            self._sym_to_pos_cache[anchor_us] = {
                s: i for i, s in enumerate(sample.symbols)
            }
        self._cohort_cache[anchor_us] = sample
        return sample

    def _extract_stock_sample(
        self, anchor_us: int, stock: str,
    ) -> dict | None:
        """Pull a single (stock, anchor) sample from the cohort cache as a
        batch dict (B=1, N=1). Returns None if the stock isn't alive at this
        anchor or assembly failed.
        """
        sample = self._get_cohort(anchor_us)
        if sample is None:
            return None
        pos_map = self._sym_to_pos_cache.get(anchor_us)
        if pos_map is None:
            return None
        pos = pos_map.get(stock)
        if pos is None:
            return None
        x_by_scale = {
            sc: v[pos].unsqueeze(0).unsqueeze(0)             # (1, 1, L, C)
            for sc, v in sample.x_by_scale.items()
        }
        return {
            "x_by_scale": x_by_scale,
            "stock_ids":  sample.stock_ids[pos].view(1, 1),
            "sector_ids": sample.sector_ids[pos].view(1, 1),
            "anchor_ts":  torch.tensor(
                [int(anchor_us)], dtype=torch.int64,
            ),
            "regime_feats": sample.regime_feats.unsqueeze(0),
        }

    def __iter__(self) -> Iterator[dict]:
        n = self.pairs.shape[0]
        order = list(range(n))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)
        rows = self.pairs.to_dicts()
        for i in order:
            row = rows[i]
            a = self._extract_stock_sample(row["anchor_a_us"], row["stock_a"])
            b = self._extract_stock_sample(row["anchor_b_us"], row["stock_b"])
            if a is None or b is None:
                continue
            yield {
                "a": a, "b": b,
                "pair_type": row["pair_type"],
                "cosine":    float(row["cosine"]),
            }
