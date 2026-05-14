"""Phase 3.4c — cross-sectional residualization of forward-return labels.

Senior-scientist Idea 1: predict the cross-sectional residual at each
anchor `t` instead of the raw `y_log_return`. Removes the part of return
variance that's mechanically explained by market beta + sector co-
movement, leaving the model to fit the *idiosyncratic* signal that
actually maps to rank-IC.

The residualization is **sector-relative cross-sectional demeaning**:

    y_resid_i = y_i - mean_{j ∈ sector(i)} y_j

This is the simplest residualization that still changes ranks (pure
universe demeaning wouldn't — adding a constant to every y leaves the
ListMLE rank order unchanged). Sector membership lifts symbols A and B
in different sectors to different baselines, so their *relative* rank
can flip.

Why not full per-symbol β·market residualization (Fama-MacBeth style)?
- Requires per-symbol rolling β estimation (~252-day window) → extra
  data infra + leakage gates.
- Sector demeaning captures most of the same effect with no β fit
  (sectors group similar-β symbols by construction in Indian large
  caps: banks ~1.1, IT ~0.9, FMCG ~0.7).
- Future enhancement: layer β residualization on top of sector once
  the simpler form is validated.

Per scientist's guidance: ONLY residualize daily horizons (1d, 2d).
Intraday β is noisier and the 30m / 120m signal is already largely
idiosyncratic at those scales.

Output: parquet per (horizon, symbol) at `{out_root}/{horizon}/{symbol}.parquet`
with the same schema as `data/labels/` plus a `y_resid` column. The
cross-section sampler picks `y` vs `y_resid` based on `RunConfig.label_mode`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from daity.data.labels import ALL_HORIZONS
from daity.utils.logging import get_logger

log = get_logger(__name__)


def _load_labels(label_root: Path, horizon: str, symbol: str) -> pl.DataFrame:
    path = label_root / horizon / f"{symbol}.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def compute_residual_labels(
    *,
    label_root: Path,
    out_root: Path,
    symbol_to_sector: dict[str, str],
    horizons: tuple[str, ...] = ("1d", "2d"),
    universe: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Compute sector-relative residualized labels for `horizons`.

    For each horizon:
      1. Load (symbol, t, y, valid) from every symbol's label parquet.
      2. Group by t. For each cross-section: for each sector, compute
         the mean `y` over valid rows in that sector.
      3. `y_resid_i = y_i - sector_mean_i_at_t`.
      4. Write `data/labels_resid/{horizon}/{symbol}.parquet` carrying
         all original columns plus `y_resid` (float32).

    Symbols with no sector mapping pool into a synthetic "_no_sector"
    bucket — the residual for those is just universe-mean demeaning.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for h in horizons:
        if h not in ALL_HORIZONS:
            msg = f"unknown horizon {h!r}; expected one of {ALL_HORIZONS}"
            raise ValueError(msg)
        h_out = out_root / h
        h_out.mkdir(parents=True, exist_ok=True)

        # Step 1: build long-format (symbol, t, y, valid_label, sector).
        rows: list[pl.DataFrame] = []
        syms = universe or sorted(
            p.stem for p in (label_root / h).glob("*.parquet")
        )
        for sym in syms:
            df = _load_labels(label_root, h, sym)
            if df.height == 0: continue
            sec = symbol_to_sector.get(sym, "_no_sector")
            df = df.with_columns(
                pl.lit(sym, dtype=pl.Utf8).alias("symbol"),
                pl.lit(sec, dtype=pl.Utf8).alias("sector"),
            )
            rows.append(df)
        if not rows:
            counts[h] = 0
            continue
        long = pl.concat(rows, how="vertical_relaxed")
        # Filter to valid only for the sector-mean computation; keep all
        # for the per-symbol write.
        valid = long.filter(pl.col("valid_label"))

        # Step 2: sector mean per (t, sector).
        sector_means = (
            valid.group_by(["t", "sector"])
                .agg(pl.col("y_log_return").mean().alias("y_sector_mean"))
        )
        # Step 3: join back and compute residual.
        joined = long.join(sector_means, on=["t", "sector"], how="left")
        joined = joined.with_columns(
            (pl.col("y_log_return") - pl.col("y_sector_mean"))
                .cast(pl.Float32)
                .alias("y_resid"),
        )

        # Step 4: write per-symbol parquet.
        for sym in syms:
            out_path = h_out / f"{sym}.parquet"
            if out_path.exists() and not overwrite:
                continue
            sub = (
                joined.filter(pl.col("symbol") == sym)
                      .drop(["symbol", "sector"])
                      .sort("t")
            )
            if sub.height == 0:
                continue
            sub.write_parquet(out_path)
            counts[h] = counts.get(h, 0) + sub.height
        log.info(
            "residualized %s: %d total rows across %d symbols",
            h, counts.get(h, 0), len(syms),
        )
    return counts


def load_sector_mapping(symbol_master_path: Path) -> dict[str, str]:
    """Load `symbol → sector` mapping from the SymbolMaster parquet.

    Falls back to '_no_sector' for any symbol without a sector value.
    Schema expected: columns `symbol`, `sector` (per
    `daity/data/symbols.py`'s SymbolMaster frame).
    """
    if not symbol_master_path.exists():
        log.warning(
            "symbol_master parquet missing at %s — all symbols will pool "
            "into '_no_sector' (residualization reduces to universe demean).",
            symbol_master_path,
        )
        return {}
    df = pl.read_parquet(symbol_master_path)
    if "symbol" not in df.columns or "sector" not in df.columns:
        log.warning(
            "symbol_master parquet at %s has columns %s — need 'symbol' "
            "and 'sector'. Skipping sector mapping.",
            symbol_master_path, df.columns,
        )
        return {}
    out: dict[str, str] = {}
    for sym, sec in zip(df["symbol"], df["sector"], strict=True):
        if sym is None: continue
        out[sym] = sec or "_no_sector"
    return out


def sector_mapping_from_default_csv(default_csv: Path) -> dict[str, str]:
    """Fallback loader: a simple `symbol,sector` CSV (no header)."""
    if not default_csv.exists():
        return {}
    out: dict[str, str] = {}
    for line in default_csv.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            out[parts[0]] = parts[1] or "_no_sector"
    return out
