"""Dump per-(date, stock, horizon) predictions from a static CohortModel ckpt.

Output schema (parquet) matches `online_walk_forward.py`'s predictions:
    date, anchor_us, stock, horizon, pred_lr, real_lr

For each trading day at the configured anchor, runs one forward pass and
records all (stock × ALL 10 horizons) predictions + realized labels.
Use this when you need v11 STATIC's predictions including the dropped
day_plus_3 / day_plus_5 horizons for downstream strategy experimentation.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from daity.data.calendar_nse import NSECalendar
from daity.data.cohort_dataset import (
    CohortAssembler,
    DEFAULT_COHORT_CHANNELS,
    N_REGIME_FEATS,
)
from daity.data.intraday_horizons import HORIZONS
from daity.data.parquet_store import ParquetStore
from daity.data.symbols import SymbolMaster
from daity.models.cohort.cohort_module import CohortModel


def _ist_to_utc(d: date, t: dtime) -> datetime:
    return (datetime.combine(d, t) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    ap.add_argument("--feature-root", type=Path, default=Path("data/features_parquet"))
    ap.add_argument("--universe-end", type=str, default="2024-12-31")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end",   type=str, required=True)
    ap.add_argument("--anchor-ist", type=str, default="15:25")
    ap.add_argument("--d-model",          type=int, default=480)
    ap.add_argument("--stock-enc-layers", type=int, default=1)
    ap.add_argument("--n-heads",          type=int, default=8)
    ap.add_argument("--ffn-ratio",        type=int, default=4)
    ap.add_argument("--n-market-layers",  type=int, default=4)
    ap.add_argument("--n-cross-layers",   type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, m = (int(x) for x in args.anchor_ist.split(":"))
    anchor_ist = dtime(h, m)

    master = SymbolMaster.from_cache(args.cache_root)
    all_syms = sorted(master.frame["symbol"].to_list())
    univ_end = datetime.fromisoformat(args.universe_end).replace(tzinfo=UTC)
    t_start = datetime(2019, 1, 1, tzinfo=UTC)
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
    all_sectors = sorted({
        (master.lookup(s) or {}).get("macro_sector") or "Unknown"
        for s in all_syms
    })
    sec_to_id = {s: i for i, s in enumerate(all_sectors)}
    sec_by = {
        s: sec_to_id[(master.lookup(s) or {}).get("macro_sector") or "Unknown"]
        for s in universe
    }

    calendar = NSECalendar.from_cache(args.cache_root)
    store = ParquetStore(args.feature_root)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    as_of = end_dt + timedelta(days=14)
    assembler = CohortAssembler(
        store=store, calendar=calendar,
        universe=universe, sector_id_by_symbol=sec_by, as_of=as_of,
        channels=DEFAULT_COHORT_CHANNELS,
    )

    # Try to detect stock_enc_layers from ckpt by inspecting state_dict.
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m_sd = sd.get("cohort_init_state_dict", sd.get("model_state_dict", sd))
    # Count backbone layers from keys like 'stock_encoder.backbone.layers.0.norm1.weight'
    layer_ids = set()
    for k in m_sd.keys():
        if k.startswith("stock_encoder.backbone.layers."):
            parts = k.split(".")
            # parts = ['stock_encoder', 'backbone', 'layers', '<idx>', ...]
            if len(parts) >= 4 and parts[3].isdigit():
                layer_ids.add(int(parts[3]))
    if layer_ids:
        bb_layers = max(layer_ids) + 1
        args.stock_enc_layers = bb_layers
        print(f"detected stock_enc_layers={bb_layers}", flush=True)

    model = CohortModel(
        scales=("5m", "15m", "60m", "day"),
        num_channels=len(DEFAULT_COHORT_CHANNELS),
        d_model=args.d_model,
        stock_enc_layers=args.stock_enc_layers,
        n_heads=args.n_heads,
        ffn_ratio=args.ffn_ratio,
        n_stocks=len(universe),
        n_sectors=len(all_sectors),
        n_market_layers=args.n_market_layers,
        n_cross_layers=args.n_cross_layers,
        market_n_heads=args.n_heads,
        cross_n_heads=args.n_heads,
        n_regime_feats=N_REGIME_FEATS,
    ).to(device).eval()
    missing, unexpected = model.load_state_dict(m_sd, strict=False)
    print(f"loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # Iterate trading days
    cur = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    days = []
    while cur <= end_date:
        if calendar.is_trading_day(cur):
            days.append(cur)
        cur = cur + timedelta(days=1)
    print(f"{len(days)} trading days from {args.start} to {args.end}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, d in enumerate(days):
        anchor_utc = _ist_to_utc(d, anchor_ist)
        sample = assembler.assemble(anchor_utc)
        if sample is None:
            continue
        batch = {
            "x_by_scale": {sc: v.unsqueeze(0).to(device)
                           for sc, v in sample.x_by_scale.items()},
            "stock_ids":  sample.stock_ids.unsqueeze(0).to(device),
            "sector_ids": sample.sector_ids.unsqueeze(0).to(device),
            "anchor_ts":  torch.tensor(
                [int(anchor_utc.timestamp() * 1_000_000)],
                dtype=torch.int64, device=device,
            ),
            "regime_feats": sample.regime_feats.unsqueeze(0).to(device),
        }
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            model_out = model(batch)
        pred = model_out["pred"] if isinstance(model_out, dict) else (
            model_out[0] if isinstance(model_out, tuple) else model_out
        )
        pred_np = pred[0].float().cpu().numpy()                       # (N, H)
        labels_np = sample.labels.cpu().numpy()
        label_v_np = sample.label_validity_per_stock.cpu().numpy()
        anchor_us = int(anchor_utc.timestamp() * 1_000_000)
        for s_idx, sym in enumerate(sample.symbols):
            for h_idx, hh in enumerate(HORIZONS):
                if not label_v_np[s_idx, h_idx]:
                    continue
                rows.append({
                    "date": d,
                    "anchor_us": anchor_us,
                    "stock": sym,
                    "horizon": hh.name,
                    "pred_lr": float(pred_np[s_idx, h_idx]),
                    "real_lr": float(labels_np[s_idx, h_idx]),
                })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(days)} days, {len(rows)} rows, "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.out)
    print(f"DONE: {len(rows)} rows → {args.out}; elapsed {time.time()-t0:.1f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
