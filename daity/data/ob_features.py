"""Phase 4 step 1 — engineered L2-snapshot features.

Per DESIGN §3.4: the OB encoder consumes K=64 snapshots × ~18 features.
Sub-phase-1 minimum (8 features) — gets us to a runnable smoke; richer
feature set lands once the encoder + fusion are proven end-to-end.

Per-snapshot feature vector (8 channels):

  0. microprice         — qty-weighted top-of-book price
  1. weighted_mid_ret   — log return of (bid1+ask1)/2 vs prior snapshot
  2. spread_bps         — passthrough from BQ (already in `spread_bps`)
  3. book_pressure_l1   — (bid1_qty - ask1_qty) / (bid1_qty + ask1_qty)
  4. queue_imb_l1       — bid1_qty / (bid1_qty + ask1_qty)
  5. ofi_l1_60s_proxy   — Δ(bid1_qty) [if bid1_price unchanged or up]
                          − Δ(ask1_qty) [if ask1_price unchanged or down]
                          ÷ (bid1_qty + ask1_qty) — Cont-Stoikov OFI proxy
                          at 60s cadence (the schema's natural granularity).
  6. log_total_buy_qty  — log1p of total_buy_qty (cumulative depth side)
  7. log_total_sell_qty — log1p of total_sell_qty

All channels are scaled to be approximately O(1) so the OB encoder's
LayerNorm + RMSNorm don't need to fight to standardize them. NaN /
inf are forward-filled within symbol then zero-filled at the start.

Output: `pl.DataFrame` with columns `(ts, f_0..f_7)`, sorted by `ts`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ----- Feature names (in canonical order). -----

OB_CHANNEL_NAMES: tuple[str, ...] = (
    "microprice",
    "weighted_mid_ret",
    "spread_bps",
    "book_pressure_l1",
    "queue_imb_l1",
    "ofi_l1_60s_proxy",
    "log_total_buy_qty",
    "log_total_sell_qty",
)
OB_N_CHANNELS: int = len(OB_CHANNEL_NAMES)


# ----- Per-snapshot derivations from the wide-format L2 row. -----


def compute_ob_features(
    df: pl.DataFrame,
    *,
    drop_warmup: bool = True,
) -> pl.DataFrame:
    """Build the per-snapshot feature vector for one symbol's OB stream.

    `df` must carry the columns from `daity/data/ob_ingest.OB_COLUMNS`.
    Output schema: `(ts, f_0..f_7)` with Float32 features.

    `drop_warmup=True` drops the first row (whose `weighted_mid_ret` and
    `ofi_l1_60s_proxy` are undefined). For training, set this; for the
    cross-section lookup we don't care about row counts that much, but
    the first row's features are NaN, so dropping is sound.
    """
    if df.height == 0:
        return pl.DataFrame(
            {"ts": [], **{f"f_{i}": [] for i in range(OB_N_CHANNELS)}}
        )

    df = df.sort("ts")
    # Cast quantity columns to float to avoid integer overflow in pressure.
    df = df.with_columns([
        pl.col("bid1_qty").cast(pl.Float64),
        pl.col("ask1_qty").cast(pl.Float64),
        pl.col("total_buy_qty").cast(pl.Float64),
        pl.col("total_sell_qty").cast(pl.Float64),
    ])

    bid_qty = pl.col("bid1_qty")
    ask_qty = pl.col("ask1_qty")
    bid_p = pl.col("bid1_price")
    ask_p = pl.col("ask1_price")
    top_sum_qty = bid_qty + ask_qty

    # f_0 microprice (qty-weighted top mid).
    microprice = (bid_qty * ask_p + ask_qty * bid_p) / (top_sum_qty + 1e-9)
    mid = (bid_p + ask_p) * 0.5

    # f_1 weighted_mid_ret — log return vs prior snapshot (NaN at row 0).
    mid_ret = (mid.log() - mid.log().shift(1))

    # f_2 spread_bps — passthrough (already in BQ schema).
    spread_bps = pl.col("spread_bps")

    # f_3 book_pressure_l1.
    book_pressure = (bid_qty - ask_qty) / (top_sum_qty + 1e-9)

    # f_4 queue_imb_l1.
    queue_imb = bid_qty / (top_sum_qty + 1e-9)

    # f_5 OFI L1 60s proxy. Following Cont-Stoikov OFI's per-event rule
    # adapted to snapshot data:
    #   If bid_price went up: add full new bid_qty.
    #   If bid_price unchanged: add delta(bid_qty).
    #   If bid_price went down: subtract old bid_qty (zeroed out level).
    # Symmetric for ask side (with sign flipped).
    # We approximate the unchanged-case Δqty as the typical 60s flow.
    bid_p_prev = bid_p.shift(1)
    ask_p_prev = ask_p.shift(1)
    bid_qty_prev = bid_qty.shift(1)
    ask_qty_prev = ask_qty.shift(1)

    delta_bid = (
        pl.when(bid_p > bid_p_prev).then(bid_qty)
          .when(bid_p < bid_p_prev).then(-bid_qty_prev)
          .otherwise(bid_qty - bid_qty_prev)
    )
    delta_ask = (
        pl.when(ask_p < ask_p_prev).then(ask_qty)
          .when(ask_p > ask_p_prev).then(-ask_qty_prev)
          .otherwise(ask_qty - ask_qty_prev)
    )
    ofi = (delta_bid - delta_ask) / (top_sum_qty + 1e-9)

    # f_6, f_7 cumulative-depth log proxies.
    log_buy = (pl.col("total_buy_qty") + 1.0).log()
    log_sell = (pl.col("total_sell_qty") + 1.0).log()

    out = df.select([
        pl.col("ts"),
        microprice.alias("f_0"),
        mid_ret.alias("f_1"),
        spread_bps.alias("f_2"),
        book_pressure.alias("f_3"),
        queue_imb.alias("f_4"),
        ofi.alias("f_5"),
        log_buy.alias("f_6"),
        log_sell.alias("f_7"),
    ])

    # NaN/inf handling: replace inf with NaN, then forward-fill, then 0-fill
    # for the head row.
    out = out.with_columns([
        pl.col(f"f_{i}").cast(pl.Float32) for i in range(OB_N_CHANNELS)
    ])
    out = out.with_columns([
        pl.when(pl.col(f"f_{i}").is_infinite()).then(None).otherwise(pl.col(f"f_{i}"))
            .forward_fill()
            .fill_null(0.0)
            .alias(f"f_{i}")
        for i in range(OB_N_CHANNELS)
    ])

    if drop_warmup:
        out = out.slice(1, out.height - 1) if out.height > 1 else out
    return out


def load_ob_features_for_symbol(
    *, ob_root: Path, symbol: str,
) -> pl.DataFrame:
    """Load and engineer OB features from `{ob_root}/{symbol}.parquet`.

    Returns an empty frame if no parquet exists (symbol absent from OB
    universe — common case, 2-ish symbols out of 205 lack OB).
    """
    path = ob_root / f"{symbol}.parquet"
    if not path.exists():
        return pl.DataFrame(
            {"ts": [], **{f"f_{i}": [] for i in range(OB_N_CHANNELS)}}
        )
    df = pl.read_parquet(path)
    return compute_ob_features(df)


def lookup_snapshot_window(
    *,
    feature_ts: np.ndarray,        # datetime64[us], shape (N,)
    feature_mat: np.ndarray,       # float32, shape (N, OB_N_CHANNELS)
    anchor_ts: np.datetime64,
    k_snapshots: int = 64,
    max_staleness_us: int = 5 * 60 * 1_000_000,  # 5 min default
) -> np.ndarray | None:
    """Return the K most-recent snapshots strictly before `anchor_ts`.

    Returns `None` if:
    - feature_ts is empty (no OB for this symbol)
    - latest available snapshot is older than `max_staleness_us` from `anchor_ts`
    - fewer than `k_snapshots` snapshots available before anchor_ts

    Otherwise: shape `(k_snapshots, OB_N_CHANNELS)` float32 with snapshots
    in chronological order (oldest first).
    """
    if feature_ts.size == 0:
        return None
    pos = int(np.searchsorted(feature_ts, anchor_ts, side="left"))
    if pos < k_snapshots:
        return None  # warmup gap
    most_recent = feature_ts[pos - 1]
    staleness = (anchor_ts - most_recent).astype("int64")  # microseconds
    if staleness > max_staleness_us:
        return None
    window = feature_mat[pos - k_snapshots:pos]
    return window
