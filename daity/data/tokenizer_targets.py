"""Build per-patch reconstruction targets + per-patch volumes.

The masked-reconstruction head predicts the **RevIN-normalized** patch
values (DESIGN §3.2: "MSE on RevIN-normalized values"), so the targets
must be normalized using the same per-(B, scale, channel) stats the
tokenizer applies to the inputs. This module separates target
construction from the LightningModule for testability.

Why a separate module: the tokenizer's RevIN layers cache stats on the
forward path. If we recompute stats from scratch here, they could drift
slightly (different RNG, different batches). Instead we **clone** the
tokenizer's tensors before its forward pass, normalize them with a
fresh-but-identical RevIN, and patch the result.

Returns:
  targets: (B, total_patches, C, patch_len) — the normalized patches the
           reconstruction head must predict.
  volumes: (B, total_patches) — mean per-patch volume in the *raw* (pre-
           RevIN) space. Volume weighting in the loss reads from this.
"""

from __future__ import annotations

import torch

from daity.models.tokenizer import MultiResTokenizer, Patcher


def mask_token_dim(tokenizer: MultiResTokenizer) -> int:
    """The d_model dimension a learnable [MASK] token must match."""
    return tokenizer.d_model


def build_targets(
    *,
    batch: dict[str, torch.Tensor],
    scales: tuple[str, ...],
    patch_len: int,
    patch_stride: int,
    tokenizer: MultiResTokenizer,
    volume_channel_index: int = 4,  # OHLCV: open=0, high=1, low=2, close=3, volume=4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct masked-recon targets in RevIN-normalized space + per-patch volumes.

    Args:
      batch: dict[scale, (B, L_scale, C)] from the dataset.
      scales: tuple of scale names matching `tokenizer.scales`.
      patch_len, patch_stride: must match the tokenizer's `Patcher`.
      tokenizer: source of the per-scale RevIN layers + scale ordering.

    Returns:
      targets: (B, total_patches, C, patch_len) where total_patches matches
               the post-tokenizer sequence (without the FORECAST token).
      volumes: (B, total_patches) — average volume per patch (raw, pre-RevIN).
    """
    patcher = Patcher(patch_len=patch_len, stride=patch_stride)
    target_chunks: list[torch.Tensor] = []
    volume_chunks: list[torch.Tensor] = []
    for sc in scales:
        x = batch[sc]                                                # (B, L, C)
        # Per-scale RevIN normalize. We CALL the tokenizer's RevIN here so
        # downstream cached stats match. mode='norm' refreshes the cache;
        # the tokenizer's own forward pass during training will refresh
        # again, ending up with the same stats since the input is the same.
        x_n = tokenizer.revins[sc](x, mode="norm")
        # Patch normalized targets.
        patches = patcher(x_n)                                       # (B, n, C, patch_len)
        target_chunks.append(patches)
        # Per-patch volume in the RAW input space (pre-RevIN).
        # patcher() works on (B, L, C); we want the mean of the volume
        # channel within each patch's window.
        vol_only = x[..., volume_channel_index : volume_channel_index + 1]  # (B, L, 1)
        vol_patches = patcher(vol_only).squeeze(-2).mean(dim=-1)     # (B, n)
        volume_chunks.append(vol_patches)
    targets = torch.cat(target_chunks, dim=1)                        # (B, total_n, C, patch_len)
    volumes = torch.cat(volume_chunks, dim=1)                        # (B, total_n)
    return targets, volumes


def build_forecast_targets(
    *,
    future_bars: torch.Tensor,
    forecast_scale: str,
    n_patches: int,
    patch_len: int,
    tokenizer: MultiResTokenizer,
) -> torch.Tensor:
    """Unfold future bars into N non-overlapping forecast patches, RevIN-normalized.

    The MultiHorizonForecastHead emits patches in the same RevIN frame as
    the input window's `forecast_scale` stream. To make the targets live
    in that same frame we MUST normalize with the **input window's**
    cached (mean, std) — never with stats computed from the future bars
    themselves (that'd leak label info into the target).

    Caller contract: `tokenizer.revins[forecast_scale]` must already have
    cached stats from the most-recent forward pass (i.e. `tokenizer(batch)`
    or `build_targets(...)` was called on the *input* window for THIS
    batch immediately before this function).

    Args:
      future_bars: (B, n_patches * patch_len, C) — raw future OHLCV bars
                   from the dataset, immediately following the input
                   window's end_ts at `forecast_scale`.
      forecast_scale: which scale these bars come from (e.g. "5m").
      n_patches: number of non-overlapping patches to emit.
      patch_len: bars per patch.
      tokenizer: source of the cached RevIN stats for `forecast_scale`.

    Returns:
      targets: (B, n_patches, C, patch_len) — non-overlapping future
               patches in the input window's RevIN frame.
    """
    if forecast_scale not in tokenizer.revins:
        msg = (
            f"forecast_scale {forecast_scale!r} not in tokenizer.revins "
            f"(have: {list(tokenizer.revins)})"
        )
        raise ValueError(msg)
    expected_len = n_patches * patch_len
    if future_bars.size(1) != expected_len:
        msg = (
            f"future_bars length {future_bars.size(1)} doesn't match "
            f"n_patches * patch_len = {expected_len}"
        )
        raise ValueError(msg)

    # Apply the SAME RevIN affine transform as the input window. The
    # tokenizer's RevIN caches (mean, std) on the input forward pass; we
    # reuse those stats by calling `mode='apply_cached'`. If the layer
    # doesn't expose that mode we fall back to manual normalization
    # using the cached attributes — see RevIN.forward().
    revin = tokenizer.revins[forecast_scale]
    x_n = revin(future_bars, mode="apply_cached")             # (B, T, C)

    # Non-overlapping unfold: stride == patch_len. We don't reuse Patcher
    # here since Patcher uses the (overlapping) patch_stride from the
    # tokenizer; for forecasts each patch is a fresh, distinct horizon.
    B, T, C = x_n.shape
    # (B, n_patches, C, patch_len)
    return x_n.view(B, n_patches, patch_len, C).permute(0, 1, 3, 2).contiguous()
