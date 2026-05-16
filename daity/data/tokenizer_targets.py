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


def build_multi_horizon_scalar_targets(
    *,
    future_bars: torch.Tensor,       # (B, F_bars, C) — raw future OHLCV (or log-return bars if input_form=log_returns)
    horizons_bars: tuple[int, ...],  # e.g. (3, 6, 9, 12, 18, 24, 36) for 15/30/45/60/90/120/180 min
    target_form: str = "log_returns",
    sqrt_time_normalize: bool = True,
    per_horizon_channel_std: torch.Tensor | None = None,  # NEW (Phase 2.5 Option C): (n_horizons, n_channels)
) -> torch.Tensor:
    """Phase 2.5 — multi-horizon scalar log-return targets.

    For each horizon h (in 5m bars), compute the cumulative log return
    between the anchor and the bar at index h (zero-indexed = h-th bar
    forward). Output: `(B, n_horizons, C)` where C is the number of
    forecast channels (typically 5: OHLCV).

    Args:
      future_bars: `(B, F, C)`. If `target_form="log_returns"`, these are
        per-bar log returns. If `target_form="absolute"`, these are raw
        prices/values and we compute log returns vs the anchor's last bar.
      horizons_bars: tuple of horizon indices in 5m-bar units. Must satisfy
        `max(horizons_bars) <= F`.
      target_form: "log_returns" (cumsum of per-bar log returns) or
        "absolute" (currently unsupported here — use 96-patch path).
      sqrt_time_normalize: if True, divide each horizon's target by
        sqrt(h_bars). Under random-walk assumptions this makes the variance
        of targets approximately constant across horizons → balanced
        gradient signal during multi-task training.

    Returns: `(B, n_horizons, C)` — multi-horizon log-return targets,
        optionally sqrt-time-normalized.
    """
    if target_form != "log_returns":
        msg = (
            f"build_multi_horizon_scalar_targets supports target_form="
            f"'log_returns' only; got {target_form!r}. Run with input_form="
            f"target_form='log_returns' in the pretrain config."
        )
        raise NotImplementedError(msg)
    if not horizons_bars:
        msg = "horizons_bars must be non-empty"
        raise ValueError(msg)
    if min(horizons_bars) <= 0:
        msg = f"horizon indices must be positive bars, got {horizons_bars}"
        raise ValueError(msg)
    F = future_bars.size(1)
    if max(horizons_bars) > F:
        msg = (
            f"max horizon {max(horizons_bars)} bars exceeds future window "
            f"length {F} bars"
        )
        raise ValueError(msg)

    # For each horizon h, target[b, c] = sum(future_bars[b, 0:h, c]).
    # Use cumulative sum along time axis, then index out.
    cum_log_returns = future_bars.cumsum(dim=1)                 # (B, F, C)
    # Index at h-1 (zero-indexed). Stack across horizons.
    # cum_log_returns[:, h-1, :] for each h ∈ horizons_bars
    h_indices = torch.tensor(
        [h - 1 for h in horizons_bars],
        device=future_bars.device, dtype=torch.long,
    )
    targets = cum_log_returns.index_select(dim=1, index=h_indices)  # (B, H, C)

    if per_horizon_channel_std is not None:
        # Phase 2.5 Option C: z-score normalize per (horizon, channel) using
        # precomputed std from training data. After this, each target has
        # ~unit variance → loss is comparable to recon/NTP at meaningful
        # scale. Supersedes sqrt-time-normalization.
        std = per_horizon_channel_std.to(targets.device, targets.dtype)
        if std.dim() == 1:
            std = std.unsqueeze(-1).expand(-1, targets.shape[-1])
        if std.shape != targets.shape[1:]:
            msg = (
                f"per_horizon_channel_std shape {tuple(std.shape)} doesn't match "
                f"targets[1:] shape {tuple(targets.shape[1:])}"
            )
            raise ValueError(msg)
        targets = targets / (std.unsqueeze(0) + 1e-9)
    elif sqrt_time_normalize:
        # Legacy fallback: divide each horizon by sqrt(h_bars) — Brownian
        # normalization. Wakes up to ~unit-target-variance only if per-bar
        # std happens to be 1, which it isn't for log returns (~0.002).
        # Use per_horizon_channel_std for proper scaling.
        scale = torch.tensor(
            [1.0 / float(h) ** 0.5 for h in horizons_bars],
            device=targets.device, dtype=targets.dtype,
        ).view(1, -1, 1)
        targets = targets * scale

    return targets
