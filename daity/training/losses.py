"""Loss functions + masking helpers for Phase 2 SSL pretraining (DESIGN §3.2).

The three SSL objectives are wired through three functions here:

  - `apply_random_patch_mask(tokens, mask_ratio, mask_token)` — masks a
    fraction of patch tokens by replacing their embeddings with a learned
    `mask_token` vector. Returns `(masked_tokens, mask)` where `mask`
    is a boolean per-patch indicator of which positions were replaced
    (the loss only scores those positions).

  - `volume_weighted_mse(pred, target, volume, mask, low_q, high_q)` —
    masked-reconstruction loss, downweighting bars whose volume is below
    the 20th percentile (×0.25) and upweighting bars in the top decile
    (×1.5) per DESIGN §3.2. Volume is the per-bar volume from the OHLCV
    input — passed in by the training step which has the original tensors.

  - `next_patch_mse(pred, target, mask)` — plain unweighted MSE on the
    causal NTP head. `mask` is the "predict-the-next-patch" alignment
    mask: usually `mask[:, :-1] = True, mask[:, -1] = False`.

  - `info_nce_loss(z_a, z_b, temperature)` — symmetric InfoNCE between two
    batches of unit-norm projections. `z_a[i]` and `z_b[i]` are positives;
    everything else in the batch is a negative. Temperature 0.07 is the
    common SimCLR default.

Volume weighting note. DESIGN §3.2 says "bars with volume below the 20th
percentile contribute 0.25× weight, bars in the top decile contribute
1.5×". We compute these percentiles per-batch over the volume tensor;
batched percentiles are stable on float32 batches of any size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MASK_RATIO = 0.4   # DESIGN §3.2: 30-40% of patches masked
DEFAULT_LOW_Q = 0.20       # bottom 20% of volume → 0.25× weight
DEFAULT_HIGH_Q = 0.90      # top 10% of volume → 1.5× weight
DEFAULT_LOW_W = 0.25
DEFAULT_HIGH_W = 1.5
DEFAULT_BASE_W = 1.0


def apply_random_patch_mask(
    tokens: torch.Tensor,
    *,
    mask_ratio: float,
    mask_token: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace `mask_ratio` of patch positions with `mask_token`.

    `tokens` is the post-tokenizer sequence `(B, T, d_model)`. We mask
    positions `[1:T]` only — position 0 is the FORECAST token and must
    never be masked (the contrastive head reads it).

    Returns `(masked_tokens, mask)` where `mask` is `(B, T)` boolean,
    True at positions that were masked (loss target).
    """
    if not 0.0 < mask_ratio < 1.0:
        msg = f"mask_ratio must be in (0, 1), got {mask_ratio}"
        raise ValueError(msg)
    B, T, D = tokens.shape
    if mask_token.shape != (D,):
        msg = (
            f"mask_token must have shape ({D},) matching d_model; "
            f"got {tuple(mask_token.shape)}"
        )
        raise ValueError(msg)

    # Mask candidates: positions [1:T] (skip FORECAST at 0).
    n_patches = T - 1
    n_to_mask = max(1, int(n_patches * mask_ratio))
    # Sample without replacement per batch row.
    rand = torch.rand(B, n_patches, generator=generator, device=tokens.device)
    # Top-k smallest random values → positions to mask.
    _, idx = rand.topk(n_to_mask, dim=-1, largest=False)         # (B, n_to_mask)
    mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, idx + 1, True)                              # +1 for FORECAST offset

    # Replace masked positions with the mask token (broadcast over B).
    masked = torch.where(
        mask.unsqueeze(-1),
        mask_token.view(1, 1, D).expand_as(tokens),
        tokens,
    )
    return masked, mask


def volume_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    volume: torch.Tensor,
    mask: torch.Tensor,
    low_q: float = DEFAULT_LOW_Q,
    high_q: float = DEFAULT_HIGH_Q,
    low_w: float = DEFAULT_LOW_W,
    high_w: float = DEFAULT_HIGH_W,
    base_w: float = DEFAULT_BASE_W,
) -> torch.Tensor:
    """Weighted-MSE loss on masked positions.

    Shapes:
      pred   : (B, P, C, patch_len)
      target : (B, P, C, patch_len)
      volume : (B, P)             — per-patch volume (mean over patch_len)
      mask   : (B, P)             — True = include in loss (the masked positions)

    Returns a scalar loss (mean over masked elements, weighted).
    """
    if pred.shape != target.shape:
        msg = f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must match"
        raise ValueError(msg)
    if volume.shape != mask.shape:
        msg = (
            f"volume {tuple(volume.shape)} and mask {tuple(mask.shape)} must match"
        )
        raise ValueError(msg)
    # Per-element squared error, then average over (C, patch_len) per patch.
    per_patch_se = (pred - target).pow(2).mean(dim=(-1, -2))     # (B, P)
    # Bucket weights by per-patch volume (within the masked positions only).
    masked_volume = volume[mask]
    if masked_volume.numel() == 0:
        return per_patch_se.new_zeros(())
    low_thresh = torch.quantile(masked_volume, low_q)
    high_thresh = torch.quantile(masked_volume, high_q)
    w = torch.full_like(volume, base_w)
    w = torch.where(volume <= low_thresh, torch.full_like(w, low_w), w)
    w = torch.where(volume >= high_thresh, torch.full_like(w, high_w), w)
    weighted = (per_patch_se * w * mask).sum()
    norm = (w * mask).sum().clamp_min(1.0)
    return weighted / norm


def next_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Plain (unweighted) MSE for the next-patch-prediction head.

    Shapes:
      pred   : (B, P, C, patch_len)  — predictions at every position
      target : (B, P, C, patch_len)
      mask   : (B, P)                — True at positions to include
    """
    per_patch_se = (pred - target).pow(2).mean(dim=(-1, -2))     # (B, P)
    n = mask.sum().clamp_min(1.0)
    return (per_patch_se * mask).sum() / n


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between two batches of unit-norm projections.

    `z_a[i]` and `z_b[i]` are a positive pair; all other pairs in the
    batch are negatives. Returns a scalar — the average of the two
    directional NT-Xent losses.

    Both tensors must be ℓ2-normalized (the contrastive head ensures
    this); we re-normalize defensively in case a caller skips it.
    """
    if z_a.shape != z_b.shape or z_a.dim() != 2:
        msg = (
            f"z_a and z_b must be (B, D) and same shape; "
            f"got {tuple(z_a.shape)} vs {tuple(z_b.shape)}"
        )
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    B = z_a.size(0)
    if B < 2:
        msg = "InfoNCE needs batch size >= 2 for at least one negative pair"
        raise ValueError(msg)
    z_a = F.normalize(z_a, p=2, dim=-1)
    z_b = F.normalize(z_b, p=2, dim=-1)
    sim = (z_a @ z_b.T) / temperature                            # (B, B)
    targets = torch.arange(B, device=z_a.device)
    loss_a = F.cross_entropy(sim, targets)
    loss_b = F.cross_entropy(sim.T, targets)
    return 0.5 * (loss_a + loss_b)
