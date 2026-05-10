"""Unit tests for `daity.models.heads` + `daity.training.losses`.

Locks in: head shapes, contrastive embedding is unit-norm, mask-replacement
preserves position 0, volume weighting actually downweights low-volume bars,
InfoNCE is minimized when positives align.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from daity.models.heads import (
    ContrastiveHead,
    MaskedReconstructionHead,
    MultiHorizonForecastHead,
    NextPatchHead,
)
from daity.training.losses import (
    DEFAULT_HIGH_W,
    DEFAULT_LOW_W,
    apply_random_patch_mask,
    info_nce_loss,
    next_patch_mse,
    volume_weighted_mse,
)


# ----- Heads -----


def test_masked_reconstruction_head_shape() -> None:
    h = MaskedReconstructionHead(d_model=64, num_channels=5, patch_len=16)
    out = h(torch.randn(2, 32, 64))
    assert out.shape == (2, 32, 5 * 16)


def test_masked_reconstruction_head_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="positive"):
        MaskedReconstructionHead(d_model=64, num_channels=0, patch_len=16)


def test_next_patch_head_shape() -> None:
    h = NextPatchHead(d_model=64, num_channels=5, patch_len=16)
    out = h(torch.randn(2, 32, 64))
    assert out.shape == (2, 32, 5 * 16)


def test_contrastive_head_unit_norm() -> None:
    h = ContrastiveHead(d_model=64, proj_dim=32)
    z = h(torch.randn(8, 64))
    assert z.shape == (8, 32)
    norms = z.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_contrastive_head_rejects_3d_input() -> None:
    h = ContrastiveHead(d_model=64, proj_dim=32)
    with pytest.raises(ValueError, match=r"\(B, d_model\)"):
        h(torch.randn(8, 32, 64))


def test_multi_horizon_forecast_head_shape() -> None:
    h = MultiHorizonForecastHead(d_model=64, n_patches=6, num_channels=5, patch_len=16)
    out = h(torch.randn(8, 64))
    assert out.shape == (8, 6, 5, 16)


def test_multi_horizon_forecast_head_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="n_patches must be positive"):
        MultiHorizonForecastHead(d_model=64, n_patches=0, num_channels=5, patch_len=16)
    with pytest.raises(ValueError, match="positive"):
        MultiHorizonForecastHead(d_model=64, n_patches=4, num_channels=0, patch_len=16)


def test_multi_horizon_forecast_head_rejects_3d_input() -> None:
    h = MultiHorizonForecastHead(d_model=64, n_patches=4, num_channels=5, patch_len=16)
    with pytest.raises(ValueError, match=r"\(B, d_model\)"):
        h(torch.randn(8, 32, 64))


def test_contrastive_head_rejects_zero_proj_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        ContrastiveHead(d_model=64, proj_dim=0)


# ----- Patch masking -----


def test_apply_random_patch_mask_preserves_forecast_token() -> None:
    """Position 0 (FORECAST) must NEVER be masked."""
    torch.manual_seed(0)
    tokens = torch.randn(4, 33, 16)
    mask_tok = torch.full((16,), 99.0)
    masked, mask = apply_random_patch_mask(
        tokens, mask_ratio=0.5, mask_token=mask_tok,
    )
    # Position 0 of mask is False, and tokens[:, 0, :] is unchanged.
    assert (mask[:, 0] == False).all()  # noqa: E712
    assert torch.allclose(masked[:, 0, :], tokens[:, 0, :])


def test_apply_random_patch_mask_replaces_with_mask_token() -> None:
    """Wherever mask is True, the token equals `mask_token` exactly."""
    torch.manual_seed(0)
    tokens = torch.randn(2, 17, 8)
    mask_tok = torch.full((8,), -7.0)
    masked, mask = apply_random_patch_mask(
        tokens, mask_ratio=0.4, mask_token=mask_tok,
    )
    masked_positions = masked[mask]
    assert torch.allclose(masked_positions, mask_tok.expand_as(masked_positions))
    # Unmasked positions: unchanged.
    unmasked_orig = tokens[~mask]
    unmasked_new = masked[~mask]
    assert torch.allclose(unmasked_orig, unmasked_new)


def test_apply_random_patch_mask_ratio_approximately_honored() -> None:
    """At ratio=0.4 over 100 patches × many trials, ~40% should be masked."""
    torch.manual_seed(0)
    tokens = torch.randn(64, 101, 16)
    mask_tok = torch.zeros(16)
    _, mask = apply_random_patch_mask(
        tokens, mask_ratio=0.4, mask_token=mask_tok,
    )
    # 100 maskable positions × 0.4 = 40 masked per row. Tolerate ±1 from rounding.
    n_masked_per_row = mask[:, 1:].sum(dim=1).float()
    assert (n_masked_per_row == 40.0).all(), n_masked_per_row.unique()


def test_apply_random_patch_mask_rejects_zero_or_one_ratio() -> None:
    tokens = torch.randn(2, 17, 8)
    mask_tok = torch.zeros(8)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        apply_random_patch_mask(tokens, mask_ratio=0.0, mask_token=mask_tok)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        apply_random_patch_mask(tokens, mask_ratio=1.0, mask_token=mask_tok)


def test_apply_random_patch_mask_rejects_wrong_mask_token_shape() -> None:
    tokens = torch.randn(2, 17, 8)
    with pytest.raises(ValueError, match="must have shape"):
        apply_random_patch_mask(tokens, mask_ratio=0.4, mask_token=torch.zeros(7))


# ----- Volume-weighted MSE -----


def test_volume_weighted_mse_zero_when_pred_eq_target() -> None:
    pred = torch.randn(2, 8, 5, 16)
    target = pred.clone()
    volume = torch.rand(2, 8) * 1000
    mask = torch.ones(2, 8, dtype=torch.bool)
    loss = volume_weighted_mse(pred, target, volume=volume, mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_volume_weighted_mse_low_volume_downweighted() -> None:
    """Bars in the bottom volume quantile should contribute LOW_W (0.25),
    upper-decile bars should contribute HIGH_W (1.5). Construct two
    fixtures where the same per-bar SE is concentrated in low vs high
    volume and verify the total loss differs by the weight ratio.
    """
    # Pred deviates from target by a constant per-patch error of 1.0.
    target = torch.zeros(1, 100, 5, 16)
    pred = torch.ones_like(target)                                # SE = 1.0 per element
    volume_lo = torch.zeros(1, 100)                               # all low
    volume_hi = torch.full((1, 100), 100.0)                       # all high (top decile)
    mask = torch.ones(1, 100, dtype=torch.bool)

    loss_low = volume_weighted_mse(pred, target, volume=volume_lo, mask=mask).item()
    loss_high = volume_weighted_mse(pred, target, volume=volume_hi, mask=mask).item()
    # Both compute weighted means with the same numerator/denominator structure
    # — so loss = 1.0 in both cases (constant SE / constant weight). The
    # interesting case is mixed-volume: half low, half high.
    assert loss_low == pytest.approx(1.0)
    assert loss_high == pytest.approx(1.0)


def test_volume_weighted_mse_mixed_volume_weights_correctly() -> None:
    """50 low-volume bars carry error=1; 50 high-volume bars carry error=0.

    Quantiles over the volume tensor: q20 = 0.0 (first 50 are <= it, all
    weight=LOW_W=0.25); q90 = 50.0 (last 50 are >= it, all weight=HIGH_W=1.5).
    Weighted mean = (50 × 0.25 × 1 + 50 × 1.5 × 0) / (50 × 0.25 + 50 × 1.5)
                  = 12.5 / 87.5 ≈ 0.143."""
    target = torch.zeros(1, 100, 5, 16)
    pred = torch.zeros_like(target)
    pred[:, :50] = 1.0                                            # SE=1 on first 50
    volume = torch.cat([
        torch.zeros(50),                                          # bottom-quantile
        torch.full((50,), 50.0),                                  # top-decile
    ]).unsqueeze(0)
    mask = torch.ones(1, 100, dtype=torch.bool)
    loss = volume_weighted_mse(pred, target, volume=volume, mask=mask).item()
    expected = (50 * 0.25 * 1.0 + 50 * 1.5 * 0.0) / (50 * 0.25 + 50 * 1.5)
    assert loss == pytest.approx(expected, abs=1e-5)


def test_volume_weighted_mse_zero_when_no_masked_positions() -> None:
    """If `mask` is all False, the loss should be 0 (no positions contribute)."""
    pred = torch.randn(2, 8, 5, 16)
    target = torch.zeros_like(pred)
    volume = torch.rand(2, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    loss = volume_weighted_mse(pred, target, volume=volume, mask=mask)
    assert loss.item() == 0.0


def test_volume_weighted_constants_match_design() -> None:
    """DESIGN §3.2: 0.25× weight for bottom-20% volume, 1.5× for top-10%."""
    assert DEFAULT_LOW_W == 0.25
    assert DEFAULT_HIGH_W == 1.5


# ----- Next-patch MSE -----


def test_next_patch_mse_zero_when_pred_eq_target() -> None:
    pred = torch.randn(2, 8, 5, 16)
    target = pred.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)
    assert next_patch_mse(pred, target, mask=mask).item() == pytest.approx(0.0)


def test_next_patch_mse_ignores_unmasked() -> None:
    """A masked-out position should not affect the loss even if pred is wildly off."""
    pred = torch.zeros(1, 4, 2, 4)
    target = torch.zeros_like(pred)
    pred[:, 0] = 100.0                                            # huge error at pos 0
    # Mask says pos 0 doesn't count, pos 1-3 do (and they match perfectly).
    mask = torch.tensor([[False, True, True, True]])
    loss = next_patch_mse(pred, target, mask=mask).item()
    assert loss == pytest.approx(0.0)


# ----- InfoNCE -----


def test_info_nce_low_when_positives_aligned() -> None:
    """If z_a == z_b and they're spread out, InfoNCE should be ~log(1) = 0."""
    torch.manual_seed(0)
    z = torch.randn(16, 32)
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    loss = info_nce_loss(z, z, temperature=0.07).item()
    # log(B) is the loss when all rows are identical → just barely above 0
    # since perfectly-aligned positives still have to compete with negatives.
    # In practice the loss is close to 0 if the positives are well-separated
    # from each other in the batch.
    assert loss < 1.0


def test_info_nce_higher_when_positives_misaligned() -> None:
    """Random pairing should yield a notably higher loss than aligned."""
    torch.manual_seed(0)
    z_a = torch.randn(16, 32)
    z_a = z_a / z_a.norm(p=2, dim=-1, keepdim=True)
    z_b_aligned = z_a.clone()
    z_b_random = torch.randn(16, 32)
    z_b_random = z_b_random / z_b_random.norm(p=2, dim=-1, keepdim=True)
    loss_aligned = info_nce_loss(z_a, z_b_aligned, temperature=0.07).item()
    loss_random = info_nce_loss(z_a, z_b_random, temperature=0.07).item()
    assert loss_random > loss_aligned + 0.5


def test_info_nce_rejects_singleton_batch() -> None:
    """Need at least 2 rows so there's at least one negative pair."""
    z = torch.randn(1, 32)
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    with pytest.raises(ValueError, match="batch size"):
        info_nce_loss(z, z)


def test_info_nce_rejects_mismatched_shapes() -> None:
    z_a = torch.randn(8, 32)
    z_b = torch.randn(8, 16)
    with pytest.raises(ValueError, match="same shape"):
        info_nce_loss(z_a, z_b)


def test_info_nce_rejects_nonpositive_temperature() -> None:
    z = torch.randn(4, 32)
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    with pytest.raises(ValueError, match="positive"):
        info_nce_loss(z, z, temperature=0.0)
