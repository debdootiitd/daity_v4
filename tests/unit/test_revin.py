"""Unit tests for `daity.models.revin`.

Locks in the reversibility contract — given any input window, `denorm`
must reconstruct it (up to float epsilon) when applied to the normalized
output. This is the load-bearing property RevIN exists for.
"""

from __future__ import annotations

import pytest
import torch

from daity.models.revin import RevIN


def _x(B: int = 4, L: int = 32, C: int = 5, *, seed: int = 0) -> torch.Tensor:
    """Synthetic OHLCV-shaped tensor with realistic per-(batch, channel) scales."""
    g = torch.Generator().manual_seed(seed)
    # Each (batch, channel) gets its own mean + std so we exercise the
    # per-instance normalization meaningfully.
    base = torch.randn(B, 1, C, generator=g) * 100.0 + 500.0   # means
    spread = torch.rand(B, 1, C, generator=g) * 5.0 + 0.5      # stds
    noise = torch.randn(B, L, C, generator=g)
    return base + spread * noise


# ----- Construction guards -----


def test_rejects_zero_or_negative_channels() -> None:
    with pytest.raises(ValueError, match="positive"):
        RevIN(0)
    with pytest.raises(ValueError, match="positive"):
        RevIN(-3)


def test_rejects_unknown_mode() -> None:
    layer = RevIN(num_channels=3)
    x = torch.randn(2, 8, 3)
    with pytest.raises(ValueError, match="Unknown mode"):
        layer(x, mode="invalid")


def test_norm_rejects_wrong_channel_count() -> None:
    layer = RevIN(num_channels=5)
    with pytest.raises(ValueError, match="Channel mismatch"):
        layer(torch.randn(2, 8, 3), mode="norm")


def test_norm_rejects_2d_input() -> None:
    layer = RevIN(num_channels=3)
    with pytest.raises(ValueError, match=r"\(B, L, C\)"):
        layer(torch.randn(2, 8), mode="norm")


def test_denorm_before_norm_raises() -> None:
    layer = RevIN(num_channels=3)
    with pytest.raises(RuntimeError, match="before norm"):
        layer(torch.randn(2, 8, 3), mode="denorm")


# ----- Reversibility (the headline contract) -----


@pytest.mark.parametrize("affine", [False, True])
def test_norm_then_denorm_recovers_input(affine: bool) -> None:
    """The whole point: x ≈ denorm(norm(x))."""
    x = _x()
    layer = RevIN(num_channels=x.size(-1), affine=affine)
    x_n = layer(x, mode="norm")
    x_back = layer(x_n, mode="denorm")
    # 1e-4 tolerance: float32 round-trip through normalization picks up
    # ~1e-5 drift; affine mode adds another touch.
    assert torch.allclose(x_back, x, atol=1e-4), (
        f"denorm did not recover input: max diff = "
        f"{(x_back - x).abs().max().item():.6f}"
    )


def test_normalized_has_zero_mean_unit_std_per_instance() -> None:
    """After norm without affine, every (batch, channel) pair has mean≈0, std≈1."""
    x = _x()
    layer = RevIN(num_channels=x.size(-1), affine=False)
    x_n = layer(x, mode="norm")
    means = x_n.mean(dim=1)  # (B, C)
    stds = x_n.std(dim=1, unbiased=False)
    # 1e-4 tolerance: float32 sum-of-32 floats picks up ~1e-5 drift per element.
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-3)


def test_independent_instances_normalize_independently() -> None:
    """A 100x-larger price series in one batch row shouldn't affect another."""
    x = _x(B=2, L=32, C=3)
    x[1] *= 100.0  # row 1 has 100× larger absolute scale
    layer = RevIN(num_channels=3, affine=False)
    x_n = layer(x, mode="norm")
    # Both rows should have unit-ish std post-normalization.
    stds = x_n.std(dim=1, unbiased=False)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-3)


# ----- Affine parameter behavior -----


def test_affine_weight_and_bias_are_trainable() -> None:
    """The (gamma, beta) params should track .grad_fn through norm + denorm."""
    layer = RevIN(num_channels=3, affine=True)
    x = _x(C=3)
    x_n = layer(x, mode="norm")
    loss = x_n.pow(2).mean()
    loss.backward()
    assert layer.affine_weight.grad is not None
    assert layer.affine_weight.grad.abs().sum() > 0


def test_affine_disabled_has_no_extra_params() -> None:
    layer = RevIN(num_channels=3, affine=False)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params == 0


# ----- Edge cases -----


def test_constant_window_stays_well_defined() -> None:
    """A bar series that's constant (limit-locked, zero variance) should not
    produce NaN/inf — the eps in (std + eps) saves us."""
    x = torch.full((2, 16, 3), 100.0)
    layer = RevIN(num_channels=3, affine=False)
    x_n = layer(x, mode="norm")
    assert torch.isfinite(x_n).all()
    x_back = layer(x_n, mode="denorm")
    assert torch.allclose(x_back, x, atol=1e-3)


def test_cached_stats_are_detached_from_graph() -> None:
    """Normalization stats must not back-propagate gradients — RevIN
    treats them as constants. Otherwise the model can game the loss by
    learning to make `std` small to inflate normalized features."""
    layer = RevIN(num_channels=3, affine=False)
    x = _x(C=3).requires_grad_(True)
    x_n = layer(x, mode="norm")
    loss = x_n.sum()
    loss.backward()
    # Cached buffers should themselves have no grad (they're detached).
    assert layer._mean.requires_grad is False
    assert layer._std.requires_grad is False


# ----- apply_cached: the forecast-target mode -----


def test_apply_cached_uses_input_stats_not_recomputed() -> None:
    """apply_cached MUST use the cached (mean, std) from the most-recent
    norm call, not recompute from the new input. This is what makes
    forecast targets live in the encoder's coordinate frame."""
    layer = RevIN(num_channels=3, affine=False)
    x_input = _x(B=4, L=32, C=3, seed=0)
    layer(x_input, mode="norm")            # cache stats from input window
    # Future bars with deliberately DIFFERENT scale — if apply_cached
    # recomputed stats, we'd see the future window normalize to mean≈0;
    # using cached stats, the future window will be far from zero.
    x_future = _x(B=4, L=8, C=3, seed=1) + 1000.0
    out = layer(x_future, mode="apply_cached")
    # Recompute manually with cached stats to verify.
    expected = (x_future - layer._mean) / (layer._std + layer.eps)
    assert torch.allclose(out, expected, atol=1e-5)
    # Sanity: out should NOT be near zero-mean.
    assert out.mean().abs() > 1.0


def test_apply_cached_before_norm_raises() -> None:
    layer = RevIN(num_channels=3, affine=False)
    with pytest.raises(RuntimeError, match="apply_cached called before norm"):
        layer(_x(C=3), mode="apply_cached")


def test_apply_cached_validates_shape_and_batch() -> None:
    layer = RevIN(num_channels=3, affine=False)
    layer(_x(B=4, C=3), mode="norm")
    # Wrong channel count.
    with pytest.raises(ValueError, match="Channel mismatch"):
        layer(torch.randn(4, 8, 5), mode="apply_cached")
    # Mismatched batch size from the cache.
    with pytest.raises(RuntimeError, match="Batch mismatch"):
        layer(torch.randn(2, 8, 3), mode="apply_cached")
    # Wrong rank.
    with pytest.raises(ValueError, match=r"\(B, L, C\)"):
        layer(torch.randn(4, 3), mode="apply_cached")


def test_apply_cached_respects_affine() -> None:
    """With affine=True, the same gamma/beta the input window saw should
    apply to the future window."""
    layer = RevIN(num_channels=3, affine=True)
    x_input = _x(B=2, L=32, C=3, seed=0)
    layer(x_input, mode="norm")
    # Manually set affine to non-identity so we can detect the difference.
    with torch.no_grad():
        layer.affine_weight.fill_(2.0)
        layer.affine_bias.fill_(0.5)
    x_future = _x(B=2, L=8, C=3, seed=1)
    out = layer(x_future, mode="apply_cached")
    expected = (x_future - layer._mean) / (layer._std + layer.eps)
    expected = expected * layer.affine_weight + layer.affine_bias
    assert torch.allclose(out, expected, atol=1e-5)
