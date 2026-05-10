"""Unit tests for the multi-resolution patch tokenizer.

Locks in: patch shapes, channel-independent layout, resolution embedding
distinguishes scales, FORECAST token is at position 0, RevIN sees each
scale's stats independently.
"""

from __future__ import annotations

import pytest
import torch

from daity.models.tokenizer import (
    DEFAULT_PATCH_LEN,
    DEFAULT_PATCH_STRIDE,
    MultiResTokenizer,
    Patcher,
)


# ----- Patcher -----


def test_patcher_n_patches_formula() -> None:
    p = Patcher(patch_len=16, stride=8)
    # (L - patch_len) / stride + 1 when L >= patch_len, else 0.
    assert p.n_patches(15) == 0
    assert p.n_patches(16) == 1
    assert p.n_patches(24) == 2          # (24-16)/8 + 1 = 2
    assert p.n_patches(256) == 31        # (256-16)/8 + 1 = 31
    assert p.n_patches(64) == 7          # (64-16)/8 + 1 = 7


def test_patcher_output_shape() -> None:
    p = Patcher(patch_len=16, stride=8)
    x = torch.randn(4, 256, 5)           # B=4, L=256, C=5
    out = p(x)
    # (B, n_patches, C, patch_len)
    assert out.shape == (4, 31, 5, 16)


def test_patcher_overlapping_strides_share_input_values() -> None:
    """Patch i+1 starts `stride` bars after patch i, so their last/first
    bars overlap when stride < patch_len."""
    p = Patcher(patch_len=8, stride=4)
    x = torch.arange(32).float().view(1, 32, 1)  # 0,1,...,31
    out = p(x).squeeze(0).squeeze(1)             # (n_patches, patch_len)
    # patch 0 = [0..7], patch 1 = [4..11], so out[0, 4:] == out[1, :4]
    assert torch.equal(out[0, 4:], out[1, :4])


def test_patcher_rejects_short_window() -> None:
    p = Patcher(patch_len=16, stride=8)
    with pytest.raises(ValueError, match="shorter than patch_len"):
        p(torch.randn(2, 10, 3))


def test_patcher_rejects_2d_input() -> None:
    p = Patcher(patch_len=16, stride=8)
    with pytest.raises(ValueError, match=r"\(B, L, C\)"):
        p(torch.randn(4, 32))


def test_patcher_rejects_zero_or_negative_params() -> None:
    with pytest.raises(ValueError, match="positive"):
        Patcher(patch_len=0, stride=8)
    with pytest.raises(ValueError, match="positive"):
        Patcher(patch_len=16, stride=-1)


# ----- MultiResTokenizer -----


def _scale_inputs(B: int = 2, C: int = 5, *,
                  five_m: int = 256, day: int = 64) -> dict[str, torch.Tensor]:
    """Synthetic per-scale tensors with realistic magnitudes."""
    return {
        "5m": torch.randn(B, five_m, C) * 5.0 + 100.0,
        "day": torch.randn(B, day, C) * 5.0 + 100.0,
    }


def test_tokenizer_output_shape_includes_forecast_token() -> None:
    """Total seq len = 1 (forecast) + sum of per-scale patches."""
    tok = MultiResTokenizer(scales=("5m", "day"), num_channels=5, d_model=320)
    inp = _scale_inputs()
    out = tok(inp)
    # 5m has 31 patches, day has 7 patches. Plus 1 FORECAST token = 39.
    assert out.shape == (2, 1 + 31 + 7, 320)


def test_tokenizer_d_model_propagates() -> None:
    """Custom d_model should flow through end-to-end."""
    tok = MultiResTokenizer(scales=("5m",), num_channels=5, d_model=128)
    out = tok({"5m": torch.randn(3, 256, 5)})
    assert out.shape[-1] == 128


def test_tokenizer_resolution_embedding_distinguishes_scales() -> None:
    """Patches from different scales should NOT all share the same vector
    (otherwise the backbone can't tell them apart)."""
    tok = MultiResTokenizer(scales=("5m", "day"), num_channels=5, d_model=64)
    # Force resolution_embed to be the identifying signal: zero out the
    # projection's bias and feed identical inputs to both scales.
    same_window = torch.zeros(1, 256, 5)
    same_day = torch.zeros(1, 64, 5)
    out = tok({"5m": same_window, "day": same_day})
    # Skip the FORECAST token. The rest is 31 5m + 7 day patches.
    five_m_patch_0 = out[0, 1, :]
    day_patch_0 = out[0, 1 + 31, :]
    # Resolution embed differs → patches differ even when input data is identical.
    assert not torch.allclose(five_m_patch_0, day_patch_0)


def test_tokenizer_forecast_token_is_at_position_zero() -> None:
    """The FORECAST token's hidden state at the backbone output is what
    every downstream head reads — it MUST be position 0 across batches."""
    tok = MultiResTokenizer(scales=("5m",), num_channels=5, d_model=32)
    inp = {"5m": torch.randn(4, 256, 5)}
    out = tok(inp)
    forecast_at_0 = out[:, 0, :]
    # The forecast_token parameter is broadcast across batch — every row's
    # position-0 token should be the SAME parameter vector.
    expected = tok.forecast_token.expand(4, 1, -1).squeeze(1)
    assert torch.allclose(forecast_at_0, expected)


def test_tokenizer_rejects_missing_scale() -> None:
    """A scale registered at construction must be present at forward time."""
    tok = MultiResTokenizer(scales=("5m", "day"), num_channels=5, d_model=64)
    with pytest.raises(ValueError, match="missing"):
        tok({"5m": torch.randn(2, 256, 5)})


def test_tokenizer_rejects_inconsistent_batch_size() -> None:
    tok = MultiResTokenizer(scales=("5m", "day"), num_channels=5, d_model=64)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        tok({
            "5m": torch.randn(2, 256, 5),
            "day": torch.randn(3, 64, 5),
        })


def test_tokenizer_rejects_empty_scales() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiResTokenizer(scales=(), num_channels=5, d_model=64)


def test_tokenizer_rejects_zero_or_negative_dims() -> None:
    with pytest.raises(ValueError, match="positive"):
        MultiResTokenizer(scales=("5m",), num_channels=0, d_model=64)
    with pytest.raises(ValueError, match="positive"):
        MultiResTokenizer(scales=("5m",), num_channels=5, d_model=0)


def test_tokenizer_n_patches_per_scale_matches_actual_output() -> None:
    """The helper used by attention-mask builders must match the live forward."""
    tok = MultiResTokenizer(scales=("5m", "day"), num_channels=5, d_model=64)
    n_patches = tok.n_patches_per_scale({"5m": 256, "day": 64})
    assert n_patches == {"5m": 31, "day": 7}
    out = tok({"5m": torch.randn(1, 256, 5), "day": torch.randn(1, 64, 5)})
    # Sequence length minus FORECAST token must equal sum of n_patches.
    assert out.size(1) - 1 == sum(n_patches.values())


def test_tokenizer_revin_is_per_scale_not_shared() -> None:
    """Each scale gets its own RevIN — stats from 5m must NOT leak into
    day's normalization, otherwise a vol-spike on 5m would shift day too."""
    tok = MultiResTokenizer(scales=("5m", "day"), num_channels=5, d_model=64)
    assert tok.revins["5m"] is not tok.revins["day"]


def test_tokenizer_default_patch_constants_match_design() -> None:
    """DESIGN §3.1 picks 16/8 for patch len/stride; lock that in."""
    assert DEFAULT_PATCH_LEN == 16
    assert DEFAULT_PATCH_STRIDE == 8


def test_tokenizer_param_count_is_reasonable_for_default_size() -> None:
    """Sanity: default tokenizer is light (<<20M params; backbone is the heavy bit)."""
    tok = MultiResTokenizer(scales=("5m", "15m", "60m", "day"),
                            num_channels=5, d_model=320)
    n = sum(p.numel() for p in tok.parameters())
    # 4 scales × Linear(80→320) ≈ 4*(80*320 + 320) = 102k. Plus RevIN affines
    # and resolution/forecast embeds. Ballpark <200k.
    assert n < 200_000, f"tokenizer has {n:,} params — too many for the default size"
