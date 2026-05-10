"""SSL heads for the Phase 2 pretraining backbone (DESIGN §3.2).

Four objectives, weighted sum (initial weights 1.0 / 0.5 / 0.3 / 0.5):

  1. **Masked patch reconstruction** (primary). A fraction of patches in
     each scale's stream is replaced with a `[MASK]` embedding before the
     backbone sees them. The reconstruction head projects the post-
     backbone hidden states back to `(C × patch_len)` and we minimize a
     volume-weighted MSE against the original RevIN-normalized values.

  2. **Next-patch prediction** (causal). A separate head reads the
     backbone's *causal* pass output and predicts the next patch from
     the prefix. Same `(C × patch_len)` output shape; loss is plain
     unweighted MSE on RevIN-normalized values.

  3. **Cross-sectional contrastive** (InfoNCE). The backbone produces a
     FORECAST token per sample (per-(symbol, end_ts) pair); the
     contrastive head projects it to a unit-norm embedding, and InfoNCE
     pulls together same-end_ts pairs from co-moving symbols, pushes
     apart unrelated pairs.

  4. **Multi-horizon forecast on FORECAST token**. Reads the *non-causal*
     pass's hidden at position 0 (which has attended to the full input
     window) and predicts the next H non-overlapping patches at a single
     forecast scale. Trains the FORECAST token to be a useful summary
     for downstream forecasting heads (Phase 3+).

This module ships:
  - `MaskedReconstructionHead` — `(B, P, d_model)` → `(B, P, C * patch_len)`.
  - `NextPatchHead` — same shape signature, used with the causal output.
  - `ContrastiveHead` — `(B, d_model)` → `(B, proj_dim)` ℓ2-normalized.
  - `MultiHorizonForecastHead` — `(B, d_model)` → `(B, H, C, patch_len)`.

The actual masking + loss code lives in `daity.training.losses` so the
heads stay focused on architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedReconstructionHead(nn.Module):
    """Predict the original `(C, patch_len)` patch values from a hidden state.

    Two-layer MLP with GELU. Input: `(B, P, d_model)`. Output: `(B, P, C * patch_len)`.
    Caller reshapes to `(B, P, C, patch_len)` if needed.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        patch_len: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if num_channels <= 0 or patch_len <= 0:
            msg = f"num_channels and patch_len must be positive ({num_channels}, {patch_len})"
            raise ValueError(msg)
        self.num_channels = num_channels
        self.patch_len = patch_len
        out_dim = num_channels * patch_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`(B, P, d_model)` → `(B, P, C * patch_len)`."""
        return self.net(hidden)


class NextPatchHead(nn.Module):
    """Predict the next patch from the causal-prefix hidden state.

    Architecturally identical to `MaskedReconstructionHead`; kept as a
    separate class so the two losses' weight init / fine-tuning don't
    interfere, and so DESIGN's "separate head" intent is explicit in code.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        patch_len: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_len = patch_len
        out_dim = num_channels * patch_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`(B, P, d_model)` → `(B, P, C * patch_len)`. Caller is responsible
        for slicing to `[:, :-1]` (the last position predicts past the end)
        before computing loss against `[:, 1:]` of the targets."""
        return self.net(hidden)


class ContrastiveHead(nn.Module):
    """Project the FORECAST hidden state to a contrastive embedding.

    Two-layer MLP + ℓ2 normalization on output. The InfoNCE loss expects
    unit-norm vectors so cosine similarity == dot product.
    """

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if proj_dim <= 0:
            msg = f"proj_dim must be positive, got {proj_dim}"
            raise ValueError(msg)
        self.proj_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, proj_dim),
        )

    def forward(self, forecast_hidden: torch.Tensor) -> torch.Tensor:
        """`(B, d_model)` → `(B, proj_dim)` with unit ℓ2 norm per row."""
        if forecast_hidden.dim() != 2:
            msg = f"Expected (B, d_model), got shape {tuple(forecast_hidden.shape)}"
            raise ValueError(msg)
        z = self.net(forecast_hidden)
        return F.normalize(z, p=2, dim=-1)


class MultiHorizonForecastHead(nn.Module):
    """Predict the next `n_patches` non-overlapping patches from the FORECAST token.

    Reads the non-causal-pass hidden at position 0 (which has attended to
    the full input window) and emits a bundle of `n_patches` future
    patches at a single forecast scale. Output shape is
    `(B, n_patches, num_channels, patch_len)`.

    Two-layer MLP with a single direct projection to the full bundle —
    deliberately not autoregressive. Multi-step direct heads outperform
    rollout-style autoregressive forecasts on patch-transformers per the
    PatchTST / Chronos / MOIRAI literature. The head is small enough
    that we can scale n_patches without bloating the param count
    (one Linear from d_model → n_patches·C·patch_len).
    """

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        num_channels: int,
        patch_len: int,
        *,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()
        if n_patches <= 0:
            msg = f"n_patches must be positive, got {n_patches}"
            raise ValueError(msg)
        if num_channels <= 0 or patch_len <= 0:
            msg = (
                f"num_channels and patch_len must be positive "
                f"({num_channels}, {patch_len})"
            )
            raise ValueError(msg)
        self.n_patches = n_patches
        self.num_channels = num_channels
        self.patch_len = patch_len
        out_dim = n_patches * num_channels * patch_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Linear(hidden_ratio * d_model, out_dim),
        )

    def forward(self, forecast_hidden: torch.Tensor) -> torch.Tensor:
        """`(B, d_model)` → `(B, n_patches, num_channels, patch_len)`."""
        if forecast_hidden.dim() != 2:
            msg = f"Expected (B, d_model), got shape {tuple(forecast_hidden.shape)}"
            raise ValueError(msg)
        B = forecast_hidden.size(0)
        flat = self.net(forecast_hidden)                 # (B, n_patches * C * patch_len)
        return flat.view(B, self.n_patches, self.num_channels, self.patch_len)
