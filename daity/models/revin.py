"""Reversible Instance Normalization (RevIN) — Kim et al. 2022.

Why RevIN here (DESIGN §2.4): equity OHLCV is non-stationary across symbols
and across time. A small-cap microcap and Reliance live on completely
different price scales; vol regimes shift across years. RevIN normalizes
each `(batch, instance, channel)` window to zero mean and unit variance at
the **input** so the same backbone parameters can fit both, then applies
the inverse normalization at the **output** so reconstructions / forecasts
land back on the absolute scale of the input.

The "Reversible" part is the core point. Every (mean, std) pair we compute
in `forward(..., mode='norm')` is cached, and `forward(..., mode='denorm')`
restores the original scale using exactly those cached stats. This makes
the model's predictions interpretable as absolute prices / log-returns
rather than as standardized residuals the caller has to invert manually.

Affine. With `affine=True` (default) we add learnable per-channel
`(gamma, beta)` after the standardization — these aren't strictly necessary
but help the backbone shift the normalized scale slightly per channel,
matching the original RevIN paper's recipe.
"""

from __future__ import annotations

import torch
from torch import nn

EPS_DEFAULT = 1e-5


class RevIN(nn.Module):
    """Reversible per-instance normalization across the time dimension.

    Input shape: `(B, L, C)` — batch × seq_len × channels.
    `forward(x, mode='norm')` computes per-(B, C) mean/std over `L`,
    standardizes, optionally applies affine, and caches the stats.
    `forward(x, mode='denorm')` reverses using the cached stats — must be
    called on the **same instance** that produced them, immediately after
    the model output is in standardized space.

    Caching is per-instance: separate `RevIN` instances on different
    streams (e.g. multi-resolution) maintain independent caches.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = EPS_DEFAULT,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            msg = f"num_channels must be positive, got {num_channels}"
            raise ValueError(msg)
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            # Per-channel learnable scale + shift, broadcast over (B, L).
            self.affine_weight = nn.Parameter(torch.ones(num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        # Cached stats (set during norm, consumed during denorm).
        self.register_buffer("_mean", torch.zeros(0), persistent=False)
        self.register_buffer("_std",  torch.zeros(0), persistent=False)

    def forward(self, x: torch.Tensor, *, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        if mode == "apply_cached":
            return self._apply_cached(x)
        msg = f"Unknown mode {mode!r}; expected 'norm', 'denorm', or 'apply_cached'"
        raise ValueError(msg)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            msg = f"Expected (B, L, C), got shape {tuple(x.shape)}"
            raise ValueError(msg)
        if x.size(-1) != self.num_channels:
            msg = (
                f"Channel mismatch: layer has {self.num_channels} channels, "
                f"input has {x.size(-1)}"
            )
            raise ValueError(msg)
        # Mean/std over the time dimension only — per-(batch, channel) pair.
        mean = x.mean(dim=1, keepdim=True)
        # Use unbiased=False to match scikit / numpy population variance —
        # otherwise tiny windows produce slightly off scales.
        std = x.std(dim=1, keepdim=True, unbiased=False)
        # Cache (detached so backward through denorm doesn't push gradients
        # back into the stats — RevIN expects them held constant).
        self._mean = mean.detach()
        self._std = std.detach()
        x_n = (x - mean) / (std + self.eps)
        if self.affine:
            x_n = x_n * self.affine_weight + self.affine_bias
        return x_n

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean.numel() == 0 or self._std.numel() == 0:
            msg = (
                "denorm called before norm; the layer has no cached stats. "
                "Always call forward(x, mode='norm') before forward(y, mode='denorm')."
            )
            raise RuntimeError(msg)
        # Affine inverse first (subtract bias, divide by weight), then
        # multiply by the original std and add the original mean.
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * (self._std + self.eps) + self._mean

    def _apply_cached(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a NEW input using the most-recent cached (mean, std).

        This is the right operation for forecast targets: future bars need
        to live in the same RevIN frame as the input window so the head's
        prediction (also in that frame) is directly comparable. Recomputing
        stats from the future bars would (a) drift relative to the input's
        frame and (b) leak label information into the target.
        """
        if self._mean.numel() == 0 or self._std.numel() == 0:
            msg = (
                "apply_cached called before norm; the layer has no cached stats. "
                "Always call forward(input, mode='norm') before "
                "forward(future, mode='apply_cached')."
            )
            raise RuntimeError(msg)
        if x.dim() != 3:
            msg = f"Expected (B, L, C), got shape {tuple(x.shape)}"
            raise ValueError(msg)
        if x.size(-1) != self.num_channels:
            msg = (
                f"Channel mismatch: layer has {self.num_channels} channels, "
                f"input has {x.size(-1)}"
            )
            raise ValueError(msg)
        if x.size(0) != self._mean.size(0):
            msg = (
                f"Batch mismatch: cached stats have batch={self._mean.size(0)} "
                f"but input has batch={x.size(0)}. apply_cached requires the same "
                f"batch the cache was filled with."
            )
            raise RuntimeError(msg)
        x_n = (x - self._mean) / (self._std + self.eps)
        if self.affine:
            x_n = x_n * self.affine_weight + self.affine_bias
        return x_n
