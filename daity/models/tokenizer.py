"""Multi-resolution patch tokenizer for the daity_v3 backbone (DESIGN §3.1).

The backbone is PatchTST/Chronos-style: each numeric channel is patched
independently, weights shared across channels and across symbols. We
provide three primitives stacked into the full tokenizer:

  - `Patcher` — reshapes a `(B, L, C)` tensor into a sequence of overlapping
    patches `(B, n_patches, patch_len * C)` (channel-mixing patches) or
    `(B, n_patches, C, patch_len)` (channel-independent). DESIGN goes
    channel-independent in the lower 6 layers, channel-mixing in upper 2,
    so the patcher itself produces a per-channel layout and the backbone
    decides how to mix.

  - `MultiResTokenizer` — stitches the patches across resolutions
    (`5m / 15m / 60m / day`). Each stream is patched separately, gets a
    learned resolution embedding, then concatenated along the patch
    sequence. A learnable `[FORECAST]` summary token is prepended; the
    backbone reads that final hidden state for downstream heads.

  - `RevIN` — see `daity.models.revin`. Applied per-(symbol, window, scale)
    so each batch sample gets its own normalization stats and the model
    sees standardized values regardless of the symbol's price level.

Output of `MultiResTokenizer.forward(...)` is the token sequence ready to
feed into the backbone: `(B, 1 + sum(n_patches per scale), d_model)`.

Patch length defaults to 16 with stride 8 (DESIGN §3.1). For a 5m stream
of 256 bars, that's `(256 - 16)/8 + 1 = 31` patches. For a daily stream of
64 bars, that's `(64 - 16)/8 + 1 = 7` patches.
"""

from __future__ import annotations

import torch
from torch import nn

from daity.models.revin import RevIN

DEFAULT_PATCH_LEN = 16
DEFAULT_PATCH_STRIDE = 8


class Patcher(nn.Module):
    """Slice a `(B, L, C)` series into overlapping patches.

    Output shape: `(B, n_patches, C, patch_len)` — channel-independent.
    The backbone projects this through a per-channel-shared linear layer
    to get `(B, n_patches, C, d_model)`, then optionally flattens the C
    dim into the sequence axis for channel-independent attention or
    averages it for channel-mixing.

    `n_patches = floor((L - patch_len) / stride) + 1` when L >= patch_len,
    else 0 (caller raises rather than silently producing an empty tensor).
    """

    def __init__(
        self,
        patch_len: int = DEFAULT_PATCH_LEN,
        stride: int = DEFAULT_PATCH_STRIDE,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            msg = f"patch_len + stride must be positive, got {patch_len}, {stride}"
            raise ValueError(msg)
        self.patch_len = patch_len
        self.stride = stride

    def n_patches(self, seq_len: int) -> int:
        """How many patches a window of length `seq_len` produces."""
        if seq_len < self.patch_len:
            return 0
        return (seq_len - self.patch_len) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, L, C)` → `(B, n_patches, C, patch_len)`."""
        if x.dim() != 3:
            msg = f"Expected (B, L, C), got shape {tuple(x.shape)}"
            raise ValueError(msg)
        B, L, C = x.shape
        n = self.n_patches(L)
        if n == 0:
            msg = (
                f"Window of length {L} is shorter than patch_len {self.patch_len}; "
                f"need at least patch_len bars."
            )
            raise ValueError(msg)
        # `unfold` along the time axis produces (B, n_patches, C, patch_len)
        # without copy. Equivalent to a strided view.
        # Move C to dim 1 first, unfold along the new time axis, then reorder.
        x_ = x.transpose(1, 2)                   # (B, C, L)
        patches = x_.unfold(-1, self.patch_len, self.stride)  # (B, C, n, patch_len)
        return patches.permute(0, 2, 1, 3).contiguous()       # (B, n, C, patch_len)


class MultiResTokenizer(nn.Module):
    """Tokenize a multi-resolution OHLCV input into a single backbone-ready sequence.

    Forward input is a dict of per-scale tensors:
        {scale_name: (B, L_scale, C)}

    For each scale:
      1. RevIN-normalize (per-(B, scale, channel) instance).
      2. Patch into (B, n_patches_scale, C, patch_len).
      3. Project each (channel × patch_len) into d_model via a
         channel-shared `Linear(patch_len → d_model)`, then collapse the
         channel axis (mean by default, or attention-pool / per-channel-
         weight via `channel_pool`). This is the Phase-2.5 fix — the
         pre-2.5 path flattened (C × patch_len) into a single Linear,
         which baked channel-mixing into the tokenizer; this version
         keeps the per-channel projection independent (each channel
         goes through the same projection) so the backbone sees
         patches whose channel structure has been preserved through
         a PatchTST-style independence layer first.
      4. Add a learnable per-scale resolution embedding (broadcast over
         the patch axis).

    Then concatenate all scales' patches along the sequence axis and
    prepend a learnable `[FORECAST]` token. Output:
        (B, 1 + sum_scales(n_patches_scale), d_model).

    `channel_independent`: if True (default for Phase 2.5+), use the
    channel-shared per-channel projection. If False, use the legacy
    flattened-channel projection (for loading Phase 2.x checkpoints).
    """

    def __init__(
        self,
        scales: tuple[str, ...],
        num_channels: int,
        d_model: int = 320,
        patch_len: int = DEFAULT_PATCH_LEN,
        patch_stride: int = DEFAULT_PATCH_STRIDE,
        *,
        revin_affine: bool = True,
        channel_independent: bool = False,    # legacy default for backward compat
                                              # with pre-Phase-2.5 checkpoints
                                              # (A2/PU10/PU15/etc).
                                              # Phase 2.5+ configs explicitly
                                              # set this to True.
        channel_pool: str = "mean",
    ) -> None:
        super().__init__()
        if not scales:
            msg = "scales must be non-empty"
            raise ValueError(msg)
        if num_channels <= 0 or d_model <= 0:
            msg = f"num_channels and d_model must be positive ({num_channels}, {d_model})"
            raise ValueError(msg)
        if channel_pool not in {"mean", "attn"}:
            msg = f"channel_pool must be 'mean' or 'attn', got {channel_pool!r}"
            raise ValueError(msg)
        self.scales = tuple(scales)
        self.num_channels = num_channels
        self.d_model = d_model
        self.channel_independent = channel_independent
        self.channel_pool = channel_pool
        self.patcher = Patcher(patch_len=patch_len, stride=patch_stride)

        # Per-scale RevIN, since each scale's normalization stats are
        # independent (5m close ranges within a session, day close across
        # months — different distributions).
        self.revins = nn.ModuleDict({
            sc: RevIN(num_channels=num_channels, affine=revin_affine)
            for sc in scales
        })

        if channel_independent:
            # Channel-shared per-channel projection: Linear(patch_len → d_model)
            # applied identically to each of the `num_channels` channels.
            # Phase 2.5+ default — preserves channel independence at the
            # tokenizer layer (PatchTST-style).
            self.scale_proj_per_channel = nn.ModuleDict({
                sc: nn.Linear(patch_len, d_model)
                for sc in scales
            })
            # Attention-based channel pool (if requested): a learned
            # per-(scale) attention vector that scores each channel,
            # softmax over channels, then weighted sum.
            if channel_pool == "attn":
                self.channel_attn_query = nn.ParameterDict({
                    sc: nn.Parameter(torch.randn(d_model) * 0.02)
                    for sc in scales
                })
            else:
                self.channel_attn_query = None
            # Keep legacy attribute None so state_dict round-trip is clean.
            self.scale_projections = None
        else:
            # Legacy path: (C * patch_len) → d_model. Loads pre-2.5 ckpts.
            self.scale_projections = nn.ModuleDict({
                sc: nn.Linear(num_channels * patch_len, d_model)
                for sc in scales
            })
            self.scale_proj_per_channel = None
            self.channel_attn_query = None

        # Learnable resolution embedding per scale, added to every patch
        # of that scale so the backbone can tell them apart.
        self.resolution_embed = nn.Parameter(torch.zeros(len(scales), d_model))
        nn.init.normal_(self.resolution_embed, std=0.02)

        # Learnable [FORECAST] summary token, prepended to the sequence.
        self.forecast_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.forecast_token, std=0.02)

    def _project_patches(self, patches: torch.Tensor, sc: str) -> torch.Tensor:
        """`(B, n, C, patch_len)` → `(B, n, d_model)`.

        Channel-independent path: applies a SHARED `Linear(P → d_model)` per
        channel, then collapses the C axis via mean (default) or attention.
        Legacy path: flattens (C, P) and projects through a single Linear.
        """
        B, n, C, P = patches.shape
        if self.channel_independent:
            # Per-channel projection (channel-shared weights):
            #   (B, n, C, P) → (B, n, C, d_model)
            per_channel = self.scale_proj_per_channel[sc](patches)
            if self.channel_pool == "attn" and self.channel_attn_query is not None:
                # Soft-attention over channels:
                # logits = per_channel @ query  → (B, n, C)
                q = self.channel_attn_query[sc]                             # (d_model,)
                logits = (per_channel * q).sum(dim=-1)                      # (B, n, C)
                weights = logits.softmax(dim=-1).unsqueeze(-1)              # (B, n, C, 1)
                tokens = (per_channel * weights).sum(dim=2)                 # (B, n, d_model)
            else:
                tokens = per_channel.mean(dim=2)                            # (B, n, d_model)
            return tokens
        # Legacy flatten-and-project:
        flat = patches.reshape(B, n, C * P)
        return self.scale_projections[sc](flat)

    def forward(self, x_by_scale: dict[str, torch.Tensor]) -> torch.Tensor:
        """`{scale: (B, L_scale, C)}` → `(B, 1 + total_patches, d_model)`."""
        missing = [sc for sc in self.scales if sc not in x_by_scale]
        if missing:
            msg = f"Tokenizer expected scales {list(self.scales)}; missing {missing}"
            raise ValueError(msg)

        per_scale_tokens: list[torch.Tensor] = []
        batch_size: int | None = None
        for idx, sc in enumerate(self.scales):
            x = x_by_scale[sc]                                    # (B, L, C)
            if batch_size is None:
                batch_size = x.size(0)
            elif x.size(0) != batch_size:
                msg = (
                    f"Batch size mismatch across scales: scale {sc!r} has "
                    f"B={x.size(0)} but earlier scales had B={batch_size}"
                )
                raise ValueError(msg)
            x_n = self.revins[sc](x, mode="norm")                # (B, L, C)
            patches = self.patcher(x_n)                          # (B, n, C, patch_len)
            tokens = self._project_patches(patches, sc)          # (B, n, d_model)
            tokens = tokens + self.resolution_embed[idx]         # (B, n, d_model)
            per_scale_tokens.append(tokens)

        seq = torch.cat(per_scale_tokens, dim=1)                  # (B, total_n, d_model)
        forecast = self.forecast_token.expand(batch_size or 0, -1, -1)
        return torch.cat([forecast, seq], dim=1)                  # (B, 1+total_n, d_model)

    def n_patches_per_scale(self, seq_lens: dict[str, int]) -> dict[str, int]:
        """Helper for callers planning attention masks / positional encodings.

        `seq_lens = {scale: L}` → `{scale: n_patches}`.
        """
        return {sc: self.patcher.n_patches(L) for sc, L in seq_lens.items()}
