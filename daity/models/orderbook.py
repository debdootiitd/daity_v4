"""Phase 4 step 2 — OB encoder (DESIGN §3.4).

A small transformer (4 layers, d_model=192) consuming K=64 engineered
snapshots and emitting `(B, K, d_model)` tokens for the fusion block to
cross-attend over.

Input: `(B, K=64, n_features=8)` — the per-snapshot feature vector built
by `daity.data.ob_features`.

Architecture:
  Input → LayerNorm (normalize feature scales) → Linear (8 → 192)
       → 4 × `TransformerBlock` (lifted from `daity.models.backbone`)
       → Output `(B, K, 192)`

No RoPE position encoding on the snapshot axis — the OB stream is a
short time-series; we use a learned positional embedding over K=64
positions, which is small enough not to overfit.
"""

from __future__ import annotations

import torch
from torch import nn

from daity.models.backbone import TransformerBlock


class OBEncoder(nn.Module):
    """4-layer transformer over K=64 engineered L2 snapshots.

    Params at d_model=192, n_layers=4, ffn_ratio=4: ~620K trainable
    (well below the Phase-4 5M budget). Position embeddings add
    K × d_model = 12K. Input projection adds n_features × d_model = 1.5K.
    """

    def __init__(
        self,
        *,
        n_features: int = 8,
        k_snapshots: int = 64,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k_snapshots = k_snapshots
        self.d_model = d_model

        # Normalize feature scales before projection (OB feature channels
        # span 6 orders of magnitude: log-quantity ~13, microprice ~1400,
        # book_pressure in [-1, 1], spread ~1.5).
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, d_model)
        # Learned positional embedding (additive, on the snapshot axis).
        self.pos_embed = nn.Parameter(torch.zeros(1, k_snapshots, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Transformer trunk — non-causal (snapshots are short context,
        # bidirectional attention is fine).
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, ffn_ratio=ffn_ratio,
                max_seq_len=k_snapshots,
                attn_dropout=dropout, ffn_dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: `(B, K, n_features)` → `(B, K, d_model)`.

        Caller must already have shape `(B, K, n_features)` — that is, K
        snapshots already aligned at the anchor. For symbols that lack
        OB at this anchor, the fusion block should NOT call us; mask
        upstream instead.
        """
        if x.shape[1] != self.k_snapshots:
            msg = (
                f"OBEncoder expects K={self.k_snapshots} snapshots, got "
                f"{x.shape[1]} (full shape {tuple(x.shape)})"
            )
            raise ValueError(msg)
        x = self.input_norm(x)
        x = self.input_proj(x)            # (B, K, d_model)
        x = x + self.pos_embed[:, :x.shape[1]]
        for layer in self.layers:
            x = layer(x, causal=False)
        return self.final_norm(x)
