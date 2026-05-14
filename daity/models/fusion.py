"""Phase 4 step 3 — gated cross-attention fusion (DESIGN §3.4).

Per spec:
- Inserted in the upper two backbone layers.
- Queries: backbone token-stream hidden `(B, 1+P, d_model_q=480)`.
- Keys/Values: OB encoder output `(B, K=64, d_model_kv=192)` projected
  to d_model_q.
- Output: residual `q' = q + gate * cross_attn(q, kv)` where `gate` is
  a learned per-token sigmoid `(B, 1+P, 1)` letting the model ignore
  OB when it's absent (no `kv` provided) or stale.
- Curriculum: a `_gate_force_zero` buffer flips True for the first
  ~1500 opt steps to keep the OB tower off while the heads re-stabilize
  on the new training distribution.

The fused-backbone wrapper (`OBFusedBackbone`) wraps the existing
(frozen) `Backbone` without modifying it — it iterates the same layers
and inserts fusion residuals after the chosen ones.
"""

from __future__ import annotations

import torch
from torch import nn

from daity.models.backbone import Backbone


class GatedCrossAttnLayer(nn.Module):
    """One layer of gated cross-attention from backbone to OB encoder.

    Q from `(B, T, d_model_q)`; K, V from OB encoder `(B, K, d_model_kv)`
    after projection to `d_model_q`. Output: `(B, T, d_model_q)` residual.

    Gate: `g = sigmoid(W_g · q + b_g) : (B, T, 1)`. Zero-initialized so
    the layer is a no-op at init.
    """

    def __init__(
        self,
        *,
        d_model_q: int = 480,
        d_model_kv: int = 192,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.d_model_q = d_model_q
        self.n_heads = n_heads
        self.head_dim = d_model_q // n_heads
        if self.head_dim * n_heads != d_model_q:
            msg = f"d_model_q={d_model_q} not divisible by n_heads={n_heads}"
            raise ValueError(msg)

        # Project KV from OB encoder's d=192 to backbone's d=480.
        self.kv_proj = nn.Linear(d_model_kv, 2 * d_model_q, bias=False)
        # Q stays in backbone space.
        self.q_proj = nn.Linear(d_model_q, d_model_q, bias=False)
        self.out_proj = nn.Linear(d_model_q, d_model_q, bias=False)

        # Pre-attention norm on the query stream (residual-friendly).
        self.norm_q = nn.LayerNorm(d_model_q)

        # Per-token sigmoid gate. Zero-init bias + weight → gate=0.5 at
        # init; we explicitly add an `init_gate_open=False` mode that
        # initializes the gate to ~0 (sigmoid(-large) ≈ 0) so the
        # residual is near-no-op at init.
        self.gate_proj = nn.Linear(d_model_q, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -4.0)  # sigmoid(-4) ≈ 0.018

        self.dropout = nn.Dropout(dropout)

        # Curriculum: forced-zero gate. `1 = forced zero`, `0 = use learned gate`.
        # Buffer so it survives state_dict round-trip.
        self.register_buffer(
            "_gate_force_zero", torch.tensor(1, dtype=torch.int8),
            persistent=True,
        )

    def forward(
        self,
        q: torch.Tensor,       # (B, T, d_model_q)
        kv: torch.Tensor | None,  # (B, K, d_model_kv) or None
    ) -> torch.Tensor:
        """Returns `q + gate * cross_attn(q, kv)`. If `kv is None`, returns `q`.

        The `gate * cross_attn` term is masked to zero when:
        - `_gate_force_zero` is 1 (curriculum), or
        - `kv` is None (no OB at inference for this symbol).
        """
        if kv is None:
            return q

        b, t, d = q.shape
        h = self.n_heads
        hd = self.head_dim

        q_n = self.norm_q(q)
        q_h = self.q_proj(q_n).reshape(b, t, h, hd).transpose(1, 2)  # (B, h, T, hd)

        kv_proj = self.kv_proj(kv)                                  # (B, K, 2*d)
        k_p, v_p = kv_proj.chunk(2, dim=-1)
        k_h = k_p.reshape(b, kv.shape[1], h, hd).transpose(1, 2)    # (B, h, K, hd)
        v_h = v_p.reshape(b, kv.shape[1], h, hd).transpose(1, 2)    # (B, h, K, hd)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q_h, k_h, v_h, dropout_p=0.0, is_causal=False,
        )                                                            # (B, h, T, hd)
        attn = attn.transpose(1, 2).reshape(b, t, d)                # (B, T, d)
        attn = self.out_proj(attn)
        attn = self.dropout(attn)

        gate = torch.sigmoid(self.gate_proj(q_n))                    # (B, T, 1)
        if self._gate_force_zero.item():
            gate = torch.zeros_like(gate)

        return q + gate * attn

    def set_gate_force_zero(self, force: bool) -> None:
        self._gate_force_zero.fill_(int(bool(force)))


class OBFusedBackbone(nn.Module):
    """Wrap a (frozen) Phase-3 `Backbone` and inject OB cross-attention.

    Behavior:
      - Iterate the backbone's transformer layers normally.
      - After each layer index in `inject_at_layers`, apply the
        corresponding `GatedCrossAttnLayer(q=layer_out, kv=ob_kv)`.
      - With `ob_kv=None` (symbol lacks OB at this anchor), the fusion
        residual is a no-op — output bit-equals frozen-Backbone output.

    NB: Backbone is referenced (not copied). LoRA banks on the frozen
    backbone are still adapted via PEFT routing in `PostTrainModule`'s
    forward path — but Phase 4's `FineTuneModule` calls THIS wrapper
    instead of `peft_backbone(...)` directly.
    """

    def __init__(
        self,
        backbone: Backbone,
        *,
        d_model_q: int = 480,
        d_model_kv: int = 192,
        n_heads: int = 8,
        inject_at_layers: tuple[int, ...] = (10, 11),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.inject_at_layers = set(inject_at_layers)
        self.fusion_layers = nn.ModuleDict({
            str(i): GatedCrossAttnLayer(
                d_model_q=d_model_q, d_model_kv=d_model_kv,
                n_heads=n_heads, dropout=dropout,
            )
            for i in inject_at_layers
        })

    def set_gate_force_zero(self, force: bool) -> None:
        """Flip the curriculum on every fusion layer at once."""
        for layer in self.fusion_layers.values():
            layer.set_gate_force_zero(force)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ob_kv: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Forward through frozen backbone with OB fusion at injection points.

        Args:
            x:       `(B, T, d_model_q)` — same input as `Backbone.forward`.
            ob_kv:   `(B, K, d_model_kv)` from `OBEncoder.forward`, or None
                     for symbols without OB at this anchor.
            causal:  passthrough to backbone (Phase 3 uses `causal=False`).

        Returns: `(B, T, d_model_q)` final hidden — same shape as backbone.
        """
        # The Backbone's forward applies layers + a final norm. We need
        # to interleave fusion residuals BETWEEN backbone layers, so
        # we call layer-by-layer ourselves and apply backbone's final
        # norm at the end.
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x, causal=causal)
            if i in self.inject_at_layers:
                x = self.fusion_layers[str(i)](x, ob_kv)
        x = self.backbone.final_norm(x)
        return x
