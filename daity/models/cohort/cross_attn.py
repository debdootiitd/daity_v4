"""Cross-attention conditioner for the cohort architecture.

Per-stock queries (the stock's patch tokens + CLS) cross-attend over the
cohort-shared market context. The output is each stock's "market-conditioned
state" — same shape as the input stock tokens, just enriched with cross-
stock + regime context.

We reuse heads.py's `_CrossAttnBlock` since it already implements the
right thing (pre-norm cross-attention + FFN).

Shapes:
  Input:
    stock_state : (B, N, 1+T, d_model) — N stocks per cohort, 1+T tokens each
    market_ctx  : (B, M, d_model)      — cohort-shared market context
                                          (M = N + n_regime_tokens)
  Output:
    conditioned : (B, N, 1+T, d_model) — same shape, market-conditioned
"""

from __future__ import annotations

import torch
from torch import nn

from daity.models.cohort.top_k_attn import TopKCrossAttnBlock
from daity.models.heads import _CrossAttnBlock


class CrossAttentionConditioner(nn.Module):
    """K cross-attention blocks: per-stock tokens query the market context.

    Stock-state tokens are flattened across (B, N) into a "batch-like" axis
    so each stock independently cross-attends to its cohort's market context.

    `top_k` (optional): if set, each stock's query attends only to the top-K
    market_ctx tokens. v5 diagnostic showed full attention collapses to
    uniform; top-K forces sparsity.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if top_k is not None and top_k > 0:
            self.blocks = nn.ModuleList([
                TopKCrossAttnBlock(d_model, n_heads, ffn_ratio, top_k=top_k)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                _CrossAttnBlock(d_model, n_heads, ffn_ratio)
                for _ in range(n_layers)
            ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        stock_state: torch.Tensor,
        market_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """`(B, N, S, d_model)` + `(B, M, d_model)` → `(B, N, S, d_model)`."""
        if stock_state.dim() != 4:
            msg = (
                f"stock_state must be (B, N, S, d_model), got {tuple(stock_state.shape)}"
            )
            raise ValueError(msg)
        if market_ctx.dim() != 3:
            msg = f"market_ctx must be (B, M, d_model), got {tuple(market_ctx.shape)}"
            raise ValueError(msg)
        B, N, S, D = stock_state.shape
        Bm, M, Dm = market_ctx.shape
        if B != Bm:
            msg = f"Cohort batch mismatch: stock_state B={B}, market_ctx B={Bm}"
            raise ValueError(msg)
        if D != Dm:
            msg = f"d_model mismatch: stock_state D={D}, market_ctx D={Dm}"
            raise ValueError(msg)

        # Flatten (B, N) into a single batch axis so each stock independently
        # cross-attends to its cohort's market context (broadcast market_ctx
        # over N).
        q = stock_state.view(B * N, S, D)                               # (B*N, S, d)
        kv = market_ctx.unsqueeze(1).expand(B, N, M, D).reshape(B * N, M, D)
        for block in self.blocks:
            q = block(q, kv)
        q = self.final_norm(q)
        return q.view(B, N, S, D)
