"""Cross-sectional + listwise loss for the cohort architecture.

Key design decision: the loss is computed on the CROSS-SECTIONAL ALPHA
component (i.e., per-cohort, per-horizon, subtract the cohort mean of
both prediction and target). This makes the loss mathematically
invariant to constant shifts in the prediction — kills the v78-style
bias problem where the model could absorb the train/val drift difference
into a free per-cohort offset.

Loss components per (cohort, horizon):
  1. SmoothL1 on (pred_centered - target_centered) — primary regression
     signal. Smooth-L1 is robust against tail returns (Huber δ=1.0).
  2. Listwise Plackett-Luce loss — aligns the cohort's predicted ranking
     to the target ranking. Computed in pure-PyTorch (no sort/topk
     gradient issues).
  3. Bias regularizer — penalize abs(cohort_mean(pred)) so the model
     doesn't drift far from zero-mean predictions. Cheap insurance.

All combined under a per-horizon weight that the trainer supplies. Heads
that are invalid for the current anchor (per `head_validity_mask`) are
masked out of the loss entirely — both their regression and rank loss
contributions become zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _cohort_demean(x: torch.Tensor) -> torch.Tensor:
    """Subtract cohort mean along the N axis. Input/output: (B, N, H)."""
    return x - x.mean(dim=1, keepdim=True)


def plackett_luce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_top_k: int = 0,
) -> torch.Tensor:
    """Listwise Plackett-Luce loss.

    Sorts each (cohort, horizon) by target descending, then for k = 0..K-1
    we compute the log-likelihood that the model selects target_rank[k]
    next from the remaining items, with probability proportional to
    softmax(pred). This is a stable, autograd-friendly listwise loss.

    `max_top_k`: if > 0, only consider the top-K positions of the ranking
    (the rest contribute nothing). This focuses the loss on the head of
    the ranking — what the strategy actually trades. 0 = full list.

    Returns scalar mean over (B, H).
    """
    if pred.shape != target.shape:
        msg = f"pred {tuple(pred.shape)} vs target {tuple(target.shape)} mismatch"
        raise ValueError(msg)
    B, N, H = pred.shape
    # Move horizon to leading dim so we sort/score per (B, H) independently.
    pred_t   = pred.permute(0, 2, 1).reshape(B * H, N)                  # (B*H, N)
    target_t = target.permute(0, 2, 1).reshape(B * H, N)                # (B*H, N)
    sort_idx = target_t.argsort(dim=1, descending=True)                 # (B*H, N)
    pred_sorted = pred_t.gather(1, sort_idx)                            # (B*H, N)

    K = N if max_top_k <= 0 else min(max_top_k, N)
    losses = []
    # logsumexp over the remaining candidates at each rank k. Standard
    # plackett-luce decomposition:
    #   P(i) = exp(s_i) / sum_{j in remaining} exp(s_j)
    #   -log P(i) = -s_i + logsumexp(s_{remaining})
    # We process ranks sequentially; at each step we mask out previously-
    # picked items.
    remaining_mask = torch.ones_like(pred_sorted, dtype=torch.bool)     # (B*H, N)
    for k in range(K):
        s_k = pred_sorted[:, k]                                         # (B*H,)
        # logsumexp over the still-eligible items at this rank.
        masked = pred_sorted.masked_fill(~remaining_mask, float("-inf"))
        lse = torch.logsumexp(masked, dim=1)                            # (B*H,)
        losses.append(-(s_k - lse))
        # Remove position k from the remaining pool.
        remaining_mask[:, k] = False
    return torch.stack(losses, dim=0).mean()


class CohortLoss(nn.Module):
    """Loss aggregator for the cohort model.

    Weights:
        w_reg:    SmoothL1 on cross-sectional residuals.
        w_rank:   Plackett-Luce on cross-sectional ranking.
        w_bias:   |cohort_mean(pred)|² regularizer.
        smooth_l1_beta: Huber δ for SmoothL1 (default 1.0).
        rank_top_k:     max top-K positions for PL loss (0 = full list).
    """

    def __init__(
        self,
        *,
        w_reg: float = 1.0,
        w_rank: float = 0.5,
        w_bias: float = 0.1,
        w_sector: float = 0.0,
        w_contrastive: float = 0.0,
        contrastive_ret_sim_thresh: float = 0.5,
        contrastive_tau: float = 0.1,
        smooth_l1_beta: float = 1.0,
        rank_top_k: int = 0,
    ) -> None:
        super().__init__()
        self.w_reg = float(w_reg)
        self.w_rank = float(w_rank)
        self.w_bias = float(w_bias)
        self.w_sector = float(w_sector)
        self.w_contrastive = float(w_contrastive)
        self.contrastive_ret_sim_thresh = float(contrastive_ret_sim_thresh)
        self.contrastive_tau = float(contrastive_tau)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.rank_top_k = int(rank_top_k)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        validity_mask: torch.Tensor,
        *,
        sector_logits: torch.Tensor | None = None,
        sector_labels: torch.Tensor | None = None,
        contrastive_embeds: torch.Tensor | None = None,
        contrastive_sector_ids: torch.Tensor | None = None,
        label_validity_per_stock: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss components and totals.

        Args:
            pred:   (B, N, H) — model output, in log-return units.
            target: (B, N, H) — realized cumulative log returns.
            validity_mask: (B, H) bool — True if horizon h is valid for
                cohort b. Per-horizon (anchor-dependent), not per-stock.
                Invalid (b, h) pairs are zeroed out of the loss.

        Returns dict with scalars: `total`, `reg`, `rank`, `bias`, and
        also `n_valid_horizons` (for logging the masking density).
        """
        if pred.shape != target.shape:
            msg = f"pred {tuple(pred.shape)} vs target {tuple(target.shape)} mismatch"
            raise ValueError(msg)
        B, N, H = pred.shape
        if validity_mask.shape != (B, H):
            msg = (
                f"validity_mask must be (B={B}, H={H}); got {tuple(validity_mask.shape)}"
            )
            raise ValueError(msg)

        # Cross-sectional demean per (B, H).
        pred_c   = _cohort_demean(pred)
        target_c = _cohort_demean(target)

        # Per-(B, H) reg loss (mean over N), masked by validity.
        # SmoothL1: sums over N implicitly via mean.
        sl1 = F.smooth_l1_loss(
            pred_c, target_c, beta=self.smooth_l1_beta, reduction="none",
        )                                                                # (B, N, H)
        sl1_per_bh = sl1.mean(dim=1)                                     # (B, H)
        sl1_masked = sl1_per_bh * validity_mask.to(sl1_per_bh.dtype)
        n_valid = validity_mask.to(sl1_per_bh.dtype).sum().clamp_min(1.0)
        loss_reg = sl1_masked.sum() / n_valid

        # Plackett-Luce rank loss on cross-sectional residuals.
        if self.w_rank > 0.0:
            # We need to mask invalid (B, H) pairs. Do this by zero-out;
            # but PL is in log-space. Trick: compute PL on ALL pairs, get
            # (B, H) by reshaping back, then mask.
            # The plackett_luce_loss function returns a scalar mean — we
            # need per-(B,H) values for masking. Re-implement inline.
            loss_rank = self._plackett_luce_masked(pred_c, target_c, validity_mask)
        else:
            loss_rank = torch.zeros((), device=pred.device, dtype=pred.dtype)

        # Bias regularizer: |cohort_mean(pred)|² per (B, H), masked.
        bias = pred.mean(dim=1).pow(2)                                   # (B, H)
        loss_bias = (bias * validity_mask.to(bias.dtype)).sum() / n_valid

        # Sector classification auxiliary loss (provides dense per-stock
        # supervision that forces the encoder to organize stocks by sector).
        if (
            self.w_sector > 0.0
            and sector_logits is not None
            and sector_labels is not None
        ):
            # sector_logits: (B, N, n_sectors); sector_labels: (B, N) long.
            sl_flat = sector_logits.reshape(-1, sector_logits.size(-1))
            lb_flat = sector_labels.reshape(-1)
            loss_sector = F.cross_entropy(sl_flat, lb_flat)
        else:
            loss_sector = torch.zeros((), device=pred.device, dtype=pred.dtype)

        # Supervised intra-cohort contrastive (InfoNCE) — mines positives from
        # (same sector AND high realized-return cosine similarity). Forces the
        # encoder's CLS to organize semantic clusters and prevents the v5-style
        # collapse to a single direction. See cohort_loss.contrastive_intra_cohort.
        if (
            self.w_contrastive > 0.0
            and contrastive_embeds is not None
            and contrastive_sector_ids is not None
        ):
            loss_contrastive = self._contrastive_intra_cohort(
                contrastive_embeds, contrastive_sector_ids, target,
                label_validity_per_stock,
            )
        else:
            loss_contrastive = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            self.w_reg          * loss_reg +
            self.w_rank         * loss_rank +
            self.w_bias         * loss_bias +
            self.w_sector       * loss_sector +
            self.w_contrastive  * loss_contrastive
        )
        return {
            "total":       total,
            "reg":         loss_reg.detach(),
            "rank":        loss_rank.detach(),
            "bias":        loss_bias.detach(),
            "sector":      loss_sector.detach(),
            "contrastive": loss_contrastive.detach(),
            "n_valid_horizons": n_valid.detach(),
        }

    def _contrastive_intra_cohort(
        self,
        embeds: torch.Tensor,                 # (B, N, d_proj)
        sector_ids: torch.Tensor,             # (B, N) long
        target: torch.Tensor,                 # (B, N, H)
        label_validity: torch.Tensor | None,  # (B, N, H) bool
    ) -> torch.Tensor:
        """Supervised InfoNCE within each cohort.

        Positives for sample (b, i): all j ≠ i such that
            - sector_ids[b, i] == sector_ids[b, j]   (same macro sector)
            - cosine(target[b, i, :], target[b, j, :]) > threshold
              (similar realized 10-horizon return signature)

        Loss = -log( sum_j∈P exp(sim_ij/τ) / sum_j≠i exp(sim_ij/τ) ).
        Averaged over (b, i) that have ≥1 positive.
        """
        if embeds.dim() != 3:
            msg = f"contrastive_embeds must be (B, N, d), got {tuple(embeds.shape)}"
            raise ValueError(msg)
        B, N, _ = embeds.shape
        # L2-normalize embeddings for cosine logits.
        z = torch.nn.functional.normalize(embeds, dim=-1)        # (B, N, d)
        sim = torch.bmm(z, z.transpose(1, 2)) / self.contrastive_tau  # (B, N, N)
        # Mask self-pairs (diagonal).
        diag_mask = torch.eye(N, device=z.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Sector equality matrix.
        sec_eq = sector_ids.unsqueeze(2) == sector_ids.unsqueeze(1)   # (B, N, N)
        # Target-return cosine similarity matrix.
        # If we have per-stock label validity, mask out invalid horizons by
        # replacing them with zeros — they then don't bias the cosine.
        t = target.clone()
        if label_validity is not None:
            t = torch.where(label_validity, t, torch.zeros_like(t))
        t_norm = torch.nn.functional.normalize(t, dim=-1, eps=1e-8)
        ret_sim = torch.bmm(t_norm, t_norm.transpose(1, 2))             # (B, N, N)

        # Positive mask = same sector AND ret_sim > threshold AND not diag.
        pos_mask = sec_eq & (ret_sim > self.contrastive_ret_sim_thresh) & (~diag_mask)

        # Compute per-row softmax of sim (over non-diagonal entries),
        # then collect log-prob of positives.
        # log_softmax handles the masked -inf entries correctly.
        log_p = torch.log_softmax(sim, dim=-1)                          # (B, N, N)
        # Per row, sum of log probs at positive positions, then take
        # log-sum-exp form: log(sum_j∈P p_j) = logsumexp_j∈P(log_p_ij).
        masked_log_p = log_p.masked_fill(~pos_mask, float("-inf"))      # (B, N, N)
        # log_sum_exp over j with -inf elsewhere → if no positives, returns -inf.
        per_row_loss = -torch.logsumexp(masked_log_p, dim=-1)            # (B, N)

        # Mask out rows that have no positives (would be -inf otherwise).
        has_pos = pos_mask.any(dim=-1)                                   # (B, N)
        # Replace -inf with 0 where no positives, and ignore in the mean.
        per_row_loss = torch.where(has_pos, per_row_loss,
                                   torch.zeros_like(per_row_loss))
        n_with_pos = has_pos.to(per_row_loss.dtype).sum().clamp_min(1.0)
        return per_row_loss.sum() / n_with_pos

    def _plackett_luce_masked(
        self,
        pred_c: torch.Tensor,
        target_c: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PL loss with per-(B, H) masking.

        Same algorithm as `plackett_luce_loss` but returns a (B, H) tensor
        of per-pair losses (so we can apply the mask before reduction).
        """
        B, N, H = pred_c.shape
        K = N if self.rank_top_k <= 0 else min(self.rank_top_k, N)
        # Move horizon to leading dim.
        p = pred_c.permute(0, 2, 1).reshape(B * H, N)
        t = target_c.permute(0, 2, 1).reshape(B * H, N)
        sort_idx = t.argsort(dim=1, descending=True)
        p_sorted = p.gather(1, sort_idx)

        remaining = torch.ones_like(p_sorted, dtype=torch.bool)
        per_step = []
        for k in range(K):
            s_k = p_sorted[:, k]
            masked = p_sorted.masked_fill(~remaining, float("-inf"))
            lse = torch.logsumexp(masked, dim=1)
            per_step.append(-(s_k - lse))
            remaining[:, k] = False
        per_pair = torch.stack(per_step, dim=0).mean(dim=0)              # (B*H,)
        per_pair = per_pair.view(B, H)
        masked_loss = per_pair * validity_mask.to(per_pair.dtype)
        n_valid = validity_mask.to(per_pair.dtype).sum().clamp_min(1.0)
        return masked_loss.sum() / n_valid
