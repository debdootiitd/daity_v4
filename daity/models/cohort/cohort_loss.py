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
        w_clf: float = 0.0,
        w_sharpe: float = 0.0,
        clf_threshold_bps: float = 50.0,
        contrastive_ret_sim_thresh: float = 0.5,
        contrastive_tau: float = 0.1,
        smooth_l1_beta: float = 1.0,
        rank_top_k: int = 0,
        sharpe_tau: float = 0.005,
        sharpe_cost_bps: float = 30.0,
    ) -> None:
        super().__init__()
        self.w_reg = float(w_reg)
        self.w_rank = float(w_rank)
        self.w_bias = float(w_bias)
        self.w_sector = float(w_sector)
        self.w_contrastive = float(w_contrastive)
        self.w_clf = float(w_clf)
        self.w_sharpe = float(w_sharpe)
        self.clf_threshold_bps = float(clf_threshold_bps)
        self.contrastive_ret_sim_thresh = float(contrastive_ret_sim_thresh)
        self.contrastive_tau = float(contrastive_tau)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.rank_top_k = int(rank_top_k)
        # Sharpe-loss hyperparams. tau (softmax temperature) controls
        # portfolio concentration: smaller → closer to top-1; larger → equal-weight.
        # In log-return units. Default 0.005 = 50 bps (matches typical alpha scale).
        self.sharpe_tau = float(sharpe_tau)
        self.sharpe_cost_bps = float(sharpe_cost_bps)

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
        clf_logits: torch.Tensor | None = None,
        cohort_bias: torch.Tensor | None = None,
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

        # Cross-sectional demean per (B, H) — used for the ranking loss
        # which only cares about relative order. Use the *masked* mean
        # (over valid stocks only) so invalid (zero-imputed) labels don't
        # bias the cohort mean toward zero. Float32 cast prevents bf16
        # cancellation error in mean over N=200.
        if label_validity_per_stock is not None:
            mask3 = label_validity_per_stock.to(torch.float32)            # (B, N, H)
            denom_n = mask3.sum(dim=1).clamp_min(1.0)                     # (B, H)
            t_f32 = target.float() * mask3
            target_cohort_mean = (t_f32.sum(dim=1) / denom_n).to(target.dtype)
            # Demean using the *masked* mean; zero out invalid cells in residuals.
            target_c = (target - target_cohort_mean.unsqueeze(1)) * mask3.to(target.dtype)
            pred_c   = (pred   - pred.float().mean(dim=1, keepdim=True).to(pred.dtype)) * mask3.to(pred.dtype)
        else:
            pred_c   = _cohort_demean(pred)
            target_c = _cohort_demean(target)
            target_cohort_mean = target.float().mean(dim=1).to(target.dtype)

        # Regression loss DECOMPOSED into:
        #   loss_alpha = SmoothL1(pred_alpha, target_alpha) — cross-sectional
        #   loss_bias  = SmoothL1(cohort_bias, target_cohort_mean) — bias head
        # When `cohort_bias` is None: pure alpha loss (legacy behavior).
        # When provided: alpha + bias added so each head MUST learn its slice
        # (model can't absorb all signal into bias).
        sl1 = F.smooth_l1_loss(
            pred_c, target_c, beta=self.smooth_l1_beta, reduction="none",
        )                                                                # (B, N, H)
        if cohort_bias is not None:
            sl1_bias = F.smooth_l1_loss(
                cohort_bias, target_cohort_mean,
                beta=self.smooth_l1_beta, reduction="none",
            )                                                             # (B, H)
        else:
            sl1_bias = None
        sl1_per_bh = sl1.mean(dim=1)                                     # (B, H)
        sl1_masked = sl1_per_bh * validity_mask.to(sl1_per_bh.dtype)
        n_valid = validity_mask.to(sl1_per_bh.dtype).sum().clamp_min(1.0)
        loss_reg_alpha = sl1_masked.sum() / n_valid
        # Per-horizon alpha regression loss (H,)
        n_valid_per_h = validity_mask.to(sl1_per_bh.dtype).sum(dim=0).clamp_min(1.0)  # (H,)
        loss_reg_alpha_per_h = sl1_masked.sum(dim=0) / n_valid_per_h     # (H,)

        # Cohort-bias loss (when bias head is enabled)
        if sl1_bias is not None:
            sl1_bias_masked = sl1_bias * validity_mask.to(sl1_bias.dtype)   # (B, H)
            loss_reg_bias = sl1_bias_masked.sum() / n_valid
            loss_reg_bias_per_h = sl1_bias_masked.sum(dim=0) / n_valid_per_h
        else:
            loss_reg_bias = torch.zeros((), device=pred.device, dtype=pred.dtype)
            loss_reg_bias_per_h = torch.zeros(pred.shape[-1], device=pred.device, dtype=pred.dtype)

        # Total regression loss = alpha + bias (forces each head to learn its slice)
        loss_reg = loss_reg_alpha + loss_reg_bias
        loss_reg_per_h = loss_reg_alpha_per_h + loss_reg_bias_per_h

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

        # Win-rate classifier head: BCE on per-(stock, horizon) win label.
        # Win = (target > threshold_bps / 10000). Per-stock label validity
        # mask (when available) keeps only valid (b, n, h) cells in the loss.
        if self.w_clf > 0.0 and clf_logits is not None:
            win_thr = self.clf_threshold_bps / 10000.0
            win_target = (target > win_thr).to(clf_logits.dtype)              # (B, N, H)
            # Per-cell BCE; mask out invalid (b, h) horizons and per-stock
            # label-validity if provided.
            bce_per_cell = F.binary_cross_entropy_with_logits(
                clf_logits, win_target, reduction="none",
            )                                                                  # (B, N, H)
            cell_mask = validity_mask.unsqueeze(1).to(bce_per_cell.dtype)      # (B, 1, H)
            if label_validity_per_stock is not None:
                cell_mask = cell_mask * label_validity_per_stock.to(bce_per_cell.dtype)
            else:
                cell_mask = cell_mask.expand_as(bce_per_cell)
            n_cells = cell_mask.sum().clamp_min(1.0)
            loss_clf = (bce_per_cell * cell_mask).sum() / n_cells
            # Per-horizon clf loss (H,)
            masked_bce_h = (bce_per_cell * cell_mask).sum(dim=(0, 1))     # (H,)
            n_cells_per_h = cell_mask.sum(dim=(0, 1)).clamp_min(1.0)       # (H,)
            loss_clf_per_h = masked_bce_h / n_cells_per_h                  # (H,)
        else:
            loss_clf = torch.zeros((), device=pred.device, dtype=pred.dtype)
            loss_clf_per_h = torch.zeros(pred.shape[-1], device=pred.device, dtype=pred.dtype)

        # ----- Differentiable Sharpe loss (per-horizon, batch-Sharpe) -----
        # For each horizon h:
        #   w[b,n,h] = softmax(pred_with_bias[b,:,h] / sharpe_tau, dim=stocks)
        #             — long-only weights, sum=1 per (b,h); concentration via tau
        #   port_lr[b,h] = sum_n w[b,n,h] * target[b,n,h]   (linear in w)
        #   net_lr[b,h] = port_lr[b,h] − 2 * cost_bps/10000
        #   Sharpe_h = mean_b(net_lr) / std_b(net_lr)   (over valid b's)
        # Loss = mean_h (-Sharpe_h) over horizons where ≥2 valid anchors.
        #
        # Invalid stocks: set their pred to -inf in the softmax so they get 0 weight.
        # If cohort_bias is provided, use pred+cohort_bias for absolute-return ranking
        # (in practice ranks identical since bias is constant per (b,h)).
        if self.w_sharpe > 0.0:
            B, N, H = pred.shape
            pred_for_sharpe = pred.float()
            if cohort_bias is not None:
                pred_for_sharpe = pred_for_sharpe + cohort_bias.float().unsqueeze(1)
            # Per-stock validity masking: -inf for invalid stocks
            if label_validity_per_stock is not None:
                mask_stock = label_validity_per_stock.to(pred_for_sharpe.dtype)  # (B,N,H)
                pred_for_sharpe = pred_for_sharpe + (mask_stock - 1.0) * 1e9
            # Softmax over stocks within each (b, h)
            w = torch.softmax(pred_for_sharpe / max(self.sharpe_tau, 1e-6), dim=1)  # (B,N,H)
            # Portfolio log-return per (b, h) = w · target
            port_lr = (w * target.float()).sum(dim=1)                              # (B, H)
            cost_lr = 2.0 * self.sharpe_cost_bps / 10000.0
            net_lr = port_lr - cost_lr                                              # (B, H)
            # Compute Sharpe per horizon over VALID anchors
            valid = validity_mask.float()                                           # (B, H)
            sharpe_h_list = []
            for h in range(H):
                vb = valid[:, h].bool()
                n_valid = vb.sum().item()
                if n_valid < 2:
                    continue
                vh = net_lr[vb, h]
                mu = vh.mean()
                # Use unbiased=False for differentiability with small N; +eps for stability
                sigma = vh.std(unbiased=False) + 1e-6
                sharpe_h = mu / sigma
                sharpe_h_list.append(-sharpe_h)   # minimize negative = maximize Sharpe
            if sharpe_h_list:
                loss_sharpe = torch.stack(sharpe_h_list).mean()
            else:
                loss_sharpe = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            loss_sharpe = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            self.w_reg          * loss_reg +
            self.w_rank         * loss_rank +
            self.w_bias         * loss_bias +
            self.w_sector       * loss_sector +
            self.w_contrastive  * loss_contrastive +
            self.w_clf          * loss_clf +
            self.w_sharpe       * loss_sharpe
        )
        return {
            "total":       total,
            "reg":         loss_reg.detach(),
            "reg_alpha":   loss_reg_alpha.detach(),       # cross-sectional
            "reg_bias":    loss_reg_bias.detach(),         # cohort-mean head
            "rank":        loss_rank.detach(),
            "bias":        loss_bias.detach(),
            "sector":      loss_sector.detach(),
            "contrastive": loss_contrastive.detach(),
            "clf":         loss_clf.detach(),
            "sharpe":      loss_sharpe.detach(),          # negative Sharpe (lower is better)
            "reg_per_h":   loss_reg_per_h.detach(),     # (H,) alpha+bias combined
            "reg_alpha_per_h": loss_reg_alpha_per_h.detach(),  # (H,)
            "reg_bias_per_h":  loss_reg_bias_per_h.detach(),   # (H,)
            "clf_per_h":   loss_clf_per_h.detach(),     # (H,)
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
