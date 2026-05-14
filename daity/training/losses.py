"""Loss functions + masking helpers for Phase 2 SSL pretraining (DESIGN §3.2).

The three SSL objectives are wired through three functions here:

  - `apply_random_patch_mask(tokens, mask_ratio, mask_token)` — masks a
    fraction of patch tokens by replacing their embeddings with a learned
    `mask_token` vector. Returns `(masked_tokens, mask)` where `mask`
    is a boolean per-patch indicator of which positions were replaced
    (the loss only scores those positions).

  - `volume_weighted_mse(pred, target, volume, mask, low_q, high_q)` —
    masked-reconstruction loss, downweighting bars whose volume is below
    the 20th percentile (×0.25) and upweighting bars in the top decile
    (×1.5) per DESIGN §3.2. Volume is the per-bar volume from the OHLCV
    input — passed in by the training step which has the original tensors.

  - `next_patch_mse(pred, target, mask)` — plain unweighted MSE on the
    causal NTP head. `mask` is the "predict-the-next-patch" alignment
    mask: usually `mask[:, :-1] = True, mask[:, -1] = False`.

  - `info_nce_loss(z_a, z_b, temperature)` — symmetric InfoNCE between two
    batches of unit-norm projections. `z_a[i]` and `z_b[i]` are positives;
    everything else in the batch is a negative. Temperature 0.07 is the
    common SimCLR default.

Volume weighting note. DESIGN §3.2 says "bars with volume below the 20th
percentile contribute 0.25× weight, bars in the top decile contribute
1.5×". We compute these percentiles per-batch over the volume tensor;
batched percentiles are stable on float32 batches of any size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MASK_RATIO = 0.4   # DESIGN §3.2: 30-40% of patches masked
DEFAULT_LOW_Q = 0.20       # bottom 20% of volume → 0.25× weight
DEFAULT_HIGH_Q = 0.90      # top 10% of volume → 1.5× weight
DEFAULT_LOW_W = 0.25
DEFAULT_HIGH_W = 1.5
DEFAULT_BASE_W = 1.0


def apply_random_patch_mask(
    tokens: torch.Tensor,
    *,
    mask_ratio: float,
    mask_token: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace `mask_ratio` of patch positions with `mask_token`.

    `tokens` is the post-tokenizer sequence `(B, T, d_model)`. We mask
    positions `[1:T]` only — position 0 is the FORECAST token and must
    never be masked (the contrastive head reads it).

    Returns `(masked_tokens, mask)` where `mask` is `(B, T)` boolean,
    True at positions that were masked (loss target).
    """
    if not 0.0 < mask_ratio < 1.0:
        msg = f"mask_ratio must be in (0, 1), got {mask_ratio}"
        raise ValueError(msg)
    B, T, D = tokens.shape
    if mask_token.shape != (D,):
        msg = (
            f"mask_token must have shape ({D},) matching d_model; "
            f"got {tuple(mask_token.shape)}"
        )
        raise ValueError(msg)

    # Mask candidates: positions [1:T] (skip FORECAST at 0).
    n_patches = T - 1
    n_to_mask = max(1, int(n_patches * mask_ratio))
    # Sample without replacement per batch row.
    rand = torch.rand(B, n_patches, generator=generator, device=tokens.device)
    # Top-k smallest random values → positions to mask.
    _, idx = rand.topk(n_to_mask, dim=-1, largest=False)         # (B, n_to_mask)
    mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, idx + 1, True)                              # +1 for FORECAST offset

    # Replace masked positions with the mask token (broadcast over B).
    masked = torch.where(
        mask.unsqueeze(-1),
        mask_token.view(1, 1, D).expand_as(tokens),
        tokens,
    )
    return masked, mask


def volume_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    volume: torch.Tensor,
    mask: torch.Tensor,
    low_q: float = DEFAULT_LOW_Q,
    high_q: float = DEFAULT_HIGH_Q,
    low_w: float = DEFAULT_LOW_W,
    high_w: float = DEFAULT_HIGH_W,
    base_w: float = DEFAULT_BASE_W,
) -> torch.Tensor:
    """Weighted-MSE loss on masked positions.

    Shapes:
      pred   : (B, P, C, patch_len)
      target : (B, P, C, patch_len)
      volume : (B, P)             — per-patch volume (mean over patch_len)
      mask   : (B, P)             — True = include in loss (the masked positions)

    Returns a scalar loss (mean over masked elements, weighted).
    """
    if pred.shape != target.shape:
        msg = f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must match"
        raise ValueError(msg)
    if volume.shape != mask.shape:
        msg = (
            f"volume {tuple(volume.shape)} and mask {tuple(mask.shape)} must match"
        )
        raise ValueError(msg)
    # Per-element squared error, then average over (C, patch_len) per patch.
    per_patch_se = (pred - target).pow(2).mean(dim=(-1, -2))     # (B, P)
    # Bucket weights by per-patch volume (within the masked positions only).
    masked_volume = volume[mask]
    if masked_volume.numel() == 0:
        return per_patch_se.new_zeros(())
    low_thresh = torch.quantile(masked_volume, low_q)
    high_thresh = torch.quantile(masked_volume, high_q)
    w = torch.full_like(volume, base_w)
    w = torch.where(volume <= low_thresh, torch.full_like(w, low_w), w)
    w = torch.where(volume >= high_thresh, torch.full_like(w, high_w), w)
    weighted = (per_patch_se * w * mask).sum()
    norm = (w * mask).sum().clamp_min(1.0)
    return weighted / norm


def next_patch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Plain (unweighted) MSE for the next-patch-prediction head.

    Shapes:
      pred   : (B, P, C, patch_len)  — predictions at every position
      target : (B, P, C, patch_len)
      mask   : (B, P)                — True at positions to include
    """
    per_patch_se = (pred - target).pow(2).mean(dim=(-1, -2))     # (B, P)
    n = mask.sum().clamp_min(1.0)
    return (per_patch_se * mask).sum() / n


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between two batches of unit-norm projections.

    `z_a[i]` and `z_b[i]` are a positive pair; all other pairs in the
    batch are negatives. Returns a scalar — the average of the two
    directional NT-Xent losses.

    Both tensors must be ℓ2-normalized (the contrastive head ensures
    this); we re-normalize defensively in case a caller skips it.
    """
    if z_a.shape != z_b.shape or z_a.dim() != 2:
        msg = (
            f"z_a and z_b must be (B, D) and same shape; "
            f"got {tuple(z_a.shape)} vs {tuple(z_b.shape)}"
        )
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    B = z_a.size(0)
    if B < 2:
        msg = "InfoNCE needs batch size >= 2 for at least one negative pair"
        raise ValueError(msg)
    z_a = F.normalize(z_a, p=2, dim=-1)
    z_b = F.normalize(z_b, p=2, dim=-1)
    sim = (z_a @ z_b.T) / temperature                            # (B, B)
    targets = torch.arange(B, device=z_a.device)
    loss_a = F.cross_entropy(sim, targets)
    loss_b = F.cross_entropy(sim.T, targets)
    return 0.5 * (loss_a + loss_b)


# ----- Phase 3 supervised losses (PHASE3.md §6.1, §6.2) -----


# The five quantiles emitted by the quantile head per PHASE3.md §6.1.
PHASE3_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def pinball_loss(
    q_pred: torch.Tensor,
    y: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Pinball / quantile loss averaged over (N, K).

    Args:
        q_pred:    (N, K) predicted quantile values.
        y:         (N,) realized targets.
        quantiles: (K,) quantile levels in (0, 1).

    Returns: scalar mean loss across all N samples × K quantiles.

    Reference: Koenker & Bassett, 1978. The standard objective for the
    Phase-3 quantile head (`PHASE3.md` §6.1) and what conformal CQR
    (§7.2) consumes as the residual distribution.
    """
    if q_pred.dim() != 2:
        msg = f"q_pred must be (N, K), got shape {tuple(q_pred.shape)}"
        raise ValueError(msg)
    if y.dim() != 1 or y.shape[0] != q_pred.shape[0]:
        msg = f"y must be (N,) matching q_pred[0]; got y={tuple(y.shape)}, q_pred={tuple(q_pred.shape)}"
        raise ValueError(msg)
    if quantiles.dim() != 1 or quantiles.shape[0] != q_pred.shape[1]:
        msg = f"quantiles must be (K,) matching q_pred[1]; got {tuple(quantiles.shape)}"
        raise ValueError(msg)
    diff = y.unsqueeze(-1) - q_pred                    # (N, K)
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff)
    return loss.mean()


def listmle_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """ListMLE rank loss within a cross-section (Xia et al. 2008).

    Args:
        scores:  (N,) predicted scalar rank scores.
        targets: (N,) the true ranking signal (e.g., realized labels or
                 cross-sectional ranks). Used only for ordering.

    Returns: scalar negative log-likelihood of the true permutation
    under the Plackett-Luce model defined by `scores`.

    Implementation:
        1. Sort `scores` by descending `targets` (the "true" order
           we want to learn).
        2. Compute, for each position i in the sorted sequence, the
           log-sum-exp of `scores[i..]` (the suffix). The probability
           the top item at position i is correct under PL is
           `exp(scores[i]) / sum(exp(scores[i..]))`.
        3. Loss = mean over i of `logsumexp(scores[i..]) - scores[i]`.

    Implemented via reverse-cumulative log-sum-exp on the flipped
    sequence, which is numerically stable and O(N).
    """
    if scores.dim() != 1 or targets.dim() != 1 or scores.shape != targets.shape:
        msg = (
            f"scores and targets must be (N,) and same shape; "
            f"got {tuple(scores.shape)} vs {tuple(targets.shape)}"
        )
        raise ValueError(msg)
    n = scores.shape[0]
    if n < 2:
        # ListMLE on N=1 is degenerate (the sum has 1 term, loss = 0).
        return scores.new_zeros(())

    # Sort scores by descending target → the "true" top of the list comes first.
    sorted_idx = torch.argsort(targets, descending=True)
    sorted_scores = scores[sorted_idx]
    # `torch.logcumsumexp` on a tensor x = (x_0, x_1, ..., x_{n-1}) returns
    # (lse(x_0), lse(x_0..x_1), ..., lse(x_0..x_{n-1})). We want the SUFFIX
    # logsumexp at each i: lse(x_i..x_{n-1}). Compute by flipping, doing a
    # forward cumulative lse, and flipping back.
    flipped = torch.flip(sorted_scores, dims=[0])
    rev_lse = torch.flip(torch.logcumsumexp(flipped, dim=0), dims=[0])  # (N,)
    per_step = rev_lse - sorted_scores
    return per_step.mean()


# ----- Phase 3X — Focal loss for 3-way tradeable-candidate classifier -----


def focal_ce_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal cross-entropy loss for class-imbalanced classification.

    Reference: Lin et al. 2017 (RetinaNet). For c-way classification with
    severe class imbalance (our case: ~7.8% LONG / 84.8% neutral / 7.4%
    SHORT), focal loss down-weights easy examples (correct + confident)
    and focuses learning on hard ones.

        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
      logits:     `(B, C)` raw class scores.
      target_idx: `(B,)` integer class indices in `[0, C)`.
      alpha:      `(C,)` per-class weight. None ⇒ uniform. For imbalanced
                  classes use inverse frequency: `α_c = 1 / freq(c)` normalized.
      gamma:      focusing parameter. γ=0 reduces to vanilla CE. γ=2 is
                  the RetinaNet default. Higher γ = more aggressive
                  down-weighting of easy examples.
      reduction:  "mean" | "sum" | "none".
    """
    log_probs = F.log_softmax(logits, dim=-1)               # (B, C)
    probs = log_probs.exp()
    n_classes = logits.shape[-1]
    target_onehot = F.one_hot(target_idx, num_classes=n_classes).float()
    pt = (probs * target_onehot).sum(dim=-1)                # (B,) P(true class)
    log_pt = (log_probs * target_onehot).sum(dim=-1)        # (B,) log P(true class)
    focal_weight = (1.0 - pt).pow(gamma)
    loss_per = -focal_weight * log_pt                       # (B,)
    if alpha is not None:
        if alpha.shape != (n_classes,):
            msg = f"alpha must be shape ({n_classes},), got {tuple(alpha.shape)}"
            raise ValueError(msg)
        per_sample_alpha = alpha.to(logits.device).index_select(0, target_idx)
        loss_per = loss_per * per_sample_alpha
    if reduction == "mean":
        return loss_per.mean()
    if reduction == "sum":
        return loss_per.sum()
    return loss_per


def direct_pnl_loss(
    scores: torch.Tensor,
    realized_pnl: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    top_k: int | None = None,
    temperature: float = 1.0,
    entropy_coef: float = 0.0,
    uniform_mix: float = 0.0,
    baseline_subtract: bool = False,
) -> torch.Tensor:
    """Offline contextual-bandit policy gradient for cross-sectional ranking.

    Policy:    π(symbol) = softmax(scores / temperature)
    Reward:    realized_pnl per symbol (net of cost, slippage-realistic)
    Objective: maximize E_π[reward] = Σ_i π(i) · realized_pnl_i
    Loss:      -E_π[reward]   [+ regularizers]

    This is REINFORCE in expectation form — we have all rewards available
    (offline data), so no sampling.

    DESIGN FIX (2026-05-14): the old implementation used `torch.where(...-inf)`
    to mask non-top-K positions before softmax. That zeroed the gradient on
    every position outside the current top-K — a hard exploration dead-zone.
    With no rank-loss anchor (pure bandit setup), the model could never
    promote a stock from rank 11 → 10, causing mode collapse (PU2: win=0.81%).

    The new implementation always softmax-normalizes over the FULL valid set
    (gradient reaches every score), and masks the REWARD instead — outside
    the deployment top-K, realized_pnl is zeroed so the policy is only
    rewarded for trades it would actually take.

    Knobs:
      `top_k`: deployment top-K. If set, realized_pnl outside top-K is zeroed.
        top_k=0 or None ⇒ no deployment masking (reward all picks).
      `temperature`: softmax temperature. Higher = flatter distribution =
        more exploration / weaker gradient signal.
      `entropy_coef`: maximize H(π) to discourage one-hot collapse. Loss
        becomes `-E[r] - entropy_coef * H(π)`. Sensible scale: 0.01-0.05
        relative to log(N) ≈ 6.4 for N=600.
      `uniform_mix`: behaviour-policy smoothing — convex-combine policy π
        with uniform 1/N. Keeps gradient flowing into every symbol.
        Sensible: 0.05-0.20.
      `baseline_subtract`: subtract cross-section mean PnL from realized_pnl
        as a REINFORCE control-variate (unbiased gradient, lower variance).
    """
    if mask is not None:
        scores = scores[mask]
        realized_pnl = realized_pnl[mask]
    if scores.numel() < 2:
        return scores.sum() * 0.0
    if baseline_subtract:
        realized_pnl = realized_pnl - realized_pnl.mean()
    # Full softmax (no -inf gate) — preserves gradient on every position.
    log_pi = F.log_softmax(scores / temperature, dim=-1)
    pi = log_pi.exp()
    # Reward masking: zero out PnL outside deployment top-K (so the policy
    # is only optimized for trades it actually takes), but keep gradient
    # flow over all positions.
    if top_k is not None and 0 < top_k < scores.numel():
        topk_vals = torch.topk(scores, top_k)[0]
        threshold = topk_vals[-1].detach()
        deploy = (scores >= threshold).float()
        realized_pnl = realized_pnl * deploy
    # Behaviour-policy smoothing keeps non-policy-favoured stocks alive.
    if uniform_mix > 0.0:
        n = scores.numel()
        w = (1.0 - uniform_mix) * pi + uniform_mix / n
    else:
        w = pi
    loss = -(w * realized_pnl).sum()
    # Entropy bonus to discourage collapse.
    if entropy_coef > 0.0:
        entropy = -(pi * log_pi).sum()
        loss = loss - entropy_coef * entropy
    return loss


def listnet_loss(
    scores: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """Listwise ranking loss (ListNet — Cao et al. 2007).

    Aligns optimization with cross-sectional ranking metrics like
    precision@k, which only care about ORDER. Pointwise classification
    losses (focal CE, BCE) penalize calibration error which is irrelevant
    for top-k retrieval.

    Forms two probability distributions over the N items in one
    cross-section:
      P*(i) = softmax(soft_targets / T)   — "true" ordering
      P (i) = softmax(scores)             — predicted ordering
    Then minimizes cross-entropy `-Σ P*(i) log P(i)`.

    Args:
      scores:        `(N,)` raw model scores (e.g. logits for LONG class).
      soft_targets:  `(N,)` ranking targets (higher = better; e.g. realized
                     forward return).
      mask:          optional `(N,)` bool, True = include. Invalid rows are
                     dropped from both distributions.
      target_temperature: temperature applied to soft_targets before
                     softmax. Lower → sharper "true" distribution
                     (concentrate mass on a few top items).

    Returns scalar loss. If fewer than 2 valid items, returns zero
    (gradient-free) — single-item ranking is undefined.
    """
    if mask is not None:
        scores = scores[mask]
        soft_targets = soft_targets[mask]
    if scores.numel() < 2:
        return scores.new_zeros(())
    p_target = F.softmax(soft_targets / target_temperature, dim=-1)
    log_p_score = F.log_softmax(scores, dim=-1)
    return -(p_target * log_p_score).sum()
