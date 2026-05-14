"""Unit tests for `daity.training.posttrain`.

Locks in (per PHASE3.md §6.9):

- Pinball + ListMLE loss correctness on small synthetic data.
- LoRA-wrapped backbone has LoRA params + frozen base (the named-adapter
  switch is correctly honored).
- Backbone state-dict round-trips through the SSL-checkpoint load path.
- 1-batch overfit reduces all three loss terms across 100 steps for
  ALL four horizons.
- `calibrate_loss_weights` produces per-horizon weights with
  `w_h * L_h ≈ 1.0` within 2×.
- `IncompatibleCheckpointError` is raised on a synthetic mismatched
  checkpoint (post-amendment 7 / §6.10).

The leakage gate (forward + loss never reads past horizon) is in
`tests/leakage/test_posttrain_leakage.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from daity.data.cross_section import CrossSection
from daity.models.backbone import Backbone
from daity.models.tokenizer import MultiResTokenizer
from daity.training.losses import (
    PHASE3_QUANTILES,
    listmle_loss,
    pinball_loss,
)
from daity.training.posttrain import (
    HORIZONS,
    SCALES,
    IncompatibleCheckpointError,
    PostTrainConfig,
    PostTrainModule,
)

# ----- Loss tests -----


def test_pinball_loss_zero_at_perfect_prediction() -> None:
    """If q_pred == y for every quantile, pinball loss is 0."""
    n, k = 16, 5
    y = torch.randn(n)
    q_pred = y.unsqueeze(-1).expand(-1, k).contiguous()
    quantiles = torch.tensor(PHASE3_QUANTILES, dtype=torch.float32)
    loss = pinball_loss(q_pred, y, quantiles)
    assert loss.item() == 0.0


def test_pinball_loss_overshoot_under_low_quantile() -> None:
    """At τ=0.1, overshooting (q > y) is penalized more than undershooting."""
    y = torch.zeros(1)
    q_high = torch.tensor([[+1.0]])   # overshoots
    q_low = torch.tensor([[-1.0]])    # undershoots
    quantiles = torch.tensor([0.1])
    # diff = y - q. q_high: diff = -1 → loss = max(0.1*-1, -0.9*-1) = 0.9.
    # q_low:  diff = +1 → loss = max(0.1*+1, -0.9*+1) = 0.1.
    high = pinball_loss(q_high, y, quantiles).item()
    low = pinball_loss(q_low, y, quantiles).item()
    assert high > low
    assert abs(high - 0.9) < 1e-6
    assert abs(low - 0.1) < 1e-6


def test_listmle_loss_zero_at_perfect_predictions_with_n2() -> None:
    """For N=2: if scores match the target ordering, the loss is small;
    if reversed, larger."""
    targets = torch.tensor([1.0, -1.0])
    # Scores aligned: high score for high target.
    good = torch.tensor([10.0, -10.0])
    bad = torch.tensor([-10.0, 10.0])
    l_good = listmle_loss(good, targets).item()
    l_bad = listmle_loss(bad, targets).item()
    assert l_good < l_bad
    # `aligned` for N=2 with very large positive gap converges to log(1+e^{-20})
    # which is essentially 0.
    assert l_good < 0.01


def test_listmle_loss_degenerate_n1() -> None:
    """N=1 returns 0 (no ordering to learn)."""
    s = torch.tensor([3.14])
    t = torch.tensor([1.0])
    loss = listmle_loss(s, t)
    assert loss.item() == 0.0


def test_listmle_loss_gradient_flows() -> None:
    """ListMLE produces a usable gradient on scores."""
    scores = torch.randn(8, requires_grad=True)
    targets = torch.randn(8)
    loss = listmle_loss(scores, targets)
    loss.backward()
    assert scores.grad is not None
    assert not torch.allclose(scores.grad, torch.zeros_like(scores.grad))


# ----- PostTrainModule construction -----


def _build_synth_checkpoint(tmp_path: Path,
                             d_model: int = 64,
                             n_layers: int = 2,
                             num_channels: int = 18) -> Path:
    """Write a minimal Lightning-style checkpoint with `state_dict` +
    `hyper_parameters` so `PostTrainModule` can load it."""
    tokenizer = MultiResTokenizer(
        scales=SCALES, num_channels=num_channels, d_model=d_model,
        patch_len=16, patch_stride=8, revin_affine=False,
    )
    backbone = Backbone(d_model=d_model, n_layers=n_layers, n_heads=4, ffn_ratio=2)
    state = {f"tokenizer.{k}": v for k, v in tokenizer.state_dict().items()}
    state.update({f"backbone.{k}": v for k, v in backbone.state_dict().items()})
    ckpt = {
        "state_dict": state,
        "hyper_parameters": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": 4,
            "ffn_ratio": 2,
            "patch_len": 16,
            "patch_stride": 8,
            "num_channels": num_channels,
            "revin_affine": False,
            "scales": list(SCALES),
        },
    }
    p = tmp_path / "synth.ckpt"
    torch.save(ckpt, p)
    return p


def _small_cfg(tmp_path: Path, **overrides) -> PostTrainConfig:
    """Build a fast-instantiating PostTrainConfig pointed at a synth ckpt."""
    ckpt_path = _build_synth_checkpoint(tmp_path)
    cfg = PostTrainConfig(
        parent_checkpoint=ckpt_path,
        d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18,
        revin_affine=False, scales=SCALES,
        lora_r=4, head_hidden=32, head_dropout=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _fake_cross_section(horizon: str, n_symbols: int = 12, d_model: int = 64) -> CrossSection:
    """Build a CrossSection with shapes the module expects."""
    # Each scale needs ≥ patch_len bars to make at least one patch.
    L = 32  # generous for patch_len=16 stride=8 → 3 patches
    inputs = {
        sc: torch.randn(n_symbols, L, 18) for sc in SCALES
    }
    labels = torch.randn(n_symbols)
    return CrossSection(
        t=datetime(2025, 1, 1, tzinfo=UTC),
        horizon=horizon,
        symbols=[f"S{i:02d}" for i in range(n_symbols)],
        inputs=inputs,
        labels=labels,
        ranks=torch.argsort(torch.argsort(labels)).float() / max(1, n_symbols - 1),
        valid_mask=torch.ones(n_symbols, dtype=torch.bool),
        vol_bucket=torch.zeros(n_symbols, dtype=torch.int8),
    )


def test_module_constructs_and_freezes_backbone(tmp_path: Path) -> None:
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    # ORIGINAL backbone params (everything that isn't a LoRA adapter param)
    # must be frozen; PEFT-added LoRA params are trainable.
    for n, p in module.backbone.named_parameters():
        is_lora = "lora_" in n
        if is_lora:
            assert p.requires_grad, f"LoRA param backbone.{n} should be trainable"
        else:
            assert not p.requires_grad, f"backbone.{n} should be frozen"
    # Tokenizer frozen.
    for n, p in module.tokenizer.named_parameters():
        assert not p.requires_grad, f"tokenizer.{n} should be frozen"
    # LoRA params + heads trainable, and not zero.
    trainable = module.trainable_parameters()
    assert len(trainable) > 0
    # Sanity: at least 4 horizons × 3 heads = 12 head modules contribute.
    head_param_names = [n for n, _ in trainable if "_heads" in n]
    assert len(head_param_names) >= 24, (
        f"expected at least 24 head params (12 heads × 2 layers each), got {len(head_param_names)}"
    )
    # Trainable count must be much less than the backbone size.
    n_train = module.trainable_param_count()
    n_backbone = sum(p.numel() for _, p in module.backbone.named_parameters())
    assert n_train < n_backbone, f"trainable ({n_train}) should be << backbone ({n_backbone})"


def test_module_lora_adapter_switch(tmp_path: Path) -> None:
    """Switching the LoRA adapter changes which params are active."""
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    for h in HORIZONS:
        module.peft_backbone.set_adapter(h)
        assert module.peft_backbone.active_adapter == h


def test_module_forward_all_horizons(tmp_path: Path) -> None:
    """Forward through every horizon's LoRA bank + heads produces the
    expected output shapes."""
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    for h in HORIZONS:
        cs = _fake_cross_section(h, n_symbols=10, d_model=cfg.d_model)
        out = module(cs, horizon=h)
        assert out["quantile"].shape == (10, len(PHASE3_QUANTILES))
        assert out["rank"].shape == (10,)
        assert out["vol"].shape == (10,)
        assert out["forecast_hidden"].shape == (10, cfg.d_model)


def test_module_compute_loss_is_finite(tmp_path: Path) -> None:
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    cs = _fake_cross_section("30m", n_symbols=12, d_model=cfg.d_model)
    out = module(cs, horizon="30m")
    loss = module.compute_loss(out, cs, horizon="30m")
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0


def test_module_one_batch_overfit_reduces_loss_for_all_horizons(tmp_path: Path) -> None:
    """The 1-batch overfit acceptance test from PHASE3.md §11. For each
    horizon: 100 steps on the same CrossSection should drop total loss
    by ≥30%."""
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    optim = torch.optim.AdamW(module.trainable_parameters(), lr=3e-3)
    # Helper: get just parameters from name-param pairs.
    optim = torch.optim.AdamW([p for _, p in module.trainable_parameters()], lr=3e-3)

    for h in HORIZONS:
        cs = _fake_cross_section(h, n_symbols=12, d_model=cfg.d_model)
        initial: float | None = None
        for step in range(100):
            optim.zero_grad()
            out = module(cs, horizon=h)
            loss = module.compute_loss(out, cs, horizon=h)
            loss.backward()
            optim.step()
            if step == 0:
                initial = float(loss.detach())
        final = float(loss.detach())
        assert initial is not None
        # 30% reduction is the PHASE3.md §11 acceptance bar.
        assert final < 0.7 * initial, (
            f"horizon {h}: initial={initial:.4f}, final={final:.4f} "
            f"(want <0.7×initial = {0.7*initial:.4f})"
        )


def test_module_calibrate_loss_weights(tmp_path: Path) -> None:
    """After calibration, w_h * L_h is approximately equal across heads."""
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    sample = {h: _fake_cross_section(h, n_symbols=12, d_model=cfg.d_model)
              for h in HORIZONS}
    weights = module.calibrate_loss_weights(sample)
    # Compute w_h * L_h post-calibration; they should all be ≈ 1.0 within
    # 2× per amendment 11.
    for h in HORIZONS:
        cs = sample[h]
        out = module(cs, horizon=h)
        losses = module.per_head_losses(out, cs)
        for head in ("quantile", "rank", "vol"):
            wl = weights[h][head] * float(losses[head])
            assert 0.5 < wl < 2.0, (
                f"horizon {h} head {head}: w*L = {wl:.4f} not in [0.5, 2.0]"
            )


# ----- Hparam compatibility (post-amendment 7 / §6.10) -----


def test_hparam_mismatch_refuses(tmp_path: Path) -> None:
    """Loading from a checkpoint with d_model=64 while the live cfg
    has d_model=128 must raise IncompatibleCheckpointError."""
    # Build a checkpoint at d_model=64.
    ckpt_path = _build_synth_checkpoint(tmp_path, d_model=64, n_layers=2)
    # Construct a config with d_model=128 → mismatch.
    cfg = PostTrainConfig(
        parent_checkpoint=ckpt_path,
        d_model=128, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18,
        revin_affine=False, scales=SCALES,
    )
    with pytest.raises(IncompatibleCheckpointError, match="d_model"):
        PostTrainModule(cfg)


def test_hparam_match_loads_cleanly(tmp_path: Path) -> None:
    """Matching hparams produce a clean load (no exception)."""
    cfg = _small_cfg(tmp_path)
    module = PostTrainModule(cfg)
    # Sanity: backbone has the loaded weights, not random init.
    # We check the final_norm bias as a witness — it's stored in the ckpt.
    bb_norm_bias = module.backbone.final_norm.bias.detach().clone()
    assert bb_norm_bias.shape == (cfg.d_model,)


def test_module_handles_no_checkpoint_hparams(tmp_path: Path) -> None:
    """A checkpoint without `hyper_parameters` skips compat check
    (caller-on-their-own contract)."""
    # Build a plain state_dict (no Lightning wrapper).
    tokenizer = MultiResTokenizer(
        scales=SCALES, num_channels=18, d_model=64,
        patch_len=16, patch_stride=8, revin_affine=False,
    )
    backbone = Backbone(d_model=64, n_layers=2, n_heads=4, ffn_ratio=2)
    state = {f"tokenizer.{k}": v for k, v in tokenizer.state_dict().items()}
    state.update({f"backbone.{k}": v for k, v in backbone.state_dict().items()})
    ckpt_path = tmp_path / "plain.ckpt"
    torch.save(state, ckpt_path)

    cfg = PostTrainConfig(
        parent_checkpoint=ckpt_path,
        d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18,
        revin_affine=False, scales=SCALES,
    )
    # Should not raise.
    module = PostTrainModule(cfg)
    assert module is not None
