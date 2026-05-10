"""Unit tests for `daity.training.pretrain.PretrainModule` + the SSL wiring.

Targets the load-bearing invariants: forward shape, all 3 losses are
finite + non-zero on a fresh model, gradient flows end-to-end through
every component, optimizer step decreases loss on a synthetic batch.

These tests run on CPU with a tiny config. The full A100 / Vast smoke
tests live in the writeup notebook.
"""

from __future__ import annotations

import pytest
import torch

from daity.data.datasets import FORECAST_FUTURE_KEY
from daity.data.tokenizer_targets import build_forecast_targets, build_targets
from daity.models.tokenizer import MultiResTokenizer
from daity.training.pretrain import PretrainConfig, PretrainModule


def _tiny_cfg(*, with_forecast: bool = False) -> PretrainConfig:
    """A tiny config that runs in <1s on CPU: 2 layers, d_model=64.

    When `with_forecast=True`, enables the multi-horizon forecast head
    with 2 patches × patch_len=16 = 32 future bars at 5m.
    """
    return PretrainConfig(
        scales=("5m", "day"),
        num_channels=5,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_ratio=2,
        patch_len=16,
        patch_stride=8,
        max_seq_len=64,
        mask_ratio=0.4,
        contrastive_proj_dim=32,
        forecast_scale="5m",
        forecast_n_patches=2 if with_forecast else 0,
        max_steps=50,
        lr=1e-3,
    )


def _batch(B: int = 4, *, with_forecast: bool = False) -> dict[str, torch.Tensor]:
    """Synthetic batch matching tiny_cfg's scales + window sizes."""
    torch.manual_seed(0)
    out: dict[str, torch.Tensor] = {
        "5m": torch.randn(B, 64, 5) * 5.0 + 100.0,
        "day": torch.randn(B, 32, 5) * 5.0 + 100.0,
    }
    if with_forecast:
        # 2 patches × 16 bars = 32 future bars at 5m.
        out[FORECAST_FUTURE_KEY] = torch.randn(B, 32, 5) * 5.0 + 100.0
    return out


# ----- Construction -----


def test_module_constructs_with_default_config() -> None:
    cfg = _tiny_cfg()
    mod = PretrainModule(cfg)
    # Sanity: hyperparameters survived save_hyperparameters.
    assert mod.hparams["d_model"] == cfg.d_model
    assert mod.hparams["mask_ratio"] == cfg.mask_ratio


def test_module_param_count_includes_mask_token() -> None:
    """The learnable [MASK] embedding must be in the parameter list so the
    optimizer updates it (otherwise reconstruction can't learn from
    masked positions)."""
    mod = PretrainModule(_tiny_cfg())
    names = {n for n, _ in mod.named_parameters()}
    assert "mask_token" in names


# ----- Forward + loss -----


def test_compute_loss_returns_finite_scalar_and_three_parts() -> None:
    mod = PretrainModule(_tiny_cfg())
    batch = _batch()
    loss, parts = mod._compute_loss(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert {"loss_recon", "loss_ntp", "loss_contrast"} <= parts.keys()
    for name, val in parts.items():
        assert torch.isfinite(val), f"non-finite {name}: {val}"


def test_compute_loss_each_objective_nonzero_at_init() -> None:
    """A randomly-initialized model should have non-trivial loss on each
    *active* objective. With forecast disabled (default `_tiny_cfg`),
    `loss_forecast` is a zero sentinel — that's expected, not a failure.
    """
    mod = PretrainModule(_tiny_cfg())
    _, parts = mod._compute_loss(_batch())
    # Forecast head off → loss_forecast is the zero sentinel.
    assert parts["loss_forecast"].item() == 0.0
    # Active objectives must be non-trivial at init.
    for name in ("loss_recon", "loss_ntp", "loss_contrast"):
        assert parts[name].item() > 0.0, f"{name} is zero at init"


def test_compute_loss_gradient_reaches_every_component() -> None:
    """Backward should produce nonzero gradients in tokenizer, backbone,
    each head, and the mask_token. If any of those misses gradient, the
    loss-weighting is broken."""
    mod = PretrainModule(_tiny_cfg())
    loss, _ = mod._compute_loss(_batch())
    loss.backward()
    # Group params by module prefix and check at least one has a non-trivial grad.
    groups: dict[str, list[float]] = {}
    for name, p in mod.named_parameters():
        if p.grad is None:
            continue
        prefix = name.split(".")[0]
        groups.setdefault(prefix, []).append(p.grad.abs().sum().item())
    expected_groups = {
        "tokenizer", "backbone", "recon_head", "ntp_head",
        "contrastive_head", "mask_token",
    }
    missing = expected_groups - groups.keys()
    assert not missing, f"no gradient reached: {missing}"
    # Each group's total grad magnitude should be > 0.
    for g, mags in groups.items():
        assert sum(mags) > 0, f"group {g} has all-zero gradients"


# ----- Optimizer + LR schedule -----


def test_configure_optimizers_returns_lambdalr_with_warmup() -> None:
    cfg = _tiny_cfg()
    cfg.max_steps = 1000
    cfg.warmup_pct = 0.10           # 100 warmup steps so the test has headroom
    mod = PretrainModule(cfg)
    res = mod.configure_optimizers()
    assert "optimizer" in res
    assert "lr_scheduler" in res
    sched = res["lr_scheduler"]["scheduler"]
    # LambdaLR applies the multiplier at construction. lr_lambda(0) = 0, so
    # the LR starts AT zero and ramps up during warmup.
    initial_lr = res["optimizer"].param_groups[0]["lr"]
    assert initial_lr == pytest.approx(0.0, abs=1e-12)
    # Step partway into warmup; LR should be > 0 but < peak.
    for _ in range(20):
        sched.step()
    in_warmup_lr = res["optimizer"].param_groups[0]["lr"]
    assert 0.0 < in_warmup_lr < cfg.lr


def test_lr_schedule_with_min_ratio_floor_does_not_decay_to_zero() -> None:
    """Phase 2.2 long-run feature — `lr_min_ratio > 0` floors the cosine
    schedule at `lr_min_ratio × peak` instead of decaying to zero. Default
    behavior (`lr_min_ratio=0.0`) is unchanged.
    """
    cfg = _tiny_cfg()
    cfg.max_steps = 100
    cfg.warmup_pct = 0.10        # 10 warmup steps
    cfg.lr_min_ratio = 0.1       # cosine floors at 10% of peak
    mod = PretrainModule(cfg)
    res = mod.configure_optimizers()
    opt, sched = res["optimizer"], res["lr_scheduler"]["scheduler"]

    # Step through warmup (10 steps) → LR == peak.
    for _ in range(10):
        sched.step()
    lr_peak = opt.param_groups[0]["lr"]
    assert lr_peak == pytest.approx(cfg.lr, rel=1e-3)

    # Step to end of training → LR should be lr_min_ratio × peak (~ 0.1 × peak).
    for _ in range(95):  # well past max_steps
        sched.step()
    lr_end = opt.param_groups[0]["lr"]
    expected_floor = cfg.lr * cfg.lr_min_ratio
    assert lr_end == pytest.approx(expected_floor, rel=1e-3), (
        f"expected LR floor {expected_floor:.6e}, got {lr_end:.6e}"
    )


def test_lr_schedule_reaches_target_lr_after_warmup_then_decays() -> None:
    """After warmup, LR should equal cfg.lr; after that it cosines down."""
    cfg = _tiny_cfg()
    cfg.max_steps = 100
    cfg.warmup_pct = 0.10  # 10 warmup steps
    mod = PretrainModule(cfg)
    res = mod.configure_optimizers()
    opt, sched = res["optimizer"], res["lr_scheduler"]["scheduler"]

    # Step through warmup (10 steps) then post-warmup peek.
    for _ in range(10):
        sched.step()
    lr_post_warmup = opt.param_groups[0]["lr"]
    assert lr_post_warmup == pytest.approx(cfg.lr, rel=1e-3)

    # Step to halfway through post-warmup → lr ~ 0.5 of peak (cosine).
    for _ in range(45):
        sched.step()
    lr_mid = opt.param_groups[0]["lr"]
    assert lr_mid < lr_post_warmup
    assert lr_mid > 0


# ----- One full training step decreases loss on a fixed batch -----


def test_overfit_one_batch_reduces_loss() -> None:
    """The acid test: enough optimizer steps on the SAME batch should drop
    the *running* loss meaningfully. We compare the average loss over the
    last 10 steps vs the first 10 — robust to step-to-step mask noise.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.lr = 5e-3              # bigger lr for the smoke
    cfg.mask_ratio = 0.3       # less stochastic
    mod = PretrainModule(cfg)
    batch = _batch()
    opt = torch.optim.AdamW(mod.parameters(), lr=cfg.lr)

    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss, _ = mod._compute_loss(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < 0.7 * early, (
        f"loss did not decrease enough: avg first-10={early:.4f} → "
        f"avg last-10={late:.4f}"
    )


def test_target_builder_volume_matches_input_volume() -> None:
    """`build_targets` extracts per-patch mean volume from channel index 4
    (OHLCV). Verify the number is correct on a hand-built batch."""
    cfg = _tiny_cfg()
    tokenizer = MultiResTokenizer(
        scales=cfg.scales, num_channels=cfg.num_channels,
        d_model=cfg.d_model,
        patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
    )
    # All-zero except volume = 1.0..L on the day scale.
    B, L_5m, L_day, C = 1, 64, 32, 5
    five_m = torch.zeros(B, L_5m, C)
    day = torch.zeros(B, L_day, C)
    # Volume column for day: 1, 2, ..., 32.
    day[..., 4] = torch.arange(1, L_day + 1, dtype=torch.float32)
    targets, volumes = build_targets(
        batch={"5m": five_m, "day": day},
        scales=cfg.scales, patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
        tokenizer=tokenizer,
    )
    # Day scale is the second batch in the concat. 5m has (64-16)/8+1 = 7 patches;
    # day has (32-16)/8+1 = 3 patches.
    n_5m = 7
    # Day patches' volumes are means of consecutive 16-bar windows of [1..32].
    # Patch 0: mean(1..16) = 8.5; patch 1: mean(9..24) = 16.5; patch 2: mean(17..32) = 24.5.
    day_vols = volumes[0, n_5m:].tolist()
    assert day_vols == pytest.approx([8.5, 16.5, 24.5], rel=1e-5)


# ----- Multi-horizon forecast head wiring -----


def test_forecast_head_disabled_when_n_patches_zero() -> None:
    """forecast_n_patches=0 should leave forecast_head as None and not
    inflate the param count beyond the 3-loss model."""
    cfg = _tiny_cfg(with_forecast=False)
    mod = PretrainModule(cfg)
    assert mod.forecast_head is None


def test_forecast_head_present_when_enabled() -> None:
    cfg = _tiny_cfg(with_forecast=True)
    mod = PretrainModule(cfg)
    assert mod.forecast_head is not None


def test_compute_loss_includes_forecast_term_when_enabled() -> None:
    cfg = _tiny_cfg(with_forecast=True)
    mod = PretrainModule(cfg)
    batch = _batch(with_forecast=True)
    loss, parts = mod._compute_loss(batch)
    assert torch.isfinite(loss)
    assert "loss_forecast" in parts
    assert parts["loss_forecast"].item() > 0.0
    # Disabling the head zeroes the term.
    cfg_off = _tiny_cfg(with_forecast=False)
    mod_off = PretrainModule(cfg_off)
    _, parts_off = mod_off._compute_loss(_batch(with_forecast=False))
    assert parts_off["loss_forecast"].item() == 0.0


def test_compute_loss_no_forecast_key_uses_zero_loss() -> None:
    """If the head is enabled but the batch doesn't carry future bars
    (e.g. validation-only paths), loss_forecast should be zero rather
    than crash."""
    cfg = _tiny_cfg(with_forecast=True)
    mod = PretrainModule(cfg)
    # Note: NO forecast key in the batch.
    loss, parts = mod._compute_loss(_batch(with_forecast=False))
    assert parts["loss_forecast"].item() == 0.0
    assert torch.isfinite(loss)


def test_forecast_gradient_reaches_forecast_head() -> None:
    """When forecast SSL is enabled, backward through the total loss must
    leave nonzero gradients on the forecast head's parameters."""
    cfg = _tiny_cfg(with_forecast=True)
    mod = PretrainModule(cfg)
    loss, _ = mod._compute_loss(_batch(with_forecast=True))
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in mod.forecast_head.parameters()
    )
    assert has_grad


def test_log_return_input_and_target_train_end_to_end() -> None:
    """Phase 2.3 — `input_form=log_returns` + `target_form=log_returns` paths.

    Locks in: the model trains forward+backward without device/shape errors
    when both forms are switched on. Loss decreases on a fixed batch over
    100 steps (same gate as the absolute-form overfit test).
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg(with_forecast=True)
    cfg.input_form = "log_returns"
    cfg.target_form = "log_returns"
    cfg.lr = 5e-3
    cfg.mask_ratio = 0.3
    mod = PretrainModule(cfg)
    # Use synthetic OHLCV that's positive (real prices); the log-return
    # transform clamps tiny values but won't tolerate negatives correctly.
    torch.manual_seed(42)
    B = 4
    batch = {
        "5m": torch.rand(B, 64, 5) * 10 + 100,    # OHLCV prices ~100-110, vol ~5
        "day": torch.rand(B, 32, 5) * 10 + 100,
        FORECAST_FUTURE_KEY: torch.rand(B, 32, 5) * 10 + 100,
    }
    opt = torch.optim.AdamW(mod.parameters(), lr=cfg.lr)

    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss, _ = mod._compute_loss(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert all(np.isfinite(v) for v in losses), "log-return path produced non-finite loss"
    assert late < 0.7 * early, (
        f"log-return path didn't reduce loss: early={early:.4f}, late={late:.4f}"
    )


def test_log_return_target_without_input_transform_raises_clean_error_or_works() -> None:
    """`target_form=log_returns` independent of `input_form` — must work,
    just less natural (model has to learn to predict log-returns from raw
    inputs)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(with_forecast=True)
    cfg.input_form = "raw"          # raw input
    cfg.target_form = "log_returns" # log-return targets
    mod = PretrainModule(cfg)
    torch.manual_seed(42)
    B = 4
    batch = {
        "5m": torch.rand(B, 64, 5) * 10 + 100,
        "day": torch.rand(B, 32, 5) * 10 + 100,
        FORECAST_FUTURE_KEY: torch.rand(B, 32, 5) * 10 + 100,
    }
    loss, parts = mod._compute_loss(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(parts["loss_forecast"])


import numpy as np  # noqa: E402  (used by the new test above)


def test_revin_targets_do_not_collapse_during_overfit() -> None:
    """Phase 2.1.1 regression — RevIN affine collapse pathology.

    With `revin_affine=True`, training would shrink gamma toward 0 over
    many steps because that shrinks the targets and trivially drives MSE
    to ~0 (validated on the H200 50K-step run: gamma collapsed to ~0.04,
    val_loss hit 1e-8). With `affine=False` (the new default), the
    targets stay in their natural standardized scale (std ≈ 1) regardless
    of how long we train.

    This test asserts: after 200 steps of overfit on a fixed batch, the
    recon-target std stays > 0.5 (not the collapse-driven < 0.1). Run
    with the default `revin_affine=False`.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.lr = 5e-3
    cfg.weight_decay = 1e-2  # the L2 reg that ACCELERATES the collapse
    mod = PretrainModule(cfg)
    batch = _batch()
    opt = torch.optim.AdamW(mod.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    for _ in range(200):
        opt.zero_grad()
        loss, _ = mod._compute_loss(batch)
        loss.backward()
        opt.step()
    # Build targets fresh after training and check their std.
    targets, _ = mod._build_targets(batch)
    target_std = targets.std().item()
    assert target_std > 0.5, (
        f"recon target std collapsed to {target_std:.4f} after 200 overfit steps. "
        f"This is the RevIN affine pathology (DESIGN amendment 12). "
        f"Verify cfg.revin_affine is False; check tokenizer.revins[scale].affine."
    )


def test_overfit_one_batch_reduces_forecast_loss() -> None:
    """The forecast loss should drop materially across 100 steps on a
    fixed batch — the head can fit a single (input, future) pair."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(with_forecast=True)
    cfg.lr = 5e-3
    cfg.mask_ratio = 0.3
    mod = PretrainModule(cfg)
    batch = _batch(with_forecast=True)
    opt = torch.optim.AdamW(mod.parameters(), lr=cfg.lr)

    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss_full, parts = mod._compute_loss(batch)
        loss_full.backward()
        opt.step()
        losses.append(parts["loss_forecast"].item())

    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < 0.7 * early, (
        f"forecast loss didn't drop enough: avg first-10={early:.4f} → "
        f"avg last-10={late:.4f}"
    )


def test_build_forecast_targets_uses_input_revin_frame() -> None:
    """Forecast targets must be normalized using the tokenizer's cached
    RevIN stats from the INPUT window — not from the future bars
    themselves. Verify by checking that wildly-different future bars
    (mean 1000 vs input mean 100) produce non-zero-mean targets."""
    cfg = _tiny_cfg()
    tokenizer = MultiResTokenizer(
        scales=cfg.scales, num_channels=cfg.num_channels,
        d_model=cfg.d_model,
        patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
    )
    # Input window: mean 100, normal noise.
    B, C = 2, 5
    input_5m = torch.randn(B, 64, C) * 5.0 + 100.0
    input_day = torch.randn(B, 32, C) * 5.0 + 100.0
    # Run the tokenizer to populate the RevIN cache.
    _ = tokenizer({"5m": input_5m, "day": input_day})
    # Future bars at much higher level — if RevIN recomputed, they'd
    # normalize to mean ~0; using cached input stats they should be
    # FAR from zero (since std ~5 → (1000-100)/5 ~= 180).
    future = torch.full((B, 32, C), 1000.0)
    targets = build_forecast_targets(
        future_bars=future, forecast_scale="5m",
        n_patches=2, patch_len=16, tokenizer=tokenizer,
    )
    assert targets.shape == (B, 2, C, 16)
    # Targets should be nowhere near zero — using cached input-window stats,
    # 1000-bar future shifted by mean-100 / std-5 is huge.
    assert targets.abs().mean() > 10.0


def test_build_forecast_targets_rejects_wrong_length() -> None:
    cfg = _tiny_cfg()
    tokenizer = MultiResTokenizer(
        scales=cfg.scales, num_channels=cfg.num_channels,
        d_model=cfg.d_model,
        patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
    )
    _ = tokenizer({"5m": torch.randn(1, 64, 5), "day": torch.randn(1, 32, 5)})
    with pytest.raises(ValueError, match="length .* doesn't match"):
        build_forecast_targets(
            future_bars=torch.randn(1, 31, 5),  # not 2*16=32
            forecast_scale="5m", n_patches=2, patch_len=16,
            tokenizer=tokenizer,
        )


def test_build_forecast_targets_rejects_unknown_scale() -> None:
    cfg = _tiny_cfg()
    tokenizer = MultiResTokenizer(
        scales=cfg.scales, num_channels=cfg.num_channels,
        d_model=cfg.d_model,
        patch_len=cfg.patch_len, patch_stride=cfg.patch_stride,
    )
    _ = tokenizer({"5m": torch.randn(1, 64, 5), "day": torch.randn(1, 32, 5)})
    with pytest.raises(ValueError, match="not in tokenizer.revins"):
        build_forecast_targets(
            future_bars=torch.randn(1, 32, 5),
            forecast_scale="60m",  # not in scales
            n_patches=2, patch_len=16, tokenizer=tokenizer,
        )
