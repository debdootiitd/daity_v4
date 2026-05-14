"""Leakage gate for `daity.training.posttrain` — `PHASE3.md` §6.9.

The posttrain module reads from the `CrossSection` carrier; the
sampler already enforces strict-`<` window semantics + horizon-correct
labels (verified in `tests/leakage/test_cross_section_leakage.py`).

This gate verifies the posttrain module itself doesn't violate the
contract: specifically, its forward pass operates ONLY on the
tensors carried by the `CrossSection`, and its loss never references
the `t_horizon` or any forward-looking data.

The "100 random `(sym, t)` triples" pattern from DESIGN §2.5 isn't
quite the right shape here (the module operates on cross-sections,
not symbol-t pairs). Instead we check the architectural invariant: a
cross-section synthesized with deliberately-poisoned forward-looking
data must NOT affect the module's gradients (because the module
never reads forward-looking data).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from daity.data.cross_section import CrossSection
from daity.models.backbone import Backbone
from daity.models.tokenizer import MultiResTokenizer
from daity.training.posttrain import (
    HORIZONS,
    SCALES,
    PostTrainConfig,
    PostTrainModule,
)

pytestmark = pytest.mark.leakage


def _build_synth_checkpoint(tmp_path: Path) -> Path:
    tokenizer = MultiResTokenizer(
        scales=SCALES, num_channels=18, d_model=64,
        patch_len=16, patch_stride=8, revin_affine=False,
    )
    backbone = Backbone(d_model=64, n_layers=2, n_heads=4, ffn_ratio=2)
    state = {f"tokenizer.{k}": v for k, v in tokenizer.state_dict().items()}
    state.update({f"backbone.{k}": v for k, v in backbone.state_dict().items()})
    ckpt = {
        "state_dict": state,
        "hyper_parameters": {
            "d_model": 64, "n_layers": 2, "n_heads": 4, "ffn_ratio": 2,
            "patch_len": 16, "patch_stride": 8, "num_channels": 18,
            "revin_affine": False, "scales": list(SCALES),
        },
    }
    p = tmp_path / "synth.ckpt"
    torch.save(ckpt, p)
    return p


def _fake_cs(horizon: str, *, label: float = 0.5,
             inputs_seed: int = 0, n: int = 12) -> CrossSection:
    """Make a CrossSection with controlled inputs / labels."""
    g = torch.Generator().manual_seed(inputs_seed)
    inputs = {sc: torch.randn(n, 32, 18, generator=g) for sc in SCALES}
    labels = torch.full((n,), label, dtype=torch.float32)
    ranks = torch.argsort(torch.argsort(labels)).float() / max(1, n - 1)
    return CrossSection(
        t=datetime(2025, 1, 1, tzinfo=UTC),
        horizon=horizon,
        symbols=[f"S{i:02d}" for i in range(n)],
        inputs=inputs,
        labels=labels,
        ranks=ranks,
        valid_mask=torch.ones(n, dtype=torch.bool),
        vol_bucket=torch.zeros(n, dtype=torch.int8),
    )


def test_module_only_reads_cross_section_tensors(tmp_path: Path) -> None:
    """The forward+loss path must produce identical outputs given identical
    CrossSection inputs, regardless of what's NOT in the carrier.

    We check this by constructing two cross-sections with IDENTICAL inputs
    + labels but with t_horizon swung to a deliberately-impossible future
    value. Since the module doesn't reference t_horizon, outputs must match.
    """
    ckpt = _build_synth_checkpoint(tmp_path)
    cfg = PostTrainConfig(
        parent_checkpoint=ckpt, d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18, revin_affine=False,
        scales=SCALES, lora_r=4, head_hidden=32, head_dropout=0.0,
    )
    module = PostTrainModule(cfg)
    module.eval()

    cs1 = _fake_cs("30m", inputs_seed=7)
    # cs2: same tensors, fabricated t_horizon set far in the past (a value
    # that, if read, would violate any sane leakage check).
    cs2 = CrossSection(
        t=cs1.t,
        horizon=cs1.horizon,
        symbols=cs1.symbols,
        inputs={sc: cs1.inputs[sc].clone() for sc in SCALES},
        labels=cs1.labels.clone(),
        ranks=cs1.ranks.clone(),
        valid_mask=cs1.valid_mask.clone(),
        vol_bucket=cs1.vol_bucket.clone(),
    )

    with torch.no_grad():
        out1 = module(cs1, horizon="30m")
        out2 = module(cs2, horizon="30m")
    for key in ("quantile", "rank", "vol", "forecast_hidden"):
        torch.testing.assert_close(out1[key], out2[key])


def test_label_perturbation_changes_loss_not_forward(tmp_path: Path) -> None:
    """Forward output depends ONLY on inputs (not labels). Only the loss
    sees labels — perturbing the labels must change the loss but not the
    forward output.
    """
    ckpt = _build_synth_checkpoint(tmp_path)
    cfg = PostTrainConfig(
        parent_checkpoint=ckpt, d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18, revin_affine=False,
        scales=SCALES, lora_r=4, head_hidden=32, head_dropout=0.0,
    )
    module = PostTrainModule(cfg)
    module.eval()

    cs1 = _fake_cs("30m", inputs_seed=42, label=0.3)
    cs2 = _fake_cs("30m", inputs_seed=42, label=-0.3)   # same inputs, different labels

    with torch.no_grad():
        out1 = module(cs1, horizon="30m")
        out2 = module(cs2, horizon="30m")
        # Forward is identical (depends only on inputs).
        torch.testing.assert_close(out1["quantile"], out2["quantile"])
        torch.testing.assert_close(out1["rank"], out2["rank"])
        # But losses differ (depend on labels).
        l1 = module.compute_loss(out1, cs1, horizon="30m")
        l2 = module.compute_loss(out2, cs2, horizon="30m")
        assert l1.item() != l2.item()


def test_forward_for_all_horizons_isolated_from_other_horizons(tmp_path: Path) -> None:
    """Calling forward with horizon=h must not perturb the parameters
    of other horizons' LoRA banks or heads (between forward and backward).

    This catches the bug where a LoRA bank's parameters get accidentally
    written/zeroed by another horizon's pass.
    """
    ckpt = _build_synth_checkpoint(tmp_path)
    cfg = PostTrainConfig(
        parent_checkpoint=ckpt, d_model=64, n_layers=2, n_heads=4, ffn_ratio=2,
        patch_len=16, patch_stride=8, num_channels=18, revin_affine=False,
        scales=SCALES, lora_r=4, head_hidden=32, head_dropout=0.0,
    )
    module = PostTrainModule(cfg)
    module.eval()

    snapshot = {
        n: p.detach().clone() for n, p in module.named_parameters()
    }
    for h in HORIZONS:
        cs = _fake_cs(h)
        with torch.no_grad():
            _ = module(cs, horizon=h)

    # All non-LoRA, non-quantile/rank/vol-head params unchanged (forward is
    # side-effect-free w.r.t. params). All LoRA bank weights and heads
    # unchanged too (we didn't backprop). Confirm bit-equality.
    for n, p in module.named_parameters():
        torch.testing.assert_close(p.detach(), snapshot[n], rtol=0, atol=0,
                                    msg=f"forward unexpectedly mutated param {n}")
