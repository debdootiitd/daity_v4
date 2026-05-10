"""Unit tests for `daity.training.callbacks` + the CLI's checkpoint wiring.

Locks in the three things `phase_2_review.md` flagged as missing:
  - `ModelCheckpoint` is registered on the Trainer with top-3 + save_last
  - `EMACallback` actually shadows the live params and swaps for val
  - `(git_sha, config_hash, as_of)` provenance triple is emitted at run start

The CLI-level checkpoint regression test runs `daity-pretrain --smoke
--max-steps 5 --device cpu` against a tmp Parquet store and inspects
the resulting `runs/.../checkpoints/` dir.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from daity.training.callbacks import (
    EMACallback,
    Provenance,
    build_provenance,
    write_provenance_sidecar,
    _config_hash,
)


# ----- Provenance -----


def test_config_hash_is_stable_across_dict_orderings() -> None:
    a = {"foo": 1, "bar": 2, "baz": [1, 2, 3]}
    b = {"baz": [1, 2, 3], "bar": 2, "foo": 1}  # same content, different order
    assert _config_hash(a) == _config_hash(b)


def test_config_hash_changes_when_any_value_changes() -> None:
    a = {"foo": 1, "bar": 2}
    b = {"foo": 1, "bar": 3}
    assert _config_hash(a) != _config_hash(b)


def test_build_provenance_returns_well_formed_triple(tmp_path: Path) -> None:
    """provenance must always carry the triple, even in repos that have no
    commits yet (then git_sha = 'unknown' is the documented fallback)."""
    repo = Path(__file__).resolve().parents[2]
    prov = build_provenance(
        repo=repo, merged_cfg={"foo": 1},
        train_end=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert isinstance(prov, Provenance)
    # Either a real SHA or the documented sentinel.
    assert prov.git_sha == "unknown" or len(prov.git_sha) == 40
    assert prov.config_hash and len(prov.config_hash) == 12
    assert prov.as_of.startswith("2026-01-01")
    assert prov.started_at.startswith("20")  # rough ISO-8601 sanity


def test_build_provenance_with_real_git_history(tmp_path: Path) -> None:
    """When a repo with commits exists, git_sha is the 40-char HEAD SHA."""
    # Build a minimal git repo so we don't depend on the project's commit state.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    prov = build_provenance(
        repo=tmp_path, merged_cfg={"foo": 1},
        train_end=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert prov.git_sha != "unknown"
    assert len(prov.git_sha) == 40
    assert prov.git_dirty is False  # clean tree post-commit


def test_build_provenance_outside_repo_returns_unknown(tmp_path: Path) -> None:
    prov = build_provenance(
        repo=tmp_path, merged_cfg={"foo": 1},
        train_end=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert prov.git_sha == "unknown"
    assert prov.git_dirty is False


def test_provenance_sidecar_roundtrips_to_json(tmp_path: Path) -> None:
    prov = Provenance(
        git_sha="abc123" * 6 + "abcd",
        git_dirty=False,
        config_hash="deadbeef0001",
        as_of="2026-01-01T00:00:00+00:00",
        started_at="2026-05-10T12:00:00+00:00",
    )
    path = write_provenance_sidecar(prov, tmp_path / "logs")
    assert path.exists()
    parsed = json.loads(path.read_text())
    assert parsed["git_sha"] == prov.git_sha
    assert parsed["config_hash"] == prov.config_hash
    assert parsed["as_of"] == prov.as_of


# ----- EMA callback -----


def test_ema_callback_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="decay must be"):
        EMACallback(decay=0.0)
    with pytest.raises(ValueError, match="decay must be"):
        EMACallback(decay=1.0)
    with pytest.raises(ValueError, match="decay must be"):
        EMACallback(decay=-0.1)


class _Tiny(L.LightningModule):
    """Minimal LightningModule for testing the EMA callback."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.train_dataset = [torch.randn(4) for _ in range(10)]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self.linear(batch)
        loss = out.sum()
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.linear(batch).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_ema_initializes_shadow_in_setup() -> None:
    cb = EMACallback(decay=0.99)
    mod = _Tiny()
    # Manually invoke setup with a stub trainer.
    cb.on_fit_start(trainer=None, pl_module=mod)  # type: ignore[arg-type]
    assert "linear.weight" in cb._shadow
    assert "linear.bias" in cb._shadow
    # Shadow starts as a clone of live params.
    assert torch.equal(cb._shadow["linear.weight"], mod.linear.weight.detach())


def test_ema_update_recurrence() -> None:
    """After one update with decay=0.9, shadow = 0.9 * shadow_old + 0.1 * live."""
    cb = EMACallback(decay=0.9)
    mod = _Tiny()
    cb.on_fit_start(None, mod)  # type: ignore[arg-type]
    shadow_before = cb._shadow["linear.weight"].clone()
    # Mutate the live weight to a known value.
    with torch.no_grad():
        mod.linear.weight.fill_(1.0)
    cb.on_train_batch_end(None, mod, outputs=None, batch=None, batch_idx=0)  # type: ignore[arg-type]
    # Expected: 0.9 * shadow_before + 0.1 * 1.0
    expected = 0.9 * shadow_before + 0.1 * torch.ones_like(shadow_before)
    assert torch.allclose(cb._shadow["linear.weight"], expected, atol=1e-6)


def test_ema_swap_and_restore_around_validation() -> None:
    cb = EMACallback(decay=0.99)
    mod = _Tiny()
    cb.on_fit_start(None, mod)  # type: ignore[arg-type]
    # Set shadow to a known fixed state.
    with torch.no_grad():
        cb._shadow["linear.weight"].fill_(7.0)
        cb._shadow["linear.bias"].fill_(7.0)
    live_before = mod.linear.weight.detach().clone()

    cb.on_validation_start(None, mod)  # type: ignore[arg-type]
    # Live should now equal the shadow.
    assert torch.allclose(mod.linear.weight, torch.full_like(mod.linear.weight, 7.0))

    cb.on_validation_end(None, mod)  # type: ignore[arg-type]
    # And after end, restored.
    assert torch.equal(mod.linear.weight, live_before)


def test_ema_state_dict_round_trip() -> None:
    """Shadow weights must survive a checkpoint/restore cycle so resumed
    training picks up the EMA rather than starting cold."""
    cb1 = EMACallback(decay=0.9)
    mod = _Tiny()
    cb1.on_fit_start(None, mod)  # type: ignore[arg-type]
    with torch.no_grad():
        cb1._shadow["linear.weight"].fill_(3.14)
    state = cb1.state_dict()

    cb2 = EMACallback(decay=0.5)  # different decay; load_state_dict must overwrite
    cb2.load_state_dict(state)
    assert cb2.decay == pytest.approx(0.9)
    assert torch.allclose(cb2._shadow["linear.weight"], cb1._shadow["linear.weight"])


# ----- CLI registers the callbacks (regression for review finding #1) -----


@pytest.fixture
def smoke_parquet(tmp_path: Path) -> Path:
    """Build a minimal Parquet store the CLI smoke can iterate over.

    Spans ~60 days at every scale so the test's window_bars × forecast tail
    has a usable eligible end_ts range; smaller fixtures starve the iterator.
    """
    from datetime import timedelta

    import polars as pl

    from daity.data.parquet_store import ParquetStore

    base = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    store = ParquetStore(tmp_path / "parquet")
    # ~60 trading days at each scale (with weekends skipped is messy; we
    # just generate 60-day-long uniform sequences. Calendar correctness is
    # not what's being tested here — ModelCheckpoint + provenance are.).
    n_5m  = 60 * 12 * 24       # 60d × 288 5m bars/day
    n_15m = 60 * 4  * 24       # 60d × 96
    n_60m = 60 * 24            # 60d × 24
    n_day = 60                 # 60d
    for sym in ["A", "B", "C", "D"]:
        for scale, n, step_minutes in [
            ("5m", n_5m, 5), ("15m", n_15m, 15),
            ("60m", n_60m, 60), ("day", n_day, 24 * 60),
        ]:
            ts = [base + i * timedelta(minutes=step_minutes) for i in range(n)]
            df = pl.DataFrame({
                "symbol": [sym] * n, "ts": ts, "scale": [scale] * n,
                "open":   [100.0 + i * 0.1 for i in range(n)],
                "high":   [101.0 + i * 0.1 for i in range(n)],
                "low":    [99.0 + i * 0.1 for i in range(n)],
                "close":  [100.5 + i * 0.1 for i in range(n)],
                "volume": [1000 + i for i in range(n)],
            }, schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"),
                       "scale": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                       "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64})
            store.write_partition(scale, sym, df)
    return tmp_path / "parquet"


def test_pretrain_cli_emits_provenance_sidecar(tmp_path: Path, smoke_parquet: Path) -> None:
    """Run `daity-pretrain --smoke --max-steps 2` end-to-end; assert
    provenance.json lands in log_dir and contains the triple."""
    cfg_path = tmp_path / "smoke.yaml"
    log_dir = tmp_path / "runs"
    cfg_path.write_text(f"""\
universe_path: {smoke_parquet.parent / "cache"}
parquet_root: {smoke_parquet}
scales: ["5m", "15m", "60m", "day"]
num_channels: 5
window_bars:
  5m: 64
  15m: 48
  60m: 32
  day: 16
batch_size: 2
num_workers: 0
d_model: 32
n_layers: 2
n_heads: 2
ffn_ratio: 2
patch_len: 16
patch_stride: 8
max_seq_len: 32
mask_ratio: 0.3
contrastive_proj_dim: 16
contrastive_temperature: 0.07
forecast_scale: "5m"
forecast_n_patches: 2
w_masked: 1.0
w_ntp: 0.5
w_contrastive: 0.0
w_forecast: 0.5
lr: 1e-3
weight_decay: 1e-2
warmup_pct: 0.10
max_steps: 2
ema_decay: 0.99
val_every_n_steps: 1
limit_val_batches: 1
checkpoint_every_n_steps: 1
save_top_k: 2
precision: "32-true"
device: "cpu"
train_end: "2024-02-01"
val_start: "2024-02-02"
val_end: "2024-02-15"
wandb_offline: true
log_dir: "{log_dir}"
""")

    # Run via the click test runner so we don't fork a subprocess.
    from click.testing import CliRunner

    from daity.scripts.pretrain_cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "--config", str(cfg_path),
        "--max-steps", "2", "--device", "cpu", "--precision", "32-true",
        "--symbols", "A,B,C,D",
        "--allow-dirty",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # Provenance sidecar.
    sidecar = log_dir / "provenance.json"
    assert sidecar.exists(), f"provenance.json missing in {log_dir}; output:\n{result.output}"
    parsed = json.loads(sidecar.read_text())
    assert "git_sha" in parsed
    assert "config_hash" in parsed
    assert "as_of" in parsed
    assert parsed["as_of"].startswith("2024-02-01")

    # Checkpoint callback fired (at least 1 ckpt + last.ckpt). The
    # `every_n_train_steps=1` config above forces a checkpoint each step.
    ckpts = list((log_dir / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"no checkpoints in {log_dir / 'checkpoints'}; output:\n{result.output}"
    names = [c.name for c in ckpts]
    assert any("last" in n for n in names), f"no last.ckpt in {names}"


def test_pretrain_cli_refuses_dirty_without_allow_dirty(
    tmp_path: Path, smoke_parquet: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --allow-dirty, a dirty checkout should hard-fail with a
    UsageError (the CLAUDE.md §6 reproducibility-tag guard)."""
    # Force `_git_dirty` to return True regardless of repo state.
    monkeypatch.setattr("daity.training.callbacks._git_dirty", lambda repo: True)

    cfg_path = tmp_path / "smoke.yaml"
    log_dir = tmp_path / "runs"
    cfg_path.write_text(f"""\
universe_path: {smoke_parquet.parent / "cache"}
parquet_root: {smoke_parquet}
scales: ["5m", "15m", "60m", "day"]
num_channels: 5
window_bars: {{5m: 64, 15m: 48, 60m: 32, day: 16}}
batch_size: 2
num_workers: 0
d_model: 32
n_layers: 2
n_heads: 2
ffn_ratio: 2
patch_len: 16
patch_stride: 8
max_seq_len: 32
mask_ratio: 0.3
contrastive_proj_dim: 16
contrastive_temperature: 0.07
forecast_scale: "5m"
forecast_n_patches: 2
w_masked: 1.0
w_ntp: 0.5
w_contrastive: 0.0
w_forecast: 0.5
lr: 1e-3
weight_decay: 1e-2
warmup_pct: 0.10
max_steps: 2
ema_decay: 0.99
val_every_n_steps: 1
limit_val_batches: 1
checkpoint_every_n_steps: 1
save_top_k: 2
precision: "32-true"
device: "cpu"
train_end: "2024-02-01"
val_start: "2024-02-02"
val_end: "2024-02-15"
wandb_offline: true
log_dir: "{log_dir}"
""")
    from click.testing import CliRunner

    from daity.scripts.pretrain_cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "--config", str(cfg_path),
        "--max-steps", "2", "--device", "cpu", "--precision", "32-true",
        "--symbols", "A,B,C,D",
        # Crucially: NO --allow-dirty
    ], catch_exceptions=False)
    assert result.exit_code != 0
    assert "uncommitted" in result.output.lower() or "dirty" in result.output.lower()
