"""Unit tests for daity.utils.env."""

from __future__ import annotations

from pathlib import Path

import pytest

from daity.utils.env import BQConfig, EnvError, GrowwConfig, load_env, load_groww


def test_bqconfig_fq_helpers() -> None:
    cfg = BQConfig(
        project="proj",
        dataset="ds",
        location="US",
        credentials_path=Path("/tmp/k.json"),
    )
    assert cfg.fq_dataset == "proj.ds"
    assert cfg.fq_table("t") == "proj.ds.t"


def test_load_env_strict_missing_raises(isolated_env: Path) -> None:  # noqa: ARG001
    with pytest.raises(EnvError):
        load_env(strict=True)


def test_load_env_non_strict_returns(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    cfg = load_env(strict=False)
    assert cfg.project == "p"
    assert cfg.dataset == "d"
    assert cfg.location is None  # auto-detect


def test_load_env_strict_with_creds(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed SA key passes strict validation."""
    import json

    fake_key = isolated_env / "fake_key.json"
    fake_key.write_text(
        json.dumps(
            {
                "type": "service_account",
                "client_email": "test@example.iam.gserviceaccount.com",
                "private_key_id": "abc123def456",
            }
        )
    )
    fake_key.chmod(0o600)
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_key))
    cfg = load_env(strict=True)
    assert cfg.credentials_path == fake_key.resolve()


def test_load_env_strict_rejects_bad_json(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_key = isolated_env / "bad.json"
    fake_key.write_text("not json")
    fake_key.chmod(0o600)
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_key))
    with pytest.raises(EnvError, match="not valid JSON"):
        load_env(strict=True)


def test_load_env_strict_rejects_non_service_account(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    fake_key = isolated_env / "user.json"
    fake_key.write_text(json.dumps({"type": "authorized_user"}))
    fake_key.chmod(0o600)
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_key))
    with pytest.raises(EnvError, match="service-account"):
        load_env(strict=True)


def test_load_env_location_unset_means_none(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When DAITY_BQ_LOCATION is empty/unset, BQConfig.location is None for auto-detect."""
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    monkeypatch.delenv("DAITY_BQ_LOCATION", raising=False)
    cfg = load_env(strict=False)
    assert cfg.location is None


# ----- Sidecar files + alias normalization -----


def test_grow_alias_mirrors_to_canonical(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar with the typo'd `GROW_*` spelling should populate `GROWW_*` env vars
    so consumers reading the canonical name see the value."""
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    (isolated_env / ".grow_cred1.env").write_text(
        "GROW_API_KEY=newkey\nGROW_API_SECRET=newsecret\n"
    )
    (isolated_env / ".grow_cred1.env").chmod(0o600)
    load_env(strict=False)
    g = load_groww(strict=False)
    assert g.api_key == "newkey"
    assert g.api_secret == "newsecret"


def test_numbered_sidecar_overrides_unnumbered(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both `.grow_cred.env` and `.grow_cred1.env` exist, the numbered file wins."""
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    (isolated_env / ".grow_cred.env").write_text("GROWW_API_KEY=oldkey\n")
    (isolated_env / ".grow_cred.env").chmod(0o600)
    (isolated_env / ".grow_cred1.env").write_text("GROW_API_KEY=newkey\n")
    (isolated_env / ".grow_cred1.env").chmod(0o600)
    load_env(strict=False)
    g = load_groww(strict=False)
    assert g.api_key == "newkey", (
        "numbered sidecar should win over base, regardless of GROW vs GROWW spelling"
    )


def test_load_groww_strict_raises_when_unset(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ARG001
    """load_groww(strict=True) must raise if no key is set (after load_env)."""
    monkeypatch.setenv("DAITY_BQ_PROJECT", "p")
    monkeypatch.setenv("DAITY_BQ_DATASET", "d")
    monkeypatch.delenv("GROWW_API_KEY", raising=False)
    monkeypatch.delenv("GROW_API_KEY", raising=False)
    load_env(strict=False)
    with pytest.raises(EnvError, match="GROWW_API_KEY"):
        load_groww(strict=True)


def test_groww_config_jwt_detection() -> None:
    """`key_is_access_token` flips True for JWT-shaped keys (eyJ prefix)."""
    assert GrowwConfig(api_key="eyJabc", api_secret=None, totp_secret=None).key_is_access_token
    assert not GrowwConfig(api_key="short-key", api_secret=None, totp_secret=None).key_is_access_token
