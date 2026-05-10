"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test in a clean directory with a scrubbed env.

    Removes daity-specific env vars so each test sets them explicitly.
    """
    monkeypatch.chdir(tmp_path)
    for k in (
        "DAITY_BQ_PROJECT",
        "DAITY_BQ_DATASET",
        "DAITY_BQ_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "DAITY_LOG_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)
    # Make sure no parent .env interferes (load_env reads cwd/.env).
    assert not (tmp_path / ".env").exists()
    return tmp_path
