"""Unit tests for the Kite login helper.

The interactive prompt + KiteConnect.generate_session call are not exercised
here; we cover the two pure helpers that hold the actual logic: request-token
extraction (accepts raw token, full URL, or query-only fragment) and the
atomic credential-file rewrite.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from daity.scripts.kite_login import _extract_request_token, _update_cred_file


def test_extract_raw_token() -> None:
    assert _extract_request_token("abc123XYZ_-") == "abc123XYZ_-"


def test_extract_from_full_redirect_url() -> None:
    url = "https://my-app.example.com/callback?action=login&request_token=tk_4ABCdef&type=login"
    assert _extract_request_token(url) == "tk_4ABCdef"


def test_extract_from_query_fragment_only() -> None:
    assert _extract_request_token("?request_token=tk_xyz_99&status=success") == "tk_xyz_99"


def test_extract_strips_whitespace() -> None:
    # Real Kite request_tokens are 30+ chars; the regex enforces a 10-char floor
    # so accidental short strings are rejected as garbage.
    assert _extract_request_token("  \nabc123XYZ_-pq \t") == "abc123XYZ_-pq"


def test_extract_rejects_empty() -> None:
    with pytest.raises(click.UsageError):
        _extract_request_token("")
    with pytest.raises(click.UsageError):
        _extract_request_token("   ")


def test_extract_rejects_garbage() -> None:
    # No `request_token=` in URL and not a plain alphanumeric token.
    with pytest.raises(click.UsageError):
        _extract_request_token("https://example.com/foo?bar=baz")


def test_update_cred_file_replaces_existing_line(tmp_path: Path) -> None:
    cred = tmp_path / ".kite_cred.env"
    cred.write_text(
        "KITE_API_KEY=apikey\n"
        "KITE_API_SECRET=secret\n"
        "KITE_ACCESS_TOKEN=stale_token_value\n"
    )
    _update_cred_file(cred, "fresh_token")
    text = cred.read_text()
    assert "KITE_API_KEY=apikey" in text
    assert "KITE_API_SECRET=secret" in text
    assert "KITE_ACCESS_TOKEN=fresh_token" in text
    assert "stale_token_value" not in text


def test_update_cred_file_appends_when_missing(tmp_path: Path) -> None:
    cred = tmp_path / ".kite_cred.env"
    cred.write_text("KITE_API_KEY=apikey\nKITE_API_SECRET=secret\n")
    _update_cred_file(cred, "first_token")
    text = cred.read_text()
    assert "KITE_ACCESS_TOKEN=first_token" in text
    # Existing lines preserved.
    assert "KITE_API_KEY=apikey" in text
    assert "KITE_API_SECRET=secret" in text


def test_update_cred_file_creates_new(tmp_path: Path) -> None:
    cred = tmp_path / ".kite_cred.env"
    assert not cred.exists()
    _update_cred_file(cred, "tok")
    assert cred.read_text() == "KITE_ACCESS_TOKEN=tok\n"
    # File created with mode 0600 (owner-only) at write time. We re-stat to confirm.
    assert (cred.stat().st_mode & 0o777) == 0o600


def test_update_cred_file_is_atomic(tmp_path: Path) -> None:
    """The temp file must be cleaned up after a successful rename."""
    cred = tmp_path / ".kite_cred.env"
    cred.write_text("KITE_API_KEY=k\n")
    _update_cred_file(cred, "tok")
    # No leftover .tmp sibling.
    leftovers = list(tmp_path.glob(".kite_cred.env.tmp"))
    assert leftovers == []
