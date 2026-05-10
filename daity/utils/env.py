"""Environment + BQ config loading.

Loads `.env` (if present), validates required vars, and surfaces a typed
`BQConfig`. Never logs the credentials path verbatim — only confirms it exists.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from daity.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BQConfig:
    """Resolved BigQuery configuration."""

    project: str
    dataset: str
    location: str | None  # `None` triggers auto-detection in BQClient
    credentials_path: Path

    @property
    def fq_dataset(self) -> str:
        """Fully-qualified dataset id, e.g. `proj.dataset`."""
        return f"{self.project}.{self.dataset}"

    def fq_table(self, table: str) -> str:
        """Fully-qualified table id, e.g. `proj.dataset.table`."""
        return f"{self.project}.{self.dataset}.{table}"


@dataclass(frozen=True, slots=True)
class KiteConfig:
    """Resolved Zerodha Kite Connect configuration.

    Kite uses an OAuth-style flow: `api_key` + `api_secret` are static, but
    `access_token` is daily-rotating (expires at 6:00 AM IST next day). The
    typical pattern is to mint a fresh token via the login redirect flow once
    per day and paste it into the sidecar file.
    """

    api_key: str
    api_secret: str | None
    access_token: str | None


@dataclass(frozen=True, slots=True)
class GrowwConfig:
    """Resolved Groww trading-API configuration.

    `key_is_access_token` flips True when `GROWW_API_KEY` looks like a JWT
    (starts with `eyJ`). In that case the SDK can be initialized directly with
    the token and we never need `secret` for auth — kept around in case a
    future refresh path needs it.
    """

    api_key: str
    api_secret: str | None
    totp_secret: str | None

    @property
    def key_is_access_token(self) -> bool:
        return self.api_key.startswith("eyJ")


class EnvError(RuntimeError):
    """Raised when required environment variables are missing or invalid."""


# Optional secret-only sidecar files we auto-merge into the environment.
# Loaded in order, with `override=True`, so a later file in the list wins over
# both earlier files and the primary `.env`. This lets you keep multiple
# numbered credential files (e.g. `.grow_cred.env`, `.grow_cred1.env`) and
# the highest-numbered one is what actually gets used.
SIDECAR_ENV_FILES: tuple[str, ...] = (
    ".grow_cred.env",
    ".grow_cred1.env",
    ".kite_cred.env",
)


# Alias normalization: when a sidecar uses a non-canonical variable name
# (commonly a typo), we mirror the value to the canonical name as we load.
# Without this, two sidecars using *different* spellings of the same secret
# leave both names in env, and the consumer has to guess which one is fresh.
# Mapping is `non_canonical_prefix -> canonical_prefix`. Match is on prefix
# so `GROW_API_KEY` mirrors to `GROWW_API_KEY`, `GROW_API_SECRET` to
# `GROWW_API_SECRET`, etc.
ENV_ALIAS_PREFIXES: dict[str, str] = {
    "GROW_": "GROWW_",  # single-W typo for Groww
}


def _validate_sa_key(path: Path) -> None:
    """Defense-in-depth checks on a service-account JSON key file.

    Raises `EnvError` for clearly-broken files; logs a warning for risky perms.
    """
    # Permission check (POSIX). On macOS the default umask 022 leaves keys
    # group/world-readable; we only warn so the dev's first run is not blocked.
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        msg = f"Could not stat {path}: {exc}"
        raise EnvError(msg) from exc
    if mode & 0o077:
        log.warning(
            "SA key %s has loose perms %o; chmod 600 recommended", path, mode
        )

    # Content check: must be a valid JSON service-account key.
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        msg = f"{path} is not valid JSON: {exc}"
        raise EnvError(msg) from exc
    if not isinstance(payload, dict) or payload.get("type") != "service_account":
        msg = (
            f"{path} does not look like a service-account key (expected "
            f"top-level JSON object with type='service_account')."
        )
        raise EnvError(msg)
    # Surface the identity (NOT the private key) for traceability.
    key_id = payload.get("private_key_id") or "<unknown>"
    log.info(
        "SA key resolved: type=%s client=%s key_id=%s",
        payload.get("type"),
        payload.get("client_email") or "<unknown>",
        key_id[:12] + "...",
    )


def load_env(env_file: Path | str | None = None, *, strict: bool = True) -> BQConfig:
    """Load `.env` and return a validated `BQConfig`.

    Args:
        env_file: optional path to a `.env` file. If `None`, looks for `.env` in cwd.
        strict: if True, raises on missing creds; if False, returns config with a
            non-existent credentials_path (useful for tests / dry-runs).
    """
    if env_file is None:
        env_file = Path.cwd() / ".env"
    env_path = Path(env_file).expanduser()
    if env_path.exists():
        # Ensure .env itself is not loose-permed.
        env_mode = env_path.stat().st_mode & 0o777
        if env_mode & 0o077 and stat.S_ISREG(env_path.stat().st_mode):
            log.warning(".env has loose perms %o; chmod 600 recommended", env_mode)
        load_dotenv(env_path, override=False)
        log.info("Loaded env from %s", env_path)
    else:
        log.warning("No .env at %s; relying on process environment", env_path)

    # Auto-merge any secret-only sidecar files (e.g. broker creds), in the
    # order declared in SIDECAR_ENV_FILES. Later files win — the convention is
    # that numbered files (e.g. `.grow_cred1.env`) supersede the unnumbered
    # base. We don't use `load_dotenv(override=True)` directly because we also
    # need to mirror non-canonical variable names (see ENV_ALIAS_PREFIXES) so
    # sidecars with typo'd names still take effect on later reads.
    for sidecar_name in SIDECAR_ENV_FILES:
        sidecar = env_path.parent / sidecar_name
        if not sidecar.exists():
            continue
        sc_mode = sidecar.stat().st_mode & 0o777
        if sc_mode & 0o077 and stat.S_ISREG(sidecar.stat().st_mode):
            log.warning("%s has loose perms %o; chmod 600 recommended", sidecar.name, sc_mode)
        for k, v in (dotenv_values(sidecar) or {}).items():
            if v is None:
                continue
            os.environ[k] = v
            for src_prefix, canonical_prefix in ENV_ALIAS_PREFIXES.items():
                if k.startswith(src_prefix) and not k.startswith(canonical_prefix):
                    os.environ[canonical_prefix + k[len(src_prefix):]] = v
        log.info("Loaded sidecar env from %s", sidecar)

    project = os.environ.get("DAITY_BQ_PROJECT")
    dataset = os.environ.get("DAITY_BQ_DATASET")
    location = os.environ.get("DAITY_BQ_LOCATION") or None  # empty string -> None
    cred_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    missing = [
        name
        for name, val in [
            ("DAITY_BQ_PROJECT", project),
            ("DAITY_BQ_DATASET", dataset),
            ("GOOGLE_APPLICATION_CREDENTIALS", cred_str),
        ]
        if not val
    ]
    if missing:
        msg = f"Missing required env vars: {', '.join(missing)}. See .env.example."
        if strict:
            raise EnvError(msg)
        log.warning(msg)

    cred_path = Path(cred_str).expanduser().resolve() if cred_str else Path("/nonexistent")
    if strict and not cred_path.exists():
        raise EnvError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to {cred_path} which does not exist."
        )
    if cred_path.exists():
        if strict:
            _validate_sa_key(cred_path)
        else:
            log.info("Using BQ credentials at %s (%d bytes)", cred_path, cred_path.stat().st_size)
        # Make sure downstream google libs see the resolved absolute path.
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)

    return BQConfig(
        project=project or "",
        dataset=dataset or "",
        location=location,
        credentials_path=cred_path,
    )


def load_groww(*, strict: bool = True) -> GrowwConfig:
    """Return a `GrowwConfig` from the current environment.

    Assumes `load_env()` has already been called (so `.env` and any sidecar
    files like `.grow_cred.env` have been merged, and any non-canonical
    spellings like `GROW_API_KEY` have been mirrored to canonical names by
    `ENV_ALIAS_PREFIXES`).
    """
    api_key = (os.environ.get("GROWW_API_KEY") or "").strip()
    api_secret = (os.environ.get("GROWW_API_SECRET") or "").strip() or None
    totp_secret = (os.environ.get("GROWW_TOTP_SECRET") or "").strip() or None

    if not api_key:
        msg = "GROWW_API_KEY not set. See .env.example or drop a .grow_cred.env sidecar."
        if strict:
            raise EnvError(msg)
        log.warning(msg)

    cfg = GrowwConfig(api_key=api_key, api_secret=api_secret, totp_secret=totp_secret)
    # Log the auth shape *only* — never the key itself, even truncated.
    log.info(
        "Groww config: key_is_access_token=%s secret=%s totp=%s",
        cfg.key_is_access_token,
        bool(api_secret),
        bool(totp_secret),
    )
    return cfg


def load_kite(*, strict: bool = True) -> KiteConfig:
    """Return a `KiteConfig` from the current environment.

    Assumes `load_env()` has already been called (so `.env` and any sidecar
    files like `.kite_cred.env` have been merged).
    """
    api_key = (os.environ.get("KITE_API_KEY") or "").strip()
    api_secret = (os.environ.get("KITE_API_SECRET") or "").strip() or None
    access_token = (os.environ.get("KITE_ACCESS_TOKEN") or "").strip() or None

    missing = [name for name, val in [("KITE_API_KEY", api_key)] if not val]
    if missing:
        msg = f"Missing required Kite env vars: {missing}. See .env.example or drop a .kite_cred.env sidecar."
        if strict:
            raise EnvError(msg)
        log.warning(msg)

    cfg = KiteConfig(api_key=api_key, api_secret=api_secret, access_token=access_token)
    # Log shape only — never the secret or token itself.
    log.info(
        "Kite config: has_secret=%s has_access_token=%s",
        bool(api_secret),
        bool(access_token),
    )
    return cfg
