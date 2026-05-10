"""Mint a fresh Kite Connect access token and update `.kite_cred.env`.

Kite Connect access tokens expire at 6:00 AM IST every day, so this helper
gets run once per session-day before any historical-data work. The flow is:

1. Print the Kite login URL pre-populated with your `KITE_API_KEY`.
2. You open it in a browser, log in, get redirected to a URL containing
   `request_token=<short-lived code>`.
3. You paste the `request_token` back (or the whole URL).
4. We exchange it for an `access_token` via `kiteconnect.generate_session`
   and atomically rewrite the `KITE_ACCESS_TOKEN=` line in `.kite_cred.env`.

Run:
    daity-kite-login
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
from rich.console import Console

from daity.utils.env import EnvError, load_env, load_kite
from daity.utils.logging import get_logger

log = get_logger(__name__)
console = Console()

ACCESS_TOKEN_LINE_RE = re.compile(r"^KITE_ACCESS_TOKEN\s*=.*$", re.MULTILINE)


def _extract_request_token(raw: str) -> str:
    """Accept either the raw token or the full redirect URL containing it."""
    raw = raw.strip()
    if not raw:
        msg = "Empty input."
        raise click.UsageError(msg)
    # If the user pasted the full URL, pull out request_token=... query param.
    m = re.search(r"[?&]request_token=([A-Za-z0-9_-]+)", raw)
    if m:
        return m.group(1)
    # Otherwise treat the whole input as the token (must be opaque alphanumerics).
    if not re.fullmatch(r"[A-Za-z0-9_-]{10,}", raw):
        msg = (
            f"Doesn't look like a request_token or login URL: {raw[:40]}...\n"
            "Paste the request_token query param, or the whole redirect URL."
        )
        raise click.UsageError(msg)
    return raw


def _update_cred_file(cred_path: Path, access_token: str) -> None:
    """Rewrite the `KITE_ACCESS_TOKEN=` line in `cred_path`. Append if missing.

    Atomic: write to a sibling temp file then rename, so a crash mid-write
    can't leave the cred file in a half-baked state.
    """
    if cred_path.exists():
        text = cred_path.read_text()
        if ACCESS_TOKEN_LINE_RE.search(text):
            new_text = ACCESS_TOKEN_LINE_RE.sub(f"KITE_ACCESS_TOKEN={access_token}", text)
        else:
            new_text = text.rstrip("\n") + f"\nKITE_ACCESS_TOKEN={access_token}\n"
    else:
        new_text = f"KITE_ACCESS_TOKEN={access_token}\n"
    tmp = cred_path.with_suffix(cred_path.suffix + ".tmp")
    tmp.write_text(new_text)
    tmp.chmod(0o600)
    tmp.replace(cred_path)


@click.command()
@click.option(
    "--cred-file",
    type=click.Path(path_type=Path),
    default=Path(".kite_cred.env"),
    show_default=True,
    help="Path to the credential file to update.",
)
@click.option(
    "--request-token",
    "request_token_arg",
    default=None,
    help="If provided, skip the prompt and exchange this token directly. "
         "Accepts the raw token or the full redirect URL.",
)
def main(cred_file: Path, request_token_arg: str | None) -> None:
    """Exchange a Kite request_token for an access_token and persist it."""
    try:
        load_env()
        cfg = load_kite(strict=False)
    except EnvError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    if not cfg.api_key:
        console.print("[red]KITE_API_KEY is not set.[/red] Drop it into .kite_cred.env first.")
        sys.exit(2)
    if not cfg.api_secret:
        console.print("[red]KITE_API_SECRET is not set.[/red] Required for the exchange step.")
        sys.exit(2)

    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={cfg.api_key}"
    console.print("[bold]1.[/bold] Open this URL in your browser and log in:\n")
    console.print(f"   [cyan]{login_url}[/cyan]\n")
    console.print("[bold]2.[/bold] After login you'll be redirected to a URL containing "
                  "[bold]request_token=...[/bold] in the query string.")
    console.print("   Copy that token (or the whole URL — both work).\n")

    if request_token_arg:
        request_token = _extract_request_token(request_token_arg)
    else:
        raw = click.prompt("Paste request_token or full redirect URL", type=str)
        request_token = _extract_request_token(raw)

    log.info("Exchanging request_token for access_token (KiteConnect.generate_session)")
    kite = KiteConnect(api_key=cfg.api_key)
    try:
        session = kite.generate_session(request_token, api_secret=cfg.api_secret)
    except KiteException as exc:
        console.print(f"[red]Exchange failed:[/red] {type(exc).__name__}: {exc}")
        console.print(
            "Common causes: request_token already used (one-time only), expired "
            "(short TTL), or wrong api_secret."
        )
        sys.exit(3)

    access_token = session["access_token"]
    user_id = session.get("user_id", "<unknown>")
    _update_cred_file(cred_file, access_token)
    console.print(
        f"[green]OK[/green] — minted access_token for [bold]{user_id}[/bold] "
        f"and updated [bold]{cred_file}[/bold]. Token is valid until ~06:00 IST tomorrow."
    )


if __name__ == "__main__":
    main()
