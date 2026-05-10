"""Project logger.

Thin wrapper around the standard library `logging` plus `rich` for colorized
console output. Use `get_logger(__name__)` at the top of every module.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from rich.logging import RichHandler

_CONFIGURED: Final[dict[str, bool]] = {"done": False}


def _configure_root() -> None:
    if _CONFIGURED["done"]:
        return
    level = os.environ.get("DAITY_LOG_LEVEL", "INFO").upper()
    handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_time=True,
        show_path=False,
        log_time_format="%H:%M:%S",
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    # Quiet down chatty libs
    for noisy in ("google.api_core", "google.auth", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED["done"] = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger by name."""
    _configure_root()
    return logging.getLogger(name)
