"""Logging configuration.

The console stays quiet -- warnings and above -- because the CLI's own output is
the user interface and a stream of INFO lines would bury it. The rotating file log
keeps the detail, which is what ``meetscribe doctor`` and a bug report need.

The redaction filter is attached to the root logger *and* to every handler before
anything else is configured, so there is no window in which a secret could be
written. That ordering is the whole point: a filter added after the first log call
is a filter that already failed once.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from meetscribe.security.paths import logs_dir
from meetscribe.security.redact import RedactingFilter

__all__ = ["setup_logging", "LOG_FILENAME"]

LOG_FILENAME = "meetscribe.log"

_CONSOLE_FORMAT = "%(levelname)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"


def setup_logging(
    level: str = "INFO",
    console_level: str = "WARNING",
    log_dir: Path | None = None,
    quiet: bool = False,
) -> Path | None:
    """Configure the root logger. Returns the log file path, if one was opened."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    redactor = RedactingFilter()
    root.addFilter(redactor)

    console = logging.StreamHandler()
    console.setLevel(
        logging.CRITICAL if quiet else getattr(logging, console_level.upper(), logging.WARNING)
    )
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    console.addFilter(redactor)
    root.addHandler(console)

    path: Path | None = None
    try:
        directory = log_dir or logs_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOG_FILENAME
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=4_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)
    except OSError:
        # A read-only or missing data directory must not stop a meeting from being
        # recorded; the console handler is already in place.
        path = None

    # These libraries are extremely chatty at INFO and say nothing this project
    # needs; their warnings still come through.
    for noisy in ("httpx", "httpcore", "anthropic", "urllib3", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return path
