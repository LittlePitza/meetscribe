"""Where the sensitive artifacts live, and who can read them.

Two decisions here are load-bearing.

**The data directory is never inside the project folder.** This repository lives
under OneDrive. Writing recordings and transcripts next to the source would upload
every meeting the user has ever had to Microsoft's cloud -- the precise outcome the
tool exists to avoid -- and it would happen silently, as a side effect of where
someone happened to clone the code. Data goes to ``%LOCALAPPDATA%\\MeetScribe``,
which is machine-local and not synchronised.

**The directory is ACL-hardened on creation.** By default a directory under
``%LOCALAPPDATA%`` is readable by other administrators on the machine. Transcripts
of private conversations should not be. Inheritance is broken and access is granted
to exactly three principals: the current user, SYSTEM, and the local
Administrators group (which cannot meaningfully be excluded on Windows anyway, as
an administrator can always take ownership -- pretending otherwise would be
security theatre).

Hardening is best-effort by design: on a filesystem that does not support ACLs, or
in a locked-down environment where ``icacls`` is unavailable, the application still
runs and ``meetscribe doctor`` reports the directory as unprotected. Refusing to
start would trade a real capability for a warning the user cannot act on.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "APP_NAME",
    "data_dir",
    "config_path",
    "db_path",
    "models_dir",
    "audio_dir",
    "exports_dir",
    "logs_dir",
    "ensure_data_dirs",
    "cloud_sync_root",
    "harden_directory",
    "describe_protection",
    "current_user_sid",
]

APP_NAME = "MeetScribe"

_SYNC_FOLDER_NAMES = frozenset(
    {"onedrive", "dropbox", "google drive", "googledrive", "icloud drive", "box"}
)

_SYSTEM_SID = "*S-1-5-18"
_ADMINISTRATORS_SID = "*S-1-5-32-544"


def data_dir() -> Path:
    """Root of everything the app persists.

    ``MEETSCRIBE_DATA_DIR`` overrides it, which is what the test suite uses so no
    test ever touches the user's real transcripts.
    """
    override = os.environ.get("MEETSCRIBE_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    return Path(
        os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    ) / "meetscribe"


def config_path() -> Path:
    return data_dir() / "config.yaml"


def db_path() -> Path:
    return data_dir() / "meetscribe.db"


def models_dir() -> Path:
    return data_dir() / "models"


def audio_dir() -> Path:
    return data_dir() / "audio"


def exports_dir() -> Path:
    return data_dir() / "exports"


def logs_dir() -> Path:
    return data_dir() / "logs"


def cloud_sync_root(path: Path) -> Path | None:
    """The cloud-sync folder containing ``path``, or ``None`` if there is none.

    Storing transcripts inside a synchronised folder uploads every meeting to a
    third party, which is the one outcome this tool exists to prevent, so the
    check has to be right in both directions.

    The environment is asked first: Windows exports ``OneDrive`` (and
    ``OneDriveCommercial`` / ``OneDriveConsumer`` when both are signed in) pointing
    at the real sync roots, which is authoritative. Only when none of those
    contains the path does it fall back to inspecting the path's own *components* --
    ``OneDrive`` exactly, or the ``OneDrive - Contoso`` form that business tenants
    use, plus the other common desktop clients.

    Matching whole components rather than the joined string is the point. A
    substring test reports a false positive for any unrelated directory whose name
    merely happens to contain the word -- a temp path keyed by project name, for
    instance -- and telling someone their data directory is unsafe when it is not
    trains them to ignore the warning that matters.
    """
    try:
        resolved = path.expanduser().resolve()
    except OSError:  # pragma: no cover - unresolvable path, nothing to compare
        return None

    for variable in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        value = os.environ.get(variable)
        if not value:
            continue
        try:
            root = Path(value).expanduser().resolve()
        except OSError:  # pragma: no cover - a stale environment entry
            continue
        if root == resolved or root in resolved.parents:
            return root

    for index, part in enumerate(resolved.parts):
        if _is_sync_component(part):
            return Path(*resolved.parts[: index + 1])
    return None


def _is_sync_component(part: str) -> bool:
    lowered = part.lower()
    if lowered in _SYNC_FOLDER_NAMES:
        return True
    # Business tenants name the root after the organisation: "OneDrive - Contoso".
    return any(lowered.startswith(f"{name} - ") for name in _SYNC_FOLDER_NAMES)


def ensure_data_dirs(harden: bool = True) -> Path:
    """Create the tree and lock it down. Idempotent and cheap enough to call often."""
    root = data_dir()
    fresh = not root.exists()
    for path in (root, models_dir(), audio_dir(), exports_dir(), logs_dir()):
        path.mkdir(parents=True, exist_ok=True)
    if harden and fresh:
        harden_directory(root)
    return root


def current_user_sid() -> str | None:
    """The running account's SID, which is stable across renames and localisation."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        log.debug("whoami failed: %s", exc)
        return None
    parts = [p.strip().strip('"') for p in out.stdout.strip().split(",")]
    if len(parts) >= 2 and parts[-1].startswith("S-1-"):
        return parts[-1]
    return None


def harden_directory(path: Path) -> bool:
    """Break inheritance and grant access to the current user only.

    Returns True when the ACL was applied. Never raises: a failure here is
    reported by ``doctor``, not allowed to prevent a meeting from being recorded.
    """
    if sys.platform != "win32":
        try:
            path.chmod(0o700)
            return True
        except OSError as exc:  # pragma: no cover - platform dependent
            log.warning("chmod failed on %s: %s", path, exc)
            return False

    principal = current_user_sid()
    grant = f"*{principal}" if principal else os.environ.get("USERNAME", "")
    if not grant:
        log.warning("cannot determine current user; skipping ACL hardening")
        return False

    command = [
        "icacls",
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"{grant}:(OI)(CI)F",
        "/grant:r",
        f"{_SYSTEM_SID}:(OI)(CI)F",
        "/grant:r",
        f"{_ADMINISTRATORS_SID}:(OI)(CI)F",
        "/T",
        "/C",
        "/Q",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("icacls unavailable (%s); %s left with inherited ACLs", exc, path)
        return False
    if result.returncode != 0:
        log.warning(
            "icacls failed on %s (rc=%d): %s",
            path,
            result.returncode,
            (result.stderr or result.stdout).strip()[:300],
        )
        return False
    log.info("hardened ACLs on %s", path)
    return True


def describe_protection(path: Path) -> tuple[bool, str]:
    """``(protected, human description in Spanish)`` for ``meetscribe doctor``."""
    if not path.exists():
        return (False, "el directorio de datos aún no existe")
    if sys.platform != "win32":
        mode = path.stat().st_mode & 0o777
        return (mode == 0o700, f"permisos {mode:o}")
    try:
        result = subprocess.run(
            ["icacls", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return (False, "no se pudo consultar la ACL (icacls no disponible)")
    text = result.stdout or ""
    inherited = "(I)" in text
    if inherited:
        return (
            False,
            (
                "el directorio hereda permisos del perfil: otros administradores "
                "pueden leer las transcripciones"
            ),
        )
    return (True, "ACL restringida al usuario actual, SYSTEM y Administradores")
