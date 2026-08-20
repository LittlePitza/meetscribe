"""Retention and deletion of the most sensitive artifacts.

**The default is that raw audio is never written.** The transcript is the artifact
worth keeping; the audio adds little and is the single most sensitive file the tool
could leave on disk -- it carries tone, hesitation, side conversations and every
word the transcript filtered out. ``capture.retain_audio`` therefore defaults to
``none``, and the other two modes are opt-in:

* ``session`` -- keep WAVs while the meeting runs (useful for re-transcribing with
  a better model afterwards) and shred them when the process exits.
* ``keep`` -- retain under ``privacy.retention_days``.

**"Shredding" is honestly named.** On an SSD with wear levelling, overwriting a
file does not reliably destroy the underlying blocks; the flash translation layer
may have written the new bytes elsewhere. The overwrite here defeats casual
recovery tools, and that is all it claims to do. Genuine destruction needs
full-disk encryption, which is a property of the machine, not of this program. The
documentation says exactly this rather than implying more.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["shred_file", "shred_session_audio", "purge_expired_sessions"]

_OVERWRITE_CHUNK = 1 << 20


def shred_file(path: Path, passes: int = 1) -> bool:
    """Overwrite with random bytes, then unlink. Best effort; never raises."""
    try:
        if not path.is_file():
            return False
        size = path.stat().st_size
        with path.open("r+b", buffering=0) as handle:
            for _ in range(max(1, passes)):
                handle.seek(0)
                remaining = size
                while remaining > 0:
                    block = min(_OVERWRITE_CHUNK, remaining)
                    handle.write(os.urandom(block))
                    remaining -= block
                handle.flush()
                os.fsync(handle.fileno())
        path.unlink()
        return True
    except OSError as exc:
        log.warning("could not shred %s: %s", path, exc)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        return True


def shred_session_audio(directory: Path) -> int:
    """Destroy every WAV under a session's audio directory. Returns the count."""
    if not directory.exists():
        return 0
    count = 0
    for path in sorted(directory.glob("*.wav")):
        if shred_file(path):
            count += 1
    # A non-empty directory is left alone: something the caller did not expect
    # is in there, and deleting it blind would be worse than leaving it.
    with contextlib.suppress(OSError):
        directory.rmdir()
    return count


def purge_expired_sessions(
    store, audio_root: Path, retention_days: int, now: dt.datetime | None = None
) -> list[str]:
    """Delete sessions older than the retention window, audio included.

    Returns the ids removed. ``retention_days <= 0`` disables the policy, which is
    the default: silently deleting a user's meeting history is not something to do
    unless they asked for it.
    """
    if retention_days <= 0:
        return []
    cutoff = (now or dt.datetime.now(dt.UTC)) - dt.timedelta(days=retention_days)
    removed: list[str] = []
    for session in store.sessions_older_than(cutoff):
        shred_session_audio(audio_root / session.session_id)
        store.delete_session(session.session_id)
        removed.append(session.session_id)
        log.info("purged session %s (older than %d days)", session.session_id, retention_days)
    return removed


def delete_session_data(store, audio_root: Path, session_id: str) -> None:
    """Remove one session completely: rows, audio, and the directory."""
    shred_session_audio(audio_root / session_id)
    shutil.rmtree(audio_root / session_id, ignore_errors=True)
    store.delete_session(session_id)
