"""Durable storage: SQLite in WAL mode.

**Why SQLite and why per-segment commits.** The scenario that shaped this module
is the process dying at minute 40 of a 60-minute meeting. Anything that buffers the
transcript in memory and writes it at the end loses everything. So each segment is
committed as it is confirmed: the durability unit is one utterance, and a crash
costs at most the sentence in flight.

WAL mode is what makes that affordable -- readers (``meetscribe show`` in another
terminal, the panel's replay) never block the writer, and a commit is an append
rather than a rewrite. ``synchronous=NORMAL`` is the deliberate setting: FULL would
fsync on every utterance, which on a laptop SSD is a real cost for a guarantee we
do not need (the failure we are protecting against is a process crash, which WAL
survives at NORMAL; only an OS-level power loss could lose the last commits).

**Identity.** A segment is unique on ``(session_id, track, seq)`` rather than on its
row id. That is what makes the draft-to-final transition an upsert instead of a
delete-and-insert, so the panel and the database agree on which line was replaced.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from meetscribe.core.models import (
    ActionItem,
    ConsentMode,
    SegmentState,
    Session,
    SessionState,
    Speaker,
    Summary,
    TrackKind,
    TranscriptSegment,
)
from meetscribe.errors import SessionNotFoundError, StoreError

log = logging.getLogger(__name__)

__all__ = ["SqliteStore", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    language            TEXT NOT NULL DEFAULT 'es',
    state               TEXT NOT NULL,
    consent             TEXT NOT NULL,
    consent_note        TEXT NOT NULL DEFAULT '',
    offline_only        INTEGER NOT NULL DEFAULT 0,
    diarization_enabled INTEGER NOT NULL DEFAULT 1,
    stt_model           TEXT NOT NULL DEFAULT '',
    audio_retained      INTEGER NOT NULL DEFAULT 0,
    metadata            TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS speakers (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    speaker_id      TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    track           TEXT NOT NULL,
    is_local        INTEGER NOT NULL DEFAULT 0,
    utterance_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, speaker_id)
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id     TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track          TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    state          TEXT NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    text           TEXT NOT NULL,
    speaker_id     TEXT NOT NULL,
    speaker_name   TEXT NOT NULL,
    language       TEXT,
    no_speech_prob REAL,
    avg_logprob    REAL,
    UNIQUE (session_id, track, seq)
);

CREATE INDEX IF NOT EXISTS idx_segments_session_time
    ON segments (session_id, start_s);

CREATE TABLE IF NOT EXISTS summaries (
    session_id   TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    executive    TEXT NOT NULL,
    payload      TEXT NOT NULL,
    partial      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS action_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    text              TEXT NOT NULL,
    owner             TEXT,
    due               TEXT,
    at_s              REAL,
    source_segment_id TEXT,
    fingerprint       TEXT NOT NULL,
    UNIQUE (session_id, fingerprint)
);
"""


class SqliteStore:
    """Thread-safe SQLite store.

    One connection guarded by a lock, rather than a connection per thread. At a few
    writes per second the lock is never contended, and it removes the entire class
    of bugs where a thread-local connection outlives its thread or a transaction
    spans two of them.
    """

    def __init__(self, path: str | Path, timeout: float = 15.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._db = sqlite3.connect(
                str(self.path), check_same_thread=False, timeout=timeout
            )
        except sqlite3.Error as exc:
            raise StoreError(
                f"cannot open {self.path}: {exc}",
                user_message=f"No se pudo abrir la base de datos en '{self.path}'.",
            ) from exc
        self._db.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA busy_timeout=15000")

    def _migrate(self) -> None:
        with self._lock:
            version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"database schema v{version} is newer than this build "
                    f"(v{SCHEMA_VERSION})",
                    user_message=(
                        "La base de datos fue creada por una versión más reciente "
                        "de MeetScribe. Actualiza la aplicación."
                    ),
                )
            self._db.executescript(_SCHEMA)
            # Future migrations append here as `if version < N: ...` blocks; the
            # schema is created idempotently so a fresh database needs no special
            # case.
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._db.commit()

    # ------------------------------------------------------------------ sessions

    def create_session(self, session: Session) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO sessions (session_id, name, started_at, ended_at,
                       language, state, consent, consent_note, offline_only,
                       diarization_enabled, stt_model, audio_retained, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session.session_id,
                    session.name,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.language,
                    session.state.value,
                    session.consent.value,
                    session.consent_note,
                    int(session.offline_only),
                    int(session.diarization_enabled),
                    session.stt_model,
                    int(session.audio_retained),
                    json.dumps(session.metadata, ensure_ascii=False),
                ),
            )
            self._db.commit()

    def update_session(self, session: Session) -> None:
        with self._lock:
            cursor = self._db.execute(
                """UPDATE sessions SET name=?, ended_at=?, language=?, state=?,
                       consent=?, consent_note=?, offline_only=?,
                       diarization_enabled=?, stt_model=?, audio_retained=?,
                       metadata=?
                   WHERE session_id=?""",
                (
                    session.name,
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.language,
                    session.state.value,
                    session.consent.value,
                    session.consent_note,
                    int(session.offline_only),
                    int(session.diarization_enabled),
                    session.stt_model,
                    int(session.audio_retained),
                    json.dumps(session.metadata, ensure_ascii=False),
                    session.session_id,
                ),
            )
            self._db.commit()
            if cursor.rowcount == 0:
                raise SessionNotFoundError(f"session {session.session_id} not found")

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def resolve_session(self, prefix: str) -> Session:
        """Look a session up by full id, unique id prefix, or exact name.

        Typing twelve hex characters to export a transcript is hostile, so a prefix
        is accepted -- but an ambiguous one is an error, never a guess.
        """
        session = self.get_session(prefix)
        if session is not None:
            return session
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions WHERE session_id LIKE ? OR name = ? "
                "ORDER BY started_at DESC",
                (f"{prefix}%", prefix),
            ).fetchall()
        if not rows:
            raise SessionNotFoundError(
                f"no session matching {prefix!r}",
                user_message=f"No hay ninguna sesión que coincida con '{prefix}'.",
            )
        if len(rows) > 1:
            names = ", ".join(f"{r['session_id']} ({r['name']})" for r in rows[:5])
            raise SessionNotFoundError(
                f"ambiguous session prefix {prefix!r}",
                user_message=f"'{prefix}' coincide con varias sesiones: {names}.",
            )
        return _row_to_session(rows[0])

    def list_sessions(self, limit: int = 50) -> list[Session]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_session(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self._db.commit()

    def sessions_older_than(self, cutoff: dt.datetime) -> list[Session]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions WHERE started_at < ? ORDER BY started_at",
                (cutoff.isoformat(),),
            ).fetchall()
        return [_row_to_session(r) for r in rows]

    # ------------------------------------------------------------------ segments

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.replace_segment(segment)

    def replace_segment(self, segment: TranscriptSegment) -> None:
        """Upsert on ``(session_id, track, seq)``.

        A final segment overwrites the draft that preceded it, keeping the original
        ``segment_id`` so anything holding a reference to the line still resolves.
        """
        with self._lock:
            existing = self._db.execute(
                "SELECT segment_id, state FROM segments "
                "WHERE session_id=? AND track=? AND seq=?",
                (segment.session_id, segment.track.value, segment.seq),
            ).fetchone()
            if existing and existing["state"] == SegmentState.FINAL.value and (
                segment.state == SegmentState.DRAFT
            ):
                # A late draft must never overwrite confirmed text.
                log.debug(
                    "ignoring late draft for %s/%s", segment.track.value, segment.seq
                )
                return
            segment_id = existing["segment_id"] if existing else segment.segment_id
            self._db.execute(
                """INSERT INTO segments (segment_id, session_id, track, seq, state,
                       start_s, end_s, text, speaker_id, speaker_name, language,
                       no_speech_prob, avg_logprob)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id, track, seq) DO UPDATE SET
                       state=excluded.state,
                       start_s=excluded.start_s,
                       end_s=excluded.end_s,
                       text=excluded.text,
                       speaker_id=excluded.speaker_id,
                       speaker_name=excluded.speaker_name,
                       language=excluded.language,
                       no_speech_prob=excluded.no_speech_prob,
                       avg_logprob=excluded.avg_logprob""",
                (
                    segment_id,
                    segment.session_id,
                    segment.track.value,
                    segment.seq,
                    segment.state.value,
                    segment.start_s,
                    segment.end_s,
                    segment.text,
                    segment.speaker_id,
                    segment.speaker_name,
                    segment.language,
                    segment.no_speech_prob,
                    segment.avg_logprob,
                ),
            )
            self._db.commit()

    def get_segments(
        self, session_id: str, final_only: bool = False
    ) -> list[TranscriptSegment]:
        query = "SELECT * FROM segments WHERE session_id=?"
        params: list[Any] = [session_id]
        if final_only:
            query += " AND state=?"
            params.append(SegmentState.FINAL.value)
        query += " ORDER BY start_s, track, seq"
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [_row_to_segment(r) for r in rows]

    def drop_drafts(self, session_id: str) -> int:
        """Remove provisional lines that never got a final version.

        Runs at session end. A draft left in the archive would be indistinguishable
        from confirmed text a month later.
        """
        with self._lock:
            cursor = self._db.execute(
                "DELETE FROM segments WHERE session_id=? AND state=?",
                (session_id, SegmentState.DRAFT.value),
            )
            self._db.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------ speakers

    def upsert_speaker(self, session_id: str, speaker: Speaker) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO speakers (session_id, speaker_id, display_name, track,
                       is_local, utterance_count)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(session_id, speaker_id) DO UPDATE SET
                       display_name=excluded.display_name,
                       utterance_count=excluded.utterance_count""",
                (
                    session_id,
                    speaker.speaker_id,
                    speaker.display_name,
                    speaker.track.value,
                    int(speaker.is_local),
                    speaker.utterance_count,
                ),
            )
            self._db.commit()

    def get_speakers(self, session_id: str) -> list[Speaker]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM speakers WHERE session_id=? ORDER BY speaker_id",
                (session_id,),
            ).fetchall()
        return [
            Speaker(
                speaker_id=r["speaker_id"],
                display_name=r["display_name"],
                track=TrackKind(r["track"]),
                is_local=bool(r["is_local"]),
                utterance_count=int(r["utterance_count"]),
            )
            for r in rows
        ]

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> int:
        """Apply a rename retroactively to every line already transcribed.

        Returns the number of segments touched. Forward segments pick the new name
        up because the pipeline reads speakers from the store.
        """
        name = name.strip()
        if not name:
            raise StoreError(
                "blank speaker name", user_message="El nombre no puede estar vacío."
            )
        with self._lock:
            updated = self._db.execute(
                "UPDATE speakers SET display_name=? WHERE session_id=? AND speaker_id=?",
                (name, session_id, speaker_id),
            ).rowcount
            if updated == 0:
                raise StoreError(
                    f"no speaker {speaker_id!r} in session {session_id}",
                    user_message=f"La sesión no tiene un hablante '{speaker_id}'.",
                )
            count = self._db.execute(
                "UPDATE segments SET speaker_name=? WHERE session_id=? AND speaker_id=?",
                (name, session_id, speaker_id),
            ).rowcount
            self._db.commit()
        return count

    def reassign_speakers(
        self, session_id: str, mapping: Mapping[tuple[str, int], str]
    ) -> int:
        """Apply the end-of-session diarization refinement."""
        if not mapping:
            return 0
        with self._lock:
            names = {
                s.speaker_id: s.display_name for s in self.get_speakers(session_id)
            }
            changed = 0
            for (track, seq), speaker_id in mapping.items():
                changed += self._db.execute(
                    "UPDATE segments SET speaker_id=?, speaker_name=? "
                    "WHERE session_id=? AND track=? AND seq=?",
                    (speaker_id, names.get(speaker_id, speaker_id), session_id, track, seq),
                ).rowcount
            self._db.commit()
        return changed

    # ----------------------------------------------------------------- summaries

    def save_summary(self, summary: Summary) -> None:
        payload = {
            "topics": list(summary.topics),
            "decisions": list(summary.decisions),
            "open_questions": list(summary.open_questions),
            "follow_ups": list(summary.follow_ups),
            "action_items": [
                {
                    "text": a.text,
                    "owner": a.owner,
                    "due": a.due,
                    "at_s": a.at_s,
                    "source_segment_id": a.source_segment_id,
                }
                for a in summary.action_items
            ],
        }
        with self._lock:
            self._db.execute(
                """INSERT INTO summaries (session_id, provider, model, generated_at,
                       executive, payload, partial)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       provider=excluded.provider, model=excluded.model,
                       generated_at=excluded.generated_at,
                       executive=excluded.executive, payload=excluded.payload,
                       partial=excluded.partial""",
                (
                    summary.session_id,
                    summary.provider,
                    summary.model,
                    summary.generated_at.isoformat(),
                    summary.executive,
                    json.dumps(payload, ensure_ascii=False),
                    int(summary.partial),
                ),
            )
            self._db.commit()
        for item in summary.action_items:
            self.add_action_item(summary.session_id, item)

    def get_summary(self, session_id: str) -> Summary | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload"])
        return Summary(
            session_id=row["session_id"],
            provider=row["provider"],
            model=row["model"],
            generated_at=dt.datetime.fromisoformat(row["generated_at"]),
            executive=row["executive"],
            topics=tuple(payload.get("topics", ())),
            decisions=tuple(payload.get("decisions", ())),
            action_items=tuple(
                ActionItem(
                    text=a["text"],
                    owner=a.get("owner"),
                    due=a.get("due"),
                    at_s=a.get("at_s"),
                    source_segment_id=a.get("source_segment_id"),
                )
                for a in payload.get("action_items", ())
            ),
            open_questions=tuple(payload.get("open_questions", ())),
            follow_ups=tuple(payload.get("follow_ups", ())),
            partial=bool(row["partial"]),
        )

    # -------------------------------------------------------------- action items

    def add_action_item(self, session_id: str, item: ActionItem) -> bool:
        """Insert unless an equivalent item is already recorded.

        The fingerprint is accent- and punctuation-folded, so the same commitment
        detected live and again in the final summary lands once.
        """
        with self._lock:
            cursor = self._db.execute(
                """INSERT OR IGNORE INTO action_items
                       (session_id, text, owner, due, at_s, source_segment_id, fingerprint)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    session_id,
                    item.text,
                    item.owner,
                    item.due,
                    item.at_s,
                    item.source_segment_id,
                    item.fingerprint,
                ),
            )
            self._db.commit()
            return cursor.rowcount > 0

    def get_action_items(self, session_id: str) -> list[ActionItem]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM action_items WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            ActionItem(
                text=r["text"],
                owner=r["owner"],
                due=r["due"],
                at_s=r["at_s"],
                source_segment_id=r["source_segment_id"],
            )
            for r in rows
        ]

    # --------------------------------------------------------------------- misc

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sessions": self._db.execute(
                    "SELECT COUNT(*) FROM sessions"
                ).fetchone()[0],
                "segments": self._db.execute(
                    "SELECT COUNT(*) FROM segments"
                ).fetchone()[0],
                "summaries": self._db.execute(
                    "SELECT COUNT(*) FROM summaries"
                ).fetchone()[0],
            }

    def close(self) -> None:
        with self._lock:
            try:
                self._db.commit()
            finally:
                self._db.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["session_id"],
        name=row["name"],
        started_at=dt.datetime.fromisoformat(row["started_at"]),
        ended_at=(
            dt.datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
        ),
        language=row["language"],
        state=SessionState(row["state"]),
        consent=ConsentMode(row["consent"]),
        consent_note=row["consent_note"],
        offline_only=bool(row["offline_only"]),
        diarization_enabled=bool(row["diarization_enabled"]),
        stt_model=row["stt_model"],
        audio_retained=bool(row["audio_retained"]),
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _row_to_segment(row: sqlite3.Row) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=row["segment_id"],
        session_id=row["session_id"],
        track=TrackKind(row["track"]),
        seq=int(row["seq"]),
        state=SegmentState(row["state"]),
        start_s=float(row["start_s"]),
        end_s=float(row["end_s"]),
        text=row["text"],
        speaker_id=row["speaker_id"],
        speaker_name=row["speaker_name"],
        language=row["language"],
        no_speech_prob=row["no_speech_prob"],
        avg_logprob=row["avg_logprob"],
    )
