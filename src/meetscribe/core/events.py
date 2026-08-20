"""The live event protocol.

One typed vocabulary shared by the pipeline, the event bus, the SSE transport and
the browser. Payload shapes are built by the ``make_*`` helpers below rather than
assembled ad hoc at each call site, so the panel's contract is defined in exactly
one place and a change to it breaks the tests rather than the UI.

Every payload is JSON-serialisable with the standard encoder -- no custom default
hook -- because the transport is a plain ``json.dumps`` inside an SSE frame.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from meetscribe.core.models import (
    ActionItem,
    LevelReading,
    Session,
    Speaker,
    Summary,
    TrackKind,
    TranscriptSegment,
)

__all__ = [
    "Event",
    "EV_SESSION",
    "EV_SEGMENT",
    "EV_SPEAKER",
    "EV_LEVEL",
    "EV_VAD",
    "EV_ACTION_ITEM",
    "EV_SUMMARY",
    "EV_NOTICE",
    "EV_HEARTBEAT",
    "ALL_EVENT_TYPES",
    "make_session",
    "make_segment",
    "make_speaker",
    "make_level",
    "make_vad",
    "make_action_item",
    "make_summary",
    "make_notice",
]

EV_SESSION = "session"
"""Session identity and lifecycle state. Sent on connect and on every change."""

EV_SEGMENT = "segment"
"""A transcript line. ``state`` is 'draft' or 'final'; a final segment replaces the
draft carrying the same (track, seq)."""

EV_SPEAKER = "speaker"
"""A speaker appeared or was renamed. Renames apply retroactively in the UI."""

EV_LEVEL = "level"
"""Per-track level meter. Emitted a few times per second; not persisted."""

EV_VAD = "vad"
"""Speech activity edge for a track, used for the 'hablando' indicator."""

EV_ACTION_ITEM = "action_item"
"""A commitment detected live, before the meeting ends."""

EV_SUMMARY = "summary"
"""The structured minutes, once generated."""

EV_NOTICE = "notice"
"""An operational message for the user: level is 'info', 'warn' or 'error'."""

EV_HEARTBEAT = "heartbeat"
"""Keeps the SSE connection and any intermediary from timing the stream out."""

ALL_EVENT_TYPES = (
    EV_SESSION,
    EV_SEGMENT,
    EV_SPEAKER,
    EV_LEVEL,
    EV_VAD,
    EV_ACTION_ITEM,
    EV_SUMMARY,
    EV_NOTICE,
    EV_HEARTBEAT,
)

VOLATILE_EVENT_TYPES = frozenset({EV_LEVEL, EV_VAD, EV_HEARTBEAT})
"""Events not worth replaying to a reconnecting browser: they describe an instant,
not a fact. Keeping them out of the replay buffer means a refresh after two hours
replays the transcript, not 20 000 stale meter ticks."""


@dataclass(frozen=True, slots=True)
class Event:
    """One published fact. ``seq`` is the replay cursor: monotonic, gap-free."""

    seq: int
    type: str
    at: float
    """Session time in seconds, or wall-clock epoch seconds before a session
    starts. The panel uses it only for display."""
    data: Mapping[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"seq": self.seq, "type": self.type, "at": round(self.at, 3), "data": self.data},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @property
    def is_volatile(self) -> bool:
        return self.type in VOLATILE_EVENT_TYPES


# ------------------------------------------------------------------- builders


def make_session(session: Session, elapsed_s: float = 0.0) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "name": session.name,
        "state": session.state.value,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "elapsed_s": round(elapsed_s, 2),
        "language": session.language,
        "diarization": session.diarization_enabled,
        "offline_only": session.offline_only,
        "stt_model": session.stt_model,
        "consent": session.consent.value,
    }


def make_segment(segment: TranscriptSegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "track": segment.track.value,
        "seq": segment.seq,
        "state": segment.state.value,
        "start_s": round(segment.start_s, 3),
        "end_s": round(segment.end_s, 3),
        "text": segment.text,
        "speaker_id": segment.speaker_id,
        "speaker_name": segment.speaker_name,
        "language": segment.language,
    }


def make_speaker(speaker: Speaker) -> dict[str, Any]:
    return {
        "speaker_id": speaker.speaker_id,
        "display_name": speaker.display_name,
        "track": speaker.track.value,
        "is_local": speaker.is_local,
        "utterance_count": speaker.utterance_count,
    }


def make_level(readings: Iterable[LevelReading]) -> dict[str, Any]:
    payload: dict[str, Any] = {"tracks": {}}
    at = 0.0
    for r in readings:
        payload["tracks"][r.track.value] = {
            "dbfs": round(r.dbfs, 1),
            "peak": round(r.peak, 4),
        }
        at = max(at, r.at_s)
    payload["at_s"] = round(at, 2)
    return payload


def make_vad(track: TrackKind, speaking: bool, at_s: float) -> dict[str, Any]:
    return {"track": track.value, "speaking": speaking, "at_s": round(at_s, 2)}


def make_action_item(item: ActionItem) -> dict[str, Any]:
    return {
        "text": item.text,
        "owner": item.owner,
        "due": item.due,
        "at_s": round(item.at_s, 2) if item.at_s is not None else None,
        "fingerprint": item.fingerprint,
    }


def make_summary(summary: Summary) -> dict[str, Any]:
    return {
        "session_id": summary.session_id,
        "provider": summary.provider,
        "model": summary.model,
        "generated_at": summary.generated_at.isoformat(),
        "executive": summary.executive,
        "topics": list(summary.topics),
        "decisions": list(summary.decisions),
        "action_items": [make_action_item(a) for a in summary.action_items],
        "open_questions": list(summary.open_questions),
        "follow_ups": list(summary.follow_ups),
        "partial": summary.partial,
    }


def make_notice(level: str, code: str, message: str) -> dict[str, Any]:
    if level not in ("info", "warn", "error"):
        raise ValueError(f"unknown notice level: {level!r}")
    return {"level": level, "code": code, "message": message}
