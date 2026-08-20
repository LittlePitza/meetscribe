"""Domain value objects.

Everything here is a frozen dataclass with no behaviour beyond derivation from its
own fields, and the module imports nothing heavier than the standard library --
``numpy`` appears only under ``TYPE_CHECKING``. That is deliberate: the whole
domain must be importable and unit-testable in milliseconds, with no audio
hardware, no model files and no network.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "PIPELINE_SAMPLE_RATE",
    "LOCAL_SPEAKER_ID",
    "UNKNOWN_SPEAKER_ID",
    "TrackKind",
    "SegmentState",
    "SessionState",
    "ConsentMode",
    "AudioFormat",
    "AudioChunk",
    "Utterance",
    "TranscriptionResult",
    "Speaker",
    "TranscriptSegment",
    "ActionItem",
    "Summary",
    "Session",
    "LevelReading",
    "new_id",
    "normalise_for_dedup",
]


PIPELINE_SAMPLE_RATE = 16_000
"""Everything downstream of capture is 16 kHz mono float32 -- what Whisper, Silero
VAD and TitaNet all consume natively. Capture is the only stage that sees 48 kHz."""

LOCAL_SPEAKER_ID = "yo"
"""The microphone track is the user, always, with certainty. No clustering needed."""

UNKNOWN_SPEAKER_ID = "remoto"
"""Fallback label when diarization is disabled or could not decide."""


def new_id() -> str:
    """A short, collision-free identifier.

    UUID4 hex truncated to 12 characters: 48 bits, ample for the tens of thousands
    of rows a personal transcript archive will ever hold, and short enough to
    paste into a CLI argument by hand.
    """
    return uuid.uuid4().hex[:12]


def normalise_for_dedup(text: str) -> str:
    """Fold case, accents, punctuation and whitespace.

    Used to collapse near-identical action items detected on consecutive passes,
    and to compare transcript text against the hallucination blocklist.
    """
    folded = unicodedata.normalize("NFD", text.lower())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", folded)).strip()


# ------------------------------------------------------------------ enumerations


class TrackKind(str, Enum):
    """The two independent capture tracks. They are never pre-mixed."""

    MIC = "mic"
    """The local microphone: the user speaking."""

    LOOPBACK = "loopback"
    """The WASAPI render-endpoint loopback: everything remote participants say."""

    @property
    def is_local(self) -> bool:
        return self is TrackKind.MIC


class SegmentState(str, Enum):
    """Draft segments may be replaced; final segments are immutable."""

    DRAFT = "draft"
    FINAL = "final"


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (SessionState.STOPPED, SessionState.FAILED)


class ConsentMode(str, Enum):
    """How participants were informed. Recorded per session, never inferred."""

    ANNOUNCED = "announced"
    """The user stated out loud that the meeting is being transcribed."""

    ALL_PARTIES = "all_parties"
    """Explicit consent was collected from every participant."""

    SELF_ONLY = "self_only"
    """Only the local track is kept; nothing remote is retained."""

    NOT_RECORDED = "not_recorded"
    """No consent was captured. Lawful in some jurisdictions, not in others."""


# ----------------------------------------------------------------------- audio


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """The shape of a stream of samples, as reported by a device or a file."""

    sample_rate: int
    channels: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")

    @property
    def is_pipeline_native(self) -> bool:
        """True when no conversion is needed before the VAD/STT stages."""
        return self.sample_rate == PIPELINE_SAMPLE_RATE and self.channels == 1


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A block of mono 16 kHz float32 audio, stamped on the session timeline.

    ``start_s`` is derived from the track's own sample counter (see
    :class:`meetscribe.core.timeline.TrackTimeline`), not from wall clock at
    delivery time, so a late-scheduled thread does not smear the transcript.
    """

    track: TrackKind
    samples: np.ndarray
    start_s: float
    sample_rate: int = PIPELINE_SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True, slots=True)
class Utterance:
    """A contiguous run of speech isolated by the VAD, ready to transcribe.

    ``is_partial`` marks a forced cut of an utterance still in progress -- a long
    monologue split so the panel does not sit silent for 40 seconds. The final cut
    of the same utterance arrives later and supersedes it.
    """

    track: TrackKind
    samples: np.ndarray
    start_s: float
    end_s: float
    seq: int
    is_partial: bool = False
    sample_rate: int = PIPELINE_SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def key(self) -> tuple[str, int]:
        """Identity shared by a partial and the final cut that replaces it."""
        return (self.track.value, self.seq)


@dataclass(frozen=True, slots=True)
class LevelReading:
    """One level-meter tick, so a dead microphone is visible instantly."""

    track: TrackKind
    dbfs: float
    peak: float
    at_s: float


# ------------------------------------------------------------------ transcription


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """What a ``Transcriber`` returns for one utterance.

    The quality fields are not decoration: they are the inputs to the
    hallucination filter, which is the difference between a usable transcript and
    one peppered with "Subtitulos realizados por la comunidad de Amara.org".
    """

    text: str
    language: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Speaker:
    """A participant label. ``speaker_id`` is stable for the life of the session."""

    speaker_id: str
    display_name: str
    track: TrackKind
    is_local: bool = False
    utterance_count: int = 0

    def renamed(self, display_name: str) -> Speaker:
        name = display_name.strip()
        if not name:
            raise ValueError("display_name cannot be blank")
        return replace(self, display_name=name)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One attributed, timestamped line of the transcript.

    A segment in ``DRAFT`` state is provisional and may be replaced by a ``FINAL``
    one carrying the same ``(track, seq)``. Once ``FINAL``, the text is immutable
    -- only ``speaker_id``/``speaker_name`` may still change, and only through the
    rename flow or the end-of-session diarization refinement.
    """

    segment_id: str
    session_id: str
    track: TrackKind
    start_s: float
    end_s: float
    text: str
    speaker_id: str
    speaker_name: str
    seq: int
    state: SegmentState = SegmentState.FINAL
    language: str | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def with_speaker(self, speaker: Speaker) -> TranscriptSegment:
        return replace(
            self, speaker_id=speaker.speaker_id, speaker_name=speaker.display_name
        )


# ---------------------------------------------------------------------- summary


@dataclass(frozen=True, slots=True)
class ActionItem:
    """A commitment extracted from the meeting.

    ``owner`` and ``due`` are frequently absent -- that is honest output, not a gap
    for the model to fill by inventing one.
    """

    text: str
    owner: str | None = None
    due: str | None = None
    source_segment_id: str | None = None
    at_s: float | None = None

    @property
    def fingerprint(self) -> str:
        """Normalised form used to suppress duplicate live detections."""
        return normalise_for_dedup(f"{self.owner or ''} {self.text}")


@dataclass(frozen=True, slots=True)
class Summary:
    """Structured meeting minutes. Produced by a ``Summarizer``; never free text."""

    session_id: str
    provider: str
    model: str
    generated_at: dt.datetime
    executive: str
    topics: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    action_items: tuple[ActionItem, ...] = ()
    open_questions: tuple[str, ...] = ()
    follow_ups: tuple[str, ...] = ()
    partial: bool = False
    """True when the transcript was summarised in chunks, so the reduce pass never
    saw the whole meeting at once."""


# ---------------------------------------------------------------------- session


@dataclass(frozen=True, slots=True)
class Session:
    """One recorded meeting."""

    session_id: str
    name: str
    started_at: dt.datetime
    ended_at: dt.datetime | None = None
    language: str = "es"
    state: SessionState = SessionState.IDLE
    consent: ConsentMode = ConsentMode.NOT_RECORDED
    consent_note: str = ""
    offline_only: bool = False
    diarization_enabled: bool = True
    stt_model: str = ""
    audio_retained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @staticmethod
    def slug(name: str) -> str:
        """A filesystem- and URL-safe form of the session name, for exports.

        Accents are folded rather than preserved: an export filename travels
        through email, chat and other people's machines, and "reunion-de-diseno"
        survives that trip where the accented original may not.
        """
        folded = unicodedata.normalize("NFD", name.lower())
        folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
        folded = re.sub(r"[^a-z0-9\s-]", "", folded).strip()
        folded = re.sub(r"[\s_-]+", "-", folded)
        return folded[:60].strip("-") or "sesion"
