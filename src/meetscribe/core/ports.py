"""Ports: the seams between the domain and everything that touches the outside world.

Every heavy dependency in this project -- WASAPI, CTranslate2, onnxruntime, SQLite,
HTTP -- sits behind one of these Protocols. The pipeline is written against the
Protocols only, so the entire orchestration can be exercised in a unit test with
in-memory fakes: no audio hardware, no model files, no network, no clock.

Protocols are structural, so an adapter never imports this module to satisfy it.
That keeps the dependency arrow pointing inward.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from meetscribe.core.models import (
    ActionItem,
    AudioFormat,
    Session,
    Speaker,
    Summary,
    TranscriptionResult,
    TranscriptSegment,
    Utterance,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from meetscribe.core.events import Event

__all__ = [
    "Clock",
    "AudioSource",
    "Transcriber",
    "SpeakerEmbedder",
    "Diarizer",
    "Summarizer",
    "SummaryRequest",
    "TranscriptStore",
    "EventBus",
    "Subscription",
]


@runtime_checkable
class Clock(Protocol):
    """Time, injected.

    Two separate readings on purpose: ``monotonic`` for durations and timeline
    arithmetic (immune to NTP steps and DST), ``now`` for the human-facing
    timestamps that get written to the database.
    """

    def monotonic(self) -> float: ...

    def now(self) -> dt.datetime: ...

    def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class AudioSource(Protocol):
    """A stream of frames in the device's own native format.

    Conversion to the pipeline's 16 kHz mono float32 is the capture stage's job,
    not the source's, so a WAV replay source and a live WASAPI endpoint are
    interchangeable without either knowing about resampling.

    ``read`` returns ``None`` at end of stream (a finite source such as a WAV
    file), and raises :class:`meetscribe.errors.DeviceLostError` when the
    endpoint disappears mid-session -- headphones plugged in, device disabled --
    which the capture manager answers by reopening rather than by failing.
    """

    @property
    def name(self) -> str: ...

    def open(self) -> AudioFormat: ...

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


@runtime_checkable
class Transcriber(Protocol):
    """Speech to text for one complete utterance.

    Deliberately not streaming: the VAD upstream has already cut the audio at a
    natural pause, which is what makes Whisper's 30-second window behave. See
    ``docs/architecture.md`` for why LocalAgreement-2 was rejected.
    """

    @property
    def name(self) -> str: ...

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult: ...

    def close(self) -> None: ...


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """Fixed-length voice embedding for a slice of speech, L2-normalised."""

    @property
    def dim(self) -> int: ...

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...

    def close(self) -> None: ...


@runtime_checkable
class Diarizer(Protocol):
    """Assigns a stable speaker label to each utterance on a mixed track.

    ``assign`` runs online and must never block the pipeline: an undecidable
    utterance gets the fallback label rather than an exception. ``refine`` runs
    once at end of session, re-clusters every stored embedding offline and returns
    the corrections to apply, keyed by :attr:`Utterance.key`.
    """

    @property
    def enabled(self) -> bool: ...

    def assign(self, utterance: Utterance) -> str: ...

    def refine(self) -> Mapping[tuple[str, int], str]: ...

    def speakers(self) -> Sequence[Speaker]: ...

    def close(self) -> None: ...


class SummaryRequest:
    """Input to a :class:`Summarizer`. A plain object so adapters stay decoupled
    from the store's row shapes."""

    __slots__ = ("session", "segments", "speakers", "language", "now")

    def __init__(
        self,
        session: Session,
        segments: Sequence[TranscriptSegment],
        speakers: Sequence[Speaker],
        language: str = "es",
        now: dt.datetime | None = None,
    ) -> None:
        self.session = session
        self.segments = segments
        self.speakers = speakers
        self.language = language
        self.now = now or dt.datetime.now(dt.UTC)


@runtime_checkable
class Summarizer(Protocol):
    """Turns a transcript into structured minutes.

    ``is_local`` is load-bearing, not informational: the session refuses to use a
    non-local summarizer when the session was started in offline-only mode.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_local(self) -> bool: ...

    def summarize(self, request: SummaryRequest) -> Summary: ...

    def detect_action_items(
        self, segments: Sequence[TranscriptSegment], language: str = "es"
    ) -> Sequence[ActionItem]: ...

    def health(self) -> str: ...


@runtime_checkable
class TranscriptStore(Protocol):
    """Durable storage. Writes are committed per segment so a crash at minute 40
    of a 60-minute meeting loses at most the utterance in flight.

    This Protocol is the whole surface the pipeline uses -- including
    ``drop_drafts`` and the action-item methods. Declaring less than the callers
    actually call would make the port a description of an older design rather than
    a contract, and a fake built from it would pass type checking and then fail at
    runtime on the first missing method.
    """

    def create_session(self, session: Session) -> None: ...

    def update_session(self, session: Session) -> None: ...

    def get_session(self, session_id: str) -> Session | None: ...

    def list_sessions(self, limit: int = 50) -> Sequence[Session]: ...

    def delete_session(self, session_id: str) -> None: ...

    def add_segment(self, segment: TranscriptSegment) -> None: ...

    def replace_segment(self, segment: TranscriptSegment) -> None: ...

    def get_segments(
        self, session_id: str, final_only: bool = False
    ) -> Sequence[TranscriptSegment]: ...

    def drop_drafts(self, session_id: str) -> int:
        """Discard provisional lines never confirmed. Returns how many went."""
        ...

    def upsert_speaker(self, session_id: str, speaker: Speaker) -> None: ...

    def get_speakers(self, session_id: str) -> Sequence[Speaker]: ...

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> int: ...

    def reassign_speakers(
        self, session_id: str, mapping: Mapping[tuple[str, int], str]
    ) -> int: ...

    def add_action_item(self, session_id: str, item: ActionItem) -> bool:
        """Store one commitment. False means it duplicated an existing one."""
        ...

    def get_action_items(self, session_id: str) -> Sequence[ActionItem]: ...

    def save_summary(self, summary: Summary) -> None: ...

    def get_summary(self, session_id: str) -> Summary | None: ...

    def close(self) -> None: ...


@runtime_checkable
class Subscription(Protocol):
    """A live feed of events with a replay cursor, so a browser refresh resumes
    exactly where it left off instead of losing the transcript.

    Two ways to consume it, because the two consumers differ: iteration for code
    that can block until the stream ends, and ``get`` with a timeout for the SSE
    handler, which has to wake up on a schedule to emit keepalives whether or not
    an event arrived.
    """

    def __iter__(self) -> Iterator[Event]: ...

    def get(self, timeout: float | None = None) -> Event | None:
        """The next event, or ``None`` on timeout or once closed."""
        ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    """Thread-safe fan-out from the pipeline threads to the web panel.

    Publishing must never block a producer: a subscriber that cannot keep up is
    disconnected, and reconnects with its cursor.
    """

    def publish(self, event_type: str, data: Mapping[str, object]) -> Event: ...

    def history(self, after_seq: int = 0, limit: int = 1000) -> Sequence[Event]: ...

    def subscribe(self, after_seq: int = 0) -> Subscription: ...

    def close(self) -> None: ...
