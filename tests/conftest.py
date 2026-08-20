"""Shared fixtures.

Two rules the whole suite depends on:

* **No test touches the user's real data.** ``isolated_data_dir`` is autouse and
  redirects ``MEETSCRIBE_DATA_DIR`` at every single test, so a bug in a path helper
  cannot delete somebody's meeting archive.
* **No test needs hardware, a network, a model file or a real clock.** Everything
  expensive sits behind a port with a fake in this file.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import pytest

from meetscribe.bus.pubsub import InMemoryEventBus
from meetscribe.core.clock import FakeClock
from meetscribe.core.models import (
    PIPELINE_SAMPLE_RATE,
    ActionItem,
    AudioChunk,
    ConsentMode,
    Session,
    SessionState,
    Summary,
    TrackKind,
    Utterance,
    new_id,
)
from meetscribe.store.sqlite_store import SqliteStore


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point every path helper at a throwaway directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("MEETSCRIBE_DATA_DIR", str(root))
    monkeypatch.setenv("MEETSCRIBE_MODELS_DIR", str(root / "models"))
    # Keep environment overrides from a developer's shell out of the tests.
    for name in (
        "ANTHROPIC_API_KEY",
        "MEETSCRIBE_OFFLINE",
        "MEETSCRIBE_PROVIDER",
        "MEETSCRIBE_STT_MODEL",
        "MEETSCRIBE_LANGUAGE",
        "MEETSCRIBE_PANEL_PORT",
        "MEETSCRIBE_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    return root


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus(history_limit=500, subscriber_queue=200)


@pytest.fixture
def store(tmp_path) -> SqliteStore:
    with SqliteStore(tmp_path / "test.db") as db:
        yield db


@pytest.fixture
def session() -> Session:
    return Session(
        session_id=new_id(),
        name="Reunión de prueba",
        started_at=dt.datetime(2026, 3, 1, 10, 0, tzinfo=dt.UTC),
        state=SessionState.IDLE,
        consent=ConsentMode.ANNOUNCED,
        stt_model="fake",
    )


# ------------------------------------------------------------------ audio data


def tone(seconds: float, freq: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    """Deterministic voiced-ish signal at the pipeline rate."""
    t = np.arange(int(seconds * PIPELINE_SAMPLE_RATE)) / PIPELINE_SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * PIPELINE_SAMPLE_RATE), dtype=np.float32)


def chunk(track: TrackKind, start_s: float, samples: np.ndarray) -> AudioChunk:
    return AudioChunk(track=track, samples=samples, start_s=start_s)


def utterance(
    track: TrackKind = TrackKind.LOOPBACK,
    seq: int = 1,
    start_s: float = 0.0,
    duration: float = 2.0,
    is_partial: bool = False,
    freq: float = 220.0,
) -> Utterance:
    return Utterance(
        track=track,
        samples=tone(duration, freq),
        start_s=start_s,
        end_s=start_s + duration,
        seq=seq,
        is_partial=is_partial,
    )


@pytest.fixture
def make_utterance():
    return utterance


# ----------------------------------------------------------------- fake engines


class ScriptedVad:
    """A VAD engine driven by a script instead of a model.

    ``script`` maps "after N fed windows" to an action: ``("emit", start, length)``
    queues a finished segment, ``("speak", start, length)`` sets an in-progress one.
    This makes the segmenter's index arithmetic testable to the sample.
    """

    def __init__(self, script: dict[int, tuple] | None = None) -> None:
        self.script = script or {}
        self.windows = 0
        self.pending: list = []
        self.current = None
        self.reset_count = 0
        self.flushed = False

    def accept_waveform(self, samples: np.ndarray) -> None:
        self.windows += 1
        action = self.script.get(self.windows)
        if action is None:
            return
        kind, start, length = action
        segment = _Segment(start, np.zeros(length, dtype=np.float32))
        if kind == "emit":
            self.pending.append(segment)
            self.current = None
        else:
            self.current = segment

    def is_speech_detected(self) -> bool:
        return self.current is not None

    def empty(self) -> bool:
        return not self.pending

    @property
    def front(self):
        return self.pending[0]

    @property
    def current_segment(self):
        return self.current

    def pop(self) -> None:
        self.pending.pop(0)

    def flush(self) -> None:
        self.flushed = True
        if self.current is not None:
            self.pending.append(self.current)
            self.current = None

    def reset(self) -> None:
        self.reset_count += 1
        self.pending.clear()
        self.current = None
        self.windows = 0


class _Segment:
    __slots__ = ("start", "samples")

    def __init__(self, start: int, samples: np.ndarray) -> None:
        self.start = start
        self.samples = samples


@pytest.fixture
def scripted_vad():
    return ScriptedVad


class RecordingSummarizer:
    """Captures the prompts it was given and replays canned JSON."""

    provider = "fake"
    is_local = True

    def __init__(self, payloads: Sequence[str] | None = None) -> None:
        self.payloads = list(payloads or [])
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-1"

    def summarize(self, request) -> Summary:
        self.calls.append(("summarize", request.session.session_id))
        return Summary(
            session_id=request.session.session_id,
            provider="fake",
            model="fake-1",
            generated_at=request.now,
            executive="Resumen de prueba.",
            topics=("tema",),
            decisions=("decisión",),
            action_items=(ActionItem(text="hacer algo", owner="Yo"),),
        )

    def detect_action_items(self, segments, language="es"):
        self.calls.append(("actions", str(len(segments))))
        return (ActionItem(text="compromiso en vivo", owner="Participante 1"),)

    def health(self) -> str:
        return "disponible (fake)"


@pytest.fixture
def fake_summarizer():
    return RecordingSummarizer
