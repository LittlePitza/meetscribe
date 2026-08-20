"""End-to-end: two tracks of real audio through the whole pipeline, no hardware.

This is the test that would catch a regression nothing else would -- a queue that
never drains, a shutdown that loses the last utterance, a draft that survives into
the archive, a speaker that never reaches the store. It runs the actual
:class:`SessionRunner` with its actual threads, against WAV replay sources, a
scripted VAD, a fake decoder and a real SQLite store.

It finishes in about a second because the replay sources declare themselves
non-realtime: the pipeline consumes the files as fast as it can while producing
exactly the timestamps a live capture would have produced.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from meetscribe.config import Config
from meetscribe.core.clock import SystemClock
from meetscribe.core.events import (
    EV_ACTION_ITEM,
    EV_NOTICE,
    EV_SEGMENT,
    EV_SESSION,
    EV_SPEAKER,
    EV_SUMMARY,
)
from meetscribe.core.models import (
    LOCAL_SPEAKER_ID,
    SegmentState,
    SessionState,
    TrackKind,
    TranscriptionResult,
)
from meetscribe.diarize.embedder import FakeEmbedder
from meetscribe.diarize.online import NullDiarizer, OnlineDiarizer
from meetscribe.errors import SummarizerUnavailableError
from meetscribe.pipeline.factory import build_replay_factory
from meetscribe.pipeline.session import SessionRunner, recover_interrupted_sessions
from meetscribe.stt.transcriber import FakeTranscriber
from meetscribe.stt.vad import WINDOW

from .conftest import RecordingSummarizer

SR = 16000


def make_wav(path: Path, seconds: float = 4.0, rate: int = 48000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    signal = (0.3 * np.sin(2 * np.pi * 260 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((signal * 32767).astype("<i2").tobytes())
    return path


class PeriodicVad:
    """Emits one finished segment every ``every`` windows.

    Deterministic in *count* rather than in timing: the file is fully consumed
    before flush, so the number of utterances depends only on how much audio there
    was, never on how the threads happened to be scheduled.
    """

    def __init__(self, every: int = 25) -> None:
        self.every = every
        self.windows = 0
        self.pending: list = []

    def accept_waveform(self, samples: np.ndarray) -> None:
        self.windows += 1
        if self.windows % self.every == 0:
            start = (self.windows - self.every) * WINDOW
            self.pending.append(_Seg(start, np.full(self.every * WINDOW, 0.2, np.float32)))

    def is_speech_detected(self) -> bool:
        return False

    def empty(self) -> bool:
        return not self.pending

    @property
    def front(self):
        return self.pending[0]

    @property
    def current_segment(self):
        return None

    def pop(self) -> None:
        self.pending.pop(0)

    def flush(self) -> None:
        return None

    def reset(self) -> None:
        self.pending.clear()


class _Seg:
    __slots__ = ("start", "samples")

    def __init__(self, start, samples):
        self.start = start
        self.samples = samples


def build_runner(
    tmp_path,
    store,
    bus,
    session,
    *,
    diarizer=None,
    summarizer=None,
    seconds: float = 4.0,
    every: int = 25,
    config: Config | None = None,
    mic: bool = True,
    loopback: bool = True,
) -> SessionRunner:
    wav = make_wav(tmp_path / "meeting.wav", seconds=seconds)
    counter = iter(range(1, 10_000))
    transcriber = FakeTranscriber(fn=lambda samples: f"intervención {next(counter)}")

    config = config or Config()
    config.summarizer.live_action_items = False

    return SessionRunner(
        session,
        config,
        store,
        bus,
        SystemClock(),
        transcriber=transcriber,
        diarizer=diarizer or OnlineDiarizer(FakeEmbedder()),
        mic_source_factory=build_replay_factory(wav, TrackKind.MIC) if mic else None,
        loopback_source_factory=(
            build_replay_factory(wav, TrackKind.LOOPBACK) if loopback else None
        ),
        vad_factory=lambda _track: PeriodicVad(every),
        summarizer=summarizer,
        realtime=False,
        audio_dir=tmp_path / "audio",
    )


def run_to_completion(runner: SessionRunner, timeout: float = 20.0, **stop_kw):
    runner.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        captures = runner._captures.values()
        if captures and all(c.finished for c in captures):  # type: ignore[attr-defined]
            break
        time.sleep(0.02)
    return runner.stop(**stop_kw)


def types_of(bus) -> list[str]:
    return [e.type for e in bus.history(limit=10_000)]


# ------------------------------------------------------------------ happy path


def test_a_full_session_produces_an_attributed_transcript(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session, summarizer=RecordingSummarizer())
    final = run_to_completion(runner)

    segments = store.get_segments(session.session_id)
    assert len(segments) == 10, "5 utterances per track over 4 s at 25 windows each"
    assert all(s.state is SegmentState.FINAL for s in segments)
    assert all(s.text.startswith("intervención") for s in segments)

    mic = [s for s in segments if s.track is TrackKind.MIC]
    remote = [s for s in segments if s.track is TrackKind.LOOPBACK]
    assert len(mic) == len(remote) == 5
    assert {s.speaker_id for s in mic} == {LOCAL_SPEAKER_ID}
    assert {s.speaker_name for s in mic} == {"Yo"}
    assert {s.speaker_id for s in remote} == {"p1"}
    assert {s.speaker_name for s in remote} == {"Participante 1"}

    assert final.state is SessionState.STOPPED
    assert final.ended_at is not None
    assert store.get_session(session.session_id).state is SessionState.STOPPED


def test_timestamps_are_ordered_and_cover_the_meeting(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session)
    run_to_completion(runner, summarize=False)

    segments = store.get_segments(session.session_id)
    starts = [s.start_s for s in segments]
    assert starts == sorted(starts)
    assert segments[0].start_s == pytest.approx(0.0, abs=0.05)
    assert segments[-1].end_s == pytest.approx(4.0, abs=0.35)


def test_the_panel_receives_every_kind_of_event(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session, summarizer=RecordingSummarizer())
    run_to_completion(runner)

    kinds = set(types_of(bus))
    assert {EV_SESSION, EV_SPEAKER, EV_SEGMENT, EV_SUMMARY, EV_NOTICE} <= kinds

    states = [
        e.data["state"] for e in bus.history(limit=10_000) if e.type == EV_SESSION
    ]
    assert states[0] == "starting"
    assert "running" in states
    assert states[-1] == "stopped"


def test_the_summary_is_generated_and_stored(tmp_path, store, bus, session):
    summarizer = RecordingSummarizer()
    runner = build_runner(tmp_path, store, bus, session, summarizer=summarizer)
    run_to_completion(runner)

    summary = store.get_summary(session.session_id)
    assert summary is not None
    assert summary.executive == "Resumen de prueba."
    assert ("summarize", session.session_id) in summarizer.calls
    # Action items from the summary are recorded for the checklist.
    assert [i.text for i in store.get_action_items(session.session_id)] == ["hacer algo"]


def test_a_single_track_session_still_works(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session, mic=False)
    run_to_completion(runner, summarize=False)

    segments = store.get_segments(session.session_id)
    assert segments and {s.track for s in segments} == {TrackKind.LOOPBACK}
    notices = [
        e.data["code"] for e in bus.history(limit=10_000) if e.type == EV_NOTICE
    ]
    assert "capture.disabled.mic" in notices


# --------------------------------------------------------------- failure paths


def test_a_failing_summarizer_never_costs_the_transcript(tmp_path, store, bus, session):
    class Broken(RecordingSummarizer):
        def summarize(self, request):
            raise SummarizerUnavailableError("api down")

    runner = build_runner(tmp_path, store, bus, session, summarizer=Broken())
    final = run_to_completion(runner)

    assert len(store.get_segments(session.session_id)) == 10
    assert store.get_summary(session.session_id) is None
    assert final.state is SessionState.STOPPED
    codes = [e.data["code"] for e in bus.history(limit=10_000) if e.type == EV_NOTICE]
    assert "summary.failed" in codes


def test_a_transcription_failure_skips_one_utterance_and_continues(
    tmp_path, store, bus, session
):
    class FlakyTranscriber(FakeTranscriber):
        def __init__(self):
            super().__init__()
            self.n = 0

        def transcribe(self, samples, *, language=None, initial_prompt=None):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("ctranslate2 blew up")
            return TranscriptionResult(
                text=f"linea {self.n}", language="es", no_speech_prob=0.01,
                avg_logprob=-0.2, compression_ratio=1.3,
            )

    runner = build_runner(tmp_path, store, bus, session)
    runner.transcriber = FlakyTranscriber()
    run_to_completion(runner, summarize=False)

    segments = store.get_segments(session.session_id)
    assert len(segments) == 9, "exactly one utterance was lost"
    codes = [e.data["code"] for e in bus.history(limit=10_000) if e.type == EV_NOTICE]
    assert "stt.utterance_failed" in codes


def test_hallucinated_output_never_reaches_the_transcript(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session)
    runner.transcriber = FakeTranscriber(
        fn=lambda s: "Subtítulos realizados por la comunidad de Amara.org"
    )
    run_to_completion(runner, summarize=False)

    assert store.get_segments(session.session_id) == []
    assert runner._assembler.rejected == 10
    assert runner.stats()["assembler"]["rejections"]["blocklist"] == 10


def test_no_summarizer_means_transcript_only(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session, summarizer=None)
    run_to_completion(runner)
    assert store.get_segments(session.session_id)
    assert store.get_summary(session.session_id) is None
    codes = [e.data["code"] for e in bus.history(limit=10_000) if e.type == EV_NOTICE]
    assert "summary.disabled" in codes


# --------------------------------------------------------------- speaker flows


def test_renaming_applies_retroactively_and_reaches_the_panel(
    tmp_path, store, bus, session
):
    runner = build_runner(tmp_path, store, bus, session)
    run_to_completion(runner, summarize=False)

    touched = runner.rename_speaker("p1", "María")
    assert touched == 5
    remote = [
        s for s in store.get_segments(session.session_id) if s.track is TrackKind.LOOPBACK
    ]
    assert {s.speaker_name for s in remote} == {"María"}
    speaker_events = [
        e.data for e in bus.history(limit=10_000) if e.type == EV_SPEAKER
    ]
    assert speaker_events[-1]["display_name"] == "María"


def test_degraded_diarization_labels_everything_remoto(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session, diarizer=NullDiarizer())
    run_to_completion(runner, summarize=False)

    remote = [
        s for s in store.get_segments(session.session_id) if s.track is TrackKind.LOOPBACK
    ]
    assert remote
    assert {s.speaker_name for s in remote} == {"Remoto"}


def test_live_action_items_are_deduplicated(tmp_path, store, bus, session):
    summarizer = RecordingSummarizer()
    runner = build_runner(tmp_path, store, bus, session, summarizer=summarizer)
    run_to_completion(runner, summarize=False)

    runner._detect_actions()
    first = store.get_action_items(session.session_id)
    assert [i.text for i in first] == ["compromiso en vivo"]

    runner._pending_action_from = 0.0
    runner._detect_actions()
    assert len(store.get_action_items(session.session_id)) == 1
    published = [e for e in bus.history(limit=10_000) if e.type == EV_ACTION_ITEM]
    assert len(published) == 1


# ------------------------------------------------------------------- shutdown


def test_stop_is_idempotent(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session)
    run_to_completion(runner, summarize=False)
    before = len(store.get_segments(session.session_id))
    runner.stop()
    assert len(store.get_segments(session.session_id)) == before


def test_unconfirmed_drafts_never_survive_into_the_archive(tmp_path, store, bus, session):
    from meetscribe.core.models import TranscriptSegment, new_id

    runner = build_runner(tmp_path, store, bus, session)
    runner.start()
    store.replace_segment(
        TranscriptSegment(
            segment_id=new_id(),
            session_id=session.session_id,
            track=TrackKind.LOOPBACK,
            start_s=99.0,
            end_s=100.0,
            text="texto provisional",
            speaker_id="p1",
            speaker_name="Participante 1",
            seq=999_999,
            state=SegmentState.DRAFT,
        )
    )
    runner.stop(summarize=False)

    texts = [s.text for s in store.get_segments(session.session_id)]
    assert "texto provisional" not in texts


def test_session_audio_is_shredded_when_retention_says_so(tmp_path, store, bus, session):
    config = Config()
    config.capture.retain_audio = "session"
    runner = build_runner(tmp_path, store, bus, session, config=config)
    run_to_completion(runner, summarize=False)

    session_audio = tmp_path / "audio" / session.session_id
    assert not list(session_audio.glob("*.wav")) if session_audio.exists() else True


def test_retained_audio_is_kept_when_asked(tmp_path, store, bus, session):
    config = Config()
    config.capture.retain_audio = "keep"
    runner = build_runner(tmp_path, store, bus, session, config=config)
    run_to_completion(runner, summarize=False)

    files = sorted((tmp_path / "audio" / session.session_id).glob("*.wav"))
    assert [f.name for f in files] == ["loopback.wav", "mic.wav"]
    assert all(f.stat().st_size > 1000 for f in files)


def test_stats_expose_every_stage(tmp_path, store, bus, session):
    runner = build_runner(tmp_path, store, bus, session)
    run_to_completion(runner, summarize=False)
    stats = runner.stats()
    assert set(stats) >= {
        "session_id", "state", "elapsed_s", "tracks", "utterance_queue",
        "assembler", "diarizer", "bus", "errors",
    }
    assert set(stats["tracks"]) == {"mic", "loopback"}
    assert stats["assembler"]["accepted"] == 10
    assert stats["utterance_queue"]["dropped_finals"] == 0
    assert stats["errors"] == []


# ------------------------------------------------------------ crash resilience


def test_an_interrupted_session_is_recovered_at_next_startup(store, session):
    import dataclasses

    store.create_session(dataclasses.replace(session, state=SessionState.RUNNING))
    recovered = recover_interrupted_sessions(store)

    assert [s.session_id for s in recovered] == [session.session_id]
    assert store.get_session(session.session_id).state is SessionState.FAILED
    # Running it again finds nothing left to fix.
    assert recover_interrupted_sessions(store) == []


def test_segments_are_durable_before_the_session_ends(tmp_path, store, bus, session):
    """The crash-at-minute-40 scenario: rows must already be committed."""
    runner = build_runner(tmp_path, store, bus, session)
    runner.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if len(store.get_segments(session.session_id)) >= 2:
            break
        time.sleep(0.02)
    assert len(store.get_segments(session.session_id)) >= 2, "written as they confirm"
    runner.stop(summarize=False)
