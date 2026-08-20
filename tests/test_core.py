"""Domain, timeline and audio primitives.

The timeline tests are the important ones in this file. Drift handling is the
subtlest logic in the project and the only place where being subtly wrong produces
a transcript that looks fine and interleaves the two speakers incorrectly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from meetscribe.audio.buffers import ChunkQueue
from meetscribe.audio.level import (
    DIGITAL_SILENCE_DBFS,
    SilenceWatchdog,
    WatchdogVerdict,
    peak_level,
    rms_dbfs,
)
from meetscribe.audio.resample import StreamResampler, downmix, resample_offline, to_float32
from meetscribe.audio.wav import read_wav_mono, write_wav
from meetscribe.core.clock import FakeClock
from meetscribe.core.events import Event, make_notice, make_segment
from meetscribe.core.models import (
    PIPELINE_SAMPLE_RATE,
    ActionItem,
    AudioChunk,
    AudioFormat,
    SegmentState,
    Session,
    SessionState,
    Speaker,
    TrackKind,
    TranscriptSegment,
    new_id,
    normalise_for_dedup,
)
from meetscribe.core.timeline import TrackTimeline

from .conftest import chunk, silence, tone

# ------------------------------------------------------------------ value objects


def test_session_slug_is_filesystem_safe():
    assert Session.slug("Reunión de Diseño #3") == "reunion-de-diseno-3"
    assert Session.slug("   ") == "sesion"
    assert len(Session.slug("x" * 200)) <= 60


def test_track_kind_locality():
    assert TrackKind.MIC.is_local
    assert not TrackKind.LOOPBACK.is_local


def test_audio_format_rejects_nonsense():
    with pytest.raises(ValueError):
        AudioFormat(sample_rate=0, channels=2)
    with pytest.raises(ValueError):
        AudioFormat(sample_rate=16000, channels=0)
    assert AudioFormat(16000, 1).is_pipeline_native
    assert not AudioFormat(48000, 2).is_pipeline_native


def test_audio_chunk_derives_its_own_duration():
    c = AudioChunk(TrackKind.MIC, tone(1.5), start_s=10.0)
    assert c.duration_s == pytest.approx(1.5)
    assert c.end_s == pytest.approx(11.5)


def test_action_item_fingerprint_folds_accents_and_punctuation():
    a = ActionItem(text="Enviar el informe, por favor.", owner="María")
    b = ActionItem(text="enviar el informe por favor", owner="Maria")
    assert a.fingerprint == b.fingerprint


def test_speaker_rename_rejects_blank():
    speaker = Speaker("p1", "Participante 1", TrackKind.LOOPBACK)
    assert speaker.renamed("  Ana  ").display_name == "Ana"
    with pytest.raises(ValueError):
        speaker.renamed("   ")


def test_session_state_terminality():
    assert SessionState.STOPPED.is_terminal
    assert SessionState.FAILED.is_terminal
    assert not SessionState.RUNNING.is_terminal


def test_new_id_is_unique_enough():
    assert len({new_id() for _ in range(2000)}) == 2000


def test_normalise_for_dedup():
    assert normalise_for_dedup("  ¿Qué   tal? ") == "que tal"


# ---------------------------------------------------------------------- events


def test_event_json_round_trip():
    segment = TranscriptSegment(
        segment_id="s1",
        session_id="x",
        track=TrackKind.MIC,
        start_s=1.234567,
        end_s=2.5,
        text="hola",
        speaker_id="yo",
        speaker_name="Yo",
        seq=7,
    )
    event = Event(seq=3, type="segment", at=1.0, data=make_segment(segment))
    import json

    decoded = json.loads(event.to_json())
    assert decoded["data"]["start_s"] == 1.235
    assert decoded["data"]["text"] == "hola"
    assert decoded["seq"] == 3


def test_notice_rejects_unknown_level():
    make_notice("warn", "x", "y")
    with pytest.raises(ValueError):
        make_notice("critical", "x", "y")


def test_level_and_heartbeat_are_volatile_but_segments_are_not():
    assert Event(1, "level", 0.0, {}).is_volatile
    assert Event(1, "heartbeat", 0.0, {}).is_volatile
    assert not Event(1, "segment", 0.0, {}).is_volatile


# -------------------------------------------------------------------- timeline


def test_timeline_offline_mode_is_pure_sample_counting():
    """A file replay has no clock domain, so wall time must be ignored entirely."""
    tl = TrackTimeline(PIPELINE_SAMPLE_RATE, realtime=False)
    assert tl.stamp(16000, arrival_s=0.0) == 0.0
    assert tl.stamp(16000, arrival_s=0.001) == pytest.approx(1.0)
    assert tl.stamp(16000, arrival_s=0.002) == pytest.approx(2.0)
    assert tl.media_s == pytest.approx(3.0)
    assert tl.reanchors == 0


def test_timeline_first_chunk_sets_the_track_offset():
    """The two tracks start at different instants; the first chunk records that."""
    tl = TrackTimeline(PIPELINE_SAMPLE_RATE)
    start = tl.stamp(16000, arrival_s=4.0)  # a 1 s chunk that arrived at t=4
    assert start == pytest.approx(3.0)


def test_timeline_is_smooth_despite_arrival_jitter():
    """Sample counting must win locally: scheduling jitter must not move segments."""
    tl = TrackTimeline(PIPELINE_SAMPLE_RATE)
    tl.stamp(1600, arrival_s=0.1)
    starts = []
    for i in range(1, 20):
        jitter = 0.02 if i % 2 else -0.02
        starts.append(tl.stamp(1600, arrival_s=0.1 * (i + 1) + jitter))
    deltas = np.diff(starts)
    assert np.allclose(deltas, 0.1, atol=1e-9)
    assert tl.reanchors == 0


def test_timeline_reanchors_when_drift_exceeds_tolerance():
    """A device running fast eventually pulls sample time away from wall time."""
    tl = TrackTimeline(PIPELINE_SAMPLE_RATE, max_drift_s=0.2)
    tl.stamp(1600, arrival_s=0.1)
    # The device delivers 0.1 s of audio every 0.09 s of wall time: 11 % fast.
    arrival = 0.1
    for _ in range(40):
        arrival += 0.09
        tl.stamp(1600, arrival_s=arrival)
    assert tl.reanchors >= 1
    assert abs(tl.drift_s) <= 0.2 + 1e-6


def test_timeline_skip_keeps_later_timestamps_true():
    """Dropped audio must advance the counter, or everything after shifts earlier."""
    tl = TrackTimeline(PIPELINE_SAMPLE_RATE, realtime=False)
    tl.stamp(16000, 0.0)
    tl.skip(16000)  # one second lost
    assert tl.stamp(16000, 0.0) == pytest.approx(2.0)
    assert tl.dropped_samples == 16000


def test_timeline_rejects_bad_input():
    with pytest.raises(ValueError):
        TrackTimeline(0)
    with pytest.raises(ValueError):
        TrackTimeline(16000, max_drift_s=0)
    with pytest.raises(ValueError):
        TrackTimeline(16000).stamp(0, 0.0)


# ------------------------------------------------------------------ resampling


def test_to_float32_handles_every_supported_dtype():
    assert to_float32(np.array([32767, -32768], dtype=np.int16))[0] == pytest.approx(
        0.99997, abs=1e-4
    )
    assert to_float32(np.array([128], dtype=np.uint8))[0] == pytest.approx(0.0)
    with pytest.raises(TypeError):
        to_float32(np.array(["x"]))


def test_downmix_averages_channels():
    interleaved = np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(downmix(interleaved, 2), [0.5, 0.5])


def test_downmix_drops_a_partial_trailing_frame():
    interleaved = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    assert downmix(interleaved, 2).size == 1


def test_resample_offline_hits_the_expected_length_and_keeps_the_tone():
    source = np.sin(
        2 * np.pi * 440 * np.arange(48000) / 48000
    ).astype(np.float32)
    out = resample_offline(source, 48000, 16000)
    assert abs(out.size - 16000) <= 2
    # The 440 Hz component must survive: check the peak of the spectrum.
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(out.size, 1 / 16000)[int(np.argmax(spectrum))]
    assert abs(peak_hz - 440) < 5


def test_stream_resampler_is_continuous_across_chunks():
    """Chunked resampling must equal whole-signal resampling, or every chunk
    boundary lands a filter transient in the middle of a word."""
    source = np.sin(2 * np.pi * 300 * np.arange(48000 * 2) / 48000).astype(np.float32)
    streamer = StreamResampler(48000, 1)
    pieces = [streamer.process(source[i : i + 4800]) for i in range(0, source.size, 4800)]
    pieces.append(streamer.flush())
    streamed = np.concatenate([p for p in pieces if p.size])
    whole = resample_offline(source, 48000, 16000)

    assert abs(streamed.size - whole.size) < 200
    n = min(streamed.size, whole.size) - 200
    # Compare away from the edges, where the two paths' filter states differ.
    assert np.allclose(streamed[100 : n], whole[100 : n], atol=0.02)


def test_stream_resampler_passthrough_when_rates_match():
    streamer = StreamResampler(16000, 2)
    out = streamer.process(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(out, [0.5, 0.5])
    assert streamer.flush().size == 0


def test_stream_resampler_refuses_use_after_flush():
    streamer = StreamResampler(48000, 1)
    streamer.flush()
    with pytest.raises(RuntimeError):
        streamer.process(np.zeros(480, dtype=np.float32))


# ------------------------------------------------------------------------ wav


def test_wav_round_trip(tmp_path):
    original = tone(0.5, 440.0)
    path = tmp_path / "a.wav"
    write_wav(path, original)
    restored = read_wav_mono(path)
    assert restored.size == original.size
    assert np.allclose(restored, original, atol=1e-3)


# --------------------------------------------------------------------- levels


def test_rms_and_peak():
    assert rms_dbfs(np.zeros(0, dtype=np.float32)) == -240.0
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) < DIGITAL_SILENCE_DBFS
    assert peak_level(tone(0.1, amplitude=0.4)) == pytest.approx(0.4, abs=0.01)
    assert isinstance(rms_dbfs(tone(0.1)), float)


def test_watchdog_fires_once_then_recovers():
    """Edge-triggered: a level-triggered warning would repeat forever and be
    ignored, which is the same as not warning at all."""
    watchdog = SilenceWatchdog(window_s=10.0, recover_s=1.0)
    assert watchdog.observe(-200.0, 0.0) == WatchdogVerdict.NONE
    assert watchdog.observe(-200.0, 9.9) == WatchdogVerdict.NONE
    assert watchdog.observe(-200.0, 10.0) == WatchdogVerdict.WENT_SILENT
    assert watchdog.observe(-200.0, 20.0) == WatchdogVerdict.NONE  # never repeats
    assert watchdog.alerted

    assert watchdog.observe(-30.0, 21.0) == WatchdogVerdict.NONE  # not yet sustained
    assert watchdog.observe(-30.0, 22.5) == WatchdogVerdict.RECOVERED
    assert not watchdog.alerted


def test_watchdog_ignores_a_single_blip_on_a_dead_device():
    watchdog = SilenceWatchdog(window_s=5.0, recover_s=2.0)
    watchdog.observe(-200.0, 0.0)
    watchdog.observe(-200.0, 5.0)
    assert watchdog.alerted
    watchdog.observe(-20.0, 6.0)      # one burst of interface noise
    assert watchdog.observe(-200.0, 6.5) == WatchdogVerdict.NONE
    assert watchdog.alerted           # still flagged, correctly


# ------------------------------------------------------------------- buffers


def test_chunk_queue_fifo_and_accounting():
    queue = ChunkQueue(max_seconds=10)
    queue.put(chunk(TrackKind.MIC, 0.0, tone(1.0)))
    queue.put(chunk(TrackKind.MIC, 1.0, tone(1.0)))
    first = queue.get(timeout=0)
    assert first is not None and first.start_s == 0.0
    assert queue.queued_seconds == pytest.approx(1.0)


def test_chunk_queue_drops_the_oldest_under_pressure():
    """Stale audio is worthless in a live panel; the newest must survive."""
    queue = ChunkQueue(max_seconds=2)
    for i in range(5):
        queue.put(chunk(TrackKind.MIC, float(i), tone(1.0)))
    remaining = []
    while (item := queue.get(timeout=0)) is not None:
        remaining.append(item.start_s)
    assert remaining == [3.0, 4.0]
    assert queue.dropped_samples == 3 * PIPELINE_SAMPLE_RATE
    assert queue.drop_events > 0


def test_chunk_queue_close_wakes_the_consumer():
    queue = ChunkQueue()
    queue.close()
    assert queue.get(timeout=0.01) is None
    assert queue.closed
    assert queue.put(chunk(TrackKind.MIC, 0.0, tone(0.1))) == 0


def test_chunk_queue_ignores_empty_chunks():
    queue = ChunkQueue()
    assert queue.put(chunk(TrackKind.MIC, 0.0, silence(0.0))) == 0
    assert len(queue) == 0


# ---------------------------------------------------------------------- clock


def test_fake_clock_advances_instead_of_blocking():
    clock = FakeClock(start=5.0, epoch=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    clock.sleep(30.0)
    assert clock.monotonic() == 35.0
    assert clock.now() == dt.datetime(2026, 1, 1, 0, 0, 35, tzinfo=dt.UTC)
    assert clock.sleeps == [30.0]


def test_segment_state_defaults_to_final():
    segment = TranscriptSegment(
        segment_id="s",
        session_id="x",
        track=TrackKind.MIC,
        start_s=0,
        end_s=1,
        text="t",
        speaker_id="yo",
        speaker_name="Yo",
        seq=1,
    )
    assert segment.state is SegmentState.FINAL
    renamed = segment.with_speaker(Speaker("yo", "Ana", TrackKind.MIC, is_local=True))
    assert renamed.speaker_name == "Ana"
    assert renamed.text == "t"


# --------------------------------------------------------------- port conformance


def test_the_real_adapters_satisfy_the_ports_they_are_written_against(tmp_path):
    """The ports must describe the adapters, not an earlier draft of them.

    A Protocol that omits methods the pipeline actually calls is worse than no
    Protocol: a fake built from it type-checks and then fails at runtime on the
    first missing call. These are ``runtime_checkable``, so this asserts that every
    declared member really exists on the shipping implementation.
    """
    from meetscribe.bus.pubsub import InMemoryEventBus
    from meetscribe.core.clock import SystemClock
    from meetscribe.core.ports import Clock, EventBus, Subscription, TranscriptStore
    from meetscribe.store.sqlite_store import SqliteStore

    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock(), Clock)

    bus = InMemoryEventBus()
    try:
        assert isinstance(bus, EventBus)
        subscription = bus.subscribe()
        try:
            assert isinstance(subscription, Subscription)
        finally:
            subscription.close()
    finally:
        bus.close()

    with SqliteStore(tmp_path / "ports.db") as store:
        assert isinstance(store, TranscriptStore)
