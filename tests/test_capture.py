"""Capture: conversion, stamping, backpressure, reconnection and the watchdog.

Run entirely against injected sources. The WASAPI and soundcard backends are thin
enough that the interesting behaviour -- what happens when a device disappears,
when a track goes silent, when the consumer falls behind -- lives in
:class:`TrackCapture`, and that is what is exercised here.
"""

from __future__ import annotations

import itertools
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from meetscribe.audio.buffers import ChunkQueue
from meetscribe.audio.level import SilenceWatchdog
from meetscribe.audio.wav import WavWriter
from meetscribe.capture.devices import DeviceInfo, pick_device
from meetscribe.capture.sources import (
    FallbackSource,
    SyntheticSource,
    WavFileSource,
)
from meetscribe.capture.track import TrackCapture
from meetscribe.core.clock import SystemClock
from meetscribe.core.models import PIPELINE_SAMPLE_RATE, AudioFormat, TrackKind
from meetscribe.errors import CaptureError, DeviceLostError, DeviceNotFoundError


def write_stereo_wav(path: Path, seconds: float = 1.0, rate: int = 48000) -> Path:
    """A real 48 kHz stereo file, so the test drives downmix and resampling."""
    t = np.arange(int(seconds * rate)) / rate
    left = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    interleaved = np.empty(left.size * 2, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(interleaved, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def drain(queue: ChunkQueue) -> list:
    out = []
    while (chunk := queue.get(timeout=0)) is not None:
        out.append(chunk)
    return out


# --------------------------------------------------------------------- sources


def test_wav_source_reports_the_files_native_format(tmp_path):
    path = write_stereo_wav(tmp_path / "a.wav")
    source = WavFileSource(path, TrackKind.LOOPBACK)
    assert source.open() == AudioFormat(sample_rate=48000, channels=2)
    assert source.read() is not None
    source.close()


def test_wav_source_ends_cleanly(tmp_path):
    path = write_stereo_wav(tmp_path / "a.wav", seconds=0.05)
    source = WavFileSource(path, TrackKind.MIC, block_frames=1024)
    source.open()
    while source.read() is not None:
        pass
    assert source.read() is None


def test_wav_source_explains_a_missing_file(tmp_path):
    source = WavFileSource(tmp_path / "nope.wav", TrackKind.MIC)
    with pytest.raises(CaptureError, match="does not exist"):
        source.open()


def test_synthetic_source_is_deterministic():
    a = SyntheticSource(TrackKind.MIC, total_blocks=4)
    b = SyntheticSource(TrackKind.MIC, total_blocks=4)
    a.open()
    b.open()
    assert np.array_equal(a.read(), b.read())
    assert a.read() is not None and a.read() is not None
    a.read()
    assert a.read() is None


def test_fallback_source_uses_the_second_backend_when_the_first_fails():
    class Broken:
        name = "roto"

        def open(self):
            raise CaptureError("exclusive mode")

        def read(self):
            raise AssertionError("must not be used")

        def close(self):
            return None

    good = SyntheticSource(TrackKind.MIC, total_blocks=2)
    source = FallbackSource([Broken(), good])
    assert source.open().sample_rate == 48000
    assert source.name == "sintético"
    assert source.read() is not None
    source.close()


def test_fallback_source_reports_every_failure_when_none_work():
    class Broken:
        name = "roto"

        def open(self):
            raise CaptureError("nope")

        def read(self):
            return None

        def close(self):
            return None

    with pytest.raises(CaptureError, match="nope") as excinfo:
        FallbackSource([Broken(), Broken()]).open()
    assert "Ningún backend" in excinfo.value.user_message


def test_fallback_source_needs_a_candidate():
    with pytest.raises(CaptureError):
        FallbackSource([])


# ------------------------------------------------------------ device selection


DEVICES = [
    DeviceInfo(1, "Microphone Array (AMD Audio Device)", 2, 48000, is_default=True),
    DeviceInfo(2, "Webcam Mic", 1, 44100),
    DeviceInfo(10, "Altavoces (Realtek(R) Audio) [Loopback]", 2, 48000,
               is_loopback=True, is_default=True),
    DeviceInfo(11, "Auriculares (Bluetooth) [Loopback]", 2, 48000, is_loopback=True),
]


def test_default_selection_prefers_the_default_endpoint():
    assert pick_device(None, TrackKind.MIC, DEVICES).index == 1
    assert pick_device(None, TrackKind.LOOPBACK, DEVICES).index == 10


def test_selection_by_index_and_by_name():
    assert pick_device(11, TrackKind.LOOPBACK, DEVICES).index == 11
    assert pick_device("webcam", TrackKind.MIC, DEVICES).index == 2
    assert pick_device("2", TrackKind.MIC, DEVICES).index == 2


def test_an_ambiguous_name_is_refused_rather_than_guessed():
    """Picking the wrong endpoint yields a plausible session with the wrong half
    of the conversation in it."""
    with pytest.raises(DeviceNotFoundError, match="ambiguous"):
        pick_device("loopback", TrackKind.LOOPBACK, DEVICES)


def test_unknown_device_specs_are_reported_clearly():
    with pytest.raises(DeviceNotFoundError, match="99"):
        pick_device(99, TrackKind.MIC, DEVICES)
    with pytest.raises(DeviceNotFoundError):
        pick_device("no existe", TrackKind.MIC, DEVICES)


def test_device_format_caps_channels_at_stereo():
    surround = DeviceInfo(3, "Surround", 8, 48000)
    assert surround.format.channels == 2
    assert "8ch" in surround.describe()


# ------------------------------------------------------------------- the track


def test_capture_converts_to_pipeline_format_and_stamps_the_timeline(tmp_path):
    path = write_stereo_wav(tmp_path / "a.wav", seconds=1.0)
    queue = ChunkQueue(max_seconds=10)
    capture = TrackCapture(
        TrackKind.LOOPBACK,
        lambda: WavFileSource(path, TrackKind.LOOPBACK),
        queue,
        SystemClock(),
        realtime=False,
    )
    capture.start()
    deadline = time.monotonic() + 5
    while not capture.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()

    chunks = drain(queue)
    assert chunks, "the file should have produced audio"
    total = sum(c.samples.size for c in chunks)
    # 1 s of 48 kHz stereo becomes ~1 s of 16 kHz mono.
    assert abs(total - PIPELINE_SAMPLE_RATE) < 400
    assert all(c.sample_rate == PIPELINE_SAMPLE_RATE for c in chunks)
    assert chunks[0].start_s == 0.0
    # Contiguous: each chunk begins where the previous one ended.
    for previous, following in itertools.pairwise(chunks):
        assert following.start_s == pytest.approx(previous.end_s, abs=1e-6)


def test_capture_reports_levels_and_never_flags_a_live_track(tmp_path):
    path = write_stereo_wav(tmp_path / "a.wav", seconds=1.0)
    levels, notices = [], []
    queue = ChunkQueue()
    capture = TrackCapture(
        TrackKind.MIC,
        lambda: WavFileSource(path, TrackKind.MIC),
        queue,
        SystemClock(),
        realtime=False,
        level_interval_s=0.1,
        on_level=levels.append,
        on_notice=lambda level, code, message: notices.append((level, code)),
    )
    capture.start()
    deadline = time.monotonic() + 5
    while not capture.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()

    assert len(levels) >= 5
    assert all(reading.dbfs > -40 for reading in levels)
    assert not [n for n in notices if n[1].startswith("capture.silent")]
    stats = capture.stats().as_dict()
    assert stats["track"] == "mic"
    assert stats["dropped_samples"] == 0


def test_a_silent_track_raises_exactly_one_warning(tmp_path):
    """The spike's microphone opened fine and returned digital zero. That must be
    visible in seconds, and must not then repeat forever."""
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros(48000 * 3, dtype="<i2").tobytes())

    notices = []
    capture = TrackCapture(
        TrackKind.MIC,
        lambda: WavFileSource(path, TrackKind.MIC),
        ChunkQueue(),
        SystemClock(),
        realtime=False,
        level_interval_s=0.1,
        watchdog=SilenceWatchdog(window_s=0.5),
        on_notice=lambda level, code, message: notices.append((level, code, message)),
    )
    capture.start()
    deadline = time.monotonic() + 5
    while not capture.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()

    silent = [n for n in notices if n[1] == "capture.silent.mic"]
    assert len(silent) == 1
    assert "Privacidad" in silent[0][2]
    assert capture.stats().silent


def test_a_lost_device_is_reconnected_without_ending_the_session():
    """Headphones plugged in mid-meeting must not end the recording."""
    attempts = {"n": 0}

    class Flaky:
        name = "inestable"

        def __init__(self):
            self.reads = 0
            attempts["n"] += 1
            self.generation = attempts["n"]

        def open(self):
            return AudioFormat(sample_rate=16000, channels=1)

        def read(self):
            self.reads += 1
            if self.generation == 1 and self.reads > 3:
                raise DeviceLostError("endpoint changed")
            return np.full(1600, 0.1, dtype=np.float32)

        def close(self):
            return None

    notices = []
    queue = ChunkQueue()
    capture = TrackCapture(
        TrackKind.LOOPBACK,
        Flaky,
        queue,
        SystemClock(),
        realtime=False,
        on_notice=lambda level, code, message: notices.append((level, code)),
    )
    capture.start()
    # The first retry backs off 0.5 s, so give it room to complete.
    deadline = time.monotonic() + 4
    while capture.stats().reconnects == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.8)
    capture.stop()

    codes = [c for _, c in notices]
    assert "capture.reconnecting" in codes
    assert "capture.reconnected" in codes
    assert attempts["n"] >= 2
    assert capture.stats().reconnects >= 1
    assert drain(queue), "audio kept flowing after the reconnect"


def test_reconnection_gives_up_and_says_so_rather_than_spinning():
    class AlwaysBroken:
        name = "muerto"

        def open(self):
            raise CaptureError("gone")

        def read(self):
            return None

        def close(self):
            return None

    notices = []
    capture = TrackCapture(
        TrackKind.MIC,
        AlwaysBroken,
        ChunkQueue(),
        SystemClock(),
        realtime=False,
        max_reconnects=2,
        on_notice=lambda level, code, message: notices.append((level, code)),
    )
    with pytest.raises(CaptureError):
        capture.start()  # the first open happens on the caller's thread
    capture.stop()


def test_backpressure_drops_audio_loudly_not_silently(tmp_path):
    """A live endpoint cannot be paused, so a full queue must drop -- and say so."""
    path = write_stereo_wav(tmp_path / "long.wav", seconds=3.0)
    notices = []
    queue = ChunkQueue(max_seconds=0.5)  # deliberately tiny; nothing consumes it
    capture = TrackCapture(
        TrackKind.LOOPBACK,
        lambda: WavFileSource(path, TrackKind.LOOPBACK),
        queue,
        SystemClock(),
        realtime=True,
        on_notice=lambda level, code, message: notices.append((level, code)),
    )
    capture.start()
    deadline = time.monotonic() + 5
    while not capture.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()

    assert queue.dropped_samples > 0
    assert ("warn", "capture.backpressure") in notices


def test_file_replay_is_lossless_even_with_a_tiny_buffer(tmp_path):
    """The opposite policy for the opposite situation.

    A file can be read more slowly, so replay must wait for room rather than drop.
    The first end-to-end run against real audio lost its opening eleven seconds to
    exactly this; the queue now blocks instead.
    """
    path = write_stereo_wav(tmp_path / "long.wav", seconds=3.0)
    queue = ChunkQueue(max_seconds=0.5)
    capture = TrackCapture(
        TrackKind.LOOPBACK,
        lambda: WavFileSource(path, TrackKind.LOOPBACK),
        queue,
        SystemClock(),
        realtime=False,
    )
    capture.start()

    consumed = 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        chunk = queue.get(timeout=0.05)
        if chunk is not None:
            consumed += chunk.samples.size
        elif capture.finished and len(queue) == 0:
            break
    capture.stop()
    while (chunk := queue.get(timeout=0)) is not None:
        consumed += chunk.samples.size

    assert queue.dropped_samples == 0
    assert abs(consumed - 3 * PIPELINE_SAMPLE_RATE) < 600


def test_capture_can_retain_raw_audio(tmp_path):
    source = tmp_path / "src.wav"
    write_stereo_wav(source, seconds=0.5)
    recorded = tmp_path / "out" / "loopback.wav"
    capture = TrackCapture(
        TrackKind.LOOPBACK,
        lambda: WavFileSource(source, TrackKind.LOOPBACK),
        ChunkQueue(),
        SystemClock(),
        realtime=False,
        recorder=WavWriter(recorded),
    )
    capture.start()
    deadline = time.monotonic() + 5
    while not capture.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()

    assert recorded.exists()
    with wave.open(str(recorded), "rb") as w:
        assert w.getframerate() == PIPELINE_SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getnframes() > 0


def test_starting_a_track_twice_is_refused(tmp_path):
    path = write_stereo_wav(tmp_path / "a.wav", seconds=0.2)
    capture = TrackCapture(
        TrackKind.MIC,
        lambda: WavFileSource(path, TrackKind.MIC),
        ChunkQueue(),
        SystemClock(),
        realtime=False,
    )
    capture.start()
    with pytest.raises(CaptureError, match="already started"):
        capture.start()
    capture.stop()
