"""Concrete :class:`~meetscribe.core.ports.AudioSource` implementations.

Four of them, all interchangeable behind the port:

* :class:`WasapiSource` -- the real one. PyAudioWPatch, WASAPI shared mode,
  blocking reads.
* :class:`SoundcardSource` -- the fallback, used when PyAudioWPatch cannot open an
  endpoint. The spike confirmed ``soundcard`` reaches the same loopback device by
  a different route, so a driver quirk that breaks one backend need not end the
  meeting.
* :class:`WavFileSource` -- replays a recording. Non-realtime, which makes an
  end-to-end pipeline test over a known meeting fully deterministic.
* :class:`SyntheticSource` -- generated tones and silence, for tests that must not
  touch the filesystem either.

None of them resample or downmix: they hand up frames in the device's own format
and let :class:`~meetscribe.capture.track.TrackCapture` convert, so all four share
one conversion path and one set of tests.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from meetscribe.audio.wav import read_wav
from meetscribe.capture.devices import DeviceInfo, default_endpoint_indices
from meetscribe.core.models import AudioFormat, TrackKind
from meetscribe.errors import CaptureError, DeviceLostError

log = logging.getLogger(__name__)

__all__ = [
    "WasapiSource",
    "SoundcardSource",
    "WavFileSource",
    "SyntheticSource",
    "FallbackSource",
]


class FallbackSource:
    """Tries several backends in order and keeps the first that opens.

    Backend choice cannot be made up front, because whether PyAudioWPatch can open
    a given endpoint depends on driver state at that instant -- another application
    holding it in exclusive mode, a device mid-reset. The spike confirmed
    ``soundcard`` reaches the same loopback endpoint by a different route, so
    trying it costs one failed open and can save the meeting.

    Reconnection re-runs the whole ladder, so a track that fell back to the
    secondary backend returns to the primary when the endpoint frees up.
    """

    __slots__ = ("candidates", "_active")

    def __init__(self, candidates) -> None:
        if not candidates:
            raise CaptureError("FallbackSource needs at least one candidate")
        self.candidates = list(candidates)
        self._active = None

    @property
    def name(self) -> str:
        return self._active.name if self._active else self.candidates[0].name

    def open(self):
        errors = []
        for candidate in self.candidates:
            try:
                fmt = candidate.open()
            except Exception as exc:
                errors.append(f"{type(candidate).__name__}: {exc}")
                continue
            self._active = candidate
            if errors:
                import logging

                logging.getLogger(__name__).warning(
                    "primary capture backend failed (%s); using %s",
                    "; ".join(errors),
                    type(candidate).__name__,
                )
            return fmt
        raise CaptureError(
            "; ".join(errors),
            user_message=(
                "Ningún backend de audio pudo abrir el dispositivo. Ejecuta "
                "'meetscribe doctor' para ver el detalle."
            ),
        )

    def read(self):
        if self._active is None:
            raise CaptureError("read() before open()")
        return self._active.read()

    def close(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None


class WasapiSource:
    """Blocking WASAPI capture from one endpoint.

    Owns its own PortAudio instance. That is not tidiness: PortAudio snapshots the
    device table at initialisation, so noticing that the default render endpoint
    changed -- the user plugged in headphones -- requires tearing the instance down
    and building a new one. Making that a per-source operation keeps the two tracks
    independent, so reconnecting the loopback never interrupts the microphone.
    """

    __slots__ = (
        "device",
        "track",
        "frames_per_buffer",
        "follow_default",
        "_pa",
        "_stream",
        "_format",
        "_expected_default",
        "_reads_since_check",
        "_check_every",
    )

    def __init__(
        self,
        device: DeviceInfo,
        track: TrackKind,
        frames_per_buffer: int = 1024,
        follow_default: bool = True,
    ) -> None:
        self.device = device
        self.track = track
        self.frames_per_buffer = frames_per_buffer
        self.follow_default = follow_default
        self._pa: Any = None
        self._stream: Any = None
        self._format: AudioFormat | None = None
        self._expected_default = -1
        self._reads_since_check = 0
        # ~1 s between endpoint checks at 1024 frames / 48 kHz. Cheap enough to be
        # invisible, frequent enough that a device switch costs one second of audio.
        self._check_every = max(1, int(48000 / max(frames_per_buffer, 1)))

    @property
    def name(self) -> str:
        return self.device.name

    def open(self) -> AudioFormat:
        import pyaudiowpatch as pyaudio

        fmt = self.device.format
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=fmt.channels,
                rate=fmt.sample_rate,
                frames_per_buffer=self.frames_per_buffer,
                input=True,
                input_device_index=self.device.index,
            )
        except Exception as exc:
            self.close()
            raise CaptureError(
                f"cannot open {self.device.name!r}: {type(exc).__name__}: {exc}",
                user_message=(
                    f"No se pudo abrir el dispositivo '{self.device.name}'. "
                    "Puede estar en uso exclusivo por otra aplicación."
                ),
            ) from exc

        self._format = fmt
        capture_default, render_default = self._read_defaults()
        self._expected_default = (
            render_default if self.device.is_loopback else capture_default
        )
        return fmt

    def read(self) -> np.ndarray | None:
        if self._stream is None or self._format is None:
            raise CaptureError("read() before open()")
        try:
            raw = self._stream.read(
                self.frames_per_buffer, exception_on_overflow=False
            )
        except Exception as exc:
            raise DeviceLostError(
                f"{self.device.name!r} read failed: {type(exc).__name__}: {exc}"
            ) from exc

        self._reads_since_check += 1
        if self.follow_default and self._reads_since_check >= self._check_every:
            self._reads_since_check = 0
            self._raise_if_default_changed()

        return np.frombuffer(raw, dtype=np.float32)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as exc:
                log.debug("closing stream for %s failed: %s", self.name, exc)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as exc:
                log.debug("terminating PortAudio failed: %s", exc)
            self._pa = None

    # ------------------------------------------------------------------ helpers

    def _read_defaults(self) -> tuple[int, int]:
        try:
            return default_endpoint_indices()
        except Exception:
            return (-1, -1)

    def _raise_if_default_changed(self) -> None:
        """Detect a hot-swap that leaves the old stream happily reading silence.

        This is the failure the level meter alone would not explain: unplugging
        headphones does not error the old stream, it just stops feeding it audio.
        Comparing against the current default endpoint catches it in about a
        second, and the manager reopens against whatever the user is now hearing.
        """
        capture_default, render_default = self._read_defaults()
        current = render_default if self.device.is_loopback else capture_default
        if current < 0 or self._expected_default < 0:
            return
        if current != self._expected_default:
            raise DeviceLostError(
                f"default {'render' if self.device.is_loopback else 'capture'} "
                f"endpoint changed {self._expected_default} -> {current}"
            )


class SoundcardSource:
    """Fallback backend. Same endpoint, different library.

    ``soundcard`` addresses loopback by speaker *name* rather than by a patched
    device index, so it survives some driver states that leave PyAudioWPatch
    unable to open the endpoint.
    """

    __slots__ = ("device", "track", "block_frames", "_recorder", "_ctx", "_format")

    def __init__(
        self, device: DeviceInfo, track: TrackKind, block_frames: int = 1024
    ) -> None:
        self.device = device
        self.track = track
        self.block_frames = block_frames
        self._recorder: Any = None
        self._ctx: Any = None
        self._format: AudioFormat | None = None

    @property
    def name(self) -> str:
        return f"{self.device.name} (soundcard)"

    def open(self) -> AudioFormat:
        try:
            import soundcard as sc
        except ImportError as exc:  # pragma: no cover - optional backend
            raise CaptureError(
                f"soundcard unavailable: {exc}",
                user_message="El backend de respaldo 'soundcard' no está instalado.",
            ) from exc

        fmt = self.device.format
        try:
            if self.device.is_loopback:
                speaker = sc.default_speaker()
                handle = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            else:
                handle = sc.default_microphone()
            self._ctx = handle.recorder(
                samplerate=fmt.sample_rate,
                channels=fmt.channels,
                blocksize=self.block_frames,
            )
            self._recorder = self._ctx.__enter__()
        except Exception as exc:
            self.close()
            raise CaptureError(
                f"soundcard cannot open {self.device.name!r}: {exc}",
                user_message=(
                    f"El backend de respaldo tampoco pudo abrir "
                    f"'{self.device.name}'."
                ),
            ) from exc
        self._format = fmt
        return fmt

    def read(self) -> np.ndarray | None:
        if self._recorder is None:
            raise CaptureError("read() before open()")
        try:
            block = self._recorder.record(numframes=self.block_frames)
        except Exception as exc:
            raise DeviceLostError(f"soundcard read failed: {exc}") from exc
        return np.asarray(block, dtype=np.float32).reshape(-1)

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception as exc:
                log.debug("closing soundcard recorder failed: %s", exc)
            self._ctx = None
        self._recorder = None


class WavFileSource:
    """Replay a WAV file as if it were a device.

    ``realtime=False`` (the default) reads as fast as the consumer drains, which is
    what makes a full-pipeline regression test finish in seconds while producing
    exactly the timestamps a real capture would have produced. ``realtime=True``
    paces the reads against a clock, for manually eyeballing the live panel with a
    recorded meeting.
    """

    __slots__ = (
        "path",
        "track",
        "block_frames",
        "realtime",
        "_samples",
        "_rate",
        "_channels",
        "_pos",
        "_clock",
        "_started",
    )

    def __init__(
        self,
        path: str | Path,
        track: TrackKind,
        block_frames: int = 1024,
        realtime: bool = False,
        clock: Any = None,
    ) -> None:
        self.path = Path(path)
        self.track = track
        self.block_frames = block_frames
        self.realtime = realtime
        self._samples = np.zeros(0, dtype=np.float32)
        self._rate = 0
        self._channels = 1
        self._pos = 0
        self._clock = clock
        self._started = 0.0

    @property
    def name(self) -> str:
        return f"{self.path.name} (archivo)"

    def open(self) -> AudioFormat:
        if not self.path.exists():
            raise CaptureError(
                f"{self.path} does not exist",
                user_message=f"No existe el archivo de audio '{self.path}'.",
            )
        self._samples, self._rate, self._channels = read_wav(self.path)
        self._pos = 0
        if self.realtime and self._clock is not None:
            self._started = self._clock.monotonic()
        return AudioFormat(sample_rate=self._rate, channels=self._channels)

    def read(self) -> np.ndarray | None:
        if self._rate == 0:
            raise CaptureError("read() before open()")
        n = self.block_frames * self._channels
        if self._pos >= self._samples.size:
            return None
        if self.realtime and self._clock is not None:
            due = self._started + (self._pos / self._channels) / self._rate
            lag = due - self._clock.monotonic()
            if lag > 0:
                self._clock.sleep(lag)
        block = self._samples[self._pos : self._pos + n]
        self._pos += n
        return np.ascontiguousarray(block)

    def close(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)
        self._pos = 0


class SyntheticSource:
    """Deterministic generated audio: alternating tone and silence.

    Exists so the capture, segmentation and backpressure tests need neither audio
    hardware nor fixture files, and so a ``doctor`` self-check can exercise the
    whole pipeline without recording the user.
    """

    __slots__ = (
        "track",
        "sample_rate",
        "channels",
        "block_frames",
        "pattern",
        "total_blocks",
        "_emitted",
        "_phase",
        "_lock",
    )

    def __init__(
        self,
        track: TrackKind,
        sample_rate: int = 48000,
        channels: int = 2,
        block_frames: int = 1024,
        pattern: tuple[tuple[float, float], ...] = ((1.0, 440.0), (0.6, 0.0)),
        total_blocks: int = 0,
    ) -> None:
        self.track = track
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_frames = block_frames
        self.pattern = pattern
        self.total_blocks = total_blocks
        self._emitted = 0
        self._phase = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "sintético"

    def open(self) -> AudioFormat:
        return AudioFormat(sample_rate=self.sample_rate, channels=self.channels)

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self.total_blocks and self._emitted >= self.total_blocks:
                return None
            start = self._phase
            self._phase += self.block_frames
            self._emitted += 1

        cycle = sum(d for d, _ in self.pattern)
        t = (np.arange(self.block_frames, dtype=np.float64) + start) / self.sample_rate
        out = np.zeros(self.block_frames, dtype=np.float32)
        within = np.mod(t, cycle)
        edge = 0.0
        for duration, freq in self.pattern:
            mask = (within >= edge) & (within < edge + duration)
            if freq > 0:
                out[mask] = (0.25 * np.sin(2 * np.pi * freq * t[mask])).astype(
                    np.float32
                )
            edge += duration
        if self.channels == 1:
            return out
        return np.repeat(out, self.channels)

    def close(self) -> None:
        return None
