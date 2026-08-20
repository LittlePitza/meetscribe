"""One capture track: a producer thread, and everything that can go wrong in it.

A ``TrackCapture`` owns exactly one audio source and turns it into a stream of
pipeline-format :class:`AudioChunk` objects on a bounded queue. It also owns the
three things that make live capture survive a real two-hour meeting:

* **Reconnection.** Devices disappear mid-session -- headphones get plugged in,
  Windows switches the default endpoint, a driver resets. The source raises
  :class:`DeviceLostError`, the track closes it, waits a capped backoff, and asks
  its factory for a *freshly resolved* device. Resolution happens inside the
  factory, so reconnecting follows the endpoint the user is actually listening to
  now rather than stubbornly reopening the one that vanished.
* **Level metering and two watchdogs.** Every quarter second the track reports RMS
  and peak, and feeds the watchdog that catches a device delivering digital zero.
  A second, separate watchdog catches the case that one cannot see at all: **no
  data arriving**. An idle WASAPI loopback endpoint does not return silence, it
  blocks ``read()`` outright until something plays, so a track can be stalled with
  the level meter frozen and nothing to measure. The idle monitor runs on its own
  thread precisely because the capture thread is the thing that is stuck.
* **Timeline stamping.** Each chunk is stamped through
  :class:`~meetscribe.core.timeline.TrackTimeline`, which is what keeps the two
  tracks interleaved correctly in the merged transcript.

Note what is *not* here: no resampling policy, no VAD, no transcription. The track
converts and stamps, then hands off. That is what lets the whole class be tested
against :class:`SyntheticSource` with a fake clock.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from meetscribe.audio.buffers import ChunkQueue
from meetscribe.audio.level import SilenceWatchdog, WatchdogVerdict, peak_level, rms_dbfs
from meetscribe.audio.resample import StreamResampler
from meetscribe.audio.wav import WavWriter
from meetscribe.core.models import (
    PIPELINE_SAMPLE_RATE,
    AudioChunk,
    LevelReading,
    TrackKind,
)
from meetscribe.core.ports import AudioSource, Clock
from meetscribe.core.timeline import TrackTimeline
from meetscribe.errors import CaptureError, DeviceLostError

log = logging.getLogger(__name__)

__all__ = ["TrackCapture", "TrackStats"]

NoticeSink = Callable[[str, str, str], None]
"""``(level, code, message)`` -- message already in Spanish, ready for the panel."""

LevelSink = Callable[[LevelReading], None]


class TrackStats:
    """Snapshot of a track's health, for ``doctor``, the panel and the logs."""

    __slots__ = (
        "track",
        "device",
        "running",
        "media_s",
        "drift_s",
        "reanchors",
        "reconnects",
        "dropped_samples",
        "queued_seconds",
        "last_dbfs",
        "last_peak",
        "silent",
    )

    def __init__(self, **kw: Any) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    def as_dict(self) -> dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class TrackCapture:
    """Producer thread for one track."""

    def __init__(
        self,
        track: TrackKind,
        source_factory: Callable[[], AudioSource],
        queue: ChunkQueue,
        clock: Clock,
        *,
        realtime: bool = True,
        session_t0: float | None = None,
        level_interval_s: float = 0.25,
        watchdog: SilenceWatchdog | None = None,
        on_level: LevelSink | None = None,
        on_notice: NoticeSink | None = None,
        recorder: WavWriter | None = None,
        max_reconnects: int = 30,
        max_drift_s: float = 0.5,
    ) -> None:
        self.track = track
        self.source_factory = source_factory
        self.queue = queue
        self.clock = clock
        self.realtime = realtime
        self.level_interval_s = level_interval_s
        self.watchdog = watchdog or SilenceWatchdog()
        self.on_level = on_level
        self.on_notice = on_notice
        self.recorder = recorder
        self.max_reconnects = max_reconnects

        self.timeline = TrackTimeline(
            PIPELINE_SAMPLE_RATE, realtime=realtime, max_drift_s=max_drift_s
        )
        self._session_t0 = session_t0
        self._source: AudioSource | None = None
        self._resampler: StreamResampler | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._finished = threading.Event()
        self._error: BaseException | None = None
        self._reconnects = 0
        self._device_name = ""
        self._meter = _LevelMeter(level_interval_s)
        self._last_dbfs = -240.0
        self._last_peak = 0.0
        self._last_chunk_end_s = 0.0
        self._monitor_thread: threading.Thread | None = None
        self._idle_reported = False

    # ------------------------------------------------------------------ control

    def start(self, session_t0: float | None = None) -> None:
        """Open the device and start producing.

        Opening happens on the caller's thread so a bad device fails the
        ``meetscribe start`` command immediately with a usable message, rather
        than half-starting a session that produces nothing.
        """
        if self._thread is not None:
            raise CaptureError(f"track {self.track.value} already started")
        if session_t0 is not None:
            self._session_t0 = session_t0
        if self._session_t0 is None:
            self._session_t0 = self.clock.monotonic()

        self._open_source()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.track.value}", daemon=True
        )
        self._thread.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor, name=f"idle-{self.track.value}", daemon=True
        )
        self._monitor_thread.start()
        self._started.set()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread, unblock it if necessary, then release the device."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(min(1.0, timeout))
            if thread.is_alive():
                # An idle loopback endpoint blocks read() until audio plays, so
                # the flag alone will not end the loop. Stopping the stream from
                # here is what makes the pending read return (verified against
                # PortAudio: stop_stream() releases a blocked reader immediately).
                log.debug("unblocking %s by detaching its stream", self.track.value)
                self._detach_source()
                thread.join(timeout)
                if thread.is_alive():
                    log.warning(
                        "capture thread for %s did not stop within %.1fs",
                        self.track.value,
                        timeout,
                    )
            self._thread = None
        if self._monitor_thread is not None:
            self._monitor_thread.join(2.0)
            self._monitor_thread = None
        self._close_source()
        if self.recorder is not None:
            self.recorder.close()
        self.queue.close()

    @property
    def finished(self) -> bool:
        """True once a finite source reached end of stream."""
        return self._finished.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def stats(self) -> TrackStats:
        return TrackStats(
            track=self.track.value,
            device=self._device_name,
            running=self._thread is not None and self._thread.is_alive(),
            media_s=round(self.timeline.media_s, 2),
            drift_s=round(self.timeline.drift_s, 4),
            reanchors=self.timeline.reanchors,
            reconnects=self._reconnects,
            dropped_samples=self.queue.dropped_samples,
            queued_seconds=round(self.queue.queued_seconds, 2),
            last_dbfs=round(self._last_dbfs, 1),
            last_peak=round(self._last_peak, 4),
            silent=self.watchdog.alerted,
        )

    # ------------------------------------------------------------------- device

    def _open_source(self) -> None:
        source = self.source_factory()
        fmt = source.open()
        self._source = source
        self._device_name = source.name
        self._resampler = StreamResampler(fmt.sample_rate, fmt.channels)
        log.info(
            "track %s open: %s (%d Hz, %d ch)",
            self.track.value,
            source.name,
            fmt.sample_rate,
            fmt.channels,
        )

    def _detach_source(self) -> None:
        """Release the device without flushing, to unblock a stuck read."""
        source, self._source = self._source, None
        if source is not None:
            try:
                source.close()
            except Exception:
                log.debug("error detaching %s", self.track.value, exc_info=True)

    def _close_source(self) -> None:
        if self._resampler is not None:
            tail = self._resampler.flush()
            if tail.size:
                self._emit(tail)
            self._resampler = None
        if self._source is not None:
            self._source.close()
            self._source = None

    def _reconnect(self, reason: str) -> bool:
        """Close, back off, reopen. Returns False once we give up on this track."""
        if self._stop.is_set():
            return False
        self._reconnects += 1
        self._close_source()
        if self._reconnects > self.max_reconnects:
            self._notice(
                "error",
                "capture.reconnect_exhausted",
                f"La pista '{self._label()}' se desconectó demasiadas veces y se "
                "detuvo. La reunión continúa con la otra pista.",
            )
            return False

        delay = min(4.0, 0.5 * (2 ** min(self._reconnects - 1, 3)))
        self._notice(
            "warn",
            "capture.reconnecting",
            f"La pista '{self._label()}' se desconectó ({reason}). "
            f"Reconectando en {delay:.1f}s...",
        )
        log.warning("track %s lost (%s); retry in %.1fs", self.track.value, reason, delay)
        self.clock.sleep(delay)
        if self._stop.is_set():
            return False
        try:
            self._open_source()
        except Exception as exc:
            log.warning("track %s reopen failed: %s", self.track.value, exc)
            return True  # try again on the next loop iteration
        self._notice(
            "info",
            "capture.reconnected",
            f"La pista '{self._label()}' se reconecto a '{self._device_name}'.",
        )
        return True

    # --------------------------------------------------------------------- loop

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._source is None:
                    if not self._reconnect("dispositivo no disponible"):
                        break
                    continue
                source = self._source
                try:
                    raw = source.read()
                except DeviceLostError as exc:
                    if not self._reconnect(str(exc.user_message)):
                        break
                    continue
                except Exception:
                    if self._stop.is_set():
                        break  # the stream was detached to unblock this read
                    raise

                if raw is None:  # finite source drained
                    self._finished.set()
                    break
                if raw.size == 0:
                    continue

                assert self._resampler is not None  # noqa: S101 - narrowing, not validation
                mono = self._resampler.process(raw)
                if mono.size:
                    self._emit(mono)
        except BaseException as exc:
            self._error = exc
            log.exception("capture thread for %s died", self.track.value)
            self._notice(
                "error",
                "capture.thread_died",
                f"La captura de '{self._label()}' fallo: {exc}",
            )
        finally:
            self._finished.set()

    def _emit(self, mono: np.ndarray) -> None:
        """Stamp, enqueue, meter, and watch for silence."""
        arrival = self.clock.monotonic() - float(self._session_t0 or 0.0)
        start_s = self.timeline.stamp(mono.size, arrival)
        self._last_chunk_end_s = start_s + mono.size / PIPELINE_SAMPLE_RATE

        chunk = AudioChunk(track=self.track, samples=mono, start_s=start_s)
        if not self.realtime:
            # A file can be read more slowly; dropping from it would be a bug, not
            # a policy. Only a live endpoint has a driver buffer that cannot wait.
            if not self.queue.put_blocking(chunk, self._stop.is_set):
                return
            dropped = 0
        else:
            dropped = self.queue.put(chunk)
        if dropped:
            # Timestamps stay correct -- every chunk carries its own start time, so
            # the consumer sees a gap rather than a shift. What is lost is content,
            # and that must be said out loud.
            self._notice(
                "warn",
                "capture.backpressure",
                f"Se descartaron {dropped / PIPELINE_SAMPLE_RATE:.1f}s de audio de "
                f"'{self._label()}': la transcripción va por detrás de la captura.",
            )

        if self.recorder is not None:
            self.recorder.write(mono)

        window = self._meter.feed(mono)
        if window is not None:
            self._last_dbfs = rms_dbfs(window)
            self._last_peak = peak_level(window)
            at_s = self.timeline.media_s
            if self.on_level is not None:
                self.on_level(
                    LevelReading(
                        track=self.track,
                        dbfs=self._last_dbfs,
                        peak=self._last_peak,
                        at_s=at_s,
                    )
                )
            self._check_silence(self._last_dbfs, at_s)

    def _monitor(self) -> None:
        """Watch for a track that has stopped delivering data altogether."""
        window = max(2.0, self.watchdog.window_s)
        last_samples = -1
        last_change = self.clock.monotonic()
        while not self._stop.wait(1.0):
            samples = self.timeline.samples
            now = self.clock.monotonic()
            if samples != last_samples:
                last_samples = samples
                last_change = now
                if self._idle_reported:
                    self._idle_reported = False
                    self._notice(
                        "info",
                        f"capture.idle_recovered.{self.track.value}",
                        f"La pista '{self._label()}' volvió a recibir audio.",
                    )
                continue
            if not self._idle_reported and now - last_change >= window:
                self._idle_reported = True
                self._notice(*self._idle_message(window))

    def _idle_message(self, window: float) -> tuple[str, str, str]:
        code = f"capture.idle.{self.track.value}"
        if self.track is TrackKind.LOOPBACK:
            return (
                "info",
                code,
                (
                    f"La pista remota lleva {window:.0f}s sin recibir nada. Es normal "
                    "si la reunión aún no ha empezado: Windows solo entrega audio de "
                    "loopback mientras algo suena."
                ),
            )
        return (
            "warn",
            code,
            (
                f"El micrófono lleva {window:.0f}s sin entregar datos. Revisa que no "
                "esté en uso exclusivo por otra aplicación."
            ),
        )

    def _check_silence(self, dbfs: float, at_s: float) -> None:
        verdict = self.watchdog.observe(dbfs, at_s)
        if verdict == WatchdogVerdict.WENT_SILENT:
            if self.track is TrackKind.MIC:
                message = (
                    "Tu micrófono lleva "
                    f"{self.watchdog.window_s:.0f}s en silencio digital (sin señal "
                    "alguna). Revisa Configuración > Privacidad > Micrófono, y que "
                    "no esté silenciado en la reunión."
                )
            else:
                message = (
                    "La pista remota lleva "
                    f"{self.watchdog.window_s:.0f}s sin audio. Si la reunión está "
                    "activa, comprueba que el sonido salga por el dispositivo "
                    f"'{self._device_name}'."
                )
            self._notice("warn", f"capture.silent.{self.track.value}", message)
        elif verdict == WatchdogVerdict.RECOVERED:
            self._notice(
                "info",
                f"capture.recovered.{self.track.value}",
                f"La pista '{self._label()}' volvió a recibir audio.",
            )

    # ------------------------------------------------------------------ helpers

    def _label(self) -> str:
        return "micrófono" if self.track is TrackKind.MIC else "remoto"

    def _notice(self, level: str, code: str, message: str) -> None:
        if self.on_notice is not None:
            self.on_notice(level, code, message)


class _LevelMeter:
    """Accumulates samples and returns a window every ``interval_s`` of audio.

    Measuring on audio time rather than wall time keeps the meter's cadence stable
    when the capture thread is scheduled unevenly, and makes it deterministic under
    a fake clock.
    """

    __slots__ = ("interval_samples", "_buf", "_n")

    def __init__(self, interval_s: float, sample_rate: int = PIPELINE_SAMPLE_RATE):
        self.interval_samples = max(1, int(interval_s * sample_rate))
        self._buf: list[np.ndarray] = []
        self._n = 0

    def feed(self, samples: np.ndarray) -> np.ndarray | None:
        self._buf.append(samples)
        self._n += samples.size
        if self._n < self.interval_samples:
            return None
        window = np.concatenate(self._buf)
        self._buf = []
        self._n = 0
        return window
