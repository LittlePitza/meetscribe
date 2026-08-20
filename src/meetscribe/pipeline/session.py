"""The session orchestrator: topology, threading model, and shutdown.

**Topology.** Six long-lived threads, with a bounded queue at every seam:

    [capture mic]      --ChunkQueue(30 s)-->  [segmenter mic]      --\\
                                                                     >-- UtteranceQueue(32)
    [capture loopback] --ChunkQueue(30 s)-->  [segmenter loopback] --/
                                                                          |
                                                                          v
                                                            [STT worker: transcribe
                                                             -> diarize -> assemble
                                                             -> persist -> publish]
                                                                          |
                              [live action items]  <---------------------/
                                                                          |
    [HTTP server thread] <---- InMemoryEventBus <-------------------------/

**Why threads and not asyncio or processes.** The expensive stages release the GIL
in C++: CTranslate2 during decoding, onnxruntime during VAD and embedding, and
PortAudio while blocked on a read. So threads genuinely run in parallel here, and
the Python-level work between them is trivial. Processes would buy true isolation
at the cost of shipping audio buffers over a pipe and duplicating a 1.6 GB model;
asyncio would buy nothing, because every one of these stages is a blocking C call,
not an I/O wait. The web server is a plain threaded HTTP server for the same
reason: with no event loop in the process, there is no event loop to starve, and
the bridge between "blocking pipeline" and "async server" -- the classic source of
subtle deadlocks in this kind of program -- simply does not exist.

**One transcriber, not two.** Measured: two ``large-v3-turbo`` instances would take
~3.2 GB and put two decoders on the same eight cores. Meeting speech is
turn-taking, so a single shared instance behind one queue is both cheaper and
faster in practice; the queue absorbs the moments when both tracks do speak at
once.

**Shutdown is ordered, and every step has a deadline.** Ctrl-C mid-meeting must not
lose the last sentence, so the sequence drains rather than aborts: stop capture ->
flush resampler tails -> close chunk queues -> segmenters flush their VADs -> close
the utterance queue -> the STT worker drains what is left -> refine diarization ->
drop unfinalised drafts -> summarise -> close. Each stage has a timeout, and
exceeding one logs and moves on rather than hanging the process.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from meetscribe.audio.buffers import ChunkQueue
from meetscribe.audio.level import SilenceWatchdog
from meetscribe.audio.wav import WavWriter
from meetscribe.config import Config
from meetscribe.core import events as ev
from meetscribe.core.models import (
    LOCAL_SPEAKER_ID,
    UNKNOWN_SPEAKER_ID,
    LevelReading,
    Session,
    SessionState,
    Speaker,
    TrackKind,
    Utterance,
)
from meetscribe.core.ports import Clock, SummaryRequest
from meetscribe.errors import MeetScribeError, SummarizerError
from meetscribe.pipeline.assembler import TranscriptAssembler, local_speaker
from meetscribe.pipeline.queues import UtteranceQueue
from meetscribe.stt.vad import VadSegmenter

log = logging.getLogger(__name__)

__all__ = ["SessionRunner", "recover_interrupted_sessions"]


def recover_interrupted_sessions(store) -> list[Session]:
    """Mark sessions the process never finished as failed, and report them.

    Run at startup. Without it a crashed session stays ``RUNNING`` forever and its
    transcript looks like a live one that simply stopped producing lines. The
    segments themselves are already durable -- this only fixes the bookkeeping, and
    the user can still summarise the recovered session afterwards.
    """
    interrupted = (SessionState.RUNNING, SessionState.STARTING, SessionState.STOPPING)
    recovered: list[Session] = []
    for session in store.list_sessions(limit=200):
        if session.state in interrupted:
            fixed = _with_state(session, SessionState.FAILED)
            store.update_session(fixed)
            recovered.append(fixed)
            log.warning("recovered interrupted session %s", session.session_id)
    return recovered


def _with_state(session: Session, state: SessionState) -> Session:
    import dataclasses

    return dataclasses.replace(session, state=state)


class SessionRunner:
    """Owns one live meeting from start to summary."""

    def __init__(
        self,
        session: Session,
        config: Config,
        store,
        bus,
        clock: Clock,
        *,
        transcriber,
        diarizer,
        mic_source_factory: Callable[[], object] | None,
        loopback_source_factory: Callable[[], object] | None,
        vad_factory: Callable[[TrackKind], object],
        summarizer=None,
        realtime: bool = True,
        audio_dir: Path | None = None,
    ) -> None:
        self.session = session
        self.config = config
        self.store = store
        self.bus = bus
        self.clock = clock
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.summarizer = summarizer
        self.vad_factory = vad_factory
        self.realtime = realtime
        self.audio_dir = audio_dir

        self._source_factories: dict[TrackKind, Callable[[], object] | None] = {
            TrackKind.MIC: mic_source_factory,
            TrackKind.LOOPBACK: loopback_source_factory,
        }
        self._chunk_queues: dict[TrackKind, ChunkQueue] = {}
        self._captures: dict[TrackKind, object] = {}
        self._segmenters: dict[TrackKind, VadSegmenter] = {}
        self._segmenter_threads: list[threading.Thread] = []
        self._recorders: dict[TrackKind, WavWriter] = {}

        self._utterances = UtteranceQueue(config.stt.queue_size)
        self._assembler = TranscriptAssembler(session.session_id)
        self._speakers: dict[str, Speaker] = {}
        self._levels: dict[TrackKind, LevelReading] = {}

        self._stt_thread: threading.Thread | None = None
        self._actions_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._t0 = 0.0
        self._lock = threading.RLock()
        self._pending_action_from = 0.0
        self._errors: list[str] = []

    # --------------------------------------------------------------------- time

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.clock.monotonic() - self._t0) if self._t0 else 0.0

    @property
    def capture_finished(self) -> bool:
        """True once every track has reached end of stream.

        Only finite sources ever reach it -- a live WASAPI endpoint never does --
        so this is what lets ``--replay`` re-transcribe a recording and stop by
        itself instead of waiting for a Ctrl-C that would never make sense.
        """
        captures = list(self._captures.values())
        return bool(captures) and all(
            c.finished for c in captures  # type: ignore[attr-defined]
        )

    # -------------------------------------------------------------------- start

    def start(self) -> None:
        """Bring the whole pipeline up, or fail cleanly having started nothing."""
        self._t0 = self.clock.monotonic()
        self.bus.set_time_source(lambda: self.elapsed_s)

        self.session = _with_state(self.session, SessionState.STARTING)
        self.store.create_session(self.session)
        self._publish_session()

        try:
            for track in (TrackKind.MIC, TrackKind.LOOPBACK):
                if self._source_factories[track] is None:
                    self._notice(
                        "warn",
                        f"capture.disabled.{track.value}",
                        f"La pista '{track.value}' no está configurada; la reunión "
                        "se grabará solo con la otra.",
                    )
                    continue
                self._start_track(track)

            if TrackKind.MIC in self._captures:
                # Only announce "Yo" when the microphone is actually being
                # recorded; a phantom participant with no lines is noise.
                self._register_speaker(local_speaker(self._local_name()))

            if not self._captures:
                raise MeetScribeError(
                    "no capture track could be started",
                    user_message=(
                        "No se pudo iniciar ninguna pista de audio. Ejecuta "
                        "'meetscribe doctor' para diagnosticar."
                    ),
                )

            self._stt_thread = threading.Thread(
                target=self._stt_loop, name="stt-worker", daemon=True
            )
            self._stt_thread.start()

            if (
                self.summarizer is not None
                and self.config.summarizer.live_action_items
            ):
                self._actions_thread = threading.Thread(
                    target=self._actions_loop, name="action-items", daemon=True
                )
                self._actions_thread.start()
        except BaseException:
            self.stop(summarize=False, failed=True)
            raise

        self.session = _with_state(self.session, SessionState.RUNNING)
        self.store.update_session(self.session)
        self._publish_session()
        log.info(
            "session %s running (%s, diarization=%s)",
            self.session.session_id,
            self.session.stt_model,
            self.diarizer.enabled,
        )

    def _start_track(self, track: TrackKind) -> None:
        from meetscribe.capture.track import TrackCapture

        queue = ChunkQueue(max_seconds=self.config.capture.buffer_seconds)
        self._chunk_queues[track] = queue

        window = (
            self.config.capture.mic_silence_window_s
            if track is TrackKind.MIC
            else self.config.capture.loopback_silence_window_s
        )
        recorder = self._make_recorder(track)
        if recorder is not None:
            self._recorders[track] = recorder

        capture = TrackCapture(
            track,
            self._source_factories[track],  # type: ignore[arg-type]
            queue,
            self.clock,
            realtime=self.realtime,
            session_t0=self._t0,
            watchdog=SilenceWatchdog(window_s=window),
            on_level=self._on_level,
            on_notice=self._notice,
            recorder=recorder,
        )
        capture.start(self._t0)
        self._captures[track] = capture

        segmenter = VadSegmenter(
            track,
            self.vad_factory(track),  # type: ignore[arg-type]
            interim_after_s=self.config.stt.interim_after_s,
            interim_every_s=self.config.stt.interim_every_s,
            on_speech_change=lambda speaking, at_s, t=track: self.bus.publish(
                ev.EV_VAD, ev.make_vad(t, speaking, at_s)
            ),
        )
        self._segmenters[track] = segmenter

        thread = threading.Thread(
            target=self._segmenter_loop,
            args=(track,),
            name=f"segmenter-{track.value}",
            daemon=True,
        )
        thread.start()
        self._segmenter_threads.append(thread)

    def _make_recorder(self, track: TrackKind) -> WavWriter | None:
        if self.config.capture.retain_audio == "none" or self.audio_dir is None:
            return None
        path = self.audio_dir / self.session.session_id / f"{track.value}.wav"
        return WavWriter(path)

    # -------------------------------------------------------------------- loops

    def _segmenter_loop(self, track: TrackKind) -> None:
        queue = self._chunk_queues[track]
        segmenter = self._segmenters[track]
        try:
            while True:
                chunk = queue.get(timeout=0.25)
                if chunk is None:
                    # Only the queue closing ends the loop: stop() always closes it
                    # after the capture thread has flushed, so draining is complete.
                    if queue.closed and len(queue) == 0:
                        break
                    continue
                for utterance in segmenter.feed(chunk):
                    self._enqueue(utterance)
            for utterance in segmenter.flush():
                self._enqueue(utterance)
        except Exception as exc:
            log.exception("segmenter %s failed", track.value)
            self._fail(f"segmentador '{track.value}'", exc)

    def _enqueue(self, utterance: Utterance) -> None:
        if not self.realtime:
            # Offline replay must be lossless: the segmenter can simply wait.
            self._utterances.put_blocking(utterance, self._stop.is_set)
            return
        outcome = self._utterances.put(utterance)
        if outcome == "dropped_final":
            self._notice(
                "warn",
                "stt.backpressure",
                "La transcripción no da abasto y se perdió una intervención. "
                "Considera un modelo más rápido (stt.model: small).",
            )

    def _stt_loop(self) -> None:
        try:
            while True:
                utterance = self._utterances.get(timeout=0.25)
                if utterance is None:
                    if self._utterances.closed and len(self._utterances) == 0:
                        break
                    continue
                self._process(utterance)
        except Exception as exc:
            log.exception("stt worker failed")
            self._fail("transcripción", exc)
        finally:
            self._finished.set()

    def _process(self, utterance: Utterance) -> None:
        if utterance.is_partial and self._assembler.is_finalised(utterance.key):
            return
        try:
            result = self.transcriber.transcribe(
                utterance.samples,
                language=self.config.stt.language,
                initial_prompt=self.config.stt.initial_prompt,
            )
        except Exception as exc:
            log.warning(
                "transcription failed for %s/%d: %s",
                utterance.track.value,
                utterance.seq,
                exc,
            )
            self._notice(
                "warn",
                "stt.utterance_failed",
                "No se pudo transcribir una intervención; la sesión continúa.",
            )
            return

        speaker = self._speaker_for(utterance)
        segment = self._assembler.build(utterance, result, speaker)
        if segment is None:
            return

        self.store.replace_segment(segment)
        self.bus.publish(ev.EV_SEGMENT, ev.make_segment(segment))

    def _speaker_for(self, utterance: Utterance) -> Speaker:
        if utterance.track is TrackKind.MIC:
            return self._speakers[LOCAL_SPEAKER_ID]

        if utterance.is_partial:
            # Drafts do not diarize: a partial utterance would both waste an
            # embedding pass and pull the speaker's centroid toward a fragment. The
            # provisional label is whoever the diarizer last saw, which is right
            # far more often than not because the draft belongs to a turn already
            # in progress. The final version corrects it moments later.
            tentative = getattr(self.diarizer, "last_speaker_id", UNKNOWN_SPEAKER_ID)
            known = self._speakers.get(tentative)
            return known if known is not None else self._fallback_remote_speaker()

        speaker_id = self.diarizer.assign(utterance)
        known = self._speakers.get(speaker_id)
        if known is not None:
            return known

        for candidate in self.diarizer.speakers():
            if candidate.speaker_id == speaker_id:
                self._register_speaker(candidate)
                return candidate
        return self._fallback_remote_speaker()

    def _fallback_remote_speaker(self) -> Speaker:
        existing = self._speakers.get(UNKNOWN_SPEAKER_ID)
        if existing is not None:
            return existing
        speaker = Speaker(
            speaker_id=UNKNOWN_SPEAKER_ID,
            display_name="Remoto",
            track=TrackKind.LOOPBACK,
        )
        self._register_speaker(speaker)
        return speaker

    def _register_speaker(self, speaker: Speaker) -> None:
        with self._lock:
            self._speakers[speaker.speaker_id] = speaker
        self.store.upsert_speaker(self.session.session_id, speaker)
        self.bus.publish(ev.EV_SPEAKER, ev.make_speaker(speaker))

    def _actions_loop(self) -> None:
        """Detect commitments during the meeting, not only at the end."""
        interval = max(30.0, self.config.summarizer.live_interval_s)
        while not self._stop.wait(interval):
            try:
                self._detect_actions()
            except SummarizerError as exc:
                log.info("live action item detection unavailable: %s", exc)
            except Exception:
                log.exception("live action item detection failed")

    def _detect_actions(self) -> None:
        if self.summarizer is None:
            return
        segments = [
            s
            for s in self.store.get_segments(self.session.session_id, final_only=True)
            if s.start_s >= self._pending_action_from
        ]
        if len(segments) < 4:
            return
        self._pending_action_from = max(s.end_s for s in segments)
        items = self.summarizer.detect_action_items(segments, self.session.language)
        for item in items:
            if self.store.add_action_item(self.session.session_id, item):
                self.bus.publish(ev.EV_ACTION_ITEM, ev.make_action_item(item))

    # --------------------------------------------------------------------- stop

    def stop(
        self,
        summarize: bool = True,
        timeout_s: float = 120.0,
        failed: bool = False,
    ) -> Session:
        """Drain everything in order, then finalise. Safe to call twice."""
        if self.session.state.is_terminal:
            return self.session

        self._stop.set()
        self.session = _with_state(self.session, SessionState.STOPPING)
        self._publish_session()
        log.info("stopping session %s", self.session.session_id)

        # 1. Capture first: stop producing, flush the resamplers' delay lines.
        for capture in self._captures.values():
            try:
                capture.stop(timeout=5.0)  # type: ignore[attr-defined]
            except Exception:
                log.exception("error stopping capture")

        # 2. Chunk queues are closed by capture.stop(); segmenters drain and flush.
        for thread in self._segmenter_threads:
            thread.join(timeout=15.0)
            if thread.is_alive():
                log.warning("segmenter %s did not finish in time", thread.name)

        # 3. No more utterances can arrive; let the decoder finish the backlog.
        self._notice("info", "session.draining", "Terminando la transcripción pendiente...")
        self._utterances.close()
        if self._stt_thread is not None:
            self._stt_thread.join(timeout=timeout_s)
            if self._stt_thread.is_alive():
                log.warning("stt worker did not drain within %.0fs", timeout_s)
                self._notice(
                    "warn",
                    "session.drain_timeout",
                    "La transcripción pendiente no terminó a tiempo; puede faltar "
                    "el final de la reunión.",
                )

        # 4. Offline diarization refinement, reconciled with what was already shown.
        if self.config.diarization.refine_at_end and self.diarizer.enabled:
            self._refine_speakers()

        # 5. Provisional lines never confirmed must not survive into the archive.
        dropped = self.store.drop_drafts(self.session.session_id)
        if dropped:
            log.info("dropped %d unconfirmed drafts", dropped)

        self._close_recorders()

        state = SessionState.FAILED if failed or self._errors else SessionState.STOPPED
        self.session = _replace_session(
            self.session, state=state, ended_at=self.clock.now()
        )
        self.store.update_session(self.session)
        self._publish_session()

        if summarize:
            self._summarise()

        self.bus.publish(
            ev.EV_NOTICE,
            ev.make_notice(
                "info",
                "session.finished",
                f"Sesión finalizada. Transcripción guardada "
                f"({self._assembler.accepted} intervenciones).",
            ),
        )
        self._finished.set()
        return self.session

    def _refine_speakers(self) -> None:
        try:
            mapping = self.diarizer.refine()
        except Exception:
            log.exception("diarization refinement failed")
            return
        for speaker in self.diarizer.speakers():
            known = self._speakers.get(speaker.speaker_id)
            merged = (
                speaker
                if known is None
                else speaker.renamed(known.display_name)
            )
            self._register_speaker(merged)
        if mapping:
            changed = self.store.reassign_speakers(self.session.session_id, mapping)
            self._notice(
                "info",
                "diarization.refined",
                f"Se reasignaron {changed} intervenciones tras revisar toda la "
                "reunión.",
            )
            # Republish only what moved. Re-sending the whole transcript would be
            # correct and would also overflow every subscriber's queue on a long
            # meeting, disconnecting the panel at the worst possible moment.
            touched = set(mapping)
            for segment in self.store.get_segments(self.session.session_id):
                if (segment.track.value, segment.seq) in touched:
                    self.bus.publish(ev.EV_SEGMENT, ev.make_segment(segment))

    def _summarise(self) -> None:
        if self.summarizer is None:
            self._notice(
                "info",
                "summary.disabled",
                "No hay proveedor de resumen configurado; solo se guardó la "
                "transcripción.",
            )
            return
        segments = self.store.get_segments(self.session.session_id, final_only=True)
        if len(segments) < 2:
            self._notice(
                "info",
                "summary.too_short",
                "La reunión es demasiado corta para resumirla.",
            )
            return

        self._notice("info", "summary.generating", "Generando el resumen...")
        try:
            summary = self.summarizer.summarize(
                SummaryRequest(
                    session=self.session,
                    segments=segments,
                    speakers=self.store.get_speakers(self.session.session_id),
                    language=self.session.language,
                    now=self.clock.now(),
                )
            )
        except MeetScribeError as exc:
            # The transcript is already durable. Only the summary is lost, and it
            # can be regenerated later without re-recording anything.
            log.warning("summarization failed: %s", exc)
            self._notice("warn", "summary.failed", exc.user_message)
            return
        except Exception as exc:
            log.exception("summarization failed")
            self._notice(
                "warn",
                "summary.failed",
                f"No se pudo generar el resumen ({exc}). La transcripción está a "
                "salvo; reintenta con 'meetscribe summarize'.",
            )
            return

        self.store.save_summary(summary)
        self.bus.publish(ev.EV_SUMMARY, ev.make_summary(summary))

    def _close_recorders(self) -> None:
        for recorder in self._recorders.values():
            recorder.close()
        if self.config.capture.retain_audio == "session":
            from meetscribe.security.retention import shred_session_audio

            if self.audio_dir is not None:
                shred_session_audio(self.audio_dir / self.session.session_id)
                log.info("session audio shredded")

    # ------------------------------------------------------------------- events

    def _on_level(self, reading: LevelReading) -> None:
        self._levels[reading.track] = reading
        self.bus.publish(ev.EV_LEVEL, ev.make_level(self._levels.values()))

    def _notice(self, level: str, code: str, message: str) -> None:
        self.bus.publish(ev.EV_NOTICE, ev.make_notice(level, code, message))
        if level == "error":
            self._errors.append(f"{code}: {message}")

    def _publish_session(self) -> None:
        self.bus.publish(ev.EV_SESSION, ev.make_session(self.session, self.elapsed_s))

    def _fail(self, stage: str, exc: BaseException) -> None:
        self._notice("error", "pipeline.stage_failed", f"Falló la etapa de {stage}: {exc}")

    def _local_name(self) -> str:
        return str(self.session.metadata.get("local_name") or "Yo")

    # ------------------------------------------------------------------ queries

    def rename_speaker(self, speaker_id: str, name: str) -> int:
        """Rename retroactively and forward, and tell the panel."""
        count = self.store.rename_speaker(self.session.session_id, speaker_id, name)
        with self._lock:
            existing = self._speakers.get(speaker_id)
            if existing is not None:
                self._speakers[speaker_id] = existing.renamed(name)
        speaker = self._speakers.get(speaker_id)
        if speaker is not None:
            self.bus.publish(ev.EV_SPEAKER, ev.make_speaker(speaker))
        return count

    def wait(self, timeout: float | None = None) -> bool:
        return self._finished.wait(timeout)

    def stats(self) -> dict[str, object]:
        return {
            "session_id": self.session.session_id,
            "state": self.session.state.value,
            "elapsed_s": round(self.elapsed_s, 1),
            "tracks": {
                track.value: capture.stats().as_dict()  # type: ignore[attr-defined]
                for track, capture in self._captures.items()
            },
            "utterance_queue": self._utterances.stats(),
            "assembler": self._assembler.stats(),
            "diarizer": self.diarizer.stats() if hasattr(self.diarizer, "stats") else {},
            "bus": {
                "published": self.bus.published,
                "subscribers": self.bus.subscriber_count,
            },
            "errors": list(self._errors),
        }


def _replace_session(
    session: Session, *, state: SessionState, ended_at: dt.datetime
) -> Session:
    import dataclasses

    return dataclasses.replace(session, state=state, ended_at=ended_at)
