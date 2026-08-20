"""Composition root: turning a :class:`Config` into a running pipeline.

All the wiring lives here, and only here. Every other module takes its
collaborators as constructor arguments, which is what makes the pipeline testable
with fakes -- the tests build a :class:`SessionRunner` directly and never touch this
file.

The one piece of real policy in this module is **graceful degradation**. If the
speaker-embedding model is missing or fails to load, the session does not abort: it
falls back to :class:`NullDiarizer`, every remote line is labelled "Remoto", and the
user is told once. A meeting with imperfect attribution is worth vastly more than a
meeting that refused to record.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from meetscribe.assets import SPEAKER_MODEL, VAD_MODEL, resolve_model
from meetscribe.capture.devices import list_devices, pick_device
from meetscribe.capture.sources import (
    FallbackSource,
    SoundcardSource,
    WasapiSource,
    WavFileSource,
)
from meetscribe.config import Config
from meetscribe.core.models import TrackKind
from meetscribe.diarize.embedder import TitanetEmbedder
from meetscribe.diarize.online import DiarizerParams, NullDiarizer, OnlineDiarizer
from meetscribe.errors import MeetScribeError
from meetscribe.stt.transcriber import FasterWhisperTranscriber
from meetscribe.stt.vad import SileroVadEngine, VadParams

log = logging.getLogger(__name__)

__all__ = [
    "build_transcriber",
    "build_diarizer",
    "build_vad_factory",
    "build_source_factory",
    "build_replay_factory",
]


def build_transcriber(config: Config) -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber(
        config.stt.model,
        language=config.stt.language,
        cpu_threads=config.stt.cpu_threads,
        beam_size=config.stt.beam_size,
        compute_type=config.stt.compute_type,
        auto_detect_min_s=config.stt.auto_detect_min_s,
    )


def build_diarizer(config: Config, on_notice: Callable[[str, str, str], None] | None = None):
    """The online diarizer, or the degraded single-label mode with an explanation."""
    if not config.diarization.enabled:
        return NullDiarizer()
    try:
        embedder = TitanetEmbedder(str(resolve_model(SPEAKER_MODEL)))
        embedder.load()
    except MeetScribeError as exc:
        log.warning("diarization unavailable: %s", exc)
        if on_notice is not None:
            on_notice(
                "warn",
                "diarization.unavailable",
                f"{exc.user_message} Todo lo remoto se etiquetara como 'Remoto'.",
            )
        return NullDiarizer()

    return OnlineDiarizer(
        embedder,
        DiarizerParams(
            threshold=config.diarization.threshold,
            min_embed_s=config.diarization.min_embed_s,
            min_new_speaker_s=config.diarization.min_new_speaker_s,
            max_speakers=config.diarization.max_speakers,
        ),
        label_prefix=config.diarization.label_prefix,
    )


def build_vad_factory(config: Config) -> Callable[[TrackKind], SileroVadEngine]:
    """One VAD instance per track -- they carry independent state and must not be
    shared, but they read the same model file."""
    model_path = str(resolve_model(VAD_MODEL))
    params = VadParams(
        threshold=config.vad.threshold,
        min_silence_duration=config.vad.min_silence_duration,
        min_speech_duration=config.vad.min_speech_duration,
        max_speech_duration=config.vad.max_speech_duration,
    )

    def factory(_: TrackKind) -> SileroVadEngine:
        return SileroVadEngine(model_path, params)

    return factory


def build_source_factory(
    config: Config, track: TrackKind, device_spec: str | int | None = None
) -> Callable[[], object]:
    """A factory that re-resolves the device on every call.

    Re-resolution is the whole point: when a track reconnects after the user plugs
    in headphones, it must attach to the endpoint they are now listening to, not
    reopen the one that disappeared.
    """
    spec = device_spec if device_spec is not None else (
        config.capture.mic_device if track is TrackKind.MIC else config.capture.loopback_device
    )
    backend = config.capture.backend

    def factory() -> object:
        device = pick_device(spec, track, list_devices())
        candidates = []
        if backend in ("auto", "pyaudio"):
            candidates.append(
                WasapiSource(
                    device,
                    track,
                    frames_per_buffer=config.capture.frames_per_buffer,
                    follow_default=config.capture.follow_default_device and spec is None,
                )
            )
        if backend in ("auto", "soundcard"):
            candidates.append(
                SoundcardSource(
                    device, track, block_frames=config.capture.frames_per_buffer
                )
            )
        return FallbackSource(candidates)

    return factory


def build_replay_factory(path: str | Path, track: TrackKind) -> Callable[[], object]:
    """Replay a WAV as a track. Used by ``--replay`` and by the regression tests."""

    def factory() -> object:
        return WavFileSource(path, track, realtime=False)

    return factory
