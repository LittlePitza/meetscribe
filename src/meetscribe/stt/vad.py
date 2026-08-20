"""Voice activity detection and utterance segmentation.

**Why segmentation and not sliding windows.** Whisper is a 30-second batch model.
Chopping the stream every N seconds and transcribing each piece produces garbage at
every boundary: cut words, invented sentence endings, duplicated text where the
windows overlap. The three ways out are (a) fixed windows with overlap and text
stitching, (b) VAD-gated segmentation -- transcribe only complete runs of speech
between pauses, (c) LocalAgreement-2, where a growing buffer is re-transcribed and
only the prefix two consecutive runs agree on is emitted.

This project uses **(b)**, and the reason is arithmetic. The measured budget
(``docs/benchmarks.md``) is RTF 0.32 for the whole pipeline against a CPU that
must also run Teams. LocalAgreement-2 re-transcribes the same audio on every step,
multiplying STT cost by three to five; that does not fit, and buying it would mean
dropping to a model with twice the word error rate. Option (a) trades boundary
artefacts for stitching heuristics that fail exactly where meetings are hardest --
fast turn-taking. Option (b) costs one pass over each second of speech, and the cut
points land in silence, which is where a human transcriber would put them too.

The price of (b) is latency: nothing appears until the speaker pauses. Two
mechanisms bound it:

* ``max_speech_duration`` makes the VAD itself force a cut inside a long monologue.
* Before that, an in-progress utterance longer than ``interim_after_s`` is
  transcribed as it stands and published as a **draft**. The draft carries the same
  ``seq`` as the eventual final segment, so the panel replaces it in place rather
  than showing the sentence twice. Draft text is explicitly provisional; final text
  is immutable.

Gaps are handled by padding with silence rather than by ignoring them. The VAD's
segment indices are sample counts into its own stream, so silently skipping a
dropped second would shift every later timestamp in the meeting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from meetscribe.core.models import PIPELINE_SAMPLE_RATE, AudioChunk, TrackKind, Utterance
from meetscribe.errors import ModelNotFoundError

log = logging.getLogger(__name__)

__all__ = ["VadEngine", "SileroVadEngine", "VadSegmenter", "VadParams"]

WINDOW = 512
"""Silero's frame size at 16 kHz. Feeding any other size is a hard error in the
model, not a quality tradeoff."""


class VadEngine(Protocol):
    """The slice of sherpa-onnx's ``VoiceActivityDetector`` this module uses.

    Declared as a Protocol so the segmenter -- where all the tricky index and
    timeline arithmetic lives -- can be tested against a scripted fake with no ONNX
    runtime and no model file.
    """

    def accept_waveform(self, samples: np.ndarray) -> None: ...

    def is_speech_detected(self) -> bool: ...

    def empty(self) -> bool: ...

    @property
    def front(self) -> Any: ...

    @property
    def current_segment(self) -> Any: ...

    def pop(self) -> None: ...

    def flush(self) -> None: ...

    def reset(self) -> None: ...


class VadParams:
    """Tuned for conversational Spanish, where speakers trail off mid-sentence."""

    __slots__ = (
        "threshold",
        "min_silence_duration",
        "min_speech_duration",
        "max_speech_duration",
        "buffer_seconds",
        "num_threads",
    )

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration: float = 0.45,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20.0,
        buffer_seconds: float = 40.0,
        num_threads: int = 1,
    ) -> None:
        # 0.5 is Silero's calibrated operating point; moving it trades missed quiet
        # speech against transcribing keyboard noise.
        self.threshold = threshold
        # 450 ms: long enough that a mid-sentence breath does not split a clause,
        # short enough that a turn handover is not glued to the reply. The spike
        # used 350 ms on clean TTS with no breathing; real speech needs more.
        self.min_silence_duration = min_silence_duration
        # Below 250 ms is a click or a lip smack, not a word -- but note this is
        # short enough to keep genuine backchannels ("si", "aja"), which carry the
        # turn-taking structure a summariser needs.
        self.min_speech_duration = min_speech_duration
        # 20 s bounds worst-case latency and keeps every utterance inside Whisper's
        # 30-second window with room for the pre-roll.
        self.max_speech_duration = max_speech_duration
        self.buffer_seconds = buffer_seconds
        self.num_threads = num_threads


class SileroVadEngine:
    """sherpa-onnx Silero VAD. 0.6 MB of ONNX, measured at RTF 0.0040."""

    __slots__ = ("_vad", "params")

    def __init__(self, model_path: str, params: VadParams | None = None) -> None:
        if not Path(model_path).exists():
            raise ModelNotFoundError(
                f"silero vad model missing at {model_path}",
                user_message=(
                    "Falta el modelo de detección de voz (silero_vad.onnx). "
                    "Ejecuta 'meetscribe models --download'."
                ),
            )
        import sherpa_onnx

        self.params = params or VadParams()
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = self.params.threshold
        cfg.silero_vad.min_silence_duration = self.params.min_silence_duration
        cfg.silero_vad.min_speech_duration = self.params.min_speech_duration
        cfg.silero_vad.max_speech_duration = self.params.max_speech_duration
        cfg.sample_rate = PIPELINE_SAMPLE_RATE
        cfg.num_threads = self.params.num_threads
        self._vad = sherpa_onnx.VoiceActivityDetector(
            cfg, buffer_size_in_seconds=self.params.buffer_seconds
        )

    def accept_waveform(self, samples: np.ndarray) -> None:
        self._vad.accept_waveform(samples)

    def is_speech_detected(self) -> bool:
        return bool(self._vad.is_speech_detected())

    def empty(self) -> bool:
        return bool(self._vad.empty())

    @property
    def front(self) -> Any:
        return self._vad.front

    @property
    def current_segment(self) -> Any:
        return self._vad.current_segment

    def pop(self) -> None:
        self._vad.pop()

    def flush(self) -> None:
        self._vad.flush()

    def reset(self) -> None:
        self._vad.reset()


class VadSegmenter:
    """Turns a track's chunk stream into utterances, in session time.

    Owned by exactly one segmenter thread per track; not thread-safe.
    """

    def __init__(
        self,
        track: TrackKind,
        engine: VadEngine,
        *,
        interim_after_s: float = 8.0,
        interim_every_s: float = 5.0,
        gap_tolerance_s: float = 0.05,
        max_pad_s: float = 10.0,
        pre_roll_s: float = 0.0,
        on_speech_change: Callable[[bool, float], None] | None = None,
    ) -> None:
        self.track = track
        self.engine = engine
        self.interim_after_s = interim_after_s
        self.interim_every_s = interim_every_s
        self.gap_tolerance_s = gap_tolerance_s
        self.max_pad_s = max_pad_s
        self.pre_roll_s = pre_roll_s
        self.on_speech_change = on_speech_change

        self._pending = np.zeros(0, dtype=np.float32)
        self._fed = 0
        self._anchor_fed = 0
        self._anchor_s: float | None = None
        self._expected_s: float | None = None
        self._seq_base = 0
        self._speaking = False
        self._draft_marks: dict[int, int] = {}
        self.padded_samples = 0
        self.resyncs = 0

    # ------------------------------------------------------------------- public

    def feed(self, chunk: AudioChunk) -> list[Utterance]:
        """Consume one chunk; return every utterance that became available."""
        if chunk.track is not self.track:
            raise ValueError(
                f"chunk from {chunk.track.value} fed to {self.track.value} segmenter"
            )
        out: list[Utterance] = []

        if self._anchor_s is None:
            self._anchor_s = chunk.start_s
            self._anchor_fed = self._fed
        else:
            out.extend(self._handle_discontinuity(chunk))

        self._pending = (
            chunk.samples
            if self._pending.size == 0
            else np.concatenate((self._pending, chunk.samples))
        )
        self._expected_s = chunk.end_s
        self._consume_windows()
        out.extend(self._collect())
        return out

    def flush(self) -> list[Utterance]:
        """End of stream: pad to a whole window, force the VAD out, collect."""
        if self._pending.size:
            missing = (-self._pending.size) % WINDOW
            if missing:
                self._pending = np.concatenate(
                    (self._pending, np.zeros(missing, dtype=np.float32))
                )
            self._consume_windows()
        self.engine.flush()
        out = self._collect(final_only=True)
        if self._speaking:
            self._speaking = False
            if self.on_speech_change is not None:
                self.on_speech_change(False, self._time_of(self._fed))
        return out

    # ------------------------------------------------------------------ internal

    def _handle_discontinuity(self, chunk: AudioChunk) -> list[Utterance]:
        """Keep the VAD's sample timeline aligned with session time.

        A dropped chunk is filled with silence, which both preserves the index
        mapping and naturally closes whatever utterance was in progress -- exactly
        what should happen when the audio it belonged to is gone. A gap too large
        to pad restarts the VAD and re-anchors.
        """
        assert self._expected_s is not None  # noqa: S101 - narrowing, not validation
        gap = chunk.start_s - self._expected_s
        if abs(gap) <= self.gap_tolerance_s:
            return []

        if 0 < gap <= self.max_pad_s:
            pad = np.zeros(round(gap * PIPELINE_SAMPLE_RATE), dtype=np.float32)
            self._pending = (
                pad if self._pending.size == 0 else np.concatenate((self._pending, pad))
            )
            self.padded_samples += int(pad.size)
            log.debug("%s: padded %.2fs gap", self.track.value, gap)
            return []

        # Either a very long outage (device reconnect) or the timeline re-anchored
        # backwards. Restart cleanly rather than emit nonsense timestamps.
        log.info("%s: resync after %.2fs discontinuity", self.track.value, gap)
        self.resyncs += 1
        out = self._drain_engine()
        self.engine.reset()
        self._seq_base += self._fed + 1
        self._pending = np.zeros(0, dtype=np.float32)
        self._fed = 0
        self._anchor_fed = 0
        self._anchor_s = chunk.start_s
        self._draft_marks.clear()
        return out

    def _consume_windows(self) -> None:
        n = (self._pending.size // WINDOW) * WINDOW
        if n == 0:
            return
        block = self._pending[:n]
        self._pending = self._pending[n:]
        for i in range(0, n, WINDOW):
            self.engine.accept_waveform(block[i : i + WINDOW])
        self._fed += n

    def _collect(self, final_only: bool = False) -> list[Utterance]:
        out = self._drain_engine()
        speaking = self.engine.is_speech_detected()
        if speaking != self._speaking:
            self._speaking = speaking
            if self.on_speech_change is not None:
                self.on_speech_change(speaking, self._time_of(self._fed))
        if not final_only and speaking:
            draft = self._maybe_draft()
            if draft is not None:
                out.append(draft)
        return out

    def _drain_engine(self) -> list[Utterance]:
        out: list[Utterance] = []
        while not self.engine.empty():
            seg = self.engine.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            seq = self._seq_base + int(seg.start)
            self._draft_marks.pop(seq, None)
            out.append(
                Utterance(
                    track=self.track,
                    samples=samples,
                    start_s=self._time_of(int(seg.start)),
                    end_s=self._time_of(int(seg.start) + samples.size),
                    seq=seq,
                    is_partial=False,
                )
            )
            self.engine.pop()
        return out

    def _maybe_draft(self) -> Utterance | None:
        """Publish provisional text for an utterance that is still running."""
        try:
            seg = self.engine.current_segment
        except Exception:
            return None
        if seg is None:
            return None
        samples = np.asarray(getattr(seg, "samples", ()), dtype=np.float32)
        if samples.size == 0:
            return None
        if samples.size < self.interim_after_s * PIPELINE_SAMPLE_RATE:
            return None

        seq = self._seq_base + int(seg.start)
        marked = self._draft_marks.get(seq)
        # The first draft only has to clear ``interim_after_s``; the rate limit
        # applies to the ones after it, so a long turn does not re-decode on
        # every window.
        if (
            marked is not None
            and samples.size - marked < self.interim_every_s * PIPELINE_SAMPLE_RATE
        ):
            return None
        self._draft_marks[seq] = int(samples.size)
        return Utterance(
            track=self.track,
            samples=samples,
            start_s=self._time_of(int(seg.start)),
            end_s=self._time_of(int(seg.start) + samples.size),
            seq=seq,
            is_partial=True,
        )

    def _time_of(self, fed_index: int) -> float:
        base = self._anchor_s if self._anchor_s is not None else 0.0
        return base + (fed_index - self._anchor_fed) / PIPELINE_SAMPLE_RATE
