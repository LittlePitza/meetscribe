"""Speech to text.

**Model choice.** Measured on the target CPU against real Spanish speech with known
ground truth (``docs/benchmarks.md``):

| model          | RTF   | speed | WER   |
|----------------|-------|-------|-------|
| tiny           | 0.024 | 42x   | 8.6 % |
| base           | 0.045 | 22x   | 7.3 % |
| small          | 0.124 | 8.1x  | 6.0 % |
| medium         | 0.348 | 2.9x  | 7.3 % |
| large-v3-turbo | 0.305 | 3.3x  | 4.6 % |

Two consequences are baked into this module. ``large-v3-turbo`` is the default:
even the largest viable model runs at 3.3x real time, so the CPU-only constraint
that looked like the project's main risk is not binding. And ``medium`` is
**refused outright** -- it is both slower and less accurate than
``large-v3-turbo``, so offering it could only ever be a mistake.

**Decoder settings, and why.**

* ``vad_filter=False``. The upstream segmenter has already cut at silence; running
  faster-whisper's own VAD again costs CPU and tends to trim short backchannels
  ("si", "aja") that carry the turn-taking structure.
* ``condition_on_previous_text=False``. Carrying decoder state across utterances
  improves coherence, and propagates hallucination loops: once the model starts
  repeating, the repetition becomes its own prompt. Utterance-level segmentation
  leaves little context to lose, and the failure mode it prevents is severe.
* Temperature fallback ``0.0 -> 1.0``. Greedy decoding first; the ladder only
  engages when the confidence thresholds reject the greedy result.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from meetscribe.core.models import PIPELINE_SAMPLE_RATE, TranscriptionResult
from meetscribe.errors import ConfigError, TranscriptionError

log = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_MODELS",
    "REJECTED_MODELS",
    "FasterWhisperTranscriber",
    "FakeTranscriber",
]

SUPPORTED_MODELS: tuple[str, ...] = ("tiny", "base", "small", "large-v3-turbo")
"""Offered to the user. Ordered fastest to most accurate."""

REJECTED_MODELS: dict[str, str] = {
    "medium": (
        "'medium' está dominado: es más lento que 'large-v3-turbo' (RTF 0.348 vs "
        "0.305) y menos preciso (WER 7.3% vs 4.6%). Usa 'large-v3-turbo'."
    ),
}


class FasterWhisperTranscriber:
    """CTranslate2 Whisper, int8, CPU.

    One instance is shared by both tracks behind a single worker thread. That is a
    measured decision, not a simplification: two instances would double resident
    memory (~1.6 GB each for ``large-v3-turbo``) and have the two decoders fight
    over the same eight cores, for no gain -- meeting speech is largely
    turn-taking, so the tracks rarely need the decoder at the same instant, and the
    queue absorbs the overlap that does occur.

    The lock is belt-and-braces for the diagnostic paths (``doctor``, re-running a
    summary) that may call in from another thread.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        *,
        language: str = "es",
        cpu_threads: int = 0,
        beam_size: int = 5,
        compute_type: str = "int8",
        download_root: str | None = None,
        auto_detect_min_s: float = 2.0,
        allowed_languages: Sequence[str] = ("es", "en"),
        local_files_only: bool = False,
    ) -> None:
        if model_size in REJECTED_MODELS:
            raise ConfigError(
                f"model {model_size!r} is dominated",
                user_message=REJECTED_MODELS[model_size],
            )
        if model_size not in SUPPORTED_MODELS:
            raise ConfigError(
                f"unknown model {model_size!r}",
                user_message=(
                    f"Modelo de transcripción desconocido: '{model_size}'. "
                    f"Opciones: {', '.join(SUPPORTED_MODELS)}."
                ),
            )
        import os

        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.compute_type = compute_type
        self.download_root = download_root
        self.auto_detect_min_s = auto_detect_min_s
        self.allowed_languages = tuple(allowed_languages)
        self.local_files_only = local_files_only
        self.cpu_threads = cpu_threads or min(8, os.cpu_count() or 4)
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model_size}/{self.compute_type}"

    def load(self) -> None:
        """Materialise the model. Separate from ``__init__`` so the CLI can show
        "descargando modelo..." before a multi-hundred-megabyte first run."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        log.info(
            "loading %s (%s, %d threads)",
            self.model_size,
            self.compute_type,
            self.cpu_threads,
        )
        try:
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=self.download_root,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"cannot load {self.model_size}: {type(exc).__name__}: {exc}",
                user_message=(
                    f"No se pudo cargar el modelo '{self.model_size}'. "
                    "Si es la primera vez, necesita descargarse (requiere red). "
                    "Ejecuta 'meetscribe doctor' para diagnosticar."
                ),
            ) from exc

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        if self._model is None:
            self.load()
        audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if audio.size == 0:
            return TranscriptionResult(text="")

        primary = language or self.language
        duration_s = audio.size / PIPELINE_SAMPLE_RATE
        requested = self._language_for(primary, duration_s)

        with self._lock:
            result = self._run(audio, requested, initial_prompt)
            # Code-switching guard: a short or noisy utterance can be auto-detected
            # as an unrelated language, which yields confident nonsense. One retry
            # pinned to the primary language is cheaper than a wrong transcript.
            if (
                requested is None
                and result.language
                and result.language not in self.allowed_languages
            ):
                log.debug(
                    "detected %s outside allowed set; retrying as %s",
                    result.language,
                    primary,
                )
                result = self._run(audio, primary, initial_prompt)
        return result

    def _language_for(self, primary: str, duration_s: float) -> str | None:
        """``None`` asks faster-whisper to detect.

        Detection is only trustworthy with enough audio, so short utterances stay
        pinned to the meeting's primary language.
        """
        if primary != "auto":
            return primary
        return None if duration_s >= self.auto_detect_min_s else "es"

    def _run(
        self, audio: np.ndarray, language: str | None, initial_prompt: str | None
    ) -> TranscriptionResult:
        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                word_timestamps=False,
            )
            segs = list(segments)
        except Exception as exc:
            raise TranscriptionError(
                f"transcription failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not segs:
            return TranscriptionResult(
                text="",
                language=getattr(info, "language", language),
                no_speech_prob=1.0,
            )

        text = " ".join(s.text.strip() for s in segs if s.text).strip()
        total = sum(max(1e-6, s.end - s.start) for s in segs)
        avg_logprob = (
            sum(s.avg_logprob * max(1e-6, s.end - s.start) for s in segs) / total
        )
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", language),
            avg_logprob=float(avg_logprob),
            # Worst case across the segments: one hallucinated tail is enough to
            # reject the whole utterance.
            no_speech_prob=float(max(s.no_speech_prob for s in segs)),
            compression_ratio=float(max(s.compression_ratio for s in segs)),
        )

    def close(self) -> None:
        with self._lock:
            self._model = None


class FakeTranscriber:
    """Deterministic stand-in for the whole pipeline's tests.

    Either scripted (a list of results consumed in order) or derived from the audio
    by a callable, so a test can assert that the *right* audio reached the decoder,
    not merely that something did.
    """

    def __init__(
        self,
        script: Sequence[str | TranscriptionResult] | None = None,
        fn: Callable[[np.ndarray], str] | None = None,
        default: str = "texto de prueba",
    ) -> None:
        self._script = list(script or [])
        self._fn = fn
        self._default = default
        self.calls: list[tuple[int, str | None]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fake"

    def load(self) -> None:
        return None

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        with self._lock:
            self.calls.append((int(np.asarray(samples).size), language))
            if self._script:
                item = self._script.pop(0)
                return (
                    item
                    if isinstance(item, TranscriptionResult)
                    else TranscriptionResult(text=item, language=language or "es")
                )
        text = self._fn(samples) if self._fn is not None else self._default
        return TranscriptionResult(
            text=text,
            language=language or "es",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            compression_ratio=1.4,
        )

    def close(self) -> None:
        return None
