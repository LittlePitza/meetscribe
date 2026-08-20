"""Speaker embeddings.

**Why sherpa-onnx and not pyannote.** pyannote.audio is the reference
implementation and it is very good, but using it means accepting gated
user-conditions on Hugging Face and carrying a Hugging Face token -- onboarding
friction for a tool whose entire premise is "runs locally, nothing to sign up
for" -- and it drags PyTorch in behind it. sherpa-onnx ships the same capability
class as a 2.2 MB wheel over onnxruntime, with no PyTorch and no gated licences.

The measured numbers (``docs/benchmarks.md``) settled it. NeMo TitaNet-small,
38 MB of ONNX, RTF 0.0091 -- 110x real time -- and on two genuinely different
voices the separation is wide:

    same speaker      cosine mean 0.909, min 0.868
    different speaker cosine mean 0.384, max 0.434
    margin            +0.524, distributions do not overlap

That gap is what makes a single global threshold viable. The operating point sits
mid-gap at 0.60 rather than hugging either distribution, so it degrades gracefully
when real meeting audio narrows the margin.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from meetscribe.core.models import PIPELINE_SAMPLE_RATE
from meetscribe.errors import DiarizationUnavailableError, ModelNotFoundError

log = logging.getLogger(__name__)

__all__ = ["TitanetEmbedder", "FakeEmbedder", "l2_normalise"]


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Project onto the unit sphere so cosine similarity is a plain dot product."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return v
    return (v / norm).astype(np.float32)


class TitanetEmbedder:
    """NeMo TitaNet-small via sherpa-onnx."""

    __slots__ = ("model_path", "num_threads", "_extractor", "_lock", "_dim")

    def __init__(self, model_path: str, num_threads: int = 2) -> None:
        if not Path(model_path).exists():
            raise ModelNotFoundError(
                f"speaker embedding model missing at {model_path}",
                user_message=(
                    "Falta el modelo de identificación de hablantes "
                    "(nemo_en_titanet_small.onnx). Ejecuta "
                    "'meetscribe models --download'."
                ),
            )
        self.model_path = model_path
        self.num_threads = num_threads
        self._extractor = None
        self._lock = threading.Lock()
        self._dim = 0

    @property
    def dim(self) -> int:
        if self._extractor is None:
            self.load()
        return self._dim

    def load(self) -> None:
        if self._extractor is not None:
            return
        try:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.model_path, num_threads=self.num_threads
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            self._dim = int(self._extractor.dim)
        except Exception as exc:
            raise DiarizationUnavailableError(
                f"cannot load speaker embedder: {type(exc).__name__}: {exc}"
            ) from exc

    def embed(
        self, samples: np.ndarray, sample_rate: int = PIPELINE_SAMPLE_RATE
    ) -> np.ndarray:
        if self._extractor is None:
            self.load()
        audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if audio.size == 0:
            return np.zeros(self._dim, dtype=np.float32)
        # sherpa's extractor is stateful per stream; a fresh stream per utterance
        # is both correct and cheap (the cost is dominated by the ONNX forward
        # pass, measured at RTF 0.0091).
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
            stream.input_finished()
            vector = self._extractor.compute(stream)
        return l2_normalise(np.array(vector, dtype=np.float32))

    def close(self) -> None:
        with self._lock:
            self._extractor = None


class FakeEmbedder:
    """Maps audio to a unit vector by a deterministic rule.

    Lets the clustering logic -- the part with the actual decisions in it -- be
    tested exhaustively without ONNX, model files or real voices.
    """

    __slots__ = ("_dim", "_rule", "calls")

    def __init__(self, dim: int = 8, rule=None) -> None:
        self._dim = dim
        self._rule = rule
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(
        self, samples: np.ndarray, sample_rate: int = PIPELINE_SAMPLE_RATE
    ) -> np.ndarray:
        self.calls += 1
        if self._rule is not None:
            return l2_normalise(np.asarray(self._rule(samples), dtype=np.float32))
        # Default rule: the mean sample value selects an axis, so a test can make
        # "speaker N" simply by generating a constant-valued buffer.
        mean = float(np.mean(samples)) if np.asarray(samples).size else 0.0
        axis = int(abs(mean) * 1000) % self._dim
        vector = np.zeros(self._dim, dtype=np.float32)
        vector[axis] = 1.0
        return vector

    def close(self) -> None:
        return None
