"""WAV input and output.

Used by the replay capture source (regression tests over a recorded meeting), by
the fixture generator, and by the optional raw-audio retention path. Deliberately
built on the standard library's ``wave`` module: no extra dependency, and the
16-bit PCM it writes opens in every tool the user already has.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from meetscribe.audio.resample import downmix, resample_offline, to_float32
from meetscribe.core.models import PIPELINE_SAMPLE_RATE

__all__ = ["read_wav", "read_wav_mono", "write_wav", "WavWriter"]


def read_wav(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM WAV as ``(interleaved float32, sample_rate, channels)``."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"{path}: unsupported sample width {width * 8}-bit")
    frames = np.frombuffer(raw, dtype=dtypes[width])
    return to_float32(frames), rate, channels


def read_wav_mono(
    path: str | Path, target_rate: int = PIPELINE_SAMPLE_RATE
) -> np.ndarray:
    """Read a WAV as mono float32 at ``target_rate``, ready for the pipeline."""
    frames, rate, channels = read_wav(path)
    mono = downmix(frames, channels)
    return np.clip(resample_offline(mono, rate, target_rate), -1.0, 1.0)


def write_wav(
    path: str | Path, samples: np.ndarray, sample_rate: int = PIPELINE_SAMPLE_RATE
) -> None:
    """Write mono float32 samples as 16-bit PCM."""
    with WavWriter(path, sample_rate=sample_rate) as writer:
        writer.write(samples)


class WavWriter:
    """Incremental 16-bit PCM writer.

    ``wave`` patches the RIFF header on close, so a file left behind by a crash is
    truncated but still readable by most tools -- which matters, because the whole
    point of retaining raw audio is to have it after something went wrong.
    """

    __slots__ = ("path", "sample_rate", "_wave", "_frames")

    def __init__(
        self, path: str | Path, sample_rate: int = PIPELINE_SAMPLE_RATE
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wave = wave.open(str(self.path), "wb")  # noqa: SIM115 - closed in close()
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(sample_rate)
        self._frames = 0

    def write(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        self._wave.writeframes((pcm * 32767.0).astype("<i2").tobytes())
        self._frames += int(pcm.size)

    @property
    def duration_s(self) -> float:
        return self._frames / self.sample_rate

    def close(self) -> None:
        if self._wave is not None:
            self._wave.close()
            self._wave = None  # type: ignore[assignment]

    def __enter__(self) -> WavWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
