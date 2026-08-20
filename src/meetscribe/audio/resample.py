"""48 kHz stereo device audio to 16 kHz mono float32, the format everything
downstream expects.

**Which resampler.** Three options were weighed against the measured CPU budget
(``docs/benchmarks.md``: the whole pipeline runs at RTF ~0.32, so resampling
must be negligible, not merely cheap):

* *Naive decimation* -- take every third sample. Free, and wrong: 48 kHz content
  above 8 kHz aliases straight down into the speech band. Whisper is trained on
  band-limited 16 kHz audio and an aliased sibilant is not a quiet artefact, it is
  a wrong phoneme.
* ``scipy.signal.resample_poly`` -- correct polyphase FIR, exact rational ratio
  1/3 for 48000 to 16000. Excellent quality, but stateless: called per chunk it
  produces a discontinuity at every boundary, and the filter's edge transient
  lands in the audio dozens of times per second.
* ``soxr`` -- the SoX resampler, which exposes a **stateful** ``ResampleStream``
  that carries filter state across chunks. Same quality class as ``resample_poly``
  at a fraction of a percent of one core, and no boundary artefacts.

``soxr`` wins for the live path precisely because of the streaming state.
``resample_poly`` is kept for the offline path (WAV fixtures, replay sources)
where the whole signal is available at once and determinism matters more than
statefulness.

The one behaviour that surprises people: a stateful resampler has *latency*. The
first ``process`` call returns fewer samples than the ratio implies, because the
filter is still filling. The deficit is constant and is repaid by ``flush``. The
timeline stamps by sample count, so this shows up as a fixed sub-10 ms offset on
both tracks equally -- not as drift.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from meetscribe.core.models import PIPELINE_SAMPLE_RATE

__all__ = ["downmix", "resample_offline", "StreamResampler", "to_float32"]


def to_float32(frames: np.ndarray) -> np.ndarray:
    """Normalise any supported sample dtype to float32 in [-1, 1]."""
    if frames.dtype == np.float32:
        return frames
    if frames.dtype == np.float64:
        return frames.astype(np.float32)
    if frames.dtype == np.int16:
        return frames.astype(np.float32) / 32768.0
    if frames.dtype == np.int32:
        return frames.astype(np.float32) / 2147483648.0
    if frames.dtype == np.uint8:
        return (frames.astype(np.float32) - 128.0) / 128.0
    raise TypeError(f"unsupported sample dtype: {frames.dtype}")


def downmix(frames: np.ndarray, channels: int) -> np.ndarray:
    """Interleaved multi-channel frames to mono, by averaging.

    Averaging rather than taking the left channel: a Teams call can put one
    participant predominantly in one ear, and dropping a channel would silently
    drop that person from the transcript.
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    flat = np.asarray(frames).reshape(-1)
    if channels == 1:
        return flat
    usable = (flat.size // channels) * channels
    if usable != flat.size:
        # A partial frame at the end of a chunk; the tail belongs to the next read.
        flat = flat[:usable]
    return flat.reshape(-1, channels).mean(axis=1, dtype=np.float32)


def resample_offline(
    samples: np.ndarray, in_rate: int, out_rate: int = PIPELINE_SAMPLE_RATE
) -> np.ndarray:
    """Whole-signal polyphase resampling, for files and fixtures.

    Deterministic and dependency-light -- the same code path the spikes used, so
    benchmark numbers and test fixtures stay comparable.
    """
    samples = to_float32(np.asarray(samples).reshape(-1))
    if in_rate == out_rate:
        return samples
    from scipy.signal import resample_poly  # imported lazily: not on the hot path

    g = gcd(int(in_rate), int(out_rate))
    return np.ascontiguousarray(
        resample_poly(samples, out_rate // g, in_rate // g).astype(np.float32)
    )


class StreamResampler:
    """Stateful device-format to pipeline-format converter, one per track.

    Not thread-safe: exactly one capture thread owns each instance. Carries the
    resampler's filter state, so consecutive chunks join without a seam.
    """

    __slots__ = ("in_rate", "in_channels", "out_rate", "_stream", "_closed")

    def __init__(
        self,
        in_rate: int,
        in_channels: int,
        out_rate: int = PIPELINE_SAMPLE_RATE,
        quality: str = "HQ",
    ) -> None:
        if in_rate <= 0 or in_channels <= 0:
            raise ValueError("in_rate and in_channels must be positive")
        self.in_rate = int(in_rate)
        self.in_channels = int(in_channels)
        self.out_rate = int(out_rate)
        self._closed = False
        self._stream = None
        if self.in_rate != self.out_rate:
            import soxr  # imported lazily so the domain stays importable without it

            self._stream = soxr.ResampleStream(
                self.in_rate, self.out_rate, 1, dtype="float32", quality=quality
            )

    def process(self, frames: np.ndarray) -> np.ndarray:
        """Downmix, resample and return mono float32 at the pipeline rate."""
        if self._closed:
            raise RuntimeError("StreamResampler is closed")
        mono = downmix(to_float32(np.asarray(frames)), self.in_channels)
        if mono.size == 0:
            return mono
        if self._stream is None:
            return np.ascontiguousarray(mono)
        return np.ascontiguousarray(
            self._stream.resample_chunk(np.ascontiguousarray(mono))
        )

    def flush(self) -> np.ndarray:
        """Drain the filter's internal delay line at end of stream.

        Called once when a track stops, so the last few hundred milliseconds of a
        closing sentence are not lost with the stream.
        """
        if self._closed or self._stream is None:
            self._closed = True
            return np.zeros(0, dtype=np.float32)
        self._closed = True
        tail = self._stream.resample_chunk(
            np.zeros(0, dtype=np.float32), last=True
        )
        return np.ascontiguousarray(np.asarray(tail, dtype=np.float32))
