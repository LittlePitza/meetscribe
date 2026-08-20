"""Timestamp alignment between the two capture tracks.

This is the subtlest part of the system, so the reasoning is written down rather
than left in the code's shape.

The microphone and the loopback endpoint live in **different clock domains**. Each
device is driven by its own crystal, and neither runs at exactly 48 000 Hz. Over a
two-hour meeting a typical 50 ppm error accumulates to about 0.36 s. If the two
tracks are timestamped independently by counting samples, they slide apart and the
merged transcript starts interleaving turns in the wrong order.

The naive alternative -- stamping each chunk with wall-clock time at the moment the
capture thread happens to be scheduled -- is worse: thread scheduling jitter on
Windows is tens of milliseconds, so timestamps become noisy at exactly the scale
that decides whether "si, de acuerdo" lands before or after the sentence it
answers.

So the design uses both, with different jobs:

* **Sample counting** provides *local* timing. Within one track it is exact and
  jitter-free, which is what makes segment boundaries land on real word edges.
* **Wall clock** provides the *global* anchor. It is noisy but does not drift, so
  it is used only to detect that the sample-derived time has wandered too far, and
  to re-anchor when it has.

The result is a piecewise-linear mapping from sample index to session time: smooth
almost everywhere, with rare corrections bounded by ``max_drift_s``. Every
correction is counted and reported, because a track that re-anchors constantly is
telling you its device is misbehaving.

A file replay source has no clock domain at all -- it is read as fast as the disk
allows -- so it declares ``realtime=False`` and gets pure sample-derived time. That
is what makes regression tests over a recorded meeting bit-for-bit deterministic.
"""

from __future__ import annotations

__all__ = ["TrackTimeline"]


class TrackTimeline:
    """Maps a track's sample counter onto the shared session timeline.

    All times are seconds relative to the start of the session. Not thread-safe by
    itself: exactly one capture thread owns each instance.
    """

    __slots__ = (
        "sample_rate",
        "realtime",
        "max_drift_s",
        "_samples",
        "_anchor_samples",
        "_anchor_media_s",
        "_drift_s",
        "_reanchors",
        "_dropped_samples",
    )

    def __init__(
        self,
        sample_rate: int,
        *,
        realtime: bool = True,
        max_drift_s: float = 0.5,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if max_drift_s <= 0:
            raise ValueError(f"max_drift_s must be positive, got {max_drift_s}")
        self.sample_rate = sample_rate
        self.realtime = realtime
        self.max_drift_s = max_drift_s
        self._samples = 0
        self._anchor_samples = 0
        self._anchor_media_s: float | None = None
        self._drift_s = 0.0
        self._reanchors = 0
        self._dropped_samples = 0

    # ------------------------------------------------------------------ queries

    @property
    def samples(self) -> int:
        """Total samples accounted for, including ones dropped under backpressure."""
        return self._samples

    @property
    def media_s(self) -> float:
        """Session time of the end of the audio stamped so far."""
        if self._anchor_media_s is None:
            return 0.0
        return self._anchor_media_s + (
            self._samples - self._anchor_samples
        ) / self.sample_rate

    @property
    def drift_s(self) -> float:
        """Most recent (sample-derived minus wall-clock) discrepancy.

        Positive means the device is producing samples faster than real time.
        """
        return self._drift_s

    @property
    def reanchors(self) -> int:
        """How many times drift exceeded the tolerance. Persistent growth here is
        a real hardware or driver problem, not a rounding artefact."""
        return self._reanchors

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    # ------------------------------------------------------------------ mutation

    def stamp(self, n_samples: int, arrival_s: float) -> float:
        """Account for ``n_samples`` delivered at session time ``arrival_s``.

        Returns the session time at which the first of those samples was captured.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if not self.realtime:
            start = self._samples / self.sample_rate
            self._samples += n_samples
            if self._anchor_media_s is None:
                self._anchor_media_s = 0.0
            return start

        # The chunk was read once it was complete, so its audio began one chunk
        # duration before the read returned.
        observed_start = arrival_s - n_samples / self.sample_rate

        if self._anchor_media_s is None:
            # The first chunk fixes this track's offset on the session timeline,
            # which is how the two tracks' different start instants are captured.
            self._anchor_media_s = max(0.0, observed_start)
            self._anchor_samples = self._samples

        nominal_start = self._anchor_media_s + (
            self._samples - self._anchor_samples
        ) / self.sample_rate
        self._drift_s = nominal_start - observed_start

        if abs(self._drift_s) > self.max_drift_s:
            self._anchor_media_s = max(0.0, observed_start)
            self._anchor_samples = self._samples
            self._reanchors += 1
            nominal_start = self._anchor_media_s

        self._samples += n_samples
        return nominal_start

    def skip(self, n_samples: int) -> None:
        """Account for audio that was captured but discarded under backpressure.

        Without this the timeline would silently compress: every later segment
        would be reported earlier than it happened, by the total dropped duration.
        Dropping audio is bad; dropping it *and* corrupting every subsequent
        timestamp is worse.
        """
        if n_samples <= 0:
            return
        self._samples += n_samples
        self._dropped_samples += n_samples
