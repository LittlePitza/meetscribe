"""Online speaker diarization for the loopback track.

The microphone track is trivially "Yo". The loopback track carries every remote
participant mixed into one stream and has to be split into stable
"Participante 1/2/3" labels, live, on CPU, with the number of speakers unknown and
growing.

**Why online centroid clustering.** Offline spectral clustering -- the accurate
choice -- needs the whole recording, which is exactly what a live panel does not
have. So the live path is incremental: embed each utterance, compare it to every
known speaker's centroid, and either attach it to the nearest one or open a new
cluster. The measured margin between same-speaker and different-speaker cosine
similarity (0.868 min vs 0.434 max, ``docs/benchmarks.md``) is what makes this
simple approach workable at all; the threshold sits at **0.60**, mid-gap.

**Three rules keep it honest as audio gets harder:**

1. *Short utterances never create a speaker.* A 400 ms "aja" produces an unstable
   embedding. It is assigned to the nearest existing centroid and, crucially, does
   **not** update that centroid -- so backchannels can be attributed without
   contaminating the model of anyone's voice.
2. *Labels are append-only during the meeting.* "Participante 2" never silently
   becomes "Participante 5" mid-session, because a rename mid-meeting makes the
   live transcript unreadable and destroys the user's trust in the attribution.
3. *A cap on speakers.* Past ``max_speakers`` the algorithm stops opening clusters
   and attaches to the nearest. Unbounded cluster creation is how online diarizers
   degrade into noise on a bad line.

**Retroactive refinement.** At end of session every stored embedding is re-clustered
offline with average-linkage agglomerative clustering, which can see the whole
meeting and fixes early mistakes made when a speaker had only one utterance. The
result is then *reconciled* with the labels already shown: each refined cluster
inherits the label the majority of its members already had, so the corrections show
up as individual lines moving between speakers rather than as a wholesale
renumbering of the transcript.

**Honest accuracy expectation.** On two clean, well-separated voices this is close
to perfect. On four to six people over a compressed Teams call with crosstalk,
expect confusions on short turns and on speakers with similar voices; overlapping
speech is attributed to whoever dominates the segment, because a single embedding
per utterance cannot represent two people at once. The design's answer to being
wrong is that it is easy to fix afterwards -- rename applies retroactively -- and
that being wrong never stops the transcript.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np

from meetscribe.core.models import (
    PIPELINE_SAMPLE_RATE,
    UNKNOWN_SPEAKER_ID,
    Speaker,
    TrackKind,
    Utterance,
)
from meetscribe.diarize.embedder import l2_normalise

log = logging.getLogger(__name__)

__all__ = ["OnlineDiarizer", "NullDiarizer", "DiarizerParams"]


@dataclass(frozen=True, slots=True)
class DiarizerParams:
    """Every tunable, with the value chosen and why."""

    threshold: float = 0.60
    """Cosine similarity above which an utterance joins an existing speaker.
    Mid-gap between the measured 0.434 (different) and 0.868 (same)."""

    min_embed_s: float = 0.35
    """Below this there is not enough voiced audio for TitaNet to say anything."""

    min_new_speaker_s: float = 1.0
    """Opening a new cluster needs a confident embedding, so it needs real audio."""

    max_speakers: int = 8
    """A hard stop on cluster proliferation. Meetings with more remote voices than
    this exist, but past it the labels stop being useful anyway."""

    max_observations: int = 4000
    """Bound on retained embeddings, ~3 MB at 192 dimensions. A four-hour meeting
    of continuous speech would reach it; the refinement pass then works from the
    first 4000, which is ample to characterise every speaker."""

    centroid_momentum: float = 0.0
    """0 means a plain running mean. Raise it to weight recent audio more heavily
    when a speaker's channel conditions change mid-call."""


@dataclass(slots=True)
class _Observation:
    key: tuple[str, int]
    embedding: np.ndarray
    speaker_id: str
    duration_s: float


@dataclass(slots=True)
class _Cluster:
    speaker_id: str
    display_name: str
    centroid: np.ndarray
    count: int = 0

    def absorb(self, embedding: np.ndarray, momentum: float = 0.0) -> None:
        if momentum > 0.0:
            blended = (1.0 - momentum) * self.centroid + momentum * embedding
        else:
            blended = self.centroid * self.count + embedding
        self.centroid = l2_normalise(blended)
        self.count += 1


class OnlineDiarizer:
    """Incremental speaker attribution with an offline refinement pass."""

    def __init__(
        self,
        embedder,
        params: DiarizerParams | None = None,
        label_prefix: str = "Participante",
    ) -> None:
        self.embedder = embedder
        self.params = params or DiarizerParams()
        self.label_prefix = label_prefix
        self._clusters: list[_Cluster] = []
        self._observations: list[_Observation] = []
        self._last_speaker_id = UNKNOWN_SPEAKER_ID
        self._lock = threading.Lock()
        self._dropped_observations = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return True

    @property
    def last_speaker_id(self) -> str:
        """Whoever the most recent decidable utterance was attributed to.

        Used as the provisional label on draft segments, which are published before
        their audio is complete enough to embed.
        """
        return self._last_speaker_id

    # ------------------------------------------------------------------- online

    def assign(self, utterance: Utterance) -> str:
        """Attribute one utterance. Never raises; a failure degrades to 'Remoto'."""
        duration = utterance.duration_s
        if duration < self.params.min_embed_s:
            # Too short to embed. Attribute it to whoever the model last saw rather
            # than inventing a speaker, and say nothing about the centroids.
            return self._last_speaker_id

        try:
            embedding = self.embedder.embed(utterance.samples, PIPELINE_SAMPLE_RATE)
        except Exception as exc:
            self.failures += 1
            log.warning("embedding failed (%s); falling back to '%s'", exc, UNKNOWN_SPEAKER_ID)
            return UNKNOWN_SPEAKER_ID
        if embedding.size == 0 or not np.isfinite(embedding).all():
            self.failures += 1
            return UNKNOWN_SPEAKER_ID

        with self._lock:
            speaker_id = self._assign_locked(embedding, duration)
            if len(self._observations) < self.params.max_observations:
                self._observations.append(
                    _Observation(utterance.key, embedding, speaker_id, duration)
                )
            else:
                self._dropped_observations += 1
            self._last_speaker_id = speaker_id
        return speaker_id

    def _assign_locked(self, embedding: np.ndarray, duration: float) -> str:
        if not self._clusters:
            if duration >= self.params.min_new_speaker_s:
                return self._open_cluster(embedding).speaker_id
            return UNKNOWN_SPEAKER_ID

        similarities = np.array(
            [float(np.dot(embedding, c.centroid)) for c in self._clusters],
            dtype=np.float32,
        )
        best = int(np.argmax(similarities))
        best_similarity = float(similarities[best])

        if best_similarity >= self.params.threshold:
            # Only well-founded utterances are allowed to move a centroid; a short
            # backchannel gets the label without shaping the voice model.
            if duration >= self.params.min_new_speaker_s:
                self._clusters[best].absorb(embedding, self.params.centroid_momentum)
            return self._clusters[best].speaker_id

        if (
            duration >= self.params.min_new_speaker_s
            and len(self._clusters) < self.params.max_speakers
        ):
            return self._open_cluster(embedding).speaker_id

        # Below threshold but not eligible for a new cluster: nearest wins, and the
        # centroid stays untouched so a wrong guess does not compound.
        return self._clusters[best].speaker_id

    def _open_cluster(self, embedding: np.ndarray) -> _Cluster:
        index = len(self._clusters) + 1
        cluster = _Cluster(
            speaker_id=f"p{index}",
            display_name=f"{self.label_prefix} {index}",
            centroid=l2_normalise(embedding),
            count=1,
        )
        self._clusters.append(cluster)
        log.info("new speaker cluster: %s", cluster.display_name)
        return cluster

    # ------------------------------------------------------------------ offline

    def refine(self) -> dict[tuple[str, int], str]:
        """Re-cluster everything; return only the assignments that changed."""
        with self._lock:
            observations = list(self._observations)
        if len(observations) < 3:
            return {}

        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
        except ImportError:  # pragma: no cover - scipy is a hard dependency
            return {}

        matrix = np.stack([o.embedding for o in observations])
        distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        # Average linkage: single linkage chains two speakers together through one
        # ambiguous utterance, complete linkage splits a speaker whose voice varies
        # across the call. Average is the standard compromise and is what the spike
        # measured 100 % purity with.
        linkage_matrix = linkage(squareform(distances, checks=False), method="average")
        labels = fcluster(
            linkage_matrix, t=1.0 - self.params.threshold, criterion="distance"
        )

        assignment: dict[tuple[str, int], str] = {}
        taken: set[str] = set()
        order = sorted(set(labels), key=lambda c: -int(np.sum(labels == c)))
        with self._lock:
            for cluster_label in order:
                members = np.flatnonzero(labels == cluster_label)
                votes = Counter(observations[i].speaker_id for i in members)
                chosen = ""
                for candidate, _ in votes.most_common():
                    if candidate not in taken and candidate != UNKNOWN_SPEAKER_ID:
                        chosen = candidate
                        break
                if not chosen:
                    chosen = self._open_cluster(
                        l2_normalise(
                            np.mean([observations[i].embedding for i in members], axis=0)
                        )
                    ).speaker_id
                taken.add(chosen)
                for i in members:
                    if observations[i].speaker_id != chosen:
                        assignment[observations[i].key] = chosen
                    observations[i].speaker_id = chosen

            # Recompute centroids and counts from the refined membership so any
            # later work (a resumed session) starts from the better model.
            by_speaker: dict[str, list[np.ndarray]] = {}
            for observation in observations:
                by_speaker.setdefault(observation.speaker_id, []).append(
                    observation.embedding
                )
            for cluster in self._clusters:
                vectors = by_speaker.get(cluster.speaker_id)
                if vectors:
                    cluster.centroid = l2_normalise(np.mean(vectors, axis=0))
                    cluster.count = len(vectors)
            self._clusters = [c for c in self._clusters if c.count > 0]
            self._observations = observations

        if assignment:
            log.info("refinement moved %d utterances", len(assignment))
        return assignment

    # ------------------------------------------------------------------ queries

    def speakers(self) -> list[Speaker]:
        with self._lock:
            return [
                Speaker(
                    speaker_id=c.speaker_id,
                    display_name=c.display_name,
                    track=TrackKind.LOOPBACK,
                    is_local=False,
                    utterance_count=c.count,
                )
                for c in self._clusters
            ]

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "speakers": len(self._clusters),
                "observations": len(self._observations),
                "dropped_observations": self._dropped_observations,
                "failures": self.failures,
                "threshold": self.params.threshold,
            }

    def close(self) -> None:
        self.embedder.close()


class NullDiarizer:
    """First-class degraded mode: everything remote is simply "Remoto".

    Used when diarization is switched off, when the model is missing, or when the
    embedder failed to load. The meeting must keep recording -- a transcript with
    imperfect attribution is worth vastly more than no transcript.
    """

    __slots__ = ("_speaker",)

    def __init__(self, display_name: str = "Remoto") -> None:
        self._speaker = Speaker(
            speaker_id=UNKNOWN_SPEAKER_ID,
            display_name=display_name,
            track=TrackKind.LOOPBACK,
        )

    @property
    def enabled(self) -> bool:
        return False

    @property
    def last_speaker_id(self) -> str:
        return self._speaker.speaker_id

    def assign(self, utterance: Utterance) -> str:
        return self._speaker.speaker_id

    def refine(self) -> dict[tuple[str, int], str]:
        return {}

    def speakers(self) -> list[Speaker]:
        return [self._speaker]

    def stats(self) -> dict[str, int | float]:
        return {"speakers": 1, "observations": 0, "failures": 0, "threshold": 0.0}

    def close(self) -> None:
        return None
