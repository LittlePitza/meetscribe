"""Online diarization: cluster creation, label stability, and refinement.

Run against a fake embedder with hand-built vectors whose pairwise cosines are
known exactly (0.95 within a speaker, 0.40 between speakers -- the same shape as
the measured TitaNet distribution in ``docs/benchmarks.md``). That isolates the
clustering decisions, which is where the interesting behaviour is; the real
embedder is exercised separately by the spikes.
"""

from __future__ import annotations

import numpy as np
import pytest

from meetscribe.core.models import UNKNOWN_SPEAKER_ID, TrackKind, Utterance
from meetscribe.diarize.embedder import FakeEmbedder, l2_normalise
from meetscribe.diarize.online import DiarizerParams, NullDiarizer, OnlineDiarizer

DIM = 6


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[: len(values)] = values
    return l2_normalise(vector)


# Cosine similarities: A~A2 = 0.95, A~B = 0.40, A~C = 0.0
VOICES = {
    1: unit(1.0, 0.0, 0.0),
    2: unit(0.95, 0.312, 0.0),      # same person, different utterance
    3: unit(0.40, 0.917, 0.0),      # a different person
    4: unit(0.0, 0.0, 1.0),         # a third person
}


def speech(voice: int, duration: float = 2.0) -> Utterance:
    """Audio whose first sample encodes which voice produced it."""
    samples = np.full(int(duration * 16000), voice / 100.0, dtype=np.float32)
    return Utterance(
        track=TrackKind.LOOPBACK,
        samples=samples,
        start_s=0.0,
        end_s=duration,
        seq=speech.counter,
        is_partial=False,
    )


speech.counter = 0


def next_utterance(voice: int, duration: float = 2.0) -> Utterance:
    speech.counter += 1
    return speech(voice, duration)


def embedder() -> FakeEmbedder:
    def rule(samples: np.ndarray) -> np.ndarray:
        key = round(float(samples[0]) * 100)
        return VOICES[key]

    return FakeEmbedder(dim=DIM, rule=rule)


@pytest.fixture(autouse=True)
def reset_counter():
    speech.counter = 0


# ------------------------------------------------------------------ clustering


def test_first_speaker_opens_a_cluster():
    diarizer = OnlineDiarizer(embedder())
    assert diarizer.assign(next_utterance(1)) == "p1"
    speakers = diarizer.speakers()
    assert [s.display_name for s in speakers] == ["Participante 1"]
    assert speakers[0].track is TrackKind.LOOPBACK


def test_same_voice_joins_the_existing_cluster():
    diarizer = OnlineDiarizer(embedder())
    first = diarizer.assign(next_utterance(1))
    again = diarizer.assign(next_utterance(2))  # cosine 0.95 against voice 1
    assert first == again
    assert len(diarizer.speakers()) == 1


def test_a_different_voice_opens_a_second_cluster():
    diarizer = OnlineDiarizer(embedder())
    diarizer.assign(next_utterance(1))
    second = diarizer.assign(next_utterance(3))  # cosine 0.40, below the 0.60 gate
    assert second == "p2"
    assert [s.display_name for s in diarizer.speakers()] == [
        "Participante 1",
        "Participante 2",
    ]


def test_labels_never_change_during_the_meeting():
    """'Participante 2' turning into 'Participante 5' mid-call destroys trust in
    the attribution, so ids are append-only while recording."""
    diarizer = OnlineDiarizer(embedder())
    ids = [
        diarizer.assign(next_utterance(voice))
        for voice in (1, 3, 4, 2, 3, 1, 4)
    ]
    assert ids == ["p1", "p2", "p3", "p1", "p2", "p1", "p3"]


def test_a_backchannel_is_attributed_without_touching_any_centroid():
    diarizer = OnlineDiarizer(embedder(), DiarizerParams(min_new_speaker_s=1.0))
    diarizer.assign(next_utterance(1))
    before = diarizer.speakers()[0].utterance_count
    # 0.5 s: long enough to embed, too short to be trusted with the voice model.
    assert diarizer.assign(next_utterance(3, duration=0.5)) == "p1"
    assert diarizer.speakers()[0].utterance_count == before
    assert len(diarizer.speakers()) == 1


def test_a_very_short_utterance_is_not_embedded_at_all():
    fake = embedder()
    diarizer = OnlineDiarizer(fake, DiarizerParams(min_embed_s=0.35))
    diarizer.assign(next_utterance(1))
    calls = fake.calls
    assert diarizer.assign(next_utterance(3, duration=0.2)) == "p1"
    assert fake.calls == calls  # the embedder was never called


def test_speaker_cap_stops_cluster_proliferation():
    diarizer = OnlineDiarizer(embedder(), DiarizerParams(max_speakers=2))
    diarizer.assign(next_utterance(1))
    diarizer.assign(next_utterance(3))
    third = diarizer.assign(next_utterance(4))
    assert third in ("p1", "p2")
    assert len(diarizer.speakers()) == 2


def test_an_embedder_failure_degrades_instead_of_raising():
    class Broken:
        dim = DIM

        def embed(self, samples, sample_rate):
            raise RuntimeError("onnxruntime exploded")

        def close(self):
            return None

    diarizer = OnlineDiarizer(Broken())
    assert diarizer.assign(next_utterance(1)) == UNKNOWN_SPEAKER_ID
    assert diarizer.failures == 1


def test_last_speaker_id_tracks_the_most_recent_decision():
    diarizer = OnlineDiarizer(embedder())
    diarizer.assign(next_utterance(1))
    assert diarizer.last_speaker_id == "p1"
    diarizer.assign(next_utterance(3))
    assert diarizer.last_speaker_id == "p2"


def test_stats_report_the_operating_point():
    diarizer = OnlineDiarizer(embedder())
    diarizer.assign(next_utterance(1))
    stats = diarizer.stats()
    assert stats["speakers"] == 1
    assert stats["observations"] == 1
    assert stats["threshold"] == 0.60


# ------------------------------------------------------------------ refinement


def test_refinement_merges_an_over_split_speaker_and_keeps_the_majority_label():
    """A high live threshold splits one voice in two; the offline pass, which can
    see every utterance at once, puts them back together."""
    diarizer = OnlineDiarizer(embedder(), DiarizerParams(threshold=0.99))
    ids = [diarizer.assign(next_utterance(v)) for v in (1, 1, 1, 2)]
    assert len(set(ids)) == 2, "the strict live threshold should over-split"

    diarizer.params = DiarizerParams(threshold=0.60)
    mapping = diarizer.refine()

    assert mapping, "refinement should move the minority utterances"
    # Everything ends up under a single label, and that label is the one the
    # majority of the utterances already carried.
    final = {mapping.get(("loopback", i), ids[i - 1]) for i in range(1, 5)}
    assert final == {"p1"}


def test_refinement_is_a_no_op_when_the_live_pass_was_already_right():
    diarizer = OnlineDiarizer(embedder())
    for voice in (1, 3, 1, 3, 1):
        diarizer.assign(next_utterance(voice))
    assert diarizer.refine() == {}


def test_refinement_needs_enough_observations_to_be_meaningful():
    diarizer = OnlineDiarizer(embedder())
    diarizer.assign(next_utterance(1))
    assert diarizer.refine() == {}


def test_observations_are_bounded_over_a_marathon_meeting():
    diarizer = OnlineDiarizer(embedder(), DiarizerParams(max_observations=5))
    for _ in range(12):
        diarizer.assign(next_utterance(1))
    assert diarizer.stats()["observations"] == 5
    assert diarizer.stats()["dropped_observations"] == 7


# ---------------------------------------------------------------- degraded mode


def test_null_diarizer_is_a_first_class_path():
    diarizer = NullDiarizer()
    assert not diarizer.enabled
    assert diarizer.assign(next_utterance(1)) == UNKNOWN_SPEAKER_ID
    assert diarizer.refine() == {}
    assert [s.display_name for s in diarizer.speakers()] == ["Remoto"]
    assert diarizer.last_speaker_id == UNKNOWN_SPEAKER_ID
    diarizer.close()


def test_l2_normalise_handles_a_zero_vector():
    assert np.allclose(l2_normalise(np.zeros(4)), np.zeros(4))
    assert float(np.linalg.norm(l2_normalise(np.array([3.0, 4.0])))) == pytest.approx(1.0)
