"""Segmentation and the hallucination defences.

The segmenter tests pin down the arithmetic that maps the VAD's internal sample
indices onto session time. Getting that wrong does not crash anything -- it
produces a transcript whose timestamps drift a little further out with every
dropped chunk, which is exactly the kind of bug nobody notices until the two
tracks are visibly out of order.
"""

from __future__ import annotations

import numpy as np
import pytest

from meetscribe.core.models import PIPELINE_SAMPLE_RATE, TrackKind, TranscriptionResult
from meetscribe.errors import ConfigError
from meetscribe.stt.hallucination import (
    HallucinationFilter,
    compression_ratio,
)
from meetscribe.stt.transcriber import (
    REJECTED_MODELS,
    SUPPORTED_MODELS,
    FakeTranscriber,
    FasterWhisperTranscriber,
)
from meetscribe.stt.vad import WINDOW, VadParams, VadSegmenter

from .conftest import ScriptedVad, chunk, tone

SR = PIPELINE_SAMPLE_RATE


def make_segmenter(script=None, **kwargs) -> tuple[VadSegmenter, ScriptedVad]:
    engine = ScriptedVad(script or {})
    segmenter = VadSegmenter(TrackKind.LOOPBACK, engine, **kwargs)
    return segmenter, engine


def audio(windows: int) -> np.ndarray:
    return tone(windows * WINDOW / SR)


# ---------------------------------------------------------------- segmentation


def test_segment_times_come_from_the_vad_sample_index():
    segmenter, _ = make_segmenter({5: ("emit", 512, 1024)})
    out = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    assert len(out) == 1
    utterance = out[0]
    assert utterance.start_s == pytest.approx(512 / SR)
    assert utterance.end_s == pytest.approx((512 + 1024) / SR)
    assert utterance.seq == 512
    assert not utterance.is_partial


def test_first_chunk_offset_shifts_every_later_timestamp():
    """A track that starts three seconds late must report three-second offsets."""
    segmenter, _ = make_segmenter({5: ("emit", 512, 1024)})
    out = segmenter.feed(chunk(TrackKind.LOOPBACK, 3.0, audio(10)))
    assert out[0].start_s == pytest.approx(3.0 + 512 / SR)


def test_a_gap_is_padded_with_silence_so_alignment_survives():
    """Skipping a dropped second would shift every later timestamp earlier."""
    segmenter, engine = make_segmenter()
    segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))     # 0.00 .. 0.32
    windows_before = engine.windows

    # Next chunk starts 0.40 s later than the previous one ended.
    engine.script = {windows_before + 30: ("emit", 8000, 1600)}
    out = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.32 + 0.40, audio(40)))

    assert segmenter.padded_samples == round(0.40 * SR)
    assert len(out) == 1
    # The mapping is still index/rate because the padding kept it linear.
    assert out[0].start_s == pytest.approx(8000 / SR)


def test_a_gap_too_large_to_pad_resyncs_and_bumps_the_sequence():
    segmenter, engine = make_segmenter(max_pad_s=1.0)
    segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    assert engine.reset_count == 0

    engine.script = {2: ("emit", 512, 512)}
    out = segmenter.feed(chunk(TrackKind.LOOPBACK, 60.0, audio(10)))

    assert engine.reset_count == 1
    assert segmenter.resyncs == 1
    assert out, "the resynced segment should still be emitted"
    # Timestamps restart from the new chunk's own start time...
    assert out[0].start_s == pytest.approx(60.0 + 512 / SR)
    # ...and the sequence never goes backwards, so old lines are not overwritten.
    assert out[0].seq > 512


def test_draft_and_final_share_a_sequence_so_the_panel_replaces_in_place():
    segmenter, engine = make_segmenter(interim_after_s=0.2, interim_every_s=0.05)
    engine.script = {4: ("speak", 0, int(0.5 * SR))}
    drafts = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    assert len(drafts) == 1
    assert drafts[0].is_partial
    assert drafts[0].seq == 0

    engine.script = {12: ("emit", 0, int(1.2 * SR))}
    finals = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.32, audio(10)))
    finals = [u for u in finals if not u.is_partial]
    assert len(finals) == 1
    assert finals[0].seq == drafts[0].seq
    assert finals[0].duration_s == pytest.approx(1.2)


def test_no_draft_before_the_interim_threshold():
    segmenter, engine = make_segmenter(interim_after_s=5.0)
    engine.script = {4: ("speak", 0, int(1.0 * SR))}
    out = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    assert out == []


def test_drafts_are_rate_limited():
    segmenter, engine = make_segmenter(interim_after_s=0.1, interim_every_s=1.0)
    engine.script = {4: ("speak", 0, int(0.5 * SR))}
    first = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    assert len(first) == 1
    # Same in-progress segment, only slightly longer: not worth another decode.
    engine.script = {14: ("speak", 0, int(0.6 * SR))}
    second = segmenter.feed(chunk(TrackKind.LOOPBACK, 0.32, audio(10)))
    assert second == []


def test_speech_state_changes_are_reported_once_per_edge():
    edges: list[tuple[bool, float]] = []
    segmenter, engine = make_segmenter(on_speech_change=lambda s, t: edges.append((s, t)))
    engine.script = {4: ("speak", 0, 512)}
    segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    engine.script = {14: ("emit", 0, 1024)}
    segmenter.feed(chunk(TrackKind.LOOPBACK, 0.32, audio(10)))
    assert [e[0] for e in edges] == [True, False]


def test_flush_drains_the_engine_and_closes_the_speech_state():
    segmenter, engine = make_segmenter()
    engine.script = {4: ("speak", 0, 4096)}
    segmenter.feed(chunk(TrackKind.LOOPBACK, 0.0, audio(10)))
    out = segmenter.flush()
    assert engine.flushed
    assert len(out) == 1
    assert not out[0].is_partial


def test_segmenter_rejects_a_chunk_from_the_wrong_track():
    segmenter, _ = make_segmenter()
    with pytest.raises(ValueError, match="mic"):
        segmenter.feed(chunk(TrackKind.MIC, 0.0, audio(2)))


def test_vad_params_defaults_are_the_documented_ones():
    params = VadParams()
    assert params.threshold == 0.5
    assert params.min_silence_duration == pytest.approx(0.45)
    assert params.max_speech_duration == 20.0


# -------------------------------------------------------------- hallucinations


def result(text: str, **kw) -> TranscriptionResult:
    defaults = dict(no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=1.5)
    defaults.update(kw)
    return TranscriptionResult(text=text, language="es", **defaults)


@pytest.mark.parametrize(
    "text",
    [
        "Subtítulos realizados por la comunidad de Amara.org",
        "subtitulos realizados por la comunidad de amara.org",
        "¡Gracias por ver el video!",
        "Subtitles by the Amara.org community",
        "No olvides suscribirte al canal",
        "...",
    ],
)
def test_known_hallucinations_are_rejected(text):
    verdict = HallucinationFilter().check(result(text))
    assert verdict.rejected


def test_real_speech_survives_every_filter():
    verdict = HallucinationFilter().check(
        result("Vamos a revisar el avance del proyecto de facturación electrónica.")
    )
    assert verdict.accepted
    assert verdict.text.startswith("Vamos a revisar")


def test_no_speech_probability_rejects():
    assert HallucinationFilter().check(result("algo", no_speech_prob=0.95)).rejected


def test_low_confidence_rejects():
    assert HallucinationFilter().check(result("algo", avg_logprob=-2.5)).rejected


def test_repetition_loop_is_caught_by_compression_ratio():
    looped = "y entonces vamos a ver. " * 40
    assert compression_ratio(looped) > 2.6
    assert HallucinationFilter().check(result(looped, compression_ratio=None)).rejected


def test_ngram_loop_is_caught_even_when_it_compresses_poorly():
    text = " ".join(f"bueno pues entonces claro {i}" for i in range(8))
    verdict = HallucinationFilter().check(result(text, compression_ratio=1.2))
    assert verdict.rejected
    assert verdict.reason == "ngram_loop"


def test_short_natural_repetition_is_not_a_loop():
    verdict = HallucinationFilter().check(result("Sí, sí, sí, claro.", compression_ratio=1.1))
    assert verdict.accepted


def test_trailing_repeated_sentences_are_trimmed_not_discarded():
    text = "El informe está listo. Lo reviso mañana. Lo reviso mañana. Lo reviso mañana."
    verdict = HallucinationFilter().check(result(text, compression_ratio=1.4))
    assert verdict.accepted
    assert verdict.text.count("Lo reviso") == 1
    assert "El informe" in verdict.text


def test_consecutive_identical_utterances_are_suppressed_after_a_few():
    filt = HallucinationFilter(max_repeats=3)
    for _ in range(2):
        assert filt.check(result("de acuerdo")).accepted
    filt.check(result("de acuerdo"))
    assert filt.check(result("de acuerdo")).rejected
    assert filt.rejections["consecutive_repeat"] >= 1


def test_rejection_reasons_are_counted_for_doctor():
    filt = HallucinationFilter()
    filt.check(result("", no_speech_prob=0.01))
    filt.check(result("hola", no_speech_prob=0.99))
    assert filt.rejections == {"empty": 1, "no_speech_prob": 1}


def test_filter_reset_clears_repeat_state():
    filt = HallucinationFilter(max_repeats=2)
    filt.check(result("uno"))
    filt.reset()
    assert filt.check(result("uno")).accepted


# ------------------------------------------------------------------ transcriber


def test_medium_is_refused_with_the_measured_reason():
    with pytest.raises(ConfigError) as excinfo:
        FasterWhisperTranscriber("medium")
    assert "dominado" in excinfo.value.user_message
    assert "medium" in REJECTED_MODELS


def test_unknown_model_is_refused_with_the_valid_options():
    with pytest.raises(ConfigError) as excinfo:
        FasterWhisperTranscriber("enorme")
    for name in SUPPORTED_MODELS:
        assert name in excinfo.value.user_message


def test_supported_models_are_ordered_and_exclude_medium():
    assert SUPPORTED_MODELS == ("tiny", "base", "small", "large-v3-turbo")


def test_language_selection_pins_short_utterances_to_the_primary_language():
    transcriber = FasterWhisperTranscriber("small", language="auto", auto_detect_min_s=2.0)
    assert transcriber._language_for("auto", 1.0) == "es"
    assert transcriber._language_for("auto", 5.0) is None
    assert transcriber._language_for("es", 5.0) == "es"


def test_fake_transcriber_records_what_reached_the_decoder():
    fake = FakeTranscriber(script=["hola", "adiós"])
    assert fake.transcribe(tone(1.0), language="es").text == "hola"
    assert fake.transcribe(tone(2.0)).text == "adiós"
    assert fake.transcribe(tone(0.5)).text == "texto de prueba"
    assert [c[0] for c in fake.calls] == [SR, 2 * SR, SR // 2]
