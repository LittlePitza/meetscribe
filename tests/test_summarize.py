"""Summarization: the contract, the long-meeting strategy, and the egress boundary.

No test here reaches the network. The base class is exercised through a fake
``complete``; the Ollama adapter through a stubbed ``httpx.Client``; the Claude
adapter's error translation directly. What is being verified is the part that can
silently go wrong -- a chunk boundary that swallows an action item, an offline-only
session that quietly sends text anyway.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import httpx
import pytest

from meetscribe.config import Config
from meetscribe.core.models import (
    ActionItem,
    SegmentState,
    Speaker,
    Summary,
    TrackKind,
    TranscriptSegment,
)
from meetscribe.core.ports import SummaryRequest
from meetscribe.errors import EgressBlockedError, SummarizerError, SummarizerUnavailableError
from meetscribe.summarize.base import NullSummarizer, StructuredSummarizer, chunk_lines
from meetscribe.summarize.factory import build_summarizer
from meetscribe.summarize.ollama_adapter import OllamaSummarizer
from meetscribe.summarize.prompts import format_timestamp, render_transcript
from meetscribe.summarize.schema import (
    ACTION_ITEMS_SCHEMA,
    SUMMARY_SCHEMA,
    merge_summaries,
    parse_action_items,
    parse_summary,
)

PAYLOAD = {
    "executive": "Se revisó el avance de facturación.",
    "topics": ["facturación", "banco"],
    "decisions": ["escalar el bloqueo de credenciales"],
    "action_items": [
        {"text": "hablar con el contacto comercial", "owner": "Yo", "due": "esta semana"},
        {"text": "subir el módulo", "owner": None, "due": None},
    ],
    "open_questions": ["¿quién firma el contrato?"],
    "follow_ups": ["revisar el lunes"],
}


def seg(seq: int, speaker: str, text: str, start: float) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"s{seq}",
        session_id="sess",
        track=TrackKind.LOOPBACK,
        start_s=start,
        end_s=start + 3.0,
        text=text,
        speaker_id=speaker.lower(),
        speaker_name=speaker,
        seq=seq,
        state=SegmentState.FINAL,
    )


class FakeSummarizer(StructuredSummarizer):
    provider = "fake"
    is_local = True

    def __init__(self, payloads=None, **kw):
        super().__init__("fake-1", **kw)
        self.payloads = list(payloads or [])
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    def complete(self, system, user, schema, max_tokens=8000):
        self.prompts.append(user)
        self.schemas.append(schema)
        if self.payloads:
            return self.payloads.pop(0)
        return json.dumps(PAYLOAD, ensure_ascii=False)

    def health(self):
        return "disponible (fake)"


def request_for(session, segments=None) -> SummaryRequest:
    segments = [
        seg(1, "Yo", "Buenos días, vamos a revisar el avance.", 0.0),
        seg(2, "Participante 1", "El módulo de validación ya está terminado.", 4.0),
        seg(3, "Yo", "Perfecto, yo hablo con el banco esta semana.", 8.0),
    ] if segments is None else segments
    return SummaryRequest(
        session=session,
        segments=segments,
        speakers=[
            Speaker("yo", "Yo", TrackKind.MIC, is_local=True),
            Speaker("p1", "Participante 1", TrackKind.LOOPBACK),
        ],
        now=dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC),
    )


# --------------------------------------------------------------------- parsing


def test_parse_summary_maps_every_field():
    summary = parse_summary(
        json.dumps(PAYLOAD), session_id="sess", provider="fake", model="m"
    )
    assert summary.executive.startswith("Se revisó")
    assert summary.topics == ("facturación", "banco")
    assert len(summary.action_items) == 2
    assert summary.action_items[0].owner == "Yo"
    assert summary.action_items[1].owner is None
    assert summary.open_questions == ("¿quién firma el contrato?",)


def test_parse_summary_tolerates_a_fenced_code_block():
    fenced = "```json\n" + json.dumps(PAYLOAD) + "\n```"
    assert parse_summary(fenced, session_id="s", provider="p", model="m").topics


def test_parse_summary_tolerates_a_wrong_shaped_field():
    """A model returning a string where a list was asked for must not lose the
    whole meeting."""
    broken = {**PAYLOAD, "topics": "no es una lista", "action_items": ["hacer algo"]}
    summary = parse_summary(broken, session_id="s", provider="p", model="m")
    assert summary.topics == ()
    assert summary.action_items[0].text == "hacer algo"


def test_parse_summary_refuses_text_that_is_not_json():
    """Substituting an empty summary would be indistinguishable from a meeting
    where nothing happened."""
    with pytest.raises(SummarizerError, match="did not return JSON"):
        parse_summary("lo siento, no puedo", session_id="s", provider="p", model="m")


def test_parse_summary_refuses_a_json_array():
    with pytest.raises(SummarizerError):
        parse_summary("[1,2,3]", session_id="s", provider="p", model="m")


def test_parse_action_items_is_forgiving_and_stamps_the_time():
    items = parse_action_items(
        json.dumps({"action_items": [{"text": "x", "owner": "Ana", "due": None}]}),
        at_s=12.5,
    )
    assert items[0].at_s == 12.5
    assert parse_action_items("no json") == ()
    assert parse_action_items({"nope": 1}) == ()


def test_merge_keeps_action_items_the_reduce_pass_dropped():
    """Losing a commitment is the worst failure this feature has, so the union is
    taken rather than trusting the final pass alone."""
    base = Summary(
        session_id="s",
        provider="p",
        model="m",
        generated_at=dt.datetime.now(dt.UTC),
        executive="final",
        action_items=(ActionItem("uno"),),
    )
    partial = dataclasses.replace(
        base, action_items=(ActionItem("uno"), ActionItem("Dos."), ActionItem("tres"))
    )
    merged = merge_summaries([partial], base)
    assert [a.text for a in merged.action_items] == ["uno", "Dos.", "tres"]


def test_schemas_are_strict_enough_for_structured_output():
    assert SUMMARY_SCHEMA["additionalProperties"] is False
    assert set(SUMMARY_SCHEMA["required"]) == set(SUMMARY_SCHEMA["properties"])
    item = SUMMARY_SCHEMA["properties"]["action_items"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"text", "owner", "due"}
    assert item["properties"]["owner"]["type"] == ["string", "null"]
    assert ACTION_ITEMS_SCHEMA["required"] == ["action_items"]


# -------------------------------------------------------------------- rendering


def test_transcript_merges_consecutive_turns_by_the_same_speaker():
    """VAD splits one thought at every breath; a transcript that repeats the
    speaker label five times wastes tokens and fragments the argument."""
    segments = [
        seg(1, "Yo", "Bueno,", 0.0),
        seg(2, "Yo", "entonces lo que veo es esto.", 3.0),
        seg(3, "Ana", "De acuerdo.", 6.0),
    ]
    rendered = render_transcript(segments)
    assert rendered.splitlines() == [
        "[00:00] Yo: Bueno, entonces lo que veo es esto.",
        "[00:06] Ana: De acuerdo.",
    ]


def test_transcript_applies_a_redactor_when_given_one():
    segments = [seg(1, "Yo", "escríbeme a ana@example.com", 0.0)]
    from meetscribe.security.redact import redact_pii

    assert "[correo]" in render_transcript(segments, redact=redact_pii)


def test_format_timestamp_grows_an_hours_field():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(3725) == "1:02:05"
    assert format_timestamp(-5) == "00:00"


# --------------------------------------------------------------------- chunking


def test_chunk_lines_never_splits_inside_an_intervention():
    transcript = "\n".join(f"[00:0{i}] Yo: línea {i}" for i in range(9))
    chunks = chunk_lines(transcript, max_chars=40)
    assert len(chunks) > 1
    assert all(line.startswith("[00:") for c in chunks for line in c.split("\n"))
    assert "\n".join(chunks) == transcript


def test_chunk_lines_rejects_a_nonsense_limit():
    with pytest.raises(ValueError):
        chunk_lines("x", 0)


def test_short_meetings_take_a_single_pass(session):
    summarizer = FakeSummarizer()
    summary = summarizer.summarize(request_for(session))
    assert len(summarizer.prompts) == 1
    assert not summary.partial
    assert summary.provider == "fake"
    assert summary.generated_at == dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC)


def test_long_meetings_map_reduce_and_say_so(session):
    summarizer = FakeSummarizer(max_chunk_chars=120)
    segments = [
        seg(i, "Yo" if i % 2 else "Ana", f"intervención número {i} con bastante texto", i * 3.0)
        for i in range(1, 15)
    ]
    summary = summarizer.summarize(request_for(session, segments))

    assert len(summarizer.prompts) > 2, "several map calls plus one reduce"
    assert "Consolida todo en UN acta" in summarizer.prompts[-1]
    assert summary.partial is True


def test_the_map_phase_carries_forward_what_earlier_chunks_said(session):
    summarizer = FakeSummarizer(max_chunk_chars=120)
    segments = [
        seg(i, "Yo" if i % 2 else "Ana", f"intervención número {i} con bastante texto", i * 3.0)
        for i in range(1, 15)
    ]
    summarizer.summarize(request_for(session, segments))
    assert "Resumen de lo tratado hasta ahora" in summarizer.prompts[1]


def test_summarizing_nothing_is_an_error_not_an_empty_acta(session):
    with pytest.raises(SummarizerError, match="nothing to summarize"):
        FakeSummarizer().summarize(request_for(session, []))


def test_live_action_item_detection_uses_the_narrow_schema(session):
    summarizer = FakeSummarizer(
        payloads=[json.dumps({"action_items": [{"text": "x", "owner": None, "due": None}]})]
    )
    items = summarizer.detect_action_items(request_for(session).segments)
    assert [i.text for i in items] == ["x"]
    assert summarizer.schemas[0] is ACTION_ITEMS_SCHEMA


# ------------------------------------------------------------- egress boundary


class RemoteFake(FakeSummarizer):
    provider = "remoto"
    is_local = False


def test_an_offline_session_refuses_a_remote_summarizer(session):
    summarizer = RemoteFake(offline_only=True)
    with pytest.raises(EgressBlockedError):
        summarizer.summarize(request_for(session))
    with pytest.raises(EgressBlockedError):
        summarizer.detect_action_items(request_for(session).segments)


def test_pii_is_redacted_before_leaving_the_machine(session):
    summarizer = RemoteFake(redact_pii_before_egress=True)
    segments = [seg(1, "Yo", "mándalo a ana@example.com o al 600 123 456", 0.0)]
    summarizer.summarize(request_for(session, segments))
    sent = summarizer.prompts[0]
    assert "ana@example.com" not in sent
    assert "[correo]" in sent


def test_a_local_summarizer_never_redacts(session):
    summarizer = FakeSummarizer(redact_pii_before_egress=True)
    segments = [seg(1, "Yo", "escribe a ana@example.com", 0.0)]
    summarizer.summarize(request_for(session, segments))
    assert "ana@example.com" in summarizer.prompts[0]


def test_null_summarizer_refuses_clearly(session):
    summarizer = NullSummarizer()
    assert summarizer.is_local
    assert summarizer.detect_action_items([]) == ()
    assert "desactivado" in summarizer.health()
    with pytest.raises(SummarizerError, match="disabled"):
        summarizer.summarize(request_for(session))


def test_factory_refuses_anthropic_in_an_offline_session():
    config = Config()
    config.summarizer.provider = "anthropic"
    with pytest.raises(EgressBlockedError):
        build_summarizer(config, offline_only=True)


def test_factory_builds_the_local_provider(monkeypatch):
    config = Config()
    config.summarizer.provider = "ollama"
    summarizer = build_summarizer(config, offline_only=True)
    assert summarizer.is_local
    assert summarizer.provider == "ollama"


def test_factory_builds_anthropic_with_a_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-value-1234567890")
    config = Config()
    summarizer = build_summarizer(config, offline_only=False)
    assert summarizer.provider == "anthropic"
    assert not summarizer.is_local
    assert summarizer.model == "claude-opus-5"


def test_factory_returns_the_null_summarizer_for_none():
    config = Config()
    config.summarizer.provider = "none"
    assert isinstance(build_summarizer(config), NullSummarizer)


# ----------------------------------------------------------------- ollama


class StubResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("POST", "http://x"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._payload


class StubClient:
    def __init__(self, response=None, error=None, **kw):
        self.response = response
        self.error = error
        self.posted: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        self.posted.append(json)
        if self.error:
            raise self.error
        return self.response

    def get(self, url):
        if self.error:
            raise self.error
        return self.response


def test_ollama_sends_the_schema_and_returns_the_content(monkeypatch, session):
    stub = StubClient(StubResponse({"message": {"content": json.dumps(PAYLOAD)}}))
    monkeypatch.setattr(httpx, "Client", lambda **kw: stub)
    summarizer = OllamaSummarizer(model="llama3.1:8b")
    summary = summarizer.summarize(request_for(session))
    assert summary.executive.startswith("Se revisó")
    assert stub.posted[0]["format"] is SUMMARY_SCHEMA
    assert stub.posted[0]["stream"] is False
    assert summarizer.is_local


def test_ollama_explains_a_missing_server(monkeypatch, session):
    stub = StubClient(error=httpx.ConnectError("refused"))
    monkeypatch.setattr(httpx, "Client", lambda **kw: stub)
    with pytest.raises(SummarizerUnavailableError) as excinfo:
        OllamaSummarizer().summarize(request_for(session))
    assert "ollama serve" in excinfo.value.user_message


def test_ollama_explains_a_model_that_was_never_pulled(monkeypatch, session):
    stub = StubClient(StubResponse({}, status=404))
    monkeypatch.setattr(httpx, "Client", lambda **kw: stub)
    with pytest.raises(SummarizerUnavailableError) as excinfo:
        OllamaSummarizer(model="mistral").summarize(request_for(session))
    assert "ollama pull mistral" in excinfo.value.user_message


def test_ollama_health_reports_a_missing_model(monkeypatch):
    stub = StubClient(StubResponse({"models": [{"name": "otro:7b"}]}))
    monkeypatch.setattr(httpx, "Client", lambda **kw: stub)
    status = OllamaSummarizer(model="llama3.1:8b").health()
    assert "no está descargado" in status
    assert "otro:7b" in status


def test_ollama_health_reports_a_down_server(monkeypatch):
    stub = StubClient(error=httpx.ConnectError("refused"))
    monkeypatch.setattr(httpx, "Client", lambda **kw: stub)
    assert "no disponible" in OllamaSummarizer().health()


# ---------------------------------------------------------------- claude errors


@pytest.mark.parametrize(
    "exc_name, expected",
    [
        ("AuthenticationError", "no es válida"),
        ("PermissionDeniedError", "permisos"),
        ("NotFoundError", "no existe"),
        ("APIConnectionError", "No hay conexión"),
    ],
)
def test_claude_errors_become_actionable_spanish(exc_name, expected):
    import anthropic

    from meetscribe.summarize.anthropic_adapter import AnthropicSummarizer

    response = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    if exc_name == "APIConnectionError":
        exc = anthropic.APIConnectionError(request=response.request)
    else:
        exc = getattr(anthropic, exc_name)(
            message="boom", response=response, body=None
        )
    translated = AnthropicSummarizer._translate(exc)
    assert expected in translated.user_message


def test_claude_adapter_declares_its_egress_and_default_model():
    from meetscribe.summarize.anthropic_adapter import DEFAULT_MODEL, AnthropicSummarizer

    summarizer = AnthropicSummarizer(api_key="sk-ant-x")
    assert not summarizer.is_local
    assert summarizer.model == DEFAULT_MODEL == "claude-opus-5"
