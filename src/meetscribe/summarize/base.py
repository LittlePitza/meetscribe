"""Shared summarizer machinery: chunking, map-reduce, and the egress boundary.

Both adapters -- Claude and Ollama -- differ only in how a prompt plus a JSON schema
becomes a JSON string. Everything above that line is identical and lives here, so
switching providers cannot change how long meetings are handled, how action items
are de-duplicated, or what gets redacted before leaving the machine.

**The long-meeting strategy.** A two-hour meeting is roughly 20 000 words, which
fits comfortably in a modern context window -- so why chunk at all? Because quality
degrades before context runs out: asked to produce six-sentence minutes from two
hours of transcript in one pass, models compress the middle and over-weight the
end. Map-reduce at ~40 000 characters per chunk gives each part of the meeting
equal attention, and the running-notes context passed forward keeps the chunks from
re-reporting the same decision. The cost is one extra call per chunk, and
``partial=True`` on the result records honestly that no single pass saw everything.

**The egress boundary is enforced here, once.** ``is_local`` is a property of the
adapter; PII redaction and the offline-only refusal are applied in this base class,
so a new adapter cannot forget them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from meetscribe.core.models import ActionItem, Summary, TranscriptSegment
from meetscribe.core.ports import SummaryRequest
from meetscribe.errors import EgressBlockedError, SummarizerError
from meetscribe.security.redact import redact_pii
from meetscribe.summarize import prompts
from meetscribe.summarize.schema import (
    ACTION_ITEMS_SCHEMA,
    SUMMARY_SCHEMA,
    merge_summaries,
    parse_action_items,
    parse_summary,
)

log = logging.getLogger(__name__)

__all__ = ["StructuredSummarizer", "NullSummarizer", "chunk_lines"]


def chunk_lines(transcript: str, max_chars: int) -> list[str]:
    """Split a rendered transcript on line boundaries, never mid-intervention.

    Splitting inside a speaker's turn would hand the model half a sentence and
    invite it to guess the rest -- the exact behaviour the prompts spend their
    length forbidding.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    lines = transcript.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        extra = len(line) + 1
        if current and size + extra > max_chars:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


class StructuredSummarizer:
    """Base class. Subclasses implement :meth:`complete` and the metadata."""

    provider = "base"
    is_local = True

    def __init__(
        self,
        model: str,
        *,
        max_chunk_chars: int = 40_000,
        redact_pii_before_egress: bool = False,
        offline_only: bool = False,
    ) -> None:
        self._model = model
        self.max_chunk_chars = max_chunk_chars
        self.redact_pii_before_egress = redact_pii_before_egress
        self.offline_only = offline_only

    # ---------------------------------------------------------------- interface

    @property
    def name(self) -> str:
        return self.provider

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> str:
        """Return the model's JSON string. Implemented by each adapter."""
        raise NotImplementedError

    def health(self) -> str:
        """Human-readable readiness, in Spanish, for ``meetscribe doctor``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers

    def _guard_egress(self) -> None:
        if self.offline_only and not self.is_local:
            raise EgressBlockedError(
                f"{self.provider} would send transcript text off the machine"
            )

    def _redactor(self) -> Callable[[str], str] | None:
        if self.is_local or not self.redact_pii_before_egress:
            return None
        return redact_pii

    # ------------------------------------------------------------------- public

    def summarize(self, request: SummaryRequest) -> Summary:
        self._guard_egress()
        segments = list(request.segments)
        if not segments:
            raise SummarizerError(
                "nothing to summarize",
                user_message="La sesión no tiene transcripción que resumir.",
            )

        transcript = prompts.render_transcript(segments, redact=self._redactor())
        duration = segments[-1].end_s

        if len(transcript) <= self.max_chunk_chars:
            payload = self.complete(
                prompts.SYSTEM_PROMPT_ES,
                prompts.build_summary_prompt(
                    request.session, segments, request.speakers, transcript
                ),
                SUMMARY_SCHEMA,
            )
            return parse_summary(
                payload,
                session_id=request.session.session_id,
                provider=self.provider,
                model=self.model,
                generated_at=request.now,
            )

        chunks = chunk_lines(transcript, self.max_chunk_chars)
        log.info("map-reduce summarization over %d chunks", len(chunks))
        partials: list[Summary] = []
        running_notes = ""
        for index, chunk in enumerate(chunks, start=1):
            payload = self.complete(
                prompts.SYSTEM_PROMPT_ES,
                prompts.build_chunk_prompt(
                    request.session, chunk, index, len(chunks), running_notes
                ),
                SUMMARY_SCHEMA,
            )
            partial = parse_summary(
                payload,
                session_id=request.session.session_id,
                provider=self.provider,
                model=self.model,
                generated_at=request.now,
                partial=True,
            )
            partials.append(partial)
            running_notes = self._condense(partials)

        final_payload = self.complete(
            prompts.SYSTEM_PROMPT_ES,
            prompts.build_reduce_prompt(
                request.session, request.speakers, self._render_partials(partials), duration
            ),
            SUMMARY_SCHEMA,
        )
        final = parse_summary(
            final_payload,
            session_id=request.session.session_id,
            provider=self.provider,
            model=self.model,
            generated_at=request.now,
            partial=True,
        )
        return merge_summaries(partials, final)

    def detect_action_items(
        self, segments: Sequence[TranscriptSegment], language: str = "es"
    ) -> Sequence[ActionItem]:
        """Live extraction over a recent slice of the meeting."""
        self._guard_egress()
        if not segments:
            return ()
        transcript = prompts.render_transcript(list(segments), redact=self._redactor())
        if not transcript.strip():
            return ()
        payload = self.complete(
            prompts.ACTION_ITEMS_SYSTEM_ES,
            prompts.build_action_items_prompt(transcript),
            ACTION_ITEMS_SCHEMA,
            max_tokens=2000,
        )
        return parse_action_items(payload, at_s=segments[-1].end_s)

    # ---------------------------------------------------------------- rendering

    @staticmethod
    def _condense(partials: Sequence[Summary]) -> str:
        """A compact running context for the next chunk. Deliberately short: it is
        there to prevent repetition, not to re-summarise the meeting."""
        lines: list[str] = []
        for summary in partials[-3:]:
            if summary.executive:
                lines.append(f"- {summary.executive}")
            lines.extend(f"  * decidido: {decision}" for decision in summary.decisions[:3])
        return "\n".join(lines)[:4000]

    @staticmethod
    def _render_partials(partials: Sequence[Summary]) -> str:
        blocks: list[str] = []
        for index, summary in enumerate(partials, start=1):
            parts = [f"[Fragmento {index}]", summary.executive]
            if summary.topics:
                parts.append("Temas: " + "; ".join(summary.topics))
            if summary.decisions:
                parts.append("Decisiones: " + "; ".join(summary.decisions))
            if summary.action_items:
                parts.append(
                    "Compromisos: "
                    + "; ".join(
                        f"{a.text}" + (f" ({a.owner})" if a.owner else "")
                        for a in summary.action_items
                    )
                )
            if summary.open_questions:
                parts.append("Abierto: " + "; ".join(summary.open_questions))
            blocks.append("\n".join(p for p in parts if p))
        return "\n\n".join(blocks)


class NullSummarizer(StructuredSummarizer):
    """Used when ``summarizer.provider`` is 'none'.

    A real object rather than ``None`` so the pipeline has one code path; it refuses
    clearly instead of pretending to work.
    """

    provider = "none"
    is_local = True

    def __init__(self) -> None:
        super().__init__(model="", max_chunk_chars=1)

    def summarize(self, request: SummaryRequest) -> Summary:
        raise SummarizerError(
            "summarization disabled",
            user_message=(
                "El resumen está desactivado (summarizer.provider: none). Solo se "
                "guardó la transcripción."
            ),
        )

    def detect_action_items(
        self, segments: Sequence[TranscriptSegment], language: str = "es"
    ) -> Sequence[ActionItem]:
        return ()

    def health(self) -> str:
        return "desactivado (summarizer.provider: none)"
