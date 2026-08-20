"""Transcript and minutes export.

Four formats, each for a different job:

* **md** -- the one people actually share. Header, minutes, then the transcript.
* **txt** -- plain, for pasting anywhere.
* **json** -- the complete record including quality metrics, for re-processing.
* **srt** -- subtitles, for lining the transcript up against a recording.

Every format states the consent mode and whether the audio was retained. A
transcript that circulates without saying how it was obtained is exactly the
artifact this project's consent design is trying to avoid producing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from meetscribe.core.models import (
    ActionItem,
    Session,
    Speaker,
    Summary,
    TranscriptSegment,
)
from meetscribe.security.consent import describe_consent
from meetscribe.summarize.prompts import format_timestamp

__all__ = ["FORMATS", "render", "render_markdown", "render_text", "render_json", "render_srt"]

FORMATS = ("md", "txt", "json", "srt")


def render(
    fmt: str,
    session: Session,
    segments: Sequence[TranscriptSegment],
    speakers: Sequence[Speaker] = (),
    summary: Summary | None = None,
    action_items: Sequence[ActionItem] = (),
) -> str:
    renderers = {
        "md": render_markdown,
        "txt": render_text,
        "json": render_json,
        "srt": render_srt,
    }
    if fmt not in renderers:
        raise ValueError(f"formato desconocido: {fmt}. Opciones: {', '.join(FORMATS)}")
    return renderers[fmt](session, segments, speakers, summary, action_items)


def _duration(segments: Sequence[TranscriptSegment], session: Session) -> float:
    if segments:
        return segments[-1].end_s
    return session.duration_s or 0.0


def render_markdown(
    session: Session,
    segments: Sequence[TranscriptSegment],
    speakers: Sequence[Speaker] = (),
    summary: Summary | None = None,
    action_items: Sequence[ActionItem] = (),
) -> str:
    lines = [
        f"# {session.name}",
        "",
        f"- **Fecha:** {session.started_at.astimezone().strftime('%Y-%m-%d %H:%M')}",
        f"- **Duración:** {format_timestamp(_duration(segments, session))}",
        f"- **Idioma:** {session.language}",
        f"- **Modelo de transcripción:** {session.stt_model or 'desconocido'}",
        f"- **Consentimiento:** {describe_consent(session.consent)}"
        + (f" ({session.consent_note})" if session.consent_note else ""),
        f"- **Audio conservado:** {'sí' if session.audio_retained else 'no'}",
        f"- **Modo solo-local:** {'sí' if session.offline_only else 'no'}",
        "",
    ]

    if speakers:
        lines += ["## Participantes", ""]
        lines += [
            f"- {s.display_name}"
            + (" (tú)" if s.is_local else "")
            + (f" — {s.utterance_count} intervenciones" if s.utterance_count else "")
            for s in speakers
        ]
        lines.append("")

    if summary is not None:
        lines += ["## Resumen ejecutivo", "", summary.executive, ""]
        for title, items in (
            ("Temas tratados", summary.topics),
            ("Decisiones", summary.decisions),
            ("Preguntas abiertas", summary.open_questions),
            ("Seguimiento", summary.follow_ups),
        ):
            if items:
                lines += [f"## {title}", ""]
                lines += [f"- {item}" for item in items]
                lines.append("")

    items = list(summary.action_items) if summary else list(action_items)
    if items:
        lines += ["## Compromisos", ""]
        for item in items:
            owner = f"**{item.owner}**: " if item.owner else ""
            due = f" _(_{item.due}_)_" if item.due else ""
            lines.append(f"- [ ] {owner}{item.text}{due}")
        lines.append("")

    lines += ["## Transcripción", ""]
    for segment in segments:
        lines.append(
            f"**[{format_timestamp(segment.start_s)}] {segment.speaker_name}:** "
            f"{segment.text}"
        )
        lines.append("")

    if summary is not None and summary.partial:
        lines += [
            "---",
            "",
            (
                "> El resumen se generó por fragmentos: ninguna pasada individual "
                "vio la reunión completa."
            ),
            "",
        ]

    lines += [
        "---",
        "",
        (
            "_Generado localmente por MeetScribe. Resumen: "
            f"{summary.provider + '/' + summary.model if summary else 'no generado'}._"
        ),
    ]
    return "\n".join(lines)


def render_text(
    session: Session,
    segments: Sequence[TranscriptSegment],
    speakers: Sequence[Speaker] = (),
    summary: Summary | None = None,
    action_items: Sequence[ActionItem] = (),
) -> str:
    lines = [
        session.name,
        "=" * len(session.name),
        (
            f"{session.started_at.astimezone().strftime('%Y-%m-%d %H:%M')} · "
            f"{format_timestamp(_duration(segments, session))} · "
            f"consentimiento: {describe_consent(session.consent)}"
        ),
        "",
    ]
    if summary is not None:
        lines += ["RESUMEN", "-------", summary.executive, ""]
        if summary.action_items:
            lines.append("COMPROMISOS")
            lines.append("-----------")
            for item in summary.action_items:
                owner = f"{item.owner}: " if item.owner else ""
                due = f" ({item.due})" if item.due else ""
                lines.append(f"  - {owner}{item.text}{due}")
            lines.append("")
    lines += ["TRANSCRIPCIÓN", "-------------"]
    for segment in segments:
        lines.append(
            f"[{format_timestamp(segment.start_s)}] {segment.speaker_name}: {segment.text}"
        )
    return "\n".join(lines)


def render_json(
    session: Session,
    segments: Sequence[TranscriptSegment],
    speakers: Sequence[Speaker] = (),
    summary: Summary | None = None,
    action_items: Sequence[ActionItem] = (),
) -> str:
    payload = {
        "session": {
            "session_id": session.session_id,
            "name": session.name,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_s": _duration(segments, session),
            "language": session.language,
            "state": session.state.value,
            "consent": session.consent.value,
            "consent_note": session.consent_note,
            "offline_only": session.offline_only,
            "diarization_enabled": session.diarization_enabled,
            "stt_model": session.stt_model,
            "audio_retained": session.audio_retained,
        },
        "speakers": [
            {
                "speaker_id": s.speaker_id,
                "display_name": s.display_name,
                "track": s.track.value,
                "is_local": s.is_local,
                "utterance_count": s.utterance_count,
            }
            for s in speakers
        ],
        "segments": [
            {
                "segment_id": s.segment_id,
                "track": s.track.value,
                "seq": s.seq,
                "start_s": round(s.start_s, 3),
                "end_s": round(s.end_s, 3),
                "speaker_id": s.speaker_id,
                "speaker_name": s.speaker_name,
                "text": s.text,
                "language": s.language,
                "no_speech_prob": s.no_speech_prob,
                "avg_logprob": s.avg_logprob,
            }
            for s in segments
        ],
        "summary": None
        if summary is None
        else {
            "provider": summary.provider,
            "model": summary.model,
            "generated_at": summary.generated_at.isoformat(),
            "partial": summary.partial,
            "executive": summary.executive,
            "topics": list(summary.topics),
            "decisions": list(summary.decisions),
            "open_questions": list(summary.open_questions),
            "follow_ups": list(summary.follow_ups),
            "action_items": [
                {"text": a.text, "owner": a.owner, "due": a.due, "at_s": a.at_s}
                for a in summary.action_items
            ],
        },
        "action_items": [
            {"text": a.text, "owner": a.owner, "due": a.due, "at_s": a.at_s}
            for a in action_items
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def render_srt(
    session: Session,
    segments: Sequence[TranscriptSegment],
    speakers: Sequence[Speaker] = (),
    summary: Summary | None = None,
    action_items: Sequence[ActionItem] = (),
) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        # Zero-length cues render as a flash or are dropped by some players; a
        # minimum duration keeps every line readable.
        end = max(segment.end_s, segment.start_s + 0.6)
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(segment.start_s)} --> {_srt_timestamp(end)}\n"
            f"{segment.speaker_name}: {segment.text}\n"
        )
    return "\n".join(blocks)
