"""Prompts, and the transcript rendering they operate on.

These are written in Spanish because the output is Spanish, and the instruction
that matters most is repeated in several forms: **do not invent**. An LLM asked to
summarise a meeting will, unprompted, promote a suggestion to a decision, assign an
owner to an unclaimed task, and smooth over the fact that half the transcript is
ambiguous. In minutes that people act on, a confidently wrong owner is worse than
an empty field.

The transcript is rendered with timestamps and speaker labels, and the prompt tells
the model to use those labels verbatim. That is what makes "Participante 2 se
se comprometio a X" traceable back to a line the user can read and, if the attribution
was wrong, fix with a rename.
"""

from __future__ import annotations

from collections.abc import Sequence

from meetscribe.core.models import Session, Speaker, TranscriptSegment

__all__ = [
    "SYSTEM_PROMPT_ES",
    "ACTION_ITEMS_SYSTEM_ES",
    "render_transcript",
    "build_summary_prompt",
    "build_chunk_prompt",
    "build_reduce_prompt",
    "build_action_items_prompt",
    "format_timestamp",
]


SYSTEM_PROMPT_ES = """\
Eres un asistente que redacta actas de reunión en español de España/Latinoamérica \
neutro, a partir de transcripciones automáticas.

Reglas que no puedes romper:

1. NO INVENTES. Si algo no está en la transcripción, no existe. No completes \
   nombres, fechas ni cifras por verosimilitud.
2. Distingue "se discutió" de "se decidió". Una decisión solo lo es si alguien la \
   cierra de forma explícita. Lo demás va en preguntas abiertas.
3. Un compromiso solo tiene responsable si alguien lo asume en voz alta. Si nadie \
   lo hace, el campo va a null. Es correcto y útil que vaya a null.
4. Usa las etiquetas de hablante tal y como aparecen ("Yo", "Participante 2", o el \
   nombre si ya fue renombrado). No inventes nombres propios.
5. La transcripción es automática y contiene errores. Si un fragmento es \
   incomprensible, ignóralo; no trates de reconstruirlo.
6. Sé concreto y breve. Un acta que repite la reunión entera no sirve para nada.
7. Responde SIEMPRE en español, y únicamente con el JSON pedido."""


ACTION_ITEMS_SYSTEM_ES = """\
Extraes compromisos de un fragmento de reunión en curso, en español.

Un compromiso es una acción concreta que alguien va a hacer después de la reunión. \
No lo son: opiniones, descripciones de lo ya hecho, ni ideas sin asumir.

Si el fragmento no contiene ningún compromiso nuevo y explícito, devuelve una lista \
vacía. Es la respuesta correcta la mayoría de las veces. No rellenes."""


def format_timestamp(seconds: float) -> str:
    """``mm:ss`` for short meetings, ``h:mm:ss`` past an hour."""
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_transcript(
    segments: Sequence[TranscriptSegment],
    *,
    with_timestamps: bool = True,
    redact=None,
) -> str:
    """One line per intervention, merging consecutive turns by the same speaker.

    Merging matters for cost and for quality: VAD segmentation splits a single
    thought into several utterances at every breath, and a transcript that reads
    "Participante 1: Bueno. / Participante 1: entonces lo que yo veo..." wastes
    tokens and makes the model treat one argument as several.
    """
    lines: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []
    start = 0.0

    def flush() -> None:
        if not buffer or current_speaker is None:
            return
        text = " ".join(buffer).strip()
        if redact is not None:
            text = redact(text)
        stamp = f"[{format_timestamp(start)}] " if with_timestamps else ""
        lines.append(f"{stamp}{current_speaker}: {text}")

    for segment in segments:
        if segment.speaker_name != current_speaker:
            flush()
            current_speaker = segment.speaker_name
            buffer = []
            start = segment.start_s
        buffer.append(segment.text.strip())
    flush()
    return "\n".join(lines)


def _header(session: Session, speakers: Sequence[Speaker], duration_s: float) -> str:
    names = ", ".join(s.display_name for s in speakers) or "desconocidos"
    return (
        f"Reunión: {session.name}\n"
        f"Fecha: {session.started_at.astimezone().strftime('%Y-%m-%d %H:%M')}\n"
        f"Duración: {format_timestamp(duration_s)}\n"
        f"Participantes detectados: {names}\n"
    )


def build_summary_prompt(
    session: Session,
    segments: Sequence[TranscriptSegment],
    speakers: Sequence[Speaker],
    transcript: str,
) -> str:
    duration = segments[-1].end_s if segments else 0.0
    return (
        f"{_header(session, speakers, duration)}\n"
        "Transcripción completa:\n"
        "-----\n"
        f"{transcript}\n"
        "-----\n\n"
        "Redacta el acta de esta reunión siguiendo el esquema JSON solicitado."
    )


def build_chunk_prompt(
    session: Session,
    transcript: str,
    index: int,
    total: int,
    previous_notes: str = "",
) -> str:
    """Map phase. Each chunk is summarised knowing what came before it.

    The running context is what keeps a map-reduce from producing minutes where the
    same decision is reported three times because three chunks each saw part of it.
    """
    context = (
        f"\nResumen de lo tratado hasta ahora (fragmentos anteriores):\n"
        f"{previous_notes}\n"
        if previous_notes
        else ""
    )
    return (
        f"Reunión: {session.name}\n"
        f"Fragmento {index} de {total} de una reunión larga.{context}\n"
        "Fragmento actual:\n"
        "-----\n"
        f"{transcript}\n"
        "-----\n\n"
        "Resume ÚNICAMENTE este fragmento con el esquema JSON solicitado. No "
        "repitas lo que ya aparece en el resumen de fragmentos anteriores."
    )


def build_reduce_prompt(
    session: Session, speakers: Sequence[Speaker], partials: str, duration_s: float
) -> str:
    """Reduce phase: consolidate the per-chunk notes into one coherent acta."""
    return (
        f"{_header(session, speakers, duration_s)}\n"
        "Esta reunión se resumió por fragmentos. A continuación están las notas de "
        "cada fragmento, en orden:\n"
        "-----\n"
        f"{partials}\n"
        "-----\n\n"
        "Consolida todo en UN acta única siguiendo el esquema JSON solicitado. "
        "Elimina duplicados, ordena por importancia y conserva TODOS los "
        "compromisos: perder uno es el peor error posible aquí."
    )


def build_action_items_prompt(transcript: str) -> str:
    return (
        "Fragmento reciente de una reunión en curso:\n"
        "-----\n"
        f"{transcript}\n"
        "-----\n\n"
        "Extrae solo los compromisos nuevos y explícitos, con el esquema JSON "
        "solicitado."
    )
