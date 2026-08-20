"""The structured output contract for meeting minutes.

**Structured, not free text.** A prose summary cannot be tested, cannot be rendered
as a checklist, and gives the model room to blur "we decided X" into "we discussed
X". A JSON schema enforced by the API turns the summary into data: action items
have an owner field that is either a name or explicitly ``null``, decisions are a
list you can count, and a regression test can assert on the shape rather than
grepping paragraphs.

**Field names are English, content is Spanish.** The schema is part of the codebase
and follows its convention; the descriptions inside it are written in Spanish and
tell the model what each field must contain, which is where the language of the
output is actually decided.

The same schema is used by both adapters, so switching between Claude and a local
Ollama model changes the quality of the minutes but never their shape.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from meetscribe.core.models import ActionItem, Summary
from meetscribe.errors import SummarizerError

__all__ = [
    "SUMMARY_SCHEMA",
    "ACTION_ITEMS_SCHEMA",
    "parse_summary",
    "parse_action_items",
    "merge_summaries",
]


_ACTION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "owner", "due"],
    "properties": {
        "text": {
            "type": "string",
            "description": (
                "El compromiso concreto, en infinitivo o presente, tal y como se "
                "acordó. Sin adornos ni interpretación."
            ),
        },
        "owner": {
            "type": ["string", "null"],
            "description": (
                "Quién se comprometió, usando la etiqueta del hablante tal cual "
                "aparece en la transcripción. null si nadie lo asumió de forma "
                "explícita. NO adivines."
            ),
        },
        "due": {
            "type": ["string", "null"],
            "description": (
                "Plazo mencionado, literal ('el viernes', 'antes de fin de mes'). "
                "null si no se mencionó ninguno."
            ),
        },
    },
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "executive",
        "topics",
        "decisions",
        "action_items",
        "open_questions",
        "follow_ups",
    ],
    "properties": {
        "executive": {
            "type": "string",
            "description": (
                "Resumen ejecutivo en español, de 3 a 6 frases. Qué se trató, a qué "
                "se llegó y qué queda pendiente. Sin listas ni títulos."
            ),
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Temas tratados, uno por elemento, en el orden en que aparecieron.",
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Decisiones tomadas. Solo lo que quedó decidido de verdad; si algo "
                "se discutió sin cerrarse, va en preguntas abiertas."
            ),
        },
        "action_items": {
            "type": "array",
            "items": _ACTION_ITEM_SCHEMA,
            "description": "Compromisos concretos asumidos durante la reunión.",
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Preguntas que quedaron sin respuesta o temas sin cerrar.",
        },
        "follow_ups": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Seguimientos sugeridos: próxima reunión, personas a consultar.",
        },
    },
}

ACTION_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action_items"],
    "properties": {
        "action_items": {
            "type": "array",
            "items": _ACTION_ITEM_SCHEMA,
            "description": (
                "Solo compromisos NUEVOS y explícitos en este fragmento. Si no hay "
                "ninguno, devuelve una lista vacía; no inventes para rellenar."
            ),
        }
    },
}


def _as_text_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v).strip() for v in value if str(v).strip())


def _as_action_items(value: Any, at_s: float | None = None) -> tuple[ActionItem, ...]:
    if not isinstance(value, list):
        return ()
    items: list[ActionItem] = []
    for raw in value:
        if isinstance(raw, str):
            text = raw.strip()
            if text:
                items.append(ActionItem(text=text, at_s=at_s))
            continue
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        owner = raw.get("owner")
        due = raw.get("due")
        items.append(
            ActionItem(
                text=text,
                owner=str(owner).strip() if owner else None,
                due=str(due).strip() if due else None,
                at_s=at_s,
            )
        )
    return tuple(items)


def parse_summary(
    payload: str | dict[str, Any],
    *,
    session_id: str,
    provider: str,
    model: str,
    generated_at: dt.datetime | None = None,
    partial: bool = False,
) -> Summary:
    """Turn a model response into a :class:`Summary`, or fail loudly.

    Tolerant about *shape* -- a model that returns a bare string where a list was
    asked for should not lose the whole meeting -- and strict about *validity*: text
    that is not JSON at all is an error, because silently substituting an empty
    summary would look identical to a meeting with nothing in it.
    """
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SummarizerError(
                f"model did not return JSON: {exc}; got {text[:200]!r}",
                user_message=(
                    "El modelo devolvió un resumen con formato inválido. La "
                    "transcripción está intacta; reintenta con "
                    "'meetscribe summarize'."
                ),
            ) from exc
    else:
        data = payload

    if not isinstance(data, dict):
        raise SummarizerError(
            f"expected a JSON object, got {type(data).__name__}",
            user_message="El modelo devolvió un resumen con una estructura inesperada.",
        )

    return Summary(
        session_id=session_id,
        provider=provider,
        model=model,
        generated_at=generated_at or dt.datetime.now(dt.UTC),
        executive=str(data.get("executive") or "").strip(),
        topics=_as_text_list(data.get("topics")),
        decisions=_as_text_list(data.get("decisions")),
        action_items=_as_action_items(data.get("action_items")),
        open_questions=_as_text_list(data.get("open_questions")),
        follow_ups=_as_text_list(data.get("follow_ups")),
        partial=partial,
    )


def parse_action_items(
    payload: str | dict[str, Any], at_s: float | None = None
) -> tuple[ActionItem, ...]:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ()
    else:
        data = payload
    if not isinstance(data, dict):
        return ()
    return _as_action_items(data.get("action_items"), at_s=at_s)


def merge_summaries(parts: list[Summary], final: Summary) -> Summary:
    """Fold the map phase's action items into the reduce phase's result.

    The reduce pass sees compressed notes, so it can lose a commitment that a chunk
    captured verbatim. Losing an action item is the worst failure this feature has,
    so the union is taken and de-duplicated on the fingerprint rather than trusting
    the final pass alone.
    """
    import dataclasses

    seen = {item.fingerprint for item in final.action_items}
    merged = list(final.action_items)
    for part in parts:
        for item in part.action_items:
            if item.fingerprint not in seen:
                seen.add(item.fingerprint)
                merged.append(item)
    return dataclasses.replace(final, action_items=tuple(merged))
