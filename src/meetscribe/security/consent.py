"""Consent, as a feature rather than a disclaimer.

Recording a conversation is regulated, and the rules differ by jurisdiction: some
require only one party to consent, many require all of them. A tool that records
private meetings and handles this with a paragraph in the README is pushing a legal
risk onto the user while making the unlawful path the easy one.

So consent is part of the session record. ``meetscribe start`` asks how the
participants were informed, stores the answer with the session, and the panel shows
a permanent recording indicator while capture is live. Nothing here can make a
recording lawful -- only the user can do that -- but the design makes the lawful
path the default one and leaves an auditable record of what was declared.

The announcement helper exists because "tell them you are recording" is easy to
intend and easy to forget in the first thirty seconds of a call. It prints a
sentence ready to read aloud, in Spanish.
"""

from __future__ import annotations

from meetscribe.core.models import ConsentMode

__all__ = [
    "CONSENT_CHOICES",
    "ANNOUNCEMENT_ES",
    "FIRST_RUN_NOTICE_ES",
    "describe_consent",
    "parse_consent",
    "consent_is_sufficient",
]

CONSENT_CHOICES: dict[str, ConsentMode] = {
    "anunciado": ConsentMode.ANNOUNCED,
    "announced": ConsentMode.ANNOUNCED,
    "todos": ConsentMode.ALL_PARTIES,
    "all": ConsentMode.ALL_PARTIES,
    "solo-yo": ConsentMode.SELF_ONLY,
    "self": ConsentMode.SELF_ONLY,
    "ninguno": ConsentMode.NOT_RECORDED,
    "none": ConsentMode.NOT_RECORDED,
}

ANNOUNCEMENT_ES = (
    "Antes de empezar: estoy usando una herramienta local para transcribir esta "
    "reunión y generar un resumen. El audio y el texto se quedan en mi equipo. "
    "Si alguien prefiere que no lo haga, lo detengo ahora mismo."
)

FIRST_RUN_NOTICE_ES = """\
MeetScribe graba y transcribe conversaciones.

  - Todo se procesa y se guarda en este equipo. No hay servidor, no hay cuenta,
    no hay bot que entre a la reunión.
  - La ÚNICA excepción es el resumen con el proveedor 'anthropic': en ese caso el
    TEXTO de la transcripción se envía a la API de Anthropic. Puedes evitarlo por
    completo con el proveedor 'ollama' o con --offline.
  - En muchos países grabar una llamada requiere el consentimiento de todos los
    participantes. Avísales: 'meetscribe anuncio' te da una frase lista
    para leer en voz alta.

Esta advertencia se muestra una sola vez."""


def describe_consent(mode: ConsentMode) -> str:
    """Spanish description, for the CLI, the panel and the exported header."""
    return {
        ConsentMode.ANNOUNCED: "se anunció en voz alta a los participantes",
        ConsentMode.ALL_PARTIES: "consentimiento explícito de todos los participantes",
        ConsentMode.SELF_ONLY: "solo se conserva la pista propia",
        ConsentMode.NOT_RECORDED: "sin registro de consentimiento",
    }[mode]


def parse_consent(value: str | None) -> ConsentMode:
    if not value:
        return ConsentMode.NOT_RECORDED
    key = value.strip().lower()
    if key not in CONSENT_CHOICES:
        options = "anunciado, todos, solo-yo, ninguno"
        raise ValueError(f"consentimiento no reconocido: '{value}'. Opciones: {options}")
    return CONSENT_CHOICES[key]


def consent_is_sufficient(mode: ConsentMode, require: bool) -> bool:
    """Whether recording may proceed under the configured policy.

    ``SELF_ONLY`` counts as sufficient because in that mode the remote track is not
    retained -- there is nobody else's speech to have consented to.
    """
    if not require:
        return True
    return mode in (
        ConsentMode.ANNOUNCED,
        ConsentMode.ALL_PARTIES,
        ConsentMode.SELF_ONLY,
    )
