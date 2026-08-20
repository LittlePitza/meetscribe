"""Building the configured summarizer, with the egress decision made once.

Every path that needs a summarizer -- starting a session, re-running a summary,
``doctor`` -- goes through here, so the rule "offline-only means no remote adapter,
ever" is enforced in one place instead of being re-implemented (and eventually
forgotten) at each call site.
"""

from __future__ import annotations

import logging

from meetscribe.config import Config
from meetscribe.errors import EgressBlockedError
from meetscribe.security.paths import config_path
from meetscribe.security.secrets import resolve_api_key
from meetscribe.summarize.anthropic_adapter import AnthropicSummarizer
from meetscribe.summarize.base import NullSummarizer, StructuredSummarizer
from meetscribe.summarize.ollama_adapter import OllamaSummarizer

log = logging.getLogger(__name__)

__all__ = ["build_summarizer", "EGRESS_WARNING_ES"]

EGRESS_WARNING_ES = """\
AVISO: el proveedor de resumen es 'anthropic'.

El TEXTO de la transcripción se enviará a la API de Anthropic para generar el acta.
El audio nunca sale de este equipo, y la transcripción permanece guardada aquí de
todos modos.

Si esta reunión es sensible:
  meetscribe start --offline            (esta sesión, sin salida de datos)
  summarizer.provider: ollama           (siempre local, en la configuración)"""


def build_summarizer(
    config: Config, offline_only: bool | None = None
) -> StructuredSummarizer:
    """Construct the adapter named by the configuration.

    ``offline_only`` overrides the config for a single session, which is how
    ``meetscribe start --offline`` works: a per-meeting decision that cannot be
    undone by a stale config file.
    """
    offline = config.privacy.offline_only if offline_only is None else offline_only
    provider = config.summarizer.provider

    if provider == "none":
        return NullSummarizer()

    if provider == "ollama":
        return OllamaSummarizer(
            model=config.summarizer.ollama_model,
            host=config.summarizer.ollama_host,
            timeout_s=config.summarizer.timeout_s,
            max_retries=config.summarizer.max_retries,
        )

    if provider == "anthropic":
        if offline:
            raise EgressBlockedError(
                "anthropic provider requested in an offline-only session",
                user_message=(
                    "Esta sesión es solo-local y el proveedor configurado "
                    "('anthropic') enviaría texto fuera de la máquina. Cambia a "
                    "'ollama' con --provider ollama, o quita --offline."
                ),
            )
        api_key = resolve_api_key(
            "anthropic",
            config_value=config.summarizer.api_key,
            config_file=config_path(),
            required=True,
        )
        assert api_key is not None  # noqa: S101 - narrowing; required=True guarantees it
        return AnthropicSummarizer(
            api_key=api_key,
            model=config.summarizer.model,
            timeout_s=config.summarizer.timeout_s,
            max_retries=config.summarizer.max_retries,
            max_chunk_chars=config.summarizer.max_chunk_chars,
            redact_pii_before_egress=config.summarizer.redact_pii,
            offline_only=offline,
            refusal_fallback=config.summarizer.refusal_fallback,
        )

    raise EgressBlockedError(
        f"unknown provider {provider!r}",
        user_message=(
            f"Proveedor de resumen desconocido: '{provider}'. "
            "Opciones: anthropic, ollama, none."
        ),
    )
