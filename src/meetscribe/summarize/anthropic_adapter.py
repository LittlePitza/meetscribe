"""The Claude adapter -- the highest-quality summaries, and the only path where
data leaves the machine.

That trade is made explicit everywhere it appears: the session records it, the CLI
warns on first use, ``privacy.offline_only`` makes it impossible, and the base class
refuses rather than silently downgrading. What is sent is the transcript *text*
(optionally PII-redacted); audio never leaves the machine under any configuration.

**API choices, and why each one.**

* ``output_config.format`` with a JSON schema, so the response is guaranteed
  parseable JSON rather than prose that happens to look like JSON. This is what
  makes the summary testable.
* **Streaming.** A two-hour transcript is a long input and the response can be
  long; a non-streaming request risks an HTTP timeout mid-generation, which costs
  the whole call. ``get_final_message()`` gives back the complete message with no
  event handling to write.
* **Adaptive thinking**, the current API for extended thinking. ``budget_tokens``
  is removed on this model family and would be rejected outright.
* **Server-side refusal fallbacks**, enabled by default. A safety classifier
  declining a request would otherwise mean the meeting simply gets no summary; with
  fallbacks the API re-runs it on another model inside the same call. The beta is
  requested defensively: if the account or endpoint does not accept it, the adapter
  retries once on the stable path rather than failing.
"""

from __future__ import annotations

import logging

from meetscribe.errors import SummarizerError, SummarizerUnavailableError
from meetscribe.summarize.base import StructuredSummarizer

log = logging.getLogger(__name__)

__all__ = ["AnthropicSummarizer", "DEFAULT_MODEL", "FALLBACK_BETA"]

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicSummarizer(StructuredSummarizer):
    """Structured meeting minutes via the Claude Messages API."""

    provider = "anthropic"
    is_local = False

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        timeout_s: float = 180.0,
        max_retries: int = 2,
        max_chunk_chars: int = 40_000,
        redact_pii_before_egress: bool = False,
        offline_only: bool = False,
        refusal_fallback: bool = True,
    ) -> None:
        super().__init__(
            model,
            max_chunk_chars=max_chunk_chars,
            redact_pii_before_egress=redact_pii_before_egress,
            offline_only=offline_only,
        )
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.refusal_fallback = refusal_fallback
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self.timeout_s,
                max_retries=self.max_retries,
            )
        return self._client

    # ------------------------------------------------------------------ request

    def complete(
        self, system: str, user: str, schema: dict, max_tokens: int = 16000
    ) -> str:
        import anthropic

        client = self._get_client()
        effort = "high" if max_tokens > 4000 else "low"
        params: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }

        try:
            message = self._stream(client, params, with_fallback=self.refusal_fallback)
        except anthropic.BadRequestError as exc:
            if not self.refusal_fallback:
                raise self._translate(exc) from exc
            # The refusal-fallback beta was rejected (older endpoint, restricted
            # account). Losing the summary over an optional resilience feature
            # would be absurd, so retry once without it.
            log.info("refusal fallback rejected (%s); retrying without it", exc)
            self.refusal_fallback = False
            try:
                message = self._stream(client, params, with_fallback=False)
            except Exception as retry_exc:
                raise self._translate(retry_exc) from retry_exc
        except Exception as exc:
            raise self._translate(exc) from exc

        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) or "desconocida"
            raise SummarizerError(
                f"model refused the request (category={category})",
                user_message=(
                    "El modelo rechazó generar el resumen de esta reunión "
                    f"(motivo: {category}). La transcripción está intacta; puedes "
                    "usar el proveedor local 'ollama'."
                ),
            )

        for block in message.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                return block.text
        raise SummarizerError(
            "empty response from the model",
            user_message="El modelo devolvió una respuestá vacía.",
        )

    def _stream(self, client, params: dict, with_fallback: bool):
        if with_fallback:
            with client.beta.messages.stream(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            ) as stream:
                return stream.get_final_message()
        with client.messages.stream(**params) as stream:
            return stream.get_final_message()

    # ----------------------------------------------------------------- failures

    @staticmethod
    def _translate(exc: BaseException) -> SummarizerError:
        """Map SDK exceptions to actionable Spanish, most specific first."""
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            return SummarizerUnavailableError(
                f"authentication failed: {exc}",
                user_message=(
                    "La clave de API de Anthropic no es válida. Actualízala con "
                    "'meetscribe config set-key'."
                ),
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return SummarizerUnavailableError(
                f"permission denied: {exc}",
                user_message="La clave de API no tiene permisos para este modelo.",
            )
        if isinstance(exc, anthropic.NotFoundError):
            return SummarizerError(
                f"model not found: {exc}",
                user_message=(
                    "El modelo configurado no existe. Revisa 'summarizer.model' "
                    "en la configuración."
                ),
            )
        if isinstance(exc, anthropic.RateLimitError):
            retry_after = "60"
            response = getattr(exc, "response", None)
            if response is not None:
                retry_after = response.headers.get("retry-after", "60")
            return SummarizerUnavailableError(
                f"rate limited: {exc}",
                user_message=(
                    f"Límite de peticiones alcanzado. Reintenta en {retry_after}s "
                    "con 'meetscribe summarize <id>'."
                ),
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return SummarizerUnavailableError(
                f"connection error: {exc}",
                user_message=(
                    "No hay conexión con la API de Anthropic. La transcripción "
                    "está guardada; reintenta el resumen cuando vuelva la red."
                ),
            )
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", 0)
            if status >= 500:
                return SummarizerUnavailableError(
                    f"server error {status}: {exc}",
                    user_message=(
                        f"La API de Anthropic devolvió un error {status}. "
                        "Reintenta más tarde."
                    ),
                )
            return SummarizerError(
                f"api error {status}: {exc}",
                user_message=f"Error de la API de Anthropic ({status}).",
            )
        if isinstance(exc, SummarizerError):
            return exc
        return SummarizerError(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------- health

    def health(self) -> str:
        import anthropic

        if self.offline_only:
            return "bloqueado por el modo solo-local"
        try:
            client = self._get_client()
            client.with_options(timeout=15.0, max_retries=0).messages.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": "ok"}],
            )
        except anthropic.AuthenticationError:
            return "clave de API invalida"
        except anthropic.APIConnectionError:
            return "sin conexión con api.anthropic.com"
        except anthropic.NotFoundError:
            return f"el modelo '{self.model}' no existe"
        except Exception as exc:
            return f"error: {type(exc).__name__}"
        return f"disponible ({self.model})"
