"""The Ollama adapter: fully local summarization, nothing leaves the machine.

This is the answer to "I want minutes of a meeting I am not willing to send
anywhere". It talks to an Ollama server on loopback and uses its structured-output
support -- ``format`` accepts a JSON schema and constrains generation to it, the
same guarantee the Claude path gets, so both adapters return the identical shape.

**Expectation setting, stated rather than implied.** An 8B model running on this
CPU produces noticeably weaker minutes than a frontier model: it misses implicit
commitments, blurs decisions into discussion, and occasionally ignores the "do not
invent an owner" instruction. It is nonetheless the right default for a sensitive
meeting, because the alternative is not "better minutes" -- it is "your customer's
name in someone else's logs".

Generation on CPU is also slow: expect a minute or more for a long meeting, which
is why the timeout defaults higher than the remote adapter's.
"""

from __future__ import annotations

import json
import logging

from meetscribe.errors import SummarizerError, SummarizerUnavailableError
from meetscribe.summarize.base import StructuredSummarizer

log = logging.getLogger(__name__)

__all__ = ["OllamaSummarizer", "DEFAULT_OLLAMA_MODEL", "DEFAULT_OLLAMA_HOST"]

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


class OllamaSummarizer(StructuredSummarizer):
    """Structured minutes from a local Ollama server."""

    provider = "ollama"
    is_local = True

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        timeout_s: float = 300.0,
        max_retries: int = 1,
        max_chunk_chars: int = 16_000,
        num_ctx: int = 8192,
    ) -> None:
        # Smaller chunks than the remote adapter on purpose: local models degrade
        # much faster with context length than frontier ones do.
        super().__init__(model, max_chunk_chars=max_chunk_chars)
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.num_ctx = num_ctx

    def complete(
        self, system: str, user: str, schema: dict, max_tokens: int = 8000
    ) -> str:
        import httpx

        body = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                # Low but non-zero: greedy decoding on small models produces
                # degenerate repetition inside long JSON arrays.
                "temperature": 0.2,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(f"{self.host}/api/chat", json=body)
                response.raise_for_status()
                payload = response.json()
                content = (payload.get("message") or {}).get("content", "")
                if not content.strip():
                    raise SummarizerError(
                        "empty response from ollama",
                        user_message="El modelo local devolvió una respuestá vacía.",
                    )
                return content
            except httpx.ConnectError as exc:
                raise SummarizerUnavailableError(
                    f"cannot reach ollama at {self.host}: {exc}",
                    user_message=(
                        f"No se pudo contactar a Ollama en {self.host}. "
                        "Comprueba que esté en ejecución ('ollama serve')."
                    ),
                ) from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise SummarizerUnavailableError(
                        f"model {self.model} not pulled: {exc}",
                        user_message=(
                            f"El modelo '{self.model}' no está descargado en Ollama. "
                            f"Ejecuta 'ollama pull {self.model}'."
                        ),
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                log.info("ollama attempt %d failed (%s); retrying", attempt + 1, last_error)

        raise SummarizerUnavailableError(
            f"ollama failed after {self.max_retries + 1} attempts: {last_error}",
            user_message=(
                "El modelo local no respondió a tiempo. La transcripción está "
                "guardada; reintenta con 'meetscribe summarize <id>'."
            ),
        )

    def health(self) -> str:
        import httpx

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
        except httpx.HTTPError:
            return f"no disponible en {self.host} (¿'ollama serve' está corriendo?)"
        except Exception as exc:
            return f"error: {type(exc).__name__}"

        if self.model in names or any(n.startswith(f"{self.model}:") for n in names):
            return f"disponible ({self.model}, local)"
        available = ", ".join(sorted(names)[:5]) or "ninguno"
        return (
            f"el modelo '{self.model}' no está descargado "
            f"(disponibles: {available}). Usa 'ollama pull {self.model}'"
        )
