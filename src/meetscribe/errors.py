"""The exception hierarchy.

One rule runs through this project: **no silent failures**. Every failure mode
worth distinguishing gets a type here, carries a Spanish, actionable message for
the user in ``user_message``, and an English technical detail for the log. A bare
``except Exception`` that swallows and continues is a bug, not defensive coding.
"""

from __future__ import annotations

__all__ = [
    "MeetScribeError",
    "ConfigError",
    "DeviceNotFoundError",
    "DeviceLostError",
    "CaptureError",
    "SilentTrackError",
    "ModelNotFoundError",
    "ModelIntegrityError",
    "TranscriptionError",
    "DiarizationUnavailableError",
    "StoreError",
    "SessionNotFoundError",
    "SummarizerError",
    "SummarizerUnavailableError",
    "EgressBlockedError",
    "ConsentRequiredError",
    "SecretNotFoundError",
    "PanelError",
]


class MeetScribeError(Exception):
    """Base class. ``user_message`` is what the CLI prints, in Spanish."""

    default_user_message = "Ocurrio un error inesperado."

    def __init__(self, detail: str = "", user_message: str | None = None) -> None:
        super().__init__(detail or self.default_user_message)
        self.detail = detail
        self.user_message = user_message or self.default_user_message

    def __str__(self) -> str:
        return self.detail or self.user_message


# ------------------------------------------------------------------- configuration


class ConfigError(MeetScribeError):
    default_user_message = "La configuración no es válida."


class SecretNotFoundError(MeetScribeError):
    default_user_message = (
        "No se encontró la clave de API. Guárdala con 'meetscribe config set-key' "
        "o exporta ANTHROPIC_API_KEY."
    )


# ------------------------------------------------------------------------- capture


class CaptureError(MeetScribeError):
    default_user_message = "Falló la captura de audio."


class DeviceNotFoundError(CaptureError):
    default_user_message = (
        "No se encontró el dispositivo de audio. Ejecuta 'meetscribe devices' "
        "para ver los disponibles."
    )


class DeviceLostError(CaptureError):
    """The endpoint vanished mid-session: headphones plugged in, device disabled.

    Recoverable by design -- the capture manager reopens rather than aborting the
    meeting, so this is raised inward and handled, never shown to the user as a
    fatal error.
    """

    default_user_message = "El dispositivo de audio cambió; reconectando."


class SilentTrackError(CaptureError):
    """A track has been below the noise floor long enough that it is certainly
    not capturing. The spike caught exactly this on the microphone array."""

    default_user_message = (
        "Una pista lleva demasiado tiempo en silencio digital. Revisa la "
        "configuración de privacidad del micrófono en Windows."
    )


# -------------------------------------------------------------------------- models


class ModelNotFoundError(MeetScribeError):
    default_user_message = (
        "Falta un modelo. Ejecuta 'meetscribe models --download' para obtenerlo."
    )


class ModelIntegrityError(MeetScribeError):
    """A downloaded model file does not match its pinned SHA-256.

    Treated as fatal: an ONNX file is executable content, and a mismatch means
    either corruption or a compromised mirror. Neither is safe to load.
    """

    default_user_message = (
        "Un archivo de modelo no coincide con su huella SHA-256 esperada. "
        "Se rechazó por seguridad."
    )


class TranscriptionError(MeetScribeError):
    default_user_message = "Falló la transcripción."


class DiarizationUnavailableError(MeetScribeError):
    """Diarization could not start. Never fatal: the session degrades to a single
    'Remoto' label and keeps recording."""

    default_user_message = (
        "La separación de hablantes no está disponible; se usará una sola "
        "etiqueta 'Remoto'."
    )


# ------------------------------------------------------------------------ storage


class StoreError(MeetScribeError):
    default_user_message = "Falló el acceso a la base de datos."


class SessionNotFoundError(StoreError):
    default_user_message = "No existe una sesión con ese identificador."


# --------------------------------------------------------------------- summarizer


class SummarizerError(MeetScribeError):
    default_user_message = "Falló la generación del resumen."


class SummarizerUnavailableError(SummarizerError):
    """The LLM could not be reached. The transcript always survives; only the
    summary is lost, and it can be regenerated later with 'meetscribe summarize'."""

    default_user_message = (
        "No se pudo contactar al modelo de resumen. La transcripción está a "
        "salvo; reintenta con 'meetscribe summarize <id>'."
    )


# -------------------------------------------------------------- privacy / consent


class EgressBlockedError(MeetScribeError):
    """A remote adapter was asked to run in a session marked offline-only.

    Raised instead of silently downgrading, because the alternative -- sending the
    transcript anyway -- is the exact failure this flag exists to prevent.
    """

    default_user_message = (
        "Esta sesión es solo-local: no se enviará ningún texto fuera de la "
        "máquina. Usa el proveedor 'ollama' o desactiva --offline."
    )


class ConsentRequiredError(MeetScribeError):
    default_user_message = (
        "Falta registrar el consentimiento. Usa --consent para indicar cómo se "
        "informó a los participantes."
    )


# ---------------------------------------------------------------------- web panel


class PanelError(MeetScribeError):
    default_user_message = "Falló el panel web local."
