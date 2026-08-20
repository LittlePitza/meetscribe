"""``meetscribe doctor`` -- prove the machine can actually record a meeting.

Written against one specific memory: the spike opened the microphone successfully
and captured **digital silence**. No exception, no warning. A self-check that only
verified "the device opened" would have passed, and the user would have discovered
the problem an hour later with half a transcript.

So the audio checks here **pull real frames and measure the level**. Everything
else follows the same rule: verify the thing that has to work, not a proxy for it.
Models are hash-verified, not merely present. The summarizer is contacted, not
merely configured.

Each check returns a status and, when something is wrong, the exact command that
fixes it. ``ERROR`` means a meeting recorded right now would produce a broken or
empty transcript; ``AVISO`` means degraded but usable.
"""

from __future__ import annotations

import platform
import socket
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from meetscribe.assets import REQUIRED_MODELS, find_model, sha256_file
from meetscribe.audio.level import DIGITAL_SILENCE_DBFS, peak_level, rms_dbfs
from meetscribe.audio.resample import StreamResampler
from meetscribe.config import Config
from meetscribe.core.models import TrackKind
from meetscribe.errors import MeetScribeError
from meetscribe.security.paths import (
    cloud_sync_root,
    data_dir,
    db_path,
    describe_protection,
)
from meetscribe.stt.transcriber import SUPPORTED_MODELS

__all__ = ["Check", "OK", "WARN", "ERROR", "run_all", "probe_track"]

OK = "OK"
WARN = "AVISO"
ERROR = "ERROR"

# Measured on the target machine; see docs/benchmarks.md.
MEASURED_RTF = {
    "tiny": 0.024,
    "base": 0.045,
    "small": 0.124,
    "large-v3-turbo": 0.305,
}
DIARIZATION_RTF = 0.017  # VAD on two tracks plus one speaker embedding


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == ERROR


# --------------------------------------------------------------------- system


def check_platform() -> Check:
    version = ".".join(map(str, sys.version_info[:3]))
    detail = f"Python {version} · {platform.system()} {platform.release()}"
    if sys.platform != "win32":
        return Check(
            "Plataforma",
            WARN,
            f"{detail} — la captura WASAPI solo existe en Windows",
            "El resto de comandos (export, summarize) sí funcionan.",
        )
    return Check("Plataforma", OK, detail)


def check_dependencies() -> Check:
    required = {
        "numpy": "cálculo numérico",
        "scipy": "remuestreo y clustering",
        "soxr": "remuestreo en streaming",
        "faster_whisper": "transcripción",
        "sherpa_onnx": "VAD y hablantes",
        "click": "CLI",
        "yaml": "configuración",
        "httpx": "descargas y Ollama",
    }
    optional = {
        "pyaudiowpatch": "captura WASAPI (obligatorio en Windows)",
        "soundcard": "backend de captura de respaldo",
        "anthropic": "resumen con Claude",
        "keyring": "guardar la clave de API de forma segura",
    }
    missing = [name for name in required if not _importable(name)]
    absent_optional = [name for name in optional if not _importable(name)]

    if missing:
        return Check(
            "Dependencias",
            ERROR,
            "faltan: " + ", ".join(missing),
            "uv sync",
        )
    if absent_optional:
        return Check(
            "Dependencias",
            WARN,
            "opcionales ausentes: " + ", ".join(absent_optional),
            "uv sync --all-extras",
        )
    return Check("Dependencias", OK, f"{len(required) + len(optional)} paquetes presentes")


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_data_dir() -> Check:
    root = data_dir()
    if not root.exists():
        return Check(
            "Directorio de datos",
            WARN,
            f"{root} aún no existe",
            "Se creará al iniciar la primera sesión.",
        )
    protected, description = describe_protection(root)
    detail = f"{root} · {description}"
    synced = cloud_sync_root(root)
    if synced is not None:
        return Check(
            "Directorio de datos",
            ERROR,
            f"{detail} — está dentro de '{synced.name}' y se sincronizaría a la nube",
            "Quita MEETSCRIBE_DATA_DIR o apúntalo a una carpeta sin sincronización.",
        )
    if not protected:
        return Check(
            "Directorio de datos",
            WARN,
            detail,
            "Ejecuta 'meetscribe config harden' para restringir los permisos.",
        )
    return Check("Directorio de datos", OK, detail)


def check_database() -> Check:
    path = db_path()
    if not path.exists():
        return Check("Base de datos", OK, f"{path} (se creará al usarla)")
    try:
        from meetscribe.store.sqlite_store import SqliteStore

        with SqliteStore(path) as store:
            stats = store.stats()
        return Check(
            "Base de datos",
            OK,
            f"{stats['sessions']} sesiones, {stats['segments']} intervenciones",
        )
    except MeetScribeError as exc:
        return Check("Base de datos", ERROR, exc.user_message, "Revisa el archivo o bórralo.")


# ---------------------------------------------------------------------- audio


def probe_track(
    config: Config, track: TrackKind, seconds: float = 2.0
) -> tuple[str, float, float]:
    """Open the real device and measure. Returns ``(device name, dBFS, peak)``."""
    from meetscribe.pipeline.factory import build_source_factory

    source = build_source_factory(config, track)()
    fmt = source.open()
    resampler = StreamResampler(fmt.sample_rate, fmt.channels)
    collected: list[np.ndarray] = []
    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline:
            raw = source.read()
            if raw is None:
                break
            mono = resampler.process(raw)
            if mono.size:
                collected.append(mono)
    finally:
        source.close()
    buffer = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    return (source.name, rms_dbfs(buffer), peak_level(buffer))


def check_audio(config: Config, track: TrackKind, seconds: float = 2.0) -> Check:
    label = "Micrófono" if track is TrackKind.MIC else "Audio de la reunión (loopback)"
    try:
        name, dbfs, peak = probe_track(config, track, seconds)
    except MeetScribeError as exc:
        return Check(label, ERROR, exc.user_message, "meetscribe devices")
    except Exception as exc:
        return Check(label, ERROR, f"{type(exc).__name__}: {exc}", "meetscribe devices")

    detail = f"{name} · {dbfs:.1f} dBFS, pico {peak:.4f}"
    if dbfs <= DIGITAL_SILENCE_DBFS:
        if track is TrackKind.MIC:
            return Check(
                label,
                ERROR,
                f"{detail} — silencio digital: el dispositivo se abre pero no "
                "entrega señal",
                "Configuración > Privacidad y seguridad > Micrófono: permite el "
                "acceso a las aplicaciones de escritorio. Comprueba también que no "
                "esté silenciado en el mezclador de Windows.",
                {"dbfs": dbfs, "peak": peak},
            )
        return Check(
            label,
            WARN,
            f"{detail} — sin audio ahora mismo",
            "Es normal si no se está reproduciendo nada. Repite la prueba con "
            "música o un video sonando.",
            {"dbfs": dbfs, "peak": peak},
        )
    return Check(label, OK, detail, extra={"dbfs": dbfs, "peak": peak})


def check_devices() -> Check:
    try:
        from meetscribe.capture.devices import list_devices, resolve_loopback, resolve_microphone

        devices = list_devices()
        loopback = resolve_loopback(devices)
        microphone = resolve_microphone(devices)
    except MeetScribeError as exc:
        return Check("Dispositivos", ERROR, exc.user_message, "meetscribe devices")
    except Exception as exc:
        return Check("Dispositivos", ERROR, f"{type(exc).__name__}: {exc}")
    return Check(
        "Dispositivos",
        OK,
        f"{len(devices)} endpoints · loopback: {loopback.name} · mic: {microphone.name}",
    )


# --------------------------------------------------------------------- models


def check_models() -> Check:
    missing = []
    corrupt = []
    for asset in REQUIRED_MODELS:
        path = find_model(asset)
        if path is None:
            missing.append(f"{asset.filename} ({asset.size_mb:.0f} MB)")
            continue
        if sha256_file(path) != asset.sha256:
            corrupt.append(str(path))
    if corrupt:
        return Check(
            "Modelos ONNX",
            ERROR,
            "huella SHA-256 incorrecta: " + ", ".join(corrupt),
            "Borra esos archivos y ejecuta 'meetscribe models --download'.",
        )
    if missing:
        return Check(
            "Modelos ONNX",
            ERROR,
            "faltan: " + ", ".join(missing),
            "meetscribe models --download",
        )
    return Check("Modelos ONNX", OK, "presentes y verificados (SHA-256)")


def check_stt_model(config: Config) -> Check:
    model = config.stt.model
    if model not in SUPPORTED_MODELS:
        return Check(
            "Modelo de transcripción",
            ERROR,
            f"'{model}' no es un modelo soportado",
            f"Opciones: {', '.join(SUPPORTED_MODELS)}",
        )
    try:
        from faster_whisper import WhisperModel

        WhisperModel(model, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        return Check(
            "Modelo de transcripción",
            WARN,
            f"'{model}' no está en la cache local",
            "Se descargará en el primer uso (requiere red). Para adelantarlo: "
            "'meetscribe models --stt'.",
        )
    return Check("Modelo de transcripción", OK, f"'{model}' en cache local")


def check_cpu_budget(config: Config) -> Check:
    import os

    cores = os.cpu_count() or 1
    threads = config.stt.cpu_threads or min(8, cores)
    stt_rtf = MEASURED_RTF.get(config.stt.model, 0.35)
    total = stt_rtf + (DIARIZATION_RTF if config.diarization.enabled else 0.008)
    detail = (
        f"{cores} núcleos, {threads} hilos para STT · "
        f"RTF estimado {total:.2f} ({1 / total:.1f}x tiempo real)"
    )
    if cores < 4:
        return Check(
            "Presupuesto de CPU",
            WARN,
            detail,
            "Con menos de 4 núcleos usa 'stt.model: small' o 'base'.",
        )
    if total > 0.7:
        return Check(
            "Presupuesto de CPU",
            WARN,
            detail,
            "El margen es escaso; considera 'stt.model: small' (RTF 0.124).",
        )
    return Check("Presupuesto de CPU", OK, detail)


# --------------------------------------------------------------- integrations


def check_summarizer(config: Config) -> Check:
    from meetscribe.summarize.factory import build_summarizer

    if config.summarizer.provider == "none":
        return Check("Resumen", WARN, "desactivado", "summarizer.provider: anthropic | ollama")
    try:
        summarizer = build_summarizer(config)
        status = summarizer.health()
    except MeetScribeError as exc:
        return Check("Resumen", WARN, exc.user_message, "meetscribe config set-key")
    except Exception as exc:
        return Check("Resumen", WARN, f"{type(exc).__name__}: {exc}")

    good = status.startswith("disponible")
    egress = "" if summarizer.is_local else " · el texto sale de esta máquina"
    return Check(
        "Resumen",
        OK if good else WARN,
        f"{config.summarizer.provider}: {status}{egress}",
        "" if good else "Sin resumen la transcripción se guarda igualmente.",
    )


def check_panel_port(config: Config) -> Check:
    if not config.panel.enabled:
        return Check("Panel web", WARN, "desactivado en la configuración")
    if config.panel.port == 0:
        return Check("Panel web", OK, "puerto automático")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        try:
            probe.bind((config.panel.host, config.panel.port))
        except OSError as exc:
            return Check(
                "Panel web",
                WARN,
                f"{config.panel.host}:{config.panel.port} ocupado ({exc.strerror or exc})",
                "Usa '--panel-port 0' para elegir un puerto libre automáticamente.",
            )
    return Check("Panel web", OK, f"{config.panel.host}:{config.panel.port} disponible")


def check_privacy(config: Config) -> Check:
    if config.privacy.offline_only:
        return Check("Privacidad", OK, "modo solo-local: ningún texto sale del equipo")
    if config.summarizer.provider == "anthropic":
        redaction = "con" if config.summarizer.redact_pii else "sin"
        return Check(
            "Privacidad",
            WARN,
            f"el resumen envía la transcripción a Anthropic ({redaction} redacción de PII)",
            "Usa --offline o 'summarizer.provider: ollama' para reuniones sensibles.",
        )
    return Check("Privacidad", OK, "el resumen es local; nada sale del equipo")


# ------------------------------------------------------------------- runner


def run_all(config: Config, with_audio: bool = True, seconds: float = 2.0) -> list[Check]:
    checks = [
        check_platform(),
        check_dependencies(),
        check_data_dir(),
        check_database(),
        check_models(),
        check_stt_model(config),
        check_cpu_budget(config),
    ]
    if with_audio and sys.platform == "win32":
        checks.append(check_devices())
        checks.append(check_audio(config, TrackKind.LOOPBACK, seconds))
        checks.append(check_audio(config, TrackKind.MIC, seconds))
    checks.append(check_panel_port(config))
    checks.append(check_summarizer(config))
    checks.append(check_privacy(config))
    return checks
