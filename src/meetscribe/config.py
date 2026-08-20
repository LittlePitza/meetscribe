"""Configuration.

Plain nested dataclasses with real defaults, loaded from one YAML file. Two
properties matter more than the shape:

**Unknown keys are an error.** A silently ignored ``diarisation:`` (British
spelling, or a typo) would leave the user convinced they had configured something
they had not. Loading raises with the offending key and the valid alternatives.

**The API key is never part of a dump.** :meth:`Config.to_public_dict` is what the
CLI prints and what the panel receives; it omits the secret entirely rather than
masking it, so there is no code path where the value could be formatted into
output at all.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from meetscribe.errors import ConfigError
from meetscribe.security.paths import config_path, data_dir

__all__ = [
    "Config",
    "CaptureConfig",
    "SttConfig",
    "VadConfig",
    "DiarizationConfig",
    "SummarizerConfig",
    "PanelConfig",
    "PrivacyConfig",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG_YAML",
]


@dataclass(slots=True)
class CaptureConfig:
    """Audio capture. ``None`` device means "resolve the current default"."""

    backend: str = "auto"
    """'auto' tries PyAudioWPatch then falls back to soundcard; 'pyaudio' and
    'soundcard' pin one backend."""

    mic_device: str | None = None
    loopback_device: str | None = None
    frames_per_buffer: int = 1024
    """21 ms at 48 kHz. Small enough for a responsive level meter, large enough
    that Python's per-callback overhead stays negligible."""

    buffer_seconds: float = 30.0
    """Per-track bound on undelivered audio. At the measured RTF of 0.32 this is
    roughly 90 seconds of headroom before anything could be dropped."""

    mic_silence_window_s: float = 12.0
    """A microphone always has a noise floor, so digital zero for 12 s is a fault."""

    loopback_silence_window_s: float = 90.0
    """Silence on the render endpoint is normal between turns; only a long stretch
    is worth reporting."""

    retain_audio: str = "none"
    """'none' keeps no audio at all (the default -- the transcript is the artifact
    worth keeping and the audio is the most sensitive thing on the disk);
    'session' keeps WAVs until the process exits and shreds them; 'keep' retains
    them under the retention policy."""

    follow_default_device: bool = True
    """Reopen against the new endpoint when Windows switches the default."""


@dataclass(slots=True)
class VadConfig:
    threshold: float = 0.5
    min_silence_duration: float = 0.45
    min_speech_duration: float = 0.25
    max_speech_duration: float = 20.0


@dataclass(slots=True)
class SttConfig:
    model: str = "large-v3-turbo"
    """Measured at RTF 0.305 and WER 4.6 % on this CPU. 'small' (RTF 0.124,
    WER 6.0 %) is the throughput alternative."""

    language: str = "es"
    """'auto' detects per utterance, which handles code-switching to English at the
    cost of occasional misdetection on short turns."""

    cpu_threads: int = 0
    """0 means min(8, cpu_count) -- leaving cores for Teams and the OS."""

    beam_size: int = 5
    compute_type: str = "int8"
    auto_detect_min_s: float = 2.0
    interim_after_s: float = 8.0
    """Publish provisional text once an utterance has run this long."""

    interim_every_s: float = 5.0
    queue_size: int = 32
    initial_prompt: str | None = None


@dataclass(slots=True)
class DiarizationConfig:
    enabled: bool = True
    threshold: float = 0.60
    min_embed_s: float = 0.35
    min_new_speaker_s: float = 1.0
    max_speakers: int = 8
    label_prefix: str = "Participante"
    refine_at_end: bool = True


@dataclass(slots=True)
class SummarizerConfig:
    provider: str = "anthropic"
    """'anthropic' (best quality, text leaves the machine), 'ollama' (fully local),
    or 'none' (transcript only)."""

    model: str = "claude-opus-5"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    timeout_s: float = 180.0
    max_retries: int = 2
    max_chunk_chars: int = 40_000
    """Map-reduce threshold. Well inside the model's context window, chosen so a
    chunk plus its instructions and the accumulating running summary always fit."""

    live_action_items: bool = True
    live_interval_s: float = 120.0
    redact_pii: bool = False
    """Strip emails, phone numbers and IDs before anything leaves the machine."""

    api_key: str | None = None
    """Discouraged. Prefer 'meetscribe config set-key', which uses the OS store."""

    refusal_fallback: bool = True
    """Let the API re-run a declined request on a fallback model automatically."""


@dataclass(slots=True)
class PanelConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    """Never 0.0.0.0. The loader rejects any other value unless
    ``allow_non_local`` is explicitly set, because binding a live transcript to a
    LAN interface is not a preference, it is a leak."""

    port: int = 8765
    """0 picks a free port."""

    open_browser: bool = True
    heartbeat_s: float = 15.0
    history_limit: int = 5000
    allow_non_local: bool = False


@dataclass(slots=True)
class PrivacyConfig:
    offline_only: bool = False
    """Makes egress impossible: remote summarizers are refused, not downgraded."""

    require_consent: bool = True

    default_consent: str = ""
    """How participants are normally informed, so ``start`` does not ask every
    time. Empty means "ask". Set by ``meetscribe setup``; ``--consentimiento``
    always overrides it, because the answer is a claim about a specific meeting and
    the person running it must be able to correct it."""

    retention_days: int = 0
    """0 disables automatic deletion. Any positive value purges sessions older
    than that on startup."""

    first_run_notice_shown: bool = False


@dataclass(slots=True)
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    log_level: str = "INFO"

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        return _build(cls, raw or {}, path="")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        """Everything except the secret. Safe to print, log or send to the panel."""
        data = self.to_dict()
        data["summarizer"].pop("api_key", None)
        data["summarizer"]["api_key_configured"] = bool(self.summarizer.api_key)
        return data

    def validate(self) -> None:
        if self.panel.host != "127.0.0.1" and not self.panel.allow_non_local:
            raise ConfigError(
                f"panel.host={self.panel.host!r} is not loopback",
                user_message=(
                    "El panel solo puede escuchar en 127.0.0.1. Exponerlo en la red "
                    "publicaría la transcripción en vivo. Si realmente lo necesitas, "
                    "activa 'panel.allow_non_local: true' de forma explícita."
                ),
            )
        if self.capture.retain_audio not in ("none", "session", "keep"):
            raise ConfigError(
                f"invalid capture.retain_audio={self.capture.retain_audio!r}",
                user_message=(
                    "capture.retain_audio debe ser 'none', 'session' o 'keep'."
                ),
            )
        if self.summarizer.provider not in ("anthropic", "ollama", "none"):
            raise ConfigError(
                f"invalid summarizer.provider={self.summarizer.provider!r}",
                user_message=(
                    "summarizer.provider debe ser 'anthropic', 'ollama' o 'none'."
                ),
            )
        if self.privacy.offline_only and self.summarizer.provider == "anthropic":
            raise ConfigError(
                "offline_only conflicts with the anthropic provider",
                user_message=(
                    "El modo solo-local está activo pero el proveedor de resumen es "
                    "'anthropic', que envía texto fuera de la máquina. Cambia a "
                    "'ollama' o desactiva privacy.offline_only."
                ),
            )
        if not 0 <= self.panel.port <= 65535:
            raise ConfigError(
                f"invalid panel.port={self.panel.port}",
                user_message="panel.port debe estar entre 0 y 65535.",
            )

    def apply_env(self) -> Config:
        """Environment overrides, for scripted runs and the test suite."""
        mapping: dict[str, tuple[Any, str]] = {
            "MEETSCRIBE_STT_MODEL": (self.stt, "model"),
            "MEETSCRIBE_LANGUAGE": (self.stt, "language"),
            "MEETSCRIBE_PROVIDER": (self.summarizer, "provider"),
            "MEETSCRIBE_PANEL_PORT": (self.panel, "port"),
            "MEETSCRIBE_LOG_LEVEL": (self, "log_level"),
        }
        for env_var, (target, attribute) in mapping.items():
            value = os.environ.get(env_var)
            if value is None or value == "":
                continue
            current = getattr(target, attribute)
            setattr(target, attribute, type(current)(value) if current is not None else value)
        if os.environ.get("MEETSCRIBE_OFFLINE") in ("1", "true", "yes"):
            self.privacy.offline_only = True
        return self


def _build(cls: type, raw: dict[str, Any], path: str) -> Any:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in raw.items():
        if key not in known:
            where = f"{path}{key}"
            options = ", ".join(sorted(known))
            raise ConfigError(
                f"unknown configuration key {where!r}",
                user_message=(
                    f"Clave de configuración desconocida: '{where}'. "
                    f"Opciones válidas en esta sección: {options}."
                ),
            )
        target_type = known[key].type
        if is_dataclass(_resolve(target_type)) and isinstance(value, dict):
            kwargs[key] = _build(_resolve(target_type), value, path=f"{path}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve(annotation: Any) -> Any:
    """Dataclass field types arrive as strings under ``from __future__``."""
    if isinstance(annotation, str):
        return {
            "CaptureConfig": CaptureConfig,
            "VadConfig": VadConfig,
            "SttConfig": SttConfig,
            "DiarizationConfig": DiarizationConfig,
            "SummarizerConfig": SummarizerConfig,
            "PanelConfig": PanelConfig,
            "PrivacyConfig": PrivacyConfig,
        }.get(annotation, annotation)
    return annotation


def load_config(path: Path | None = None) -> Config:
    """Read the config file, apply environment overrides, validate."""
    target = path or config_path()
    raw: dict[str, Any] = {}
    if target.exists():
        import yaml

        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"{target} is not valid YAML: {exc}",
                user_message=(
                    f"El archivo de configuración '{target}' tiene un error de "
                    "sintaxis YAML."
                ),
            ) from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(
                    f"{target} must contain a mapping",
                    user_message=f"'{target}' debe contener un mapa de claves.",
                )
            raw = loaded
    config = Config.from_dict(raw).apply_env()
    config.validate()
    return config


def save_config(config: Config, path: Path | None = None) -> Path:
    """Write the config back out, without the secret."""
    import yaml

    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    data["summarizer"].pop("api_key", None)
    target.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return target


DEFAULT_CONFIG_YAML = f"""\
# MeetScribe -- configuración
# Datos y grabaciones: {data_dir()}
# La clave de API NO se guarda aquí: usa 'meetscribe config set-key'.

stt:
  model: large-v3-turbo   # tiny | base | small | large-v3-turbo
  language: es            # 'auto' para detectar por intervención

diarization:
  enabled: true
  threshold: 0.60

summarizer:
  provider: anthropic     # anthropic | ollama | none
  model: claude-opus-5

privacy:
  offline_only: false     # true impide que cualquier texto salga de la máquina
  require_consent: true

panel:
  port: 8765
"""
