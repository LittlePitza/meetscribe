"""WASAPI device discovery.

Kept separate from the capture sources so ``meetscribe devices`` and
``meetscribe doctor`` can enumerate hardware without opening a stream, and so the
selection logic -- "which loopback endpoint belongs to the current default
speaker?" -- is testable against a recorded device table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meetscribe.core.models import AudioFormat, TrackKind
from meetscribe.errors import CaptureError, DeviceNotFoundError

__all__ = [
    "DeviceInfo",
    "list_devices",
    "resolve_loopback",
    "resolve_microphone",
    "pick_device",
    "default_endpoint_indices",
]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One WASAPI endpoint, in the terms the rest of the app cares about."""

    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int
    is_loopback: bool = False
    is_default: bool = False

    @property
    def format(self) -> AudioFormat:
        """The format this endpoint will be opened with.

        Capped at 2 channels: some render endpoints advertise 8 for surround, and
        opening all of them costs bandwidth for audio that is downmixed to mono
        one step later anyway.
        """
        return AudioFormat(
            sample_rate=self.default_sample_rate,
            channels=min(2, max(1, self.max_input_channels)),
        )

    def describe(self) -> str:
        # Most drivers already put "[Loopback]" in the name; do not say it twice.
        tag = (
            " [loopback]"
            if self.is_loopback and "loopback" not in self.name.lower()
            else ""
        )
        star = " *" if self.is_default else ""
        return (
            f"[{self.index}] {self.name}{tag}{star} "
            f"({self.max_input_channels}ch @ {self.default_sample_rate} Hz)"
        )


def _pyaudio() -> Any:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise CaptureError(
            f"PyAudioWPatch no disponible: {exc}",
            user_message=(
                "PyAudioWPatch no está instalado. Es la única forma de capturar "
                "el audio de la reunión en Windows: instala las dependencias con "
                "'uv sync'."
            ),
        ) from exc
    return pyaudio


def list_devices() -> list[DeviceInfo]:
    """Every WASAPI capture endpoint, with the loopback companions marked.

    PyAudioWPatch exposes loopback endpoints **twice**: once in the ordinary
    host-API device enumeration and once through
    ``get_loopback_device_info_generator``. They carry the same index, so the
    generator is used as a *set of indices to mark*, not as a second source of
    devices. Treating it as the latter is how an earlier version of this function
    ended up reporting "no loopback endpoint" on a machine that had two -- the
    de-duplication silently discarded exactly the devices it was meant to add.
    """
    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_in = int(wasapi.get("defaultInputDevice", -1))
        default_out = int(wasapi.get("defaultOutputDevice", -1))
        default_out_name = ""
        if default_out >= 0:
            default_out_name = str(pa.get_device_info_by_index(default_out)["name"])

        loopback_raw: dict[int, Any] = {}
        for raw in pa.get_loopback_device_info_generator():
            loopback_raw[int(raw["index"])] = raw

        devices: dict[int, DeviceInfo] = {}
        for i in range(int(wasapi["deviceCount"])):
            raw = pa.get_device_info_by_host_api_device_index(wasapi["index"], i)
            if int(raw["maxInputChannels"]) <= 0:
                continue
            devices[int(raw["index"])] = _to_device(
                raw, loopback_raw, default_in, default_out_name
            )

        # A loopback endpoint the host-API enumeration did not list (driver
        # dependent) still has to be usable.
        for index, raw in loopback_raw.items():
            if index not in devices:
                devices[index] = _to_device(
                    raw, loopback_raw, default_in, default_out_name
                )

        return sorted(devices.values(), key=lambda d: d.index)
    finally:
        pa.terminate()


def _to_device(
    raw: Any,
    loopback_raw: dict[int, Any],
    default_in: int,
    default_out_name: str,
) -> DeviceInfo:
    index = int(raw["index"])
    is_loopback = index in loopback_raw
    if is_loopback:
        # The loopback that belongs to the endpoint the user is listening on.
        is_default = bool(default_out_name and default_out_name in str(raw["name"]))
    else:
        is_default = index == default_in
    return DeviceInfo(
        index=index,
        name=str(raw["name"]),
        max_input_channels=int(raw["maxInputChannels"]),
        default_sample_rate=int(raw["defaultSampleRate"]),
        is_loopback=is_loopback,
        is_default=is_default,
    )


def default_endpoint_indices() -> tuple[int, int]:
    """``(default capture index, default render index)`` as WASAPI reports them.

    Polled during capture: when the render index changes the user has plugged in
    headphones or switched output, and the loopback stream must be reopened
    against the new endpoint.
    """
    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        return (
            int(wasapi.get("defaultInputDevice", -1)),
            int(wasapi.get("defaultOutputDevice", -1)),
        )
    finally:
        pa.terminate()


def resolve_loopback(devices: list[DeviceInfo] | None = None) -> DeviceInfo:
    """The loopback endpoint that carries what the user is currently hearing."""
    devices = devices if devices is not None else list_devices()
    loopbacks = [d for d in devices if d.is_loopback]
    if not loopbacks:
        raise DeviceNotFoundError(
            "no WASAPI loopback endpoint exposed",
            user_message=(
                "No se encontró ningún endpoint de loopback WASAPI. Sin él no se "
                "puede capturar lo que dicen los participantes remotos."
            ),
        )
    for d in loopbacks:
        if d.is_default:
            return d
    return loopbacks[0]


def resolve_microphone(devices: list[DeviceInfo] | None = None) -> DeviceInfo:
    """The default capture endpoint: the user's own voice."""
    devices = devices if devices is not None else list_devices()
    mics = [d for d in devices if not d.is_loopback]
    if not mics:
        raise DeviceNotFoundError(
            "no WASAPI capture endpoint available",
            user_message="No se encontró ningún micrófono disponible.",
        )
    for d in mics:
        if d.is_default:
            return d
    return mics[0]


def pick_device(
    spec: str | int | None, track: TrackKind, devices: list[DeviceInfo] | None = None
) -> DeviceInfo:
    """Resolve a user-supplied device selector.

    Accepts a device index or a case-insensitive substring of the name; ``None``
    means "the sensible default for this track". Ambiguous substrings are an error
    rather than a silent first-match, because picking the wrong endpoint yields a
    plausible-looking session with the wrong half of the conversation in it.
    """
    devices = devices if devices is not None else list_devices()
    if spec is None or spec == "":
        return (
            resolve_loopback(devices)
            if track is TrackKind.LOOPBACK
            else resolve_microphone(devices)
        )

    if isinstance(spec, int) or (isinstance(spec, str) and spec.lstrip("-").isdigit()):
        index = int(spec)
        for d in devices:
            if d.index == index:
                return d
        raise DeviceNotFoundError(
            f"no device with index {index}",
            user_message=f"No existe un dispositivo con índice {index}.",
        )

    needle = str(spec).lower()
    matches = [d for d in devices if needle in d.name.lower()]
    if not matches:
        raise DeviceNotFoundError(
            f"no device matching {spec!r}",
            user_message=f"Ningún dispositivo coincide con '{spec}'.",
        )
    if len(matches) > 1:
        names = ", ".join(f"[{d.index}] {d.name}" for d in matches[:5])
        raise DeviceNotFoundError(
            f"ambiguous device spec {spec!r}: {names}",
            user_message=(
                f"'{spec}' coincide con varios dispositivos ({names}). "
                "Usa el índice numérico."
            ),
        )
    return matches[0]
