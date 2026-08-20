"""Model files: where they live, and proving they are what they claim to be.

Two ONNX files are needed at runtime: Silero VAD (0.6 MB) and NeMo TitaNet-small
(38 MB). Whisper weights are handled by faster-whisper's own Hugging Face cache and
are not managed here.

**Every downloaded file is verified against a pinned SHA-256, and a mismatch is
fatal.** An ONNX file is not data, it is a program the runtime will execute, and a
model file that does not match its hash means either a corrupted download or a
compromised mirror. There is no benign third case, so there is no "continue anyway"
path -- the file is deleted and the error explains what happened.

The hashes were computed from the exact files the spikes measured, so the
reproducibility claim in ``docs/benchmarks.md`` holds byte for byte.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from meetscribe.errors import ModelIntegrityError, ModelNotFoundError
from meetscribe.security.paths import models_dir

log = logging.getLogger(__name__)

__all__ = [
    "ModelAsset",
    "VAD_MODEL",
    "SPEAKER_MODEL",
    "REQUIRED_MODELS",
    "resolve_model",
    "install_model",
    "sha256_file",
    "verify_model",
]


@dataclass(frozen=True, slots=True)
class ModelAsset:
    name: str
    filename: str
    sha256: str
    size_bytes: int
    url: str
    purpose: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


VAD_MODEL = ModelAsset(
    name="silero-vad",
    filename="silero_vad.onnx",
    sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    size_bytes=643_854,
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    purpose="detección de voz (VAD)",
)

SPEAKER_MODEL = ModelAsset(
    name="titanet-small",
    filename="nemo_en_titanet_small.onnx",
    sha256="ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    size_bytes=40_257_283,
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/nemo_en_titanet_small.onnx"
    ),
    purpose="identificación de hablantes (diarización)",
)

REQUIRED_MODELS: tuple[ModelAsset, ...] = (VAD_MODEL, SPEAKER_MODEL)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def search_paths() -> list[Path]:
    """Where a model may already be, most authoritative first.

    Only absolute locations appear here. An entry relative to the current working
    directory would make model resolution depend on where the command was typed,
    which is the kind of difference that only shows up on someone else's machine.
    ``MEETSCRIBE_MODELS_DIR`` is the supported way to point at a shared cache.
    """
    paths = [models_dir()]
    override = os.environ.get("MEETSCRIBE_MODELS_DIR")
    if override:
        paths.insert(0, Path(override).expanduser())
    return paths


def find_model(asset: ModelAsset) -> Path | None:
    for directory in search_paths():
        candidate = directory / asset.filename
        if candidate.is_file():
            return candidate
    return None


def verify_model(asset: ModelAsset, path: Path) -> None:
    """Raise unless the file matches the pinned digest exactly."""
    actual = sha256_file(path)
    if actual != asset.sha256:
        raise ModelIntegrityError(
            f"{path} sha256 {actual} != expected {asset.sha256}",
            user_message=(
                f"El archivo '{asset.filename}' no coincide con su huella SHA-256 "
                "esperada. Se rechazó por seguridad: un modelo ONNX es código que "
                "se ejecuta. Bórralo y vuelve a descargarlo con "
                "'meetscribe models --download'."
            ),
        )


def resolve_model(asset: ModelAsset, verify: bool = True) -> Path:
    """Locate a model, verifying it. Raises with instructions when absent."""
    path = find_model(asset)
    if path is None:
        raise ModelNotFoundError(
            f"{asset.filename} not found in {[str(p) for p in search_paths()]}",
            user_message=(
                f"Falta el modelo de {asset.purpose} ({asset.filename}, "
                f"{asset.size_mb:.0f} MB). Ejecuta 'meetscribe models --download'."
            ),
        )
    if verify:
        verify_model(asset, path)
    return path


def install_model(
    asset: ModelAsset, progress=None, timeout_s: float = 300.0
) -> tuple[Path, str]:
    """Put a verified copy in the data directory. Returns ``(path, how)``.

    Prefers an existing verified copy elsewhere on the machine over the network:
    it is faster, it works offline, and it avoids re-downloading 38 MB because the
    file happened to be in the wrong directory.
    """
    target_dir = models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / asset.filename

    if target.is_file():
        try:
            verify_model(asset, target)
            return (target, "ya estaba instalado")
        except ModelIntegrityError:
            log.warning("%s failed verification; replacing", target)
            target.unlink()

    for directory in search_paths():
        candidate = directory / asset.filename
        if candidate.is_file() and candidate != target:
            if sha256_file(candidate) == asset.sha256:
                shutil.copy2(candidate, target)
                return (target, f"copiado de {candidate.parent}")
            log.warning("local copy at %s has an unexpected digest; ignoring", candidate)

    return (_download(asset, target, progress, timeout_s), "descargado")


def _download(
    asset: ModelAsset, target: Path, progress, timeout_s: float
) -> Path:
    import httpx

    temporary = target.with_suffix(target.suffix + ".part")
    log.info("downloading %s from %s", asset.filename, asset.url)
    try:
        with (
            httpx.Client(timeout=timeout_s, follow_redirects=True) as client,
            client.stream("GET", asset.url) as response,
        ):
                response.raise_for_status()
                total = int(response.headers.get("Content-Length") or asset.size_bytes)
                done = 0
                with temporary.open("wb") as handle:
                    for block in response.iter_bytes(1 << 16):
                        handle.write(block)
                        done += len(block)
                        if progress is not None:
                            progress(done, total)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ModelNotFoundError(
            f"download of {asset.filename} failed: {type(exc).__name__}: {exc}",
            user_message=(
                f"No se pudo descargar '{asset.filename}' desde {asset.url}. "
                "Comprueba la conexión, o copia el archivo manualmente a "
                f"'{models_dir()}'."
            ),
        ) from exc

    try:
        verify_model(asset, temporary)
    except ModelIntegrityError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)
    return target
