"""API key resolution and the local panel's session token.

**Resolution order for the API key**, first match wins:

1. ``ANTHROPIC_API_KEY`` in the environment -- what CI and one-off runs use.
2. The Windows Credential Manager, via ``keyring``. This is the recommended place:
   the value is stored by the OS under the user's account, encrypted with their
   credentials, and never appears in a file the user might paste into an issue.
3. ``api_key`` in the config file -- supported because people will do it anyway,
   but the file's permissions are checked first and a warning is emitted, so
   nobody stores a key in a world-readable file without being told.

The key is never returned as part of any config dump, never written to the
database, and cannot reach a log file (see :mod:`meetscribe.security.redact`).

**The session token** protects the local panel. Binding to 127.0.0.1 stops the
network, but it does not stop *another user account on the same machine* from
opening the port and reading a live transcript. A 256-bit token in the URL closes
that, and every comparison uses ``secrets.compare_digest`` so timing does not leak
it.
"""

from __future__ import annotations

import logging
import os
import secrets
import stat
import sys
from pathlib import Path

from meetscribe.errors import SecretNotFoundError

log = logging.getLogger(__name__)

__all__ = [
    "KEYRING_SERVICE",
    "resolve_api_key",
    "store_api_key",
    "delete_api_key",
    "keyring_available",
    "new_session_token",
    "token_matches",
    "check_file_permissions",
]

KEYRING_SERVICE = "meetscribe"

ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


def keyring_available() -> bool:
    try:
        import keyring

        backend = keyring.get_keyring()
        return backend is not None and "fail" not in type(backend).__name__.lower()
    except Exception:
        return False


def resolve_api_key(
    provider: str = "anthropic",
    config_value: str | None = None,
    config_file: Path | None = None,
    required: bool = True,
) -> str | None:
    """Find the key, or explain precisely how to provide one."""
    env_var = ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
    value = os.environ.get(env_var, "").strip()
    if value:
        log.debug("api key for %s resolved from %s", provider, env_var)
        return value

    if keyring_available():
        try:
            import keyring

            stored = keyring.get_password(KEYRING_SERVICE, provider)
            if stored:
                log.debug("api key for %s resolved from the OS keyring", provider)
                return stored.strip()
        except Exception as exc:
            log.warning("keyring lookup failed for %s: %s", provider, exc)

    if config_value:
        if config_file is not None:
            problem = check_file_permissions(config_file)
            if problem:
                log.warning(
                    "la clave de API está en %s y %s; muévela al almacén del "
                    "sistema con 'meetscribe config set-key'",
                    config_file,
                    problem,
                )
        return config_value.strip()

    if not required:
        return None
    raise SecretNotFoundError(
        f"no api key for provider {provider!r}",
        user_message=(
            f"No hay clave de API para '{provider}'. Guárdala de forma segura con "
            f"'meetscribe config set-key --provider {provider}', o exporta "
            f"{env_var}."
        ),
    )


def store_api_key(provider: str, key: str) -> str:
    """Write the key to the OS credential store. Returns where it landed."""
    key = key.strip()
    if not key:
        raise SecretNotFoundError(
            "refusing to store an empty key",
            user_message="La clave está vacía; no se guardó nada.",
        )
    if not keyring_available():
        raise SecretNotFoundError(
            "no keyring backend available",
            user_message=(
                "No hay un almacén de credenciales disponible en este sistema. "
                f"Exporta {ENV_VARS.get(provider, provider.upper() + '_API_KEY')} "
                "en su lugar."
            ),
        )
    import keyring

    keyring.set_password(KEYRING_SERVICE, provider, key)
    return "el almacén de credenciales de Windows"


def delete_api_key(provider: str) -> bool:
    if not keyring_available():
        return False
    try:
        import keyring

        keyring.delete_password(KEYRING_SERVICE, provider)
        return True
    except Exception:
        return False


def check_file_permissions(path: Path) -> str | None:
    """Return a Spanish description of the problem, or ``None`` when acceptable."""
    if not path.exists():
        return None
    if sys.platform == "win32":
        # Windows ACLs are checked at the directory level in
        # meetscribe.security.paths; a per-file check here would duplicate it
        # without adding information.
        return None
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        return "es legible por otros usuarios"
    return None


def new_session_token() -> str:
    """256 bits of entropy, URL-safe."""
    return secrets.token_urlsafe(32)


def token_matches(expected: str, provided: str | None) -> bool:
    """Constant-time comparison. Never short-circuit on a secret."""
    if not expected or not provided:
        return False
    return secrets.compare_digest(expected, provided)
