"""Redaction: for logs, and optionally for text leaving the machine.

Two distinct jobs share one module because they share the patterns.

**Log redaction is structural, not disciplinary.** "Remember not to log the API
key" is not a control -- it fails the first time someone adds a debug line during
an incident. Instead a :class:`RedactingFilter` is installed on the root logger, so
a secret cannot reach a log file even if some code path formats it into a message.
The filter rewrites the message *and* the arguments, since ``log.info("key=%s",
key)`` puts the secret in ``record.args``.

**PII redaction is opt-in and applies before egress.** When the Claude adapter is
in use, transcript text leaves the machine. For users who want the quality of a
frontier model on a meeting that mentions customer data, this strips the obvious
direct identifiers first. It is deliberately conservative: emails, phone numbers,
long digit runs, and Spanish national IDs. It is not a guarantee -- no regex
understands that "the patient in room 4" is identifying -- and the docs say so.
The honest control for genuinely sensitive meetings is offline-only mode.
"""

from __future__ import annotations

import logging
import re

__all__ = ["RedactingFilter", "redact_secrets", "redact_pii", "install_log_redaction"]


_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Anthropic keys have a stable, distinctive prefix.
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "sk-ant-[REDACTADO]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-[REDACTADO]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}"), "hf_[REDACTADO]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE), "Bearer [REDACTADO]"),
    # The panel's own session token, whenever a URL with it is logged.
    (re.compile(r"([?&]t=)[A-Za-z0-9_\-]{16,}"), r"\1[REDACTADO]"),
    # Generic "key/token/secret/password = value" assignments in any format.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)"
            r"(\s*[:=]\s*)(\"|')?([^\s\"',;]{8,})"
        ),
        r"\1\2\3[REDACTADO]",
    ),
)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)\d{9,}(?!\d)")
_SPANISH_ID = re.compile(r"\b\d{7,8}[- ]?[A-HJ-NP-TV-Z]\b", re.IGNORECASE)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}\b")


def redact_secrets(text: str) -> str:
    """Remove anything that looks like a credential."""
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_pii(text: str, *, phones: bool = True) -> str:
    """Strip direct identifiers before text leaves the machine.

    ``phones=False`` exists because the phone pattern is the greediest one here and
    can swallow dates and figures in a meeting full of numbers.
    """
    if not text:
        return text
    text = _EMAIL.sub("[correo]", text)
    text = _IBAN.sub("[iban]", text)
    text = _SPANISH_ID.sub("[documento]", text)
    # Long digit runs are matched first: the phone pattern is the greediest one
    # here and would otherwise swallow account and case numbers.
    text = _LONG_DIGITS.sub("[número]", text)
    if phones:
        text = _PHONE.sub("[teléfono]", text)
    return text


class RedactingFilter(logging.Filter):
    """Scrubs secrets from every record that passes through a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_secrets(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact_secrets(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def install_log_redaction(logger: logging.Logger | None = None) -> RedactingFilter:
    """Attach the filter to a logger and every handler already on it.

    Both, on purpose: a filter on the logger does not apply to records that
    propagate up from child loggers, while a filter on the handlers catches those.
    """
    target = logger or logging.getLogger()
    redactor = RedactingFilter()
    target.addFilter(redactor)
    for handler in target.handlers:
        handler.addFilter(redactor)
    return redactor
