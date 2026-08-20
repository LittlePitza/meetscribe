"""Defences against Whisper inventing speech that was never said.

This is not a theoretical concern. Whisper's decoder is autoregressive and was
trained on subtitle corpora, so when it is handed audio with little or no speech
in it -- a hold tone, background music, a cough, room noise -- it falls back on
what subtitle files usually contain. In Spanish the signature outputs are
"Subtitulos realizados por la comunidad de Amara.org", "Gracias por ver el video"
and long runs of one repeated phrase.

Silently writing those into a meeting transcript is worse than dropping the
segment: the user has no way to tell the invented line from a real one, and a
summariser downstream will happily treat it as content.

Four independent filters, because each catches a different failure:

1. **``no_speech_prob``** -- the model's own estimate that the window contains no
   speech. The single most reliable signal.
2. **``avg_logprob``** -- overall decoder confidence. Low means it was guessing.
3. **``compression_ratio``** -- the ratio of raw text length to its zlib-compressed
   length. Repetition loops compress extremely well, so a high ratio is a direct
   measure of "the model got stuck".
4. **A blocklist plus repetition detection** on the text itself, which catches the
   known phrases and the loops that slipped past the numeric thresholds.

Every rejection is logged with its reason. A filter that silently eats real speech
would be its own kind of failure, so the reasons are counted and surfaced by
``meetscribe doctor``.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

from meetscribe.core.models import TranscriptionResult, normalise_for_dedup

__all__ = [
    "HallucinationFilter",
    "FilterVerdict",
    "BLOCKLIST_PATTERNS",
    "compression_ratio",
]


BLOCKLIST_PATTERNS: tuple[str, ...] = (
    # Subtitle-corpus boilerplate, Spanish
    r"subt[ií]tulos?\s+(realizados?|creados?|por)",
    r"amara\.org",
    r"subtitulado\s+por",
    r"traducci[oó]n\s+(y\s+)?subt[ií]tulos",
    r"gracias\s+por\s+ver\s+(el|este)\s+(video|v[ií]deo)",
    r"no\s+olvides\s+suscribirte",
    r"suscr[ií]bete\s+al\s+canal",
    r"m[aá]s\s+informaci[oó]n\s+en\s+www",
    # English equivalents, for code-switched audio
    r"subtitles?\s+by",
    r"thanks?\s+for\s+watching",
    r"please\s+subscribe",
    r"transcription\s+by\s+castingwords",
    # Bare URLs and channel handles that only ever arrive as hallucination
    r"www\.(youtube|amara)\.",
    r"^\s*[\W_]*\s*$",  # punctuation-only output
)

_BLOCKLIST = tuple(re.compile(p, re.IGNORECASE) for p in BLOCKLIST_PATTERNS)


def compression_ratio(text: str) -> float:
    """Length of the UTF-8 text divided by the length of its zlib compression.

    Normal speech lands around 1.2-2.0. A repetition loop -- the same clause
    emitted forty times -- compresses far better and shows up above 3.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / max(1, len(zlib.compress(raw, 6)))


@dataclass(frozen=True, slots=True)
class FilterVerdict:
    """Outcome of filtering one transcription."""

    accepted: bool
    text: str
    reason: str = ""

    @property
    def rejected(self) -> bool:
        return not self.accepted


class HallucinationFilter:
    """Stateful across a session, so it can also suppress immediate repeats.

    The thresholds are conservative on purpose. A false reject costs one short
    utterance -- annoying, and visible as a gap in the timeline. A false accept
    puts a sentence nobody said into the minutes, which is the failure that
    destroys trust in the whole tool.
    """

    __slots__ = (
        "max_no_speech_prob",
        "min_avg_logprob",
        "max_compression_ratio",
        "max_repeats",
        "min_chars",
        "_last_text",
        "_repeat_count",
        "rejections",
    )

    def __init__(
        self,
        max_no_speech_prob: float = 0.6,
        min_avg_logprob: float = -1.0,
        max_compression_ratio: float = 2.6,
        max_repeats: int = 3,
        min_chars: int = 2,
    ) -> None:
        self.max_no_speech_prob = max_no_speech_prob
        self.min_avg_logprob = min_avg_logprob
        self.max_compression_ratio = max_compression_ratio
        self.max_repeats = max_repeats
        self.min_chars = min_chars
        self._last_text = ""
        self._repeat_count = 0
        self.rejections: dict[str, int] = {}

    def check(self, result: TranscriptionResult) -> FilterVerdict:
        text = _tidy(result.text)

        if len(text) < self.min_chars:
            return self._reject("empty", text)

        if (
            result.no_speech_prob is not None
            and result.no_speech_prob > self.max_no_speech_prob
        ):
            return self._reject("no_speech_prob", text)

        if (
            result.avg_logprob is not None
            and result.avg_logprob < self.min_avg_logprob
        ):
            return self._reject("low_confidence", text)

        # Measured on the raw output, never on the tidied text: trimming a
        # repetition loop would otherwise hide that loop from its own detector.
        ratio = (
            result.compression_ratio
            if result.compression_ratio is not None
            else compression_ratio(result.text)
        )
        if ratio > self.max_compression_ratio:
            return self._reject("repetition_loop", text)

        for pattern in _BLOCKLIST:
            if pattern.search(text):
                return self._reject("blocklist", text)

        if _has_ngram_loop(text):
            return self._reject("ngram_loop", text)

        normalised = normalise_for_dedup(text)
        if normalised and normalised == self._last_text:
            self._repeat_count += 1
            if self._repeat_count >= self.max_repeats:
                return self._reject("consecutive_repeat", text)
        else:
            self._last_text = normalised
            self._repeat_count = 0

        return FilterVerdict(accepted=True, text=text)

    def _reject(self, reason: str, text: str) -> FilterVerdict:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return FilterVerdict(accepted=False, text=text, reason=reason)

    def reset(self) -> None:
        self._last_text = ""
        self._repeat_count = 0


_WS = re.compile(r"\s+")


def _tidy(text: str) -> str:
    """Trim, collapse whitespace, and drop a trailing run of identical sentences.

    Whisper often ends a hallucinated segment by repeating its last sentence a few
    times. Removing the tail keeps the genuine opening of the segment instead of
    discarding the whole thing.
    """
    text = _WS.sub(" ", text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?…])\s+", text)
    if len(parts) < 3:
        return text
    kept: list[str] = []
    previous = ""
    for part in parts:
        key = normalise_for_dedup(part)
        if key and key == previous:
            continue
        previous = key
        kept.append(part)
    return " ".join(kept).strip()


def _has_ngram_loop(text: str, n: int = 4, threshold: int = 4) -> bool:
    """True when one n-gram of words repeats more than ``threshold`` times.

    Catches the loops that are long enough to defeat the compression heuristic --
    "y entonces vamos a ver, y entonces vamos a ver, ..." -- without flagging the
    genuinely repetitive short phrases that occur in speech ("si, si, si").
    """
    words = normalise_for_dedup(text).split()
    if len(words) < n * (threshold + 1):
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] > threshold:
            return True
    return False
