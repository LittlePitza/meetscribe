"""Turning a transcription into a transcript line.

Small, pure, and heavily used: given an utterance, what the decoder returned and
who was speaking, decide whether this becomes a transcript segment at all, and what
it looks like if so. Keeping it separate from the threading in
:mod:`meetscribe.pipeline.session` means the rules -- when a draft is allowed to
replace a final, which speaker a partial inherits, what the hallucination filter
rejects -- are testable without starting a single thread.
"""

from __future__ import annotations

import logging

from meetscribe.core.models import (
    LOCAL_SPEAKER_ID,
    SegmentState,
    Speaker,
    TrackKind,
    TranscriptionResult,
    TranscriptSegment,
    Utterance,
    new_id,
)
from meetscribe.stt.hallucination import HallucinationFilter

log = logging.getLogger(__name__)

__all__ = ["TranscriptAssembler", "local_speaker"]


def local_speaker(display_name: str = "Yo") -> Speaker:
    """The microphone track. Known with certainty; no clustering involved."""
    return Speaker(
        speaker_id=LOCAL_SPEAKER_ID,
        display_name=display_name,
        track=TrackKind.MIC,
        is_local=True,
    )


class TranscriptAssembler:
    """Applies the filters and builds segments. One instance per session."""

    def __init__(
        self,
        session_id: str,
        hallucination_filter: HallucinationFilter | None = None,
        min_final_chars: int = 2,
    ) -> None:
        self.session_id = session_id
        self.filter = hallucination_filter or HallucinationFilter()
        self.min_final_chars = min_final_chars
        self._finalised: set[tuple[str, int]] = set()
        self.rejected = 0
        self.accepted = 0

    def build(
        self,
        utterance: Utterance,
        result: TranscriptionResult,
        speaker: Speaker,
    ) -> TranscriptSegment | None:
        """``None`` means "this utterance produces no transcript line"."""
        if utterance.is_partial and utterance.key in self._finalised:
            # The final version won the race while this draft sat in the queue.
            # Publishing it now would replace confirmed text with provisional text.
            return None

        verdict = self.filter.check(result)
        if verdict.rejected:
            self.rejected += 1
            log.debug(
                "rejected %s/%d (%s): %r",
                utterance.track.value,
                utterance.seq,
                verdict.reason,
                verdict.text[:80],
            )
            return None

        text = verdict.text
        if not utterance.is_partial and len(text) < self.min_final_chars:
            self.rejected += 1
            return None

        if not utterance.is_partial:
            self._finalised.add(utterance.key)
        self.accepted += 1

        return TranscriptSegment(
            segment_id=new_id(),
            session_id=self.session_id,
            track=utterance.track,
            seq=utterance.seq,
            state=SegmentState.DRAFT if utterance.is_partial else SegmentState.FINAL,
            start_s=utterance.start_s,
            end_s=utterance.end_s,
            text=text,
            speaker_id=speaker.speaker_id,
            speaker_name=speaker.display_name,
            language=result.language,
            no_speech_prob=result.no_speech_prob,
            avg_logprob=result.avg_logprob,
        )

    def is_finalised(self, key: tuple[str, int]) -> bool:
        return key in self._finalised

    def stats(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejections": dict(self.filter.rejections),
        }
