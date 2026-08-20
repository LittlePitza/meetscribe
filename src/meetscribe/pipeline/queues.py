"""The queue between segmentation and transcription, and its overflow policy.

This is where backpressure actually bites. Capture is cheap and constant;
transcription is the expensive stage, and it is the one that can fall behind if the
machine is oversubscribed -- a Teams screen share, a build running, thermal
throttling.

**The policy: drafts are sacrificed before finals, and supersession is free.**

* A draft is provisional text for an utterance still in progress. If the utterance
  has since finished, its final version makes the draft worthless -- so enqueueing a
  final *removes* any pending draft with the same key. That is not just a memory
  saving: it stops the decoder spending CPU on audio whose successor is already
  waiting, which is exactly the wrong thing to do while behind.
* When the queue is genuinely full, the oldest **draft** is dropped first. Losing a
  draft costs nothing permanent; the final still arrives.
* Only when no drafts remain is a final dropped, and that is counted and reported
  as a warning, because it is a real hole in the transcript.

At the measured RTF of 0.32 with a 32-slot queue this should never engage. It exists
because "should never" is not a guarantee, and because the alternative -- unbounded
growth -- turns a slow machine into a two-hour memory leak.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable

from meetscribe.core.models import Utterance

log = logging.getLogger(__name__)

__all__ = ["UtteranceQueue"]


class UtteranceQueue:
    """Bounded, thread-safe, with draft supersession and a stated drop order."""

    def __init__(self, maxsize: int = 32) -> None:
        if maxsize < 2:
            raise ValueError("maxsize must be at least 2")
        self.maxsize = maxsize
        self._items: deque[Utterance] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.dropped_drafts = 0
        self.dropped_finals = 0
        self.superseded = 0

    def put(self, utterance: Utterance) -> str:
        """Enqueue. Returns 'ok', 'dropped_draft' or 'dropped_final'."""
        outcome = "ok"
        with self._not_empty:
            if self._closed:
                return "closed"

            if not utterance.is_partial:
                before = len(self._items)
                self._items = deque(
                    u for u in self._items if not (u.is_partial and u.key == utterance.key)
                )
                removed = before - len(self._items)
                if removed:
                    self.superseded += removed

            while len(self._items) >= self.maxsize:
                victim = self._pick_victim()
                if victim is None:
                    break
                self._items.remove(victim)
                if victim.is_partial:
                    self.dropped_drafts += 1
                    outcome = "dropped_draft"
                else:
                    self.dropped_finals += 1
                    outcome = "dropped_final"
                    log.warning(
                        "utterance queue full: dropped final %s/%d (%.1fs of speech)",
                        victim.track.value,
                        victim.seq,
                        victim.duration_s,
                    )

            self._items.append(utterance)
            self._not_empty.notify()
        return outcome

    def put_blocking(
        self,
        utterance: Utterance,
        should_stop: Callable[[], bool] | None = None,
        poll: float = 0.2,
    ) -> bool:
        """Enqueue, waiting for room. For finite sources only.

        Re-transcribing a recording must never lose an utterance: unlike a live
        meeting, the input can simply be consumed more slowly.
        """
        while True:
            with self._not_empty:
                if self._closed:
                    return False
                if len(self._items) < self.maxsize:
                    self._items.append(utterance)
                    self._not_empty.notify_all()
                    return True
                self._not_empty.wait(poll)
            if should_stop is not None and should_stop():
                return False

    def _pick_victim(self) -> Utterance | None:
        for item in self._items:
            if item.is_partial:
                return item
        return self._items[0] if self._items else None

    def get(self, timeout: float | None = 0.5) -> Utterance | None:
        with self._not_empty:
            if not self._items and not self._closed:
                self._not_empty.wait(timeout)
            if not self._items:
                return None
            item = self._items.popleft()
            self._not_empty.notify_all()  # a blocked producer may now have room
            return item

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "queued": len(self._items),
                "dropped_drafts": self.dropped_drafts,
                "dropped_finals": self.dropped_finals,
                "superseded": self.superseded,
            }
