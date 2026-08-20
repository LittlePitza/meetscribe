"""The bounded buffer between a capture thread and its segmenter.

**The policy, stated plainly.** The queue holds at most ``max_seconds`` of audio
per track. When it is full the *oldest* chunk is discarded, never the newest.

That direction is the important one. If transcription falls behind during a burst
of dense speech, the useful thing to keep is what is being said now -- a live panel
showing a transcript from ninety seconds ago is worse than useless, because the
user cannot tell it is stale. Dropping the newest audio would also mean the panel
freezes at the exact moment the meeting gets busy.

Nothing is dropped silently. Every discarded sample is counted, the producer feeds
that count to :meth:`TrackTimeline.skip` so later timestamps stay true, and the
session raises a warning notice in the panel. At the measured RTF of 0.32 with a
30-second buffer this should never trigger; if it does, it is a real signal that
the machine is oversubscribed, and it must be visible.

**One case must not drop: file replay.** A WAV source is read as fast as the disk
allows, which floods the queue in milliseconds -- and unlike a live endpoint, a file
can simply be read more slowly. :meth:`put_blocking` waits for room instead. That
is what makes re-transcribing a recording lossless, and it is a real bug fixed
after the first end-to-end run against real audio silently lost the opening eleven
seconds of the meeting.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from meetscribe.core.models import PIPELINE_SAMPLE_RATE, AudioChunk

__all__ = ["ChunkQueue"]


class ChunkQueue:
    """Thread-safe, bounded queue of audio chunks.

    ``put`` never blocks (live capture); ``put_blocking`` waits for room (file
    replay). Which one a producer uses is decided by whether its source can be
    slowed down, not by preference.
    """

    __slots__ = (
        "max_samples",
        "sample_rate",
        "_items",
        "_queued_samples",
        "_lock",
        "_not_empty",
        "_closed",
        "_dropped_samples",
        "_drop_events",
    )

    def __init__(
        self, max_seconds: float = 30.0, sample_rate: int = PIPELINE_SAMPLE_RATE
    ) -> None:
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self._items: deque[AudioChunk] = deque()
        self._queued_samples = 0
        self._lock = threading.Lock()
        # One condition for both "something arrived" and "room appeared"; the rate
        # is a few dozen notifications per second, so the extra wakeups are free.
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self._dropped_samples = 0
        self._drop_events = 0

    # ---------------------------------------------------------------- producer

    def put(self, chunk: AudioChunk) -> int:
        """Enqueue a chunk. Returns how many samples had to be dropped to fit it.

        Never blocks and never raises on a full queue: a blocked capture thread
        means a WASAPI buffer overrun, which loses audio anyway *and* corrupts the
        device state. Dropping deliberately is strictly better than dropping by
        accident.
        """
        n = int(chunk.samples.size)
        if n == 0:
            return 0
        dropped = 0
        with self._not_empty:
            if self._closed:
                return 0
            self._items.append(chunk)
            self._queued_samples += n
            while self._queued_samples > self.max_samples and len(self._items) > 1:
                old = self._items.popleft()
                freed = int(old.samples.size)
                self._queued_samples -= freed
                dropped += freed
            if dropped:
                self._dropped_samples += dropped
                self._drop_events += 1
            self._not_empty.notify()
        return dropped

    def put_blocking(
        self,
        chunk: AudioChunk,
        should_stop: Callable[[], bool] | None = None,
        poll: float = 0.2,
    ) -> bool:
        """Enqueue, waiting for room. Returns False only when told to stop.

        For finite sources only. Calling this from a live capture thread would
        block a WASAPI read and cause the driver-level overrun this class exists
        to avoid.
        """
        n = int(chunk.samples.size)
        if n == 0:
            return True
        while True:
            with self._not_empty:
                if self._closed:
                    return False
                if self._queued_samples + n <= self.max_samples or not self._items:
                    self._items.append(chunk)
                    self._queued_samples += n
                    self._not_empty.notify_all()
                    return True
                self._not_empty.wait(poll)
            if should_stop is not None and should_stop():
                return False

    # ---------------------------------------------------------------- consumer

    def get(self, timeout: float | None = 0.5) -> AudioChunk | None:
        """Pop the oldest chunk, waiting up to ``timeout``.

        Returns ``None`` on timeout or once the queue is closed and drained, which
        is the consumer thread's signal to check for shutdown and loop.
        """
        with self._not_empty:
            if not self._items and not self._closed:
                self._not_empty.wait(timeout)
            if not self._items:
                return None
            chunk = self._items.popleft()
            self._queued_samples -= int(chunk.samples.size)
            self._not_empty.notify_all()  # a blocked producer may now have room
            return chunk

    def close(self) -> None:
        """Stop accepting input and wake every waiting consumer."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    # ------------------------------------------------------------------ status

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def queued_seconds(self) -> float:
        with self._lock:
            return self._queued_samples / self.sample_rate

    @property
    def dropped_samples(self) -> int:
        with self._lock:
            return self._dropped_samples

    @property
    def drop_events(self) -> int:
        with self._lock:
            return self._drop_events

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
