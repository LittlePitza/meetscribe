"""The event bus: pipeline threads to browser, without either knowing about the other.

Three properties, each chosen against a specific failure:

**Publishing never blocks.** A producer here is a capture or transcription thread.
If a browser tab that is paused in a background window could apply backpressure to
the transcriber, the user's meeting would be damaged by their window manager. Each
subscriber therefore gets its own bounded queue, and a subscriber that cannot keep
up is *disconnected* rather than waited for.

**Disconnection is not data loss.** Every event carries a monotonic ``seq``, and
the bus keeps a replay ring buffer. A dropped or refreshed browser reconnects with
its last cursor -- SSE does this natively via ``Last-Event-ID`` -- and receives
exactly what it missed. So the panel survives a refresh, a laptop sleeping, and a
tab being throttled, without the pipeline ever knowing.

**Volatile events are not replayed.** Level meters and heartbeats describe an
instant, not a fact. Replaying two hours of meter ticks after a refresh would bury
the transcript and could exceed the buffer on its own, so they are published live
and never stored.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

from meetscribe.core.events import Event

log = logging.getLogger(__name__)

__all__ = ["InMemoryEventBus", "BusSubscription"]

_CLOSED = object()


class BusSubscription:
    """One consumer's view of the stream. Iterating yields events until closed."""

    __slots__ = ("_queue", "_bus", "_closed", "cursor", "dropped")

    def __init__(self, bus: InMemoryEventBus, maxsize: int, cursor: int) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._bus = bus
        self._closed = False
        self.cursor = cursor
        self.dropped = False

    def offer(self, event: Event) -> bool:
        """Called by the bus under its lock. False means "this consumer is stuck"."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self.dropped = True
            return False

    def __iter__(self) -> Iterator[Event]:
        while True:
            item = self._queue.get()
            if item is _CLOSED:
                return
            event = item
            self.cursor = event.seq
            yield event

    def get(self, timeout: float | None = None) -> Event | None:
        """Blocking single read, for transports that manage their own loop."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _CLOSED:
            return None
        self.cursor = item.seq
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus.unsubscribe(self)
        # A full queue means the reader is already gone; nothing to wake.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_CLOSED)

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> BusSubscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InMemoryEventBus:
    """Thread-safe fan-out with a bounded replay buffer."""

    def __init__(self, history_limit: int = 5000, subscriber_queue: int = 2000) -> None:
        self._history: deque[Event] = deque(maxlen=history_limit)
        self._subscribers: list[BusSubscription] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._subscriber_queue = subscriber_queue
        self._closed = False
        self._clock_source = None
        self.published = 0
        self.disconnected = 0

    def set_time_source(self, fn) -> None:
        """Supply session-relative time. Without it, ``at`` is 0."""
        self._clock_source = fn

    def publish(self, event_type: str, data: Mapping[str, Any]) -> Event:
        at = 0.0
        if self._clock_source is not None:
            try:
                at = float(self._clock_source())
            except Exception:
                at = 0.0

        with self._lock:
            if self._closed:
                return Event(seq=self._seq, type=event_type, at=at, data=dict(data))
            self._seq += 1
            event = Event(seq=self._seq, type=event_type, at=at, data=dict(data))
            self.published += 1
            if not event.is_volatile:
                self._history.append(event)
            stuck = [s for s in self._subscribers if not s.offer(event)]
            for subscription in stuck:
                self._subscribers.remove(subscription)
                self.disconnected += 1
        for subscription in stuck:
            log.info("subscriber fell behind at seq=%d; disconnecting", event.seq)
            subscription.close()
        return event

    def history(self, after_seq: int = 0, limit: int = 1000) -> list[Event]:
        with self._lock:
            return [e for e in self._history if e.seq > after_seq][-limit:]

    def subscribe(self, after_seq: int = 0) -> BusSubscription:
        """Attach a consumer, pre-loaded with whatever it missed.

        Replay happens inside the lock so no event can slip between the history
        snapshot and the subscription going live -- the seam that would otherwise
        drop exactly one transcript line per reconnect.
        """
        subscription = BusSubscription(self, self._subscriber_queue, after_seq)
        with self._lock:
            for event in self._history:
                if event.seq > after_seq and not subscription.offer(event):
                    break
            if not self._closed:
                self._subscribers.append(subscription)
        return subscription

    def unsubscribe(self, subscription: BusSubscription) -> None:
        with self._lock:
            if subscription in self._subscribers:
                self._subscribers.remove(subscription)

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscription in subscribers:
            subscription.close()
