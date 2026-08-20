"""Clock implementations.

The system clock is a dependency like any other. Injecting it is what makes the
timeline logic, the silent-track watchdog and the backpressure policy testable
without ``time.sleep`` in the test suite.
"""

from __future__ import annotations

import datetime as dt
import threading
import time

__all__ = ["SystemClock", "FakeClock"]


class SystemClock:
    """The real clock."""

    __slots__ = ()

    def monotonic(self) -> float:
        return time.perf_counter()

    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """A clock the test drives by hand.

    ``sleep`` advances time instead of blocking, so a test covering a 30-minute
    watchdog window runs in microseconds. Thread-safe, because several pipeline
    threads read it concurrently in the integration tests.
    """

    __slots__ = ("_t", "_epoch", "_lock", "sleeps")

    def __init__(
        self, start: float = 0.0, epoch: dt.datetime | None = None
    ) -> None:
        self._t = start
        self._epoch = epoch or dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        with self._lock:
            return self._t

    def now(self) -> dt.datetime:
        with self._lock:
            return self._epoch + dt.timedelta(seconds=self._t)

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self._t += max(0.0, seconds)

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._t += seconds
            return self._t
