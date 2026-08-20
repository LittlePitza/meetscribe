"""Signal level, and the watchdog that catches a track producing nothing.

The spike found the microphone array opening successfully and returning **digital
silence** -- peak 0.0000, RMS -240 dBFS. No exception, no warning, just an empty
half of the transcript discovered an hour later. That failure mode is the reason
this module exists and the reason the panel shows a level meter per track: a dead
input has to be visible in the first ten seconds, not the first hour.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DIGITAL_SILENCE_DBFS",
    "rms_dbfs",
    "peak_level",
    "SilenceWatchdog",
    "WatchdogVerdict",
]

DIGITAL_SILENCE_DBFS = -80.0
"""Below this, a track is not "quiet", it is disconnected.

A real room with a muted speaker still floors around -65 to -55 dBFS of preamp and
fan noise. Genuine digital zero reads as the floor of the log, around -240. The
threshold sits between them with a wide margin on both sides, so a genuinely quiet
office does not trip it and a dead device cannot avoid it.
"""


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS level in dBFS. Returns the log floor for an empty or silent buffer."""
    if samples.size == 0:
        return -240.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def peak_level(samples: np.ndarray) -> float:
    """Absolute peak in [0, 1]."""
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


class WatchdogVerdict:
    """What :meth:`SilenceWatchdog.observe` decided about this instant."""

    NONE = "none"
    """Nothing to report."""

    WENT_SILENT = "went_silent"
    """The track just crossed the threshold. Warn the user, once."""

    RECOVERED = "recovered"
    """Signal came back. Clear the warning."""


class SilenceWatchdog:
    """Edge-triggered detector for a track stuck below the noise floor.

    Edge-triggered on purpose: a level-triggered check would emit a warning on
    every audio chunk for the rest of the meeting, which trains the user to ignore
    warnings. This fires once on the way down and once on the way back up.

    Pure and clock-injected -- a test covering the 12-second window runs instantly.
    """

    __slots__ = (
        "threshold_dbfs",
        "window_s",
        "recover_s",
        "_silent_since",
        "_loud_since",
        "_alerted",
    )

    def __init__(
        self,
        threshold_dbfs: float = DIGITAL_SILENCE_DBFS,
        window_s: float = 12.0,
        recover_s: float = 1.0,
    ) -> None:
        self.threshold_dbfs = threshold_dbfs
        self.window_s = window_s
        self.recover_s = recover_s
        self._silent_since: float | None = None
        self._loud_since: float | None = None
        self._alerted = False

    @property
    def alerted(self) -> bool:
        return self._alerted

    def observe(self, dbfs: float, at_s: float) -> str:
        """Feed one level reading; returns a :class:`WatchdogVerdict` constant."""
        if dbfs <= self.threshold_dbfs:
            self._loud_since = None
            if self._silent_since is None:
                self._silent_since = at_s
            elif not self._alerted and at_s - self._silent_since >= self.window_s:
                self._alerted = True
                return WatchdogVerdict.WENT_SILENT
            return WatchdogVerdict.NONE

        self._silent_since = None
        if self._loud_since is None:
            self._loud_since = at_s
        # Require sustained signal before clearing, so one burst of interface
        # noise on a dead device does not cancel a correct warning.
        if self._alerted and at_s - self._loud_since >= self.recover_s:
            self._alerted = False
            self._loud_since = None
            return WatchdogVerdict.RECOVERED
        return WatchdogVerdict.NONE

    def reset(self) -> None:
        self._silent_since = None
        self._loud_since = None
        self._alerted = False
