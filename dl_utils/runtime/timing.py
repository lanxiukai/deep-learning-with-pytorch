"""Elapsed-time measurement and duration formatting for runtime operations."""

import time

import numpy as np


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self._paused_elapsed = 0.0
        self._running = False
        self.start()

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self._paused_elapsed = 0.0
        self._running = True

    def stop(self) -> float:
        """Stop the timer and record the accumulated time in a list."""
        if self._running:
            self._paused_elapsed += time.time() - self.start_time
            self._running = False
        self.times.append(self._paused_elapsed)
        self._paused_elapsed = 0.0
        return self.times[-1]

    def avg(self) -> float:
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self) -> float:
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the cumulative time."""
        return np.array(self.times).cumsum().tolist()

    def format_time(self, seconds: float | None = None, precision: int = 1) -> str:
        """Format given seconds or the accumulated time using ``_time_str``."""
        total = self.sum() if seconds is None else seconds
        return _time_str(total, precision=precision)


def _time_str(seconds: float, precision: int = 1) -> str:
    """Return a formatted string given seconds in non-zero units format (d, h, min and sec)."""
    total = seconds

    # Decompose total seconds into days, hours, minutes and seconds
    remainder = total
    days = int(remainder // 86400)
    remainder -= days * 86400
    hours = int(remainder // 3600)
    remainder -= hours * 3600
    minutes = int(remainder // 60)
    secs = remainder - minutes * 60

    def _fmt_secs(value: float) -> str:
        return f"{value:.{precision}f}"

    # Build parts starting from the highest non-zero unit
    parts: list[str] = []
    if days > 0:
        parts.append(f"{days} d")
        parts.append(f"{hours} h")
        parts.append(f"{minutes} min")
        parts.append(f"{_fmt_secs(secs)} sec")
    elif hours > 0:
        parts.append(f"{hours} h")
        parts.append(f"{minutes} min")
        parts.append(f"{_fmt_secs(secs)} sec")
    elif minutes > 0:
        parts.append(f"{minutes} min")
        parts.append(f"{_fmt_secs(secs)} sec")
    else:
        # All higher units are zero, only show seconds
        parts.append(f"{_fmt_secs(secs)} sec")

    return " ".join(parts)


__all__ = ["Timer"]
