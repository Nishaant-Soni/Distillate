"""Sliding-window rate limits: one per visitor's session and one per app instance.

Cloud Run runs a few instances at most (``max_instances`` in the service config), so a limit
per instance bounds the total without shared state. The serving API limits requests too; this
one is for the visitor: a clear "try again in N seconds" instead of an error from the API.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindow:
    """At most ``limit`` events in any ``window_s`` seconds. Safe to share between threads."""

    def __init__(self, limit: int, window_s: float = 60,
                 clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        while self._events and self._events[0] <= now - self.window_s:
            self._events.popleft()

    def wait_s(self) -> float:
        """Seconds until one more event is allowed (0 if allowed now). Records nothing."""
        with self._lock:
            now = self.clock()
            self._expire(now)
            if len(self._events) < self.limit:
                return 0.0
            return self._events[0] + self.window_s - now

    def acquire(self) -> float:
        """Record an event and return 0 if allowed; else record nothing and return the wait."""
        with self._lock:
            now = self.clock()
            self._expire(now)
            if len(self._events) < self.limit:
                self._events.append(now)
                return 0.0
            return self._events[0] + self.window_s - now


def check(session: SlidingWindow, instance: SlidingWindow) -> float:
    """0 if this request may go (and it is counted in both windows); else seconds to wait.

    A request refused by the visitor's own limit doesn't use up the instance's.
    """
    wait = session.wait_s()
    if wait:
        return wait
    wait = instance.acquire()
    if wait:
        return wait
    session.acquire()
    return 0.0
