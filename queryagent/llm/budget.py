"""Rate-limit bookkeeping for free tiers.

Reactive, not predictive. A predictive token bucket assumes a long-lived warm
process, which is wrong for serverless -- every cold container would start with
a full bucket and cheerfully re-exceed a limit the provider is still
enforcing. So the authority is the provider's own 429: observing one opens a
circuit breaker for that provider until its Retry-After elapses.

The in-process request counter is a secondary courtesy check only.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Dict, Optional

log = logging.getLogger(__name__)

DEFAULT_BACKOFF_SECONDS = 60.0


class Budget:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_until: Dict[str, float] = {}
        self._recent: Dict[str, deque] = {}
        self._calls: Dict[str, int] = {}
        self._failures: Dict[str, int] = {}

    def allow(self, provider: str, rpm: Optional[int] = None) -> bool:
        """False if the breaker is open, or we are clearly over the RPM."""
        now = time.time()
        with self._lock:
            until = self._open_until.get(provider, 0.0)
            if now < until:
                return False
            if rpm:
                window = self._recent.setdefault(provider, deque())
                while window and now - window[0] > 60:
                    window.popleft()
                if len(window) >= rpm:
                    return False
            return True

    def record_attempt(self, provider: str) -> None:
        now = time.time()
        with self._lock:
            self._recent.setdefault(provider, deque()).append(now)
            self._calls[provider] = self._calls.get(provider, 0) + 1

    def record_success(self, provider: str) -> None:
        with self._lock:
            self._open_until.pop(provider, None)
            self._failures[provider] = 0

    def record_rate_limited(self, provider: str, retry_after: Optional[float]) -> None:
        wait = retry_after if retry_after and retry_after > 0 else DEFAULT_BACKOFF_SECONDS
        with self._lock:
            self._open_until[provider] = time.time() + wait
            self._failures[provider] = self._failures.get(provider, 0) + 1
        log.warning("%s rate limited; skipping it for %.0fs", provider, wait)

    def record_failure(self, provider: str) -> None:
        with self._lock:
            self._failures[provider] = self._failures.get(provider, 0) + 1

    def stats(self) -> dict:
        now = time.time()
        with self._lock:
            return {
                "calls": dict(self._calls),
                "failures": dict(self._failures),
                "breakers_open": {
                    p: round(u - now, 1) for p, u in self._open_until.items() if u > now
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._open_until.clear()
            self._recent.clear()
            self._calls.clear()
            self._failures.clear()


_budget = Budget()


def get_budget() -> Budget:
    return _budget
