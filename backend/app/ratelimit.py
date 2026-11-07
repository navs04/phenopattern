"""Small in-memory sliding-window rate limiter (no extra dependency).

Per key (client IP) it remembers the timestamps of recent requests. A request is allowed if
fewer than `limit` happened in the last `window_s` seconds. Memory is bounded: when more than
`max_keys` clients are tracked, expired ones are dropped, then the least recently active.

Limits of this design (stated, not hidden): the state lives in ONE process, so it resets on
restart and is not shared if you run several workers (fine for a single free-tier instance).
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Mapping


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if limit < 1 or window_s <= 0 or max_keys < 1:
            raise ValueError("limit, window_s and max_keys must be positive")
        self.limit, self.window_s, self.max_keys, self._clock = limit, window_s, max_keys, clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()   # LRU order: oldest activity first
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request attempt. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = self._hits[key] = deque()
            else:
                self._hits.move_to_end(key)
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, math.ceil(q[0] + self.window_s - now))
            q.append(now)
            if len(self._hits) > self.max_keys:
                self._evict(now)
            return True, 0

    def _evict(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or q[-1] <= now - self.window_s]:
            del self._hits[k]
        while len(self._hits) > self.max_keys:
            self._hits.popitem(last=False)

    def __len__(self) -> int:
        return len(self._hits)


def client_ip(headers: Mapping[str, str], client_host: str | None, trust_proxy: bool) -> str:
    """Key used for rate limiting.

    X-Forwarded-For is trivially spoofable by clients, so it is only read when explicitly
    enabled (TRUST_PROXY_HEADERS) because the service sits behind a proxy that sets it. Verify
    your host's behaviour before enabling (see DEPLOYMENT.md).
    """
    if trust_proxy:
        xff = headers.get("x-forwarded-for", "")
        first = xff.split(",")[0].strip()
        if first:
            return first[:64]
    return (client_host or "unknown")[:64]
