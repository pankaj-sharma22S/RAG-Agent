"""Sliding-window rate limiter for protecting API endpoints."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class RateLimiter:
    """Sliding-window in-memory rate limiter per key (e.g. session_id or IP) and endpoint."""

    def __init__(self) -> None:
        self._enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in {"true", "1", "yes"}
        self._default_limit = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
        self._default_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(
        self,
        identifier: str,
        endpoint: str = "default",
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> tuple[bool, int, float]:
        """Check if request is allowed.

        Returns:
            (allowed: bool, remaining_requests: int, retry_after_seconds: float)
        """
        if not self._enabled:
            return True, 999, 0.0

        limit = max_requests if max_requests is not None else self._default_limit
        window = window_seconds if window_seconds is not None else self._default_window
        now = time.time()
        bucket_key = f"{identifier}:{endpoint}"

        with self._lock:
            # Purge timestamps outside current sliding window
            timestamps = [ts for ts in self._requests[bucket_key] if now - ts < window]
            self._requests[bucket_key] = timestamps

            if len(timestamps) < limit:
                self._requests[bucket_key].append(now)
                remaining = limit - len(self._requests[bucket_key])
                return True, remaining, 0.0
            else:
                oldest = timestamps[0]
                retry_after = max(0.0, window - (now - oldest))
                return False, 0, round(retry_after, 2)

    def reset(self, identifier: str | None = None) -> None:
        """Reset rate limiter state for a specific key or globally."""
        with self._lock:
            if identifier:
                for k in list(self._requests.keys()):
                    if k.startswith(identifier):
                        del self._requests[k]
            else:
                self._requests.clear()


global_rate_limiter = RateLimiter()
