"""Rate limiter proxy delegating all rate-limiting checks to the centralized RedisService."""

from __future__ import annotations

from typing import Tuple, Optional
try:
    from app.services.redis_service import redis_service
except ModuleNotFoundError:
    from services.redis_service import redis_service


class RateLimiter:
    """Sliding-window rate limiter powered by the centralized non-critical RedisService."""

    def __init__(self) -> None:
        self.service = redis_service

    def is_allowed(
        self,
        identifier: str,
        endpoint: str = "default",
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> Tuple[bool, int, float]:
        """Check rate limit via RedisService.
        
        Returns:
            (allowed: bool, remaining_requests: int, retry_after_seconds: float)
        """
        return self.service.check_rate_limit(
            identifier=identifier,
            endpoint=endpoint,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def reset(self, identifier: Optional[str] = None) -> None:
        """Reset rate limiter state (no-op or handled gracefully)."""
        pass


global_rate_limiter = RateLimiter()
