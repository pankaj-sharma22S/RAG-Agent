"""HTTP middleware for request tracking, correlation IDs, timing, and security headers."""

from __future__ import annotations

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("RequestLogger")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Assigns unique Request IDs, tracks request duration, and logs request lifecycle."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Extract existing X-Request-ID header or generate a new UUID
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)

            # Do not log noisy health check polling unless error
            if request.url.path not in {"/health", "/api/v1/health"} or response.status_code >= 400:
                logger.info(
                    "[%s] %s %s -> status=%d duration=%.2fms client=%s",
                    request_id,
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                    client_ip,
                )
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "[%s] %s %s -> UNHANDLED EXCEPTION after %.2fms: %s",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
                str(exc),
            )
            raise
