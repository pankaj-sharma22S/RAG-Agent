"""Centralized FastAPI exception handlers with structured JSON error envelopes."""

from __future__ import annotations

import logging
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .exceptions import AppException

logger = logging.getLogger("APIExceptionHandler")


def format_error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: str | None = None,
    details: Any | None = None,
) -> JSONResponse:
    """Construct a clean, standardized error response envelope."""
    payload = {
        "error": {
            "code": error_code,
            "message": message,
            "request_id": request_id,
            "details": details,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "Domain exception [%s] on %s %s (RequestID: %s): %s",
        exc.error_code,
        request.method,
        request.url.path,
        request_id,
        exc.message,
    )
    return format_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        request_id=request_id,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        408: "REQUEST_TIMEOUT",
        413: "PAYLOAD_TOO_LARGE",
        422: "UNPROCESSABLE_ENTITY",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }
    error_code = error_code_map.get(exc.status_code, "HTTP_ERROR")
    message = str(exc.detail) if isinstance(exc.detail, str) else "An HTTP error occurred."
    details = exc.detail if not isinstance(exc.detail, str) else None

    logger.warning(
        "HTTP exception %d [%s] on %s %s (RequestID: %s): %s",
        exc.status_code,
        error_code,
        request.method,
        request.url.path,
        request_id,
        message,
    )
    return format_error_response(
        status_code=exc.status_code,
        error_code=error_code,
        message=message,
        request_id=request_id,
        details=details,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    formatted_errors = []
    for err in exc.errors():
        loc = " -> ".join(str(item) for item in err.get("loc", []))
        formatted_errors.append({
            "location": loc,
            "message": err.get("msg", "Invalid parameter"),
            "type": err.get("type", "validation_error"),
        })

    logger.warning(
        "Validation error on %s %s (RequestID: %s): %d issues",
        request.method,
        request.url.path,
        request_id,
        len(formatted_errors),
    )
    return format_error_response(
        status_code=422,
        error_code="VALIDATION_ERROR",
        message="The request payload or parameters failed validation.",
        request_id=request_id,
        details=formatted_errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    # Log full traceback server-side only
    logger.exception(
        "Unhandled server exception on %s %s (RequestID: %s): %s",
        request.method,
        request.url.path,
        request_id,
        str(exc),
    )
    # Return safe, non-leaking message to client
    return format_error_response(
        status_code=500,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred. Please contact support with the request ID.",
        request_id=request_id,
        details=None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global error handlers with the FastAPI application."""
    from starlette.exceptions import HTTPException as StarletteHTTPException
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
