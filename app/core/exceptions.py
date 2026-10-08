"""Centralized application exception hierarchy for structured error handling."""

from __future__ import annotations

from typing import Any


class AppException(Exception):
    """Base exception for all domain and application errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        error_code: str | None = None,
        status_code: int | None = None,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.details = details


class AuthenticationException(AppException):
    """Raised when authentication fails, token is missing, expired, or invalid signature."""

    status_code = 401
    error_code = "AUTHENTICATION_FAILED"

    def __init__(self, message: str = "Authentication failed. Please provide a valid Bearer token.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class AuthorizationException(AppException):
    """Raised when an authenticated user lacks permission for a resource or action."""

    status_code = 403
    error_code = "FORBIDDEN"

    def __init__(self, message: str = "You do not have permission to access this resource.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class ValidationException(AppException):
    """Raised when request payload fails semantic or domain validation rules."""

    status_code = 422
    error_code = "VALIDATION_ERROR"

    def __init__(self, message: str = "Invalid request parameters.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class NotFoundException(AppException):
    """Raised when a requested resource (thread, document, user) is not found."""

    status_code = 404
    error_code = "NOT_FOUND"

    def __init__(self, message: str = "The requested resource was not found.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class RateLimitException(AppException):
    """Raised when a user or client IP exceeds configured request quotas."""

    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(self, message: str = "Too many requests. Please try again later.", retry_after: int | None = None) -> None:
        details = {"retry_after_seconds": retry_after} if retry_after is not None else None
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class TimeoutException(AppException):
    """Raised when an agent, RAG step, or external tool execution times out."""

    status_code = 504
    error_code = "GATEWAY_TIMEOUT"

    def __init__(self, message: str = "The operation timed out. Please try again.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class DatabaseException(AppException):
    """Raised when a database query or checkpoint transaction fails."""

    status_code = 500
    error_code = "DATABASE_ERROR"

    def __init__(self, message: str = "A database operation failed.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class AgentException(AppException):
    """Raised when LangGraph agent or RAG execution encounters an unrecoverable failure."""

    status_code = 500
    error_code = "AGENT_EXECUTION_ERROR"

    def __init__(self, message: str = "Agent execution failed.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)


class InternalException(AppException):
    """Raised for unexpected internal server errors."""

    status_code = 500
    error_code = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str = "An internal server error occurred.", details: Any | None = None) -> None:
        super().__init__(message=message, error_code=self.error_code, status_code=self.status_code, details=details)
