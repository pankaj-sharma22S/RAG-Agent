"""Fallback manager and error classifier for multi-tier provider failovers."""

from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel

try:
    from app.schemas.gateway_schemas import FallbackDecision
except ModuleNotFoundError:
    from schemas.gateway_schemas import FallbackDecision

logger = logging.getLogger("LLMGateway.Fallback")

# Error indicators that warrant provider failover
FAILOVER_ERROR_KEYWORDS = [
    "timeout",
    "timed out",
    "connection refused",
    "connecterror",
    "apiconnectionerror",
    "rate limit",
    "ratelimit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "service unavailable",
    "bad gateway",
    "not found",
    "model_not_found",
    "unauthorized",
    "invalid api key",
]


class FallbackManager:
    """Classifies provider failures and manages failover state."""

    @staticmethod
    def should_fallback(error: Exception) -> tuple[bool, str]:
        """Determine if an exception warrants failing over to the next provider.

        Returns:
            (should_failover: bool, categorized_reason: str)
        """
        error_msg = str(error).lower()
        error_cls = type(error).__name__.lower()

        # Categorize
        if "timeout" in error_msg or "timeouterror" in error_cls:
            return True, "timeout"
        if "connect" in error_msg or "connection" in error_cls:
            return True, "connection_failure"
        if "rate limit" in error_msg or "429" in error_msg or "ratelimit" in error_cls:
            return True, "rate_limit_exceeded"
        if "503" in error_msg or "unavailable" in error_msg or "overloaded" in error_msg:
            return True, "provider_unavailable"
        if "500" in error_msg or "502" in error_msg or "504" in error_msg:
            return True, "server_error"
        if "auth" in error_msg or "unauthorized" in error_msg or "401" in error_msg or "api key" in error_msg:
            return True, "authentication_failure"
        if "not found" in error_msg or "404" in error_msg:
            return True, "model_unavailable"

        # Check keyword heuristics
        for kw in FAILOVER_ERROR_KEYWORDS:
            if kw in error_msg or kw in error_cls:
                return True, f"provider_error_{kw.replace(' ', '_')}"

        # General unexpected provider failure
        return True, f"generic_provider_failure_{type(error).__name__}"

    @staticmethod
    def build_fallback_record(
        primary_model: str,
        primary_provider: str,
        fallback_model: str,
        fallback_provider: str,
        error: Exception,
    ) -> FallbackDecision:
        """Create a structured FallbackDecision record."""
        _, reason = FallbackManager.should_fallback(error)
        logger.warning(
            f"LLM Failover triggered: [{primary_provider}/{primary_model}] -> "
            f"[{fallback_provider}/{fallback_model}] Reason: {reason} ({error})"
        )
        return FallbackDecision(
            primary_model=primary_model,
            primary_provider=primary_provider,
            fallback_model=fallback_model,
            fallback_provider=fallback_provider,
            fallback_reason=str(error)[:200],
            error_type=reason,
        )
