"""Output guardrail inspecting AI responses for secret leakage, system prompts, credentials, and formatting integrity."""

from __future__ import annotations

import re
import ipaddress
import os
from urllib.parse import urlparse
from typing import Any
from pydantic import BaseModel

try:
    from app.guardrails.pii import global_pii_guard
    from app.schemas.guardrail_schemas import GuardrailResult
except ModuleNotFoundError:
    from guardrails.pii import global_pii_guard
    from schemas.guardrail_schemas import GuardrailResult


BLOCKED_IMAGE_URL = "[BLOCKED_SUSPICIOUS_IMAGE_URL]"
DEFAULT_IMAGE_DOMAINS = {"storage.googleapis.com", "s3.amazonaws.com"}


def sanitize_image_url(image_url: str | None) -> str | None:
    """Allow only HTTPS image URLs hosted on configured enterprise domains."""
    if not image_url or not isinstance(image_url, str):
        return None

    parsed = urlparse(image_url.strip())
    hostname = (parsed.hostname or "").lower().rstrip(".")
    configured = os.getenv("IMAGE_ALLOWED_DOMAINS", "")
    allowed_domains = {
        item.strip().lower().rstrip(".")
        for item in configured.split(",")
        if item.strip()
    } or DEFAULT_IMAGE_DOMAINS

    is_ip = False
    try:
        ip = ipaddress.ip_address(hostname)
        is_ip = True
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return BLOCKED_IMAGE_URL
    except ValueError:
        pass

    allowed_host = any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains)
    if parsed.scheme != "https" or not hostname or is_ip or not allowed_host:
        return BLOCKED_IMAGE_URL
    return image_url.strip()


class OutputGuard:
    """Validates model responses before returning to users or downstream workflows."""

    @staticmethod
    def validate_output(
        content: str,
        expected_schema: type[BaseModel] | None = None,
    ) -> tuple[bool, str, str | None]:
        """Verify output safety and redact any leaked passwords, keys, tokens, or PII.

        Returns:
            (passed: bool, sanitized_or_repaired_content: str, error_detail: str | None)
        """
        if not content or not content.strip():
            return False, "I apologize, but I was unable to generate a response. Please try again.", "Empty model response."

        # Scan and redact any credentials, passwords, API keys, tokens, or PII
        pii_res = global_pii_guard.scan_and_mask(content)
        sanitized_content = pii_res.sanitized_text

        # Validate schema if expected
        if expected_schema and issubclass(expected_schema, BaseModel):
            try:
                expected_schema.model_validate_json(sanitized_content)
            except Exception as err:
                return False, sanitized_content, f"Structured output schema validation failed: {err}"

        return True, sanitized_content, None


global_output_guard = OutputGuard()
