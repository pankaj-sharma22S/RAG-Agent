"""Output guardrail inspecting AI responses for secret leakage, system prompts, credentials, and formatting integrity."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel

try:
    from app.guardrails.pii import global_pii_guard
    from app.schemas.guardrail_schemas import GuardrailResult
except ModuleNotFoundError:
    from guardrails.pii import global_pii_guard
    from schemas.guardrail_schemas import GuardrailResult


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
