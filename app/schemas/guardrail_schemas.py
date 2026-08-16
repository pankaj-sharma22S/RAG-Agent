"""Pydantic schemas for input/output guardrails, PII detection, and injection checking."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PIIEntity(BaseModel):
    """Detected PII entity occurrence."""

    entity_type: Literal[
        "email", "phone", "api_key", "secret_key", "password", "ssn", "credit_card", "jwt_token", "private_key", "connection_string", "bearer_token"
    ] = Field(description="Type of sensitive information.")
    original_text: str = Field(description="Matched sensitive string.")
    masked_text: str = Field(description="Redacted replacement string.")
    start: int = Field(description="Starting index in input string.")
    end: int = Field(description="Ending index in input string.")


class PIIResult(BaseModel):
    """Result of PII scanning and redaction."""

    contains_pii: bool = Field(default=False, description="Whether sensitive data was detected.")
    entities: list[PIIEntity] = Field(default_factory=list, description="List of detected entities.")
    sanitized_text: str = Field(..., description="Text with sensitive items masked.")


class InjectionResult(BaseModel):
    """Result of prompt injection and jailbreak risk analysis."""

    is_injection_detected: bool = Field(default=False, description="Whether prompt injection was detected.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk confidence score.")
    pattern_matched: str | None = Field(default=None, description="Triggered injection pattern or signature.")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(default="low", description="Assessed risk.")
    explanation: str | None = Field(default=None, description="Explanation of why text was flagged.")


class SanitizedInput(BaseModel):
    """Final sanitized user input envelope ready for downstream pipeline processing."""

    original_input: str
    cleaned_input: str
    is_safe: bool = True
    pii_result: PIIResult
    injection_result: InjectionResult
    truncated: bool = False
    rejection_reason: str | None = None


class GuardrailResult(BaseModel):
    """Overall status of input or output safety validation."""

    passed: bool = Field(default=True, description="Whether the check passed without blocking.")
    action: Literal["allow", "mask", "block", "retry"] = Field(
        default="allow", description="Enforced guardrail action."
    )
    details: str | None = Field(default=None, description="Detailed explanation of findings.")
