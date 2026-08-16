"""Prompt injection detection and untrusted external context isolation."""

from __future__ import annotations

import re
from typing import Any

try:
    from app.schemas.guardrail_schemas import InjectionResult
except ModuleNotFoundError:
    from schemas.guardrail_schemas import InjectionResult

INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|developer|system)\s+(\w+\s+)?(instructions|directives|rules|prompts)", re.IGNORECASE), "ignore_instructions", "high"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|developer|system)\s+(\w+\s+)?(instructions|rules|prompts)", re.IGNORECASE), "disregard_instructions", "high"),
    (re.compile(r"reveal\s+(all\s+)?(your\s+|the\s+)?(system\s+prompt|initial\s+instructions|system\s+instructions|secret\s+key|password)", re.IGNORECASE), "system_prompt_leakage", "critical"),
    (re.compile(r"you\s+are\s+now\s+(in\s+)?(DAN\s+mode|unrestricted\s+mode|developer\s+mode|unfiltered)", re.IGNORECASE), "jailbreak_mode", "critical"),
    (re.compile(r"forget\s+(everything\s+)?(you\s+know|your\s+rules|all\s+instructions)", re.IGNORECASE), "forget_rules", "high"),
    (re.compile(r"override\s+(all\s+)?(safety|system|developer)\s+protocols", re.IGNORECASE), "safety_override", "critical"),
    (re.compile(r"system:\s*you\s+must", re.IGNORECASE), "role_impersonation", "medium"),
]


class InjectionGuard:
    """Detects adversarial injection attacks and encapsulates external context."""

    @staticmethod
    def check_injection(text: str) -> InjectionResult:
        """Scan input query for common adversarial injection signatures."""
        if not text:
            return InjectionResult(is_injection_detected=False, risk_level="low")

        for pattern, name, severity in INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return InjectionResult(
                    is_injection_detected=True,
                    confidence=0.92,
                    pattern_matched=name,
                    risk_level=severity,  # type: ignore
                    explanation=f"Matched adversarial pattern: '{match.group(0)}'",
                )

        return InjectionResult(is_injection_detected=False, confidence=0.0, risk_level="low")

    @staticmethod
    def wrap_untrusted_content(content: str, source_name: str = "document", content_type: str = "text") -> str:
        """Encapsulate external untrusted content within strict security boundary markers.

        Informs the LLM that any directives or instructions enclosed within the envelope
        are passive data elements to analyze, NOT system-level executable commands.
        """
        header = f"<<< UNTRUSTED_EXTERNAL_DATA [Source: {source_name} | Type: {content_type}] >>>"
        footer = "<<< END_UNTRUSTED_EXTERNAL_DATA >>>"
        return f"{header}\n{content}\n{footer}"


global_injection_guard = InjectionGuard()
