"""Unified input guardrail pipeline coordinating sanitization, PII redaction, and injection scanning."""

from __future__ import annotations

try:
    from app.guardrails.pii import global_pii_guard
    from app.guardrails.prompt_injection import global_injection_guard
    from app.guardrails.sanitizer import sanitize_text
    from app.schemas.guardrail_schemas import SanitizedInput
except ModuleNotFoundError:
    from guardrails.pii import global_pii_guard
    from guardrails.prompt_injection import global_injection_guard
    from guardrails.sanitizer import sanitize_text
    from schemas.guardrail_schemas import SanitizedInput


try:
    from app.tracing import safe_traceable
except Exception:
    try:
        from tracing import safe_traceable
    except Exception:
        safe_traceable = lambda **kw: lambda f: f


class InputGuard:
    """End-to-end user input pre-processor."""

    @staticmethod
    @safe_traceable(name="input_guardrails", run_type="chain", tags=["guardrails", "security", "pii", "injection"])
    def process(text: str) -> SanitizedInput:
        """Run full input sanitization, PII masking, and injection scanning."""
        cleaned, truncated = sanitize_text(text)
        pii_res = global_pii_guard.scan_and_mask(cleaned)
        inj_res = global_injection_guard.check_injection(pii_res.sanitized_text)

        is_safe = True
        rejection = None
        if inj_res.is_injection_detected and inj_res.risk_level == "critical":
            is_safe = False
            rejection = "Input flagged by security guardrail for potential prompt injection."

        return SanitizedInput(
            original_input=text,
            cleaned_input=pii_res.sanitized_text,
            is_safe=is_safe,
            pii_result=pii_res,
            injection_result=inj_res,
            truncated=truncated,
            rejection_reason=rejection,
        )


global_input_guard = InputGuard()
