"""PII (Personally Identifiable Information), Password, and Credential Detection & Redaction Guard."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Literal
from dotenv import load_dotenv

load_dotenv()

try:
    from app.schemas.guardrail_schemas import PIIEntity, PIIResult
except ModuleNotFoundError:
    from schemas.guardrail_schemas import PIIEntity, PIIResult

# Comprehensive regex suite for secrets, credentials, tokens, keys, and PII
SECRET_AND_PII_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"),
    "password": re.compile(r"(?i)\b(?:password|passwd|pwd|db_pass|admin_pass|root_pass|user_pass|secret_key|api_secret)\s*[:=]\s*['\"]?([A-Za-z0-9!@#$%^&*()_+=\-{}[\]:;\"'<>,.?/~\\]{3,})['\"]?"),
    "connection_string": re.compile(r"\b(?:postgresql|postgres|mysql|mongodb|mongodb\+srv|redis|sqlite|mssql|oracle):\/\/[A-Za-z0-9_-]+:[^@\s]+@[A-Za-z0-9.-]+(?::\d+)?\/[A-Za-z0-9_.-]*"),
    "bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_-]{10,}\b"),
    "api_key": re.compile(r"\b(?:AIzaSy[A-Za-z0-9_-]{10,60}|sk-[A-Za-z0-9_-]{6,80}|ghp_[A-Za-z0-9]{10,60}|github_pat_[A-Za-z0-9_]{10,85}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{8,72}|sk_live_[0-9a-zA-Z]{10,34})\b"),
    "jwt_token": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


class PIIGuard:
    """Thread-safe content scanner that detects, classifies, and redacts sensitive credentials and PII."""

    def __init__(self) -> None:
        self.mode: Literal["allow", "mask", "block"] = os.getenv("PII_GUARD_MODE", "mask").lower()  # type: ignore
        self._scan_cache: dict[str, PIIResult] = {}

    def scan_and_mask(self, text: str) -> PIIResult:
        """Scan text for passwords, API keys, private keys, connection strings, tokens, and PII."""
        if not text or not isinstance(text, str):
            return PIIResult(contains_pii=False, entities=[], sanitized_text=text or "")

        if self.mode == "allow":
            return PIIResult(contains_pii=False, entities=[], sanitized_text=text)

        # Fast content hash cache check
        content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        if content_hash in self._scan_cache:
            return self._scan_cache[content_hash]

        entities: list[PIIEntity] = []
        sanitized = text

        for entity_type, pattern in SECRET_AND_PII_PATTERNS.items():
            for match in pattern.finditer(text):
                # For password regex with captured group, mask the value
                if entity_type == "password" and match.lastindex and match.lastindex >= 1:
                    val_start = match.start(1)
                    val_end = match.end(1)
                    original_val = match.group(1)
                    mask_label = "[REDACTED]"
                    entity = PIIEntity(
                        entity_type="password",  # type: ignore
                        original_text=original_val,
                        masked_text=mask_label,
                        start=val_start,
                        end=val_end,
                    )
                    entities.append(entity)
                else:
                    original = match.group(0)
                    mask_label = f"[{entity_type.upper()}_REDACTED]" if entity_type != "password" else "[REDACTED]"
                    entity = PIIEntity(
                        entity_type=entity_type,  # type: ignore
                        original_text=original,
                        masked_text=mask_label,
                        start=match.start(),
                        end=match.end(),
                    )
                    entities.append(entity)

        # Deduplicate overlapping entities
        entities = sorted(entities, key=lambda e: e.start, reverse=True)
        filtered_entities: list[PIIEntity] = []
        last_start = float("inf")
        for ent in entities:
            if ent.end <= last_start:
                filtered_entities.append(ent)
                last_start = ent.start

        # Apply redactions
        for entity in filtered_entities:
            sanitized = sanitized[: entity.start] + entity.masked_text + sanitized[entity.end :]

        result = PIIResult(
            contains_pii=len(filtered_entities) > 0,
            entities=filtered_entities,
            sanitized_text=sanitized,
        )

        # Cache result (bounded cache)
        if len(self._scan_cache) > 2000:
            self._scan_cache.clear()
        self._scan_cache[content_hash] = result

        return result


global_pii_guard = PIIGuard()
