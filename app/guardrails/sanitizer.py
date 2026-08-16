"""Input sanitization and normalization for user queries and retrieved content."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

MAX_INPUT_LENGTH = 30_000

# Control characters to strip (excluding \n, \r, \t)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize_text(text: str, max_chars: int = MAX_INPUT_LENGTH) -> tuple[str, bool]:
    """Normalize whitespace and strip harmful control characters without breaking source code.

    Returns:
        (sanitized_text: str, is_truncated: bool)
    """
    if not text:
        return "", False

    # Normalize unicode to standard form
    normalized = unicodedata.normalize("NFC", text)

    # Remove non-printable control characters
    cleaned = CONTROL_CHAR_RE.sub("", normalized)

    is_truncated = False
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
        is_truncated = True

    return cleaned, is_truncated


def validate_url(url: str) -> bool:
    """Check if a URL string is structurally valid with http or https protocol."""
    try:
        result = urlparse(url.strip())
        return all([result.scheme in {"http", "https"}, result.netloc])
    except Exception:
        return False
