"""Safe text extraction for public web pages with strict SSRF protection and untrusted content wrapping."""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from core.document import create_document
    from guardrails.pii import global_pii_guard
    from guardrails.prompt_injection import global_injection_guard
except ModuleNotFoundError:
    from app.core.document import create_document
    from app.guardrails.pii import global_pii_guard
    from app.guardrails.prompt_injection import global_injection_guard

BLOCKED_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/Cloud metadata
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_ssrf_safe_url(url: str) -> None:
    """Validate that URL does not target private or localhost network resources (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must begin with http:// or https:// and contain a valid hostname.")

    hostname = parsed.hostname or ""
    if hostname.lower() in BLOCKED_HOSTNAMES or hostname.endswith((".local", ".internal", ".lan")):
        raise ValueError(f"Security Alert: Access to local/internal host '{hostname}' is forbidden (SSRF protection).")

    # Resolve IP and verify not private range
    try:
        ip_str = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip_str)
        for net in PRIVATE_IP_RANGES:
            if ip_obj in net:
                raise ValueError(f"Security Alert: Destination IP '{ip_str}' is in a private network range (SSRF protection).")
    except socket.gaierror:
        raise ValueError(f"Unable to resolve hostname '{hostname}'.")


class _PageTextParser(HTMLParser):
    """Collect visible text while ignoring scripts, styles, and navigation markup."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "iframe"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "iframe"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        if not self._ignored_depth:
            self.parts.append(text)


def load_web_page(url: str):
    """Download a public HTML page safely, enforce SSRF protection, scan secrets, and wrap content as untrusted."""
    _validate_ssrf_safe_url(url)
    parsed = urlparse(url)

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise ValueError(f"This link returns {content_type}, not an HTML web page.")
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read(5_000_000).decode(charset, errors="replace")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not open this link: {error}") from error

    parser = _PageTextParser()
    parser.feed(html)
    text = "\n".join(parser.parts)
    if len(text.strip()) < 20:
        raise ValueError("No readable text was found on this page.")

    # 1. Content Security Scan: Redact passwords, API keys, private keys, connection strings, PII
    pii_res = global_pii_guard.scan_and_mask(text)
    sanitized_text = pii_res.sanitized_text

    # 2. Encapsulate in untrusted security boundary envelope
    wrapped_text = global_injection_guard.wrap_untrusted_content(
        sanitized_text, source_name=url, content_type="webpage_content"
    )

    return create_document(
        wrapped_text,
        {"source": url, "type": "web", "title": parser.title or parsed.netloc},
    )
