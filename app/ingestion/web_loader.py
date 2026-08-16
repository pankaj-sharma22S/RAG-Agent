"""Safe, dependency-free text extraction for public web pages."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from core.document import create_document


class _PageTextParser(HTMLParser):
    """Collect visible text while ignoring scripts, styles, and navigation markup."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer"}:
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
    """Download a public HTML page and turn its readable text into a document."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete public URL beginning with http:// or https://")

    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Knowledge Workspace)"})
    try:
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"This link returns {content_type}, not a web page.")
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read(5_000_000).decode(charset, errors="replace")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not open this link: {error}") from error

    parser = _PageTextParser()
    parser.feed(html)
    text = "\n".join(parser.parts)
    if len(text.strip()) < 30:
        raise ValueError("No readable text was found on this page. It may require sign-in or JavaScript.")

    return create_document(text, {"source": url, "type": "web", "title": parser.title or parsed.netloc})
