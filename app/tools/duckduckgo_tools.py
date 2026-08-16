"""DuckDuckGo web search tool for RAG and GitHub Analyzer agents."""

from __future__ import annotations

from typing import Any
from ddgs import DDGS


def search_web_ddg(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo and return structured web results."""
    try:
        raw_results = DDGS().text(
            query.strip(),
            backend="duckduckgo",
            max_results=max_results,
        )
        return [
            {
                "title": item.get("title", "Untitled"),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
            for item in raw_results
            if item.get("href") and item.get("body")
        ]
    except Exception as error:
        return [{"title": "Search Error", "url": "", "snippet": f"Web search failed: {error}"}]
