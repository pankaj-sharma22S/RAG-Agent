"""Context-aware in-memory response cache for the LLM Gateway."""

from __future__ import annotations

import hashlib
import json
import os
import time
from threading import Lock
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class ResponseCache:
    """Thread-safe context-isolated cache with TTL for LLM completions."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._default_ttl = int(os.getenv("LLM_CACHE_TTL", str(default_ttl_seconds)))
        self._enabled = os.getenv("LLM_CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}
        self._hits = 0
        self._misses = 0

    def generate_cache_key(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        session_id: str = "global",
        structured_schema: str | None = None,
    ) -> str:
        """Create a deterministic SHA-256 cache key including session and schema context."""
        hasher = hashlib.sha256()
        hasher.update(session_id.encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update(f"{temperature:.2f}".encode("utf-8"))
        if structured_schema:
            hasher.update(structured_schema.encode("utf-8"))
        
        # Serialize messages deterministically
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            hasher.update(f"{role}:{content}".encode("utf-8"))

        return hasher.hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve a cached entry if available and unexpired."""
        if not self._enabled:
            return None

        with self._lock:
            entry = self._cache.get(cache_key)
            if not entry:
                self._misses += 1
                return None

            now = time.time()
            if now > entry["expires_at"]:
                del self._cache[cache_key]
                self._misses += 1
                return None

            self._hits += 1
            return entry["data"]

    def set(self, cache_key: str, data: dict[str, Any], ttl_seconds: int | None = None) -> None:
        """Store an entry with TTL."""
        if not self._enabled:
            return

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        with self._lock:
            # Simple eviction if cache exceeds 2000 entries
            if len(self._cache) > 2000:
                self._purge_expired()

            self._cache[cache_key] = {
                "data": data,
                "created_at": time.time(),
                "expires_at": time.time() + ttl,
            }

    def _purge_expired(self) -> None:
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v["expires_at"]]
        for k in expired_keys:
            del self._cache[k]
        if len(self._cache) > 1500:
            # Evict oldest 500
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
            for k in sorted_keys[:500]:
                del self._cache[k]

    def clear(self) -> None:
        """Purge all cached responses."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100.0) if total > 0 else 0.0
            return {
                "enabled": self._enabled,
                "cached_entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
            }


global_response_cache = ResponseCache()
