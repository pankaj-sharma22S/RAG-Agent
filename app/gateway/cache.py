"""Response cache adapter delegating to the centralized RedisService."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

try:
    from app.services.redis_service import redis_service
except ModuleNotFoundError:
    from services.redis_service import redis_service


class ResponseCache:
    """Centralized response cache interface delegating to the non-critical RedisService."""

    def __init__(self) -> None:
        self.service = redis_service

    def generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        session_id: str = "global",
        structured_schema: Optional[str] = None,
    ) -> str:
        """Create a deterministic SHA-256 cache key."""
        hasher = hashlib.sha256()
        hasher.update(session_id.encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update(f"{temperature:.2f}".encode("utf-8"))
        if structured_schema:
            hasher.update(structured_schema.encode("utf-8"))
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            hasher.update(f"{role}:{content}".encode("utf-8"))
        return hasher.hexdigest()

    def get(self, query_or_key: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Retrieve cached response via Redis semantic cache."""
        res = self.service.get_semantic_cache(query=query_or_key, context=context)
        if res is not None:
            if isinstance(res, dict) and "content" in res:
                return res
            if isinstance(res, dict) and "answer" in res:
                return res["answer"]
            return res
        return None

    def set(
        self,
        query_or_key: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store response in Redis semantic cache."""
        self.service.set_semantic_cache(
            query=query_or_key,
            response_data=data,
            context=context,
            ttl_seconds=ttl_seconds,
        )

    def clear(self) -> None:
        """Purge cached responses."""
        self.service.clear_cache()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return self.service.stats()


global_response_cache = ResponseCache()
