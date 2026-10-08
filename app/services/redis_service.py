"""Centralized Redis Service for non-critical Rate Limiting and Semantic Response Caching.

Design Guarantees:
1. NON-CRITICAL: If Redis is unavailable, disconnected, or times out, all operations
   gracefully degrade. Requests NEVER fail with 500 errors due to Redis.
2. LOW LATENCY: Strict connection and socket timeouts prevent request delays.
3. CONTEXT ISOLATION: Semantic cache partitions lookups by user, tenant, route, and schema
   context to prevent cache leakage across incompatible boundaries.
4. EMBEDDING REUSE: Leverages the project's existing high-performance embedding singleton.
5. NO FILE EMBEDDINGS: Redis is strictly used for rate limiting and semantic response caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis

try:
    from app.core.config import settings
except ModuleNotFoundError:
    from core.config import settings

logger = logging.getLogger("RedisService")


class RedisService:
    """Centralized Redis manager for rate limiting and semantic response caching."""

    def __init__(self) -> None:
        self._client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None
        self._hits: int = 0
        self._misses: int = 0
        self._rate_limited_count: int = 0
        self._last_warn_time: float = 0.0

    def _warn_throttled(self, op: str, err: Exception) -> None:
        """Log Redis warning at most once every 30 seconds to prevent log flooding when Redis is down."""
        now = time.time()
        if now - self._last_warn_time > 30.0:
            self._last_warn_time = now
            logger.warning("Redis non-critical fallback on '%s': %s (continuing normally without Redis)", op, err)

    def get_client(self) -> Optional[redis.Redis]:
        """Return connected Redis client instance with strict fast timeouts."""
        if self._client is not None:
            return self._client

        try:
            connect_timeout = getattr(settings, "REDIS_CONNECT_TIMEOUT", 0.5)
            socket_timeout = getattr(settings, "REDIS_SOCKET_TIMEOUT", 0.5)
            redis_url = getattr(settings, "REDIS_URL", None)

            if redis_url and redis_url.startswith("redis://"):
                self._client = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=connect_timeout,
                    socket_timeout=socket_timeout,
                    decode_responses=True,
                )
            else:
                host = getattr(settings, "REDIS_HOST", "localhost")
                port = getattr(settings, "REDIS_PORT", 6379)
                password = getattr(settings, "REDIS_PASSWORD", None)
                db = getattr(settings, "REDIS_DB", 0)
                self._client = redis.Redis(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    socket_connect_timeout=connect_timeout,
                    socket_timeout=socket_timeout,
                    decode_responses=True,
                )
            return self._client
        except Exception as err:
            self._warn_throttled("get_client", err)
            return None

    def is_available(self) -> bool:
        """Check if Redis server is reachable within timeout."""
        try:
            client = self.get_client()
            if client is None:
                return False
            return bool(client.ping())
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # 1. Rate Limiting (Sliding Window in Redis)
    # -------------------------------------------------------------------------

    def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = "default",
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> Tuple[bool, int, float]:
        """Check sliding-window rate limit for a client/user identifier and endpoint.

        Returns:
            (allowed: bool, remaining_requests: int, retry_after_seconds: float)
        """
        if not getattr(settings, "RATE_LIMIT_ENABLED", True):
            return True, 999, 0.0

        if not identifier or not identifier.strip():
            identifier = "anonymous"

        limit = max_requests if max_requests is not None else getattr(settings, "DEFAULT_RATE_LIMIT_PER_MINUTE", 60)
        window = window_seconds if window_seconds is not None else 60

        client = self.get_client()
        if client is None:
            # Redis unavailable -> Allow request gracefully
            return True, limit, 0.0

        bucket_key = f"ratelimit:{endpoint}:{identifier.strip()}"
        now = time.time()

        try:
            pipe = client.pipeline(transaction=True)
            # Remove timestamps older than the sliding window
            pipe.zremrangebyscore(bucket_key, 0, now - window)
            # Count current valid requests in the window
            pipe.zcard(bucket_key)
            # Fetch the oldest request in the window for calculating retry_after
            pipe.zrange(bucket_key, 0, 0, withscores=True)
            results = pipe.execute()

            current_count = int(results[1])
            oldest_entries = results[2]

            if current_count < limit:
                # Add current request with a unique member key and timestamp score
                unique_member = f"{now}_{uuid.uuid4().hex[:6]}"
                pipe2 = client.pipeline(transaction=True)
                pipe2.zadd(bucket_key, {unique_member: now})
                pipe2.expire(bucket_key, window + 10)
                pipe2.execute()
                remaining = max(0, limit - current_count - 1)
                return True, remaining, 0.0
            else:
                # Rate limit exceeded
                self._rate_limited_count += 1
                if oldest_entries and len(oldest_entries) > 0:
                    oldest_ts = float(oldest_entries[0][1])
                    retry_after = max(0.1, window - (now - oldest_ts))
                else:
                    retry_after = float(window)
                return False, 0, round(retry_after, 2)

        except (redis.RedisError, ConnectionError, TimeoutError, OSError, Exception) as exc:
            self._warn_throttled("check_rate_limit", exc)
            # Redis failure: fail-open to ensure user request is not blocked
            return True, limit, 0.0

    # -------------------------------------------------------------------------
    # 2. Semantic Response Caching
    # -------------------------------------------------------------------------

    def _build_context_namespace(self, context: Optional[Dict[str, Any]]) -> str:
        """Create a deterministic hash string representing the context partition."""
        if not context:
            return "global"
        try:
            # Filter out non-partitioning dynamic keys like timestamps or ephemeral request IDs
            filtered_context = {
                k: v for k, v in context.items()
                if k in {"user_id", "tenant_id", "schema_version", "route", "collection", "endpoint", "database"}
                and v is not None
            }
            if not filtered_context:
                return "global"
            serialized = json.dumps(filtered_context, sort_keys=True)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "global"

    def _get_query_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding vector using the project's existing singleton embedding model."""
        try:
            from app.embedding.embedding_model import get_embedding_model
        except ModuleNotFoundError:
            try:
                from embedding.embedding_model import get_embedding_model
            except Exception as err:
                logger.warning("Could not import embedding model for semantic cache: %s", err)
                return None

        try:
            embed_model = get_embedding_model()
            return embed_model.embed_query(text.strip())
        except Exception as err:
            logger.warning("Embedding generation for semantic cache failed: %s", err)
            return None

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two float vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_semantic_cache(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Lookup semantically similar previous answer in Redis.

        Returns:
            cached_response dict if similarity >= threshold and context matches, else None.
        """
        if not getattr(settings, "SEMANTIC_CACHE_ENABLED", True):
            return None

        clean_query = query.strip()
        if not clean_query:
            return None

        client = self.get_client()
        if client is None:
            return None

        sim_threshold = (
            threshold
            if threshold is not None
            else getattr(settings, "SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92)
        )

        ns = self._build_context_namespace(context)
        index_key = f"semcache:idx:{ns}"

        try:
            # 1. Fetch candidate entry IDs in this context partition
            entry_ids = client.smembers(index_key)
            if not entry_ids:
                self._misses += 1
                return None

            # 2. Compute embedding for the incoming query
            query_vec = self._get_query_embedding(clean_query)
            if query_vec is None:
                self._misses += 1
                return None

            # 3. Batch fetch cached entries via Redis pipeline
            entry_keys = [f"semcache:entry:{ns}:{eid}" for eid in entry_ids]
            pipe = client.pipeline(transaction=False)
            for k in entry_keys:
                pipe.get(k)
            raw_entries = pipe.execute()

            best_match: Optional[Dict[str, Any]] = None
            best_similarity: float = -1.0
            expired_ids: List[str] = []

            for eid, raw in zip(entry_ids, raw_entries):
                if not raw:
                    expired_ids.append(eid)
                    continue

                try:
                    entry = json.loads(raw)
                    cached_vec = entry.get("embedding")
                    if not cached_vec or len(cached_vec) != len(query_vec):
                        continue

                    similarity = self._cosine_similarity(query_vec, cached_vec)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry
                except Exception:
                    continue

            # Clean up expired index references in the background
            if expired_ids:
                try:
                    client.srem(index_key, *expired_ids)
                except Exception:
                    pass

            # 4. Check if highest similarity exceeds threshold
            if best_match and best_similarity >= sim_threshold:
                self._hits += 1
                logger.info(
                    "Semantic Cache HIT [ns=%s, sim=%.4f >= %.2f] query: '%s' -> cached: '%s'",
                    ns,
                    best_similarity,
                    sim_threshold,
                    clean_query[:50],
                    best_match.get("query", "")[:50],
                )
                response_data = best_match.get("response")
                if isinstance(response_data, dict):
                    return response_data
                return {"answer": str(response_data)}

            self._misses += 1
            return None

        except (redis.RedisError, ConnectionError, TimeoutError, OSError, Exception) as exc:
            self._warn_throttled("get_semantic_cache", exc)
            self._misses += 1
            return None

    def set_semantic_cache(
        self,
        query: str,
        response_data: Any,
        context: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Store query embedding and response in Redis semantic cache with TTL."""
        if not getattr(settings, "SEMANTIC_CACHE_ENABLED", True):
            return False

        clean_query = query.strip()
        if not clean_query or not response_data:
            return False

        # Do not cache error responses or empty responses
        if isinstance(response_data, str) and (
            response_data.startswith("I encountered an issue") or response_data.startswith("⚠️")
        ):
            return False
        if isinstance(response_data, dict):
            ans = str(response_data.get("answer", ""))
            if ans.startswith("I encountered an issue") or ans.startswith("⚠️"):
                return False

        client = self.get_client()
        if client is None:
            return False

        query_vec = self._get_query_embedding(clean_query)
        if query_vec is None:
            return False

        ns = self._build_context_namespace(context)
        entry_id = hashlib.sha256(clean_query.lower().encode("utf-8")).hexdigest()[:16]
        entry_key = f"semcache:entry:{ns}:{entry_id}"
        index_key = f"semcache:idx:{ns}"
        ttl = ttl_seconds if ttl_seconds is not None else getattr(settings, "SEMANTIC_CACHE_TTL_SECONDS", 3600)

        payload = {
            "query": clean_query,
            "embedding": query_vec,
            "response": response_data,
            "context": context or {},
            "created_at": time.time(),
        }

        try:
            pipe = client.pipeline(transaction=True)
            pipe.setex(entry_key, ttl, json.dumps(payload))
            pipe.sadd(index_key, entry_id)
            pipe.expire(index_key, ttl + 600)
            pipe.execute()
            return True
        except (redis.RedisError, ConnectionError, TimeoutError, OSError, Exception) as exc:
            self._warn_throttled("set_semantic_cache", exc)
            return False

    def clear_cache(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Purge semantic cache entries for a given context partition or globally."""
        client = self.get_client()
        if client is None:
            return False

        try:
            if context:
                ns = self._build_context_namespace(context)
                index_key = f"semcache:idx:{ns}"
                entry_ids = client.smembers(index_key)
                if entry_ids:
                    keys = [f"semcache:entry:{ns}:{eid}" for eid in entry_ids]
                    keys.append(index_key)
                    client.delete(*keys)
            else:
                # Global purge of semantic cache keys
                cursor = 0
                while True:
                    cursor, keys = client.scan(cursor=cursor, match="semcache:*", count=500)
                    if keys:
                        client.delete(*keys)
                    if cursor == 0:
                        break
            return True
        except Exception as err:
            self._warn_throttled("clear_cache", err)
            return False

    def stats(self) -> Dict[str, Any]:
        """Return operational health, rate-limit, and semantic cache statistics."""
        available = self.is_available()
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100.0) if total_requests > 0 else 0.0

        return {
            "redis_connected": available,
            "rate_limiting_enabled": getattr(settings, "RATE_LIMIT_ENABLED", True),
            "rate_limited_count": self._rate_limited_count,
            "semantic_cache": {
                "enabled": getattr(settings, "SEMANTIC_CACHE_ENABLED", True),
                "similarity_threshold": getattr(settings, "SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.92),
                "ttl_seconds": getattr(settings, "SEMANTIC_CACHE_TTL_SECONDS", 3600),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
            },
        }


# Global Centralized Redis Service Singleton
redis_service = RedisService()
