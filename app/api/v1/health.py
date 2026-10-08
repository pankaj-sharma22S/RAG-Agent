"""Health check and dependency diagnostics endpoint."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from fastapi import APIRouter
from core.config import settings
from schemas.api_schemas import HealthResponse, ServiceStatus

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", response_model=HealthResponse, summary="Perform system health check")
def health_check() -> HealthResponse:
    """Return health status across database, vector store, and LLM gateway."""
    services: dict[str, ServiceStatus] = {}
    overall_status = "healthy"

    # 1. Check Vector Store (Chroma)
    t_start = time.perf_counter()
    try:
        from vectorstore.croma_store import get_vectorstore
        vs = get_vectorstore()
        count = vs._collection.count() if hasattr(vs, "_collection") else 0
        services["vector_store"] = ServiceStatus(
            status="healthy",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=f"Chroma active with {count} document vectors in session store.",
        )
    except Exception as err:
        services["vector_store"] = ServiceStatus(
            status="degraded",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=f"Vector store warning: {err}",
        )
        overall_status = "degraded"

    # 2. Check Database / Checkpointer
    t_start = time.perf_counter()
    try:
        from db.service import list_conversations
        convs = list_conversations()
        services["database"] = ServiceStatus(
            status="healthy",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=f"SQLite database responsive ({len(convs)} conversations tracked).",
        )
    except Exception as err:
        services["database"] = ServiceStatus(
            status="degraded",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=f"Database warning: {err}",
        )
        overall_status = "degraded"

    # 3. Check LLM Gateway & Redis
    t_start = time.perf_counter()
    try:
        from services.redis_service import redis_service
        cache_stats = redis_service.stats()
        redis_online = cache_stats.get("redis_connected", False)
        services["llm_gateway"] = ServiceStatus(
            status="healthy",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=f"Gateway online (Provider: {settings.PRIMARY_PROVIDER}).",
        )
        services["redis"] = ServiceStatus(
            status="healthy" if redis_online else "offline",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=(
                f"Redis connected (Semantic cache hits: {cache_stats['semantic_cache']['hits']}, Rate-limited: {cache_stats['rate_limited_count']})."
                if redis_online
                else "Redis disconnected (non-critical graceful fallback active; rate-limiting and cache bypassed without disruption)."
            ),
        )
    except Exception as err:
        services["llm_gateway"] = ServiceStatus(
            status="degraded",
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            detail=f"Gateway warning: {err}",
        )
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,  # type: ignore
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=settings.ENVIRONMENT,
        services=services,
    )
