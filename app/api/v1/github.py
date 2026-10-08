"""FastAPI router for GitHub Repository Analyzer Agent."""

from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from agents.github_analyzer_agent import analyze_github_repository
from core.auth import AuthenticatedUser, get_optional_user
from core.exceptions import RateLimitException
from services.redis_service import redis_service
from schemas.api_schemas import GitHubAnalysisRequest, GitHubAnalysisResponse

logger = logging.getLogger("GitHubAPI")
router = APIRouter(prefix="/github", tags=["GitHub Analyzer Agent"])


@router.get("/status", summary="Health check for GitHub Analyzer")
def github_analyzer_status() -> dict[str, str]:
    """Health check for GitHub Analyzer subsystem."""
    return {"status": "online", "service": "GitHub Repository Analyzer Agent"}


@router.post("/analyze", response_model=GitHubAnalysisResponse, summary="Execute deep GitHub repository analysis")
async def analyze_repo_endpoint(
    request: GitHubAnalysisRequest,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> GitHubAnalysisResponse:
    """Execute read-only multi-stage repository inspection and generate an architectural report."""
    user_id = current_user.user_id
    effective_thread = request.thread_id if request.thread_id != "default" else user_id

    # 1. Rate Limit Check
    allowed, remaining, retry_after = redis_service.check_rate_limit(user_id, endpoint="github_analyze")
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    # 2. Semantic Cache Check
    cache_query = f"{request.repo_url} | focus: {request.focus_query or 'overview'} | branch: {request.branch or 'default'}"
    cache_ctx = {"user_id": user_id, "endpoint": "github_analyze"}
    cached = redis_service.get_semantic_cache(query=cache_query, context=cache_ctx)
    if cached and isinstance(cached, dict) and cached.get("status") == "completed":
        return GitHubAnalysisResponse(
            status="completed",
            repo_info=cached.get("repo_info", {}),
            markdown_report=cached.get("markdown_report", ""),
            findings=cached.get("findings", []),
            error=None,
        )

    # 3. Execute Analysis on Cache MISS
    try:
        result = analyze_github_repository(
            repo_url=request.repo_url,
            focus_query=request.focus_query or "",
            branch=request.branch,
            thread_id=effective_thread,
        )
        if result.get("error"):
            return GitHubAnalysisResponse(
                status="failed",
                error=result["error"],
            )

        resp_data = {
            "status": "completed",
            "repo_info": result.get("repo_info", {}),
            "markdown_report": result.get("markdown_report", ""),
            "findings": result.get("findings", []),
        }

        # 4. Store in Redis Semantic Cache (TTL: 2 hours)
        redis_service.set_semantic_cache(
            query=cache_query,
            response_data=resp_data,
            context=cache_ctx,
            ttl_seconds=7200,
        )

        return GitHubAnalysisResponse(
            status="completed",
            repo_info=result.get("repo_info", {}),
            markdown_report=result.get("markdown_report", ""),
            findings=result.get("findings", []),
            error=None,
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Repository analysis failed: %s", error)
        raise HTTPException(status_code=500, detail=f"Repository analysis failed: {error}") from error
