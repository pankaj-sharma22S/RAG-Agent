"""FastAPI router for GitHub Repository Analyzer Agent."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from app.agents.github_analyzer_agent import analyze_github_repository
    from app.gateway import global_rate_limiter
except ModuleNotFoundError:
    from agents.github_analyzer_agent import analyze_github_repository
    from gateway import global_rate_limiter

router = APIRouter(prefix="/github", tags=["GitHub Analyzer"])


class GitHubAnalysisRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="Public or authenticated GitHub repository URL (e.g. 'https://github.com/owner/repo').",
        examples=["https://github.com/langchain-ai/langgraph"],
    )
    focus_query: str = Field(
        default="",
        description="Optional focus query, question, or specific area to inspect.",
        examples=["Analyze authentication architecture and error handling."],
    )
    branch: str | None = Field(
        default=None,
        description="Optional target branch to analyze. Defaults to the repository's default branch.",
    )
    thread_id: str = Field(
        default="api_user",
        description="Session / thread ID for tracing isolation.",
    )


@router.get("/status")
def github_analyzer_status():
    """Health check for GitHub Analyzer subsystem."""
    return {"status": "online", "service": "GitHub Repository Analyzer Agent"}


@router.post("/analyze")
async def analyze_repo_endpoint(request: GitHubAnalysisRequest) -> dict[str, Any]:
    """Execute end-to-end multi-stage repository analysis and generate an evidence-based report."""
    allowed, remaining, retry_after = global_rate_limiter.is_allowed(request.thread_id, endpoint="github_analyze")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Please retry after {retry_after} seconds.",
        )

    try:
        result = analyze_github_repository(
            repo_url=request.repo_url,
            focus_query=request.focus_query,
            branch=request.branch,
            thread_id=request.thread_id,
        )
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Repository analysis failed: {error}") from error
