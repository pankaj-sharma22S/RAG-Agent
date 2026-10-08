"""Standardized Pydantic request, response, and SSE streaming models for FastAPI API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


# =========================================================================
# 1. Health & Status Models
# =========================================================================

class ServiceStatus(BaseModel):
    status: Literal["healthy", "degraded", "offline"]
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "offline"]
    version: str
    timestamp: str
    environment: str
    services: Dict[str, ServiceStatus]


# =========================================================================
# 2. Authentication & User Models
# =========================================================================

class UserProfileResponse(BaseModel):
    user_id: str
    name: Optional[str] = None
    profession: Optional[str] = None
    technical_stack: List[str] = Field(default_factory=list)
    custom_rules: List[str] = Field(default_factory=list)
    last_updated: str


class CurrentUserResponse(BaseModel):
    user_id: str
    email: Optional[str] = None
    role: str
    is_anonymous: bool
    profile: Optional[UserProfileResponse] = None


# =========================================================================
# 3. Chat & Conversation Models
# =========================================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="The user's query or instruction for the AI agent.",
        examples=["What are the API rate limits in the uploaded policy document?"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional session / thread identifier. Generated automatically if omitted.",
        examples=["thread_abc123"],
    )
    active_source: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional active source dictionary metadata (document, video, or GitHub URL).",
    )


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    route: str
    retrieved_chunks: int = 0
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)
    latency_ms: float


class ChatStreamEvent(BaseModel):
    """Base model for Server-Sent Events (SSE) in streaming chat."""
    event: Literal["status", "token", "error", "done"]
    data: Dict[str, Any]


# =========================================================================
# 4. Ingestion & Document Models
# =========================================================================

class UploadResponse(BaseModel):
    filename: str
    content_type: Optional[str] = None
    file_size_bytes: int
    location: str
    status: Literal["uploaded", "indexed", "failed"]
    chunks_indexed: int = 0
    thread_id: str


class URLIngestRequest(BaseModel):
    url: HttpUrl = Field(
        ...,
        description="Public webpage or YouTube video link to ingest.",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    thread_id: Optional[str] = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Target session / thread ID for indexed chunk isolation.",
    )


class URLIngestResponse(BaseModel):
    url: str
    chunks_indexed: int
    source_type: str
    status: Literal["indexed", "failed"]
    thread_id: str


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4_000,
        description="Question to answer using indexed document chunks.",
    )
    thread_id: Optional[str] = Field(
        default="default",
        min_length=1,
        max_length=128,
    )


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: int
    route: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)


# =========================================================================
# 5. GitHub Analyzer Models
# =========================================================================

class GitHubAnalysisRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="GitHub repository or topic URL.",
        examples=["https://github.com/langchain-ai/langgraph"],
    )
    focus_query: Optional[str] = Field(
        default="",
        description="Specific question or inspection area.",
    )
    branch: Optional[str] = Field(
        default=None,
        description="Optional branch to analyze.",
    )
    thread_id: Optional[str] = Field(
        default="default",
        min_length=1,
        max_length=128,
    )


class GitHubAnalysisResponse(BaseModel):
    status: Literal["completed", "failed"]
    repo_info: Dict[str, Any] = Field(default_factory=dict)
    markdown_report: str = ""
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


# =========================================================================
# 6. Threads & History Models
# =========================================================================

class ThreadSummary(BaseModel):
    thread_id: str
    user_id: str
    title: str
    turn_count: int
    created_at: str
    updated_at: str


class ThreadListResponse(BaseModel):
    threads: List[ThreadSummary]
    total: int


class ConversationTurnSchema(BaseModel):
    turn_id: str
    user_query: str
    agent_response: str
    route: Optional[str] = None
    created_at: str


class ThreadDetailResponse(BaseModel):
    thread_id: str
    user_id: str
    title: str
    turns: List[ConversationTurnSchema]
    created_at: str
    updated_at: str


# =========================================================================
# 7. Gateway Telemetry & Cache Models
# =========================================================================

class GatewayStatsResponse(BaseModel):
    cost_and_usage: Dict[str, Any]
    cache: Dict[str, Any]
