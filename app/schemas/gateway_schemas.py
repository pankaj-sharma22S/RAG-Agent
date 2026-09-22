"""Gateway schemas for LLM requests, responses, routing decisions, and usage tracking."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Standardized request envelope passed into the LLM Gateway."""

    messages: list[dict[str, str]] = Field(..., description="List of role/content chat messages.")
    complexity: Literal["simple", "medium", "complex"] = Field(
        default="medium", description="Task complexity level for smart model routing."
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate.")
    session_id: str = Field(default="default", description="Session / thread isolation ID.")
    timeout: float | None = Field(default=None, description="Request timeout in seconds.")
    response_format: Any | None = Field(default=None, description="Pydantic model or schema for structured output.")
    use_cache: bool = Field(default=True, description="Whether to check response cache.")
    force_provider: Literal["ollama", "gemini", "openrouter"] | None = Field(
        default=None, description="Optional manual override provider."
    )


class UsageMetadata(BaseModel):
    """Token usage and timing metrics for a completed completion."""

    input_tokens: int = Field(default=0, description="Number of prompt tokens consumed.")
    output_tokens: int = Field(default=0, description="Number of completion tokens generated.")
    total_tokens: int = Field(default=0, description="Total tokens consumed.")
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds.")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD.")


class FallbackDecision(BaseModel):
    """Information regarding a failover event during model invocation."""

    primary_model: str = Field(..., description="The initially attempted model.")
    primary_provider: str = Field(..., description="The initially attempted provider.")
    fallback_model: str = Field(..., description="The model successfully or subsequently invoked.")
    fallback_provider: str = Field(..., description="The provider successfully or subsequently invoked.")
    fallback_reason: str = Field(..., description="Explanation of why failover was triggered.")
    error_type: str = Field(..., description="Categorized failure reason (e.g. timeout, rate_limit, connection_error).")


class ModelRouteDecision(BaseModel):
    """Smart routing decision selecting provider and model based on complexity."""

    complexity: Literal["simple", "medium", "complex"] = Field(
        description="Assessed request complexity."
    )
    selected_provider: Literal["ollama", "gemini", "openrouter"] = Field(
        description="Selected primary provider."
    )
    selected_model: str = Field(description="Target model identifier.")
    reason: str = Field(description="Rationale for selected route.")


class GatewayRoutingDecision(BaseModel):
    """Deterministic semantic-routing decision for an incoming user request."""

    intent: str
    target_route: Literal[
        "direct_chat",
        "internal_rag",
        "web_search",
        "hybrid_crag",
        "clarification_required",
    ]
    recommended_model_tier: Literal["cheap_fast", "standard_reasoning", "deep_reasoning"]
    is_mutative_action: bool
    requires_multimodal_vit: bool
    rewritten_search_queries: list[str] = Field(default_factory=list)
    clarification_message: str | None = None


class LLMResponse(BaseModel):
    """Standardized response envelope returned by the LLM Gateway."""

    content: str = Field(..., description="Generated text content.")
    structured_data: Any | None = Field(default=None, description="Parsed structured output if requested.")
    model: str = Field(..., description="Actual model that generated the completion.")
    provider: str = Field(..., description="Actual provider that fulfilled the request.")
    cache_hit: bool = Field(default=False, description="Whether response was served from cache.")
    fallback_used: bool = Field(default=False, description="Whether a fallback was triggered.")
    fallback_info: FallbackDecision | None = Field(default=None, description="Details of failover if used.")
    usage: UsageMetadata = Field(default_factory=UsageMetadata, description="Token and cost metrics.")


class CostRecord(BaseModel):
    """Single request audit record for centralized cost tracking."""

    request_id: str
    session_id: str
    timestamp: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    cache_hit: bool
    fallback_used: bool
