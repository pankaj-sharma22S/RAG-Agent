"""Pydantic schemas for Self-RAG, Hybrid Retrieval, Cross-Encoder, CRAG, and Query Refinement."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class KnowledgeDecision(BaseModel):
    """Self-RAG pre-check decision classifying knowledge requirement."""

    answer_from_model: bool = Field(
        description="Whether the query can reasonably and accurately be answered solely from general model knowledge."
    )
    requires_documents: bool = Field(
        description="Whether answering the query requires user-uploaded or session-indexed document evidence."
    )
    requires_web: bool = Field(
        description="Whether answering the query requires external real-time or current web research."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score in the knowledge assessment."
    )
    reason: str = Field(description="Explanation of why external evidence is or is not needed.")


class CRAGDecision(BaseModel):
    """Corrective RAG (CRAG) evaluation of retrieved document evidence."""

    decision: Literal["good", "partial", "bad"] = Field(
        description=(
            "good: retrieved evidence sufficiently and reliably answers the query; "
            "partial: evidence partially addresses the query but critical details are missing; "
            "bad: evidence is irrelevant, contradictory, or empty."
        )
    )
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Assessed relevance of the evidence to the query."
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Assessed completeness in answering the whole query."
    )
    contradiction: bool = Field(
        default=False, description="Whether retrieved evidence contains conflicting or contradictory statements."
    )
    missing_information: list[str] = Field(
        default_factory=list, description="Specific factual elements, topics, or aspects still missing."
    )
    reason: str = Field(description="Detailed diagnostic justification for the evaluation decision.")
    should_refine: bool = Field(
        default=False, description="Whether a refined retrieval attempt should be performed."
    )


class QueryRefinement(BaseModel):
    """Output of adaptive query refinement for improving retrieval performance."""

    original_query: str = Field(description="The user's original unrefined query.")
    refined_query: str = Field(
        description="Optimized, terminology-enriched, or focused query addressing why previous retrieval failed."
    )
    strategy: Literal[
        "query_rewrite",
        "terminology_expansion",
        "entity_expansion",
        "metadata_focused",
        "multi_query",
        "broader_query",
        "narrower_query",
        "temporal_clarification",
        "decomposition",
    ] = Field(description="Refinement heuristic strategy applied.")
    reason: str = Field(description="Rationale for how the refined query targets the diagnosed failure.")
    missing_information: list[str] = Field(
        default_factory=list, description="List of missing facets targeted by the refinement."
    )
    search_focus: list[str] = Field(
        default_factory=list, description="Key concepts or keywords prioritized in the new query."
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Optional list of decomposed sub-queries for multi-query retrieval.",
    )


class RetrievalAttempt(BaseModel):
    """Audit record capturing each retrieval and evaluation step."""

    attempt: int = Field(description="1-based attempt sequence number.")
    query: str = Field(description="Search query used in this attempt.")
    strategy: str = Field(default="hybrid", description="Retrieval strategy employed.")
    dense_count: int = Field(default=0, description="Number of candidate chunks returned by dense retrieval.")
    sparse_count: int = Field(default=0, description="Number of candidate chunks returned by sparse BM25.")
    fused_count: int = Field(default=0, description="Number of candidates after RRF fusion.")
    reranked_count: int = Field(default=0, description="Number of candidates retained after Cross-Encoder reranking.")
    top_rerank_score: float = Field(default=0.0, description="Highest Cross-Encoder score in this attempt.")
    relevance_score: float = Field(default=0.0, description="CRAG relevance score.")
    completeness_score: float = Field(default=0.0, description="CRAG completeness score.")
    crag_decision: str = Field(default="good", description="CRAG evaluation outcome (good/partial/bad).")
    refinement_reason: str | None = Field(default=None, description="Reason for subsequent refinement if needed.")
