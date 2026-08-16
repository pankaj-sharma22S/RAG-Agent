"""Workflow-level schemas for intent, ambiguity, retrieval evaluation, multimodal orchestration, and structured RAG answers."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class MultimodalDecision(BaseModel):
    """Internal orchestration and routing decision object for Multimodal RAG."""

    intent: str = Field(description="Classified user intent")
    source_required: bool = Field(default=False, description="Whether answer requires active document or source")
    modalities: list[Literal["TEXT", "IMAGE", "TABLE", "CHART", "DIAGRAM", "AUDIO", "VIDEO", "MIXED"]] = Field(
        default_factory=lambda: ["TEXT"], description="Detected media and data modalities"
    )
    needs_web: bool = Field(default=False, description="Whether external web comparison or live info is required")
    needs_ocr: bool = Field(default=False, description="Whether text extraction via OCR is needed")
    needs_visual_retrieval: bool = Field(default=False, description="Whether visual embedding or image/frame retrieval is needed")
    needs_vision_model: bool = Field(default=False, description="Whether a vision-capable LLM analysis is required")
    needs_table_extraction: bool = Field(default=False, description="Whether structured table extraction is needed")
    needs_chart: bool = Field(default=False, description="Whether chart generation/visualization is requested")
    needs_calculation: bool = Field(default=False, description="Whether numerical calculation tool is needed")
    needs_refinement: bool = Field(default=False, description="Whether query refinement is required")
    parallel_tasks: list[str] = Field(default_factory=list, description="Independent retrieval/tool tasks to run concurrently")
    max_refinements: int = Field(default=1, ge=0, le=2, description="Maximum allowable refinement iterations")


class IntentDecision(BaseModel):
    """Classification of user intent and query type."""

    intent: Literal[
        "chat",
        "document_question",
        "web_research",
        "hybrid_research",
        "github_analysis",
        "clarification_required",
    ] = Field(description="Classified user query intent.")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence.")
    is_ambiguous: bool = Field(default=False, description="Whether the request is underspecified or ambiguous.")
    missing_information: str | None = Field(
        default=None, description="Description of what context or detail is missing."
    )
    recommended_route: Literal["chat", "clarify", "research", "github"] = Field(
        description="Recommended initial LangGraph path."
    )


class AmbiguityDecision(BaseModel):
    """Contextual clarification query when user input lacks sufficient reference."""

    is_ambiguous: bool = Field(description="Whether query requires clarification before executing RAG.")
    clarification_question: str = Field(
        description="Polite, context-aware clarification question presenting specific candidate topics."
    )
    reason: str = Field(description="Explanation of why clarification is needed.")


class EvaluationDecision(BaseModel):
    """Structured decision returned by retrieval evaluator after reviewing chunks."""

    route: Literal["answer", "web", "combined"] = Field(
        description="answer=document chunks only, web=current web only, combined=documents plus web"
    )
    chunk_confidence: float = Field(ge=0, le=1, description="Confidence that chunks answer the request")
    reason: str = Field(description="Short reason for the selected route")
    web_query: str = Field(
        description="Focused current-information search query; empty only for document-only answers"
    )


class UserBiography(BaseModel):
    """LLM-selected long-term facts explicitly shared by the user."""

    facts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Short, stable, useful user facts keyed by a clear lowercase name, such "
            "as name, profession, current_project, preference, skill, or goal."
        ),
    )


class RAGAnswer(BaseModel):
    """Structured final answer envelope with citations and evidence."""

    answer: str = Field(description="The primary markdown-formatted response.")
    cited_sources: list[str] = Field(default_factory=list, description="Referenced document files or web URLs.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall answer confidence.")
