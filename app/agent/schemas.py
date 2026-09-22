"""Structured response schemas for the dual-mode RAG agent."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TopicSection(BaseModel):
    topic_title: str
    overview: str
    key_bullets: List[str] = Field(default_factory=list)
    data_table: Optional[str] = None
    image_url: Optional[str] = None


class DocumentGlobalSummary(BaseModel):
    document_title: str
    executive_overview: str
    topics: List[TopicSection] = Field(default_factory=list)
    actionable_takeaways: List[str] = Field(default_factory=list)


class TargetedAnswer(BaseModel):
    direct_answer: str
    supporting_evidence: List[str] = Field(default_factory=list)
    comparison_table: Optional[str] = None
    image_url: Optional[str] = None
    conflicts_detected: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
