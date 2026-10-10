"""Pydantic models for the three-tier persistent memory architecture."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserPreferences(BaseModel):
    user_id: str
    name: Optional[str] = None
    profession: Optional[str] = None
    technical_stack: List[str] = Field(default_factory=list)
    custom_rules: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=utc_now)


class ExtractedUserProfile(BaseModel):
    name: Optional[str] = None
    profession: Optional[str] = None
    technical_stack: List[str] = Field(default_factory=list)
    custom_rules: List[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    user_query: str
    agent_response: str
    route: Optional[str] = None
    suggested_followups: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)
    created_at: Optional[datetime] = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThreadConversation(BaseModel):
    thread_id: str
    user_id: str
    title: str = "New Conversation"
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgentStructuredOutput(BaseModel):
    answer: str
    suggested_followups: List[str] = Field(default_factory=list, min_length=2, max_length=3)
