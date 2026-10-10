"""Conversation threads, session history, and semantic memory endpoints."""

from __future__ import annotations

import logging
from typing import Any, List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from core.auth import AuthenticatedUser, get_current_user, get_optional_user
from core.exceptions import AuthorizationException, NotFoundException
from memory.service import global_memory_service
from schemas.api_schemas import (
    ConversationTurnSchema,
    ThreadDetailResponse,
    ThreadListResponse,
    ThreadSummary,
    UserProfileResponse,
)

logger = logging.getLogger("ThreadsAPI")
router = APIRouter(tags=["Threads & Memory"])


@router.get("/threads", response_model=ThreadListResponse, summary="List conversation threads for current user")
def list_user_threads(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> ThreadListResponse:
    """Retrieve all persisted chat threads owned by the authenticated user or guest."""
    user_id = current_user.user_id
    raw_threads = global_memory_service.store.list_user_threads(user_id)

    # If guest has no threads yet, also include any recent demo threads
    if not raw_threads and current_user.is_anonymous:
        raw_threads = global_memory_service.store.list_user_threads("default")

    summaries = [
        ThreadSummary(
            thread_id=t["thread_id"],
            user_id=t["user_id"],
            title=t["title"],
            turn_count=t.get("turn_count", 0),
            created_at=t["created_at"],
            updated_at=t["updated_at"],
        )
        for t in raw_threads
    ]

    paginated = summaries[offset : offset + limit]
    return ThreadListResponse(threads=paginated, total=len(summaries))


@router.get("/threads/{thread_id}", response_model=ThreadDetailResponse, summary="Get full conversation history for a thread")
def get_thread_details(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> ThreadDetailResponse:
    """Fetch all chronological conversation turns for a given thread with user ownership validation."""
    user_id = current_user.user_id
    thread = global_memory_service.store.get_thread(thread_id, user_id)
    owns_thread = global_memory_service.store.thread_exists(thread_id, user_id)

    if current_user.is_anonymous and not owns_thread:
        default_thread = global_memory_service.store.get_thread(thread_id, "default")
        if global_memory_service.store.thread_exists(thread_id, "default"):
            thread = default_thread
        else:
            raise NotFoundException(f"Thread '{thread_id}' not found.")
    elif not owns_thread:
        raise NotFoundException(f"Thread '{thread_id}' not found.")

    turns_schemas = []
    for t in (thread.turns or []):
        ts = getattr(t, "created_at", None) or getattr(t, "timestamp", None)
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")
        turns_schemas.append(
            ConversationTurnSchema(
                turn_id=str(getattr(t, "turn_id", "")),
                user_query=str(getattr(t, "user_query", "")),
                agent_response=str(getattr(t, "agent_response", "")),
                route=getattr(t, "route", None),
                created_at=ts_str,
            )
        )

    t_created = getattr(thread, "created_at", None)
    t_updated = getattr(thread, "updated_at", None)
    return ThreadDetailResponse(
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        title=thread.title,
        turns=turns_schemas,
        created_at=t_created.isoformat() if hasattr(t_created, "isoformat") else str(t_created or ""),
        updated_at=t_updated.isoformat() if hasattr(t_updated, "isoformat") else str(t_updated or ""),
    )


@router.delete("/threads/{thread_id}", summary="Delete a conversation thread")
def delete_thread_endpoint(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> dict[str, Any]:
    """Delete a conversation thread and its associated turns from SQLite store."""
    user_id = current_user.user_id
    success = global_memory_service.store.delete_thread(thread_id, user_id)
    return {"status": "success" if success else "not_found", "thread_id": thread_id}


class RenameThreadPayload(BaseModel):
    title: str


@router.patch("/threads/{thread_id}", summary="Rename a conversation thread")
def rename_thread_endpoint(
    thread_id: str,
    payload: RenameThreadPayload,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> dict[str, Any]:
    """Update the title of an existing conversation thread."""
    user_id = current_user.user_id
    success = global_memory_service.store.rename_thread(thread_id, user_id, payload.title)
    return {"status": "success" if success else "failed", "thread_id": thread_id, "title": payload.title}


@router.get("/profile", response_model=UserProfileResponse, summary="Get semantic profile memory for current user")
def get_user_profile_endpoint(
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> UserProfileResponse:
    """Retrieve durable user profile facts and custom preferences."""
    user_id = current_user.user_id
    profile = global_memory_service.store.get_user_profile(user_id)
    return UserProfileResponse(
        user_id=profile.user_id if profile else user_id,
        name=profile.name if profile and profile.name else ("Demo Analyst" if current_user.is_anonymous else "Verified User"),
        profession=profile.profession if profile and profile.profession else "AI Engineer",
        technical_stack=profile.technical_stack if profile and profile.technical_stack else ["Python", "FastAPI", "React", "ChromaDB", "LangGraph"],
        custom_rules=profile.custom_rules if profile else [],
        last_updated=profile.last_updated.isoformat() if profile else "2026-10-08T00:00:00",
    )


@router.get("/threads/{thread_id}/suggestions", summary="Get document and memory grounded suggestions")
def get_thread_suggestions_endpoint(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> dict[str, Any]:
    """Return contextual, document-grounded suggestions based on active source and thread history."""
    try:
        from db.service import get_active_source
    except ModuleNotFoundError:
        from app.db.service import get_active_source

    active_src = get_active_source(thread_id)
    thread = global_memory_service.store.get_thread(thread_id, current_user.user_id) or global_memory_service.store.get_thread(thread_id, "default")

    suggestions: list[dict[str, str]] = []

    if active_src and active_src.get("title"):
        title = active_src.get("title")
        src_type = active_src.get("source_type", "document")
        if src_type == "pdf" or "pdf" in str(active_src.get("source_url", "")).lower():
            suggestions.append({
                "category": "Document Core",
                "label": f"Explain Key Architecture in {title[:25]}",
                "query": f"What is the main architecture and core mechanism presented in {title}?",
                "icon": "FileText",
            })
            suggestions.append({
                "category": "Deep Dive",
                "label": "Masking & Positional Encoding",
                "query": "Explain how masking and positional encoding operate in this document with citations.",
                "icon": "Layers",
            })
            suggestions.append({
                "category": "Visual Evidence",
                "label": "Inspect Extracted Figures & Diagrams",
                "query": "What diagrams and figures are included in this document, and what do they show?",
                "icon": "ImageIcon",
            })
            suggestions.append({
                "category": "Executive Summary",
                "label": "Hierarchical Summary with Takeaways",
                "query": f"Provide a complete structured summary of {title} with key takeaways.",
                "icon": "Sparkles",
            })
        elif src_type == "youtube":
            suggestions.append({
                "category": "Video Summary",
                "label": f"Deep Summary of Video ({title[:20]})",
                "query": "Give me a deep structured summary of this video transcript with timestamps.",
                "icon": "Video",
            })
            suggestions.append({
                "category": "Key Insights",
                "label": "Core Arguments & Takeaways",
                "query": "What are the primary arguments, techniques, or lessons shared by the speaker in this video?",
                "icon": "Sparkles",
            })
        elif "github" in src_type:
            suggestions.append({
                "category": "Codebase Audit",
                "label": f"Component Flow in {title[:25]}",
                "query": "What is the component architecture and request lifecycle in this repository?",
                "icon": "GitBranch",
            })
            suggestions.append({
                "category": "Weakness Scan",
                "label": "Identify Anti-Patterns & Bottlenecks",
                "query": "What are the key architectural weaknesses, missing tests, or concurrency risks in this repository?",
                "icon": "Shield",
            })

    # If recent turns exist in thread, suggest natural follow-up
    if thread and thread.turns and len(thread.turns) > 0:
        last_turn = thread.turns[-1]
        last_q = last_turn.user_query
        if len(suggestions) < 4:
            suggestions.append({
                "category": "Follow-Up Context",
                "label": f"Elaborate on: '{last_q[:25]}...'",
                "query": f"Can you elaborate further on your previous explanation of '{last_q}' with more detail?",
                "icon": "Sparkles",
            })

    # Default starter prompts if no active document or turns
    if not suggestions:
        suggestions = [
            {
                "category": "Document Intelligence",
                "label": "Summarize Uploaded PDF",
                "query": "Give me a structured hierarchical summary of the uploaded document with key takeaways.",
                "icon": "FileText",
            },
            {
                "category": "System Architecture",
                "label": "Transformer & Attention Mechanisms",
                "query": "Explain Multi-Head Attention, Positional Encoding, and Masking mechanisms in detail.",
                "icon": "Layers",
            },
            {
                "category": "GitHub Codebase Audit",
                "label": "Inspect FastAPI Architecture",
                "query": "Analyze the architecture and routing mechanisms of https://github.com/fastapi/fastapi",
                "icon": "GitBranch",
            },
            {
                "category": "Real-Time Intelligence",
                "label": "Current AI Agent Trends",
                "query": "What are the latest breakthrough developments in autonomous AI agent frameworks this year?",
                "icon": "Globe",
            },
        ]

    return {"thread_id": thread_id, "suggestions": suggestions}

