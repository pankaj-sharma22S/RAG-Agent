"""Conversation threads, session history, and semantic memory endpoints."""

from __future__ import annotations

import logging
from typing import List
from fastapi import APIRouter, Depends, Query

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

    if not thread:
        # Fallback to check without user_id if guest
        if current_user.is_anonymous:
            thread = global_memory_service.store.get_thread(thread_id, "default")

    if not thread:
        raise NotFoundException(f"Thread '{thread_id}' not found.")

    turns_schemas = [
        ConversationTurnSchema(
            turn_id=t.turn_id,
            user_query=t.user_query,
            agent_response=t.agent_response,
            route=t.route,
            created_at=t.created_at.isoformat(),
        )
        for t in (thread.turns or [])
    ]

    return ThreadDetailResponse(
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        title=thread.title,
        turns=turns_schemas,
        created_at=thread.created_at.isoformat(),
        updated_at=thread.updated_at.isoformat(),
    )


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
