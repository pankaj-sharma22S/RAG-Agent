"""REST endpoints for persistent memory-backed chat."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

try:
    from app.memory.service import global_memory_service
except ModuleNotFoundError:
    from memory.service import global_memory_service


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, object]:
    result = await global_memory_service.chat(
        request.query,
        {"configurable": {"thread_id": request.thread_id, "user_id": request.user_id}},
    )
    return result.model_dump()


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str, user_id: str) -> dict[str, object]:
    return global_memory_service.store.get_thread(thread_id, user_id).model_dump(mode="json")


@router.get("/threads/user/{user_id}")
def list_threads(user_id: str) -> list[dict[str, object]]:
    return global_memory_service.store.list_user_threads(user_id)


@router.get("/profile/{user_id}")
def get_profile(user_id: str) -> dict[str, object]:
    return global_memory_service.store.get_user_profile(user_id).model_dump(mode="json")
