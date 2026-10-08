"""Chat endpoints with non-streaming, token streaming, and Server-Sent Events (SSE) support.

Integrated with Centralized Redis Service:
1. Redis rate-limit check (fails-open if Redis unavailable; raises 429 if limit exceeded).
2. Redis semantic-cache check (skips expensive LLM pipeline on similarity HIT).
3. Existing Agent / RAG pipeline execution on cache MISS.
4. Response storage in Redis semantic cache with context isolation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from core.auth import AuthenticatedUser, get_optional_user
from core.exceptions import AgentException, RateLimitException
from memory.service import global_memory_service
from rag_workflow import direct_query_router, stream_response
from schemas.api_schemas import ChatRequest, ChatResponse
from services.redis_service import redis_service

logger = logging.getLogger("ChatAPI")
router = APIRouter(prefix="/chat", tags=["Agentic Chat & Streaming"])


@router.post("", response_model=ChatResponse, summary="Send message to AI Agent (Standard response)")
async def chat_endpoint(
    request: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> ChatResponse:
    """Execute end-to-end agentic workflow with persistent memory, semantic cache, and rate limiting."""
    user_id = current_user.user_id
    thread_id = request.thread_id or f"thread_{user_id[:8]}_{int(time.time())}"
    t_start = time.perf_counter()

    # 1. Redis Rate-Limit Check
    allowed, remaining, retry_after = redis_service.check_rate_limit(
        identifier=user_id,
        endpoint="chat",
    )
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    # 2. Redis Semantic-Cache Check
    cache_context = {"user_id": user_id, "endpoint": "chat"}
    cached_entry = redis_service.get_semantic_cache(
        query=request.message,
        context=cache_context,
    )
    if cached_entry:
        cached_answer = cached_entry.get("answer", "") if isinstance(cached_entry, dict) else str(cached_entry)
        followups = cached_entry.get("suggested_followups", []) if isinstance(cached_entry, dict) else []
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return ChatResponse(
            answer=cached_answer,
            thread_id=thread_id,
            route="semantic_cache",
            retrieved_chunks=0,
            sources=[],
            suggested_followups=followups,
            latency_ms=elapsed_ms,
        )

    # 3. Cache MISS -> Execute Existing Pipeline
    try:
        result = await global_memory_service.chat(
            query=request.message,
            config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 4. Store Final Response in Redis Semantic Cache
        if result.answer and not result.answer.strip().startswith("I encountered an issue") and not result.answer.strip().startswith("⚠️"):
            redis_service.set_semantic_cache(
                query=request.message,
                response_data={
                    "answer": result.answer,
                    "suggested_followups": result.suggested_followups or [],
                },
                context=cache_context,
            )

        return ChatResponse(
            answer=result.answer,
            thread_id=thread_id,
            route="memory_chat",
            retrieved_chunks=0,
            sources=[],
            suggested_followups=result.suggested_followups or [],
            latency_ms=elapsed_ms,
        )
    except Exception as err:
        logger.exception("Chat execution failed: %s", err)
        raise AgentException(f"Agent chat processing failed: {err}") from err


@router.post("/stream", summary="Stream tokens token-by-token (Plain text stream)")
async def chat_stream_endpoint(
    request: ChatRequest,
    http_request: Request,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> StreamingResponse:
    """Stream AI response tokens with semantic cache and rate limiting."""
    user_id = current_user.user_id
    thread_id = request.thread_id or f"thread_{user_id[:8]}_{int(time.time())}"

    # 1. Redis Rate-Limit Check
    allowed, remaining, retry_after = redis_service.check_rate_limit(
        identifier=user_id,
        endpoint="chat_stream",
    )
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    # 2. Redis Semantic-Cache Check
    cache_context = {"user_id": user_id, "endpoint": "chat_stream"}
    cached_entry = redis_service.get_semantic_cache(
        query=request.message,
        context=cache_context,
    )

    async def token_generator() -> AsyncGenerator[str, None]:
        # CACHE HIT
        if cached_entry:
            cached_text = cached_entry.get("answer", "") if isinstance(cached_entry, dict) else str(cached_entry)
            words = cached_text.split(" ")
            for idx, word in enumerate(words):
                if await http_request.is_disconnected():
                    break
                sep = " " if idx < len(words) - 1 else ""
                yield word + sep
                await asyncio.sleep(0.01)
            return

        # CACHE MISS -> Stream from existing pipeline
        collected_tokens: list[str] = []
        try:
            token_iter = stream_response(
                question=request.message,
                messages=[HumanMessage(content=request.message)],
                session_id=thread_id,
            )
            for token in token_iter:
                if await http_request.is_disconnected():
                    logger.info("Client disconnected during stream (thread: %s)", thread_id)
                    break
                collected_tokens.append(token)
                yield token
                await asyncio.sleep(0.001)

            # Store in semantic cache
            full_text = "".join(collected_tokens)
            if full_text and not full_text.strip().startswith("I encountered an issue") and not full_text.strip().startswith("⚠️"):
                redis_service.set_semantic_cache(
                    query=request.message,
                    response_data={"answer": full_text},
                    context=cache_context,
                )
        except Exception as err:
            logger.error("Streaming error in thread %s: %s", thread_id, err)
            yield f"\n\n[Error: {err}]"

    return StreamingResponse(token_generator(), media_type="text/plain")


@router.post("/sse", summary="Stream AI Agent events via Server-Sent Events (SSE)")
async def chat_sse_endpoint(
    request: ChatRequest,
    http_request: Request,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> StreamingResponse:
    """Stream structured agent events via SSE with rate limiting, semantic caching, and SQLite persistence."""
    user_id = current_user.user_id
    thread_id = request.thread_id or f"thread_{user_id[:8]}_{int(time.time())}"

    # 1. Redis Rate-Limit Check
    allowed, remaining, retry_after = redis_service.check_rate_limit(
        identifier=user_id,
        endpoint="chat_sse",
    )
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    # 2. Redis Semantic-Cache Check
    cache_context = {"user_id": user_id, "endpoint": "chat_sse"}
    cached_entry = redis_service.get_semantic_cache(
        query=request.message,
        context=cache_context,
    )

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        t_start = time.perf_counter()

        # CACHE HIT
        if cached_entry:
            cached_answer = cached_entry.get("answer", "") if isinstance(cached_entry, dict) else str(cached_entry)
            status_payload = {
                "stage": "routing",
                "route": "semantic_cache",
                "confidence": 1.0,
                "thread_id": thread_id,
                "message": "Response retrieved instantly from Redis Semantic Cache.",
            }
            yield f"event: status\ndata: {json.dumps(status_payload)}\n\n"
            await asyncio.sleep(0.01)

            # Stream cached tokens smoothly
            words = cached_answer.split(" ")
            for idx, word in enumerate(words):
                if await http_request.is_disconnected():
                    return
                token = word + (" " if idx < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
                await asyncio.sleep(0.01)

            # Persist cached turn to SQLite thread history
            try:
                from memory.schemas import ConversationTurn
                turn = ConversationTurn(
                    turn_id=str(uuid.uuid4()),
                    user_query=request.message,
                    agent_response=cached_answer,
                    route="semantic_cache",
                )
                global_memory_service.store.add_turn_to_thread(
                    thread_id=thread_id,
                    user_id=user_id,
                    turn=turn,
                )
            except Exception as save_err:
                logger.warning("Could not persist cached turn to SQLite: %s", save_err)

            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            done_payload = {
                "answer": cached_answer,
                "thread_id": thread_id,
                "route": "semantic_cache",
                "latency_ms": elapsed_ms,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            return

        # CACHE MISS -> Stream through existing pipeline
        collected_tokens: list[str] = []
        try:
            # 1. Initial Routing Status Event
            route, conf = direct_query_router(request.message, session_id=thread_id)
            status_payload = {
                "stage": "routing",
                "route": route,
                "confidence": conf,
                "thread_id": thread_id,
                "message": f"Query routed to [{route}] pipeline.",
            }
            yield f"event: status\ndata: {json.dumps(status_payload)}\n\n"
            await asyncio.sleep(0.001)

            # 2. Stream Generated Content Tokens
            token_iter = stream_response(
                question=request.message,
                messages=[HumanMessage(content=request.message)],
                session_id=thread_id,
            )

            for token in token_iter:
                if await http_request.is_disconnected():
                    logger.info("Client disconnected from SSE stream (thread: %s)", thread_id)
                    return

                collected_tokens.append(token)
                token_payload = {"content": token}
                yield f"event: token\ndata: {json.dumps(token_payload)}\n\n"
                await asyncio.sleep(0.001)

            full_answer = "".join(collected_tokens)
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            # 3. Persist turn to SQLite thread history
            if full_answer and not full_answer.strip().startswith("⚠️"):
                try:
                    from memory.schemas import ConversationTurn
                    turn = ConversationTurn(
                        turn_id=str(uuid.uuid4()),
                        user_query=request.message,
                        agent_response=full_answer,
                        route=route,
                    )
                    global_memory_service.store.add_turn_to_thread(
                        thread_id=thread_id,
                        user_id=user_id,
                        turn=turn,
                    )
                except Exception as save_err:
                    logger.warning("Could not persist SSE turn to SQLite: %s", save_err)

                # 4. Store Response in Redis Semantic Cache
                redis_service.set_semantic_cache(
                    query=request.message,
                    response_data={"answer": full_answer},
                    context=cache_context,
                )

            done_payload = {
                "answer": full_answer,
                "thread_id": thread_id,
                "route": route,
                "latency_ms": elapsed_ms,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

        except Exception as err:
            logger.error("SSE stream error in thread %s: %s", thread_id, err)
            error_payload = {
                "code": "STREAMING_ERROR",
                "message": str(err),
                "thread_id": thread_id,
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
