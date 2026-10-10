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

try:
    from app.tracing import safe_traceable
except Exception:
    try:
        from tracing import safe_traceable
    except Exception:
        safe_traceable = lambda **kw: lambda f: f

logger = logging.getLogger("ChatAPI")
router = APIRouter(prefix="/chat", tags=["Agentic Chat & Streaming"])


def resolve_request_api_key(request: ChatRequest, raw_req: Request | None = None) -> str | None:
    """Extract dynamic Gemini/LLM API key from request body or HTTP headers."""
    if request.gemini_api_key and request.gemini_api_key.strip():
        return request.gemini_api_key.strip()
    if request.api_key and request.api_key.strip():
        return request.api_key.strip()
    if raw_req:
        headers = raw_req.headers
        for header_name in ["x-gemini-api-key", "x-api-key", "x-google-api-key", "gemini-api-key"]:
            if val := headers.get(header_name):
                if val.strip():
                    return val.strip()
    return None


@router.post("", response_model=ChatResponse, summary="Send message to AI Agent (Standard response)")
async def chat_endpoint(
    request: ChatRequest,
    raw_req: Request,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> ChatResponse:
    """Execute end-to-end agentic workflow with persistent memory, semantic cache, and rate limiting."""
    user_id = current_user.user_id
    thread_id = request.thread_id or f"thread_{user_id[:8]}_{int(time.time())}"
    t_start = time.perf_counter()
    runtime_key = resolve_request_api_key(request, raw_req)

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
    cache_context = {"user_id": user_id, "thread_id": thread_id, "endpoint": "chat"}
    cached_entry = redis_service.get_semantic_cache(
        query=request.message,
        context=cache_context,
    )
    if cached_entry:
        cached_answer = cached_entry.get("answer", "") if isinstance(cached_entry, dict) else str(cached_entry)
        cached_sources = cached_entry.get("sources", []) if isinstance(cached_entry, dict) else []
        followups = cached_entry.get("suggested_followups", []) if isinstance(cached_entry, dict) else []
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return ChatResponse(
            answer=cached_answer,
            thread_id=thread_id,
            route="semantic_cache",
            retrieved_chunks=len(cached_sources),
            sources=cached_sources,
            suggested_followups=followups,
            latency_ms=elapsed_ms,
        )

    # 3. Cache MISS -> Execute End-to-End RAG Pipeline
    try:
        if runtime_key:
            from app.gateway import gateway_chat_model
            gateway_chat_model.api_key = runtime_key

        route, _ = direct_query_router(request.message, session_id=thread_id)
        sources_list: list[dict[str, Any]] = []
        route_list: list[str] = []
        # Hydrate thread conversation history
        history_messages = global_memory_service.hydrate_working_memory(
            user_id=user_id,
            thread_id=thread_id,
            max_turns=6,
        )
        full_messages = [*history_messages, HumanMessage(content=request.message)]
        collected_tokens: list[str] = []
        token_iter = stream_response(
            question=request.message,
            messages=full_messages,
            session_id=thread_id,
            sources_collector=sources_list,
            route_collector=route_list,
        )
        for token in token_iter:
            collected_tokens.append(token)

        full_answer = "".join(collected_tokens)
        final_route = route_list[-1] if route_list else route
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 4. Persist to SQLite Memory Store
        if full_answer and not full_answer.strip().startswith("⚠️"):
            try:
                from memory.schemas import ConversationTurn
                turn = ConversationTurn(
                    turn_id=str(uuid.uuid4()),
                    user_query=request.message,
                    agent_response=full_answer,
                    route=final_route,
                )
                global_memory_service.store.add_turn_to_thread(
                    thread_id=thread_id,
                    user_id=user_id,
                    turn=turn,
                )
            except Exception as save_err:
                logger.warning("Could not persist turn to SQLite: %s", save_err)

            # Store in Redis Semantic Cache
            redis_service.set_semantic_cache(
                query=request.message,
                response_data={"answer": full_answer, "sources": sources_list},
                context=cache_context,
            )

        return ChatResponse(
            answer=full_answer,
            thread_id=thread_id,
            route=final_route,
            retrieved_chunks=len(sources_list),
            sources=sources_list,
            suggested_followups=[],
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

    runtime_key = resolve_request_api_key(request, http_request)
    if runtime_key:
        from app.gateway import gateway_chat_model
        gateway_chat_model.api_key = runtime_key

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
    cache_context = {"user_id": user_id, "thread_id": thread_id, "endpoint": "chat_stream"}
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
            # Hydrate thread conversation history
            history_messages = global_memory_service.hydrate_working_memory(
                user_id=user_id,
                thread_id=thread_id,
                max_turns=6,
            )
            full_messages = [*history_messages, HumanMessage(content=request.message)]

            token_iter = stream_response(
                question=request.message,
                messages=full_messages,
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

    runtime_key = resolve_request_api_key(request, http_request)
    if runtime_key:
        from app.gateway import gateway_chat_model
        gateway_chat_model.api_key = runtime_key

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
    cache_context = {"user_id": user_id, "thread_id": thread_id, "endpoint": "chat_sse"}
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
            cached_sources = cached_entry.get("sources", []) if isinstance(cached_entry, dict) else []
            done_payload = {
                "answer": cached_answer,
                "thread_id": thread_id,
                "route": "semantic_cache",
                "latency_ms": elapsed_ms,
                "sources": cached_sources,
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
            sources_list: list[dict[str, Any]] = []
            route_list: list[str] = []
            history_messages = global_memory_service.hydrate_working_memory(
                user_id=user_id,
                thread_id=thread_id,
                max_turns=6,
            )
            full_messages = [*history_messages, HumanMessage(content=request.message)]

            token_iter = stream_response(
                question=request.message,
                messages=full_messages,
                session_id=thread_id,
                sources_collector=sources_list,
                route_collector=route_list,
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
            final_route = route_list[-1] if route_list else route
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            # 3. Persist turn to SQLite thread history
            if full_answer and not full_answer.strip().startswith("⚠️"):
                try:
                    from memory.schemas import ConversationTurn
                    turn = ConversationTurn(
                        turn_id=str(uuid.uuid4()),
                        user_query=request.message,
                        agent_response=full_answer,
                        route=final_route,
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
                    response_data={"answer": full_answer, "sources": sources_list},
                    context=cache_context,
                )

            done_payload = {
                "answer": full_answer,
                "thread_id": thread_id,
                "route": final_route,
                "latency_ms": elapsed_ms,
                "sources": sources_list,
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
