"""Document upload, web/YouTube URL ingestion, and document retrieval query endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage

from core.auth import AuthenticatedUser, get_optional_user
from core.config import settings
from core.exceptions import RateLimitException, ValidationException
from gateway import global_cost_tracker
from services.redis_service import redis_service
from ingest import ingest_file, ingest_url
from schemas.api_schemas import (
    GatewayStatsResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
    URLIngestRequest,
    URLIngestResponse,
)

logger = logging.getLogger("UploadAPI")
router = APIRouter(tags=["Ingestion & Knowledge Base"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/gateway/stats", response_model=GatewayStatsResponse, summary="Get LLM Gateway usage and caching statistics")
def get_gateway_stats() -> GatewayStatsResponse:
    """Return aggregated cost, token consumption, and cache metrics."""
    return GatewayStatsResponse(
        cost_and_usage=global_cost_tracker.get_global_stats(),
        cache=redis_service.stats(),
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a document, audio, video, or photo",
    description=(
        "Supported formats: PDF, DOCX, PPTX, XLS/XLSX; images in PNG, JPG, "
        "JPEG, WEBP, BMP, TIFF; audio/video in MP3, WAV, M4A, MP4, MKV, AVI, MOV."
    ),
)
async def upload_file(
    file: UploadFile = File(...),
    thread_id: str = "default",
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> UploadResponse:
    """Upload a file, save to storage, parse content, and index into Chroma vector database."""
    user_id = current_user.user_id
    effective_thread = thread_id if thread_id != "default" else user_id

    allowed, remaining, retry_after = redis_service.check_rate_limit(user_id, endpoint="upload")
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    filename = Path(file.filename or "").name
    if not filename:
        raise ValidationException("Please choose a valid file before uploading.")

    file_path = UPLOAD_DIR / filename
    total_bytes = 0

    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_UPLOAD_BYTES:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. The maximum upload size is {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                buffer.write(chunk)

        chunks = ingest_file(str(file_path), session_id=effective_thread)

        return UploadResponse(
            filename=filename,
            content_type=file.content_type,
            file_size_bytes=total_bytes,
            location=str(file_path),
            status="uploaded" if chunks else "uploaded",
            chunks_indexed=len(chunks),
            thread_id=effective_thread,
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Upload ingestion error: %s", error)
        raise HTTPException(status_code=400, detail=f"Upload ingestion failed: {error}") from error
    finally:
        await file.close()


@router.post("/url", response_model=URLIngestResponse, summary="Ingest webpage or YouTube transcript")
async def ingest_public_url(
    request: URLIngestRequest,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> URLIngestResponse:
    """Extract and index text from a public URL or YouTube video transcript."""
    user_id = current_user.user_id
    effective_thread = request.thread_id if request.thread_id != "default" else user_id

    allowed, remaining, retry_after = redis_service.check_rate_limit(user_id, endpoint="url")
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    try:
        url_str = str(request.url)
        chunks = ingest_url(url_str, session_id=effective_thread)
        chunk_count = len(chunks) if isinstance(chunks, list) else 1
        source_type = "youtube" if "youtube.com" in url_str or "youtu.be" in url_str else "web"

        return URLIngestResponse(
            url=url_str,
            chunks_indexed=chunk_count,
            source_type=source_type,
            status="indexed",
            thread_id=effective_thread,
        )
    except Exception as error:
        logger.exception("URL ingestion error: %s", error)
        raise HTTPException(status_code=400, detail=f"Link ingestion failed: {error}") from error


@router.post("/query", response_model=QueryResponse, summary="Query indexed documents directly")
async def query_documents(
    request: QueryRequest,
    current_user: AuthenticatedUser = Depends(get_optional_user),
) -> QueryResponse:
    """Answer a question directly from document vectors in Chroma without full conversational state."""
    user_id = current_user.user_id
    effective_thread = request.thread_id if request.thread_id != "default" else user_id

    # 1. Rate Limit Check
    allowed, remaining, retry_after = redis_service.check_rate_limit(user_id, endpoint="query")
    if not allowed:
        raise RateLimitException(
            f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            retry_after=retry_after,
        )

    # 2. Semantic Cache Check
    cache_ctx = {"user_id": effective_thread, "endpoint": "query"}
    cached = redis_service.get_semantic_cache(query=request.question, context=cache_ctx)
    if cached and isinstance(cached, dict):
        return QueryResponse(
            answer=str(cached.get("answer", "")),
            retrieved_chunks=int(cached.get("retrieved_chunks", 0)),
            route=str(cached.get("route", "semantic_cache")),
            sources=cached.get("sources", []),
        )

    # 3. Cache MISS -> Execute Pipeline
    try:
        from rag_workflow import graph_app
        from tracing import get_tracing_config

        tracing_config = get_tracing_config(
            thread_id=effective_thread,
            session_id=effective_thread,
            tags=["query", "api"],
            metadata={"query_type": "document_query", "force_document_retrieval": True},
            run_name="API Document Query",
        )

        result = graph_app.invoke(
            {
                "question": request.question,
                "messages": [HumanMessage(content=request.question)],
                "force_document_retrieval": True,
                "session_id": effective_thread,
            },
            config=tracing_config,
        )

        documents = result.get("documents", [])
        answer_text = str(result["messages"][-1].content)
        route_text = str(result.get("route", "document_rag"))
        sources_list = [doc.metadata for doc in documents]

        # 4. Store in Semantic Cache
        if answer_text and not answer_text.startswith("I encountered an issue"):
            redis_service.set_semantic_cache(
                query=request.question,
                response_data={
                    "answer": answer_text,
                    "retrieved_chunks": len(documents),
                    "route": route_text,
                    "sources": sources_list,
                },
                context=cache_ctx,
            )

        return QueryResponse(
            answer=answer_text,
            retrieved_chunks=len(documents),
            route=route_text,
            sources=sources_list,
        )
    except Exception as error:
        logger.exception("Document query error: %s", error)
        raise HTTPException(status_code=500, detail=f"Document query failed: {error}") from error
