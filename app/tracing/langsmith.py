"""LangSmith tracing configuration and observability utilities for RAG-Agent."""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Sequence
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-agent")
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
VECTOR_STORE_NAME = "chroma"


def setup_tracing(project_name: str | None = None) -> bool:
    """Initialize LangSmith environment variables and verify configuration.

    Returns True if tracing is active and configured.
    """
    load_dotenv()
    if project_name:
        os.environ["LANGCHAIN_PROJECT"] = project_name
    elif "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = DEFAULT_PROJECT

    enabled = is_tracing_enabled()
    return enabled


def is_tracing_enabled() -> bool:
    """Check whether LangSmith tracing is explicitly enabled in the environment."""
    tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "false").strip().lower()
    return tracing_flag in {"true", "1", "yes", "on"}


def get_tracing_config(
    session_id: str | None = None,
    thread_id: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Construct a LangChain/LangGraph RunnableConfig dictionary with standard observability.

    Attaches session isolation IDs, model and vector store identifiers, tags, and run metadata.
    """
    active_thread = (thread_id or session_id or "default").strip()
    active_session = (session_id or thread_id or "default").strip()

    base_tags = ["rag-agent"]
    if tags:
        for tag in tags:
            if tag and tag not in base_tags:
                base_tags.append(str(tag))

    base_metadata: dict[str, Any] = {
        "thread_id": active_thread,
        "session_id": active_session,
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_store": VECTOR_STORE_NAME,
    }

    if metadata:
        for key, value in metadata.items():
            # Avoid logging sensitive fields
            if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                continue
            base_metadata[key] = value

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": active_thread,
        },
        "tags": base_tags,
        "metadata": base_metadata,
    }

    if run_name:
        config["run_name"] = run_name

    return config


def safe_traceable(
    name: str | None = None,
    run_type: str = "chain",
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Safe decorator wrapping functions with LangSmith @traceable.

    If langsmith is unavailable or disabled, gracefully falls back to the original function.
    """
    try:
        from langsmith import traceable

        return traceable(
            name=name,
            run_type=run_type,
            tags=list(tags) if tags else None,
            metadata=metadata,
        )
    except Exception:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper
        return decorator


def trace_stage(
    stage_name: str,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_type: str = "chain",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator to trace a specific stage in the RAG pipeline."""
    combined_tags = ["rag-agent"]
    if tags:
        for t in tags:
            if t not in combined_tags:
                combined_tags.append(t)

    combined_metadata: dict[str, Any] = {
        "stage": stage_name,
        "model": MODEL_NAME,
        "vector_store": VECTOR_STORE_NAME,
    }
    if metadata:
        combined_metadata.update(metadata)

    return safe_traceable(
        name=stage_name,
        run_type=run_type,
        tags=combined_tags,
        metadata=combined_metadata,
    )
