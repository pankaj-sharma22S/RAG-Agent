"""LangSmith tracing configuration, payload sanitization, and enterprise observability utilities."""

from __future__ import annotations

import functools
import logging
import os
import socket
from typing import Any, Callable, Sequence
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("LangSmithTracing")

# Automatically sync LANGSMITH_* to standard LANGCHAIN_* environment variables
if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes", "on"}:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-app")
if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

DEFAULT_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-app")
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
VECTOR_STORE_NAME = "chroma"
HOSTNAME = socket.gethostname()
PID = os.getpid()
INSTANCE_ID = f"{HOSTNAME}:{PID}"


def sanitize_trace_payload(data: Any, max_string_len: int = 2500, depth: int = 0) -> Any:
    """Recursively redact secrets and truncate oversized strings to prevent trace bloat."""
    if depth > 8:
        return "<max_depth_exceeded>"

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            lower_key = str(key).lower()
            if any(s in lower_key for s in ["key", "secret", "password", "token", "auth", "cookie", "cert", "credential"]):
                sanitized[key] = "[REDACTED_SECRET]"
            else:
                sanitized[key] = sanitize_trace_payload(value, max_string_len=max_string_len, depth=depth + 1)
        return sanitized

    if isinstance(data, (list, tuple)):
        return [sanitize_trace_payload(item, max_string_len=max_string_len, depth=depth + 1) for item in data[:50]]

    if isinstance(data, str):
        if len(data) > max_string_len:
            return data[:max_string_len] + f" ... [truncated {len(data) - max_string_len} chars]"
        return data

    return data


def setup_tracing(project_name: str | None = None) -> bool:
    """Initialize LangSmith environment variables and verify configuration.

    Returns True if tracing is active and configured.
    """
    load_dotenv()
    if project_name:
        os.environ["LANGCHAIN_PROJECT"] = project_name
    elif "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = DEFAULT_PROJECT

    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes", "on"}:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

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
    """Construct a LangChain/LangGraph RunnableConfig dictionary with standard observability and thread grouping."""
    active_thread = (thread_id or session_id or "default").strip()
    active_session = (session_id or thread_id or "default").strip()

    base_tags = ["rag-app", "multimodal-rag", f"thread:{active_thread}"]
    if tags:
        for tag in tags:
            if tag and tag not in base_tags:
                base_tags.append(str(tag))

    base_metadata: dict[str, Any] = {
        "thread_id": active_thread,
        "session_id": active_session,
        "conversation_id": active_thread,
        "thread": active_thread,
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_store": VECTOR_STORE_NAME,
        "backend_instance": INSTANCE_ID,
    }

    if metadata:
        clean_meta = sanitize_trace_payload(metadata)
        if isinstance(clean_meta, dict):
            base_metadata.update(clean_meta)

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": active_thread,
            "session_id": active_session,
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
    reduce_fn: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Safe decorator wrapping functions with LangSmith @traceable.

    If langsmith is unavailable or disabled, gracefully falls back to the original function.
    """
    try:
        from langsmith import traceable

        combined_tags = ["rag-app"]
        if tags:
            for t in tags:
                if t and t not in combined_tags:
                    combined_tags.append(str(t))

        combined_metadata: dict[str, Any] = {
            "backend_instance": INSTANCE_ID,
            "vector_store": VECTOR_STORE_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
        if metadata:
            clean_meta = sanitize_trace_payload(metadata)
            if isinstance(clean_meta, dict):
                combined_metadata.update(clean_meta)

        return traceable(
            name=name,
            run_type=run_type,
            tags=combined_tags,
            metadata=combined_metadata,
            reduce_fn=reduce_fn,
        )
    except Exception as err:
        logger.debug("LangSmith traceable initialization bypassed: %s", err)

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
    combined_tags = ["rag-app", "pipeline-stage"]
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
        clean_meta = sanitize_trace_payload(metadata)
        if isinstance(clean_meta, dict):
            combined_metadata.update(clean_meta)

    return safe_traceable(
        name=stage_name,
        run_type=run_type,
        tags=combined_tags,
        metadata=combined_metadata,
    )
