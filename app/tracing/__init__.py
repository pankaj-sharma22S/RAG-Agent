"""Centralized LangSmith tracing and observability package for RAG-Agent."""

from .langsmith import (
    DEFAULT_PROJECT,
    EMBEDDING_MODEL_NAME,
    MODEL_NAME,
    VECTOR_STORE_NAME,
    get_tracing_config,
    is_tracing_enabled,
    safe_traceable,
    setup_tracing,
    trace_stage,
)

__all__ = [
    "DEFAULT_PROJECT",
    "MODEL_NAME",
    "EMBEDDING_MODEL_NAME",
    "VECTOR_STORE_NAME",
    "setup_tracing",
    "is_tracing_enabled",
    "get_tracing_config",
    "safe_traceable",
    "trace_stage",
]
