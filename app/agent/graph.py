"""Dual-mode LangGraph RAG agent for global summaries and targeted answers."""

from __future__ import annotations

import os
from typing import Any, List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .schemas import DocumentGlobalSummary, TargetedAnswer

try:
    from app.tracing import safe_traceable
except Exception:
    try:
        from tracing import safe_traceable
    except Exception:
        safe_traceable = lambda **kw: lambda f: f


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    user_id: str
    thread_id: str
    intent: Literal["summary", "targeted"]
    retrieved_docs: List[Document]
    structured_result: DocumentGlobalSummary | TargetedAnswer
    generation: str
    has_images: bool
    image_urls: List[str]
    memory_messages: List[Any]
    suggested_followups: List[str]


SUMMARY_TERMS = ("summarize", "summary", "overview", "topic wise", "topic-wise", "summarise")


def _memory_identity(state: AgentState, config: RunnableConfig | None = None) -> tuple[str, str]:
    configurable = (config or {}).get("configurable", {}) if config else {}
    return (
        str(state.get("user_id") or configurable.get("user_id") or "default_user"),
        str(state.get("thread_id") or configurable.get("thread_id") or "default_thread"),
    )


@safe_traceable(name="hydrate_memory", run_type="chain", tags=["dual-mode-rag", "memory"])
def hydrate_memory_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Load semantic preferences and recent episodic turns into working memory."""
    try:
        from app.memory.service import global_memory_service
    except ModuleNotFoundError:
        from memory.service import global_memory_service
    user_id, thread_id = _memory_identity(state, config)
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "memory_messages": global_memory_service.hydrate_working_memory(user_id, thread_id),
    }


def get_hybrid_retriever():
    """Return the repository's hybrid retriever without eager model startup."""
    try:
        from app.retriever.hybrid_retriever import global_hybrid_retriever
    except ModuleNotFoundError:
        from retriever.hybrid_retriever import global_hybrid_retriever
    return global_hybrid_retriever


def _resolve_parents(documents: List[Document]) -> List[Document]:
    try:
        from app.retriever.hybrid_retriever import resolve_parent_documents
    except ModuleNotFoundError:
        from retriever.hybrid_retriever import resolve_parent_documents
    return resolve_parent_documents(documents)


@safe_traceable(name="classify_intent", run_type="chain", tags=["dual-mode-rag", "intent"])
def classify_intent_node(state: AgentState) -> dict[str, Any]:
    """Route empty or summary-oriented questions to global document summarization."""
    question = state.get("question", "").strip()
    normalized = question.lower()
    is_summary = not normalized or any(term in normalized for term in SUMMARY_TERMS)
    return {"intent": "summary" if is_summary else "targeted"}


def _route_intent(state: AgentState) -> Literal["summary", "targeted"]:
    return state.get("intent", "targeted")


def _image_metadata(documents: List[Document]) -> tuple[bool, List[str]]:
    try:
        from app.guardrails.output_guard import sanitize_image_url
    except ModuleNotFoundError:
        from guardrails.output_guard import sanitize_image_url

    urls: list[str] = []
    for document in documents:
        meta = document.metadata or {}
        # Single primary image_url
        if primary := meta.get("image_url"):
            if (safe := sanitize_image_url(primary)) and safe != "[BLOCKED_SUSPICIOUS_IMAGE_URL]":
                urls.append(safe)
        # List of image_urls on this page/document
        for raw_url in meta.get("image_urls", []):
            if (safe := sanitize_image_url(raw_url)) and safe != "[BLOCKED_SUSPICIOUS_IMAGE_URL]":
                urls.append(safe)

    unique_urls = list(dict.fromkeys(urls))
    return bool(unique_urls), unique_urls



@safe_traceable(name="retrieve_summary", run_type="retriever", tags=["dual-mode-rag", "retriever", "summary"])
def retrieve_summary_node(state: AgentState) -> dict[str, Any]:
    """Fetch a broad set of macro-parent contexts for document-level summaries."""
    try:
        from app.vectorstore.croma_store import get_vectorstore
    except ModuleNotFoundError:
        from vectorstore.croma_store import get_vectorstore

    query = state.get("question", "").strip() or "document overview topics key facts and actionable takeaways"
    search_kwargs: dict[str, Any] = {"k": 20}
    if state.get("session_id"):
        search_kwargs["filter"] = {"session_id": state["session_id"]}

    child_docs = get_vectorstore().similarity_search(query, **search_kwargs)
    documents = _resolve_parents(child_docs)
    has_images, image_urls = _image_metadata(documents)
    return {
        "retrieved_docs": documents,
        "has_images": has_images,
        "image_urls": image_urls,
    }


@safe_traceable(name="retrieve_targeted", run_type="retriever", tags=["dual-mode-rag", "retriever", "targeted"])
def retrieve_targeted_node(state: AgentState) -> dict[str, Any]:
    """Use weighted hybrid retrieval, then reconstruct unique parent contexts."""
    session_id = state.get("session_id", "")
    results, _dense_count, _sparse_count = get_hybrid_retriever().retrieve(
        query=state.get("question", ""),
        session_id=session_id,
        top_k=8,
        filter_dict={"session_id": session_id} if session_id else None,
    )
    child_docs = [document for document, _score in results]
    documents = _resolve_parents(child_docs)
    has_images, image_urls = _image_metadata(documents)
    return {
        "retrieved_docs": documents,
        "has_images": has_images,
        "image_urls": image_urls,
    }


def _get_llm():
    provider = os.getenv("RAG_LLM_PROVIDER", "gateway").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"))
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"))

    try:
        from app.gateway import gateway_chat_model
    except ModuleNotFoundError:
        from gateway import gateway_chat_model
    return gateway_chat_model


def _context(documents: List[Document]) -> str:
    try:
        from app.guardrails.output_guard import sanitize_image_url
    except ModuleNotFoundError:
        from guardrails.output_guard import sanitize_image_url

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source", metadata.get("source_id", "unknown"))
        date = metadata.get("date", metadata.get("timestamp", "unknown"))
        image_url = sanitize_image_url(metadata.get("image_url"))
        sections.append(
            f"[Context {index}]\n[Source] {source}\n[Date/Timestamp] {date}\n"
            f"{document.page_content}"
            + (f"\n[image_url] {image_url}" if image_url else "")
        )
    return "\n\n".join(sections) or "No retrieved context is available."


def _working_memory(state: AgentState) -> str:
    messages = state.get("memory_messages", [])
    if not messages:
        return "No persistent working memory is available."
    return "\n".join(f"{message.type}: {message.content}" for message in messages)


def _followups(question: str) -> list[str]:
    topic = question.strip().rstrip("?!. ") or "this topic"
    return [
        f"What are the key details of {topic}?",
        f"What should I investigate next about {topic}?",
    ]


CONFLICT_AWARE_SYSTEM_PROMPT = """You are a precise enterprise AI assistant. Answer using ONLY the retrieved context.

If different documents state conflicting facts, do not silently choose, merge, or guess. Check [Source] and [Date/Timestamp]. Prefer a clearly newer or authoritative source and explain that it supersedes the older one. If unresolved, explicitly state both facts and their sources. Never invent unsupported details.
For zero relevant context, state exactly: "The requested information is not available in the indexed documentation." For partial context, answer only supported components and append a section titled "### ⚠️ Missing Information / Context" listing what could not be verified.
"""


@safe_traceable(name="generate_summary", run_type="llm", tags=["dual-mode-rag", "generate", "summary"])
def generate_summary_node(state: AgentState) -> dict[str, Any]:
    """Generate a structured topic-tree summary from diverse parent contexts."""
    prompt = f"""{CONFLICT_AWARE_SYSTEM_PROMPT}
Create a complete global document summary as a topic tree. Include an executive overview,
major topics, key bullets, data tables when useful, image URLs from context, and actionable takeaways.
Return only the requested structured schema.

Retrieved context:
{_context(state.get('retrieved_docs', []))}

Persistent working memory:
{_working_memory(state)}
"""
    result = _get_llm().with_structured_output(DocumentGlobalSummary).invoke(prompt)
    return {"structured_result": result}


@safe_traceable(name="generate_targeted", run_type="llm", tags=["dual-mode-rag", "generate", "targeted"])
def generate_targeted_node(state: AgentState) -> dict[str, Any]:
    """Generate a structured targeted answer with evidence, citations, and conflict reporting."""
    prompt = f"""{CONFLICT_AWARE_SYSTEM_PROMPT}
Answer the user's targeted question. Put the direct answer first, then supporting evidence.
Use a Markdown comparison table when comparing values. Include image_url only when supplied by context.
Return only the requested structured schema.

Question:
{state.get('question', '')}

Retrieved context:
{_context(state.get('retrieved_docs', []))}

Persistent working memory:
{_working_memory(state)}
"""
    result = _get_llm().with_structured_output(TargetedAnswer).invoke(prompt)
    return {"structured_result": result}


def _table(value: str | None) -> str:
    return f"\n\n{value.strip()}" if value and value.strip() else ""


@safe_traceable(name="format_markdown", run_type="chain", tags=["dual-mode-rag", "format"])
def format_markdown_node(state: AgentState) -> dict[str, Any]:
    """Render either structured response schema as clean Markdown."""
    try:
        from app.guardrails.output_guard import sanitize_image_url
    except ModuleNotFoundError:
        from guardrails.output_guard import sanitize_image_url

    result = state.get("structured_result")
    if isinstance(result, DocumentGlobalSummary):
        lines = [f"# {result.document_title}", "", result.executive_overview, "", "## Topics"]
        for topic in result.topics:
            lines.extend([f"### {topic.topic_title}", topic.overview])
            lines.extend(f"- {bullet}" for bullet in topic.key_bullets)
            lines.append(_table(topic.data_table))
            if topic.image_url:
                safe_url = sanitize_image_url(topic.image_url)
                lines.extend(["", f"![Image]({safe_url})" if safe_url and safe_url != "[BLOCKED_SUSPICIOUS_IMAGE_URL]" else "[BLOCKED_SUSPICIOUS_IMAGE_URL]"])
        lines.extend(["", "## Actionable Takeaways"])
        lines.extend(f"- {takeaway}" for takeaway in result.actionable_takeaways)
        return {
            "generation": "\n".join(lines).strip(),
            "suggested_followups": _followups(state.get("question", "")),
        }

    if isinstance(result, TargetedAnswer):
        lines = ["# Answer", result.direct_answer, "", "## Supporting Evidence"]
        lines.extend(f"- {evidence}" for evidence in result.supporting_evidence)
        lines.append(_table(result.comparison_table))
        if result.conflicts_detected:
            lines.extend(["", "## Conflicts Detected", result.conflicts_detected])
        if result.citations:
            lines.extend(["", "## Citations"])
            lines.extend(f"- {citation}" for citation in result.citations)
        if result.image_url:
            safe_url = sanitize_image_url(result.image_url)
            lines.extend(["", f"![Image]({safe_url})" if safe_url and safe_url != "[BLOCKED_SUSPICIOUS_IMAGE_URL]" else "[BLOCKED_SUSPICIOUS_IMAGE_URL]"])
        return {
            "generation": "\n".join(lines).strip(),
            "suggested_followups": _followups(state.get("question", "")),
        }

    return {
        "generation": "No structured answer was generated.",
        "suggested_followups": _followups(state.get("question", "")),
    }


@safe_traceable(name="persist_memory", run_type="chain", tags=["dual-mode-rag", "memory", "sqlite"])
def persist_memory_node(state: AgentState) -> dict[str, Any]:
    """Persist the final graph answer and UI follow-up chips to SQLite."""
    try:
        from app.memory.schemas import ConversationTurn
        from app.memory.service import global_memory_service
    except ModuleNotFoundError:
        from memory.schemas import ConversationTurn
        from memory.service import global_memory_service
    user_id, thread_id = _memory_identity(state)
    answer = state.get("generation", "")
    if answer:
        global_memory_service.store.add_turn_to_thread(
            thread_id,
            user_id,
            ConversationTurn(
                user_query=state.get("question", ""),
                agent_response=answer,
                suggested_followups=state.get("suggested_followups", [])[:3],
                metadata={"intent": state.get("intent", "")},
            ),
        )
    return {}


workflow = StateGraph(AgentState)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_summary", retrieve_summary_node)
workflow.add_node("retrieve_targeted", retrieve_targeted_node)
workflow.add_node("generate_summary", generate_summary_node)
workflow.add_node("generate_targeted", generate_targeted_node)
workflow.add_node("format_markdown", format_markdown_node)
workflow.add_node("hydrate_memory", hydrate_memory_node)
workflow.add_node("persist_memory", persist_memory_node)
workflow.add_edge(START, "hydrate_memory")
workflow.add_edge("hydrate_memory", "classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    _route_intent,
    {"summary": "retrieve_summary", "targeted": "retrieve_targeted"},
)
workflow.add_edge("retrieve_summary", "generate_summary")
workflow.add_edge("retrieve_targeted", "generate_targeted")
workflow.add_edge("generate_summary", "format_markdown")
workflow.add_edge("generate_targeted", "format_markdown")
workflow.add_edge("format_markdown", "persist_memory")
workflow.add_edge("persist_memory", END)

rag_agent_app = workflow.compile()


__all__ = [
    "AgentState",
    "classify_intent_node",
    "format_markdown_node",
    "generate_summary_node",
    "generate_targeted_node",
    "get_hybrid_retriever",
    "rag_agent_app",
    "retrieve_summary_node",
    "retrieve_targeted_node",
    "hydrate_memory_node",
    "persist_memory_node",
    "workflow",
]
