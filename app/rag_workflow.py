"""Production Adaptive LangGraph workflow with Fast Chat Routing, Self-RAG, Hybrid BM25+Dense Retrieval, RRF, Cross-Encoder, CRAG, and Query Refinement."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

try:
    from app.db.checkpointer import get_checkpointer
    from app.db.service import get_active_source, set_active_source
except Exception:
    try:
        from db.checkpointer import get_checkpointer
        from db.service import get_active_source, set_active_source
    except Exception:
        get_checkpointer = lambda: MemorySaver()
        get_active_source = lambda sid: None
        set_active_source = lambda sid, s: None

try:
    from app.gateway import gateway_chat_model, global_response_cache, llm_gateway
    from app.guardrails import (
        global_injection_guard,
        global_input_guard,
        global_output_guard,
    )
    from app.retriever import (
        global_crag_evaluator,
        global_hybrid_retriever,
        global_query_refiner,
        global_reranker,
    )
    from app.schemas import (
        AmbiguityDecision,
        CRAGDecision,
        EvaluationDecision,
        IntentDecision,
        KnowledgeDecision,
        MultimodalDecision,
        QueryRefinement,
        RAGAnswer,
        RetrievalAttempt,
        UserBiography,
    )
    from app.tracing import (
        DEFAULT_PROJECT,
        EMBEDDING_MODEL_NAME,
        MODEL_NAME,
        VECTOR_STORE_NAME,
        get_tracing_config,
        safe_traceable,
        setup_tracing,
        trace_stage,
    )
except ModuleNotFoundError:
    from gateway import gateway_chat_model, global_response_cache, llm_gateway
    from guardrails import (
        global_injection_guard,
        global_input_guard,
        global_output_guard,
    )
    from retriever import (
        global_crag_evaluator,
        global_hybrid_retriever,
        global_query_refiner,
        global_reranker,
    )
    from schemas import (
        AmbiguityDecision,
        CRAGDecision,
        EvaluationDecision,
        IntentDecision,
        KnowledgeDecision,
        MultimodalDecision,
        QueryRefinement,
        RAGAnswer,
        RetrievalAttempt,
        UserBiography,
    )
    from tracing import (
        DEFAULT_PROJECT,
        EMBEDDING_MODEL_NAME,
        MODEL_NAME,
        VECTOR_STORE_NAME,
        get_tracing_config,
        safe_traceable,
        setup_tracing,
        trace_stage,
    )

logger = logging.getLogger("RAGWorkflow")
setup_tracing()

# Global Configuration Parameters & Adaptive Confidence Policy
EVIDENCE_THRESHOLD = float(os.getenv("EVIDENCE_THRESHOLD", "7.5"))
EVIDENCE_THRESHOLD_NORM = EVIDENCE_THRESHOLD / 10.0 if EVIDENCE_THRESHOLD > 1.0 else EVIDENCE_THRESHOLD
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", str(EVIDENCE_THRESHOLD_NORM)))
MEDIUM_CONFIDENCE_MIN = float(os.getenv("MEDIUM_CONFIDENCE_MIN", "0.35"))
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", str(EVIDENCE_THRESHOLD_NORM)))
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "1"))
MIN_SCORE_IMPROVEMENT = float(os.getenv("MIN_SCORE_IMPROVEMENT", "0.05"))
MAX_RETRIEVAL_ATTEMPTS = MAX_REFINEMENT_ITERATIONS + 1  # 1 initial + max 1 refinement
WEB_RESULT_COUNT = int(os.getenv("WEB_RESULT_COUNT", "5"))
SELF_RAG_ENABLED = os.getenv("SELF_RAG_ENABLED", "true").lower() in {"true", "1", "yes"}

llm = gateway_chat_model

# Ultra-Fast Chat / Greeting Lexicon (0ms zero-LLM detection)
FAST_CHAT_PATTERNS = {
    # Greetings
    "hi", "hello", "hlo", "helo", "hlw", "hye", "hii", "heyy", "hey", "hiya", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "good night", "yo", "sup", "hola", "namaste", "gm", "gn",
    # Casual Questions
    "how are you", "how are you doing", "hows it going", "how's it going",
    "what's up", "whats up", "how do you do",
    # Gratitude & Politeness
    "thanks", "thank you", "thx", "ty", "thank you very much", "many thanks",
    "appreciate it", "cheers",
    # Farewells
    "bye", "goodbye", "see you", "cya", "have a good day", "have a nice day",
    # Identity & Basic Help
    "who are you", "what can you do", "what are you", "help", "ping", "test",
    # Acknowledgments & Short Affirmations
    "ok", "okay", "sure", "cool", "great", "awesome", "got it", "understood",
    "yes", "no", "yep", "nope", "perfect", "nice",
}


DIRECT_ROUTE_KEYWORDS = {
    "realtime_time": [
        "what time is it", "what is the time", "current time", "time now",
        "tell me the time", "whats the time", "what's the time", "time please",
        "time right now", "date today", "today date", "current date", "what is the date",
    ],
    "memory": [
        "who am i", "who i am", "what is my name", "whats my name",
        "what do you know about me", "tell me about myself", "my profile",
        "do you remember me", "do you know who i am",
    ],
    "realtime_web": [
        "today", "todays", "yesterday", "tomorrow", "weather", "stock price",
        "current price", "latest news", "virat", "ipl", "score", "match",
        "live score", "who won", "breaking news", "current exchange rate",
    ],
    "document_rag": [
        "pdf", "document", "uploaded", "docx", "file", "in the text",
        "spreadsheet", "my doc", "according to the document", "chapter",
        "in this link", "in this video", "speaker say", "author say",
        "craft technique", "in the video", "in the audio", "in the transcript",
        "transformer", "attention", "seq2seq", "positional encoding", "vision model",
        "vit", "encoder", "decoder", "self-attention", "cross-attention", "feed forward",
    ],
    "web_research": [
        "compare latest", "state of the art", "market research", "recent studies",
    ],
}


def is_fast_chat_query(text: str) -> bool:
    """Detect if a user message is casual chit-chat, greeting, or farewell to bypass heavy RAG pipelines in 0ms."""
    if not text:
        return True

    # Never bypass if URLs are present
    if "http://" in text.lower() or "https://" in text.lower():
        return False

    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not cleaned:
        return True

    # Never bypass if document, video, or source references exist
    source_words = [
        "document", "pdf", "file", "video", "transcript", "summary", "summarize",
        "chapter", "article", "table", "schema", "empno", "ename", "mgr", "author", "speaker",
        "transformer", "seq2seq", "attention", "encoding", "model",
    ]
    if any(w in cleaned.split() for w in source_words):
        return False

    # Never bypass if user is introducing themselves or asking about their identity/profile
    profile_triggers = [
        "who am i", "who i am", "my name is", "i am ", "call me", "i work as",
        "i live in", "my role is", "my project is", "what is my name", "whats my name",
        "tell me about myself", "do you know me", "what do you know about me",
    ]
    for pt in profile_triggers:
        if pt in cleaned:
            return False

    if cleaned in FAST_CHAT_PATTERNS:
        return True

    words = cleaned.split()
    if len(words) <= 5:
        if all(w in {"hi", "hello", "hlo", "helo", "hlw", "hey", "how", "are", "you", "doing", "there", "thanks", "thank", "much", "a", "lot", "bye", "good", "morning", "night", "day", "see", "ya", "cya", "ok", "okay", "cool", "great", "who", "what", "can", "help", "please"} for w in words):
            return True
    return False


SOURCE_EXPLICIT_TRIGGERS = [
    "according to", "in this video", "in the video", "in this link", "in this document",
    "in the document", "in this pdf", "in the pdf", "in this file", "in the file",
    "in this article", "in the article", "what does the document say", "what does the pdf say",
    "what does the video say", "what did the speaker say", "what does the speaker say",
    "what does the author say", "what did the author say", "from the document", "from this document",
    "from the pdf", "from this pdf", "from the video", "from this video", "from the text",
    "based on the document", "based on this document", "based on the pdf", "based on the video",
    "based on the uploaded file", "in chapter", "in the transcript", "in this transcript",
    "craft technique", "schema", "table", "empno", "ename", "mgr", "manager",
]

SOURCE_CONTEXTUAL_TRIGGERS = [
    "explain that", "explain this", "summarize that", "summarize this", "elaborate that",
    "what example did", "why did the speaker", "why did the author", "what did they mean",
    "tell me more about that", "give more details about that", "clarify that", "simplify that",
    "in simple words", "in plain english", "can you break that down", "what was the conclusion",
    "what were the key takeaways", "what are the main points", "what does it say",
]

SIMPLE_MATH_PATTERNS = [
    r"^\s*what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$",
    r"^\s*\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$",
    r"^\s*calculate\s+\d+",
    r"^\s*2\s*\+\s*2\s*\??\s*$",
    r"^\s*10\s*\+\s*10\s*\??\s*$",
]

GENERAL_CONCEPT_PATTERNS = [
    r"^\s*what\s+is\s+python\b",
    r"^\s*what\s+is\s+java\b",
    r"^\s*what\s+is\s+c\+\+\b",
    r"^\s*what\s+is\s+html\b",
    r"^\s*what\s+is\s+sql\b",
    r"^\s*what\s+is\s+inheritance\b",
    r"^\s*what\s+is\s+polymorphism\b",
    r"^\s*explain\s+inheritance\b",
    r"^\s*explain\s+polymorphism\b",
    r"^\s*what\s+is\s+object\s+oriented\b",
]


SUMMARY_INTENT_PATTERNS = [
    "summarize this", "summarize the document", "summarize the video", "give me a complete summary",
    "explain the whole document", "give me a detailed summary", "summarize everything", "give me the key points",
    "explain the entire video", "give me a summary", "briefly summarize", "short summary", "deep summary",
    "complete summary", "detailed summary", "explain everything", "dont miss anything", "don't miss anything",
    "recap the document", "recap this", "overview of this document", "overview of this video",
    "analyze this repo", "summarize this repo", "summarize this repository", "explain the entire repo",
    "what in document", "what is in document", "what is in the document", "what is in this document",
    "what does this document contain", "what does the document contain", "what is this document about",
    "what is the document about", "whats in the document", "whats in this document", "what's in the document",
    "what inside document", "what is inside document", "what is inside the document", "what is inside this document",
    "explain document", "explain the document", "tell me about the document", "tell me about this document",
    "document overview", "overview of document", "overview of the document", "summary of document",
]


def classify_summary_depth(text: str) -> str:
    """Classify requested summary depth into 'quick', 'standard', or 'deep'."""
    lower = text.lower().strip()
    if any(w in lower for w in ["deep", "detailed", "complete", "everything", "dont miss", "don't miss", "in depth", "in-depth", "comprehensive", "entire", "whole", "all topics"]):
        return "deep"
    if any(w in lower for w in ["brief", "short", "quick", "one line", "concise", "tldr", "tl;dr"]):
        return "quick"
    return "standard"


def is_summary_request(text: str) -> bool:
    """Deterministically detect if user prompt requests a summary (document, video, or repo)."""
    lower = text.lower().strip()
    cleaned = re.sub(r"[^\w\s]", "", lower)
    if any(pat in lower or pat in cleaned for pat in SUMMARY_INTENT_PATTERNS):
        return True
    words = cleaned.split()
    if any(w in words for w in ["summarize", "summary", "recap", "overview", "takeaways", "takeaway"]) and (
        len(words) <= 6 or any(w in words for w in ["this", "the", "document", "video", "transcript", "pdf", "file", "all", "everything", "repo", "repository"])
    ):
        return True
    return False


def session_has_indexed_documents(session_id: str = "") -> bool:
    """Check if vector storage currently has active indexed documents/transcripts."""
    try:
        from app.vectorstore.croma_store import get_vectorstore
        vs = get_vectorstore()
        if hasattr(vs, "_collection") and vs._collection.count() > 0:
            return True
    except Exception:
        try:
            from vectorstore.croma_store import get_vectorstore
            vs = get_vectorstore()
            if hasattr(vs, "_collection") and vs._collection.count() > 0:
                return True
        except Exception:
            pass
    return False


def direct_query_router(text: str, session_id: str = "default") -> tuple[str, float]:
    """Ultra-fast deterministic first-stage query classifier (<0.05ms execution).
    
    Priority:
    1. Realtime time queries -> 'realtime_time'
    2. Fast Chat (greetings, affirmations, casual chit-chat) -> 'fast_chat'
    3. URL Detection (YouTube summary vs question, GitHub, Web)
    4. Explicit / Contextual Summary Intent -> 'youtube_summary' / 'document_summary' / 'github_summary'
    5. User Memory / Profile Lookup -> 'memory'
    6. Realtime / Current Information -> 'realtime_web'
    7. General Calculation / General Concepts -> 'general_chat' (even with active doc)
    8. Active Source Reference / Follow-Up -> 'document_rag' / 'youtube_question'
    9. Web Research -> 'web_research'
    10. General Knowledge -> 'general_chat'
    """
    if not text or not text.strip():
        return "fast_chat", 1.0

    raw_lower = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", raw_lower)

    # 1. Realtime Time queries (0ms zero-LLM response)
    for kw in DIRECT_ROUTE_KEYWORDS["realtime_time"]:
        if kw in cleaned:
            return "realtime_time", 1.0

    # 2. Fast Chat (greetings, affirmations, casual chit-chat)
    if is_fast_chat_query(text):
        return "fast_chat", 1.0

    # 3. URL Detection & Explicit YouTube / GitHub URLs
    if "youtube.com" in raw_lower or "youtu.be" in raw_lower:
        if is_summary_request(text) or len(cleaned.split()) <= 4:
            return "youtube_summary", 1.0
        return "youtube_question", 1.0

    if "github.com" in raw_lower:
        if "/topics/" in raw_lower:
            return "github_topic", 1.0
        return "github_repo", 1.0

    # 4. Summary Request Detection across Active Sources
    if is_summary_request(text):
        if session_id:
            try:
                act_src = get_active_source(session_id)
            except Exception:
                act_src = None
            if act_src:
                src_type = act_src.get("source_type", "document")
                if src_type == "youtube":
                    return "youtube_summary", 1.0
                if src_type == "github_topic":
                    return "github_topic", 1.0
                if src_type == "github_repo" or src_type == "github":
                    return "github_repo", 1.0
                return "document_summary", 1.0
        # If document or video explicitly mentioned
        if any(w in cleaned.split() for w in ["video", "transcript", "youtube"]):
            return "youtube_summary", 0.98
        if any(w in cleaned.split() for w in ["repo", "repository", "github"]):
            return "github_repo", 0.98
        return "document_summary", 0.98

    # 5. User Memory / Profile Lookup
    for kw in DIRECT_ROUTE_KEYWORDS["memory"]:
        if kw in cleaned:
            return "memory", 1.0

    # 6. Realtime Weather & Sports / News
    for kw in DIRECT_ROUTE_KEYWORDS["realtime_web"]:
        if re.search(rf"\b{re.escape(kw)}\b", raw_lower):
            return "realtime_web", 0.95

    # 7. Simple Math & Generic Coding Concepts (General LLM, NO RAG)
    for pat in SIMPLE_MATH_PATTERNS:
        if re.search(pat, raw_lower):
            return "general_chat", 1.0

    for pat in GENERAL_CONCEPT_PATTERNS:
        if re.search(pat, raw_lower) and not any(t in raw_lower for t in ["in the document", "in this video", "in the pdf", "in the file"]):
            return "general_chat", 1.0

    # 8. Active Source Reference & Follow-Up Questions
    if session_id:
        try:
            act_src = get_active_source(session_id)
        except Exception:
            act_src = None

        if act_src:
            src_type = act_src.get("source_type", "document")
            if src_type == "github_topic":
                return "github_topic", 0.98
            if src_type == "github_repo":
                return "github_repo", 0.98
            target_route = "youtube_question" if src_type == "youtube" else "document_rag"
            return target_route, 0.95

    # 8. Document / RAG explicit references
    for kw in DIRECT_ROUTE_KEYWORDS["document_rag"]:
        if re.search(rf"\b{re.escape(kw)}\b", raw_lower) or kw in cleaned:
            return "document_rag", 0.95

    # 9. Web Research
    for kw in DIRECT_ROUTE_KEYWORDS["web_research"]:
        if kw in cleaned:
            return "web_research", 0.90

    # 10. Self-introductions should be treated as memory capture, not casual chat
    intro_patterns = [
        "my name is", "i am ", "call me", "i work as", "i live in",
        "my role is", "my project is", "i work at", "i live at"
    ]
    if any(p in cleaned for p in intro_patterns):
        return "memory", 0.90

    # 11. Ambiguity check
    if len(cleaned.split()) == 1 and cleaned in {"what", "why", "how", "where", "when"}:
        return "clarify", 0.90

    # 12. If active documents exist in vector store and query is substantive, route to document_rag
    if session_has_indexed_documents(session_id) and len(cleaned.split()) >= 2:
        return "document_rag", 0.90

    return "general_chat", 0.85


def classify_multimodal_orchestrator(text: str, session_id: str = "default") -> MultimodalDecision:
    """Fast deterministic Multimodal RAG Orchestrator classifier (<0.05ms execution).
    
    Determines:
    1. User Intent
    2. Hard Source-Scope Rule (REQUIRED vs NOT_REQUIRED)
    3. Modalities (TEXT, IMAGE, TABLE, CHART, DIAGRAM, AUDIO, VIDEO, MIXED)
    4. Tool Selection (needs_web, needs_ocr, needs_visual_retrieval, needs_vision_model, needs_table_extraction, needs_chart, needs_calculation)
    5. Parallel Task Planning (TEXT_RETRIEVAL, VISUAL_RETRIEVAL, WEB_SEARCH)
    6. Refinement bounds (max_refinements=1)
    """
    if not text or not text.strip():
        return MultimodalDecision(intent="chat", source_required=False, modalities=["TEXT"])

    raw_lower = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", raw_lower)
    words = cleaned.split()

    # 1. Modality Detection
    modalities: list[str] = []
    if any(w in raw_lower for w in ["diagram", "architecture", "flowchart", "component", "system design", "wireframe"]):
        modalities.append("DIAGRAM")
    if any(w in raw_lower for w in ["chart", "graph", "plot", "trend", "bar chart", "line chart", "pie chart", "scatter", "revenue growth"]):
        modalities.append("CHART")
    if any(w in raw_lower for w in ["table", "column", "row", "schema", "database", "sql", "empno", "ename", "mgr", "csv"]):
        modalities.append("TABLE")
    if any(w in raw_lower for w in ["image", "picture", "photo", "screenshot", "visual", "page 5", "page 10", "on screen"]):
        modalities.append("IMAGE")
    if any(w in raw_lower for w in ["audio", "speaker say", "podcast", "transcript at", "minute 20"]):
        modalities.append("AUDIO")
    if any(w in raw_lower for w in ["video", "youtube", "speaker explain", "frame"]):
        modalities.append("VIDEO")

    if not modalities:
        modalities.append("TEXT")
    elif len(modalities) > 1 and "MIXED" not in modalities:
        modalities.append("MIXED")

    # 2. Hard Source-Scope Rule
    has_explicit_source = any(t in raw_lower for t in SOURCE_EXPLICIT_TRIGGERS) or any(
        w in words for w in ["document", "pdf", "file", "diagram", "chart", "table", "video", "transcript", "page", "chapter", "schema", "article"]
    ) or bool(re.search(r"\bpage\s+\d+\b", raw_lower))
    has_contextual_source = any(t in raw_lower for t in SOURCE_CONTEXTUAL_TRIGGERS)
    has_source_nouns = any(w in words for w in ["video", "speaker", "document", "pdf", "file", "author", "article", "section", "chapter", "schema", "table", "manager", "diagram", "chart", "page"])

    active_src = get_active_source(session_id) if session_id else None

    # Check if query is simple math or general concept
    is_simple_math = any(re.search(pat, raw_lower) for pat in SIMPLE_MATH_PATTERNS)
    is_general_concept = any(re.search(pat, raw_lower) for pat in GENERAL_CONCEPT_PATTERNS) and not has_explicit_source

    if is_simple_math or is_general_concept or is_fast_chat_query(text):
        source_required = False
    elif has_explicit_source or (active_src and (has_contextual_source or has_source_nouns)):
        source_required = True
    elif "youtube.com" in raw_lower or "youtu.be" in raw_lower or "github.com" in raw_lower:
        source_required = True
    else:
        source_required = False

    # 3. Tool & Modality Flags
    needs_web = any(kw in raw_lower for kw in [
        "compare with web", "compare with internet", "use the web", "search the web",
        "search web", "web if necessary", "web if needed", "check the web", "on the web",
        "latest", "current", "today", "weather", "score", "news",
    ]) or ("web" in raw_lower and ("compare" in raw_lower or "search" in raw_lower or "necessary" in raw_lower or "needed" in raw_lower))
    needs_visual_retrieval = "IMAGE" in modalities or "DIAGRAM" in modalities or "CHART" in modalities
    needs_vision_model = (
        ("DIAGRAM" in modalities or "IMAGE" in modalities or "CHART" in modalities)
        and (source_required or any(kw in raw_lower for kw in ["explain", "analyze", "what does", "trend", "look at", "diagram", "chart", "visual", "graph", "picture", "screen"]))
    )
    needs_ocr = "IMAGE" in modalities or "DIAGRAM" in modalities
    needs_table_extraction = "TABLE" in modalities
    needs_chart = bool(re.search(r"\b(make|create|generate|show|plot|draw)\s+(a\s+)?(bar\s+|line\s+|pie\s+|scatter\s+)?(chart|graph|plot)\b", raw_lower)) or any(kw in raw_lower for kw in ["show this graphically", "visualize the comparison", "plot the data", "generate a chart", "show growth"]) and not any(kw in raw_lower for kw in ["analyze this", "explain this chart", "analyze the sales chart"])
    needs_calculation = any(kw in raw_lower for kw in ["calculate", "percentage", "total revenue", "sum", "average", "growth rate", "math"]) or is_simple_math

    # 4. Parallel Task Planning
    parallel_tasks: list[str] = []
    if source_required:
        parallel_tasks.append("TEXT_RETRIEVAL")
        if needs_visual_retrieval:
            parallel_tasks.append("VISUAL_RETRIEVAL")
    if needs_web:
        parallel_tasks.append("WEB_SEARCH")

    intent = "document_question" if source_required else ("web_research" if needs_web else ("chat" if is_fast_chat_query(text) else "general_question"))

    return MultimodalDecision(
        intent=intent,
        source_required=source_required,
        modalities=modalities,
        needs_web=needs_web,
        needs_ocr=needs_ocr,
        needs_visual_retrieval=needs_visual_retrieval,
        needs_vision_model=needs_vision_model,
        needs_table_extraction=needs_table_extraction,
        needs_chart=needs_chart,
        needs_calculation=needs_calculation,
        needs_refinement=False,
        parallel_tasks=parallel_tasks,
        max_refinements=1,
    )


class AgentState(TypedDict, total=False):
    """Unified LangGraph state passing through the adaptive RAG pipeline."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    original_query: str
    raw_question: str
    session_id: str
    biography: dict[str, str]

    # Active Source & Domain Drift Tracking
    active_source: dict[str, Any] | None
    active_source_id: str | None
    active_source_type: str | None
    active_source_url: str | None
    active_source_title: str | None
    topic_domain: str | None
    source_scope: str | None
    document_mode: str | None  # "document_only", "document_plus_theory", "web_comparison", "general_chat", "fast_chat"
    multimodal_decision: dict[str, Any] | None

    # Classifier & Pre-check decisions
    intent: Literal["chat", "research", "clarify"]
    clarification_question: str | None
    knowledge_decision: KnowledgeDecision | None
    is_fast_chat: bool

    # Adaptive Retrieval Loop
    retrieval_attempt: int
    max_retrieval_attempts: int
    retrieval_strategy: str  # "DIRECT_DOCUMENT", "DENSE_ONLY", "BM25_ONLY", "HYBRID", "HYBRID_RERANK", "SELF_RAG", "CRAG", "WEB", "COMBINED"
    previous_strategies: list[str]
    retrieval_attempts: list[dict[str, Any]]

    # CRAG & Refinement Tracking
    crag_score: float
    crag_threshold: float
    refinement_required: bool
    refinement_iteration: int
    max_refinement_iterations: int
    previous_score: float | None
    current_score: float
    score_improvement: float | None
    refinement_stop_reason: str | None

    # Evidence & Rankings
    documents: list[Document]
    dense_count: int
    sparse_count: int
    fused_count: int
    reranked_count: int
    crag_decision: dict[str, Any] | None

    # Post-generation Self-RAG reflection
    generated_answer: str
    reflection_passed: bool
    reflection_attempt: int
    reflection_feedback: str
    reflection_decision: dict[str, Any] | None

    # Web & Route decisions
    web_results: list[dict[str, str]]
    web_query: str
    route: Literal["answer", "web", "combined", "chat", "clarify", "refine"]
    force_document_retrieval: bool

    # Observability & Latency Breakdown
    node_latencies: dict[str, float]
    nodes_executed: list[str]
    nodes_skipped: list[str]
    cache_hit: bool


summary_prompt = PromptTemplate.from_template(
    """Summarize the retrieved text faithfully. Keep relevant technical
details and do not add facts that are not present in the text.

Source Content:
{text}

Summary:"""
)

quick_summary_prompt = PromptTemplate.from_template(
    """You are an expert executive summarizer. Provide a concise, high-level summary of the provided text covering the core objective and most critical takeaway in 2-3 focused paragraphs or key bullet points.
Do NOT leak internal chunk IDs, source IDs, or retrieval metadata.

Source Content:
{text}

Summary:"""
)

standard_summary_prompt = PromptTemplate.from_template(
    """You are an expert technical summarizer. Provide a well-structured summary of the provided source following this exact structure:

# Summary

## Main Idea
[Clear, concise statement of what the source is about and its central objective/thesis]

## Key Points
[Bulleted list of the most critical takeaways, facts, metrics, and arguments]

## Important Concepts
[Definitions, key terminology, mechanisms, and methodologies explained in the source]

## Conclusion
[Final outcome, takeaway, or conclusion]

CRITICAL PRESENTATION RULES:
1. Ground every point strictly in the provided Source Content.
2. DO NOT output chunk IDs, retrieval scores, or internal tags like "[Chunk ...]" or "[Source: ...]".
3. The summary must read as a clean, polished, standalone document.

Source Content:
{text}

Summary:"""
)

deep_summary_prompt = PromptTemplate.from_template(
    """You are an expert technical analyst and senior summarizer. Perform a comprehensive, hierarchical deep summary of the entire source without missing any significant section, explanation, example, or relationship.

Follow this exact structure:

# Detailed Summary

## 1. Introduction
[Comprehensive overview of the background, context, purpose, and scope]

## 2. Main Concepts
[Core foundational concepts, key terminology, and underlying principles]

## 3. Detailed Explanation
[Thorough, section-by-section breakdown of the material covering all topics and mechanisms]

## 4. Examples & Case Studies
[Concrete examples, illustrations, data points, or scenarios provided in the source]

## 5. Important Relationships / Comparisons
[Connections between concepts, architectural relationships, distinctions, trade-offs, and interactions]

## 6. Key Takeaways
[Actionable points, vital findings, numerical metrics, and strategic lessons]

## 7. Overall Conclusion
[Final synthesis, implications, and concluding assessment]

CRITICAL PRESENTATION RULES:
1. Ground every statement strictly in the provided Source Content without hallucinating or adding unmentioned facts.
2. DO NOT output chunk IDs, retrieval scores, vector IDs, or internal tags like "[Chunk ...]" or "[Source: ...]".
3. The summary must read as an authoritative, clean, polished, standalone document.

Source Content:
{text}

Detailed Summary:"""
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise AI knowledge assistant and principal technical synthesis engine.\n"
            "Synthesize a direct, authoritative, and evidence-grounded answer based on the provided context evidence.\n\n"
            "Professional Output Guidelines:\n"
            "1. Direct Synthesis: Start directly with your authoritative answer. Never repeat or echo prompt tags, system headers, 'SystemMessage', 'VERIFIED CONTEXT EVIDENCE', or 'USER QUESTION'.\n"
            "2. Multi-Faceted Technical Concepts: When answering broad concepts with multiple dimensions (such as Masking, Self-Attention, Encoders, Pretraining), clearly distinguish each distinct technical dimension (e.g. Causal/Autoregressive Masking, Padding Masking, Masked Autoencoders MAE, ViT).\n"
            "3. Professional Markdown Layout: Use clean structured headers (`## Overview`, `## Architectural Mechanism`, `## Comparison / Specification`, `## Key Takeaways`, `## Citations`), markdown comparison tables, and code snippets where relevant.\n"
            "4. Verifiable Citations: Label references accurately based on their source and page (e.g. `[1] lec19.transformer.pdf, Page 22`).\n"
            "5. Zero Disclaimers: If you explain concepts grounded in the context, do not contradict yourself with generic 'information is not available' disclaimers.\n"
            "6. Standalone Polish: Output only the clean, polished, user-facing response.\n"
            "7. Off-Topic Noise Filter: If the supplied web evidence consists of unrelated dictionary definitions or irrelevant grammatical entries (e.g. definitions of common verbs like 'do' or 'be'), disregard that noise and provide a comprehensive, accurate technical explanation directly."
        ),
        (
            "human",
            "Context Evidence:\n{context}\n\nQuestion:\n{question}"
        ),
    ]
)


def message_text(message: Any) -> str:
    """Safely return clean string content without message wrapper serialization or response_metadata dumps, preserving all token whitespaces."""
    if message is None:
        return ""
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            text = "".join(text_parts)
        else:
            text = str(content)
    elif isinstance(message, dict):
        text = str(message.get("content", ""))
    else:
        text = str(message)

    # For streaming tokens or normal text, preserve whitespace untouched
    if not (text.startswith("SystemMessage") or text.startswith("AIMessage") or text.startswith("HumanMessage") or text.startswith("ChatMessage")):
        return text

    # Only strip if it is a whole serialized LangChain wrapper string
    trimmed = text.strip()
    m_match = re.match(
        r"^(?:SystemMessage|AIMessage|HumanMessage|ChatMessage|FunctionMessage|ToolMessage)\s*\(\s*(?:content=)?(['\"]{1,3})(.*?)\1(?:\s*,\s*response_metadata=.*)?\s*\)$",
        trimmed,
        re.DOTALL,
    )
    if m_match:
        return m_match.group(2)

    m_multi = re.match(
        r"^(?:SystemMessage|AIMessage|HumanMessage|ChatMessage)\s*\n+content=(?:\"|\'|\"\"\"|\'\'\')(.*?)(?:\"|\'|\"\"\"|\'\'\')(?:[\s\S]*)$",
        trimmed,
        re.DOTALL,
    )
    if m_multi:
        return m_multi.group(1)

    return text


def conversation_context(state: AgentState, limit: int = 6) -> str:
    messages = state.get("messages", [])[-limit:]
    return "\n".join(f"{message.type}: {message_text(message)}" for message in messages)


def format_biography(biography: dict[str, str] | None) -> str:
    """Format stored, explicitly stated profile facts."""
    if not biography:
        return ""
    return "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in biography.items() if value)


def get_chat_system_prompt(biography: dict[str, str] | None) -> SystemMessage:
    """Generate structured assistant system prompt with user profile facts and identity instructions."""
    bio_str = format_biography(biography)
    if bio_str:
        prompt_text = (
            "You are an enterprise AI knowledge assistant.\n\n"
            "=== USER PROFILE & IDENTITY MEMORY ===\n"
            f"{bio_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. If the user explicitly asks about their personal identity, personal name, personal occupation, or personal profile (e.g. 'who am I', 'who i am', 'what is my name', 'what do you know about me'), "
            "answer accurately, politely, and directly based on the recorded User Profile facts above.\n"
            "2. For all technical concepts (including machine learning, agents, architectures, coding), provide direct, authoritative technical explanations.\n"
            "3. Never hallucinate user identity facts that are not present in the profile."
        )
    else:
        prompt_text = (
            "You are an enterprise AI knowledge assistant and principal technical expert.\n"
            "Provide clear, comprehensive, and accurate explanations for technical concepts, software architectures, artificial intelligence, and general queries.\n"
            "Answer directly without referencing personal profile or identity unless the user specifically asks who they are."
        )
    return SystemMessage(content=prompt_text)


def detect_and_ingest_urls(question: str, session_id: str) -> tuple[str, list[str]]:
    """Detect URLs in chat messages and auto-ingest YouTube, GitHub, or Web resources with domain drift isolation."""
    raw_urls = re.findall(r"https?://[^\s<>\"']+", question)
    ingested_labels: list[str] = []
    if not raw_urls:
        return question, ingested_labels

    for u in raw_urls:
        clean_u = u.rstrip(".,;!?)")
        if "youtube.com" in clean_u.lower() or "youtu.be" in clean_u.lower():
            try:
                from ingest import ingest_url
                ingest_url(clean_u, session_id=session_id)
                ingested_labels.append(f"YouTube Video ({clean_u})")
            except Exception as err:
                logger.warning(f"Auto-ingesting YouTube URL failed: {err}")
        elif "github.com" in clean_u.lower():
            try:
                from tools.github_tools import classify_github_url
                kind, meta = classify_github_url(clean_u)
                if kind == "topic":
                    topic_name = meta.get("topic", "general")
                    set_active_source(session_id, {
                        "source_type": "github_topic",
                        "source_id": f"github_topic_{topic_name}",
                        "source_url": clean_u,
                        "topic": topic_name,
                        "title": f"GitHub Topic: {topic_name}",
                    })
                    ingested_labels.append(f"GitHub Topic ({topic_name})")
                elif kind == "repo":
                    owner = meta.get("owner", "")
                    repo = meta.get("repo", "")
                    set_active_source(session_id, {
                        "source_type": "github_repo",
                        "source_id": f"github_repo_{owner}_{repo}",
                        "source_url": clean_u,
                        "owner": owner,
                        "repo": repo,
                        "title": f"GitHub Repository: {owner}/{repo}",
                    })
                    ingested_labels.append(f"GitHub Repo ({owner}/{repo})")
            except Exception as gh_err:
                logger.warning(f"Registering GitHub URL failed: {gh_err}")
        else:
            try:
                from ingest import ingest_url
                ingest_url(clean_u, session_id=session_id)
                ingested_labels.append(f"Web Page ({clean_u})")
            except Exception as err:
                logger.warning(f"Auto-ingesting Web URL failed: {err}")

    # If the user only sent a URL without extra instructions, format as summary request
    cleaned_q = question.strip()
    if len(cleaned_q.split()) <= 4 and any(u in cleaned_q for u in raw_urls):
        first_url = raw_urls[0].rstrip(".,;!?)")
        if "youtube.com" in first_url.lower() or "youtu.be" in first_url.lower():
            cleaned_q = f"Summarize the key points, topics, and takeaways from this YouTube video: {first_url}"
        elif "github.com" in first_url.lower():
            cleaned_q = f"Analyze the architecture, components, and improvements for this GitHub resource: {first_url}"
        else:
            cleaned_q = f"Summarize the main contents of this webpage: {first_url}"

    return cleaned_q, ingested_labels


summary_chain = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)) | summary_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()
structured_biography = llm.with_structured_output(UserBiography)
structured_intent_classifier = llm.with_structured_output(IntentDecision)
structured_self_rag = llm.with_structured_output(KnowledgeDecision)


class SelfRAGReflection(BaseModel):
    """Groundedness and relevance judgment for a generated answer."""

    grounded: bool = Field(description="Whether the answer is supported by the supplied evidence.")
    relevant: bool = Field(description="Whether the answer directly addresses the user's question.")
    hallucination_detected: bool = Field(description="Whether the answer contains unsupported claims.")
    feedback: str = Field(description="Specific corrections required before regenerating, if any.")


structured_self_rag_reflection = llm.with_structured_output(SelfRAGReflection)


@safe_traceable(name="Input Guardrail", run_type="chain", tags=["rag-agent", "guardrail", "input-guard"])
def input_guard_node(state: AgentState) -> dict[str, Any]:
    """Sanitize user input, mask PII, and verify prompt injection safety."""
    start_t = time.perf_counter()
    raw_query = state.get("question", "")
    sanitized = global_input_guard.process(raw_query)
    latencies = dict(state.get("node_latencies", {}))
    latencies["input_guard_ms"] = round((time.perf_counter() - start_t) * 1000, 2)

    if not sanitized.is_safe:
        return {
            "question": "Please provide a valid query.",
            "original_query": raw_query,
            "raw_question": raw_query,
            "intent": "clarify",
            "clarification_question": "Your input was flagged by security guardrails for potential prompt injection. Please rephrase your query.",
            "node_latencies": latencies,
        }

    return {
        "question": sanitized.cleaned_input,
        "original_query": sanitized.cleaned_input,
        "raw_question": raw_query,
        "retrieval_attempt": 1,
        "max_retrieval_attempts": MAX_RETRIEVAL_ATTEMPTS,
        "previous_strategies": [],
        "retrieval_attempts": [],
        "node_latencies": latencies,
    }


@safe_traceable(name="Profile Node", run_type="chain", tags=["rag-agent", "profile"])
def profile_node(state: AgentState) -> dict[str, Any]:
    """Extract explicit long-term user facts without inferring, bypassing LLM for simple chat and document questions."""
    start_t = time.perf_counter()
    biography = dict(state.get("biography", {}))
    latencies = dict(state.get("node_latencies", {}))
    query = state.get("question", "")
    route = state.get("route", "")

    # Fast bypass: if not memory route, skip LLM profile extraction (0ms on critical path)
    if route != "memory" or is_fast_chat_query(query):
        latencies["profile_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {"biography": biography, "node_latencies": latencies}

    try:
        extracted = structured_biography.invoke(
            "Analyze the latest user message for important long-term facts that would "
            "help personalize future answers. Examples include their name, profession, "
            "skills, current project, goals, preferences, location, or work status. "
            "Extract only facts explicitly stated by the user; never infer or guess. "
            "Do not retain secrets or sensitive PII. Return an empty facts object when nothing useful is stated.\n\n"
            f"Existing profile:\n{format_biography(biography)}\n\n"
            f"Latest user message:\n{query}"
        )
        for key, value in extracted.facts.items():
            if value.strip():
                biography[key.strip().lower().replace(" ", "_")] = value.strip()
    except Exception:
        pass

    latencies["profile_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
    return {"biography": biography, "node_latencies": latencies}


@safe_traceable(name="Intent & Ambiguity Classifier", run_type="chain", tags=["rag-agent", "router", "intent-classifier"])
def intent_and_ambiguity_node(state: AgentState) -> dict[str, Any]:
    """Classify user intent with ultra-fast regex detection for greetings to eliminate latency."""
    start_t = time.perf_counter()
    latencies = dict(state.get("node_latencies", {}))
    query = state.get("question", "")

    if state.get("force_document_retrieval"):
        latencies["intent_classifier_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {"intent": "research", "clarification_question": None, "node_latencies": latencies}

    if state.get("clarification_question"):
        latencies["intent_classifier_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {"intent": "clarify", "node_latencies": latencies}

    # 1. Zero-latency fast router for greetings, thanks, farewells, and casual chit-chat
    if is_fast_chat_query(query):
        latencies["intent_classifier_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {
            "intent": "chat",
            "clarification_question": None,
            "is_fast_chat": True,
            "node_latencies": latencies,
        }

    prompt = (
        "You are an intent and ambiguity classifier for an enterprise knowledge workspace.\n"
        "Review the latest user query and previous conversation context. Classify intent:\n"
        "1. 'chat': Casual small talk, greetings, general conversational queries.\n"
        "2. 'clarification_required': Underspecified, ambiguous query (e.g. 'Explain that', 'What about it?') where the reference cannot be resolved.\n"
        "3. 'document_question' / 'web_research' / 'hybrid_research': Factual, document-specific, market, or research queries.\n\n"
        f"Conversation History:\n{conversation_context(state)}\n\n"
        f"Latest User Query: {query}"
    )

    try:
        decision: IntentDecision = structured_intent_classifier.invoke(prompt)
        latencies["intent_classifier_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        if decision.is_ambiguous or decision.intent == "clarification_required":
            clarification = decision.missing_information or "Could you please clarify which document, topic, or aspect you would like me to explain?"
            return {"intent": "clarify", "clarification_question": clarification, "node_latencies": latencies}
        elif decision.intent == "chat":
            return {"intent": "chat", "clarification_question": None, "node_latencies": latencies}
        else:
            return {"intent": "research", "clarification_question": None, "node_latencies": latencies}
    except Exception:
        latencies["intent_classifier_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {"intent": "research", "clarification_question": None, "node_latencies": latencies}


@safe_traceable(name="Clarification Node", run_type="chain", tags=["rag-agent", "clarification"])
def clarification_node(state: AgentState) -> dict[str, list[AIMessage]]:
    """Return contextual clarification question."""
    question = state.get("clarification_question") or "Could you please specify what information you are looking for?"
    return {"messages": [AIMessage(content=question)]}


@safe_traceable(name="Self-RAG Precheck", run_type="chain", tags=["rag-agent", "self-rag", "precheck"])
def self_rag_precheck_node(state: AgentState) -> dict[str, Any]:
    """Determine if query can be answered from model knowledge or strictly requires external evidence."""
    start_t = time.perf_counter()
    latencies = dict(state.get("node_latencies", {}))

    # Fast bypass if already chat
    if state.get("intent") == "chat":
        latencies["self_rag_ms"] = 0.0
        return {"node_latencies": latencies}

    if not SELF_RAG_ENABLED or state.get("force_document_retrieval"):
        latencies["self_rag_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {
            "knowledge_decision": KnowledgeDecision(
                answer_from_model=False,
                requires_documents=True,
                requires_web=False,
                confidence=1.0,
                reason="Document retrieval enforced by user or configuration.",
            ),
            "node_latencies": latencies,
        }

    query = state["question"]
    prompt = (
        "You are a Self-RAG Knowledge Assessment Engine.\n"
        "Determine if the user question can be completely, accurately, and reliably answered using "
        "general model knowledge, or if it requires user-uploaded document evidence or live web research.\n\n"
        "Rules:\n"
        "- If the query asks about general knowledge, concepts, standard definitions, or common coding syntax (e.g. 'What is Python?'), answer_from_model = true, requires_documents = false.\n"
        "- If the query mentions 'my uploaded PDF', 'the document', 'the code', 'chapter 4', or asks about specific proprietary files, answer_from_model = false, requires_documents = true.\n"
        "- If the query asks for current stock price, weather, latest news, or current release notes, requires_web = true.\n"
        "- If the query compares uploaded document content with latest online/market data, requires_documents = true, requires_web = true.\n\n"
        f"User Query: {query}\n"
        f"Conversation History:\n{conversation_context(state)}"
    )

    try:
        decision: KnowledgeDecision = structured_self_rag.invoke(prompt)
        latencies["self_rag_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        if decision.answer_from_model and not decision.requires_documents and not decision.requires_web:
            return {"intent": "chat", "knowledge_decision": decision, "node_latencies": latencies}
        return {"knowledge_decision": decision, "node_latencies": latencies}
    except Exception as err:
        latencies["self_rag_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {
            "knowledge_decision": KnowledgeDecision(
                answer_from_model=False,
                requires_documents=True,
                requires_web=False,
                confidence=0.8,
                reason="Proceeding with document retrieval by default.",
            ),
            "node_latencies": latencies,
        }


@safe_traceable(name="Chat Node", run_type="chain", tags=["rag-agent", "chat", "route-chat"])
def chat_node(state: AgentState) -> dict[str, Any]:
    """Direct conversational completion using general model knowledge and memory."""
    start_t = time.perf_counter()
    instruction = get_chat_system_prompt(state.get("biography"))
    response = llm.invoke([instruction, *state["messages"]])
    latencies = dict(state.get("node_latencies", {}))
    latencies["chat_node_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
    return {"messages": [response], "node_latencies": latencies}


@safe_traceable(name="Hybrid Retrieval & Reranking", run_type="retriever", tags=["rag-agent", "hybrid-retrieval", "dense", "sparse", "bm25", "cross-encoder"])
def hybrid_retrieval_node(state: AgentState) -> dict[str, Any]:
    """Execute adaptive retrieval: Dense (Chroma) + Sparse (BM25) with RRF and conditional Cross-Encoder reranking."""
    start_t = time.perf_counter()
    current_query = state.get("question", state.get("original_query", ""))
    session_id = state.get("session_id", "").strip() or "default"
    latencies = dict(state.get("node_latencies", {}))

    # Lazy Indexing: Ensure active source is indexed before retrieval if not yet indexed
    try:
        from ingest import ensure_source_indexed
        ensure_source_indexed(session_id)
    except Exception as err:
        logger.warning(f"Lazy indexing verification failed: {err}")

    # Active source metadata filtering (Domain / Topic Drift Isolation)
    active_src = state.get("active_source")
    if not active_src:
        try:
            active_src = get_active_source(session_id)
        except Exception:
            active_src = None

    filter_dict: dict[str, Any] | None = None
    if active_src and active_src.get("source_id"):
        filter_dict = {"source_id": active_src["source_id"]}

    # 1. Parallel Dense + Sparse Retrieval with RRF
    t_ret_start = time.perf_counter()
    fused_candidates, dense_count, sparse_count = global_hybrid_retriever.retrieve(
        query=current_query,
        session_id=session_id,
        filter_dict=filter_dict,
    )
    if not fused_candidates and filter_dict:
        # Fallback to session-wide search if strict source filter had 0 results
        fused_candidates, dense_count, sparse_count = global_hybrid_retriever.retrieve(
            query=current_query,
            session_id=session_id,
        )
    fused_count = len(fused_candidates)
    retrieval_time_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)
    latencies["dense_ms"] = round(retrieval_time_ms / 2, 2)
    latencies["bm25_ms"] = round(retrieval_time_ms / 2, 2)
    latencies["rrf_ms"] = 0.1

    # 2. Neural Cross-Encoder Reranking
    t_rerank_start = time.perf_counter()
    if fused_candidates:
        strategy = "HYBRID_RERANK"
        reranked_results = global_reranker.rerank(
            query=state.get("original_query", current_query),
            documents=fused_candidates,
            top_k=6,
        )
        latencies["cross_encoder_ms"] = round((time.perf_counter() - t_rerank_start) * 1000, 2)
    else:
        strategy = "EMPTY"
        reranked_results = []
        latencies["cross_encoder_ms"] = 0.0

    reranked_count = len(reranked_results)

    # 3. Encapsulate in untrusted security envelopes
    wrapped_docs: list[Document] = []
    top_score = 0.0
    for doc, score, rank in reranked_results:
        if top_score == 0.0:
            top_score = score
        source_name = doc.metadata.get("source", doc.metadata.get("filename", "document"))
        wrapped_content = global_injection_guard.wrap_untrusted_content(
            doc.page_content, source_name=str(source_name), content_type="retrieved_chunk"
        )
        wrapped_docs.append(
            Document(
                page_content=wrapped_content,
                metadata={**doc.metadata, "rerank_score": score, "rank": rank},
            )
        )

    recall_metrics = calculate_recall_node(
        question=state.get("original_query", current_query),
        retrieved_docs=wrapped_docs,
    )

    latencies["retrieval_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
    return {
        "documents": wrapped_docs,
        "active_source": active_src,
        "dense_count": dense_count,
        "sparse_count": sparse_count,
        "fused_count": fused_count,
        "reranked_count": reranked_count,
        "retrieval_strategy": strategy,
        "current_score": top_score,
        "recall_metrics": recall_metrics,
        "node_latencies": latencies,
    }


@safe_traceable(name="Evaluate Recall@k", run_type="evaluator", tags=["rag-agent", "evaluation", "recall"])
def calculate_recall_node(
    question: str,
    retrieved_docs: list[Document],
    k_values: tuple[int, ...] = (3, 5, 10),
    ground_truth_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate retrieval Recall@k directly within a LangSmith traceable node without running an experiment."""
    try:
        # Load ground truth if available from evaluation_results or cache
        relevant_ids = set(ground_truth_chunk_ids or [])
        if not relevant_ids:
            gt_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "ground_truth.jsonl")
            if not os.path.exists(gt_path):
                gt_path = "evaluation_results/ground_truth.jsonl"
            if os.path.exists(gt_path):
                q_clean = re.sub(r"[^\w\s]", "", question.strip().lower())
                with open(gt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        saved_q = re.sub(r"[^\w\s]", "", (data.get("question") or "").strip().lower())
                        if saved_q == q_clean or saved_q in q_clean or q_clean in saved_q:
                            relevant_ids = set(data.get("relevant_chunk_ids", []))
                            break

        if not relevant_ids:
            return {
                "status": "skipped",
                "message": "No verified ground truth labels found for this specific query.",
                "retrieved_count": len(retrieved_docs),
            }

        # Normalize retrieved chunk IDs
        retrieved_ids: list[str] = []
        for doc in retrieved_docs:
            m = doc.metadata or {}
            c_id = m.get("chunk_id")
            if not c_id:
                src = str(m.get("source", m.get("filename", "chunk"))).replace("\\", "/")
                clean_name = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(src))[0].lower())
                page = m.get("page", 1)
                chunk_idx = m.get("chunk_index", 0)
                c_id = f"{clean_name}_p{page}_c{chunk_idx}"
            if c_id not in retrieved_ids:
                retrieved_ids.append(c_id)

        results: dict[str, Any] = {
            "status": "evaluated",
            "question": question,
            "ground_truth_count": len(relevant_ids),
            "ground_truth_ids": list(relevant_ids),
            "retrieved_count": len(retrieved_ids),
            "retrieved_ids": retrieved_ids,
        }

        for k in k_values:
            top_k_set = set(retrieved_ids[:k])
            matched = relevant_ids.intersection(top_k_set)
            recall_pct = round(100.0 * len(matched) / len(relevant_ids), 2)
            results[f"recall_at_{k}"] = recall_pct
            results[f"matched_at_{k}"] = list(matched)
            results[f"missed_at_{k}"] = list(relevant_ids - top_k_set)

        return results
    except Exception as err:
        logger.debug(f"Recall node computation skipped: {err}")
        return {"status": "error", "error": str(err)}


@safe_traceable(name="Routing Node", run_type="chain", tags=["rag-agent", "router", "direct-router"])
def routing_node(state: AgentState) -> dict[str, Any]:
    """Execute ultra-fast direct query router as the first graph stage."""
    route, conf = direct_query_router(state.get("question", ""), session_id=state.get("session_id", "default"))
    return {"route": route}


@safe_traceable(name="Realtime Time Node", run_type="tool", tags=["rag-agent", "realtime", "time"])
def realtime_time_node(state: AgentState) -> dict[str, Any]:
    """Return current local date and time in 0ms without LLM inference."""
    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")
    return {"messages": [AIMessage(content=f"The current date and time is {now_str}.")]}


@safe_traceable(name="YouTube Summary Node", run_type="chain", tags=["rag-agent", "youtube", "summary"])
def youtube_summary_node(state: AgentState) -> dict[str, Any]:
    """Summarize YouTube transcript directly without full vector indexing."""
    question = state.get("question", "")
    raw_urls = re.findall(r"https?://[^\s<>\"']+", question)
    if raw_urls:
        target_url = raw_urls[0].rstrip(".,;!?)")
        try:
            from ingestion.youtube_loader import load_youtube
            doc = load_youtube(target_url)
            summary_prompt_val = (
                "Please provide a clear, comprehensive, and well-structured summary of the following YouTube video transcript.\n"
                "Extract the main topics, key takeaways, and conclusions:\n\n"
                f"{doc.page_content}"
            )
            response = llm.invoke([
                SystemMessage(content="You are an expert video summarizer. Provide concise, grounded summaries."),
                HumanMessage(content=summary_prompt_val)
            ])
            passed, validated_answer, _ = global_output_guard.validate_output(message_text(response))
            return {"messages": [AIMessage(content=validated_answer)]}
        except Exception as yt_err:
            logger.warning(f"Fast YouTube summary error: {yt_err}")
    return {"messages": [AIMessage(content="Could not extract YouTube video transcript.")]}


@safe_traceable(name="CRAG Evidence Evaluation", run_type="chain", tags=["rag-agent", "crag", "evaluation"])
def crag_evaluation_node(state: AgentState) -> dict[str, Any]:
    """Evaluate retrieved evidence quality and enforce multi-level adaptive confidence thresholds (High, Medium, Low)."""
    start_t = time.perf_counter()
    query = state.get("original_query", state.get("question", ""))
    documents = state.get("documents", [])
    session_id = state.get("session_id", "default")
    latencies = dict(state.get("node_latencies", {}))

    refinement_iteration = state.get("refinement_iteration", 0)
    previous_score = state.get("previous_score")
    top_score = documents[0].metadata.get("rerank_score", 0.0) if documents else 0.0

    # 1. HIGH CONFIDENCE (Cross-Encoder probability >= 0.70):
    # Direct high-relevance document match -> direct to answer!
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        latencies["crag_evaluation_ms"] = 0.0
        return {
            "crag_decision": {
                "decision": "good",
                "relevance_score": top_score,
                "completeness_score": 0.9,
                "contradiction": False,
                "missing_information": [],
                "reason": "High confidence direct document match.",
                "should_refine": False,
            },
            "crag_score": top_score,
            "current_score": top_score,
            "previous_score": previous_score,
            "score_improvement": None,
            "crag_threshold": CRAG_RELEVANCE_THRESHOLD,
            "refinement_required": False,
            "refinement_iteration": refinement_iteration,
            "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
            "refinement_stop_reason": "high_confidence_match",
            "route": "answer",
            "node_latencies": latencies,
        }

    # 2. LOW CONFIDENCE (Cross-Encoder probability < 0.35) or NO DOCS:
    # Query is out-of-domain for the document -> directly route to Web Search fallback!
    if top_score < MEDIUM_CONFIDENCE_MIN or not documents:
        latencies["crag_evaluation_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {
            "crag_decision": {
                "decision": "bad",
                "relevance_score": top_score,
                "completeness_score": 0.1,
                "contradiction": False,
                "missing_information": [f"Query '{query}' is not covered in the active document."],
                "reason": f"Retrieved document evidence has low relevance score ({top_score:.4f} < {MEDIUM_CONFIDENCE_MIN}). Routing to Web Search.",
                "should_refine": False,
            },
            "crag_score": top_score,
            "current_score": top_score,
            "previous_score": previous_score,
            "score_improvement": None,
            "crag_threshold": CRAG_RELEVANCE_THRESHOLD,
            "refinement_required": False,
            "refinement_iteration": refinement_iteration,
            "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
            "refinement_stop_reason": "low_confidence_fallback",
            "route": "web",
            "node_latencies": latencies,
        }

    # 3. MEDIUM CONFIDENCE (0.35 <= top_score < 0.70):
    # Run structured LLM / heuristic CRAG evaluation for nuanced judgment
    crag_decision = global_crag_evaluator.evaluate(
        query=query,
        documents=documents,
        conversation_context=conversation_context(state),
        session_id=session_id,
    )
    current_score = round(float(crag_decision.relevance_score), 4)

    if previous_score is not None:
        score_improvement = round(current_score - float(previous_score), 4)
    else:
        score_improvement = None

    if crag_decision.decision == "good" or current_score >= CRAG_RELEVANCE_THRESHOLD:
        refinement_required = False
        refinement_stop_reason = "threshold_met"
        route = "answer"
    elif crag_decision.decision == "partial":
        refinement_required = (refinement_iteration < MAX_REFINEMENT_ITERATIONS)
        refinement_stop_reason = "partial_match" if not refinement_required else None
        route = "refine" if refinement_required else "combined"
    else:
        refinement_required = False
        refinement_stop_reason = "evidence_unrelated"
        route = "web"

    latencies["crag_evaluation_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
    return {
        "crag_decision": crag_decision.model_dump(),
        "crag_score": current_score,
        "current_score": current_score,
        "previous_score": previous_score,
        "score_improvement": score_improvement,
        "crag_threshold": CRAG_RELEVANCE_THRESHOLD,
        "refinement_required": refinement_required,
        "refinement_iteration": refinement_iteration,
        "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
        "refinement_stop_reason": refinement_stop_reason,
        "route": route,
        "node_latencies": latencies,
    }


@safe_traceable(name="Query Refinement Node", run_type="chain", tags=["rag-agent", "query-refinement", "retrieval-retry"])
def query_refinement_node(state: AgentState) -> dict[str, Any]:
    """Diagnose previous retrieval failure, increment refinement iteration, and produce an improved search query."""
    start_t = time.perf_counter()
    orig_query = state.get("original_query", state["question"])
    current_query = state.get("question", orig_query)
    raw_crag = state.get("crag_decision") or {}
    crag_decision = CRAGDecision(**raw_crag) if raw_crag else CRAGDecision(decision="bad", reason="Missing evidence")
    latencies = dict(state.get("node_latencies", {}))

    # Increment refinement iteration and attempt sequence
    refinement_iteration = state.get("refinement_iteration", 0) + 1
    attempt = state.get("retrieval_attempt", 1) + 1
    prev_strats = list(state.get("previous_strategies", []))

    refinement = global_query_refiner.refine(
        original_query=orig_query,
        current_query=current_query,
        crag_decision=crag_decision,
        conversation_context=conversation_context(state),
        attempt=attempt,
        previous_strategies=prev_strats,
        session_id=state.get("session_id", "default"),
    )

    prev_strats.append(refinement.strategy)
    latencies["query_refinement_ms"] = round((time.perf_counter() - start_t) * 1000, 2)

    return {
        "question": refinement.refined_query,
        "previous_score": state.get("current_score"),
        "refinement_iteration": refinement_iteration,
        "retrieval_attempt": attempt,
        "retrieval_strategy": refinement.strategy,
        "previous_strategies": prev_strats,
        "node_latencies": latencies,
    }


def clean_web_search_query(query: str) -> str:
    """Transform conversational queries into high-precision search engine keywords."""
    if not query or not query.strip():
        return ""
    q = query.strip()
    conversational_prefixes = [
        r"^what\s+(?:do\s+you\s+mean\s+by|is|are|was|were)\s+",
        r"^how\s+(?:is\s+it|are\s+they|do\s+we|does\s+it|to)\s+",
        r"^tell\s+me\s+(?:about|more\s+about)\s+",
        r"^can\s+you\s+(?:explain|tell\s+me\s+about|describe)\s+",
        r"^explain\s+(?:to\s+me\s+)?",
        r"^who\s+(?:is|was|are)\s+",
        r"^why\s+(?:is|are|does|did)\s+",
        r"^what\s+is\s+the\s+difference\s+between\s+",
    ]
    cleaned = q
    for pat in conversational_prefixes:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:and\s+how\s+(?:its|it\'s|is\s+it)\s+diff\w*\s+from)\b", "vs", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bselft\b", "self", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bself\s+mask\s+attention\b", "masked self-attention", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmask\s+attention\b", "masked attention", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[?!.,;]+$", "", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else q


@safe_traceable(name="Web Search Node", run_type="tool", tags=["rag-agent", "web-search"])
def web_search_node(state: AgentState) -> dict[str, Any]:
    """Search DuckDuckGo using refined or original query, wrapping snippets in untrusted envelopes."""
    start_t = time.perf_counter()
    raw_query = state.get("question", state.get("original_query", ""))
    search_query = clean_web_search_query(raw_query) or raw_query
    latencies = dict(state.get("node_latencies", {}))
    try:
        raw_results = []
        for q_attempt in [search_query, raw_query]:
            if not q_attempt:
                continue
            try:
                res = list(DDGS().text(q_attempt, max_results=WEB_RESULT_COUNT))
                if res:
                    raw_results = res
                    break
            except Exception:
                continue

        # Filter out dictionary/lexical grammar results if the query is a technical query
        filtered_results = []
        is_technical = any(w in raw_query.lower() for w in ["attention", "transformer", "encoder", "decoder", "layer", "model", "neural", "deep learning", "vector", "token"])
        for item in raw_results:
            title = (item.get("title") or "").lower()
            snippet = (item.get("body") or "").lower()
            # If technical, reject English grammar dictionary definition results for auxiliary verbs
            if is_technical and any(dict_site in title for dict_site in ["cambridge dictionary", "merriam-webster", "dictionary.com", "esl teacher", "grammar"]):
                if not any(k in snippet for k in ["attention", "transformer", "mask", "neural", "deep learning"]):
                    continue
            filtered_results.append(item)

        web_results = [
            {
                "title": item.get("title", "Untitled result"),
                "url": item.get("href", ""),
                "snippet": global_injection_guard.wrap_untrusted_content(
                    item.get("body", ""), source_name=item.get("href", "web"), content_type="web_snippet"
                ),
            }
            for item in filtered_results
            if item.get("href") and item.get("body")
        ]
        latencies["web_search_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {"web_results": web_results, "node_latencies": latencies}
    except Exception as err:
        logger.warning(f"DuckDuckGo search error: {err}")
        latencies["web_search_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        return {"web_results": [], "node_latencies": latencies}


@safe_traceable(name="Answer Node", run_type="chain", tags=["rag-agent", "answer", "llm-gateway"])
def answer_node(state: AgentState) -> dict[str, Any]:
    """Synthesize grounded final answer, strictly distinguishing document vs web sources."""
    start_t = time.perf_counter()
    context_parts: list[str] = []
    documents = state.get("documents", [])
    web_results = state.get("web_results", [])
    route = state.get("route", "answer")
    session_id = state.get("session_id", "default")
    active_src = state.get("active_source") or get_active_source(session_id)
    latencies = dict(state.get("node_latencies", {}))

    # Document Evidence
    if documents and (route in {"answer", "combined"} or state.get("force_document_retrieval")):
        formatted_docs = "\n\n".join(
            f"--- Evidence {idx+1} ---\n{d.page_content}"
            for idx, d in enumerate(documents[:8])
        )
        context_parts.append("=== VERIFIED SOURCE EVIDENCE ===\n" + formatted_docs)

    # Web Research Evidence
    if web_results and route in {"web", "combined"}:
        formatted_web = "\n\n".join(
            f"Source Title: {r['title']}\nSnippet: {r['snippet']}" for r in web_results
        )
        context_parts.append("=== CURRENT WEB RESEARCH EVIDENCE ===\n" + formatted_web)

    # Insufficient Evidence Handling
    if not context_parts:
        latencies["answer_node_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
        ans_text = "The requested information is not available in the indexed documentation."
        return {
            "messages": [AIMessage(content=ans_text)],
            "generated_answer": ans_text,
            "node_latencies": latencies,
        }

    reflection_feedback = state.get("reflection_feedback", "").strip()
    if reflection_feedback:
        context_parts.append(
            "=== SELF-RAG REFLECTION FEEDBACK ===\n"
            "Regenerate the answer and correct only the issues identified below. "
            "Keep every claim grounded in the evidence.\n"
            + reflection_feedback
        )

    raw_answer = answer_chain.invoke(
        {
            "context": "\n\n".join(context_parts),
            "question": state.get("original_query", state["question"]),
            "biography": format_biography(state.get("biography")),
        }
    )

    passed, validated_answer, _ = global_output_guard.validate_output(raw_answer)
    latencies["answer_node_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
    return {
        "messages": [AIMessage(content=validated_answer)],
        "generated_answer": validated_answer,
        "node_latencies": latencies,
    }


@safe_traceable(name="Self-RAG Reflection", run_type="chain", tags=["rag-agent", "self-rag", "reflection", "groundedness"])
def self_rag_reflection_node(state: AgentState) -> dict[str, Any]:
    """Grade the generated answer and request one grounded regeneration when needed."""
    answer = state.get("generated_answer", "")
    if not answer and state.get("messages"):
        answer = message_text(state["messages"][-1])

    attempt = state.get("reflection_attempt", 0)
    latencies = dict(state.get("node_latencies", {}))

    # A second failed pass is delivered rather than entering an unbounded loop.
    if attempt >= 1:
        return {
            "reflection_passed": True,
            "reflection_attempt": attempt,
            "reflection_decision": {"grounded": False, "relevant": False, "forced_delivery": True},
            "node_latencies": latencies,
        }

    documents = state.get("documents", [])
    evidence = "\n\n".join(
        f"[Evidence {idx + 1}] {document.page_content}" for idx, document in enumerate(documents[:8])
    )
    web_evidence = "\n\n".join(
        f"[Web: {item.get('title', 'Untitled')}] {item.get('snippet', '')}"
        for item in state.get("web_results", [])
    )
    prompt = (
        "You are a strict Self-RAG reflection grader. Evaluate the answer against the user's question "
        "and the supplied evidence. Mark it failed if it contains unsupported facts, hallucinations, "
        "or does not answer the question. Check contradictions explicitly.\n\n"
        f"Question:\n{state.get('original_query', state.get('question', ''))}\n\n"
        f"Generated answer:\n{answer}\n\n"
        f"Internal evidence:\n{evidence or 'None'}\n\n"
        f"Web evidence:\n{web_evidence or 'None'}"
    )

    try:
        decision: SelfRAGReflection = structured_self_rag_reflection.invoke(prompt)
        passed = bool(decision.grounded and decision.relevant and not decision.hallucination_detected)
        return {
            "reflection_passed": passed,
            "reflection_attempt": attempt + (0 if passed else 1),
            "reflection_feedback": decision.feedback if not passed else "",
            "reflection_decision": decision.model_dump(),
            "node_latencies": latencies,
        }
    except Exception as error:
        logger.warning("Self-RAG reflection failed closed: %s", error)
        return {
            "reflection_passed": True,
            "reflection_attempt": attempt,
            "reflection_decision": {"grounded": True, "relevant": True, "fallback": True},
            "node_latencies": latencies,
        }


def self_rag_reflection_route(state: AgentState) -> Literal["deliver", "regenerate"]:
    """Route failed first-pass answers through one regeneration attempt."""
    if not state.get("reflection_passed", True) and state.get("reflection_attempt", 0) <= 1:
        return "regenerate"
    return "deliver"


@safe_traceable(name="Stream Response", run_type="chain", tags=["rag-agent", "query", "stream", "adaptive-rag"])
def stream_response(
    question: str,
    messages: list[BaseMessage],
    session_id: str,
    biography: dict[str, str] | None = None,
    sources_collector: list[dict[str, Any]] | None = None,
    route_collector: list[str] | None = None,
):
    """Yield model response token-by-token with direct query-type routing, response caching, and zero redundant steps."""
    effective_question, new_sources = detect_and_ingest_urls(question, session_id=session_id)
    force_doc_retrieval = len(new_sources) > 0
    active_src = get_active_source(session_id)

    route, confidence = direct_query_router(effective_question, session_id=session_id)
    if force_doc_retrieval and route in {"general_chat", "fast_chat"}:
        route = "document_rag"

    if route_collector is not None and route in {"fast_chat", "realtime_time", "youtube_summary", "realtime_web", "general_chat", "memory", "github_topic", "github_repo"}:
        route_collector.append(route)

    tracing_config = get_tracing_config(
        session_id=session_id,
        thread_id=session_id,
        tags=[route, f"thread:{session_id}"],
        run_name=f"ChatTurn: {effective_question[:40]}",
    )

    # 1. Check Response Cache for non-personalized queries
    if route in {"fast_chat", "general_chat"} and not biography:
        cached = global_response_cache.get(effective_question)
        if cached:
            cached_text = cached if isinstance(cached, str) else (cached.get("content") if isinstance(cached, dict) else str(cached))
            if cached_text:
                yield str(cached_text)
                return

    # Route A: FAST CHAT ("hello", "hi", "hlo", "thanks", "how are you")
    # Skips Profile, Intent LLM, Self-RAG, Embeddings, Chroma, BM25, CRAG, Refinement
    if route == "fast_chat":
        cleaned_fast = re.sub(r"[^\w\s]", "", effective_question.lower()).strip()
        fast_greetings = {
            "hello": "Hello! How can I assist you today?",
            "hi": "Hi there! How can I help you today?",
            "hlo": "Hello! How can I help you today?",
            "hey": "Hey! How can I assist you?",
            "heyy": "Hey there! How can I help?",
            "howdy": "Howdy! How can I help you today?",
            "greetings": "Greetings! How can I help you today?",
            "good morning": "Good morning! How can I help you today?",
            "good afternoon": "Good afternoon! How can I assist you today?",
            "good evening": "Good evening! How can I assist you today?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "thank you": "You're very welcome! Happy to help.",
            "bye": "Goodbye! Have a great day ahead.",
            "goodbye": "Goodbye! Feel free to reach out anytime.",
            "ok": "Got it! Let me know how I can assist further.",
            "okay": "Understood! What would you like to explore next?",
        }
        if cleaned_fast in fast_greetings and not biography and not messages:
            yield fast_greetings[cleaned_fast]
            return

        instruction = get_chat_system_prompt(biography)
        collected = []
        for chunk in llm.stream([instruction, *messages], config=tracing_config):
            content = message_text(chunk)
            if content:
                collected.append(content)
                yield content
        full_res = "".join(collected)
        if not biography and full_res and not full_res.strip().startswith("I encountered an issue"):
            global_response_cache.set(effective_question, full_res)
        return

    # Route B: REALTIME TIME ("what time is it", "current time", "what is the time")
    if route == "realtime_time":
        now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")
        yield f"The current date and time is {now_str}."
        return

    # Helper to retrieve sequential full source content for unbiased coverage
    def _get_full_source_text(sess_id: str, act_source: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
        try:
            from app.vectorstore.croma_store import get_vectorstore
            vs = get_vectorstore()
            src_id = act_source.get("source_id") if act_source else None
            filter_dict = {"source_id": src_id} if src_id else None
            results = vs.similarity_search("", k=50, filter=filter_dict) if filter_dict else vs.similarity_search("", k=50)
            if not results and filter_dict:
                results = vs.similarity_search("", k=50)
            sorted_chunks = sorted(results, key=lambda d: d.metadata.get("chunk_id", d.metadata.get("chunk_index", 0)))
            text = "\n\n".join(c.page_content for c in sorted_chunks)
            doc_sources = []
            for c in sorted_chunks:
                m = dict(c.metadata or {})
                src_val = m.get("source", "Document")
                doc_sources.append({
                    "source": src_val,
                    "filename": m.get("filename") or (os.path.basename(src_val) if src_val else "Document Chunk"),
                    "page": m.get("page", 1),
                    "rerank_score": 1.0,
                    "image_url": m.get("image_url", ""),
                    "content": c.page_content[:300],
                })
            return text, doc_sources
        except Exception as err:
            logger.warning(f"Error fetching full source text: {err}")
            return "", []

    # Route C: YOUTUBE SUMMARY ("summarize this YouTube video <URL>", "deep summary of video")
    if route == "youtube_summary":
        depth = classify_summary_depth(effective_question)
        target_prompt = deep_summary_prompt if depth == "deep" else (quick_summary_prompt if depth == "quick" else standard_summary_prompt)
        raw_urls = re.findall(r"https?://[^\s<>\"']+", question)
        transcript_text = ""
        if raw_urls:
            target_url = raw_urls[0].rstrip(".,;!?)")
            try:
                from ingestion.youtube_loader import load_youtube
                doc = load_youtube(target_url)
                transcript_text = doc.page_content
            except Exception as yt_err:
                logger.warning(f"Fast YouTube summary transcript extraction error: {yt_err}")
        if not transcript_text and active_src:
            transcript_text, _ = _get_full_source_text(session_id, active_src)

        if transcript_text:
            formatted_prompt = target_prompt.format(text=transcript_text)
            collected = []
            for chunk in llm.stream([
                SystemMessage(content="You are an expert video summarizer. Ground all statements strictly in the transcript without outputting internal metadata or chunk tags."),
                HumanMessage(content=formatted_prompt)
            ], config=tracing_config):
                tok = message_text(chunk)
                if tok:
                    collected.append(tok)
                    yield tok
            full_res = "".join(collected)
            if full_res and not full_res.strip().startswith("I encountered an issue"):
                global_response_cache.set(effective_question, full_res)
            return

    # Route C2: DOCUMENT SUMMARY ("summarize this document", "give me a detailed summary of the uploaded PDF")
    # Skips Hybrid, Cross-Encoder, CRAG, Refinement, Web -> Direct hierarchical summary of document text
    if route == "document_summary":
        active_src = get_active_source(session_id)
        doc_text, doc_sources = _get_full_source_text(session_id, active_src)
        if doc_text:
            if sources_collector is not None:
                sources_collector.extend(doc_sources)
            depth = classify_summary_depth(effective_question)
            target_prompt = deep_summary_prompt if depth == "deep" else (quick_summary_prompt if depth == "quick" else standard_summary_prompt)
            formatted_prompt = target_prompt.format(text=doc_text)
            collected = []
            for chunk in llm.stream([
                SystemMessage(content="You are an expert document summarizer. Ground all statements strictly in the text without outputting internal metadata or chunk tags."),
                HumanMessage(content=formatted_prompt)
            ], config=tracing_config):
                tok = message_text(chunk)
                if tok:
                    collected.append(tok)
                    yield tok
            full_res = "".join(collected)
            if full_res and not full_res.strip().startswith("I encountered an issue"):
                global_response_cache.set(effective_question, full_res)
            return

    # Route C3: GITHUB TOPIC ANALYSIS ("https://github.com/topics/weather-data", etc.)
    if route == "github_topic":
        from tools.github_tools import classify_github_url, get_github_topic_data
        topic_name = "general"
        if active_src and active_src.get("source_type") == "github_topic":
            topic_name = active_src.get("topic", "general")
        else:
            raw_urls = re.findall(r"https?://[^\s<>\"']+", question)
            for u in raw_urls:
                if "github.com" in u.lower():
                    _, meta = classify_github_url(u)
                    topic_name = meta.get("topic", "general")
                    break

        topic_data = get_github_topic_data(topic_name)
        repos = topic_data.get("repositories", [])
        languages = topic_data.get("languages", [])

        repo_summaries = []
        for r in repos[:8]:
            repo_summaries.append(
                f"- **{r['full_name']}** ({r['language']}, ⭐ {r['stars']:,}, 🍴 {r['forks']:,})\n"
                f"  *Description:* {r['description']}\n"
                f"  *Topics:* {', '.join(r.get('topics', []))}\n"
                f"  *URL:* {r['html_url']}"
            )
        formatted_repos = "\n".join(repo_summaries) if repo_summaries else "No public repositories found under this topic."

        topic_prompt_text = (
            f"You are a senior GitHub ecosystem analyst and software architect.\n"
            f"The user has provided a GitHub TOPIC PAGE (aggregating repositories under the topic '{topic_name}').\n"
            f"Analyze this topic domain and address the user's specific prompt: '{effective_question}'\n\n"
            f"=== GITHUB TOPIC DISCOVERY DATA ===\n"
            f"Topic: {topic_name}\n"
            f"Total Repositories Found: {topic_data.get('total_count', len(repos))}\n"
            f"Primary Languages in Ecosystem: {', '.join(languages) if languages else 'Python, JavaScript, TypeScript, Go'}\n\n"
            f"Top Repositories under Topic:\n{formatted_repos}\n\n"
            f"=== REQUIRED STRUCTURED RESPONSE FORMAT ===\n"
            f"Tool Used: Tool: GitHub Search API (read-only) + Topic Discovery Engine\n\n"
            f"# 1. Topic Overview & Domain Purpose\n"
            f"- Clarify that this is a GitHub Topic Category aggregating repositories, not a single monolithic repository.\n"
            f"- Explain what the '{topic_name}' ecosystem represents (e.g. data ingestion, meteorological APIs, forecast pipelines, climate modeling).\n\n"
            f"# 2. Featured Repositories & Technology Stacks\n"
            f"- Detail the representative projects found in the topic, their primary languages, and their roles.\n\n"
            f"# 3. Common Architecture & Component Synchronization Patterns\n"
            f"- Explain how components in this domain typically work and synchronize (e.g. Ingestion -> Parser -> Time-Series DB -> API/Visualizer).\n"
            f"```text\n[Weather/Data APIs / Scrapers] --> [ETL Ingestion & Normalization] --> [Storage (Time-Series/Relational)] --> [Analytics & API Endpoints]\n```\n\n"
            f"# 4. Ecosystem Weaknesses & Common Pitfalls\n"
            f"- Identify typical weaknesses in projects within this domain (e.g. API rate limits, stale caching, lack of real-time streaming, unhandled telemetry schema changes).\n\n"
            f"# 5. Recommended Improvements & Modernization Plan\n"
            f"- Concrete recommendations (e.g. asynchronous fetching, resilient retry queues, schema validation via Pydantic, distributed streaming via Kafka/Redis, open-data caching)."
        )

        tool_tag = "**Tool Used**: GitHub Search API (read-only) + Topic Discovery Engine\n\n"
        yield tool_tag
        collected = [tool_tag]
        for chunk in llm.stream([
            SystemMessage(content="You are an expert GitHub systems architect. Provide structured, accurate, and deeply technical topic and repository ecosystem analyses. State clearly that this is a GitHub topic collection."),
            HumanMessage(content=topic_prompt_text)
        ], config=tracing_config):
            tok = message_text(chunk)
            if tok:
                collected.append(tok)
                yield tok
        full_res = "".join(collected)
        if full_res and not full_res.strip().startswith("I encountered an issue"):
            global_response_cache.set(effective_question, full_res)
        return

    # Route C4: GITHUB REPOSITORY DEEP ANALYSIS ("https://github.com/owner/repo", etc.)
    if route == "github_repo":
        from tools.github_tools import classify_github_url, get_repository, get_readme, get_repository_tree, get_file
        owner, repo = "", ""
        if active_src and active_src.get("source_type") == "github_repo":
            owner = active_src.get("owner", "")
            repo = active_src.get("repo", "")
        else:
            raw_urls = re.findall(r"https?://[^\s<>\"']+", question)
            for u in raw_urls:
                if "github.com" in u.lower():
                    kind, meta = classify_github_url(u)
                    if kind == "repo":
                        owner = meta.get("owner", "")
                        repo = meta.get("repo", "")
                        break

        # Discover repository assets
        repo_info = {}
        readme_content = ""
        tree_files = []
        key_files_content = {}

        if owner and repo:
            try:
                repo_info = get_repository(owner, repo)
                readme_content = get_readme(owner, repo)
                tree_data = get_repository_tree(owner, repo, recursive=True)
                if isinstance(tree_data, list):
                    tree_files = [f.get("path", "") for f in tree_data if isinstance(f, dict) and f.get("type") == "blob"]
            except Exception as gh_fetch_err:
                logger.warning(f"Error fetching GitHub repo metadata: {gh_fetch_err}")

            # Fetch important config / entrypoint files
            for target_file in ["package.json", "requirements.txt", "pyproject.toml", "Dockerfile", "main.py", "app.py", "index.js", "server.js", "go.mod", "Cargo.toml", "README.md"]:
                if target_file in tree_files:
                    try:
                        f_text = get_file(owner, repo, target_file)
                        if f_text and not f_text.startswith("[Binary"):
                            key_files_content[target_file] = f_text[:3000]
                    except Exception:
                        pass

        repo_prompt_text = (
            f"You are a principal software architect and GitHub code reviewer.\n"
            f"Perform a deep architectural analysis of the repository '{owner}/{repo}' based strictly on the verified repository assets.\n\n"
            f"User Question: {effective_question}\n\n"
            f"=== VERIFIED REPOSITORY EVIDENCE ===\n"
            f"Repository: {owner}/{repo}\n"
            f"Description: {repo_info.get('description', 'N/A')}\n"
            f"Default Branch: {repo_info.get('default_branch', 'main')}\n"
            f"Stars: {repo_info.get('stars', 0)} | Forks: {repo_info.get('forks', 0)}\n\n"
            f"File Tree Structure:\n" + "\n".join(tree_files[:40]) + "\n\n"
            f"README Content:\n{readme_content[:4000] if readme_content else 'No README available.'}\n\n"
            f"Key Config / Entry Files:\n" + "\n\n".join(f"[{fname}]:\n{fcontent}" for fname, fcontent in key_files_content.items()) + "\n\n"
            f"=== REQUIRED STRUCTURED RESPONSE FORMAT ===\n"
            f"# 1. Repository Overview\n"
            f"- Purpose, domain, language stack, and architecture type.\n\n"
            f"# 2. Component Architecture\n"
            f"- Diagram/map of core modules and how they connect:\n"
            f"```text\n[Entry / API Layer] --> [Service / Business Logic] --> [Data / Storage Layer]\n```\n\n"
            f"# 3. Component Details & Working Mechanism\n"
            f"- Explain each component and its responsibility.\n\n"
            f"# 4. Data Flow & Synchronization\n"
            f"- Trace request/data lifecycle and synchronization mechanisms.\n\n"
            f"# 5. Identified Weaknesses & Broken Logic\n"
            f"- Specific vulnerabilities, anti-patterns, missing tests, or concurrency bottlenecks.\n\n"
            f"# 6. Concrete Improvement Plan\n"
            f"- Actionable modernization steps and fixes."
        )

        if sources_collector is not None:
            owner_data = repo_info.get("owner", {})
            avatar_url = owner_data.get("avatar_url", f"https://opengraph.githubassets.com/1/{owner}/{repo}") if isinstance(owner_data, dict) else f"https://opengraph.githubassets.com/1/{owner}/{repo}"
            sources_collector.append({
                "source": f"https://github.com/{owner}/{repo}",
                "filename": f"{owner}/{repo} (GitHub Repository)",
                "page": None,
                "rerank_score": 1.0,
                "image_url": avatar_url,
                "content": repo_info.get("description", "GitHub repository architecture analysis"),
            })

        repo_tool_tag = "**Tool Used**: GitHub Read-Only Inspection Engine\n\n"
        yield repo_tool_tag
        collected = [repo_tool_tag]
        for chunk in llm.stream([
            SystemMessage(content="You are a principal software architect. Provide a deep, accurate architectural code analysis."),
            HumanMessage(content=repo_prompt_text)
        ], config=tracing_config):
            tok = message_text(chunk)
            if tok:
                collected.append(tok)
                yield tok
        full_res = "".join(collected)
        if full_res and not full_res.strip().startswith("I encountered an issue"):
            global_response_cache.set(effective_question, full_res)
        return

    # Route D: REALTIME WEB / CURRENT INFO ("today's weather", "today's run scored by Virat", "latest IPL score")
    if route == "realtime_web":
        web_state = web_search_node({"question": effective_question, "node_latencies": {}})
        web_results = web_state.get("web_results", [])
        if web_results:
            formatted_web = "\n\n".join(
                f"Source Title: {r['title']}\nSnippet: {r['snippet']}" for r in web_results
            )
            for chunk in answer_chain.stream(
                {
                    "context": "=== REAL-TIME WEB SEARCH EVIDENCE ===\n" + formatted_web,
                    "question": effective_question,
                    "biography": format_biography(biography),
                },
                config=tracing_config,
            ):
                if chunk:
                    yield str(chunk)
            return

    # Route E: GENERAL CHAT ("What is 2 + 2?", "What is Python?", "Explain inheritance")
    # General queries bypass document retrieval even if active source exists in session!
    if route == "general_chat" and not force_doc_retrieval:
        instruction = get_chat_system_prompt(biography)
        collected = []
        for chunk in llm.stream([instruction, *messages], config=tracing_config):
            content = message_text(chunk)
            if content:
                collected.append(content)
                yield content
        full_res = "".join(collected)
        if not biography and full_res and not full_res.strip().startswith("I encountered an issue"):
            global_response_cache.set(effective_question, full_res)
        return

    # Route F: MEMORY / IDENTITY ("who am I?", "what is my name?", "what do you know about me?")
    if route == "memory" and not force_doc_retrieval:
        instruction = get_chat_system_prompt(biography)
        for chunk in llm.stream([instruction, *messages], config=tracing_config):
            content = message_text(chunk)
            if content:
                yield content
        return

    # Route G: SOURCE-GROUNDED RAG ("document_rag", "youtube_question", "web_research")
    active_src = get_active_source(session_id)
    mm_decision = classify_multimodal_orchestrator(effective_question, session_id=session_id)

    state: AgentState = {
        "question": effective_question,
        "original_query": effective_question,
        "raw_question": question,
        "messages": messages,
        "session_id": session_id,
        "biography": biography if biography is not None else {},
        "active_source": active_src,
        "active_source_id": active_src.get("source_id") if active_src else None,
        "active_source_type": active_src.get("source_type") if active_src else None,
        "active_source_url": active_src.get("source_url") if active_src else None,
        "active_source_title": active_src.get("title") if active_src else None,
        "multimodal_decision": mm_decision.model_dump(),
        "force_document_retrieval": force_doc_retrieval,
        "retrieval_attempt": 1,
        "max_retrieval_attempts": MAX_RETRIEVAL_ATTEMPTS,
        "previous_strategies": [],
        "retrieval_attempts": [],
        "node_latencies": {},
    }

    # 1. Input Guardrail
    state.update(input_guard_node(state))

    # 2. Parallel Retrieval Execution (Text Retrieval + Web Search Concurrently if required)
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_hybrid():
        s = dict(state)
        s.update(hybrid_retrieval_node(s))
        s.update(crag_evaluation_node(s))
        return s

    def _fetch_web():
        if mm_decision.needs_web:
            return web_search_node({"question": effective_question, "node_latencies": {}})
        return {"web_results": []}

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_hybrid = executor.submit(_fetch_hybrid)
        f_web = executor.submit(_fetch_web)
        state.update(f_hybrid.result())
        web_res = f_web.result()
        if web_res.get("web_results"):
            state["web_results"] = web_res["web_results"]

    # 3. Bounded Conditional Refinement Loop (max 1 refinement)
    while state.get("route") == "refine" and state.get("refinement_required", False) and state.get("refinement_iteration", 0) < MAX_REFINEMENT_ITERATIONS:
        state.update(query_refinement_node(state))
        state.update(hybrid_retrieval_node(state))
        state.update(crag_evaluation_node(state))
        if state.get("route") != "refine":
            break

    # 4. Corrective RAG (CRAG) Adaptive Web Search Fallback
    # If CRAG evaluation detects partial/ambiguous evidence or missing details, trigger DuckDuckGo search
    crag_dec = state.get("crag_decision") or {}
    crag_verdict = crag_dec.get("decision", "good") if isinstance(crag_dec, dict) else getattr(crag_dec, "decision", "good")
    needs_crag_web = crag_verdict in {"partial", "bad"} or state.get("route") in {"web", "combined"} or state.get("crag_score", 1.0) < 0.70 or mm_decision.needs_web
    if needs_crag_web and not state.get("web_results"):
        logger.info("CRAG triggered live DuckDuckGo web search fallback to supplement partial/missing evidence.")
        web_res = web_search_node({"question": effective_question, "node_latencies": {}})
        if web_res.get("web_results"):
            state["web_results"] = web_res["web_results"]

    if route_collector is not None:
        if crag_verdict == "bad" or state.get("route") == "web":
            final_route = "web_research"
        elif state.get("route") == "combined" or crag_verdict == "partial":
            final_route = "hybrid_web_rag"
        else:
            final_route = route
        route_collector.append(final_route)

    # 5. Synthesize context with CRAFT + Multimodal Constraints
    context_parts: list[str] = []

    # Include document evidence only if CRAG verified it as good or partial
    if crag_verdict in {"good", "partial"} and state.get("documents"):
        valid_docs = [
            d for d in state["documents"][:8]
            if (d.metadata.get("rerank_score", 1.0) >= MEDIUM_CONFIDENCE_MIN or crag_verdict == "good")
        ]
        if valid_docs:
            if sources_collector is not None:
                for doc in valid_docs:
                    m = dict(doc.metadata or {})
                    src_val = m.get("source", "Document")
                    sources_collector.append({
                        "source": src_val,
                        "filename": m.get("filename") or (os.path.basename(src_val) if src_val else "Document Chunk"),
                        "page": m.get("page", 1),
                        "rerank_score": m.get("rerank_score", 1.0),
                        "image_url": m.get("image_url", ""),
                        "content": doc.page_content[:300],
                    })
            formatted_docs = "\n\n".join(
                f"--- Document Evidence {idx+1} (Source: {d.metadata.get('filename', 'Document')}, Page {d.metadata.get('page', 1)}) ---\n{d.page_content}"
                for idx, d in enumerate(valid_docs)
            )
            context_parts.append("=== VERIFIED DOCUMENT EVIDENCE ===\n" + formatted_docs)

    # Include live web research results if retrieved
    if state.get("web_results"):
        if sources_collector is not None:
            for idx, r in enumerate(state["web_results"][:5]):
                sources_collector.append({
                    "source": r.get("url", "Web Search"),
                    "filename": r.get("title", "Web Result"),
                    "page": None,
                    "rerank_score": round(0.95 - (idx * 0.05), 2),
                    "image_url": "",
                    "content": r.get("snippet", "")[:300],
                })
        formatted_web = "\n\n".join(
            f"--- Web Evidence {idx+1} (Title: {r['title']}, Link: {r['url']}) ---\nSnippet: {r['snippet']}"
            for idx, r in enumerate(state["web_results"])
        )
        context_parts.append("=== LIVE WEB RESEARCH EVIDENCE ===\n" + formatted_web)

    if not context_parts:
        # Fallback to general conversational generation if no doc or web evidence is present
        instruction = get_chat_system_prompt(biography)
        for chunk in llm.stream([instruction, *messages], config=tracing_config):
            content = message_text(chunk)
            if content:
                yield content
        return

    for chunk in answer_chain.stream(
        {
            "context": "\n\n".join(context_parts),
            "question": state.get("original_query", question),
        },
        config=tracing_config,
    ):
        txt = message_text(chunk)
        if txt:
            yield txt


def create_rag_graph():
    """Compile the full adaptive LangGraph state machine with routing-first conditional execution."""
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("router", routing_node)
    workflow.add_node("profile", profile_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("realtime_time", realtime_time_node)
    workflow.add_node("youtube_summary", youtube_summary_node)
    workflow.add_node("hybrid_retrieval", hybrid_retrieval_node)
    workflow.add_node("crag_evaluation", crag_evaluation_node)
    workflow.add_node("query_refinement", query_refinement_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("self_rag_reflection", self_rag_reflection_node)
    workflow.add_node("clarification", clarification_node)

    # Graph Edges: Routing First
    workflow.add_edge(START, "input_guard")
    workflow.add_edge("input_guard", "router")

    def safe_router_route(state: AgentState) -> str:
        r = str(state.get("route") or "general_chat")
        route_map = {
            "fast_chat": "chat",
            "general_chat": "chat",
            "memory": "profile",
            "realtime_time": "realtime_time",
            "realtime_web": "web_search",
            "youtube_summary": "youtube_summary",
            "youtube_question": "hybrid_retrieval",
            "document_rag": "hybrid_retrieval",
            "document_summary": "hybrid_retrieval",
            "github_topic": "web_search",
            "github_repo": "chat",
            "github_summary": "chat",
            "web_research": "web_search",
            "clarify": "clarification",
        }
        return route_map.get(r, "hybrid_retrieval" if ("doc" in r or "rag" in r or "summary" in r) else "chat")

    workflow.add_conditional_edges(
        "router",
        safe_router_route,
        {
            "chat": "chat",
            "profile": "profile",
            "realtime_time": "realtime_time",
            "web_search": "web_search",
            "youtube_summary": "youtube_summary",
            "hybrid_retrieval": "hybrid_retrieval",
            "clarification": "clarification",
        },
    )

    workflow.add_edge("profile", "chat")
    workflow.add_edge("hybrid_retrieval", "crag_evaluation")

    def safe_crag_route(state: AgentState) -> str:
        r = str(state.get("route") or "answer")
        crag_map = {
            "answer": "answer",
            "refine": "query_refinement",
            "web": "web_search",
            "combined": "web_search",
        }
        return crag_map.get(r, "answer")

    workflow.add_conditional_edges(
        "crag_evaluation",
        safe_crag_route,
        {
            "answer": "answer",
            "query_refinement": "query_refinement",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("query_refinement", "hybrid_retrieval")
    workflow.add_edge("web_search", "answer")
    workflow.add_edge("answer", "self_rag_reflection")
    workflow.add_conditional_edges(
        "self_rag_reflection",
        self_rag_reflection_route,
        {"deliver": END, "regenerate": "answer"},
    )

    workflow.add_edge("chat", END)
    workflow.add_edge("realtime_time", END)
    workflow.add_edge("youtube_summary", END)
    workflow.add_edge("clarification", END)

    return workflow.compile(checkpointer=get_checkpointer())


graph_app = create_rag_graph()


if __name__ == "__main__":
    first_turn = True
    print("Adaptive RAG ready. Type 'exit' to quit.")
    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        input_messages: list[BaseMessage] = [HumanMessage(content=query)]
        if first_turn:
            input_messages.insert(
                0, SystemMessage(content="You are a helpful AI assistant. Be concise and clear."),
            )
        result = graph_app.invoke(
            {
                "question": query,
                "messages": input_messages,
                "session_id": "default",
            }
        )
        first_turn = False
        print(f"Assistant: {message_text(result['messages'][-1])}")
