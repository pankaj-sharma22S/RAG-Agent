"""Polished Streamlit interface for the session-isolated knowledge workspace.

Run from the repository root with ``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import re
import sys
import uuid
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

try:
    from app.memory.schemas import ConversationTurn
    from app.memory.sqlite_store import SQLiteMemoryStore
except ImportError:
    from memory.schemas import ConversationTurn
    from memory.sqlite_store import SQLiteMemoryStore

memory_store = SQLiteMemoryStore()

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from tracing import get_tracing_config, setup_tracing
    setup_tracing()
except ImportError:
    pass

try:
    from app.db.service import (
        create_or_get_conversation,
        delete_conversation,
        get_messages,
        list_conversations,
        save_conversation_metadata,
        save_message,
    )
except ImportError:
    try:
        from db.service import (
            create_or_get_conversation,
            delete_conversation,
            get_messages,
            list_conversations,
            save_conversation_metadata,
            save_message,
        )
    except ImportError:
        create_or_get_conversation = lambda session_id, title=None: {"session_id": session_id, "title": title or "Chat", "metadata": {}}
        list_conversations = lambda: []
        save_conversation_metadata = lambda session_id, title=None, biography=None, files=None: None
        save_message = lambda session_id, role, content, node_name=None, metadata=None: None
        get_messages = lambda session_id: []
        delete_conversation = lambda session_id: None

try:
    from ingest import ingest_file, ingest_url
except ImportError:
    from ingest import ingest_document, ingest_file
    from ingestion.web_loader import load_web_page
    from ingestion.youtube_loader import load_youtube

    def ingest_url(url: str, session_id: str):
        document = load_youtube(url) if ("youtube.com" in url.lower() or "youtu.be" in url.lower()) else load_web_page(url)
        return ingest_document(document, session_id)

try:
    from rag_workflow import stream_response
except ImportError:
    from rag_workflow import graph_app

    def stream_response(question: str, messages, session_id: str, biography=None):
        try:
            from tracing import get_tracing_config
            tracing_config = get_tracing_config(
                thread_id=session_id,
                session_id=session_id,
                tags=["streamlit", "query"],
                run_name="Streamlit Graph Query",
            )
        except Exception:
            tracing_config = {"configurable": {"thread_id": session_id}}

        result = graph_app.invoke(
            {"question": question, "messages": messages, "session_id": session_id, "biography": biography or {}},
            config=tracing_config,
        )
        yield str(result["messages"][-1].content)


UPLOAD_ROOT = APP_DIR / "data" / "uploads"
DOC_TYPES = ["pdf", "docx", "pptx", "xlsx", "xls", "txt", "csv", "md", "json"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
MEDIA_TYPES = ["mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a"]
SUPPORTED_TYPES = DOC_TYPES + IMAGE_TYPES + MEDIA_TYPES


def new_thread() -> str:
    return f"thread-{uuid.uuid4().hex[:12]}"


def initialise_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = os.getenv("DEFAULT_USER_ID", "default_user")
    if "threads" not in st.session_state:
        st.session_state.threads = {}
        # Load persistent conversations from database
        db_convs = list_conversations()
        if db_convs:
            for conv in db_convs:
                sid = conv["session_id"]
                msgs = get_messages(sid)
                meta = conv.get("metadata", {})
                st.session_state.threads[sid] = {
                    "title": conv.get("title") or "Chat",
                    "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                    "files": meta.get("files", []),
                    "biography": meta.get("biography", {}),
                    "created_at": conv.get("created_at"),
                }
            st.session_state.active_thread = db_convs[0]["session_id"]
        else:
            first_thread = new_thread()
            create_or_get_conversation(first_thread, title="New Chat")
            st.session_state.threads = {
                first_thread: {
                    "title": "New Chat",
                    "messages": [],
                    "files": [],
                    "biography": {},
                    "created_at": datetime.now().isoformat(),
                }
            }
            st.session_state.active_thread = first_thread
        st.session_state.upload_version = 0
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename).name).strip(". ")
    if not cleaned:
        raise ValueError("The uploaded file has an invalid name.")
    return cleaned


def save_upload(uploaded_file, thread_id: str) -> Path:
    thread_dir = UPLOAD_ROOT / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    destination = thread_dir / f"{uuid.uuid4().hex[:8]}_{safe_filename(uploaded_file.name)}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def add_thread() -> None:
    thread_id = new_thread()
    create_or_get_conversation(thread_id, title="New Chat")
    st.session_state.threads[thread_id] = {
        "title": "New Chat",
        "messages": [],
        "files": [],
        "biography": {},
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.active_thread = thread_id
    st.session_state.upload_version += 1


def chat_messages(messages: list[dict[str, str]]):
    """Convert the visible conversation into LangChain messages for streaming."""
    return [
        HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(content=item["content"])
        for item in messages
    ]


st.set_page_config(page_title="Knowledge Workspace", page_icon="⚡", layout="wide")
initialise_state()

st.markdown(
    """
    <style>
      html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
      [data-testid="stMain"], [data-testid="stSidebar"] { background: #111827; color: #eef4ff; }
      .block-container { max-width: 920px; padding-top: 1.2rem; padding-bottom: 7rem; }
      [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #2b3850; }
      [data-testid="stSidebar"] * { color: #eef4ff; }
      [data-testid="stSidebar"] .stButton { margin-bottom: .55rem; }
      [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        border-radius: 7px; height: 40px; width: 100%; padding: 0 .8rem;
        border: 1px solid #4f8dff; background: #2878ed; color: #ffffff;
        font-size: .9rem; font-weight: 600; display: block;
      }
      [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background: #4d91ff; border-color: #73a9ff; }
      [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        border: 1px solid #374862; background: #1e2b40; border-radius: 999px;
        min-height: 36px; padding: 0 .9rem; text-align: left; font-size: .86rem;
      }
      [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover { background: #2b3c57; border-color: #526b91; }
      .rag-brand { color: #f3f7ff; text-align: right; font-size: 1.15rem; font-weight: 700; letter-spacing: .08em; margin: .1rem 0 .65rem; }
      .add-bar { border: 1px solid #2e405e; border-radius: 10px; padding: .4rem .8rem .75rem; margin-bottom: 1.25rem; background: #18243a; }
      [data-testid="stChatInput"] { max-width: 900px; margin: auto; }
      [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"]:hover,
      [data-testid="stChatInput"] > div, [data-testid="stTextInput"] input,
      [data-baseweb="tab-panel"], [data-baseweb="tab-list"] { background: #18243a !important; color: #eef4ff !important; }
      [data-testid="stFileUploaderDropzone"] { border-color: #465b7d; min-height: 88px; }
      [data-testid="stChatMessage"] { background: transparent; }
      .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
      .stTabs [data-baseweb="tab"] { font-size: .83rem; padding: .35rem .65rem; }
      .stMarkdown, .stCaption, [data-testid="stFileUploaderDropzone"] *,
      [data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea,
      [data-baseweb="tab"], [data-testid="stStatusWidget"] * { color: #eef4ff !important; }
      [data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea::placeholder { color: #aab8cd !important; }
      [data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea { background: #18243a !important; border-color: #465b7d !important; }
      [data-testid="stMain"] .stButton > button { background: #233552; color: #eef4ff; border-color: #465b7d; }
      [data-testid="stMain"] .stButton > button:hover { background: #2c456b; border-color: #6684b4; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### User Memory")
    st.text_input(
        "User ID",
        key="user_id",
        help="Persistent semantic preferences and conversation memory are scoped to this ID.",
    )
    st.caption("Memory is stored locally in data/memory.db")
    st.divider()
    st.markdown("### Conversations")
    if st.button("+  New Chat", key="new_thread", type="primary", help="Start a new conversation"):
        add_thread()
        st.rerun()
    st.caption("SAVED THREADS")
    thread_ids = list(st.session_state.threads)
    for index, value in enumerate(thread_ids):
        t_data = st.session_state.threads[value]
        label = t_data.get("title") or (f"Conversation {index + 1}" if value != thread_ids[-1] else "Active Chat")
        if st.button(label, key=f"thread_{value}", type="secondary", use_container_width=True):
            st.session_state.active_thread = value
            # Reload messages from DB to ensure freshest state
            db_msgs = get_messages(value)
            st.session_state.threads[value]["messages"] = [{"role": m["role"], "content": m["content"]} for m in db_msgs]
            st.rerun()

    try:
        from gateway import global_cost_tracker, global_response_cache
        stats = global_cost_tracker.get_global_stats()
        cache_stats = global_response_cache.stats()
        with st.expander("⚡ Gateway & Observability", expanded=False):
            st.caption(f"**Total Tokens:** {stats.get('total_tokens', 0):,}")
            st.caption(f"**Est. Cost:** ${stats.get('total_cost_usd', 0.0):.4f}")
            st.caption(f"**Cache Hits:** {cache_stats.get('hits', 0)} ({cache_stats.get('hit_rate_pct', 0.0)}%)")
            st.caption(f"**Avg Latency:** {stats.get('avg_latency_ms', 0.0)} ms")
    except Exception:
        pass

thread_id = st.session_state.active_thread
thread = st.session_state.threads[thread_id]

st.markdown("<div class='rag-brand'>RAG AGENT</div>", unsafe_allow_html=True)

try:
    from agents.github_analyzer_agent import analyze_github_repository
except ImportError:
    try:
        from app.agents.github_analyzer_agent import analyze_github_repository
    except ImportError:
        analyze_github_repository = None

# Top Dedicated Tabs for Direct YouTube, GitHub, Web Link, and File Uploads
st.markdown("<div class='add-bar'>", unsafe_allow_html=True)
yt_tab, gh_tab, web_tab, file_tab = st.tabs(["🎥 YouTube Video URL", "🐙 GitHub Repository", "🌐 Web Page Link", "📁 Upload Documents & Media"])

with yt_tab:
    col_yt1, col_yt2 = st.columns([4, 1])
    with col_yt1:
        yt_link = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...", key=f"yt_{thread_id}", label_visibility="collapsed")
    with col_yt2:
        btn_yt = st.button("Add YouTube", disabled=not yt_link.strip(), use_container_width=True)
    if btn_yt and yt_link.strip():
        try:
            with st.spinner("Extracting YouTube video transcript and indexing..."):
                ingest_url(yt_link.strip(), session_id=thread_id)
            thread["files"].append({"name": f"YouTube Video ({yt_link.strip()})", "path": yt_link.strip(), "kind": "youtube"})
            save_conversation_metadata(thread_id, files=thread["files"])
            st.success("YouTube video transcript successfully indexed!")
        except Exception as err:
            st.error(f"YouTube processing error: {err}")

with gh_tab:
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repository", key=f"gh_url_{thread_id}", label_visibility="collapsed")
    col1, col2 = st.columns([3, 1])
    with col1:
        focus_query = st.text_input("Analysis Focus (optional)", placeholder="e.g. Architecture, auth flow, logic breaks", key=f"gh_focus_{thread_id}", label_visibility="collapsed")
    with col2:
        branch_name = st.text_input("Branch (optional)", placeholder="main", key=f"gh_branch_{thread_id}", label_visibility="collapsed")
    if st.button("Analyze GitHub Repo (Read-Only)", disabled=not repo_url.strip(), use_container_width=True):
        if analyze_github_repository is not None:
            try:
                with st.spinner("Analyzing repository structure, code, commits, and documentation..."):
                    result = analyze_github_repository(
                        repo_url=repo_url.strip(),
                        focus_query=focus_query.strip(),
                        branch=branch_name.strip() or None,
                        thread_id=thread_id,
                    )
                if result.get("error"):
                    st.error(result["error"])
                else:
                    report_md = result.get("markdown_report", "")
                    thread["messages"].append({"role": "user", "content": f"Analyze repository: {repo_url.strip()}"})
                    thread["messages"].append({"role": "assistant", "content": report_md})
                    thread["files"].append({"name": result.get("repo_name", repo_url.strip()), "path": repo_url.strip(), "kind": "github"})
                    save_message(thread_id, "user", f"Analyze repository: {repo_url.strip()}")
                    save_message(thread_id, "assistant", report_md)
                    save_conversation_metadata(thread_id, files=thread["files"])
                    st.success(f"Analyzed {result.get('repo_name', repo_url.strip())}")
                    st.rerun()
            except Exception as err:
                st.error(f"GitHub analysis failed: {err}")
        else:
            st.error("GitHub Analyzer subsystem not loaded.")

with web_tab:
    col_w1, col_w2 = st.columns([4, 1])
    with col_w1:
        web_link = st.text_input("Web Page URL", placeholder="https://example.com/article", key=f"web_{thread_id}", label_visibility="collapsed")
    with col_w2:
        btn_web = st.button("Add Web Link", disabled=not web_link.strip(), use_container_width=True)
    if btn_web and web_link.strip():
        try:
            with st.spinner("Downloading and indexing web page safely..."):
                ingest_url(web_link.strip(), session_id=thread_id)
            thread["files"].append({"name": web_link.strip(), "path": web_link.strip(), "kind": "web"})
            save_conversation_metadata(thread_id, files=thread["files"])
            st.success("Web page added and indexed!")
        except Exception as err:
            st.error(f"Could not add this link: {err}")

with file_tab:
    uploads = st.file_uploader(
        "Upload Documents, Images, Audio, or Videos",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key=f"uploads_{st.session_state.upload_version}",
        label_visibility="collapsed",
        help="PDFs, DOCX, PPTX, XLSX, TXT, CSV, MD, Images, Audio, and Video files are supported.",
    )
    if st.button("Process & Index Files", disabled=not uploads, use_container_width=True):
        for uploaded in uploads or []:
            try:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    path = save_upload(uploaded, thread_id)
                    ingest_file(str(path), session_id=thread_id)
                suffix = Path(uploaded.name).suffix.lower().lstrip(".")
                thread["files"].append({"name": uploaded.name, "path": str(path), "kind": suffix})
                save_conversation_metadata(thread_id, files=thread["files"])
                st.success(f"Indexed {uploaded.name}")
            except Exception as error:
                st.error(f"Could not add {uploaded.name}: {error}")
        st.session_state.upload_version += 1

st.markdown("</div>", unsafe_allow_html=True)

# Display Chat Messages
st.divider()
for message in thread["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

followups = thread.get("suggested_followups", [])
if followups:
    st.caption("Suggested follow-ups")
    followup_columns = st.columns(len(followups))
    for index, followup in enumerate(followups):
        if followup_columns[index].button(followup, key=f"followup_{thread_id}_{index}"):
            st.session_state.pending_question = followup
            st.rerun()

# Chat Input Area (supports questions as well as pasted YouTube, GitHub, and Web URLs automatically)
question = st.chat_input("Ask a question, or paste a YouTube / GitHub / Web link...")
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    thread["messages"].append({"role": "user", "content": question})
    save_message(thread_id, "user", question)
    
    # Auto-update conversation title if first turn
    if len(thread["messages"]) <= 2:
        short_title = question[:35] + "..." if len(question) > 35 else question
        thread["title"] = short_title
        save_conversation_metadata(thread_id, title=short_title)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            progress = st.status("Finding the best answer...", expanded=False)

            def answer_stream():
                first_token = True
                for token in stream_response(
                    question,
                    chat_messages(thread["messages"]),
                    thread_id,
                    thread.setdefault("biography", {}),
                ):
                    if first_token:
                        progress.update(label="Writing answer...", state="complete", expanded=False)
                        first_token = False
                    yield token
                if first_token:
                    progress.update(label="Answer ready", state="complete", expanded=False)

            response = st.write_stream(answer_stream())
            thread["messages"].append({"role": "assistant", "content": response})
            followups = [
                f"What are the key details of {question.strip().rstrip('?!. ')}?",
                f"What should I investigate next about {question.strip().rstrip('?!. ')}?",
            ]
            thread["suggested_followups"] = followups
            memory_store.add_turn_to_thread(
                thread_id,
                st.session_state.user_id or "default_user",
                ConversationTurn(
                    user_query=question,
                    agent_response=response,
                    suggested_followups=followups,
                    metadata={"source": "streamlit"},
                ),
            )
            save_message(thread_id, "assistant", response)
            save_conversation_metadata(thread_id, biography=thread.get("biography"))
        except Exception as error:
            st.error(f"Unable to answer: {error}")
