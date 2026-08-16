"""Polished Streamlit interface for the session-isolated knowledge workspace.

Run from the repository root with ``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Keep the interface usable if Streamlit is holding an older ``ingest`` module
# in memory during a hot reload. A full restart will use the normal import.
try:
    from ingest import ingest_file, ingest_url
except ImportError:
    from ingest import ingest_document, ingest_file
    from ingestion.web_loader import load_web_page
    from ingestion.youtube_loader import load_youtube

    def ingest_url(url: str, session_id: str):
        document = load_youtube(url) if ("youtube.com" in url.lower() or "youtu.be" in url.lower()) else load_web_page(url)
        return ingest_document(document, session_id)
# Streamlit can briefly retain a previous module object after source changes.
# Use the normal token stream when available, with a safe graph fallback until
# the server completes its next clean reload.
try:
    from rag_workflow import stream_response
except ImportError:
    from rag_workflow import graph_app

    def stream_response(question: str, messages, session_id: str, biography=None):
        result = graph_app.invoke(
            {"question": question, "messages": messages, "session_id": session_id, "biography": biography or {}},
            config={"configurable": {"thread_id": session_id}},
        )
        yield str(result["messages"][-1].content)


UPLOAD_ROOT = APP_DIR / "data" / "uploads"
SUPPORTED_TYPES = [
    "pdf", "docx", "pptx", "xlsx", "xls", "png", "jpg", "jpeg", "webp",
    "bmp", "tif", "tiff", "mp3", "wav", "m4a", "mp4", "mkv", "avi", "mov",
]
IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
VIDEO_TYPES = {"mp4", "mkv", "avi", "mov"}


def new_thread() -> str:
    return f"thread-{uuid.uuid4().hex[:12]}"


def initialise_state() -> None:
    if "threads" not in st.session_state:
        first_thread = new_thread()
        st.session_state.threads = {first_thread: {"messages": [], "files": [], "biography": {}, "created_at": datetime.now().isoformat()}}
        st.session_state.active_thread = first_thread
        st.session_state.upload_version = 0


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
    st.session_state.threads[thread_id] = {"messages": [], "files": [], "biography": {}, "created_at": datetime.now().isoformat()}
    st.session_state.active_thread = thread_id
    st.session_state.upload_version += 1


def chat_messages(messages: list[dict[str, str]]):
    """Convert the visible conversation into LangChain messages for streaming."""
    return [
        HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(content=item["content"])
        for item in messages
    ]


st.set_page_config(page_title="Knowledge Workspace", page_icon="K", layout="wide")
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
      .add-bar { border: 1px solid #2e405e; border-radius: 10px; padding: .2rem .7rem .65rem; margin-bottom: 1.25rem; background: #18243a; }
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
    st.markdown("### Conversations")
    if st.button("+  New Chat", key="new_thread", type="primary", help="Start a new conversation"):
        add_thread()
        st.rerun()
    st.caption("HISTORY")
    thread_ids = list(st.session_state.threads)
    for index, value in enumerate(thread_ids):
        label = "New chat" if value == thread_ids[-1] else f"Conversation {index + 1}"
        if st.button(label, key=f"thread_{value}", type="secondary", use_container_width=True):
            st.session_state.active_thread = value
            st.rerun()

thread_id = st.session_state.active_thread
thread = st.session_state.threads[thread_id]

st.markdown("<div class='rag-brand'>RAG</div>", unsafe_allow_html=True)

st.markdown("<div class='add-bar'>", unsafe_allow_html=True)
upload_tab, link_tab = st.tabs(["Files", "Link"])
with upload_tab:
    uploads = st.file_uploader(
        "Add files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key=f"uploads_{st.session_state.upload_version}",
        label_visibility="collapsed",
        help="PDFs, documents, images, audio, and video are supported.",
    )
    if st.button("Add files", disabled=not uploads, use_container_width=True):
        for uploaded in uploads or []:
            try:
                with st.spinner(f"Adding {uploaded.name}..."):
                    path = save_upload(uploaded, thread_id)
                    ingest_file(str(path), session_id=thread_id)
                suffix = Path(uploaded.name).suffix.lower().lstrip(".")
                thread["files"].append({"name": uploaded.name, "path": str(path), "kind": suffix})
                st.success(f"Added {uploaded.name}")
            except Exception as error:
                st.error(f"Could not add {uploaded.name}: {error}")
        st.session_state.upload_version += 1
with link_tab:
    link = st.text_input("Add a public link", placeholder="Paste a webpage or YouTube link", label_visibility="collapsed")
    if st.button("Add link", disabled=not link.strip(), use_container_width=True):
        try:
            with st.spinner("Reading link..."):
                ingest_url(link.strip(), session_id=thread_id)
            label = link.strip().replace("https://", "").replace("http://", "")
            thread["files"].append({"name": label, "path": link.strip(), "kind": "link"})
            st.success("Link added")
        except Exception as error:
            st.error(f"Could not add this link: {error}")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()
for message in thread["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask anything about what you added..."):
    thread["messages"].append({"role": "user", "content": question})
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
        except Exception as error:
            st.error(f"Unable to answer: {error}")
