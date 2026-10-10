"""End-to-end LangSmith live trace verification script."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from app.tracing import setup_tracing
setup_tracing("rag-app")

print(f"Tracing active: {os.getenv('LANGCHAIN_TRACING_V2')} | Project: {os.getenv('LANGCHAIN_PROJECT')}", flush=True)

# 1. Test Single Traceable Function (Embeddings / Semantic Cache / Retriever / Guardrails)
from app.guardrails.input_guard import global_input_guard
from app.retriever.hybrid_retriever import global_hybrid_retriever
from app.retriever.reranker import global_reranker

print("\n--- 1. Testing Isolated Components Tracing ---", flush=True)
guard_res = global_input_guard.process("Explain multi-head attention in Transformer architecture.")
print(f"Input Guard Processed (Safe: {guard_res.is_safe})", flush=True)

ret_docs, d_cnt, s_cnt = global_hybrid_retriever.retrieve(
    query="multi-head attention",
    session_id="test_verify_session",
    top_k=4,
)
print(f"Hybrid Retriever (Dense: {d_cnt}, Sparse: {s_cnt}, Ret: {len(ret_docs)})", flush=True)

if ret_docs:
    reranked = global_reranker.rerank("multi-head attention", ret_docs, top_k=2)
    print(f"CrossEncoder Reranked: {len(reranked)} docs", flush=True)

# 2. Test End-to-End RAG Workflow Streaming Trace
print("\n--- 2. Testing End-to-End RAG Workflow Streaming ---", flush=True)
from app.rag_workflow import stream_response
from langchain_core.messages import HumanMessage

sources = []
tokens = []
for tok in stream_response(
    question="What is the role of self-attention in transformers?",
    messages=[HumanMessage(content="What is the role of self-attention in transformers?")],
    session_id="test_verify_session",
    sources_collector=sources,
):
    tokens.append(tok)

print(f"Stream Response complete: {len(tokens)} chunks yielded, {len(sources)} source citations collected.", flush=True)

# 3. Allow background HTTP queue to push runs to LangSmith
print("\n--- 3. Waiting 5 seconds for LangSmith asynchronous batch flush ---", flush=True)
time.sleep(5)

# 4. Fetch and Verify Actual Recorded Runs from LangSmith REST API
print("\n--- 4. Querying LangSmith REST API for Project 'rag-app' ---", flush=True)
from langsmith import Client

client = Client()
runs = list(client.list_runs(project_name="rag-app", limit=20))

print(f"Total Runs Found in LangSmith: {len(runs)}\n", flush=True)
print(f"{'RUN TYPE':<12} | {'STATUS':<8} | {'LATENCY (ms)':<12} | {'RUN NAME':<30} | {'TAGS'}", flush=True)
print("-" * 100, flush=True)

for r in runs[:15]:
    dur = (
        round((r.end_time - r.start_time).total_seconds() * 1000, 2)
        if (r.end_time and r.start_time)
        else "N/A"
    )
    tags_str = ", ".join(r.tags or [])
    print(f"{str(r.run_type).upper():<12} | {str(r.status):<8} | {str(dur):<12} | {str(r.name):<30} | {tags_str}", flush=True)

print("\nLangSmith trace verification complete!", flush=True)
