"""Incremental repository indexer tracking file hashes, commit SHAs, and checkpointed sync."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any
from langchain_core.documents import Document

try:
    from app.chunking.splitter import split_documents
    from app.retriever.sparse_retriever import global_sparse_retriever
    from app.tools.github_tools import (
        BINARY_EXTENSIONS,
        get_file,
        get_repository,
        get_repository_tree,
        parse_github_url,
    )
    from app.vectorstore.croma_store import get_vectorstore
except ModuleNotFoundError:
    from chunking.splitter import split_documents
    from retriever.sparse_retriever import global_sparse_retriever
    from tools.github_tools import (
        BINARY_EXTENSIONS,
        get_file,
        get_repository,
        get_repository_tree,
        parse_github_url,
    )
    from vectorstore.croma_store import get_vectorstore

logger = logging.getLogger("GitHubIndexer")

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "repo_index_state.json"
CHECKPOINT_BATCH_SIZE = 10


class GitHubIncrementalIndexer:
    """Stateful, checkpointed incremental indexer for GitHub repositories."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ensure_state_file()

    def _ensure_state_file(self) -> None:
        os.makedirs(STATE_FILE.parent, exist_ok=True)
        if not STATE_FILE.exists():
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_all_state(self) -> dict[str, Any]:
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_all_state(self, state: dict[str, Any]) -> None:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def get_repo_state(self, repo_key: str) -> dict[str, Any]:
        with self._lock:
            all_state = self._load_all_state()
            return all_state.get(repo_key, {"files": {}, "last_commit_sha": None})

    def save_repo_state(self, repo_key: str, repo_state: dict[str, Any]) -> None:
        with self._lock:
            all_state = self._load_all_state()
            all_state[repo_key] = repo_state
            self._save_all_state(all_state)

    def index_repository(
        self,
        repo_url: str,
        branch: str | None = None,
        session_id: str = "default",
        max_files: int = 50,
    ) -> dict[str, Any]:
        """Incrementally index repository into ChromaDB and BM25 sparse index.

        Detects added, modified, deleted, and unchanged files using git tree blob SHAs.
        Only indexes added and modified files; skips unchanged files; cleans deleted files.
        """
        owner, repo, url_branch = parse_github_url(repo_url)
        target_branch = branch or url_branch
        if not target_branch:
            repo_info = get_repository(owner, repo)
            target_branch = repo_info["default_branch"]

        repo_key = f"{owner}/{repo}:{target_branch}"
        current_state = self.get_repo_state(repo_key)
        indexed_files: dict[str, str] = current_state.get("files", {})  # path -> blob_sha

        # 1. Fetch current repository tree (blobs)
        tree = get_repository_tree(owner, repo, branch=target_branch, recursive=True)
        code_blobs = [
            item for item in tree
            if item.get("type") == "blob"
            and os.path.splitext(item.get("path", ""))[1].lower() not in BINARY_EXTENSIONS
            and not item.get("path", "").startswith((".git/", "node_modules/", "venv/", ".venv/"))
        ]

        current_tree_map: dict[str, str] = {
            item["path"]: item.get("sha", "") for item in code_blobs if item.get("path")
        }

        # 2. Compute file diff categories
        added_paths = [p for p in current_tree_map if p not in indexed_files]
        modified_paths = [p for p in current_tree_map if p in indexed_files and indexed_files[p] != current_tree_map[p]]
        deleted_paths = [p for p in indexed_files if p not in current_tree_map]
        unchanged_paths = [p for p in current_tree_map if p in indexed_files and indexed_files[p] == current_tree_map[p]]

        files_to_index = (added_paths + modified_paths)[:max_files]

        logger.info(
            f"Incremental Index [{repo_key}]: {len(added_paths)} added, {len(modified_paths)} modified, "
            f"{len(deleted_paths)} deleted, {len(unchanged_paths)} unchanged. Processing {len(files_to_index)} files."
        )

        total_chunks = 0
        chroma = get_vectorstore()

        # 3. Process Deleted Files
        if deleted_paths:
            for d_path in deleted_paths:
                try:
                    # Remove from Chroma
                    chroma.delete(where={"$and": [{"source": d_path}, {"session_id": session_id}]})
                except Exception:
                    pass
                indexed_files.pop(d_path, None)

        # 4. Fetch and Index Added & Modified Files with Checkpointing
        new_docs: list[Document] = []
        for idx, path in enumerate(files_to_index, start=1):
            try:
                content = get_file(owner, repo, path, branch=target_branch)
                if content and not content.startswith("["):
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": path,
                            "repo": f"{owner}/{repo}",
                            "branch": target_branch,
                            "session_id": session_id,
                            "type": "github_file",
                            "sha": current_tree_map[path],
                        },
                    )
                    new_docs.append(doc)
                    indexed_files[path] = current_tree_map[path]

                # Checkpoint in batches of CHECKPOINT_BATCH_SIZE
                if idx % CHECKPOINT_BATCH_SIZE == 0 or idx == len(files_to_index):
                    if new_docs:
                        chunks = split_documents(new_docs)
                        for c in chunks:
                            c.metadata["session_id"] = session_id
                        chroma.add_documents(chunks)
                        global_sparse_retriever.index_documents(chunks, session_id=session_id)
                        total_chunks += len(chunks)
                        new_docs = []

                    # Save checkpoint
                    self.save_repo_state(repo_key, {
                        "files": indexed_files,
                        "last_indexed_at": os.getenv("CURRENT_TIME", "2026-08-16T21:00:00Z"),
                    })
            except Exception as err:
                logger.warning(f"Failed indexing file {path}: {err}")

        # Final state save
        self.save_repo_state(repo_key, {
            "files": indexed_files,
            "last_indexed_at": os.getenv("CURRENT_TIME", "2026-08-16T21:00:00Z"),
        })

        return {
            "status": "indexed",
            "repo": repo_key,
            "added_count": len(added_paths),
            "modified_count": len(modified_paths),
            "deleted_count": len(deleted_paths),
            "unchanged_count": len(unchanged_paths),
            "indexed_files": len(files_to_index),
            "chunks_indexed": total_chunks,
        }


global_github_indexer = GitHubIncrementalIndexer()
