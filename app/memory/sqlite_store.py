"""SQLite-backed durable storage for semantic and episodic memory."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .schemas import ConversationTurn, ThreadConversation, UserPreferences


class SQLiteMemoryStore:
    """Small zero-configuration SQLite repository for persistent memory."""

    def __init__(self, db_path: str = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    data JSON NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    data JSON NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(user_id);
                """
            )

    def get_user_profile(self, user_id: str) -> UserPreferences:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return UserPreferences.model_validate_json(row["data"]) if row else UserPreferences(user_id=user_id)

    def save_user_profile(self, profile: UserPreferences) -> None:
        now = profile.last_updated.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles(user_id, data, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
                """,
                (profile.user_id, profile.model_dump_json(), now),
            )

    def get_thread(self, thread_id: str, user_id: str) -> ThreadConversation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data FROM chat_threads WHERE thread_id = ? AND user_id = ?",
                (thread_id, user_id),
            ).fetchone()
        if row:
            return ThreadConversation.model_validate_json(row["data"])
        return ThreadConversation(thread_id=thread_id, user_id=user_id)

    def save_thread(self, thread: ThreadConversation) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_threads(thread_id, user_id, title, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    title = excluded.title,
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (
                    thread.thread_id,
                    thread.user_id,
                    thread.title,
                    thread.model_dump_json(),
                    thread.created_at.isoformat(),
                    thread.updated_at.isoformat(),
                ),
            )

    def add_turn_to_thread(self, thread_id: str, user_id: str, turn: ConversationTurn) -> None:
        thread = self.get_thread(thread_id, user_id)
        if not thread.turns:
            title = turn.user_query.strip().replace("\n", " ")[:80] or "New Conversation"
            thread.title = title
        thread.turns.append(turn)
        thread.updated_at = datetime.now(timezone.utc)
        self.save_thread(thread)

    def list_user_threads(self, user_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT thread_id, title, created_at, updated_at, data FROM chat_threads "
                "WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        summaries: List[Dict[str, Any]] = []
        for row in rows:
            thread = ThreadConversation.model_validate_json(row["data"])
            summaries.append(
                {
                    "thread_id": thread.thread_id,
                    "user_id": thread.user_id,
                    "title": thread.title,
                    "turn_count": len(thread.turns),
                    "created_at": thread.created_at.isoformat(),
                    "updated_at": thread.updated_at.isoformat(),
                }
            )
        return summaries
