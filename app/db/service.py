"""Database helper service for managing conversations, message histories, and user memory."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from sqlalchemy import desc, select

try:
    from app.db.database import get_db_session
    from app.db.models import CheckpointModel, CheckpointWriteModel, ConversationModel, MessageModel
except ModuleNotFoundError:
    from db.database import get_db_session
    from db.models import CheckpointModel, CheckpointWriteModel, ConversationModel, MessageModel

logger = logging.getLogger("DatabaseService")


def create_or_get_conversation(session_id: str, title: str | None = None) -> dict[str, Any]:
    """Ensure a conversation record exists in the database and return its metadata."""
    with get_db_session() as session:
        conv = session.execute(
            select(ConversationModel).where(ConversationModel.session_id == session_id)
        ).scalars().first()

        if not conv:
            conv = ConversationModel(
                session_id=session_id,
                title=title or "New Chat",
                metadata_json=json.dumps({"biography": {}, "files": []}),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(conv)
            session.flush()

        meta = json.loads(conv.metadata_json or "{}")
        return {
            "session_id": conv.session_id,
            "title": conv.title,
            "metadata": meta,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        }


def list_conversations() -> list[dict[str, Any]]:
    """List all saved conversations ordered by latest activity."""
    with get_db_session() as session:
        rows = session.execute(
            select(ConversationModel).order_by(desc(ConversationModel.updated_at))
        ).scalars().all()

        results = []
        for r in rows:
            meta = json.loads(r.metadata_json or "{}")
            results.append({
                "session_id": r.session_id,
                "title": r.title,
                "metadata": meta,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            })
        return results


def save_conversation_metadata(session_id: str, title: str | None = None, biography: dict | None = None, files: list | None = None) -> None:
    """Update conversation title, biography memory, or file attachment lists."""
    with get_db_session() as session:
        conv = session.execute(
            select(ConversationModel).where(ConversationModel.session_id == session_id)
        ).scalars().first()

        if not conv:
            conv = ConversationModel(
                session_id=session_id,
                title=title or "New Chat",
                metadata_json=json.dumps({"biography": biography or {}, "files": files or []}),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(conv)
        else:
            meta = json.loads(conv.metadata_json or "{}")
            if title:
                conv.title = title
            if biography is not None:
                meta["biography"] = biography
            if files is not None:
                meta["files"] = files
            conv.metadata_json = json.dumps(meta)
            conv.updated_at = datetime.utcnow()


def save_message(
    session_id: str,
    role: str,
    content: str,
    node_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a chat message into persistent database storage."""
    with get_db_session() as session:
        # Also ensure conversation exists
        create_or_get_conversation(session_id)

        msg = MessageModel(
            session_id=session_id,
            role=role,
            content=content,
            node_name=node_name,
            metadata_json=json.dumps(metadata or {}),
            created_at=datetime.utcnow(),
        )
        session.add(msg)


def get_messages(session_id: str) -> list[dict[str, Any]]:
    """Retrieve ordered chat messages for a session/thread."""
    with get_db_session() as session:
        rows = session.execute(
            select(MessageModel)
            .where(MessageModel.session_id == session_id)
            .order_by(MessageModel.created_at)
        ).scalars().all()

        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "node_name": r.node_name,
                "metadata": json.loads(r.metadata_json or "{}"),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def delete_conversation(session_id: str) -> None:
    """Delete a conversation and all its associated messages and checkpoints."""
    with get_db_session() as session:
        session.query(MessageModel).filter(MessageModel.session_id == session_id).delete()
        session.query(ConversationModel).filter(ConversationModel.session_id == session_id).delete()
        session.query(CheckpointModel).filter(CheckpointModel.thread_id == session_id).delete()
        session.query(CheckpointWriteModel).filter(CheckpointWriteModel.thread_id == session_id).delete()


def set_active_source(session_id: str, source_info: dict[str, Any]) -> None:
    """Set the active source metadata for a conversation session/thread."""
    if not session_id or not source_info:
        return
    with get_db_session() as session:
        conv = session.execute(
            select(ConversationModel).where(ConversationModel.session_id == session_id)
        ).scalars().first()

        if not conv:
            conv = ConversationModel(
                session_id=session_id,
                title=source_info.get("title", "Active Source Chat"),
                metadata_json=json.dumps({"biography": {}, "files": [], "active_source": source_info}),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(conv)
        else:
            meta = json.loads(conv.metadata_json or "{}")
            meta["active_source"] = source_info
            conv.metadata_json = json.dumps(meta)
            conv.updated_at = datetime.utcnow()


def get_active_source(session_id: str) -> dict[str, Any] | None:
    """Retrieve active source metadata for a conversation session/thread."""
    if not session_id:
        return None
    try:
        with get_db_session() as session:
            conv = session.execute(
                select(ConversationModel).where(ConversationModel.session_id == session_id)
            ).scalars().first()

            if conv:
                meta = json.loads(conv.metadata_json or "{}")
                return meta.get("active_source")
    except Exception:
        pass
    return None
