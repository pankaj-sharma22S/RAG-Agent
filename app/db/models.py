"""SQLAlchemy ORM models for LangGraph checkpoints, writes, conversations, and memory."""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)

try:
    from app.db.database import Base
except ModuleNotFoundError:
    from db.database import Base


class CheckpointModel(Base):
    """LangGraph state checkpoint storage model."""

    __tablename__ = "checkpoints"

    thread_id = Column(String(255), primary_key=True, nullable=False)
    checkpoint_ns = Column(String(255), primary_key=True, nullable=False, default="")
    checkpoint_id = Column(String(255), primary_key=True, nullable=False)
    parent_checkpoint_id = Column(String(255), nullable=True)
    type = Column(String(50), nullable=True)
    checkpoint_blob = Column(LargeBinary, nullable=False)
    metadata_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_checkpoints_thread_ns", "thread_id", "checkpoint_ns"),
        Index("idx_checkpoints_created_at", "created_at"),
    )


class CheckpointWriteModel(Base):
    """LangGraph channel intermediate writes storage model."""

    __tablename__ = "checkpoint_writes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), nullable=False)
    checkpoint_ns = Column(String(255), nullable=False, default="")
    checkpoint_id = Column(String(255), nullable=False)
    task_id = Column(String(255), nullable=False)
    idx = Column(Integer, nullable=False, default=0)
    channel = Column(String(255), nullable=False)
    type = Column(String(50), nullable=True)
    value_blob = Column(LargeBinary, nullable=True)
    task_path = Column(String(255), nullable=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_checkpoint_writes_lookup", "thread_id", "checkpoint_ns", "checkpoint_id"),
        UniqueConstraint("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx", name="uq_checkpoint_write"),
    )


class ConversationModel(Base):
    """Conversation thread metadata storage."""

    __tablename__ = "conversations"

    session_id = Column(String(255), primary_key=True, nullable=False)
    title = Column(String(500), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class MessageModel(Base):
    """Chat message history and node execution traces."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    node_name = Column(String(100), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
