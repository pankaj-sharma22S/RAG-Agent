"""SQLAlchemy-based persistent LangGraph CheckpointSaver for SQLite and PostgreSQL."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator, Sequence
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import desc, select

try:
    from app.db.database import get_db_session, init_db
    from app.db.models import CheckpointModel, CheckpointWriteModel
except ModuleNotFoundError:
    from db.database import get_db_session, init_db
    from db.models import CheckpointModel, CheckpointWriteModel

logger = logging.getLogger("SqlAlchemyCheckpointSaver")


class SqlAlchemyCheckpointSaver(BaseCheckpointSaver):
    """Thread-safe persistent checkpointer using SQLAlchemy (works with SQLite & PostgreSQL)."""

    def __init__(self) -> None:
        super().__init__(serde=JsonPlusSerializer())
        # Ensure database tables exist
        init_db()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch the latest or specific checkpoint tuple for a given thread."""
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with get_db_session() as session:
            query = select(CheckpointModel).where(
                CheckpointModel.thread_id == thread_id,
                CheckpointModel.checkpoint_ns == checkpoint_ns,
            )
            if checkpoint_id:
                query = query.where(CheckpointModel.checkpoint_id == str(checkpoint_id))
            else:
                query = query.order_by(desc(CheckpointModel.created_at))

            row = session.execute(query).scalars().first()
            if not row:
                return None

            # Deserialize checkpoint and metadata
            type_str = row.type or "msgpack"
            checkpoint: Checkpoint = self.serde.loads_typed((type_str, row.checkpoint_blob))
            metadata: CheckpointMetadata = self.serde.loads_typed((type_str, row.metadata_blob))

            # Fetch any pending writes for this checkpoint
            writes_query = (
                select(CheckpointWriteModel)
                .where(
                    CheckpointWriteModel.thread_id == thread_id,
                    CheckpointWriteModel.checkpoint_ns == checkpoint_ns,
                    CheckpointWriteModel.checkpoint_id == row.checkpoint_id,
                )
                .order_by(CheckpointWriteModel.idx)
            )
            writes_rows = session.execute(writes_query).scalars().all()
            pending_writes: list[PendingWrite] = []
            for w in writes_rows:
                val = self.serde.loads_typed((w.type or "msgpack", w.value_blob)) if w.value_blob else None
                pending_writes.append((w.task_id, w.channel, val))

            # Fetch parent checkpoint config if present
            parent_config: RunnableConfig | None = None
            if row.parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.parent_checkpoint_id,
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoint tuples matching criteria."""
        with get_db_session() as session:
            query = select(CheckpointModel)
            if config:
                thread_id = str(config["configurable"].get("thread_id", ""))
                checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
                if thread_id:
                    query = query.where(CheckpointModel.thread_id == thread_id)
                if checkpoint_ns:
                    query = query.where(CheckpointModel.checkpoint_ns == checkpoint_ns)

            if before:
                before_id = before["configurable"].get("checkpoint_id")
                if before_id:
                    query = query.where(CheckpointModel.checkpoint_id < str(before_id))

            query = query.order_by(desc(CheckpointModel.created_at))
            if limit:
                query = query.limit(limit)

            rows = session.execute(query).scalars().all()
            for row in rows:
                type_str = row.type or "msgpack"
                checkpoint: Checkpoint = self.serde.loads_typed((type_str, row.checkpoint_blob))
                metadata: CheckpointMetadata = self.serde.loads_typed((type_str, row.metadata_blob))

                parent_config: RunnableConfig | None = None
                if row.parent_checkpoint_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": row.thread_id,
                            "checkpoint_ns": row.checkpoint_ns,
                            "checkpoint_id": row.parent_checkpoint_id,
                        }
                    }

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": row.thread_id,
                            "checkpoint_ns": row.checkpoint_ns,
                            "checkpoint_id": row.checkpoint_id,
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a new checkpoint state in the database."""
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        type_str, checkpoint_blob = self.serde.dumps_typed(checkpoint)
        _, metadata_blob = self.serde.dumps_typed(metadata)

        with get_db_session() as session:
            # Check if exists
            existing = session.execute(
                select(CheckpointModel).where(
                    CheckpointModel.thread_id == thread_id,
                    CheckpointModel.checkpoint_ns == checkpoint_ns,
                    CheckpointModel.checkpoint_id == checkpoint_id,
                )
            ).scalars().first()

            if existing:
                existing.parent_checkpoint_id = parent_checkpoint_id
                existing.type = type_str
                existing.checkpoint_blob = checkpoint_blob
                existing.metadata_blob = metadata_blob
            else:
                model = CheckpointModel(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    type=type_str,
                    checkpoint_blob=checkpoint_blob,
                    metadata_blob=metadata_blob,
                )
                session.add(model)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Record intermediate writes from graph execution tasks."""
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        with get_db_session() as session:
            for idx, (channel, value) in enumerate(writes):
                type_str, value_blob = self.serde.dumps_typed(value) if value is not None else (None, None)
                
                # Check for existing write to prevent constraint violation
                existing_write = session.execute(
                    select(CheckpointWriteModel).where(
                        CheckpointWriteModel.thread_id == thread_id,
                        CheckpointWriteModel.checkpoint_ns == checkpoint_ns,
                        CheckpointWriteModel.checkpoint_id == checkpoint_id,
                        CheckpointWriteModel.task_id == task_id,
                        CheckpointWriteModel.idx == idx,
                    )
                ).scalars().first()

                if existing_write:
                    existing_write.channel = channel
                    existing_write.type = type_str
                    existing_write.value_blob = value_blob
                    existing_write.task_path = task_path
                else:
                    write_model = CheckpointWriteModel(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        idx=idx,
                        channel=channel,
                        type=type_str,
                        value_blob=value_blob,
                        task_path=task_path,
                    )
                    session.add(write_model)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes belonging to a thread."""
        with get_db_session() as session:
            session.query(CheckpointModel).filter(CheckpointModel.thread_id == thread_id).delete()
            session.query(CheckpointWriteModel).filter(CheckpointWriteModel.thread_id == thread_id).delete()


_global_checkpointer: SqlAlchemyCheckpointSaver | None = None


def get_checkpointer() -> SqlAlchemyCheckpointSaver:
    """Retrieve or initialize singleton persistent SqlAlchemy checkpoint saver."""
    global _global_checkpointer
    if _global_checkpointer is None:
        _global_checkpointer = SqlAlchemyCheckpointSaver()
    return _global_checkpointer
