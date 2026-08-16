"""Database and persistence package supporting SQLite and PostgreSQL."""

try:
    from app.db.checkpointer import SqlAlchemyCheckpointSaver, get_checkpointer
    from app.db.database import Base, SessionLocal, engine, get_db_session, init_db
    from app.db.models import CheckpointModel, CheckpointWriteModel, ConversationModel, MessageModel
except ModuleNotFoundError:
    from db.checkpointer import SqlAlchemyCheckpointSaver, get_checkpointer
    from db.database import Base, SessionLocal, engine, get_db_session, init_db
    from db.models import CheckpointModel, CheckpointWriteModel, ConversationModel, MessageModel

__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "init_db",
    "get_db_session",
    "CheckpointModel",
    "CheckpointWriteModel",
    "ConversationModel",
    "MessageModel",
    "SqlAlchemyCheckpointSaver",
    "get_checkpointer",
]
