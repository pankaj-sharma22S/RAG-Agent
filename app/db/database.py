"""SQLAlchemy Database configuration supporting SQLite (dev) and PostgreSQL (prod) via DATABASE_URL."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
logger = logging.getLogger("Database")

# Base directory
APP_DIR = Path(__file__).resolve().parent.parent

# Database URL configuration
# Development SQLite: sqlite:///./app/data/dev.db
# Production PostgreSQL: postgresql://user:password@localhost:5432/rag_db
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{APP_DIR / 'data' / 'dev.db'}")

if DATABASE_URL.startswith("sqlite"):
    # Ensure directory for SQLite DB exists
    db_path = DATABASE_URL.replace("sqlite:///", "")
    db_file = Path(db_path)
    if not db_file.is_absolute():
        db_file = (APP_DIR.parent / db_path).resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_file.as_posix()}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    logger.info(f"Initialized SQLite database at {db_file}")

elif DATABASE_URL.startswith(("postgresql", "postgres")):
    # PostgreSQL configuration with connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        pool_pre_ping=True,
        echo=False,
    )
    logger.info(f"Initialized PostgreSQL database connection pool")

else:
    engine = create_engine(DATABASE_URL, echo=False)
    logger.info(f"Initialized database with URL: {DATABASE_URL}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Create all required tables automatically if they do not exist."""
    try:
        from app.db import models  # noqa: F401
    except ModuleNotFoundError:
        import db.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized and verified.")


@contextmanager
def get_db_session() -> Generator:
    """Context manager for thread-safe database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
