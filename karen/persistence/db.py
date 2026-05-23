"""Async SQLite database engine and session management."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from karen.persistence.models import Base

DB_PATH = Path(__file__).parent.parent.parent / "data" / "karen.db"

_engine = create_async_engine(
    f"sqlite+aiosqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"Database initialised at {DB_PATH}")


async def get_session() -> AsyncSession:
    """Return a new async session (caller must close it)."""
    return AsyncSessionLocal()
