"""Async SQLAlchemy engine + session factory.

A single engine is created at import time; FastAPI dependencies hand out short-lived
`AsyncSession` objects. The engine pool is sized small — most of our latency budget
is in LLM/external API calls, not DB queries.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from finsight.settings import settings

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    future=True,
)

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager for ad-hoc usage outside FastAPI dependency injection."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def dispose_engine() -> None:
    await engine.dispose()
