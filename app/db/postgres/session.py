"""
Async SQLAlchemy engine and session factory.

Provides:
  - `engine`          — the shared AsyncEngine instance
  - `get_session()`   — FastAPI dependency that yields an AsyncSession per request
  - `init_db()`       — creates all tables on startup (idempotent via CREATE IF NOT EXISTS)

Usage in a FastAPI route:
    from app.db.postgres.session import get_session
    from sqlmodel.ext.asyncio.session import AsyncSession

    @router.get("/topics")
    async def list_topics(session: AsyncSession = Depends(get_session)):
        ...
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

@asynccontextmanager
async def get_async_session_context():
    """
    Async context manager for database sessions outside of request context.
    Used by Celery tasks which run outside FastAPI's dependency injection.
    """
    engine = create_async_engine(settings.POSTGRES_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        try:
            yield session
        finally:
            await engine.dispose()

# Engine 

def _create_engine() -> AsyncEngine:
    """
    Creates the async SQLAlchemy engine.

    NullPool is used because FastAPI's async request handlers each get their
    own short-lived connection via the session dependency — a connection pool
    would not provide meaningful benefits in this single-user, low-concurrency
    context, and NullPool avoids connection state issues between Celery workers
    and the FastAPI process.

    For a multi-user deployment, replace NullPool with AsyncConnectionPool
    and configure pool_size and max_overflow appropriately.
    """
    return create_async_engine(
        settings.POSTGRES_URL,
        echo=(settings.ENVIRONMENT == "development"),  # log SQL in dev only
        poolclass=NullPool,
    )


engine: AsyncEngine = _create_engine()

#  Session dependency 

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a scoped AsyncSession.

    The session is automatically committed on success and rolled back on
    exception. Always closed at the end of the request regardless of outcome.

    Use as a FastAPI Depends:
        session: AsyncSession = Depends(get_session)
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

#  Startup initialization 

async def init_db() -> None:
    """
    Creates all SQLModel tables in the database if they do not already exist.

    This is called once on application startup (from app/main.py lifespan).
    It is safe to call multiple times (CREATE TABLE IF NOT EXISTS semantics).

    For production schema migrations, use Alembic instead of this function.
    This function is appropriate for initial setup and development resets.

    Note: All SQLModel table classes must be imported before this is called
    so that SQLModel.metadata contains all table definitions. The import is
    done here explicitly to guarantee this.
    """
    # These imports register the table metadata with SQLModel.metadata.
    # They must happen before create_all — do not remove.
    import app.db.postgres.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)