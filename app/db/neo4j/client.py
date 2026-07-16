"""
Neo4j async driver setup.

Provides:
  - `neo4j_driver`     — the shared AsyncDriver instance (module-level singleton)
  - `get_session()`    — async context manager yielding an AsyncSession
  - `close_driver()`   — graceful shutdown (called in app lifespan)

The Neo4j driver maintains its own connection pool internally.
Unlike PostgreSQL (where we use NullPool), the Neo4j driver's built-in
pooling is appropriate and should not be disabled.

Usage:
    from app.db.neo4j.client import get_session

    async with get_session() as session:
        result = await session.run("MATCH (c:Concept) RETURN c.name LIMIT 5")
        records = await result.values()
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from app.config import settings

log = structlog.get_logger(__name__)

#  Driver singleton 

def _create_driver() -> AsyncDriver:
    """
    Creates the Neo4j async driver.

    The driver is configured with:
      - connection_timeout: 10s — fail fast if Neo4j is unreachable on startup.
      - max_connection_lifetime: 30 min — prevents stale connections.
      - max_connection_pool_size: 10 — generous for single-user workload.
      - encrypted: False — self-hosted, local network only.
    """
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        max_connection_lifetime=30 * 60,   # 30 minutes
        max_connection_pool_size=10,
        connection_timeout=10.0,
        encrypted=False,
    )


neo4j_driver: AsyncDriver = _create_driver()


#  Session context manager 

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a Neo4j AsyncSession.

    Automatically closes the session when the block exits, regardless of
    whether an exception was raised.

    Usage:
        async with get_session() as session:
            result = await session.run("MATCH (n) RETURN count(n)")
    """
    async with neo4j_driver.session() as session:
        yield session


#  Connectivity check 

async def verify_connectivity() -> bool:
    """
    Pings Neo4j to verify the connection is alive.
    Called on startup by the init script and health check endpoint.

    Returns True if connected, False if unreachable.
    """
    try:
        await neo4j_driver.verify_connectivity()
        log.info("neo4j.connected", uri=settings.NEO4J_URI)
        return True
    except Exception as exc:
        log.error("neo4j.connection_failed", uri=settings.NEO4J_URI, error=str(exc))
        return False


async def close_driver() -> None:
    """
    Closes the Neo4j driver and releases all pooled connections.
    Must be called during application shutdown (app/main.py lifespan).
    """
    await neo4j_driver.close()
    log.info("neo4j.driver_closed")