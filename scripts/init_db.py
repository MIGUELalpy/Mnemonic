"""
scripts/init_db.py
───────────────────
One-time bootstrapping script. Run this once after Easypanel brings up
all the containers (PostgreSQL, Qdrant, Neo4j, Redis).

What it does:
  1. Verifies connectivity to all three databases.
  2. Creates all PostgreSQL tables (idempotent via CREATE IF NOT EXISTS).
  3. Creates Neo4j uniqueness constraints and seeds lookup nodes.
  4. Creates the Qdrant `knowledge_base` collection with payload indexes.
  5. Creates the initial user row in PostgreSQL.

Usage:
    # From the project root:
    python -m scripts.init_db --user "Your Name" --lang EN

    # To reset and re-initialize (drops all Neo4j data, Qdrant collection, PG tables):
    python -m scripts.init_db --user "Your Name" --reset
"""

import argparse
import asyncio
import sys

import structlog

# Configure minimal structlog output for the CLI context
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True),
    ]
)
log = structlog.get_logger()


async def check_postgres() -> bool:
    from sqlalchemy import text
    from app.db.postgres.session import engine
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("postgres.connected")
        return True
    except Exception as e:
        log.error("postgres.connection_failed", error=str(e))
        return False


async def check_qdrant() -> bool:
    from app.db.qdrant.client import qdrant_client
    try:
        await qdrant_client.get_collections()
        log.info("qdrant.connected")
        return True
    except Exception as e:
        log.error("qdrant.connection_failed", error=str(e))
        return False


async def check_neo4j() -> bool:
    from app.db.neo4j.client import verify_connectivity
    return await verify_connectivity()


async def init_all(display_name: str, ui_language: str, reset: bool = False) -> None:
    """
    Bootstraps all databases in the correct order.
    """
    log.info("init.starting", reset=reset)

    # Step 1: Connectivity checks 
    log.info("init.step", step=1, description="Checking connectivity to all databases")
    results = await asyncio.gather(
        check_postgres(),
        check_qdrant(),
        check_neo4j(),
        return_exceptions=False,
    )
    if not all(results):
        log.error(
            "init.connectivity_failed",
            postgres=results[0],
            qdrant=results[1],
            neo4j=results[2],
            message="One or more databases are unreachable. "
                    "Check Easypanel containers and .env file.",
        )
        sys.exit(1)

    # Step 2: Reset (optional, destructive) 
    if reset:
        log.warning("init.reset_requested", message="Dropping all existing data...")

        from app.db.neo4j.schema import drop_all_data
        from app.db.qdrant.client import qdrant_client
        from app.config import settings
        from sqlmodel import SQLModel
        from app.db.postgres.session import engine

        await drop_all_data()

        exists = await qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME)
        if exists:
            await qdrant_client.delete_collection(settings.QDRANT_COLLECTION_NAME)
            log.warning("qdrant.collection_deleted", name=settings.QDRANT_COLLECTION_NAME)

        import app.db.postgres.models  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
        log.warning("postgres.tables_dropped")

    #  Step 3: PostgreSQL tables 
    log.info("init.step", step=3, description="Creating PostgreSQL tables")
    from app.db.postgres.session import init_db
    await init_db()
    log.info("postgres.tables_ready")

    #  Step 4: Neo4j schema 
    log.info("init.step", step=4, description="Applying Neo4j constraints and seeding lookup nodes")
    from app.db.neo4j.schema import init_schema
    await init_schema()
    log.info("neo4j.schema_ready")

    # Step 5: Qdrant collection 
    log.info("init.step", step=5, description="Creating Qdrant collection and payload indexes")
    from app.db.qdrant.client import init_collection
    await init_collection()
    log.info("qdrant.collection_ready")

    #  Step 6: Initial user 
    log.info("init.step", step=6, description="Creating initial user record")
    from app.db.postgres.session import AsyncSession, engine
    from app.db.postgres.crud import get_user, create_user

    async with AsyncSession(engine, expire_on_commit=False) as session:
        existing_user = await get_user(session)
        if existing_user:
            log.info(
                "postgres.user_exists",
                user_id="(already exists)",
                name="(already exists)",
                message="User already exists — skipping creation.",
            )
        else:
            user = await create_user(
                session,
                display_name=display_name,
                ui_language=ui_language.upper(),
            )
            user_id = user.id
            user_name = user.display_name
            await session.commit()
            log.info(
                "postgres.user_created",
                user_id=user_id,
                name=user_name,
            )

    #  Done 
    log.info(
        "init.complete",
        message="All databases initialized successfully. Mnemonic is ready.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize Mnemonic databases (PostgreSQL, Neo4j, Qdrant)."
    )
    parser.add_argument(
        "--user",
        type=str,
        default="Developer",
        help="Display name for the initial user record (default: 'Developer').",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="EN",
        choices=["EN", "PT", "DE", "ZH"],
        help="Preferred UI language for the user (default: EN).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all existing data before initializing. DESTRUCTIVE — use with caution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(init_all(
        display_name=args.user,
        ui_language=args.lang,
        reset=args.reset,
    ))