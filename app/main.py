from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field as PydanticField
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.core.auth import verify_token
from app.db.neo4j.client import close_driver, verify_connectivity as neo4j_ping
from app.db.neo4j.operations import get_graph_stats, get_all_documents, get_all_concepts
from app.db.postgres.crud import create_topic, get_all_topics, get_due_topics, get_user
from app.db.postgres.session import get_session, init_db
from app.db.qdrant.client import get_collection_info, init_collection
from app.routers.ingestion import router as ingestion_router
from app.routers.classifier import router as classifier_router
from app.routers.chat import router as chat_router
from app.routers.challenge import router as challenge_router
from app.routers.submission import router as submission_router
from app.routers.logs import router as logs_router

log = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log.info("mnemonic.startup", environment=settings.ENVIRONMENT)
    await init_db()
    await init_collection()
    if not await neo4j_ping():
        log.warning("neo4j.startup_unreachable")
    log.info("mnemonic.ready")
    yield
    await close_driver()
    log.info("mnemonic.shutdown")

app = FastAPI(
    title="Mnemonic API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url=None,
)

app.include_router(ingestion_router)
app.include_router(classifier_router)
app.include_router(chat_router)
app.include_router(challenge_router)
app.include_router(submission_router)
app.include_router(logs_router)

app.mount("/static", StaticFiles(directory="/app/app/static"), name="static")

@app.get("/ui", include_in_schema=False)
async def serve_ui():
    return FileResponse("/app/app/static/index.html")

@app.get("/health", tags=["system"])
async def health_check(session: AsyncSession = Depends(get_session)) -> dict:
    try:
        await session.exec(text("SELECT 1"))  # type: ignore[call-overload]
        postgres_ok = True
    except Exception:
        postgres_ok = False
    neo4j_ok = await neo4j_ping()
    try:
        await get_collection_info()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    all_ok = postgres_ok and neo4j_ok and qdrant_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "services": {
            "postgres": "ok" if postgres_ok else "unreachable",
            "neo4j": "ok" if neo4j_ok else "unreachable",
            "qdrant": "ok" if qdrant_ok else "unreachable",
        },
        "environment": settings.ENVIRONMENT,
    }

@app.get("/admin/db", tags=["admin"], dependencies=[Depends(verify_token)])
async def database_stats() -> dict:
    return {"qdrant": await get_collection_info(), "neo4j": await get_graph_stats()}

@app.get("/admin/db/documents", tags=["admin"], dependencies=[Depends(verify_token)])
async def list_knowledge_base_documents() -> dict:
    docs = await get_all_documents()
    return {"count": len(docs), "documents": docs}

@app.get("/admin/db/concepts", tags=["admin"], dependencies=[Depends(verify_token)])
async def list_concepts() -> dict:
    concepts = await get_all_concepts()
    return {"count": len(concepts), "concepts": concepts}

class TopicCreate(BaseModel):
    name: str = PydanticField(max_length=200)
    programming_language: str = PydanticField(description="C++ | Python | C#")
    description: str | None = None

@app.get("/topics", tags=["topics"], dependencies=[Depends(verify_token)])
async def list_topics(session: AsyncSession = Depends(get_session)) -> dict:
    user = await get_user(session)
    if not user:
        return {"topics": [], "message": "No user found. Run scripts/init_db.py first."}
    topics = await get_all_topics(session, user.id)
    return {"count": len(topics), "topics": [
        {"id": t.id, "name": t.name, "programming_language": t.programming_language,
         "fsrs_retrievability": round(t.fsrs_retrievability, 4),
         "next_review_date": str(t.next_review_date) if t.next_review_date else None,
         "total_reviews": t.total_reviews}
        for t in topics
    ]}

@app.get("/topics/due", tags=["topics"], dependencies=[Depends(verify_token)])
async def list_due_topics(session: AsyncSession = Depends(get_session)) -> dict:
    user = await get_user(session)
    if not user:
        return {"topics": [], "message": "No user found. Run scripts/init_db.py first."}
    topics = await get_due_topics(session, user.id)
    return {"count": len(topics), "topics": [
        {"id": t.id, "name": t.name, "programming_language": t.programming_language,
         "fsrs_retrievability": round(t.fsrs_retrievability, 4),
         "next_review_date": str(t.next_review_date)}
        for t in topics
    ]}

@app.post("/topics", tags=["topics"], dependencies=[Depends(verify_token)], status_code=201)
async def add_topic(body: TopicCreate, session: AsyncSession = Depends(get_session)) -> dict:
    if body.programming_language not in ("C++", "Python", "C#"):
        raise HTTPException(status_code=422, detail="programming_language must be C++, Python, or C#")
    user = await get_user(session)
    if not user:
        raise HTTPException(status_code=400, detail="No user found. Run scripts/init_db.py first.")
    topic = await create_topic(session, user_id=user.id, name=body.name,
        programming_language=body.programming_language,
        description=body.description)
    return {"id": topic.id, "name": topic.name, "programming_language": topic.programming_language}