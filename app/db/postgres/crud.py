"""
CRUD (Create, Read, Update, Delete) operations for every PostgreSQL table.

All functions accept an AsyncSession and return typed SQLModel instances.
They do NOT commit — the session dependency in session.py handles commits,
which keeps transaction control at the request boundary.

FSRS integration note: functions that update FSRS state accept pre-computed
values from the fsrs-python library. This module does not run the FSRS
algorithm itself — it only persists the results.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.postgres.models import Challenge, GenerationLog, ReviewLog, Topic, User

log = structlog.get_logger(__name__)

# users

async def get_user(session: AsyncSession) -> User | None:
    """
    Returns the single user row, or None if the database has not been seeded.
    In a single-user system there is always at most one row in the users table.
    """
    result = await session.execute(select(User).limit(1))
    return result.scalars().first()


async def create_user(session: AsyncSession, display_name: str, ui_language: str = "EN") -> User:
    """
    Creates the initial user record. Should only be called once during setup.
    Raises ValueError if a user already exists.
    """
    existing = await get_user(session)
    if existing:
        raise ValueError(
            f"User already exists (id={existing.id}). "
            "Mnemonic is a single-user system."
        )
    user = User(display_name=display_name, ui_language=ui_language)
    session.add(user)
    await session.flush()  # flush to get the generated id without committing
    log.info("user.created", user_id=user.id, display_name=display_name)
    return user


async def update_streak(session: AsyncSession, user_id: int, current: int, longest: int) -> None:
    """Updates streak counters after a session is completed."""
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().one()
    user.current_streak = current
    user.longest_streak = max(longest, current)
    session.add(user)

# topics

async def get_due_topics(session: AsyncSession, user_id: int) -> list[Topic]:
    """
    Returns topics due for review today, ordered by ascending retrievability
    (most forgotten first). The FSRS threshold is < 0.90 retrievability.

    This is the primary query that drives every session start.
    """
    stmt = (
        select(Topic)
        .where(
            Topic.user_id == user_id,
            Topic.next_review_date <= date.today(),
            Topic.fsrs_retrievability < 0.90,
        )
        .order_by(Topic.fsrs_retrievability.asc())
    )
    result = await session.execute(stmt)
    topics = result.scalars().all()
    log.info("topics.due_fetched", count=len(topics), user_id=user_id)
    return list(topics)


async def get_all_topics(session: AsyncSession, user_id: int) -> list[Topic]:
    """Returns all topics for a user, ordered by programming language then name."""
    stmt = (
        select(Topic)
        .where(Topic.user_id == user_id)
        .order_by(Topic.programming_language, Topic.name)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_topic_by_id(session: AsyncSession, topic_id: int) -> Topic | None:
    """Returns a topic by primary key, or None if not found."""
    return await session.get(Topic, topic_id)


async def create_topic(
    session: AsyncSession,
    user_id: int,
    name: str,
    programming_language: str,
    description: str | None = None,
) -> Topic:
    """
    Creates a new study topic with default FSRS state (unreviewed).
    next_review_date is set to today so the first challenge is generated
    on the next Celery run or session start.
    """
    topic = Topic(
        user_id=user_id,
        name=name,
        programming_language=programming_language,
        description=description,
        next_review_date=date.today(),
    )
    session.add(topic)
    await session.flush()
    log.info("topic.created", topic_id=topic.id, name=name, lang=programming_language)
    return topic


async def update_topic_fsrs(
    session: AsyncSession,
    topic_id: int,
    *,
    stability: float,
    difficulty: float,
    retrievability: float,
    next_review_date: date,
) -> Topic:
    """
    Persists FSRS state after a review session.

    The caller (the review submission handler) runs the fsrs-python algorithm
    and passes the computed values here. This function only writes to the DB.

    Args:
        stability:        New memory stability value in days (from FSRS output).
        difficulty:       New difficulty value on the 1–10 scale (from FSRS output).
        retrievability:   New recall probability 0.0–1.0 (from FSRS output).
        next_review_date: Computed date of the next review (from FSRS output).
    """
    result = await session.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalars().one()

    topic.fsrs_stability = stability
    topic.fsrs_difficulty = difficulty
    topic.fsrs_retrievability = retrievability
    topic.next_review_date = next_review_date
    topic.last_review_date = date.today()
    topic.total_reviews += 1

    session.add(topic)
    log.info(
        "topic.fsrs_updated",
        topic_id=topic_id,
        stability=round(stability, 3),
        retrievability=round(retrievability, 3),
        next_review=str(next_review_date),
    )
    return topic

# challenges

async def create_challenge(
    session: AsyncSession,
    topic_id: int,
    difficulty: int,
    vector_chunk_ids: list[str],
    graph_node_ids: list[str],
    title: str | None = None,
    generated_by: str = "qwen-32b-async",
) -> Challenge:
    """
    Records a newly generated challenge.

    The actual challenge content (problem statement, evaluation rubric) is
    NOT stored here — it lives in Qdrant/Neo4j and is reconstructed on demand
    using vector_chunk_ids and graph_node_ids.
    """
    challenge = Challenge(
        topic_id=topic_id,
        title=title,
        difficulty=difficulty,
        vector_chunk_ids=vector_chunk_ids,
        graph_node_ids=graph_node_ids,
        generated_by=generated_by,
    )
    session.add(challenge)
    await session.flush()
    log.info(
        "challenge.created",
        challenge_id=challenge.id,
        topic_id=topic_id,
        difficulty=difficulty,
    )
    return challenge


async def get_challenge_by_id(session: AsyncSession, challenge_id: int) -> Challenge | None:
    """Returns a challenge by primary key, or None if not found."""
    return await session.get(Challenge, challenge_id)


async def get_challenges_for_topic(
    session: AsyncSession, topic_id: int, limit: int = 20
) -> list[Challenge]:
    """Returns the most recent challenges for a topic, newest first."""
    stmt = (
        select(Challenge)
        .where(Challenge.topic_id == topic_id)
        .order_by(Challenge.generated_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

# review_logs

async def create_review_log(
    session: AsyncSession,
    challenge_id: int,
    topic_id: int,
    code_submitted: str,
    piston_output: dict[str, Any],
    compile_success: bool,
    llm_grade: int,
    execution_ms: int | None = None,
    llm_feedback: str | None = None,
) -> ReviewLog:
    """
    Appends an immutable record of a code submission.

    Called after:
      1. Piston has executed the code (provides piston_output, compile_success, execution_ms)
      2. The Synthesizer agent has evaluated the logic (provides llm_grade, llm_feedback)

    Grade scale:
      1 = Blackout  — failed to compile or completely wrong
      2 = Hard      — compiles but incorrect or severely inefficient
      3 = Good      — correct, working solution
      4 = Perfect   — correct, efficient, and idiomatic
    """
    if llm_grade not in (1, 2, 3, 4):
        raise ValueError(f"llm_grade must be 1–4, got {llm_grade}")

    log_entry = ReviewLog(
        challenge_id=challenge_id,
        topic_id=topic_id,
        code_submitted=code_submitted,
        piston_output=piston_output,
        compile_success=compile_success,
        execution_ms=execution_ms,
        llm_grade=llm_grade,
        llm_feedback=llm_feedback,
    )
    session.add(log_entry)
    await session.flush()
    log.info(
        "review_log.created",
        log_id=log_entry.id,
        challenge_id=challenge_id,
        grade=llm_grade,
        compiled=compile_success,
    )
    return log_entry


async def get_review_history(
    session: AsyncSession, topic_id: int, limit: int = 50
) -> list[ReviewLog]:
    """Returns the most recent review logs for a topic, newest first."""
    stmt = (
        select(ReviewLog)
        .where(ReviewLog.topic_id == topic_id)
        .order_by(ReviewLog.submitted_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

# generation_logs

async def create_generation_log(
    session: AsyncSession,
    challenge_id: int | None,
    vector_chunks: list[dict[str, Any]],
    graph_filter: dict[str, Any],
    prompt_tokens: int | None = None,
    model_used: str | None = None,
    latency_ms: int | None = None,
    reviewer_verdict: str | None = None,
) -> GenerationLog:
    """
    Records a retrieval and generation event for observability.

    Should be called after every Synthesizer agent invocation, both for
    real-time responses and async Celery jobs.

    The prompt_tokens field is compared against settings.PROMPT_TOKEN_WARN_THRESHOLD
    here — a warning is logged if the threshold is exceeded so the operator can
    tune top_k or chunk size before KV cache spill occurs.
    """
    from app.config import settings  # avoid circular import at module level

    if prompt_tokens is not None and prompt_tokens > settings.PROMPT_TOKEN_WARN_THRESHOLD:
        log.warning(
            "kv_cache.budget_warning",
            prompt_tokens=prompt_tokens,
            threshold=settings.PROMPT_TOKEN_WARN_THRESHOLD,
            challenge_id=challenge_id,
            message="Prompt approaching KV cache limit on RTX 4060. "
                    "Consider reducing top_k or chunk size.",
        )

    entry = GenerationLog(
        challenge_id=challenge_id,
        prompt_tokens=prompt_tokens,
        vector_chunks=vector_chunks,
        graph_filter=graph_filter,
        model_used=model_used,
        latency_ms=latency_ms,
        reviewer_verdict=reviewer_verdict,
    )
    session.add(entry)
    await session.flush()
    log.info(
        "generation_log.created",
        log_id=entry.id,
        challenge_id=challenge_id,
        chunks_retrieved=len(vector_chunks),
        prompt_tokens=prompt_tokens,
        latency_ms=latency_ms,
        verdict=reviewer_verdict,
    )
    return entry


async def get_generation_logs_for_challenge(
    session: AsyncSession, challenge_id: int
) -> list[GenerationLog]:
    """Returns all generation logs associated with a specific challenge."""
    stmt = (
        select(GenerationLog)
        .where(GenerationLog.challenge_id == challenge_id)
        .order_by(GenerationLog.created_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_high_token_generations(
    session: AsyncSession,
    threshold: int,
    limit: int = 20,
) -> list[GenerationLog]:
    """
    Returns the most token-heavy recent generations, for KV cache budget auditing.

    Use this query to proactively identify requests that are approaching
    the VRAM limit before they cause latency degradation.
    """
    stmt = (
        select(GenerationLog)
        .where(
            GenerationLog.prompt_tokens.is_not(None),
            GenerationLog.prompt_tokens > threshold,
        )
        .order_by(GenerationLog.prompt_tokens.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def create_challenge(
    session: AsyncSession,
    *,
    user_id: int,
    topic_name: str,
    programming_language: str,
    difficulty: str,
    title: str,
    description: str,
    requirements: list,
    starter_code: str,
    expected_behavior: str,
    source_ids: list,
    knowledge_refs: list,
) -> Challenge:
    challenge = Challenge(
        user_id=user_id,
        topic_name=topic_name,
        programming_language=programming_language,
        difficulty=difficulty,
        title=title,
        description=description,
        requirements=requirements,
        starter_code=starter_code,
        expected_behavior=expected_behavior,
        source_ids=source_ids,
        knowledge_refs=knowledge_refs,
        status="pending",
    )
    session.add(challenge)
    await session.commit()
    await session.refresh(challenge)
    return challenge
 
 
async def get_challenge(session: AsyncSession, challenge_id: int) -> Challenge | None:
    result = await session.execute(
        select(Challenge).where(Challenge.id == challenge_id)
    )
    return result.scalars().first()
 
 
async def get_pending_challenges(session: AsyncSession, user_id: int) -> list[Challenge]:
    result = await session.execute(
        select(Challenge)
        .where(Challenge.user_id == user_id)
        .where(Challenge.status == "pending")
        .order_by(Challenge.created_at.desc())
    )
    return result.scalars().all()
 
 
async def mark_challenge_submitted(
    session: AsyncSession,
    challenge_id: int,
) -> Challenge | None:
    challenge = await get_challenge(session, challenge_id)
    if challenge:
        challenge.status = "submitted"
        session.add(challenge)
        await session.commit()
        await session.refresh(challenge)
    return challenge

async def get_or_create_topic(
    session: AsyncSession,
    *,
    user_id: int,
    name: str,
    programming_language: str,
) -> "Topic":
    from sqlmodel import select
    result = await session.execute(
        select(Topic)
        .where(Topic.user_id == user_id)
        .where(Topic.name == name)
        .where(Topic.programming_language == programming_language)
    )
    topic = result.scalars().first()
    if topic:
        return topic
    topic = Topic(
        user_id=user_id,
        name=name,
        programming_language=programming_language,
    )
    session.add(topic)
    await session.commit()
    await session.refresh(topic)
    return topic
 
 
async def update_topic_fsrs(
    session: AsyncSession,
    *,
    topic_id: int,
    new_stability: float,
    new_difficulty: float,
    new_retrievability: float,
    next_review_date: "date",
) -> None:
    from sqlmodel import select
    from datetime import date as date_type
    result = await session.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalars().first()
    if not topic:
        return
    topic.fsrs_stability = new_stability
    topic.fsrs_difficulty = new_difficulty
    topic.fsrs_retrievability = new_retrievability
    topic.next_review_date = next_review_date
    topic.last_review_date = date_type.today()
    topic.total_reviews = (topic.total_reviews or 0) + 1
    session.add(topic)
    await session.commit()
 
 
async def create_review_log(
    session: AsyncSession,
    *,
    challenge_id: int,
    topic_id: int,
    code_submitted: str,
    piston_output: dict,
    compile_success: bool,
    execution_ms: int | None,
    llm_grade: int,
    llm_feedback: str | None,
) -> "ReviewLog":
    log = ReviewLog(
        challenge_id=challenge_id,
        topic_id=topic_id,
        code_submitted=code_submitted,
        piston_output=piston_output,
        compile_success=compile_success,
        execution_ms=execution_ms,
        llm_grade=llm_grade,
        llm_feedback=llm_feedback,
    )
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log
 
 
async def get_recent_review_logs(
    session: AsyncSession,
    limit: int = 10,
) -> list["ReviewLog"]:
    from sqlmodel import select
    result = await session.execute(
        select(ReviewLog)
        .order_by(ReviewLog.submitted_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
 
async def update_review_log_notes(session: AsyncSession, log_id: int, notes: str) -> None:
    from sqlmodel import select
    result = await session.execute(select(ReviewLog).where(ReviewLog.id == log_id))
    log = result.scalars().first()
    if log:
        log.notes = notes
        session.add(log)
        await session.commit()

async def get_review_history_with_topics(session: AsyncSession) -> list[dict]:
    from sqlmodel import select
    from app.db.postgres.models import Challenge
    result = await session.execute(
        select(ReviewLog, Challenge)
        .join(Challenge, ReviewLog.challenge_id == Challenge.id, isouter=True)
        .order_by(ReviewLog.submitted_at.desc())
        .limit(500)
    )
    rows = result.all()
    return [
        {
            "log_id": log.id,
            "topic_name": challenge.topic_name if challenge else "Unknown",
            "programming_language": challenge.programming_language if challenge else "",
            "date": str(log.submitted_at)[:10],
            "grade": log.llm_grade,
            "notes": getattr(log, "notes", "") or "",
        }
        for log, challenge in rows
    ]