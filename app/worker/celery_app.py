"""
Celery application configuration and startup task.

The worker_ready signal fires once when the Celery worker finishes
booting. We use it to trigger challenge pre-generation for all topics
whose next_review_date is today or earlier.

This replaces the 3AM cron approach — challenges are generated whenever
you start the system, so they're ready by the time you sit down to study.
"""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_ready

from app.config import settings

celery_app = Celery(
    "mnemonic",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,        # 30 minute hard limit per task
    task_soft_time_limit=25 * 60,   # soft limit triggers SoftTimeLimitExceeded
    worker_prefetch_multiplier=1,   # one task at a time (we're single-worker)
    result_expires=3600,
)

# Register task modules
celery_app.autodiscover_tasks(["app.worker"])

@worker_ready.connect
def on_worker_ready(sender, **kwargs) -> None:
    """
    Fires once when the Celery worker finishes booting.
    Triggers challenge pre-generation for all due topics.
    """
    import structlog
    log = structlog.get_logger(__name__)
    log.info("worker.ready", message="Triggering startup challenge generation.")
    generate_due_challenges.delay()

#  Tasks 

@celery_app.task(name="mnemonic.generate_due_challenges", bind=True, max_retries=2)
def generate_due_challenges(self) -> dict:
    """
    Finds all topics with next_review_date <= today and pre-generates
    one challenge per topic. Runs at startup and can also be triggered manually.

    Returns a summary dict with counts.
    """
    import asyncio
    import structlog
    log = structlog.get_logger(__name__)

    try:
        result = asyncio.run(_generate_due_challenges_async())
        log.info("worker.generate_due_challenges.complete", **result)
        return result
    except Exception as e:
        log.error("worker.generate_due_challenges.failed", error=str(e))
        raise self.retry(exc=e, countdown=60)

async def _generate_due_challenges_async() -> dict:
    """
    Async implementation of challenge pre-generation.
    Runs inside asyncio.run() called from the Celery task.
    """
    import structlog
    from datetime import date

    from sqlmodel import select

    from app.db.postgres.models import Challenge, Topic, User
    from app.db.postgres.session import get_async_session_context
    from app.routers.challenge import _generate_challenge_for_topic

    log = structlog.get_logger(__name__)
    generated = 0
    skipped = 0
    failed = 0
    today = date.today()

    async with get_async_session_context() as session:
        # Get the single user
        user_result = await session.execute(select(User).limit(1))
        user = user_result.scalars().first()
        if not user:
            log.warning("worker.no_user_found")
            return {"generated": 0, "skipped": 0, "failed": 0}

        # Find all due topics (next_review_date is today or in the past, or never reviewed)
        topics_result = await session.execute(
            select(Topic).where(
                Topic.user_id == user.id,
            )
        )
        topics = topics_result.scalars().all()

        due_topics = [
            t for t in topics
            if t.next_review_date is None or t.next_review_date <= today
        ]

        log.info(
            "worker.due_topics_found",
            total_topics=len(topics),
            due_topics=len(due_topics),
        )

        for topic in due_topics:
            # Skip if there's already a pending challenge for this topic
            pending_result = await session.execute(
                select(Challenge).where(
                    Challenge.user_id == user.id,
                    Challenge.topic_name == topic.name,
                    Challenge.programming_language == topic.programming_language,
                    Challenge.status == "pending",
                )
            )
            existing = pending_result.scalars().first()
            if existing:
                log.info(
                    "worker.challenge_already_pending",
                    topic=topic.name,
                    challenge_id=existing.id,
                )
                skipped += 1
                continue

            # Generate challenge for this topic
            try:
                await _generate_challenge_for_topic(
                    session=session,
                    user_id=user.id,
                    topic_name=topic.name,
                    programming_language=topic.programming_language,
                    difficulty=_difficulty_from_fsrs(topic.fsrs_difficulty),
                )
                generated += 1
                log.info(
                    "worker.challenge_generated",
                    topic=topic.name,
                    language=topic.programming_language,
                )
            except Exception as e:
                failed += 1
                log.error(
                    "worker.challenge_generation_failed",
                    topic=topic.name,
                    error=str(e),
                )

    return {"generated": generated, "skipped": skipped, "failed": failed}

def _difficulty_from_fsrs(fsrs_difficulty: float) -> str:
    """
    Maps FSRS difficulty (1-10) to challenge difficulty (easy/intermediate/advanced).
    New topics start at 5.0 → intermediate.
    """
    if fsrs_difficulty <= 3.5:
        return "easy"
    elif fsrs_difficulty <= 6.5:
        return "intermediate"
    else:
        return "advanced"