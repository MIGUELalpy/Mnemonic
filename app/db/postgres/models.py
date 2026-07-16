"""
SQLModel table definitions for every PostgreSQL table.

Relationships (back_populates) are intentionally omitted. All queries in
crud.py use explicit select() statements, so ORM relationship accessors are
not needed. Removing them also eliminates all SQLAlchemy forward-reference
annotation issues entirely.
"""

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.utcnow()


def _today() -> date:
    return datetime.now(timezone.utc).date()


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    display_name: str = Field(max_length=100)
    current_streak: int = Field(default=0, ge=0)
    longest_streak: int = Field(default=0, ge=0)
    daily_goal_minutes: int = Field(default=30, ge=5, le=480)
    ui_language: str = Field(default="EN", max_length=5)
    created_at: datetime = Field(default_factory=_now_utc)


class Topic(SQLModel, table=True):
    __tablename__ = "topics"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=200)
    programming_language: str = Field(max_length=20)
    description: Optional[str] = Field(default=None)

    # FSRS state
    fsrs_stability: float = Field(default=1.0)
    fsrs_difficulty: float = Field(default=5.0, ge=1.0, le=10.0)
    fsrs_retrievability: float = Field(default=1.0, ge=0.0, le=1.0)

    # Scheduling
    next_review_date: Optional[date] = Field(default=None)
    last_review_date: Optional[date] = Field(default=None)
    total_reviews: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_now_utc)


class Challenge(SQLModel, table=True):
    __tablename__ = "challenges"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    topic_name: str = Field(max_length=200)
    programming_language: str = Field(max_length=20)
    difficulty: str = Field(max_length=20, default="intermediate")
    title: str = Field(max_length=300)
    description: str
    requirements: List[Any] = Field(default_factory=list, sa_column=Column(JSONB))
    starter_code: str
    expected_behavior: str
    source_ids: List[Any] = Field(default_factory=list, sa_column=Column(JSONB))
    knowledge_refs: List[Any] = Field(default_factory=list, sa_column=Column(JSONB))
    status: str = Field(default="pending", max_length=20)
    created_at: datetime = Field(default_factory=_now_utc)


class ReviewLog(SQLModel, table=True):
    __tablename__ = "review_logs"

    notes: Optional[str] = Field(default=None)
    id: Optional[int] = Field(default=None, primary_key=True)
    challenge_id: Optional[int] = Field(default=None, foreign_key="challenges.id", index=True)
    topic_id: int = Field(foreign_key="topics.id", index=True)
    submitted_at: datetime = Field(default_factory=_now_utc)
    code_submitted: str
    piston_output: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    compile_success: bool
    execution_ms: Optional[int] = Field(default=None, ge=0)
    # Grade: 1=Blackout 2=Hard 3=Good 4=Perfect
    llm_grade: int = Field(ge=1, le=4)
    llm_feedback: Optional[str] = Field(default=None)


class GenerationLog(SQLModel, table=True):
    __tablename__ = "generation_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    challenge_id: Optional[int] = Field(default=None, foreign_key="challenges.id", index=True)
    created_at: datetime = Field(default_factory=_now_utc)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    vector_chunks: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB))
    graph_filter: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    model_used: Optional[str] = Field(default=None)
    latency_ms: Optional[int] = Field(default=None, ge=0)
    reviewer_verdict: Optional[str] = Field(default=None)