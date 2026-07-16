"""
Code submission endpoint — closes the learning loop.

Flow for POST /submit/{challenge_id}:
  1. Load the challenge from PostgreSQL
  2. Execute the submitted code via Piston (sandboxed)
  3. Run the agent pipeline in code_submission mode
     (Synthesizer grades 1-4 using knowledge base as rubric)
  4. Update FSRS scheduling for the topic in PostgreSQL
  5. Log the review in review_logs table
  6. Mark the challenge as submitted
  7. Return grade, feedback, execution result, and next review date

Endpoints:
  POST /submit/{challenge_id}   — submit code for a challenge
  GET  /submit/history          — recent submission history
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.graph import agent_graph
from app.core.auth import verify_token
from app.db.postgres.crud import (
    create_review_log,
    get_challenge,
    get_or_create_topic,
    get_recent_review_logs,
    get_review_history_with_topics,
    get_user,
    mark_challenge_submitted,
    update_topic_fsrs,
)
from app.db.postgres.session import get_session
from app.services.fsrs import schedule
from app.services.piston import PistonResult, check_piston_health, execute

router = APIRouter(prefix="/submit", tags=["submission"])

#  Request / Response schemas 

class SubmitRequest(BaseModel):
    code: str = Field(
        description="Your code solution.",
        min_length=1,
        max_length=10000,
    )
    language_version: str | None = Field(
        default=None,
        description="Optional language version hint (e.g. '3.11'). "
                    "If omitted, the latest installed version is used.",
    )

class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    compile_stderr: str
    exit_code: int
    compile_success: bool
    execution_ms: int
    timed_out: bool

class SubmissionResponse(BaseModel):
    challenge_id: int
    grade: int                          # 1-4
    grade_rationale: str
    feedback: str
    citations: list[dict]
    execution: ExecutionResult
    fsrs: dict                          # next_review_date, interval_days, etc.
    elapsed_ms: int

#  Endpoints 

@router.post("/{challenge_id}", dependencies=[Depends(verify_token)])
async def submit_challenge(
    challenge_id: int,
    request: SubmitRequest,
    session: AsyncSession = Depends(get_session),
) -> SubmissionResponse:
    """
    Submits code for a challenge, executes it, grades it, and updates FSRS.

    The grade (1-4) determines how many days until this topic appears again:
      1 (Blackout) → 1 day
      2 (Hard)     → 2-4 days
      3 (Good)     → 1-2 weeks
      4 (Perfect)  → several weeks

    The secondary node must be running for knowledge-base-grounded feedback.
    Piston must be running for code execution.
    """
    t_start = time.time()

    #  Load challenge 
    challenge = await get_challenge(session, challenge_id)
    if not challenge:
        raise HTTPException(status_code=404, detail=f"Challenge {challenge_id} not found.")

    user = await get_user(session)
    if not user:
        raise HTTPException(status_code=400, detail="No user found.")

    #  Step 1: Execute code via Piston 
    piston_result: PistonResult = await execute(
        code=request.code,
        programming_language=challenge.programming_language,
    )

    #  Step 2: Grade via agent pipeline 
    # Build a rich query that includes the challenge context and execution result
    submission_query = _build_submission_query(
        code=request.code,
        challenge_title=challenge.title,
        challenge_description=challenge.description,
        requirements=challenge.requirements,
        expected_behavior=challenge.expected_behavior,
        piston_result=piston_result,
        programming_language=challenge.programming_language,
    )

    agent_state = {
        "query": submission_query,
        "query_type": "code_submission",
        "programming_language": challenge.programming_language,
        "code_submitted": request.code,
        "piston_output": {
            "stdout": piston_result.stdout,
            "stderr": piston_result.stderr,
            "exit_code": piston_result.exit_code,
            "compile_success": piston_result.compile_success,
        },
        "compile_success": piston_result.compile_success,
        "execution_ms": piston_result.execution_ms,
    }

    try:
        final_state = await agent_graph.ainvoke(agent_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading pipeline failed: {str(e)}")

    # Extract grade — default to 2 if grading failed
    grade = final_state.get("grade")
    if grade is None or not isinstance(grade, int) or grade not in range(1, 11):
        # Fallback: derive grade from execution result
        grade = _infer_grade_from_execution(piston_result)

    grade_rationale = final_state.get("grade_rationale") or ""
    feedback = final_state.get("final_response") or final_state.get("response") or ""
    citations = final_state.get("citations") or []

    #  Step 3: Update FSRS 
    topic = await get_or_create_topic(
        session,
        user_id=user.id,
        name=challenge.topic_name,
        programming_language=challenge.programming_language,
    )

    fsrs_result = schedule(
        current_stability=topic.fsrs_stability,
        current_difficulty=topic.fsrs_difficulty,
        grade=grade,
        total_reviews=topic.total_reviews,
    )

    await update_topic_fsrs(
        session,
        topic_id=topic.id,
        new_stability=fsrs_result.new_stability,
        new_difficulty=fsrs_result.new_difficulty,
        new_retrievability=fsrs_result.new_retrievability,
        next_review_date=fsrs_result.next_review_date,
    )

    #  Step 4: Log the review 
    await create_review_log(
        session,
        challenge_id=challenge_id,
        topic_id=topic.id,
        code_submitted=request.code,
        piston_output={
            "stdout": piston_result.stdout,
            "stderr": piston_result.stderr,
            "exit_code": piston_result.exit_code,
            "compile_success": piston_result.compile_success,
            "execution_ms": piston_result.execution_ms,
        },
        compile_success=piston_result.compile_success,
        execution_ms=piston_result.execution_ms,
        llm_grade=grade,
        llm_feedback=feedback,
    )

    # Step 5: Mark challenge submitted 
    await mark_challenge_submitted(session, challenge_id)

    elapsed_ms = int((time.time() - t_start) * 1000)

    return SubmissionResponse(
        challenge_id=challenge_id,
        grade=grade,
        grade_rationale=grade_rationale,
        feedback=feedback,
        citations=[
            {
                "source_title": c.get("source_title", ""),
                "source_page": c.get("source_page", 1),
                "exact_source_quote": c.get("exact_source_quote", ""),
            }
            for c in citations
        ],
        execution=ExecutionResult(
            stdout=piston_result.stdout[:2000],
            stderr=piston_result.stderr[:1000],
            compile_stderr=piston_result.compile_stderr[:1000],
            exit_code=piston_result.exit_code,
            compile_success=piston_result.compile_success,
            execution_ms=piston_result.execution_ms,
            timed_out=piston_result.timed_out,
        ),
        fsrs={
            "next_review_date": str(fsrs_result.next_review_date),
            "interval_days": fsrs_result.interval_days,
            "new_stability": fsrs_result.new_stability,
            "new_difficulty": fsrs_result.new_difficulty,
        },
        elapsed_ms=elapsed_ms,
    )

@router.get("/topic-history", dependencies=[Depends(verify_token)])
async def get_topic_grade_history(
    session: AsyncSession = Depends(get_session),
) -> dict:
    entries = await get_review_history_with_topics(session)
    by_topic: dict[str, list] = {}
    for e in entries:
        key = f"{e['topic_name']}|{e['programming_language']}"
        if key not in by_topic:
            by_topic[key] = []
        by_topic[key].append({
            "log_id": e["log_id"],
            "date": e["date"],
            "grade": e["grade"],
            "notes": e.get("notes", ""),
        })
    return {"topics": by_topic}

@router.get("/history", dependencies=[Depends(verify_token)])
async def submission_history(
    session: AsyncSession = Depends(get_session),
    limit: int = 10,
) -> dict:
    """Returns the most recent submission logs for the current user."""
    logs = await get_recent_review_logs(session, limit=limit)
    return {
        "count": len(logs),
        "logs": [
            {
                "id": log.id,
                "challenge_id": log.challenge_id,
                "submitted_at": str(log.submitted_at),
                "compile_success": log.compile_success,
                "llm_grade": log.llm_grade,
                "execution_ms": log.execution_ms,
            }
            for log in logs
        ],
    }

#  Helpers 

def _build_submission_query(
    code: str,
    challenge_title: str,
    challenge_description: str,
    requirements: list,
    expected_behavior: str,
    piston_result: PistonResult,
    programming_language: str,
) -> str:
    """Builds the rich query string sent to the agent pipeline for grading."""
    execution_summary = []
    if not piston_result.compile_success:
        execution_summary.append(f"COMPILE ERROR:\n{piston_result.compile_stderr[:500]}")
    elif piston_result.timed_out:
        execution_summary.append("EXECUTION TIMED OUT after 10 seconds.")
    elif piston_result.exit_code != 0:
        execution_summary.append(
            f"RUNTIME ERROR (exit code {piston_result.exit_code}):\n"
            f"{piston_result.stderr[:500]}"
        )
    else:
        execution_summary.append(
            f"EXECUTED SUCCESSFULLY\n"
            f"stdout: {piston_result.stdout[:300] or '(no output)'}\n"
            f"stderr: {piston_result.stderr[:200] or '(none)'}"
        )

    req_text = "\n".join(f"- {r}" for r in (requirements or []))

    return (
        f"Challenge: {challenge_title}\n"
        f"Language: {programming_language}\n\n"
        f"Description: {challenge_description}\n\n"
        f"Requirements:\n{req_text}\n\n"
        f"Expected behavior: {expected_behavior}\n\n"
        f"Submitted code:\n```{programming_language.lower()}\n{code}\n```\n\n"
        f"Execution result:\n{chr(10).join(execution_summary)}"
    )

def _infer_grade_from_execution(result: PistonResult) -> int:
    """Fallback grade (1-10 scale) when LLM grading fails — derived from execution result."""
    if not result.compile_success:
        return 1
    if result.timed_out:
        return 2
    if result.exit_code != 0:
        return 4
    return 7  # compiled and ran successfully → at least a 7