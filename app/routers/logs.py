"""
Study log endpoints.

Endpoints:
  POST /logs/submission     — write submission result to JSONL log
  GET  /logs/recent         — fetch recent log entries
  POST /logs/sync           — check log status and return sync command
  PATCH /logs/note/{log_id} — save a custom note on a review log entry
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import verify_token
from app.db.postgres.session import get_session

router = APIRouter(prefix="/logs", tags=["logs"])

LOG_DIR = Path("/app/data/logs")
LOG_FILE = LOG_DIR / "study_log.jsonl"

class LogEntry(BaseModel):
    challenge_id: int
    topic_name: str
    programming_language: str
    difficulty: str
    title: str
    code_submitted: str
    grade: int
    grade_rationale: str
    feedback: str
    citations: list[dict]
    execution: dict
    fsrs: dict

@router.post("/submission", dependencies=[Depends(verify_token)])
async def log_submission(entry: LogEntry) -> dict:
    """Appends a submission result to the study log JSONL file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry.model_dump(),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"logged": True, "file": str(LOG_FILE)}

@router.get("/recent", dependencies=[Depends(verify_token)])
async def get_recent_logs(limit: int = 20) -> dict:
    """Returns the most recent submission log entries."""
    if not LOG_FILE.exists():
        return {"count": 0, "entries": []}
    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]
    recent = lines[-limit:]
    entries = [json.loads(l) for l in reversed(recent)]
    return {"count": len(entries), "entries": entries}

@router.post("/sync", dependencies=[Depends(verify_token)])
async def sync_logs_to_debian() -> dict:
    """
    Checks log file status and returns the sync command to run on the host.
    The actual rsync must run on the host since SSH keys are on the host machine.
    """
    exists = LOG_FILE.exists()
    size = LOG_FILE.stat().st_size if exists else 0
    line_count = 0
    if exists:
        with open(LOG_FILE, encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())

    return {
        "log_exists": exists,
        "entries": line_count,
        "size_bytes": size,
        "sync_command": "bash ~/Documentos/ProjetoAM/scripts/sync_logs.sh",
        "message": f"Log file has {line_count} entries ({size} bytes). "
                   f"Run the sync command in your terminal to push to Debian.",
    }

@router.patch("/note/{log_id}", dependencies=[Depends(verify_token)])
async def save_submission_note(
    log_id: int,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Saves a custom note on a review log entry in PostgreSQL."""
    from app.db.postgres.crud import update_review_log_notes
    notes = str(body.get("notes", "")).strip()
    await update_review_log_notes(session, log_id, notes)
    return {"saved": True, "log_id": log_id}