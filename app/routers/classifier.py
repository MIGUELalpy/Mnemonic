"""
FastAPI router for the AI classification workflow.

Endpoints:
  POST /admin/classify/run      — run classifier on all PDFs in inbox/
  GET  /admin/classify/pending  — view current pending_review.json
  POST /admin/classify/confirm  — move confirmed files and trigger ingestion
  PUT  /admin/classify/edit     — edit a single proposal before confirming
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import verify_token
from app.ingestion.classifier import classify_inbox, confirm_classifications
from app.ingestion.pipeline import ingest_directory

router = APIRouter(prefix="/admin/classify", tags=["classification"])

DATA_ROOT = Path("/app/data")
INTAKE_DIR = DATA_ROOT / "intake"
INBOX_DIR = INTAKE_DIR / "inbox"
PROCESSED_DIR = DATA_ROOT / "processed"
FAILED_DIR = DATA_ROOT / "failed"
PENDING_REVIEW_PATH = INTAKE_DIR / "pending_review.json"

#  Request schemas 

class ConfirmRequest(BaseModel):
    filenames: list[str] | None = Field(
        default=None,
        description="Specific filenames to confirm. If null, confirms ALL pending proposals.",
        examples=[["bishop_prml.pdf", "attention_is_all_you_need.pdf"]],
    )
    run_ingestion: bool = Field(
        default=True,
        description="If True, immediately runs the ingestion pipeline after moving files.",
    )

class EditProposalRequest(BaseModel):
    filename: str = Field(description="The filename to edit.")
    proposed_tier: str | None = None
    proposed_domain: str | None = None
    proposed_programming_language: str | None = None
    proposed_doc_language: str | None = None
    proposed_document_type: str | None = None
    status: str | None = Field(
        default=None,
        description="Set to 'skipped' to exclude this file from confirmation.",
    )

# Endpoints

@router.post("/run", dependencies=[Depends(verify_token)])
async def run_classifier() -> dict:
    """
    Runs the AI classifier on all PDFs in data/intake/inbox/.

    The classifier (Qwen2.5-Coder-7B via Ollama) reads the first 3 pages
    of each PDF and proposes a classification. Results are written to
    data/intake/pending_review.json for human review before any files are moved.

    Re-running is safe — already-classified files are skipped unless
    their status is still "pending".
    """
    if not INBOX_DIR.exists():
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "message": "Inbox directory created but is empty.",
            "inbox_path": str(INBOX_DIR),
            "hint": "Place your PDF files in data/intake/inbox/ and run again.",
        }

    pdf_count = len(list(INBOX_DIR.glob("*.pdf")))
    if pdf_count == 0:
        return {
            "message": "No PDFs found in inbox.",
            "inbox_path": str(INBOX_DIR),
        }

    result = await classify_inbox(
        inbox_dir=INBOX_DIR,
        pending_review_path=PENDING_REVIEW_PATH,
    )

    return {
        "total_pdfs": result.total,
        "classified": result.classified,
        "failed": result.failed,
        "elapsed_s": result.elapsed_s,
        "pending_review_path": str(PENDING_REVIEW_PATH),
        "next_step": "Review data/intake/pending_review.json, edit any incorrect "
                     "proposals, then call POST /admin/classify/confirm.",
        "proposals": [
            {
                "filename": p.filename,
                "proposed_tier": p.proposed_tier,
                "proposed_domain": p.proposed_domain,
                "proposed_programming_language": p.proposed_programming_language,
                "suggested_path": p.suggested_path,
                "confidence": p.confidence,
                "reasoning": p.reasoning,
            }
            for p in result.proposals
        ],
    }

@router.get("/pending", dependencies=[Depends(verify_token)])
async def get_pending_proposals() -> dict:
    """
    Returns the current contents of pending_review.json.
    This is your review interface — check this after running the classifier.
    """
    if not PENDING_REVIEW_PATH.exists():
        return {
            "message": "No pending review file found. Run POST /admin/classify/run first.",
            "proposals": [],
        }

    proposals = json.loads(PENDING_REVIEW_PATH.read_text())
    pending = [p for p in proposals if p.get("status") == "pending"]
    confirmed = [p for p in proposals if p.get("status") == "confirmed"]
    skipped = [p for p in proposals if p.get("status") == "skipped"]

    return {
        "summary": {
            "pending": len(pending),
            "confirmed": len(confirmed),
            "skipped": len(skipped),
            "total": len(proposals),
        },
        "proposals": proposals,
    }

@router.put("/edit", dependencies=[Depends(verify_token)])
async def edit_proposal(request: EditProposalRequest) -> dict:
    """
    Edits a single proposal in pending_review.json before confirming.

    Use this to correct the model's classification on individual files
    without editing the JSON file directly.
    The suggested_path is automatically recomputed from the new values.
    """
    if not PENDING_REVIEW_PATH.exists():
        raise HTTPException(status_code=404, detail="No pending_review.json found.")

    proposals = json.loads(PENDING_REVIEW_PATH.read_text())
    found = False

    for p in proposals:
        if p["filename"] != request.filename:
            continue
        found = True

        if request.proposed_tier is not None:
            p["proposed_tier"] = request.proposed_tier
        if request.proposed_domain is not None:
            p["proposed_domain"] = request.proposed_domain
        if request.proposed_programming_language is not None:
            p["proposed_programming_language"] = request.proposed_programming_language
        if request.proposed_doc_language is not None:
            p["proposed_doc_language"] = request.proposed_doc_language
        if request.proposed_document_type is not None:
            p["proposed_document_type"] = request.proposed_document_type
        if request.status is not None:
            p["status"] = request.status

        # Recompute suggested_path from updated values
        tier = p["proposed_tier"]
        domain = p["proposed_domain"]
        prog_lang = p.get("proposed_programming_language")
        filename = p["filename"]

        if tier == "theory":
            p["suggested_path"] = f"theory/{domain}/{filename}"
        else:
            lang_dir = {"C++": "cpp", "Python": "python", "C#": "csharp"}.get(
                prog_lang or "", "python"
            )
            p["suggested_path"] = f"applied/{lang_dir}/{domain}/{filename}"

        break

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"No proposal found for filename: {request.filename}",
        )

    PENDING_REVIEW_PATH.write_text(
        json.dumps(proposals, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {"message": f"Proposal for {request.filename} updated successfully."}

@router.post("/confirm", dependencies=[Depends(verify_token)])
async def confirm_and_ingest(request: ConfirmRequest) -> dict:
    """
    Moves confirmed PDFs from inbox/ to their classified paths and
    optionally triggers the full ingestion pipeline.

    Only files with status="pending" in pending_review.json are moved.
    Set a file's status to "skipped" via PUT /edit to exclude it.

    The ingestion pipeline automatically skips any PDFs already in Qdrant,
    so re-running confirm is always safe.
    """
    # Step 1: Move files
    move_result = await confirm_classifications(
        inbox_dir=INBOX_DIR,
        intake_root=INTAKE_DIR,
        pending_review_path=PENDING_REVIEW_PATH,
        filenames=request.filenames,
    )

    if "error" in move_result:
        raise HTTPException(status_code=400, detail=move_result["error"])

    response = {
        "files_moved": move_result["moved"],
        "files_skipped": move_result["skipped_already_processed"],
        "move_errors": move_result["errors"],
        "moved_files": move_result["moved_files"],
        "error_details": move_result["error_details"],
    }

    # Step 2: Optionally run ingestion on newly placed files
    if request.run_ingestion and move_result["moved"] > 0:
        ingest_results = await ingest_directory(
            intake_root=INTAKE_DIR,
            processed_dir=PROCESSED_DIR,
            failed_dir=FAILED_DIR,
            force_reingest=False,
        )

        newly_ingested = sum(
            1 for r in ingest_results if r.success and r.chunks_stored != -1
        )
        skipped_ingestion = sum(1 for r in ingest_results if r.chunks_stored == -1)
        failed_ingestion = sum(1 for r in ingest_results if not r.success)

        response["ingestion"] = {
            "newly_ingested": newly_ingested,
            "skipped_already_processed": skipped_ingestion,
            "failed": failed_ingestion,
            "total_chunks_stored": sum(
                r.chunks_stored for r in ingest_results
                if r.success and r.chunks_stored != -1
            ),
        }
    elif not request.run_ingestion:
        response["ingestion"] = "Skipped — run POST /admin/ingest/dir to ingest manually."

    return response