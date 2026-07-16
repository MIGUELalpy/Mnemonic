"""
FastAPI router for the ingestion pipeline admin endpoints.

Endpoints:
  GET  /admin/ingest/status  — check secondary node health and PDF count
  POST /admin/ingest/file    — ingest a single PDF by path
  POST /admin/ingest/dir     — ingest all PDFs in the intake directory
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import verify_token
from app.ingestion.embedder import check_secondary_node_health
from app.ingestion.pipeline import IngestionResult, ingest_directory, ingest_pdf

router = APIRouter(prefix="/admin/ingest", tags=["ingestion"])

DATA_ROOT = Path("/app/data")
INTAKE_DIR = DATA_ROOT / "intake"
PROCESSED_DIR = DATA_ROOT / "processed"
FAILED_DIR = DATA_ROOT / "failed"

class IngestFileRequest(BaseModel):
    pdf_path: str = Field(
        description="Absolute path to the PDF inside the container, under /app/data/intake/.",
        examples=["/app/data/intake/theory/machine_learning/bishop_prml.pdf",
                  "/app/data/intake/applied/python/deep_learning/chollet_dl.pdf"],
    )
    programming_language: str | None = Field(
        default=None,
        description="Override inferred language: C++ | Python | C#",
    )
    document_type: str | None = Field(
        default=None,
        description="Override inferred type: Textbook | Scientific Paper | RFC",
    )
    doc_language: str | None = Field(
        default=None,
        description="Override inferred natural language: EN | PT | DE | ZH",
    )
    authors: list[str] | None = None
    force_reingest: bool = Field(
        default=False,
        description="Delete existing data for this document and re-ingest from scratch.",
    )

class IngestDirRequest(BaseModel):
    force_reingest: bool = Field(default=False)
    subdirectory: str | None = Field(
        default=None,
        description="Restrict ingestion to a subdirectory, e.g. 'theory/machine_learning'.",
    )

@router.get("/status", dependencies=[Depends(verify_token)])
async def ingestion_status() -> dict:
    """
    Checks secondary node health and counts available PDFs by tier.
    Run this before starting ingestion to confirm the system is ready.
    """
    is_healthy = await check_secondary_node_health()

    theory_pdfs = len(list((INTAKE_DIR / "theory").rglob("*.pdf"))) \
        if (INTAKE_DIR / "theory").exists() else 0
    applied_pdfs = len(list((INTAKE_DIR / "applied").rglob("*.pdf"))) \
        if (INTAKE_DIR / "applied").exists() else 0

    return {
        "secondary_node_healthy": is_healthy,
        "intake_dir": str(INTAKE_DIR),
        "pdfs": {
            "theory": theory_pdfs,
            "applied": applied_pdfs,
            "total": theory_pdfs + applied_pdfs,
        },
        "processed_count": len(list(PROCESSED_DIR.glob("*.md"))) \
            if PROCESSED_DIR.exists() else 0,
    }

@router.post("/file", dependencies=[Depends(verify_token)])
async def ingest_single_file(request: IngestFileRequest) -> dict:
    """Ingests a single PDF file into the knowledge base."""
    pdf_path = Path(request.pdf_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"File not found: {request.pdf_path}")

    if not str(pdf_path).startswith(str(INTAKE_DIR)):
        raise HTTPException(status_code=400,
                            detail=f"File must be under {INTAKE_DIR}")

    result: IngestionResult = await ingest_pdf(
        pdf_path=pdf_path,
        intake_root=INTAKE_DIR,
        processed_dir=PROCESSED_DIR,
        failed_dir=FAILED_DIR,
        force_reingest=request.force_reingest,
        doc_language=request.doc_language,
        programming_language=request.programming_language,
        document_type=request.document_type,
        authors=request.authors,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    if result.chunks_stored == -1:
        return {"status": "skipped", "reason": "Already ingested. Use force_reingest=true to reprocess.",
                "source_id": result.source_id, "source_title": result.source_title}

    return {
        "status": "ingested",
        "source_id": result.source_id,
        "source_title": result.source_title,
        "chunks_created": result.chunks_created,
        "chunks_embedded": result.chunks_embedded,
        "chunks_stored": result.chunks_stored,
        "extraction_tool": result.extraction_tool,
        "elapsed_s": result.elapsed_s,
    }

@router.post("/dir", dependencies=[Depends(verify_token)])
async def ingest_directory_endpoint(request: IngestDirRequest) -> dict:
    """
    Ingests all PDFs found under the intake directory (or a subdirectory).
    Already-ingested documents are automatically skipped.
    """
    target_dir = INTAKE_DIR
    if request.subdirectory:
        target_dir = INTAKE_DIR / request.subdirectory
        if not target_dir.exists():
            raise HTTPException(status_code=404,
                                detail=f"Subdirectory not found: {target_dir}")

    results = await ingest_directory(
        intake_root=target_dir,
        processed_dir=PROCESSED_DIR,
        failed_dir=FAILED_DIR,
        force_reingest=request.force_reingest,
    )

    successful = [r for r in results if r.success and r.chunks_stored != -1]
    skipped = [r for r in results if r.chunks_stored == -1]
    failed = [r for r in results if not r.success]

    return {
        "total_pdfs": len(results),
        "newly_ingested": len(successful),
        "skipped_already_processed": len(skipped),
        "failed": len(failed),
        "total_chunks_stored": sum(r.chunks_stored for r in successful),
        "failed_files": [{"path": r.pdf_path, "error": r.error} for r in failed],
        "results": [
            {"source_id": r.source_id, "title": r.source_title,
             "chunks_stored": r.chunks_stored, "elapsed_s": r.elapsed_s}
            for r in successful
        ],
    }