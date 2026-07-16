"""
AI-powered PDF classifier using Ollama + Qwen2.5-Coder-7B.

Workflow:
  1. Scan data/intake/inbox/ for all PDF files.
  2. For each PDF, extract the first 3 pages using pdfplumber (fast,
     no Marker needed — we only need enough text to classify, not full fidelity).
  3. Send the extracted text to Ollama with a structured classification prompt.
  4. Parse the JSON response into a ClassificationProposal.
  5. Write all proposals to data/intake/pending_review.json for human review.

After human review (edit the JSON file directly), the confirm endpoint:
  - Moves each PDF from inbox/ to its proposed path under theory/ or applied/
  - Triggers the ingestion pipeline for newly placed files

The classifier never moves files autonomously — all file operations require
explicit human confirmation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import structlog

from app.config import settings

log = structlog.get_logger(__name__)

PENDING_REVIEW_FILENAME = "pending_review.json"

#  Valid taxonomy values 

VALID_TIERS = {"theory", "applied"}

VALID_THEORY_DOMAINS = {
    "machine_learning", "deep_learning", "computer_vision",
    "nlp", "data_mining", "databases", "general",
}

VALID_APPLIED_DOMAINS = {
    "algorithms", "systems", "architecture", "machine_learning",
    "computer_vision", "nlp", "data_mining", "databases",
    "scientific_computing", "general",
}

VALID_LANGUAGES = {"C++", "Python", "C#"}
VALID_DOC_LANGUAGES = {"EN", "PT", "DE", "ZH"}
VALID_DOCUMENT_TYPES = {"Textbook", "Scientific Paper", "RFC"}


@dataclass
class ClassificationProposal:
    """
    A single PDF's proposed classification.
    Written to pending_review.json and edited by the user before confirming.
    """
    filename: str                        # original filename in inbox/
    proposed_tier: str                   # "theory" | "applied"
    proposed_domain: str                 # domain subdirectory name (snake_case)
    proposed_programming_language: str | None  # None for theory tier
    proposed_doc_language: str           # EN | PT | DE | ZH
    proposed_document_type: str          # Textbook | Scientific Paper | RFC
    suggested_path: str                  # relative path under intake/ after move
    confidence: float                    # 0.0–1.0
    reasoning: str                       # model's explanation
    status: str = "pending"              # "pending" | "confirmed" | "skipped"

@dataclass
class ClassificationResult:
    """Result of classifying all PDFs in the inbox."""
    total: int
    classified: int
    failed: int
    proposals: list[ClassificationProposal]
    elapsed_s: float

#  PDF text extraction (lightweight, for classification only) 

def _extract_preview(pdf_path: Path, max_pages: int = 3) -> str:
    """
    Extracts a plain-text preview from the first N pages of a PDF.
    Uses pdfplumber (fast, no GPU needed).
    This text is sent to the classifier — it doesn't need perfect fidelity,
    just enough to identify the subject matter and language.
    """
    try:
        import pdfplumber
        pages_text: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())
        preview = "\n\n".join(pages_text)
        # Truncate to ~3000 chars — enough for classification, fits in context
        return preview[:3000]
    except Exception as e:
        log.error("classifier.preview_failed", pdf=pdf_path.name, error=str(e))
        # Fall back to just the filename — the model can still make a decent
        # guess from a descriptive filename like "bishop_pattern_recognition.pdf"
        return f"[Preview extraction failed. Filename: {pdf_path.stem}]"

#  Ollama prompt 

_SYSTEM_PROMPT = """You are a technical librarian classifying academic documents
for a software engineering learning platform.

The platform covers these domains:
  THEORY (language-agnostic academic papers and research):
    machine_learning, deep_learning, computer_vision, nlp,
    data_mining, databases, general

  APPLIED (language-specific implementation books):
    algorithms, systems, architecture, machine_learning,
    computer_vision, nlp, data_mining, databases, scientific_computing, general

Programming languages (for applied tier only): C++, Python, C#

Document types: Textbook, Scientific Paper, RFC

Natural languages: EN (English), PT (Portuguese), DE (German), ZH (Chinese)

Classification rules:
  - Use "theory" if the document is primarily mathematical, conceptual, or
    research-oriented with no strong tie to a specific programming language.
  - Use "applied" if the document teaches implementation in a specific language.
  - A book on "Machine Learning with Python" is applied/python/machine_learning.
  - A paper on "Attention Is All You Need" is theory/deep_learning.
  - A C++ algorithms textbook is applied/cpp/algorithms.

Respond ONLY with a valid JSON object. No preamble, no explanation outside JSON."""

_USER_TEMPLATE = """Classify this document.

Filename: {filename}

Content preview (first few pages):
---
{preview}
---

Respond with ONLY this JSON structure:
{{
  "proposed_tier": "theory" or "applied",
  "proposed_domain": "snake_case domain name",
  "proposed_programming_language": "C++" or "Python" or "C#" or null,
  "proposed_doc_language": "EN" or "PT" or "DE" or "ZH",
  "proposed_document_type": "Textbook" or "Scientific Paper" or "RFC",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence explaining your classification."
}}"""

#  Ollama API call 

async def _classify_with_ollama(filename: str, preview: str) -> dict:
    """
    Sends a classification request to the Ollama API.
    Returns the parsed JSON response from the model.
    Raises on network errors or JSON parse failures.
    """
    prompt = _USER_TEMPLATE.format(filename=filename, preview=preview)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",           # Ollama JSON mode — forces valid JSON output
                "options": {
                    "temperature": 0.1,     # low temperature for deterministic classification
                    "num_predict": 300,     # classifications are short
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_content = data["message"]["content"]

    # Parse the JSON response — strip markdown fences if present
    clean = re.sub(r"```(?:json)?|```", "", raw_content).strip()
    return json.loads(clean)

#  Path builder 

def _build_suggested_path(
    filename: str,
    tier: str,
    domain: str,
    programming_language: str | None,
) -> str:
    """
    Builds the suggested relative path under data/intake/ after classification.

    Examples:
      theory/machine_learning/bishop_prml.pdf
      applied/python/deep_learning/chollet_dl.pdf
    """
    if tier == "theory":
        return f"theory/{domain}/{filename}"
    else:
        lang_dir = {
            "C++": "cpp",
            "Python": "python",
            "C#": "csharp",
        }.get(programming_language or "", "python")
        return f"applied/{lang_dir}/{domain}/{filename}"

def _validate_and_fix(raw: dict, filename: str) -> dict:
    """
    Validates the model's output against the taxonomy and fixes common errors.
    Falls back to safe defaults rather than raising.
    """
    tier = raw.get("proposed_tier", "theory")
    if tier not in VALID_TIERS:
        tier = "theory"

    domain = raw.get("proposed_domain", "general")
    valid_domains = VALID_THEORY_DOMAINS if tier == "theory" else VALID_APPLIED_DOMAINS
    if domain not in valid_domains:
        domain = "general"

    prog_lang = raw.get("proposed_programming_language")
    if tier == "theory":
        prog_lang = None        # theory is always language-agnostic
    elif prog_lang not in VALID_LANGUAGES:
        prog_lang = "Python"    # safe default for applied tier

    doc_lang = raw.get("proposed_doc_language", "EN")
    if doc_lang not in VALID_DOC_LANGUAGES:
        doc_lang = "EN"

    doc_type = raw.get("proposed_document_type", "Textbook")
    if doc_type not in VALID_DOCUMENT_TYPES:
        doc_type = "Textbook" if tier == "applied" else "Scientific Paper"

    confidence = float(raw.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "proposed_tier": tier,
        "proposed_domain": domain,
        "proposed_programming_language": prog_lang,
        "proposed_doc_language": doc_lang,
        "proposed_document_type": doc_type,
        "confidence": confidence,
        "reasoning": str(raw.get("reasoning", "No reasoning provided."))[:300],
    }

#  Main classifier 

async def classify_inbox(
    inbox_dir: Path,
    pending_review_path: Path,
) -> ClassificationResult:
    """
    Classifies all PDFs in the inbox directory and writes proposals to
    pending_review.json for human review.

    Already-proposed files (present in pending_review.json with any status)
    are skipped unless their status is "pending" — so re-running is safe.

    Args:
        inbox_dir:            Path to data/intake/inbox/
        pending_review_path:  Path to data/intake/pending_review.json

    Returns:
        ClassificationResult with counts and all proposals.
    """
    t_start = time.time()

    # Load existing proposals to avoid re-classifying already-processed files
    existing: dict[str, dict] = {}
    if pending_review_path.exists():
        try:
            existing_list = json.loads(pending_review_path.read_text())
            existing = {p["filename"]: p for p in existing_list}
            log.info("classifier.loaded_existing", count=len(existing))
        except Exception:
            existing = {}

    pdf_files = list(inbox_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("classifier.inbox_empty", inbox=str(inbox_dir))
        return ClassificationResult(0, 0, 0, [], 0.0)

    log.info("classifier.starting", total_pdfs=len(pdf_files))

    proposals: list[ClassificationProposal] = []
    classified = 0
    failed = 0

    for pdf_path in pdf_files:
        filename = pdf_path.name

        # Skip if already classified and not still pending
        if filename in existing and existing[filename].get("status") != "pending":
            log.info("classifier.skipping", filename=filename,
                     status=existing[filename].get("status"))
            # Restore from existing
            p = existing[filename]
            proposals.append(ClassificationProposal(**{
                k: p[k] for k in ClassificationProposal.__dataclass_fields__
            }))
            continue

        log.info("classifier.processing", filename=filename)

        # Extract text preview
        preview = _extract_preview(pdf_path)

        # Call Ollama
        try:
            raw_result = await _classify_with_ollama(filename, preview)
            validated = _validate_and_fix(raw_result, filename)

            suggested_path = _build_suggested_path(
                filename=filename,
                tier=validated["proposed_tier"],
                domain=validated["proposed_domain"],
                programming_language=validated["proposed_programming_language"],
            )

            proposal = ClassificationProposal(
                filename=filename,
                suggested_path=suggested_path,
                status="pending",
                **validated,
            )
            proposals.append(proposal)
            classified += 1

            log.info(
                "classifier.classified",
                filename=filename,
                tier=validated["proposed_tier"],
                domain=validated["proposed_domain"],
                confidence=validated["confidence"],
            )

        except Exception as e:
            log.error("classifier.failed", filename=filename, error=str(e))
            # Create a low-confidence fallback proposal so the user
            # can still review and manually correct it
            proposals.append(ClassificationProposal(
                filename=filename,
                proposed_tier="theory",
                proposed_domain="general",
                proposed_programming_language=None,
                proposed_doc_language="EN",
                proposed_document_type="Scientific Paper",
                suggested_path=f"theory/general/{filename}",
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)[:100]}. "
                           f"Please classify manually.",
                status="pending",
            ))
            failed += 1

    # Merge new proposals with any existing confirmed/skipped ones
    all_proposals = proposals
    elapsed = round(time.time() - t_start, 1)

    # Write to pending_review.json
    pending_review_path.parent.mkdir(parents=True, exist_ok=True)
    pending_review_path.write_text(
        json.dumps([asdict(p) for p in all_proposals], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info(
        "classifier.complete",
        total=len(pdf_files),
        classified=classified,
        failed=failed,
        pending_review=str(pending_review_path),
        elapsed_s=elapsed,
    )

    return ClassificationResult(
        total=len(pdf_files),
        classified=classified,
        failed=failed,
        proposals=all_proposals,
        elapsed_s=elapsed,
    )

async def confirm_classifications(
    inbox_dir: Path,
    intake_root: Path,
    pending_review_path: Path,
    filenames: list[str] | None = None,
) -> dict:
    """
    Moves confirmed PDFs from inbox/ to their classified paths under intake/.

    Args:
        inbox_dir:            Path to data/intake/inbox/
        intake_root:          Root of the intake directory (data/intake/)
        pending_review_path:  Path to pending_review.json
        filenames:            If provided, only confirm these specific files.
            If None, confirms ALL pending proposals.

    Returns:
        Dict with counts of moved, skipped, and failed files.
    """
    if not pending_review_path.exists():
        return {"error": "No pending_review.json found. Run classify first."}

    proposals_raw = json.loads(pending_review_path.read_text())
    moved = []
    skipped = []
    errors = []

    for p_dict in proposals_raw:
        filename = p_dict["filename"]

        # Filter to requested filenames if specified
        if filenames and filename not in filenames:
            continue

        if p_dict.get("status") != "pending":
            skipped.append(filename)
            continue

        source = inbox_dir / filename
        if not source.exists():
            log.warning("classifier.confirm_missing", filename=filename)
            errors.append({"filename": filename, "error": "File not found in inbox"})
            continue

        dest = intake_root / p_dict["suggested_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            source.rename(dest)
            p_dict["status"] = "confirmed"
            moved.append(filename)
            log.info("classifier.moved", filename=filename, dest=str(dest))
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
            log.error("classifier.move_failed", filename=filename, error=str(e))

    # Update pending_review.json with confirmed statuses
    pending_review_path.write_text(
        json.dumps(proposals_raw, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "moved": len(moved),
        "skipped_already_processed": len(skipped),
        "errors": len(errors),
        "moved_files": moved,
        "error_details": errors,
    }