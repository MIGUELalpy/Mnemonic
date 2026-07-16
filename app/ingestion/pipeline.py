from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
import structlog
from app.db.neo4j.operations import link_concept_to_document, upsert_concept, upsert_document
from app.db.qdrant.operations import ChunkPayload, delete_chunks_by_source_id, document_exists, upsert_chunks_batch
from app.ingestion.chunker import SemanticChunker, extract_concept_tags
from app.ingestion.embedder import SecondaryNodeClient
from app.ingestion.extractor import extract_pdf, make_source_id

log = structlog.get_logger(__name__)

LANGUAGE_MAP = {
    "cpp": "C++", "c++": "C++",
    "python": "Python", "py": "Python",
    "csharp": "C#", "c#": "C#",
}

DOMAIN_MAP = {
    "machine_learning": "Machine Learning",
    "deep_learning": "Deep Learning",
    "computer_vision": "Computer Vision",
    "nlp": "NLP & Text Processing",
    "data_mining": "Data Mining & Analytics",
    "databases": "Database Management",
    "algorithms": "Algorithms & Data Structures",
    "systems": "Systems Programming",
    "architecture": "Software Architecture",
    "scientific_computing": "Scientific Computing",
    "general": "General",
}

NATURAL_LANGUAGE_HINTS = {
    "de_": "DE", "_de_": "DE", "_de.": "DE", "german": "DE", "deutsch": "DE",
    "zh_": "ZH", "_zh_": "ZH", "_zh.": "ZH", "chinese": "ZH",
    "pt_": "PT", "_pt_": "PT", "_pt.": "PT", "portuguese": "PT",
}

@dataclass
class IngestionResult:
    source_id: str
    source_title: str
    pdf_path: str
    success: bool
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    extraction_tool: str = ""
    elapsed_s: float = 0.0
    error: str | None = None

def infer_metadata_from_path(pdf_path: Path, intake_root: Path) -> dict:
    try:
        relative = pdf_path.relative_to(intake_root)
        parts = [p.lower() for p in relative.parts]
    except ValueError:
        parts = [pdf_path.name.lower()]

    filename_lower = pdf_path.stem.lower()
    doc_language = "EN"
    for hint, lang_code in NATURAL_LANGUAGE_HINTS.items():
        if hint in filename_lower:
            doc_language = lang_code
            break

    if parts[0] == "theory":
        domain_dir = parts[1] if len(parts) > 2 else "general"
        return {
            "tier": "theory",
            "programming_language": None,
            "document_type": "Scientific Paper",
            "domain": DOMAIN_MAP.get(domain_dir, "General"),
            "doc_language": doc_language,
        }

    if parts[0] == "applied":
        lang_dir = parts[1] if len(parts) > 1 else ""
        domain_dir = parts[2] if len(parts) > 3 else "general"
        return {
            "tier": "applied",
            "programming_language": LANGUAGE_MAP.get(lang_dir, "Python"),
            "document_type": "Textbook",
            "domain": DOMAIN_MAP.get(domain_dir, "General"),
            "doc_language": doc_language,
        }

    log.warning("pipeline.unexpected_path", path=str(pdf_path))
    return {
        "tier": "theory",
        "programming_language": None,
        "document_type": "Scientific Paper",
        "domain": "General",
        "doc_language": doc_language,
    }

async def ingest_pdf(
    pdf_path: Path,
    intake_root: Path,
    processed_dir: Path,
    failed_dir: Path,
    force_reingest: bool = False,
    doc_language: str | None = None,
    programming_language: str | None = None,
    document_type: str | None = None,
    authors: list[str] | None = None,
) -> IngestionResult:
    t_start = time.time()
    source_id = make_source_id(pdf_path)
    source_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
    inferred = infer_metadata_from_path(pdf_path, intake_root)
    final_prog_lang = programming_language or inferred["programming_language"]
    final_doc_type = document_type or inferred["document_type"]
    final_doc_lang = doc_language or inferred["doc_language"]
    final_domain = inferred["domain"]
    final_tier = inferred["tier"]

    log.info("pipeline.starting", source_id=source_id, title=source_title,
             tier=final_tier, prog_lang=final_prog_lang or "language-agnostic",
             domain=final_domain, doc_type=final_doc_type, doc_lang=final_doc_lang)

    if force_reingest:
        await delete_chunks_by_source_id(source_id)
        log.info("pipeline.existing_data_cleared", source_id=source_id)
    else:
        if await document_exists(source_id):
            log.info("pipeline.already_ingested", source_id=source_id, title=source_title)
            return IngestionResult(source_id=source_id, source_title=source_title,
                                   pdf_path=str(pdf_path), success=True, chunks_stored=-1,
                                   extraction_tool="skipped")

    extraction = extract_pdf(pdf_path, processed_dir, force_reextract=force_reingest)
    if not extraction.success or not extraction.pages:
        failed_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.rename(failed_dir / pdf_path.name)
        log.error("pipeline.extraction_failed", source_id=source_id, error=extraction.error)
        return IngestionResult(source_id=source_id, source_title=source_title,
                               pdf_path=str(pdf_path), success=False,
                               error=extraction.error or "Extraction produced no pages")

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(extraction.pages, source_id)
    if not chunks:
        return IngestionResult(source_id=source_id, source_title=source_title,
            pdf_path=str(pdf_path), success=False,
            error="Chunking produced no chunks")

    async with SecondaryNodeClient() as client:
        embedded = await client.embed_batch(chunks)

    if not embedded:
        return IngestionResult(source_id=source_id, source_title=source_title,
        pdf_path=str(pdf_path), success=False,
        chunks_created=len(chunks),
        extraction_tool=extraction.extraction_tool,
        error="Embedding returned no results")

    qdrant_prog_lang = final_prog_lang or "LANG_AGNOSTIC"
    qdrant_points = []
    for chunk, vector in embedded:
        concept_tags = extract_concept_tags(chunk.text, qdrant_prog_lang)
        concept_tags.extend([final_domain, final_tier])
        payload = ChunkPayload(
            source_id=source_id, source_title=source_title,
            source_page=chunk.source_page, doc_language=final_doc_lang,
            programming_language=qdrant_prog_lang, document_type=final_doc_type,
            chunk_text=chunk.text, concept_tags=list(set(concept_tags)),
            chunk_index=chunk.chunk_index, token_count=chunk.token_count,
        )
        qdrant_points.append((vector, payload))

    stored_point_ids = await upsert_chunks_batch(qdrant_points)

    await upsert_document(source_id=source_id, source_title=source_title,
        document_type=final_doc_type, doc_language=final_doc_lang,
        programming_language=qdrant_prog_lang,
        total_pages=extraction.total_pages, authors=authors or [])

    all_tags: set[str] = {final_domain, final_tier}
    for chunk, _ in embedded:
        all_tags.update(extract_concept_tags(chunk.text, qdrant_prog_lang))

    for tag in all_tags:
        await upsert_concept(tag)
        await link_concept_to_document(tag, source_id)

    elapsed = round(time.time() - t_start, 1)
    log.info("pipeline.complete", source_id=source_id, title=source_title,
        chunks_stored=len(stored_point_ids), elapsed_s=elapsed)

    return IngestionResult(source_id=source_id, source_title=source_title,
        pdf_path=str(pdf_path), success=True,
        chunks_created=len(chunks), chunks_embedded=len(embedded),
        chunks_stored=len(stored_point_ids),
        extraction_tool=extraction.extraction_tool, elapsed_s=elapsed)

async def ingest_directory(
    intake_root: Path,
    processed_dir: Path,
    failed_dir: Path,
    force_reingest: bool = False,
) -> list[IngestionResult]:
    pdf_files = list(intake_root.rglob("*.pdf"))
    if not pdf_files:
        log.warning("pipeline.no_pdfs_found", intake_dir=str(intake_root))
        return []

    log.info("pipeline.directory_scan", total_pdfs=len(pdf_files))
    results = []
    for pdf_path in pdf_files:
        result = await ingest_pdf(pdf_path=pdf_path, intake_root=intake_root,
        processed_dir=processed_dir, failed_dir=failed_dir,
        force_reingest=force_reingest)
        results.append(result)

    successful = sum(1 for r in results if r.success and r.chunks_stored != -1)
    skipped = sum(1 for r in results if r.chunks_stored == -1)
    failed = sum(1 for r in results if not r.success)
    log.info("pipeline.directory_complete", total=len(results),
        newly_ingested=successful, skipped=skipped, failed=failed)
    return results