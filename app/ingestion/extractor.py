"""
PDF extraction using Marker (primary) or pdfplumber (fallback).

Marker is a visual model that converts technical PDFs to clean Markdown,
preserving code blocks, mathematical notation, and multi-column layouts.
Standard text extractors (pdfplumber) are used only as a fallback because
they corrupt code indentation and lose LaTeX math.

Output Markdown is cached in data/processed/ as {source_id}.md.
Re-extraction is skipped if the cache file already exists.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    _MARKER_AVAILABLE = True
except ImportError:
    _MARKER_AVAILABLE = False
    log.warning("marker.not_available", message="Marker not installed. Using pdfplumber fallback.")

class ExtractionResult:
    def __init__(self, source_id, source_title, pages, total_pages,
                 extraction_tool, success, error=None):
        self.source_id = source_id
        self.source_title = source_title
        self.pages = pages          # list of (page_number, markdown_text)
        self.total_pages = total_pages
        self.extraction_tool = extraction_tool
        self.success = success
        self.error = error

    def full_text(self) -> str:
        return "\n\n---\n\n".join(text for _, text in self.pages)

def make_source_id(pdf_path: Path) -> str:
    """Stable source_id from filename — join key between Qdrant and Neo4j."""
    raw = pdf_path.stem.strip().lower()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def extract_pdf(pdf_path: Path, processed_dir: Path, force_reextract: bool = False) -> ExtractionResult:
    """
    Extracts a PDF to Markdown. Returns cached result if available.
    """
    source_id = make_source_id(pdf_path)
    source_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
    cache_path = processed_dir / f"{source_id}.md"

    if cache_path.exists() and not force_reextract:
        log.info("extractor.cache_hit", source_id=source_id)
        cached_text = cache_path.read_text(encoding="utf-8")
        pages = _split_into_pages(cached_text)
        return ExtractionResult(source_id, source_title, pages, len(pages), "cache", True)

    log.info("extractor.starting", source_id=source_id, pdf=pdf_path.name)

    if _MARKER_AVAILABLE:
        result = _extract_with_marker(pdf_path, source_id, source_title)
    else:
        result = _extract_with_pdfplumber(pdf_path, source_id, source_title)

    if result.success:
        processed_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result.full_text(), encoding="utf-8")
        log.info("extractor.cached", source_id=source_id, pages=result.total_pages,
                 tool=result.extraction_tool)

    return result

def _extract_with_marker(pdf_path, source_id, source_title):
    try:
        models = load_all_models()
        full_text, _meta, _images = convert_single_pdf(str(pdf_path), models, max_pages=None, langs=None, batch_multiplier=1)
        pages = _split_into_pages(full_text)
        return ExtractionResult(source_id, source_title, pages, len(pages), "marker", True)
    except Exception as e:
        log.error("extractor.marker_failed", source_id=source_id, error=str(e))
        return _extract_with_pdfplumber(pdf_path, source_id, source_title)

def _extract_with_pdfplumber(pdf_path, source_id, source_title):
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
        return ExtractionResult(source_id, source_title, pages, len(pages), "pdfplumber", True)
    except Exception as e:
        log.error("extractor.pdfplumber_failed", source_id=source_id, error=str(e))
        return ExtractionResult(source_id, source_title, [], 0, "pdfplumber", False, str(e))

def _split_into_pages(text: str) -> list[tuple[int, str]]:
    page_break = re.compile(r"\n---\n|\f|\[PAGE_BREAK\]", re.IGNORECASE)
    raw_pages = page_break.split(text)
    pages = [(i + 1, p.strip()) for i, p in enumerate(raw_pages) if p.strip()]
    return pages or [(1, text.strip())]