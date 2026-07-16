"""
Semantic chunker — splits Markdown into vector-ready chunks.

Core invariant: a code block and its surrounding paragraphs always
stay in the same chunk, even if this slightly exceeds the token target.

Targets: 256–512 tokens per chunk (lower end preferred for KV cache).
Overlap:  50 tokens between adjacent chunks.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
import structlog

log = structlog.get_logger(__name__)

CHUNK_TARGET_TOKENS = 400
CHUNK_MAX_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 50

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)

@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_page: int
    token_count: int
    has_code: bool = False
    concept_tags: list[str] = field(default_factory=list)


class SemanticChunker:
    _CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _PARAGRAPH_SEP = re.compile(r"\n{2,}")

    def chunk_document(self, pages: list[tuple[int, str]], source_id: str) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        chunk_index = 0
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            page_chunks = self._chunk_page(page_text, page_num, chunk_index)
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)
        log.info("chunker.complete", source_id=source_id, total_chunks=len(all_chunks),
                 avg_tokens=round(sum(c.token_count for c in all_chunks) /
                                  max(1, len(all_chunks)), 1))
        return all_chunks

    def _chunk_page(self, text: str, page_num: int, start_index: int) -> list[Chunk]:
        sections = self._split_into_sections(text)
        chunks: list[Chunk] = []
        current_sections: list[str] = []
        current_tokens = 0
        pending_code_context = False
        overlap_text = ""

        for section in sections:
            section_tokens = _approx_tokens(section)
            is_code = section.strip().startswith("```")
            would_exceed = (current_tokens + section_tokens) > CHUNK_MAX_TOKENS
            safe_to_split = not pending_code_context and not is_code

            if would_exceed and safe_to_split and current_sections:
                chunk_text = overlap_text + "\n\n".join(current_sections)
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    chunk_index=start_index + len(chunks),
                    source_page=page_num,
                    token_count=_approx_tokens(chunk_text),
                    has_code=any("```" in s for s in current_sections),
                ))
                overlap_text = self._build_overlap(current_sections)
                current_sections = []
                current_tokens = _approx_tokens(overlap_text)

            current_sections.append(section)
            current_tokens += section_tokens

            if is_code:
                pending_code_context = True
            elif pending_code_context:
                pending_code_context = False

        if current_sections:
            chunk_text = overlap_text + "\n\n".join(current_sections)
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    chunk_index=start_index + len(chunks),
                    source_page=page_num,
                    token_count=_approx_tokens(chunk_text),
                    has_code=any("```" in s for s in current_sections),
                ))
        return chunks

    def _split_into_sections(self, text: str) -> list[str]:
        sections: list[str] = []
        last_end = 0
        for match in self._CODE_BLOCK_RE.finditer(text):
            before = text[last_end:match.start()]
            if before.strip():
                sections.extend(self._split_prose(before))
            sections.append(match.group(0))
            last_end = match.end()
        remaining = text[last_end:]
        if remaining.strip():
            sections.extend(self._split_prose(remaining))
        return [s for s in sections if s.strip()]

    def _split_prose(self, text: str) -> list[str]:
        parts = self._PARAGRAPH_SEP.split(text)
        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _approx_tokens(part) > CHUNK_MAX_TOKENS:
                current = ""
                for sub in part.split("\n"):
                    if _approx_tokens(current + sub) > CHUNK_TARGET_TOKENS and current:
                        result.append(current.strip())
                        current = sub
                    else:
                        current = current + "\n" + sub if current else sub
                if current.strip():
                    result.append(current.strip())
            else:
                result.append(part)
        return result

    def _build_overlap(self, sections: list[str]) -> str:
        overlap_chars = CHUNK_OVERLAP_TOKENS * 4
        prose = [s for s in sections if not s.strip().startswith("```")]
        if not prose:
            return ""
        tail = prose[-1]
        if len(tail) > overlap_chars:
            tail = tail[-overlap_chars:]
            idx = tail.find(". ")
            if idx > 0:
                tail = tail[idx + 2:]
        return tail.strip() + "\n\n" if tail.strip() else ""

def extract_concept_tags(chunk_text: str, programming_language: str) -> list[str]:
    tags: list[str] = []
    for match in re.finditer(r"^#{1,3}\s+(.+)", chunk_text, re.MULTILINE):
        heading = match.group(1).strip()
        if 3 < len(heading) < 60:
            tags.append(heading.title())
    tags.append(programming_language)
    return list(set(tags))[:10]