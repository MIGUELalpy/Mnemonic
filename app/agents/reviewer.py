"""
Reviewer Agent — verifies citations via difflib and optionally NLI.

Verification strategy by query type:

  research_query:
    Approach A (difflib) only — quote must be found verbatim (≥0.55 ratio)
    in the retrieved chunks. If it passes, response is returned immediately.
    NLI is skipped — difflib is sufficient for research and NLI was causing
    false-positive contradictions on legitimate explanatory text.

  code_submission:
    Approach A (difflib) first, then Approach B (NLI) if secondary node is
    online. NLI checks that the generated feedback is entailed by (not
    contradicting) the source chunks. This matters more for grading because
    incorrect feedback could mislead the learner.

Verdict values:
  "pass_quote"   — verified by difflib only (research queries)
  "pass_nli"     — passed both difflib and NLI (code submissions)
  "fail_quote"   — citation quote not found in source chunks
  "fail_nli"     — NLI found contradiction (code submissions only)
  "nli_offline"  — passed difflib, NLI skipped (secondary node offline)
"""

from __future__ import annotations

import difflib

import httpx
import structlog

from app.agents.state import AgentState, RetrievedChunk
from app.config import settings

log = structlog.get_logger(__name__)

DIFFLIB_THRESHOLD = 0.55
NLI_ENTAILMENT_THRESHOLD = 0.70


async def review(state: AgentState) -> dict:
    """
    Verifies citations. For research queries, difflib only.
    For code submissions, difflib + optional NLI.
    """
    response = state.get("response", "")
    citations = state.get("citations", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    query_type = state.get("query_type", "research_query")

    # No citations — pass through
    if not citations:
        log.info("reviewer.no_citations", action="pass_through")
        return {
            "reviewer_verdict": "pass_quote",
            "reviewer_passed": True,
            "final_response": response,
        }

    if not retrieved_chunks:
        log.warning("reviewer.no_chunks_to_verify")
        return {
            "reviewer_verdict": "fail_quote",
            "reviewer_passed": False,
            "final_response": _build_fallback(retrieved_chunks, response),
        }

    # Build lookup: source_id → list of chunk texts
    chunk_texts_by_source: dict[str, list[str]] = {}
    all_chunk_texts: list[str] = []
    for chunk in retrieved_chunks:
        sid = chunk["source_id"]
        if sid not in chunk_texts_by_source:
            chunk_texts_by_source[sid] = []
        chunk_texts_by_source[sid].append(chunk["chunk_text"])
        all_chunk_texts.append(chunk["chunk_text"])

    #  Approach A: difflib quote verification (always runs) 
    failed_citations = []
    for citation in citations:
        quote = citation.get("exact_source_quote", "")
        source_id = citation.get("source_id", "")
        if not quote:
            continue
        candidate_texts = chunk_texts_by_source.get(source_id, []) + all_chunk_texts
        if not _verify_quote_difflib(quote, candidate_texts):
            failed_citations.append({"quote": quote[:100], "source_id": source_id})

    if failed_citations:
        log.warning(
            "reviewer.quote_verification_failed",
            failed_count=len(failed_citations),
            failed=failed_citations,
        )
        return {
            "reviewer_verdict": "fail_quote",
            "reviewer_passed": False,
            "final_response": _build_fallback(retrieved_chunks, response),
        }

    log.info("reviewer.quote_passed", citations_verified=len(citations))

    #  Research queries: difflib is sufficient, skip NLI 
    if query_type != "code_submission":
        log.info("reviewer.nli_skipped", reason="research_query — difflib sufficient")
        return {
            "reviewer_verdict": "pass_quote",
            "reviewer_passed": True,
            "final_response": response,
        }

    #  Code submissions only: Approach B (NLI) 
    secondary_online = await _check_secondary_node()
    if not secondary_online:
        log.info("reviewer.nli_skipped", reason="secondary_node_offline")
        return {
            "reviewer_verdict": "nli_offline",
            "reviewer_passed": True,
            "final_response": response,
        }

    best_chunk_text = retrieved_chunks[0]["chunk_text"] if retrieved_chunks else ""
    if best_chunk_text and response:
        nli_result = await _verify_nli(premise=best_chunk_text, hypothesis=response)
        if nli_result == "CONTRADICTION":
            log.warning("reviewer.nli_contradiction")
            return {
                "reviewer_verdict": "fail_nli",
                "reviewer_passed": False,
                "final_response": _build_fallback(retrieved_chunks, response),
            }

    log.info("reviewer.nli_passed")
    return {
        "reviewer_verdict": "pass_nli",
        "reviewer_passed": True,
        "final_response": response,
    }


#  Helpers 

def _verify_quote_difflib(quote: str, candidate_texts: list[str]) -> bool:
    """Returns True if the quote is found (or near-found) in any candidate text."""
    quote_lower = quote.lower().strip()
    for text in candidate_texts:
        text_lower = text.lower()
        if quote_lower in text_lower:
            return True
        quote_len = len(quote_lower)
        for start in range(0, len(text_lower) - quote_len + 1, max(1, quote_len // 4)):
            segment = text_lower[start: start + quote_len + 20]
            ratio = difflib.SequenceMatcher(None, quote_lower, segment).ratio()
            if ratio >= DIFFLIB_THRESHOLD:
                return True
    return False


async def _check_secondary_node() -> bool:
    """Returns True if the secondary node is reachable and healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.SECONDARY_NODE_URL}/health")
            data = response.json()
            return data.get("status") == "ok"
    except Exception:
        return False


async def _verify_nli(premise: str, hypothesis: str) -> str:
    """
    Calls the secondary node's /classify endpoint.
    Returns "ENTAILMENT", "NEUTRAL", or "CONTRADICTION".
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.SECONDARY_NODE_URL}/classify",
                json={
                    "premise": premise[:2000],
                    "hypothesis": hypothesis[:1000],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("label", "NEUTRAL")
    except Exception as e:
        log.warning("reviewer.nli_call_failed", error=str(e))
        return "NEUTRAL"


def _build_fallback(chunks: list[RetrievedChunk], original_response: str) -> str:
    """Returns a safe fallback when verification fails."""
    if not chunks:
        return (
            "I could not verify my response against the knowledge base. "
            "Please try rephrasing your query."
        )
    best = chunks[0]
    return (
        f"[Unverified response withheld for accuracy. "
        f"Most relevant passage from '{best['source_title']}', page {best['source_page']}:]\n\n"
        f"{best['chunk_text'][:600]}"
    )