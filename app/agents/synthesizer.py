"""
Synthesizer Agent — generates cited response from retrieved chunks.

Grade rubric (code submissions) — 1 to 10 scale:
  10    = Flawless. Optimal algorithm, handles all edge cases, clean and documented.
  7-9   = Good. Correct and working, minor improvements possible.
  4-6   = Average. Mostly correct but has bugs, inefficiencies, or missing edge cases.
  1-3   = Poor. Fundamentally incorrect, does not compile, or completely wrong approach.
"""

from __future__ import annotations
import structlog
from app.agents.state import AgentState, Citation, RetrievedChunk
from app.config import settings
from app.llm.client import chat_json, system_user

log = structlog.get_logger(__name__)

_SYSTEM_RESEARCH = """You are a technical mentor for a software engineering learning platform.
Answer the user's question using ONLY the provided knowledge base chunks.
Every factual claim must be supported by a direct citation from a chunk.

Rules:
- Ground factual claims in the provided chunks when possible
- You MAY generate code examples to illustrate concepts, even if not in the chunks
- Always cite the chunk that inspired or supports each explanation
- You MUST cite at least 3 different sources if 3 or more chunks are provided
- Each citation must come from a DIFFERENT source_id — do not cite the same source twice
- COPY-PASTE the exact quote word-for-word from the chunk — do NOT paraphrase
- Be precise and technical — the user is an advanced developer
- If chunks are insufficient to answer well, say so clearly
- Keep response under 500 words

Example of correct citations format with multiple sources:
{
  "response": "Backpropagation works by...[claim 1]...Furthermore...[claim 2]...Additionally...[claim 3]",
  "citations": [
    {
      "source_id": "abc123",
      "source_title": "Deep Learning by Goodfellow",
      "source_page": 193,
      "exact_source_quote": "exact verbatim sentence from chunk 1"
    },
    {
      "source_id": "def456",
      "source_title": "Hands-On Machine Learning",
      "source_page": 762,
      "exact_source_quote": "exact verbatim sentence from chunk 2"
    },
    {
      "source_id": "ghi789",
      "source_title": "Bishop PRML",
      "source_page": 241,
      "exact_source_quote": "exact verbatim sentence from chunk 3"
    }
  ]
}

Respond ONLY with this JSON — no other text."""

_SYSTEM_CODE = """You are a code reviewer for a software engineering learning platform.
Evaluate the submitted code using the provided knowledge base chunks as your reference.

Grade rubric (1-10 scale):
  10    = Flawless. Optimal algorithm, handles all edge cases, clean and well documented.
  7-9   = Good. Correct and working, minor style or efficiency improvements possible.
  4-6   = Average. Mostly correct but has bugs, inefficiencies, or missing edge cases.
  1-3   = Poor. Fundamentally incorrect, does not compile, or completely wrong approach.

Rules:
- You MUST cite at least 3 different sources if 3 or more chunks are provided
- Each citation must come from a DIFFERENT source_id — do not cite the same source twice
- COPY-PASTE the exact quote word-for-word from the chunk — do NOT paraphrase
- Cite specific concepts from the knowledge base that relate to the code quality
- Be constructive — explain concretely how to improve where needed
- Reference algorithm complexity, design patterns, and best practices from the chunks
- Keep feedback under 500 words

Respond ONLY with this JSON:
{
  "response": "your feedback here",
  "grade": 1 to 10,
  "grade_rationale": "one sentence explaining the grade",
  "citations": [
    {
      "source_id": "abc123",
      "source_title": "Title of book 1",
      "source_page": 1,
      "exact_source_quote": "verbatim quote from chunk 1"
    },
    {
      "source_id": "def456",
      "source_title": "Title of book 2",
      "source_page": 2,
      "exact_source_quote": "verbatim quote from chunk 2"
    },
    {
      "source_id": "ghi789",
      "source_title": "Title of book 3",
      "source_page": 3,
      "exact_source_quote": "verbatim quote from chunk 3"
    }
  ]
}"""


async def synthesize(state: AgentState) -> dict:
    query = state.get("query", "")
    query_type = state.get("query_type", "research_query")
    retrieved_chunks = state.get("retrieved_chunks", [])
    prog_lang = state.get("programming_language")

    if not retrieved_chunks:
        log.warning("synthesizer.no_chunks")
        return {
            "response": "I could not find relevant information in the knowledge base for your query.",
            "citations": [],
            "final_response": "I could not find relevant information in the knowledge base for your query.",
            "prompt_tokens": 0,
        }

    context = _build_context(retrieved_chunks)

    if query_type == "code_submission":
        system_prompt = _SYSTEM_CODE
        user_message = f"Code to evaluate:\n```\n{query}\n```\n\nKnowledge base context:\n{context}"
        if prog_lang:
            user_message = f"Language: {prog_lang}\n\n" + user_message
    else:
        system_prompt = _SYSTEM_RESEARCH
        user_message = f"Question: {query}\n\nKnowledge base context:\n{context}"

    prompt_tokens = len(user_message) // 4 + len(system_prompt) // 4

    if prompt_tokens > settings.PROMPT_TOKEN_WARN_THRESHOLD:
        log.warning("synthesizer.prompt_too_long", tokens=prompt_tokens)

    try:
        result = await chat_json(
            messages=system_user(system_prompt, user_message),
            temperature=0.2,
            max_tokens=1024,
            timeout=180.0,
        )

        response = result.get("response", "")
        citations_raw = result.get("citations", [])

        citations: list[Citation] = []
        for c in citations_raw:
            if isinstance(c, dict) and c.get("exact_source_quote"):
                citations.append({
                    "source_id": str(c.get("source_id", "")),
                    "source_title": str(c.get("source_title", "")),
                    "source_page": int(c.get("source_page", 1)),
                    "exact_source_quote": str(c.get("exact_source_quote", "")),
                })

        # Replace fabricated quotes with real extracted sentences
        for i, citation in enumerate(citations):
            best_chunk = retrieved_chunks[i] if i < len(retrieved_chunks) else retrieved_chunks[0]
            citation["exact_source_quote"] = _extract_best_quote(
                citation["exact_source_quote"],
                best_chunk["chunk_text"],
            )

        output = {
            "response": response,
            "citations": citations,
            "prompt_tokens": prompt_tokens,
        }

        if query_type == "code_submission":
            grade = result.get("grade")
            try:
                grade = max(1, min(10, int(grade))) if grade is not None else None
            except (ValueError, TypeError):
                grade = None
            output["grade"] = grade
            output["grade_rationale"] = str(result.get("grade_rationale", ""))

        log.info("synthesizer.complete", query_type=query_type,
                 citations_count=len(citations), response_len=len(response),
                 grade=output.get("grade"))
        return output

    except Exception as e:
        log.error("synthesizer.failed", error=str(e))
        fallback = _build_fallback_response(retrieved_chunks, query)
        return {
            "response": fallback,
            "citations": [],
            "prompt_tokens": prompt_tokens,
            "error": f"Synthesizer failed: {str(e)}",
        }


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk['source_title']} (page {chunk['source_page']}, id: {chunk['source_id']})"
        parts.append(f"{header}\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def _extract_best_quote(claim: str, chunk_text: str) -> str:
    import difflib
    sentences = [s.strip() for s in chunk_text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    if not sentences:
        return chunk_text[:200]
    return max(sentences, key=lambda s: difflib.SequenceMatcher(None, claim.lower(), s.lower()).ratio())


def _build_fallback_response(chunks: list[RetrievedChunk], query: str) -> str:
    if not chunks:
        return "No relevant information found."
    best = chunks[0]
    return (f"Based on '{best['source_title']}' (page {best['source_page']}):\n\n"
            f"{best['chunk_text'][:500]}...")