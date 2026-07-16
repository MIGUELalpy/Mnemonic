"""
Query Expander Agent — second node in the LangGraph pipeline.

Takes the user's raw query and expands it into 2-3 precise, specific
retrieval queries optimised for vector search over academic papers.

Why expand queries?
  The user's phrasing is often informal or ambiguous ("how does attention work").
  Vector search works better with technical, specific queries that match the
  vocabulary used in academic papers ("self-attention mechanism transformer
  architecture query key value").

  Multiple queries also improve recall — each targets a different facet
  of the question, and the retrieval agent deduplicates results.
"""

from __future__ import annotations

import structlog

from app.agents.state import AgentState
from app.llm.client import chat_json, system_user

log = structlog.get_logger(__name__)

_SYSTEM = """You are a query expansion specialist for a technical knowledge retrieval system.
The knowledge base contains academic papers on machine learning, deep learning, computer vision,
NLP, data mining, algorithms, systems programming, and software architecture.

Your task: expand the user's query into 2-3 precise retrieval queries that will find the
most relevant academic content. Each query should:
- Use technical vocabulary that appears in research papers
- Target a specific aspect of the question
- Be different enough to retrieve distinct relevant chunks

For code submissions (when code is provided), generate queries about the algorithms,
data structures, and design patterns used in the code.

Respond ONLY with this JSON:
{
  "queries": ["query1", "query2", "query3"]
}

Rules:
- 2-3 queries maximum, each 5-15 words
- Use precise technical terminology
- No questions, only declarative search phrases
- If programming_language is provided, include it in at least one query"""


async def expand_query(state: AgentState) -> dict:
    """
    Expands the user query into 2-3 precise retrieval queries.
    Returns partial state update with expanded_queries.
    """
    query = state.get("query", "")
    query_type = state.get("query_type", "research_query")
    prog_lang = state.get("programming_language")

    # Build context for the expander
    context_parts = [f"User query: {query}"]
    if query_type == "code_submission":
        context_parts.append("Type: code submission (focus on algorithms and patterns used)")
    if prog_lang:
        context_parts.append(f"Programming language: {prog_lang}")

    user_message = "\n".join(context_parts)

    try:
        result = await chat_json(
            messages=system_user(_SYSTEM, user_message),
            temperature=0.1,
            max_tokens=300,
        )

        queries = result.get("queries", [])

        # Validate and clean
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        queries = queries[:3]   # cap at 3

        # Always include the original query as a fallback
        if query not in queries:
            queries.insert(0, query)
        queries = queries[:3]

        log.info("query_expander.expanded", original=query, expanded=queries)

        return {"expanded_queries": queries}

    except Exception as e:
        log.error("query_expander.failed", error=str(e))
        # Fall back to original query only
        return {"expanded_queries": [query]}