"""
Router Agent — first node in the LangGraph pipeline.

Classifies the user's input into one of two types:
  research_query   — a question about concepts, algorithms, theory
  code_submission  — code the user wants evaluated

Also detects the target programming language (C++, Python, C#) when
relevant, and sets it to None for language-agnostic research questions.
"""

from __future__ import annotations

import structlog

from app.agents.state import AgentState
from app.llm.client import chat_json, system_user

log = structlog.get_logger(__name__)

_SYSTEM = """You are a routing classifier for a software engineering learning platform.
Classify the user's input and respond ONLY with a valid JSON object.

Classification rules:
- "research_query": a question about concepts, algorithms, theory, how something works,
  explanations, comparisons, or any request to learn or understand something.
- "code_submission": the user has submitted code (a function, class, solution, or program)
  that they want evaluated, reviewed, or graded.

Programming language detection:
- Detect from explicit mentions ("in Python", "C++ code", "using C#") or from code syntax.
- Set to null if the query is language-agnostic (e.g., "explain transformers").
- Only use: "C++", "Python", or "C#".

Respond with ONLY this JSON:
{
  "query_type": "research_query" or "code_submission",
  "programming_language": "C++" or "Python" or "C#" or null,
  "reasoning": "one sentence explanation"
}"""


async def route(state: AgentState) -> dict:
    """
    Classifies the user query and detects programming language.
    Returns partial state update with query_type and programming_language.
    """
    query = state.get("query", "")

    if not query.strip():
        log.warning("router.empty_query")
        return {
            "query_type": "research_query",
            "programming_language": None,
            "error": "Empty query received.",
        }

    try:
        result = await chat_json(
            messages=system_user(_SYSTEM, query),
            temperature=0.0,
            max_tokens=150,
        )

        query_type = result.get("query_type", "research_query")
        if query_type not in ("research_query", "code_submission"):
            query_type = "research_query"

        prog_lang = result.get("programming_language")
        if prog_lang not in ("C++", "Python", "C#", None):
            prog_lang = None

        log.info(
            "router.classified",
            query_type=query_type,
            programming_language=prog_lang,
            reasoning=result.get("reasoning", ""),
        )

        return {
            "query_type": query_type,
            "programming_language": prog_lang,
        }

    except Exception as e:
        log.error("router.failed", error=str(e))
        return {
            "query_type": "research_query",
            "programming_language": None,
            "error": f"Router failed: {str(e)}",
        }