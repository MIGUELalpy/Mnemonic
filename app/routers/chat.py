"""
Chat endpoint — the primary interface for querying the knowledge base.

Accepts a user query, runs it through the full LangGraph agent pipeline
(Router → Query Expander → Retrieval → Synthesizer → Reviewer), and
returns a cited, verified response.

Endpoints:
  POST /chat          — submit a query or code submission
  GET  /chat/health   — check that Ollama and the agent graph are ready
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.graph import agent_graph
from app.core.auth import verify_token
from app.llm.client import check_ollama_health

router = APIRouter(prefix="/chat", tags=["chat"])

#  Request / Response schemas 

class ChatRequest(BaseModel):
    query: str = Field(
        description="Your question or code submission.",
        min_length=3,
        max_length=4000,
        examples=[
            "How does the attention mechanism work in transformers?",
            "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
        ],
    )
    programming_language: str | None = Field(
        default=None,
        description="Override detected language: C++ | Python | C#",
    )
    user_language: str = Field(
        default="EN",
        description="Preferred response language: EN | PT | DE | ZH",
    )


class CitationOut(BaseModel):
    source_title: str
    source_page: int
    exact_source_quote: str


class ChatResponse(BaseModel):
    response: str
    citations: list[CitationOut]
    query_type: str
    reviewer_verdict: str
    grade: int | None = None
    grade_rationale: str | None = None
    elapsed_ms: int

#  Endpoints 

@router.get("/health", dependencies=[Depends(verify_token)])
async def chat_health() -> dict:
    """
    Checks that Ollama is reachable and the model is loaded.
    Run this before making your first query.
    """
    ollama_ok = await check_ollama_health()
    return {
        "ollama_ready": ollama_ok,
        "model": "qwen2.5-coder:7b",
        "hint": "If ollama_ready is false, run: docker compose exec ollama ollama pull qwen2.5-coder:7b",
    }

@router.post("", dependencies=[Depends(verify_token)], response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Runs the full agent pipeline on the user's query.

    The pipeline:
      1. Router        — classifies request type and language
      2. Query Expander — generates precise retrieval queries
      3. Retrieval      — Neo4j pre-filter + Qdrant vector search
      4. Synthesizer    — generates cited response from chunks
      5. Reviewer       — verifies citations via difflib / NLI

    Note: The secondary node (GTX 1650) must be running for retrieval
    to work (it embeds the query). If it is offline, the pipeline will
    still run but retrieval will return no chunks and the response will
    indicate that no relevant information was found.
    """
    t_start = time.time()

    # Build initial state
    initial_state = {
        "query": request.query.strip(),
        "user_language": request.user_language,
    }

    # Allow explicit language override
    if request.programming_language:
        initial_state["programming_language"] = request.programming_language

    # Run the pipeline
    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline failed: {str(e)}",
        )

    elapsed_ms = int((time.time() - t_start) * 1000)

    # Check for pipeline error
    if final_state.get("error") and not final_state.get("final_response"):
        raise HTTPException(
            status_code=500,
            detail=final_state["error"],
        )

    # Build citations output
    citations_out = [
        CitationOut(
            source_title=c.get("source_title", ""),
            source_page=c.get("source_page", 1),
            exact_source_quote=c.get("exact_source_quote", ""),
        )
        for c in final_state.get("citations", [])
    ]

    return ChatResponse(
        response=final_state.get("final_response") or final_state.get("response", ""),
        citations=citations_out,
        query_type=final_state.get("query_type", "research_query"),
        reviewer_verdict=final_state.get("reviewer_verdict", "unknown"),
        grade=final_state.get("grade"),
        grade_rationale=final_state.get("grade_rationale"),
        elapsed_ms=elapsed_ms,
    )