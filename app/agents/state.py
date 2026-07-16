"""
Shared state object that flows through all five LangGraph agents.

Every agent reads from and writes to this state. LangGraph passes it
automatically between nodes — no agent needs to call the next one directly.

State fields by agent that writes them:

  Router Agent:
    query_type        — "research_query" | "code_submission"
    programming_language — detected from the query or submission

  Query Expander Agent:
    expanded_queries  — list of precise retrieval queries

  Retrieval Agent:
    retrieved_chunks  — list of chunk dicts from Qdrant
    graph_filter      — Neo4j filter params used (for generation_logs)
    source_ids        — source_ids returned by Neo4j pre-filter

  Synthesizer Agent:
    response          — the generated explanation text
    citations         — list of citation dicts with source info + exact quote
    grade             — 1-4 (for code submissions only)
    grade_rationale   — explanation of the grade (code submissions only)
    prompt_tokens     — token count of the full prompt sent to the model

  Reviewer Agent:
    reviewer_verdict  — "pass_quote" | "pass_nli" | "fail_quote" | "fail_nli" | "nli_offline"
    reviewer_passed   — True if response is verified, False if rejected
    final_response    — the response returned to the user (may differ from `response` if the reviewer rejected and fell back to raw chunks)

  Code submission fields (populated by Router if query_type == "code_submission"):
    code_submitted    — the raw code string
    piston_output     — raw JSON result from Piston execution
    compile_success   — True if Piston reported successful compilation
    execution_ms      — wall-clock execution time from Piston
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class RetrievedChunk(TypedDict):
    """A single chunk retrieved from Qdrant with its metadata."""
    point_id: str
    source_id: str
    source_title: str
    source_page: int
    doc_language: str
    programming_language: str
    document_type: str
    chunk_text: str
    score: float


class Citation(TypedDict):
    """A citation produced by the Synthesizer agent."""
    source_id: str
    source_title: str
    source_page: int
    exact_source_quote: str    # the verbatim quote from the chunk


class AgentState(TypedDict, total=False):
    """
    The shared state passed between all LangGraph agents.

    All fields are optional (total=False) because each agent only
    populates its own outputs — earlier agents don't know about later fields.

    Required on input (must be set before the graph starts):
      query            — the user's raw input text
    """

    #  Input (set before graph runs) 
    query: str                          # raw user input
    user_language: str                  # preferred response language (EN/PT/DE/ZH)

    #  Router Agent outputs 
    query_type: str                     # "research_query" | "code_submission"
    programming_language: Optional[str] # detected: "C++" | "Python" | "C#" | None
    topic_id: Optional[int]             # PostgreSQL topic ID if known

    #  Code submission fields (Router populates if code_submission) 
    code_submitted: Optional[str]
    piston_output: Optional[dict[str, Any]]
    compile_success: Optional[bool]
    execution_ms: Optional[int]

    #  Query Expander Agent outputs 
    expanded_queries: list[str]         # 2-3 precise retrieval queries

    #  Retrieval Agent outputs 
    source_ids: list[str]               # source_ids from Neo4j pre-filter
    graph_filter: dict[str, Any]        # filter params used in Neo4j query
    retrieved_chunks: list[RetrievedChunk]

    #  Synthesizer Agent outputs 
    response: str                       # generated explanation
    citations: list[Citation]
    grade: Optional[int]                # 1-4, code submissions only
    grade_rationale: Optional[str]      # explanation of grade
    prompt_tokens: int                  # token count for generation_logs

    #  Reviewer Agent outputs 
    reviewer_verdict: str               # see module docstring for values
    reviewer_passed: bool
    final_response: str                 # what actually gets returned to user

    #  Error tracking 
    error: Optional[str]                # set if any agent hits an unrecoverable error