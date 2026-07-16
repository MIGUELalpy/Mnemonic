"""
Challenge generation endpoint.

Flow:
  1. Accept a topic name + programming language
  2. Query the knowledge base for relevant chunks (via retrieval agent)
  3. Use Qwen2.5-Coder-7B to generate a concrete coding challenge
     grounded in the retrieved academic content
  4. Store the challenge in PostgreSQL
  5. Return it ready to solve

Endpoints:
  POST /challenge/generate   — generate a new challenge for a topic
  GET  /challenge/{id}       — retrieve a stored challenge by ID
  GET  /challenge/pending    — list all unsolved challenges
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agents.retrieval import retrieve
from app.agents.state import AgentState
from app.core.auth import verify_token
from app.db.postgres.crud import (
    create_challenge,
    get_challenge,
    get_pending_challenges,
    get_user,
)
from app.db.postgres.session import get_session
from app.llm.client import chat_json, system_user

router = APIRouter(prefix="/challenge", tags=["challenge"])

#  Request / Response schemas 

class GenerateRequest(BaseModel):
    topic_name: str = Field(
        description="The topic to generate a challenge for.",
        examples=["Attention Mechanism", "K-Means Clustering", "Memory Management in C++"],
    )
    programming_language: str = Field(
        description="Target language: C++ | Python | C#",
        examples=["Python", "C++", "C#"],
    )
    difficulty: str = Field(
        default="intermediate",
        description="easy | intermediate | advanced",
    )

class ChallengeOut(BaseModel):
    id: int
    topic_name: str
    programming_language: str
    title: str
    description: str
    requirements: list[str]
    starter_code: str
    expected_behavior: str
    difficulty: str
    knowledge_refs: list[str]
    status: str
    created_at: str

#  Challenge generation prompt 

_SYSTEM_CHALLENGE = """You are a coding challenge designer for an advanced developer learning platform.
Generate a concrete, implementable coding challenge grounded in the provided knowledge base content.

The challenge must:
- Be directly implementable in the specified programming language
- Reference specific concepts from the knowledge base chunks
- Have clear, testable requirements
- Include starter code with the function signature and docstring
- Be appropriate for the specified difficulty level

Difficulty guidelines:
  easy:         single function, straightforward algorithm, no edge cases
  intermediate: multiple functions or classes, efficiency matters, handle edge cases
  advanced:     complex system design, optimization required, research-level concepts

Respond ONLY with this JSON:
{
  "title": "Short descriptive title",
  "description": "2-3 paragraph explanation of what to implement and why it matters",
  "requirements": ["requirement 1", "requirement 2", "requirement 3"],
  "starter_code": "complete starter code with function signature, type hints, and docstring",
  "expected_behavior": "what the correct implementation should do, including edge cases",
  "knowledge_refs": ["one sentence per chunk explaining how it informed this challenge"]
}"""

#  Endpoints 

@router.post("/generate", dependencies=[Depends(verify_token)])
async def generate_challenge(
    request: GenerateRequest,
    session: AsyncSession = Depends(get_session),
) -> ChallengeOut:
    """
    Generates a new coding challenge for the given topic and language.

    Uses the knowledge base to ground the challenge in academic content,
    then stores it in PostgreSQL for later submission and evaluation.

    The secondary node must be running for knowledge base retrieval.
    """
    t_start = time.time()

    if request.programming_language not in ("C++", "Python", "C#"):
        raise HTTPException(
            status_code=422,
            detail="programming_language must be C++, Python, or C#",
        )

    if request.difficulty not in ("easy", "intermediate", "advanced"):
        raise HTTPException(
            status_code=422,
            detail="difficulty must be easy, intermediate, or advanced",
        )

    #  Step 1: Retrieve relevant knowledge base chunks 
    retrieval_state: AgentState = {
        "query": request.topic_name,
        "expanded_queries": [
            request.topic_name,
            f"{request.topic_name} algorithm implementation",
            f"{request.topic_name} {request.programming_language}",
        ],
        "programming_language": request.programming_language,
    }

    retrieval_result = await retrieve(retrieval_state)
    retrieved_chunks = retrieval_result.get("retrieved_chunks", [])

    #  Step 2: Build context from chunks 
    if retrieved_chunks:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            context_parts.append(
                f"[{i}] From '{chunk['source_title']}' (page {chunk['source_page']}):\n"
                f"{chunk['chunk_text'][:600]}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        # Fallback: generate without knowledge base (warn but don't fail)
        context = f"No specific knowledge base content found for '{request.topic_name}'. Generate based on general knowledge."

    #  Step 3: Generate the challenge 
    user_message = (
        f"Topic: {request.topic_name}\n"
        f"Programming language: {request.programming_language}\n"
        f"Difficulty: {request.difficulty}\n\n"
        f"Knowledge base context:\n{context}"
    )

    try:
        result = await chat_json(
            messages=system_user(_SYSTEM_CHALLENGE, user_message),
            temperature=0.3,      # slight creativity for variety
            max_tokens=1500,
            timeout=180.0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Challenge generation failed: {str(e)}",
        )

    # Step 4: Validate and extract fields 
    title = str(result.get("title", f"{request.topic_name} Challenge"))
    description = str(result.get("description", ""))
    requirements = result.get("requirements", [])
    starter_code = str(result.get("starter_code", "# Your implementation here\n"))
    # Strip language fences that the model sometimes includes
    import re
    starter_code = re.sub(r'^```[a-zA-Z+#]*\n?', '', starter_code.strip())
    starter_code = re.sub(r'\n?```$', '', starter_code.strip())
    expected_behavior = str(result.get("expected_behavior", ""))
    knowledge_refs = result.get("knowledge_refs", [])

    if not isinstance(requirements, list):
        requirements = [requirements]
    if not isinstance(knowledge_refs, list):
        knowledge_refs = [str(knowledge_refs)]

    source_ids = [c["source_id"] for c in retrieved_chunks[:3]]

    # Step 5: Store in PostgreSQL 
    user = await get_user(session)
    if not user:
        raise HTTPException(status_code=400, detail="No user found. Run make init first.")

    challenge = await create_challenge(
        session=session,
        user_id=user.id,
        topic_name=request.topic_name,
        programming_language=request.programming_language,
        difficulty=request.difficulty,
        title=title,
        description=description,
        requirements=requirements,
        starter_code=starter_code,
        expected_behavior=expected_behavior,
        source_ids=source_ids,
        knowledge_refs=knowledge_refs,
    )

    elapsed = round(time.time() - t_start, 1)

    return ChallengeOut(
        id=challenge.id,
        topic_name=request.topic_name,
        programming_language=request.programming_language,
        title=title,
        description=description,
        requirements=requirements,
        starter_code=starter_code,
        expected_behavior=expected_behavior,
        difficulty=request.difficulty,
        knowledge_refs=knowledge_refs,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

@router.get("/pending", dependencies=[Depends(verify_token)])
async def list_pending_challenges(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Returns all unsolved challenges stored for the current user."""
    user = await get_user(session)
    if not user:
        return {"challenges": [], "count": 0}

    challenges = await get_pending_challenges(session, user.id)
    return {
        "count": len(challenges),
        "challenges": [
            {
                "id": c.id,
                "topic_name": c.topic_name,
                "programming_language": c.programming_language,
                "title": c.title,
                "difficulty": c.difficulty,
                "status": c.status,
                "created_at": str(c.created_at),
            }
            for c in challenges
        ],
    }

@router.get("/{challenge_id}", dependencies=[Depends(verify_token)])
async def get_challenge_by_id(
    challenge_id: int,
    session: AsyncSession = Depends(get_session),
) -> ChallengeOut:
    """Retrieves a stored challenge by its ID."""
    challenge = await get_challenge(session, challenge_id)
    if not challenge:
        raise HTTPException(status_code=404, detail=f"Challenge {challenge_id} not found.")

    return ChallengeOut(
        id=challenge.id,
        topic_name=challenge.topic_name,
        programming_language=challenge.programming_language,
        title=challenge.title,
        description=challenge.description,
        requirements=json.loads(challenge.requirements) if isinstance(challenge.requirements, str) else challenge.requirements,
        starter_code=challenge.starter_code,
        expected_behavior=challenge.expected_behavior,
        difficulty=challenge.difficulty,
        knowledge_refs=json.loads(challenge.knowledge_refs) if isinstance(challenge.knowledge_refs, str) else challenge.knowledge_refs,
        status=challenge.status,
        created_at=str(challenge.created_at),
    )