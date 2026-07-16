"""
Two-stage retrieval:
  1. Neo4j concept pre-filter — finds source_ids of documents related to
     the query's domain concepts, narrowing the Qdrant search space.
  2. Qdrant vector search — embeds each expanded query and searches for
     the top-k most similar chunks, filtered to the pre-selected source_ids.

The secondary node (GTX 1650) must be running for embedding.
If it is offline, the agent falls back to an unfiltered Qdrant search
using a keyword-based query (lower quality but still functional).

Deduplication: chunks retrieved by multiple expanded queries are merged,
keeping the highest-scoring instance of each chunk.
"""

from __future__ import annotations

import structlog

from app.agents.state import AgentState, RetrievedChunk
from app.config import settings
from app.db.neo4j.operations import get_source_ids_by_concepts
from app.db.qdrant.operations import search_chunks

log = structlog.get_logger(__name__)


async def retrieve(state: AgentState) -> dict:
    """
    Runs Neo4j pre-filter + Qdrant vector search.
    Returns partial state with retrieved_chunks, source_ids, graph_filter.
    """
    expanded_queries = state.get("expanded_queries", [])
    prog_lang = state.get("programming_language")
    query = state.get("query", "")

    if not expanded_queries:
        expanded_queries = [query]

    #  Stage 1: Neo4j concept pre-filter 
    # Extract candidate concept names from the queries (simple keyword extraction)
    concept_candidates = _extract_concept_candidates(expanded_queries)

    source_ids: list[str] = []
    graph_filter: dict = {}

    if concept_candidates:
        try:
            source_ids = await get_source_ids_by_concepts(concept_candidates)
            graph_filter = {"concepts": concept_candidates, "source_ids_found": len(source_ids)}
            log.info(
                "retrieval.graph_filter",
                concepts=concept_candidates,
                source_ids_count=len(source_ids),
            )
        except Exception as e:
            log.warning("retrieval.graph_filter_failed", error=str(e))
            source_ids = []

    #  Stage 2: Qdrant vector search 
    # Embed each expanded query and search Qdrant
    all_chunks: dict[str, RetrievedChunk] = {}  # point_id → chunk (deduplicates)
    top_k = settings.RETRIEVAL_TOP_K_REALTIME

    for query_text in expanded_queries:
        try:
            # Embed via secondary node
            query_vector = await _embed_query(query_text)

            if query_vector is None:
                log.warning("retrieval.embed_failed", query=query_text[:50])
                continue

            # Search Qdrant with optional filters
            chunks = await search_chunks(
                query_vector=query_vector,
                programming_language=prog_lang,
                source_ids=source_ids if source_ids else None,
                top_k=top_k,
            )

            for chunk in chunks:
                point_id = chunk["point_id"]
                # Keep highest-scoring version of each chunk
                if point_id not in all_chunks or chunk["score"] > all_chunks[point_id]["score"]:
                    all_chunks[point_id] = chunk

        except Exception as e:
            log.error("retrieval.search_failed", query=query_text[:50], error=str(e))
            continue

    retrieved = sorted(all_chunks.values(), key=lambda c: c["score"], reverse=True)

    # Enforce source diversity — max N chunks per source_id
    max_per_source = getattr(settings, 'RETRIEVAL_MAX_CHUNKS_PER_SOURCE', 2)
    source_counts: dict[str, int] = {}
    diverse_chunks = []
    for chunk in retrieved:
        sid = chunk["source_id"]
        if source_counts.get(sid, 0) < max_per_source:
            diverse_chunks.append(chunk)
            source_counts[sid] = source_counts.get(sid, 0) + 1
        if len(diverse_chunks) >= top_k:
            break

    retrieved = diverse_chunks

    if not retrieved:
        log.warning("retrieval.no_results", queries=expanded_queries)

    log.info(
        "retrieval.complete",
        queries=len(expanded_queries),
        chunks_retrieved=len(retrieved),
        top_score=retrieved[0]["score"] if retrieved else 0.0,
    )

    return {
        "retrieved_chunks": retrieved,
        "source_ids": source_ids,
        "graph_filter": graph_filter,
    }


#  Helpers 

async def _embed_query(query_text: str) -> list[float] | None:
    """
    Embeds a query string via the secondary node's /embed endpoint.
    Returns None if the secondary node is offline.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=settings.SECONDARY_NODE_TIMEOUT) as client:
            response = await client.post(
                f"{settings.SECONDARY_NODE_URL}/embed",
                json={"text": query_text, "normalize": True},
            )
            response.raise_for_status()
            return response.json()["vector"]
    except Exception as e:
        log.warning(
            "retrieval.secondary_node_offline",
            error=str(e),
            hint="Start the secondary node with ./start.sh to enable vector search.",
        )
        return None


def _extract_concept_candidates(queries: list[str]) -> list[str]:
    """
    Extracts candidate concept names from expanded queries for Neo4j lookup.
    Uses a simple multi-word extraction — the Neo4j query does fuzzy matching.
    """
    # Known domain concepts to look for (matched case-insensitively)
    domain_keywords = {
    # existing entries...
    "machine learning", "deep learning", "neural network", "transformer",
    "attention mechanism", "convolutional", "recurrent", "lstm", "bert", "gpt",
    "reinforcement learning", "gradient descent", "backpropagation", "optimization",
    "computer vision", "object detection", "image segmentation", "nlp",
    "natural language processing", "data mining", "clustering", "classification",
    "regression", "decision tree", "random forest", "support vector",
    "graph neural network", "federated learning", "transfer learning",
    "algorithms", "data structures", "dynamic programming", "sorting",
    "database", "indexing", "query optimization", "systems programming",
    "memory management", "concurrency", "architecture", "design patterns",
    "c++", "python", "c#", "cuda", "parallel computing",
    "softmax", "relu", "sigmoid", "activation function", "weight initialization",
    "dropout", "batch normalization", "regularization", "overfitting",
    "supervised learning", "unsupervised learning", "cross-validation",
    "principal component analysis", "k-means", "naive bayes", "linear regression",
    "logistic regression", "random forest", "support vector machine",
    "autoencoder", "generative adversarial", "embedding", "tokenization",
    "named entity", "sentiment analysis", "language model", "encoder decoder",
    "object detection", "image segmentation", "feature extraction", "convolution",
    "sql", "nosql", "acid", "normalization", "indexing", "query optimization",
    "recursion", "hash table", "binary tree", "binary search", "heap",
    "greedy", "divide and conquer", "graph algorithm", "shortest path",
    "pointer", "template", "iterator", "lambda", "generics",
    "decorator", "generator", "context manager", "async await",
    "inheritance", "polymorphism", "encapsulation", "solid", "singleton",
    "numerical methods", "linear algebra", "eigenvalue", "matrix",
    "probability", "statistics", "bayesian", "monte carlo",
    "association rules", "anomaly detection", "pagerank", "community detection",
}

    candidates = []
    combined = " ".join(queries).lower()

    for keyword in domain_keywords:
        if keyword in combined:
            # Capitalize for Neo4j matching
            candidates.append(keyword.title())

    return candidates[:10]  # cap at 10 concepts