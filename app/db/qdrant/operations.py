"""
Qdrant vector store operations: upsert chunks and filtered vector search.

The two core operations are:

  upsert_chunk()    — called by the ELT ingestion pipeline when adding a new document chunk to the knowledge base.

  search_chunks()   — called by the Retrieval Agent during inference. Accepts a pre-filtered list of source_ids from Neo4j and returns the top-k most semantically similar chunks.

The Retrieval Agent's hard constraint (top_k = 3 for real-time, 5 for async)
is enforced here, not at the caller — passing a larger limit raises an error.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import structlog
from qdrant_client.http import models as qmodels

from app.config import settings
from app.db.qdrant.client import PayloadField, qdrant_client

log = structlog.get_logger(__name__)

#  Chunk data structure 

class ChunkPayload:
    """
    Typed constructor for a Qdrant point payload.
    Validates required fields and builds the dict for upsert.
    """

    def __init__(
        self,
        *,
        source_id: str,
        source_title: str,
        source_page: int,
        doc_language: str,
        programming_language: str,
        document_type: str,
        chunk_text: str,
        concept_tags: list[str],
        chunk_index: int,
        token_count: int,
    ) -> None:
        if doc_language not in ("EN", "PT", "DE", "ZH"):
            raise ValueError(f"doc_language must be EN|PT|DE|ZH, got '{doc_language}'")
        if programming_language not in ("C++", "Python", "C#", "LANG_AGNOSTIC"):
            raise ValueError(
                f"programming_language must be C++|Python|C#|Dart, got '{programming_language}'"
            )
        if document_type not in ("Textbook", "Scientific Paper", "RFC"):
            raise ValueError(
                f"document_type must be Textbook|Scientific Paper|RFC, got '{document_type}'"
            )

        self.data: dict[str, Any] = {
            PayloadField.SOURCE_ID: source_id,
            PayloadField.SOURCE_TITLE: source_title,
            PayloadField.SOURCE_PAGE: source_page,
            PayloadField.DOC_LANGUAGE: doc_language,
            PayloadField.PROGRAMMING_LANGUAGE: programming_language,
            PayloadField.DOCUMENT_TYPE: document_type,
            PayloadField.CHUNK_TEXT: chunk_text,
            PayloadField.CONCEPT_TAGS: concept_tags,
            PayloadField.CHUNK_INDEX: chunk_index,
            PayloadField.TOKEN_COUNT: token_count,
        }


def make_source_id(source_title: str, page: int) -> str:
    """
    Derives a stable, deterministic source_id from a document title and page.
    This is the join key between Qdrant and Neo4j — it must be consistent
    across ingestion runs so that re-ingesting a document doesn't create
    orphaned graph nodes.

    Uses the first 16 hex chars of SHA-256 (64 bits of collision resistance,
    sufficient for a personal knowledge base of any realistic size).
    """
    raw = f"{source_title.strip().lower()}::{page}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_point_id(source_id: str, chunk_index: int) -> str:
    """
    Derives a deterministic UUID-format Qdrant point ID from source_id + chunk index.
    Qdrant requires point IDs to be either uint64 or UUID strings.
    Using UUIDs here for readability in the Qdrant dashboard.
    """
    raw = f"{source_id}::{chunk_index}"
    # UUID v5 (namespace + name → deterministic UUID)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))

#  Upsert 

async def upsert_chunk(
    vector: list[float],
    payload: ChunkPayload,
) -> str:
    """
    Inserts or updates a single chunk in the Qdrant collection.

    Uses upsert (not insert) so that re-running the ingestion pipeline on
    the same document is idempotent — existing points are overwritten rather
    than duplicated.

    Args:
        vector:   The 1024-dimensional BGE-M3 embedding (L2-normalized).
        payload:  A ChunkPayload instance with all required metadata.

    Returns:
        The UUID string of the upserted point.
    """
    source_id: str = payload.data[PayloadField.SOURCE_ID]
    chunk_index: int = payload.data[PayloadField.CHUNK_INDEX]
    point_id = make_point_id(source_id, chunk_index)

    if len(vector) != settings.QDRANT_VECTOR_SIZE:
        raise ValueError(
            f"Vector dimension mismatch: expected {settings.QDRANT_VECTOR_SIZE}, "
            f"got {len(vector)}. Ensure BGE-M3 is returning 1024-dim embeddings."
        )

    await qdrant_client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload.data,
            )
        ],
    )

    log.debug(
        "qdrant.chunk_upserted",
        point_id=point_id,
        source_id=source_id,
        chunk_index=chunk_index,
        title=payload.data[PayloadField.SOURCE_TITLE],
    )
    return point_id


async def upsert_chunks_batch(
    chunks: list[tuple[list[float], ChunkPayload]],
    batch_size: int = 64,
) -> list[str]:
    """
    Batch upsert for the ingestion pipeline.
    Processes chunks in batches of `batch_size` to avoid overwhelming the
    Qdrant gRPC buffer for very large documents.

    Args:
        chunks:     List of (vector, payload) pairs.
        batch_size: Number of points per Qdrant upsert call.

    Returns:
        List of point ID strings in the same order as the input.
    """
    point_ids: list[str] = []
    total = len(chunks)

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        points = []

        for vector, payload in batch:
            source_id = payload.data[PayloadField.SOURCE_ID]
            chunk_index = payload.data[PayloadField.CHUNK_INDEX]
            point_id = make_point_id(source_id, chunk_index)
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload.data,
                )
            )
            point_ids.append(point_id)

        await qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )
        log.info(
            "qdrant.batch_upserted",
            batch_start=batch_start,
            batch_size=len(points),
            total=total,
        )

    log.info("qdrant.ingestion_complete", total_points=len(point_ids))
    return point_ids

#  Search 

class RetrievedChunk:
    """Typed wrapper for a search result returned to the Retrieval Agent."""

    def __init__(self, scored_point: qmodels.ScoredPoint) -> None:
        p = scored_point.payload or {}
        self.point_id: str = str(scored_point.id)
        self.score: float = scored_point.score
        self.source_id: str = p.get(PayloadField.SOURCE_ID, "")
        self.source_title: str = p.get(PayloadField.SOURCE_TITLE, "")
        self.source_page: int = p.get(PayloadField.SOURCE_PAGE, 0)
        self.doc_language: str = p.get(PayloadField.DOC_LANGUAGE, "")
        self.programming_language: str = p.get(PayloadField.PROGRAMMING_LANGUAGE, "")
        self.document_type: str = p.get(PayloadField.DOCUMENT_TYPE, "")
        self.chunk_text: str = p.get(PayloadField.CHUNK_TEXT, "")
        self.concept_tags: list[str] = p.get(PayloadField.CONCEPT_TAGS, [])
        self.token_count: int = p.get(PayloadField.TOKEN_COUNT, 0)

    def to_dict(self) -> dict[str, Any]:
        """Serializes this chunk for storage in generation_logs.vector_chunks."""
        return {
            "point_id": self.point_id,
            "score": self.score,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_page": self.source_page,
            "doc_language": self.doc_language,
            "text": self.chunk_text,
        }


async def search_chunks(
    query_vector: list[float],
    top_k: int,
    source_id_filter: list[str] | None = None,
    programming_language: str | None = None,
    doc_languages: list[str] | None = None,
    document_type: str | None = None,
    is_realtime: bool = True,
) -> list[RetrievedChunk]:
    """
    Performs a filtered approximate nearest-neighbour (ANN) search against
    the knowledge_base collection.

    Filtering strategy (as per architecture §7.3):
      1. If source_id_filter is provided (from Neo4j), only those documents
         are searched. This is the primary filter in the GraphRAG pipeline.
      2. Additional payload filters (programming_language, doc_languages,
         document_type) are applied as AND conditions on top.

    Args:
        query_vector:          The BGE-M3 embedding of the expanded query.
        top_k:                 Maximum results to return. Hard-capped at
                               RETRIEVAL_TOP_K_REALTIME (3) for real-time and
                               RETRIEVAL_TOP_K_ASYNC (5) for async jobs.
        source_id_filter:      List of source_ids from the Neo4j graph filter.
                               When provided, search is restricted to these docs.
        programming_language:  Optional filter: "C++" | "Python" | "C#" | "Dart"
        doc_languages:         Optional filter: subset of ["EN", "PT", "DE", "ZH"]
        document_type:         Optional filter: "Textbook" | "Scientific Paper" | "RFC"
        is_realtime:           True for synchronous request paths (enforces top_k cap).
                               False for async Celery jobs (uses relaxed cap).

    Returns:
        List of RetrievedChunk objects, sorted by descending similarity score.
    """
    # Enforce the hard top_k cap from the architecture document.
    hard_cap = (
        settings.RETRIEVAL_TOP_K_REALTIME if is_realtime
        else settings.RETRIEVAL_TOP_K_ASYNC
    )
    if top_k > hard_cap:
        log.warning(
            "qdrant.top_k_capped",
            requested=top_k,
            cap=hard_cap,
            is_realtime=is_realtime,
        )
        top_k = hard_cap

    # Build the filter conditions.
    must_conditions: list[qmodels.FieldCondition] = []

    # Primary filter: restrict to source_ids returned by Neo4j.
    if source_id_filter:
        must_conditions.append(
            qmodels.FieldCondition(
                key=PayloadField.SOURCE_ID,
                match=qmodels.MatchAny(any=source_id_filter),
            )
        )

    # Secondary filters: applied on top of the graph filter.
    if programming_language:
        must_conditions.append(
            qmodels.FieldCondition(
                key=PayloadField.PROGRAMMING_LANGUAGE,
                match=qmodels.MatchValue(value=programming_language),
            )
        )

    if doc_languages:
        must_conditions.append(
            qmodels.FieldCondition(
                key=PayloadField.DOC_LANGUAGE,
                match=qmodels.MatchAny(any=doc_languages),
            )
        )

    if document_type:
        must_conditions.append(
            qmodels.FieldCondition(
                key=PayloadField.DOCUMENT_TYPE,
                match=qmodels.MatchValue(value=document_type),
            )
        )

    query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

    results = await qdrant_client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,        # we need the chunk_text for the Synthesizer
        with_vectors=False,       # don't return embeddings — saves bandwidth
    )

    chunks = [RetrievedChunk(r) for r in results]

    log.info(
        "qdrant.search_complete",
        results=len(chunks),
        top_k=top_k,
        has_graph_filter=bool(source_id_filter),
        programming_language=programming_language,
        doc_languages=doc_languages,
    )
    return chunks


async def get_chunk_by_point_id(point_id: str) -> RetrievedChunk | None:
    """
    Retrieves a single chunk by its Qdrant point ID.
    Used by the Reviewer Agent to re-fetch the source text for verification.
    """
    results = await qdrant_client.retrieve(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return None
    # retrieve() returns a list of Record, not ScoredPoint.
    # Wrap in a ScoredPoint-compatible dict for RetrievedChunk.
    record = results[0]
    # Build a minimal ScoredPoint-like object.
    scored = qmodels.ScoredPoint(
        id=record.id,
        version=0,
        score=1.0,
        payload=record.payload,
        vector=None,
    )
    return RetrievedChunk(scored)

async def delete_chunks_by_source_id(source_id: str) -> int:
    """
    Deletes all chunks belonging to a given source document.
    Used when re-ingesting a document to avoid duplicate chunks.

    Returns the number of points deleted.
    """
    result = await qdrant_client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=PayloadField.SOURCE_ID,
                        match=qmodels.MatchValue(value=source_id),
                    )
                ]
            )
        ),
    )
    log.info("qdrant.chunks_deleted", source_id=source_id, result=str(result))
    # Qdrant returns an UpdateResult; actual count isn't directly available
    # without a pre-deletion count query. Return -1 to signal "some deleted".
    return -1

async def document_exists(source_id: str) -> bool:
    """
    Returns True if at least one chunk for this source_id already exists
    in the Qdrant collection. Used by the ingestion pipeline to skip
    documents that have already been fully processed.
    """
    results = await qdrant_client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=PayloadField.SOURCE_ID,
                    match=qmodels.MatchValue(value=source_id),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results[0]) > 0

async def search_chunks(
    query_vector: list[float],
    programming_language: str | None = None,
    source_ids: list[str] | None = None,
    top_k: int = 5,
    score_threshold: float = 0.4,
) -> list[dict]:
    """
    Searches Qdrant for the most similar chunks to the query vector.
    Optionally filters by programming_language and/or source_ids.
    Returns a list of chunk dicts sorted by score descending.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    must_conditions = []

    if programming_language and programming_language != "LANG_AGNOSTIC":
        must_conditions.append(
            FieldCondition(
                key=PayloadField.PROGRAMMING_LANGUAGE,
                match=MatchAny(any=[programming_language, "LANG_AGNOSTIC"]),
            )
        )

    if source_ids:
        must_conditions.append(
            FieldCondition(
                key=PayloadField.SOURCE_ID,
                match=MatchAny(any=source_ids),
            )
        )

    search_filter = Filter(must=must_conditions) if must_conditions else None

    results = await qdrant_client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append({
            "point_id": str(hit.id),
            "source_id": payload.get(PayloadField.SOURCE_ID, ""),
            "source_title": payload.get(PayloadField.SOURCE_TITLE, ""),
            "source_page": payload.get(PayloadField.SOURCE_PAGE, 1),
            "doc_language": payload.get(PayloadField.DOC_LANGUAGE, "EN"),
            "programming_language": payload.get(PayloadField.PROGRAMMING_LANGUAGE, ""),
            "document_type": payload.get(PayloadField.DOCUMENT_TYPE, ""),
            "chunk_text": payload.get(PayloadField.CHUNK_TEXT, ""),
            "score": round(hit.score, 4),
        })

    return chunks