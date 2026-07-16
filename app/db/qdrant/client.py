"""
Qdrant async client setup and collection initialization.

The `knowledge_base` collection holds all ingested document chunks.
Each point in the collection represents one semantically chunked text
fragment with its BGE-M3 embedding (1024 dimensions, cosine distance)
and a rich metadata payload for filtered retrieval.

Collection initialization is idempotent — calling init_collection() on
an existing collection is safe and results in a no-op.
"""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings

log = structlog.get_logger(__name__)

#  Client singleton 

def _create_client() -> AsyncQdrantClient:
    """
    Creates the async Qdrant client.

    Connects via REST (default port 6333). gRPC (port 6334) is available
    for higher-throughput batch operations if needed in the ingestion pipeline.
    """
    kwargs: dict = {
        "host": settings.QDRANT_HOST,
        "port": settings.QDRANT_PORT,
    }
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY

    return AsyncQdrantClient(**kwargs)


# Module-level singleton. Import this in operations.py and agents.
qdrant_client: AsyncQdrantClient = _create_client()


#  Collection schema 

# Payload field names used in the collection — centralised here to avoid
# magic strings scattered across the codebase.
class PayloadField:
    SOURCE_ID = "source_id"               # stable hash: sha256(title + page)
    SOURCE_TITLE = "source_title"         # human-readable document title
    SOURCE_PAGE = "source_page"           # page number in the source document
    DOC_LANGUAGE = "doc_language"         # natural language: EN | PT | DE | ZH
    PROGRAMMING_LANGUAGE = "programming_language"  # C++ | Python | C# | Dart
    DOCUMENT_TYPE = "document_type"       # Textbook | Scientific Paper | RFC
    CONCEPT_TAGS = "concept_tags"         # list[str] of topic tags
    CHUNK_TEXT = "chunk_text"             # full text of the chunk (stored in payload)
    CHUNK_INDEX = "chunk_index"           # position of this chunk within its source doc
    TOKEN_COUNT = "token_count"           # approximate token count of this chunk

#  Collection initialization 

async def init_collection() -> None:
    """
    Creates the `knowledge_base` collection if it does not exist.

    Collection configuration:
      - Vector size:    1024 (BGE-M3 output dimensionality)
      - Distance:       Cosine (standard for normalized semantic embeddings)
      - Quantization:   Scalar INT8 (halves memory footprint, ~1% recall loss)
      - On-disk:        Payload stored on disk; only vectors loaded into RAM.
     Keeps Qdrant's RAM footprint manageable on the primary node.

    Indexed payload fields:
      All fields used as filters in vector searches must be indexed here.
      Qdrant skips un-indexed fields during filtering, causing full scans.
    """
    exists = await qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME)

    if exists:
        log.info("qdrant.collection_exists", collection=settings.QDRANT_COLLECTION_NAME)
        return

    log.info("qdrant.collection_creating", collection=settings.QDRANT_COLLECTION_NAME)

    await qdrant_client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=settings.QDRANT_VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
            # Store vectors on disk (mmap) rather than fully in RAM.
            # Essential for large knowledge bases on the primary node.
            on_disk=True,
        ),
        # Scalar quantization compresses float32 → int8, halving memory usage.
        # The quantile=0.99 setting clips only the most extreme outlier values,
        # preserving similarity quality for well-distributed BGE-M3 embeddings.
        quantization_config=qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,  # keep quantized vectors in RAM for fast ANN
            )
        ),
        # Store payload (metadata) on disk — only loaded on demand during retrieval.
        on_disk_payload=True,
    )

    # Create payload indexes for all fields used in filtering.
    # Without these, Qdrant performs a full scan on every filtered query.
    indexed_fields = [
        (PayloadField.SOURCE_ID, qmodels.PayloadSchemaType.KEYWORD),
        (PayloadField.DOC_LANGUAGE, qmodels.PayloadSchemaType.KEYWORD),
        (PayloadField.PROGRAMMING_LANGUAGE, qmodels.PayloadSchemaType.KEYWORD),
        (PayloadField.DOCUMENT_TYPE, qmodels.PayloadSchemaType.KEYWORD),
        (PayloadField.SOURCE_PAGE, qmodels.PayloadSchemaType.INTEGER),
        (PayloadField.TOKEN_COUNT, qmodels.PayloadSchemaType.INTEGER),
        # CONCEPT_TAGS is a list[str] — TEXT index enables substring matching.
        (PayloadField.CONCEPT_TAGS, qmodels.PayloadSchemaType.TEXT),
    ]

    for field_name, field_type in indexed_fields:
        await qdrant_client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name=field_name,
            field_schema=field_type,
        )
        log.info("qdrant.index_created", field=field_name, type=str(field_type))

    log.info(
        "qdrant.collection_ready",
        collection=settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )


async def get_collection_info() -> dict:
    """Returns current collection stats: point count, vector count, status."""
    info = await qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
    return {
        "name": settings.QDRANT_COLLECTION_NAME,
        "status": str(info.status),
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "indexed_vectors_count": info.indexed_vectors_count,
    }