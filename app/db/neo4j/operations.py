"""
Neo4j graph operations: node creation, relationship management,
and the Cypher queries used by the Retrieval Agent.

The primary consumer of this module is the Retrieval Agent, which calls
`get_source_ids_for_query()` to get a filtered list of document source_ids
before passing them to Qdrant as a payload filter.

The ingestion pipeline calls `upsert_document()` and `link_concept_to_document()`
when adding new literature to the knowledge base.
"""

from __future__ import annotations

import structlog

from app.db.neo4j.client import get_session
from app.db.neo4j.schema import NodeLabel, RelType

log = structlog.get_logger(__name__)

# Document node operations (called by the ingestion pipeline)

async def upsert_document(
    *,
    source_id: str,
    source_title: str,
    document_type: str,
    doc_language: str,
    programming_language: str,
    total_pages: int | None = None,
    authors: list[str] | None = None,
) -> None:
    """
    Creates or updates a Document node and establishes its relationships to
    Language, Programming_Language, and Document_Type lookup nodes.

    Uses MERGE on source_id so re-ingesting a document updates its properties
    without creating duplicate nodes.

    Args:
        source_id:             Stable hash identifying this document (shared with Qdrant).
        source_title:          Human-readable document title.
        document_type:         "Textbook" | "Scientific Paper" | "RFC"
        doc_language:          "EN" | "PT" | "DE" | "ZH"
        programming_language:  "C++" | "Python" | "C#" | "Dart"
        total_pages:           Optional: total page count (informational).
        authors:               Optional: list of author name strings.
    """
    async with get_session() as session:
        # 1. Upsert the Document node.
        await session.run(
            """
            MERGE (d:Document {source_id: $source_id})
            ON CREATE SET
                d.source_title = $source_title,
                d.total_pages = $total_pages,
                d.authors = $authors,
                d.created_at = timestamp()
            ON MATCH SET
                d.source_title = $source_title,
                d.total_pages = $total_pages,
                d.authors = $authors,
                d.updated_at = timestamp()
            """,
            source_id=source_id,
            source_title=source_title,
            total_pages=total_pages,
            authors=authors or [],
        )

        # 2. WRITTEN_IN → Language
        await session.run(
            """
            MATCH (d:Document {source_id: $source_id})
            MATCH (l:Language {code: $code})
            MERGE (d)-[:WRITTEN_IN]->(l)
            """,
            source_id=source_id,
            code=doc_language,
        )

        # 3. APPLIES_TO → Programming_Language
        await session.run(
            """
            MATCH (d:Document {source_id: $source_id})
            MATCH (pl:Programming_Language {name: $name})
            MERGE (d)-[:APPLIES_TO]->(pl)
            """,
            source_id=source_id,
            name=programming_language,
        )

        # 4. IS_SUBTYPE_OF → Document_Type
        await session.run(
            """
            MATCH (d:Document {source_id: $source_id})
            MATCH (dt:Document_Type {name: $name})
            MERGE (d)-[:IS_SUBTYPE_OF]->(dt)
            """,
            source_id=source_id,
            name=document_type,
        )

    log.info(
        "neo4j.document_upserted",
        source_id=source_id,
        title=source_title,
        lang=doc_language,
        prog_lang=programming_language,
    )

# Concept node operations (called by the ingestion pipeline)

async def upsert_concept(name: str, description: str | None = None) -> None:
    """
    Creates or updates a Concept node.
    Concepts are the semantic backbone of the graph ontology — they connect
    related documents across languages and document types.
    """
    async with get_session() as session:
        await session.run(
            """
            MERGE (c:Concept {name: $name})
            ON CREATE SET c.description = $description, c.created_at = timestamp()
            ON MATCH SET c.description = $description
            """,
            name=name,
            description=description,
        )
    log.debug("neo4j.concept_upserted", name=name)


async def link_concept_to_document(concept_name: str, source_id: str) -> None:
    """
    Creates an IS_EXPLAINED_IN relationship between a Concept and a Document.
    This is the core relationship that enables concept-based retrieval filtering.
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (c:Concept {name: $concept_name})
            MATCH (d:Document {source_id: $source_id})
            MERGE (c)-[:IS_EXPLAINED_IN]->(d)
            RETURN count(*) AS linked
            """,
            concept_name=concept_name,
            source_id=source_id,
        )
        record = await result.single()
        if record and record["linked"] == 0:
            log.warning(
                "neo4j.link_failed",
                concept=concept_name,
                source_id=source_id,
                reason="One or both nodes not found",
            )


async def link_concept_prerequisite(
    prerequisite_name: str, dependent_name: str
) -> None:
    """
    Creates a PREREQUISITE_OF relationship: (prerequisite) -[:PREREQUISITE_OF]→ (dependent).
    Encodes the learning order between concepts (e.g. "Pointers" must precede "Smart Pointers").
    """
    async with get_session() as session:
        await session.run(
            """
            MATCH (pre:Concept {name: $pre_name})
            MATCH (dep:Concept {name: $dep_name})
            MERGE (pre)-[:PREREQUISITE_OF]->(dep)
            """,
            pre_name=prerequisite_name,
            dep_name=dependent_name,
        )
    log.debug(
        "neo4j.prerequisite_linked",
        prerequisite=prerequisite_name,
        dependent=dependent_name,
    )

# Retrieval Agent queries (the core GraphRAG filtering operations)

async def get_source_ids_for_query(
    *,
    programming_language: str | None = None,
    doc_languages: list[str] | None = None,
    document_type: str | None = None,
    concept_names: list[str] | None = None,
) -> list[str]:
    """
    The primary Retrieval Agent query.

    Returns a list of source_ids matching the given metadata filters.
    This list is passed to Qdrant as a payload filter, restricting the ANN
    search to only the relevant documents from the knowledge base.

    If no filters are provided, returns all document source_ids (fallback
    to pure vector search across the full collection).

    Args:
        programming_language: Filter to documents covering this language.
        doc_languages:        Filter to documents written in these natural languages.
        document_type:        Filter to "Textbook" | "Scientific Paper" | "RFC".
        concept_names:        Filter to documents explaining at least one of these concepts.

    Returns:
        List of source_id strings. May be empty if no documents match the filters.
        An empty list passed to search_chunks() falls back to unfiltered search.

    Example (from architecture §7.3):
        # Only scientific papers in Chinese or English covering Memory Management
        source_ids = await get_source_ids_for_query(
            programming_language="Python",
            doc_languages=["ZH", "EN"],
            document_type="Scientific Paper",
            concept_names=["Memory Management"],
        )
    """
    # Build the Cypher query dynamically based on which filters are active.
    conditions: list[str] = []
    params: dict = {}

    if programming_language:
        conditions.append(
            "EXISTS { (d)-[:APPLIES_TO]->(:Programming_Language {name: $prog_lang}) }"
        )
        params["prog_lang"] = programming_language

    if doc_languages:
        conditions.append(
            "EXISTS { (d)-[:WRITTEN_IN]->(l:Language) WHERE l.code IN $doc_languages }"
        )
        params["doc_languages"] = doc_languages

    if document_type:
        conditions.append(
            "EXISTS { (d)-[:IS_SUBTYPE_OF]->(:Document_Type {name: $doc_type}) }"
        )
        params["doc_type"] = document_type

    if concept_names:
        conditions.append(
            "EXISTS { (d)-[:HAS_CONCEPT]->(c:Concept) WHERE c.name IN $concept_names }"
        )
        params["concept_names"] = concept_names

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    cypher = f"""
        MATCH (d:{NodeLabel.DOCUMENT})
        {where_clause}
        RETURN d.source_id AS source_id
    """

    async with get_session() as session:
        result = await session.run(cypher, **params)
        records = await result.values("source_id")

    source_ids = [r[0] for r in records if r[0]]

    log.info(
        "neo4j.source_ids_retrieved",
        count=len(source_ids),
        prog_lang=programming_language,
        doc_languages=doc_languages,
        doc_type=document_type,
        concepts=concept_names,
    )
    return source_ids


async def get_concept_prerequisites(concept_name: str) -> list[str]:
    """
    Returns the names of concepts that are prerequisites for the given concept.
    Used by the Query Expander Agent to broaden queries when a concept
    has foundational dependencies.

    Example: get_concept_prerequisites("Smart Pointers") → ["Pointers", "Memory Management"]
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (pre:Concept)-[:PREREQUISITE_OF]->(dep:Concept {name: $name})
            RETURN pre.name AS prerequisite
            """,
            name=concept_name,
        )
        records = await result.values("prerequisite")

    return [r[0] for r in records if r[0]]


async def get_related_concepts(concept_name: str, depth: int = 2) -> list[str]:
    """
    Returns concept names related to the given concept within `depth` hops
    in the prerequisite graph. Used to broaden query expansion.

    Args:
        concept_name: The starting concept.
        depth:        Max relationship hops to traverse (default 2 to avoid over-expansion).

    Returns:
        List of related concept names, excluding the input concept itself.
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (c:Concept {name: $name})-[:PREREQUISITE_OF*1..$depth]-(related:Concept)
            WHERE related.name <> $name
            RETURN DISTINCT related.name AS name
            """,
            name=concept_name,
            depth=depth,
        )
        records = await result.values("name")

    return [r[0] for r in records if r[0]]

# Inspection and admin queries

async def get_graph_stats() -> dict:
    """
    Returns a summary of the current graph contents.
    Used by the health check and admin endpoints.
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (d:Document) WITH count(d) AS docs
            MATCH (c:Concept)  WITH docs, count(c) AS concepts
            MATCH ()-[r]->()   WITH docs, concepts, count(r) AS rels
            RETURN docs, concepts, rels
            """
        )
        record = await result.single()
        if record is None:
            return {"documents": 0, "concepts": 0, "relationships": 0}

        return {
            "documents": record["docs"],
            "concepts": record["concepts"],
            "relationships": record["rels"],
        }


async def get_all_documents() -> list[dict]:
    """
    Returns all Document nodes with their key properties.
    Used by the admin ingestion endpoint to show the current knowledge base.
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:WRITTEN_IN]->(l:Language)
            OPTIONAL MATCH (d)-[:APPLIES_TO]->(pl:Programming_Language)
            OPTIONAL MATCH (d)-[:IS_SUBTYPE_OF]->(dt:Document_Type)
            RETURN d.source_id AS source_id,
                   d.source_title AS title,
                   l.code AS language,
                   pl.name AS programming_language,
                   dt.name AS document_type
            ORDER BY d.source_title
            """
        )
        records = await result.data()
    return records


async def get_all_concepts() -> list[str]:
    """Returns all concept names, alphabetically sorted."""
    async with get_session() as session:
        result = await session.run(
            "MATCH (c:Concept) RETURN c.name AS name ORDER BY c.name"
        )
        records = await result.values("name")
    return [r[0] for r in records if r[0]]


async def delete_document_and_relationships(source_id: str) -> None:
    """
    Removes a Document node and all its relationships from the graph.
    Called when a document is removed from the knowledge base.
    Note: the corresponding Qdrant chunks must also be deleted separately
    using qdrant/operations.py:delete_chunks_by_source_id().
    """
    async with get_session() as session:
        await session.run(
            "MATCH (d:Document {source_id: $source_id}) DETACH DELETE d",
            source_id=source_id,
        )
    log.info("neo4j.document_deleted", source_id=source_id)


async def get_source_ids_by_concepts(concept_names: list[str]) -> list[str]:
    """
    Convenience wrapper for the Retrieval Agent.
    Returns source_ids of documents linked to any of the given concept names.
    Delegates to get_source_ids_for_query with concept_names filter.
    """
    return await get_source_ids_for_query(concept_names=concept_names)