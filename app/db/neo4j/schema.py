"""
Neo4j schema initialization: constraints and indexes.

This module defines and applies the graph schema as described in
the architecture document (§7). It is idempotent — safe to run on
an already-initialized database.

Node labels and relationship types are defined as constants here.
Import them in operations.py to avoid magic strings.

Schema summary:
  Nodes:   Concept, Document, Document_Type, Language, Programming_Language
  Edges:   IS_EXPLAINED_IN, WRITTEN_IN, APPLIES_TO, IS_SUBTYPE_OF, PREREQUISITE_OF

Constraints enforce uniqueness on the natural key of each node type.
Indexes accelerate the Cypher queries used during retrieval filtering.
"""

import structlog

from app.db.neo4j.client import get_session

log = structlog.get_logger(__name__)

#  Node label constants 

class NodeLabel:
    CONCEPT = "Concept"
    DOCUMENT = "Document"
    DOCUMENT_TYPE = "Document_Type"
    LANGUAGE = "Language"
    PROGRAMMING_LANGUAGE = "Programming_Language"


#  Relationship type constants 

class RelType:
    IS_EXPLAINED_IN = "IS_EXPLAINED_IN"      # (Concept) → (Document)
    WRITTEN_IN = "WRITTEN_IN"                # (Document) → (Language)
    APPLIES_TO = "APPLIES_TO"               # (Document) → (Programming_Language)
    IS_SUBTYPE_OF = "IS_SUBTYPE_OF"         # (Document) → (Document_Type)
    PREREQUISITE_OF = "PREREQUISITE_OF"     # (Concept) → (Concept)


#  Schema definition
# Each entry: (node_label, property_name)
# A uniqueness constraint also creates an implicit index on the property.
_UNIQUENESS_CONSTRAINTS = [
    # Concept nodes are uniquely identified by their canonical name.
    (NodeLabel.CONCEPT, "name"),
    # Document nodes are uniquely identified by source_id (sha256 hash).
    # This is the join key shared with Qdrant.
    (NodeLabel.DOCUMENT, "source_id"),
    # Lookup nodes — small, static sets.
    (NodeLabel.DOCUMENT_TYPE, "name"),
    (NodeLabel.LANGUAGE, "code"),              # "EN" | "PT" | "DE" | "ZH"
    (NodeLabel.PROGRAMMING_LANGUAGE, "name"),  # "C++" | "Python" | "C#" | "Dart"
]

# Additional non-unique indexes for properties used in WHERE clauses
# that are not covered by uniqueness constraints.
_ADDITIONAL_INDEXES = [
    # Concepts are often queried by partial name match in the Query Expander.
    (NodeLabel.CONCEPT, "name"),       # already covered by constraint, listed for clarity
    # Documents are filtered by title in the admin ingestion flow.
    (NodeLabel.DOCUMENT, "source_title"),
]


async def init_schema() -> None:
    """
    Applies all uniqueness constraints and indexes to the Neo4j database.

    Idempotent: uses IF NOT EXISTS on all CREATE statements, so calling this
    multiple times (e.g. on every startup) is safe and results in no-ops after
    the first run.
    """
    log.info("neo4j.schema_init_start")

    async with get_session() as session:
        for label, prop in _UNIQUENESS_CONSTRAINTS:
            constraint_name = f"unique_{label.lower()}_{prop}"
            cypher = (
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )
            await session.run(cypher)
            log.info("neo4j.constraint_applied", label=label, property=prop)

        # Seed the static lookup nodes that are referenced by all documents.
        # These rarely-changing nodes (Language, Programming_Language, Document_Type)
        # are created with MERGE so they are idempotent.
        await _seed_lookup_nodes(session)

    log.info("neo4j.schema_init_complete")


async def _seed_lookup_nodes(session) -> None:
    """
    Creates the static lookup nodes used as relationship targets.

    These nodes form the vocabulary of the ontology. They are created once
    and referenced by document nodes via WRITTEN_IN, APPLIES_TO, and IS_SUBTYPE_OF.
    """

    # Language nodes
    languages = [
        ("EN", "English"),
        ("PT", "Portuguese"),
        ("DE", "German"),
        ("ZH", "Chinese"),
    ]
    for code, name in languages:
        await session.run(
            "MERGE (:Language {code: $code, name: $name})",
            code=code, name=name,
        )

    # Programming_Language nodes
    prog_languages = ["C++", "Python", "C#"]
    for lang in prog_languages:
        await session.run(
            "MERGE (:Programming_Language {name: $name})",
            name=lang,
        )

    # Document_Type nodes
    doc_types = ["Textbook", "Scientific Paper", "RFC"]
    for dtype in doc_types:
        await session.run(
            "MERGE (:Document_Type {name: $name})",
            name=dtype,
        )

    log.info("neo4j.lookup_nodes_seeded")


async def drop_all_data() -> None:
    """
    Deletes all nodes and relationships from the database.
    WARNING: Destructive and irreversible. Use only in development/testing.
    The schema constraints are preserved — only the data is deleted.
    """
    async with get_session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    log.warning("neo4j.all_data_deleted")