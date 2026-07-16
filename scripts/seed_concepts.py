"""
Populates Neo4j with ~200 fundamental CS/ML concepts and links them
to existing documents using the actual graph schema:
  - source_title keyword matching
  - APPLIES_TO Programming_Language relationships
  - Broad linkage for foundational concepts

Run once after ingestion:
  docker compose exec app python -m scripts.seed_concepts
"""

import asyncio
import re
import structlog

log = structlog.get_logger(__name__)

#  Concept taxonomy 
# Format: { concept_name: {
#     "keywords": [...],      # matched against source_title (case-insensitive)
#     "languages": [...],     # matched against APPLIES_TO Programming_Language
#     "broad": bool           # if True, link to ALL documents
# }}

CONCEPTS = {
    # Deep Learning — broad foundational
    "Backpropagation":             {"keywords": ["deep learning", "neural", "machine learning", "goodfellow", "geron", "bishop"], "languages": [], "broad": False},
    "Gradient Descent":            {"keywords": ["deep learning", "neural", "machine learning", "goodfellow", "geron", "bishop", "numerical"], "languages": [], "broad": False},
    "Stochastic Gradient Descent": {"keywords": ["deep learning", "machine learning", "goodfellow", "geron"], "languages": [], "broad": False},
    "Adam Optimizer":              {"keywords": ["deep learning", "machine learning"], "languages": [], "broad": False},
    "Learning Rate":               {"keywords": ["deep learning", "machine learning", "goodfellow", "geron"], "languages": [], "broad": False},
    "Batch Normalization":         {"keywords": ["deep learning", "goodfellow"], "languages": [], "broad": False},
    "Dropout":                     {"keywords": ["deep learning", "machine learning", "goodfellow"], "languages": [], "broad": False},
    "Regularization":              {"keywords": ["deep learning", "machine learning", "bishop", "goodfellow"], "languages": [], "broad": False},
    "Overfitting":                 {"keywords": ["deep learning", "machine learning", "bishop", "goodfellow", "geron"], "languages": [], "broad": False},
    "Neural Network":              {"keywords": ["deep learning", "neural", "machine learning", "goodfellow", "geron", "bishop"], "languages": [], "broad": False},
    "Convolutional Neural Network":{"keywords": ["deep learning", "computer vision", "goodfellow", "geron"], "languages": [], "broad": False},
    "Recurrent Neural Network":    {"keywords": ["deep learning", "goodfellow", "speech", "language"], "languages": [], "broad": False},
    "LSTM":                        {"keywords": ["deep learning", "speech", "language", "goodfellow"], "languages": [], "broad": False},
    "Transformer":                 {"keywords": ["deep learning", "speech", "language", "goodfellow"], "languages": [], "broad": False},
    "Attention Mechanism":         {"keywords": ["deep learning", "speech", "language"], "languages": [], "broad": False},
    "Self-Attention":              {"keywords": ["deep learning", "speech", "language"], "languages": [], "broad": False},
    "Transfer Learning":           {"keywords": ["deep learning", "machine learning", "geron", "goodfellow"], "languages": [], "broad": False},
    "Embedding":                   {"keywords": ["deep learning", "speech", "language", "nlp"], "languages": [], "broad": False},
    "Word2Vec":                    {"keywords": ["speech", "language", "nlp"], "languages": [], "broad": False},
    "BERT":                        {"keywords": ["speech", "language", "nlp"], "languages": [], "broad": False},
    "Softmax":                     {"keywords": ["deep learning", "machine learning", "bishop", "goodfellow"], "languages": [], "broad": False},
    "ReLU":                        {"keywords": ["deep learning", "goodfellow"], "languages": [], "broad": False},
    "Cross-Entropy Loss":          {"keywords": ["deep learning", "machine learning", "bishop", "goodfellow"], "languages": [], "broad": False},
    "Activation Function":         {"keywords": ["deep learning", "neural", "goodfellow"], "languages": [], "broad": False},
    "Weight Initialization":       {"keywords": ["deep learning", "goodfellow"], "languages": [], "broad": False},
    "Vanishing Gradient":          {"keywords": ["deep learning", "goodfellow"], "languages": [], "broad": False},
    "Autoencoder":                 {"keywords": ["deep learning", "goodfellow"], "languages": [], "broad": False},
    "GAN":                         {"keywords": ["deep learning", "goodfellow"], "languages": [], "broad": False},

    # Machine Learning
    "Supervised Learning":         {"keywords": ["machine learning", "bishop", "geron", "data mining"], "languages": [], "broad": False},
    "Unsupervised Learning":       {"keywords": ["machine learning", "bishop", "geron", "data mining"], "languages": [], "broad": False},
    "Reinforcement Learning":      {"keywords": ["machine learning", "deep learning"], "languages": [], "broad": False},
    "Classification":              {"keywords": ["machine learning", "data mining", "bishop", "geron"], "languages": [], "broad": False},
    "Regression":                  {"keywords": ["machine learning", "bishop", "geron"], "languages": [], "broad": False},
    "Decision Tree":               {"keywords": ["machine learning", "data mining", "geron"], "languages": [], "broad": False},
    "Random Forest":               {"keywords": ["machine learning", "data mining", "geron"], "languages": [], "broad": False},
    "Support Vector Machine":      {"keywords": ["machine learning", "bishop", "geron"], "languages": [], "broad": False},
    "K-Nearest Neighbors":         {"keywords": ["machine learning", "geron"], "languages": [], "broad": False},
    "Naive Bayes":                 {"keywords": ["machine learning", "bishop"], "languages": [], "broad": False},
    "Linear Regression":           {"keywords": ["machine learning", "bishop", "geron"], "languages": [], "broad": False},
    "Logistic Regression":         {"keywords": ["machine learning", "bishop", "geron"], "languages": [], "broad": False},
    "Principal Component Analysis":{"keywords": ["machine learning", "bishop", "data mining"], "languages": [], "broad": False},
    "K-Means Clustering":          {"keywords": ["machine learning", "data mining", "bishop"], "languages": [], "broad": False},
    "Cross-Validation":            {"keywords": ["machine learning", "bishop", "geron"], "languages": [], "broad": False},
    "Bias-Variance Tradeoff":      {"keywords": ["machine learning", "bishop", "geron"], "languages": [], "broad": False},
    "Feature Engineering":         {"keywords": ["machine learning", "data mining", "geron"], "languages": [], "broad": False},
    "Ensemble Methods":            {"keywords": ["machine learning", "geron"], "languages": [], "broad": False},
    "Bayesian Inference":          {"keywords": ["machine learning", "bishop"], "languages": [], "broad": False},
    "Dimensionality Reduction":    {"keywords": ["machine learning", "bishop", "data mining"], "languages": [], "broad": False},

    # Algorithms & Data Structures — link to algorithms books
    "Sorting":                     {"keywords": ["algorithm", "cormen", "introduction to algorithm"], "languages": [], "broad": False},
    "Binary Search":               {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Dynamic Programming":         {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Graph Algorithms":            {"keywords": ["algorithm", "cormen", "graph"], "languages": [], "broad": False},
    "Depth-First Search":          {"keywords": ["algorithm", "cormen", "graph"], "languages": [], "broad": False},
    "Breadth-First Search":        {"keywords": ["algorithm", "cormen", "graph"], "languages": [], "broad": False},
    "Dijkstra's Algorithm":        {"keywords": ["algorithm", "cormen", "graph"], "languages": [], "broad": False},
    "Hash Table":                  {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Binary Tree":                 {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Heap":                        {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Dynamic Programming":         {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Time Complexity":             {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Big O Notation":              {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Recursion":                   {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Greedy Algorithm":            {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Divide And Conquer":          {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Minimum Spanning Tree":       {"keywords": ["algorithm", "cormen", "graph"], "languages": [], "broad": False},
    "Hashing":                     {"keywords": ["algorithm", "cormen"], "languages": [], "broad": False},
    "Graph":                       {"keywords": ["algorithm", "cormen", "graph database"], "languages": [], "broad": False},

    # C++ specific
    "Memory Management":           {"keywords": [], "languages": ["C++"], "broad": False},
    "Pointer":                     {"keywords": [], "languages": ["C++"], "broad": False},
    "RAII":                        {"keywords": [], "languages": ["C++"], "broad": False},
    "Move Semantics":              {"keywords": [], "languages": ["C++"], "broad": False},
    "Smart Pointer":               {"keywords": [], "languages": ["C++"], "broad": False},
    "Template":                    {"keywords": [], "languages": ["C++"], "broad": False},
    "STL":                         {"keywords": [], "languages": ["C++"], "broad": False},
    "Iterator":                    {"keywords": [], "languages": ["C++"], "broad": False},
    "Const Correctness":           {"keywords": [], "languages": ["C++"], "broad": False},
    "Lambda":                      {"keywords": [], "languages": ["C++", "Python", "C#"], "broad": False},

    # C# specific
    "Delegate":                    {"keywords": [], "languages": ["C#"], "broad": False},
    "LINQ":                        {"keywords": [], "languages": ["C#"], "broad": False},
    "Async Await":                 {"keywords": [], "languages": ["C#", "Python"], "broad": False},
    "Generics":                    {"keywords": [], "languages": ["C#", "C++"], "broad": False},
    "Extension Method":            {"keywords": [], "languages": ["C#"], "broad": False},

    # Python specific
    "Decorator":                   {"keywords": [], "languages": ["Python"], "broad": False},
    "Generator":                   {"keywords": [], "languages": ["Python"], "broad": False},
    "Context Manager":             {"keywords": [], "languages": ["Python"], "broad": False},
    "List Comprehension":          {"keywords": [], "languages": ["Python"], "broad": False},
    "Type Hints":                  {"keywords": [], "languages": ["Python"], "broad": False},

    # OOP — all languages
    "Object-Oriented Programming": {"keywords": ["pattern", "design", "game programming"], "languages": ["C++", "Python", "C#"], "broad": False},
    "Inheritance":                 {"keywords": ["pattern", "design"], "languages": ["C++", "Python", "C#"], "broad": False},
    "Polymorphism":                {"keywords": ["pattern", "design"], "languages": ["C++", "Python", "C#"], "broad": False},
    "Encapsulation":               {"keywords": ["pattern", "design"], "languages": ["C++", "Python", "C#"], "broad": False},
    "Design Patterns":             {"keywords": ["pattern", "design", "game programming"], "languages": ["C++", "Python", "C#"], "broad": False},
    "SOLID Principles":            {"keywords": ["pattern", "design"], "languages": [], "broad": False},

    # Concurrency
    "Concurrency":                 {"keywords": ["architecture", "system", "operating"], "languages": ["C++", "C#", "Python"], "broad": False},
    "Threading":                   {"keywords": ["architecture", "system"], "languages": ["C++", "C#", "Python"], "broad": False},
    "Deadlock":                    {"keywords": ["architecture", "system", "operating"], "languages": [], "broad": False},

    # Architecture
    "Virtual Memory":              {"keywords": ["architecture", "computer architecture", "quantitative"], "languages": [], "broad": False},
    "Cache":                       {"keywords": ["architecture", "computer architecture", "quantitative"], "languages": [], "broad": False},
    "Pipeline":                    {"keywords": ["architecture", "computer architecture", "quantitative"], "languages": [], "broad": False},
    "Instruction Set Architecture":{"keywords": ["architecture", "computer architecture", "quantitative"], "languages": [], "broad": False},
    "Compilation":                 {"keywords": ["architecture", "system"], "languages": [], "broad": False},

    # Databases
    "SQL":                         {"keywords": ["database", "silberschatz", "graph database"], "languages": [], "broad": False},
    "NoSQL":                       {"keywords": ["database", "graph database"], "languages": [], "broad": False},
    "Indexing":                    {"keywords": ["database", "silberschatz"], "languages": [], "broad": False},
    "Query Optimization":          {"keywords": ["database", "silberschatz"], "languages": [], "broad": False},
    "Normalization":               {"keywords": ["database", "silberschatz"], "languages": [], "broad": False},
    "ACID Properties":             {"keywords": ["database", "silberschatz"], "languages": [], "broad": False},
    "Transactions":                {"keywords": ["database", "silberschatz"], "languages": [], "broad": False},
    "Graph Database":              {"keywords": ["graph database", "oreilly"], "languages": [], "broad": False},
    "Relational Database":         {"keywords": ["database", "silberschatz"], "languages": [], "broad": False},

    # NLP
    "Natural Language Processing": {"keywords": ["speech", "language", "jurafsky", "nlp"], "languages": [], "broad": False},
    "Tokenization":                {"keywords": ["speech", "language", "jurafsky"], "languages": [], "broad": False},
    "Named Entity Recognition":    {"keywords": ["speech", "language", "jurafsky"], "languages": [], "broad": False},
    "Sentiment Analysis":          {"keywords": ["speech", "language", "jurafsky"], "languages": [], "broad": False},
    "Language Model":              {"keywords": ["speech", "language", "jurafsky", "deep learning"], "languages": [], "broad": False},
    "Machine Translation":         {"keywords": ["speech", "language", "jurafsky"], "languages": [], "broad": False},
    "Question Answering":          {"keywords": ["speech", "language", "jurafsky"], "languages": [], "broad": False},

    # Computer Vision
    "Object Detection":            {"keywords": ["computer vision", "szeliski"], "languages": [], "broad": False},
    "Image Segmentation":          {"keywords": ["computer vision", "szeliski"], "languages": [], "broad": False},
    "Image Classification":        {"keywords": ["computer vision", "szeliski", "deep learning"], "languages": [], "broad": False},
    "Feature Extraction":          {"keywords": ["computer vision", "szeliski", "machine learning"], "languages": [], "broad": False},
    "Convolution":                 {"keywords": ["computer vision", "deep learning", "goodfellow"], "languages": [], "broad": False},

    # Data Mining
    "Association Rules":           {"keywords": ["data mining", "han"], "languages": [], "broad": False},
    "Anomaly Detection":           {"keywords": ["data mining", "machine learning"], "languages": [], "broad": False},
    "Data Preprocessing":          {"keywords": ["data mining", "machine learning", "geron"], "languages": [], "broad": False},
    "CRISP-DM":                    {"keywords": ["data mining", "han"], "languages": [], "broad": False},
    "PageRank":                    {"keywords": ["data mining", "graph"], "languages": [], "broad": False},

    # Scientific Computing
    "Linear Algebra":              {"keywords": ["numerical", "scientific", "mathematics", "machine learning"], "languages": [], "broad": False},
    "Matrix Factorization":        {"keywords": ["numerical", "machine learning", "bishop"], "languages": [], "broad": False},
    "Eigenvalue":                  {"keywords": ["numerical", "scientific", "machine learning"], "languages": [], "broad": False},
    "Singular Value Decomposition":{"keywords": ["numerical", "machine learning", "bishop"], "languages": [], "broad": False},
    "Optimization":                {"keywords": ["numerical", "scientific", "machine learning", "deep learning"], "languages": [], "broad": False},
    "Probability":                 {"keywords": ["machine learning", "bishop", "deep learning"], "languages": [], "broad": False},
    "Statistics":                  {"keywords": ["machine learning", "data mining", "bishop"], "languages": [], "broad": False},
    "Monte Carlo":                 {"keywords": ["numerical", "scientific"], "languages": [], "broad": False},
    "Numerical Methods":           {"keywords": ["numerical", "scientific"], "languages": [], "broad": False},
}


async def seed_concepts() -> None:
    from neo4j import AsyncGraphDatabase
    from app.config import settings

    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    created = 0
    linked = 0

    try:
        async with driver.session() as session:
            # Get all documents with their titles and programming languages
            result = await session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:APPLIES_TO]->(pl:Programming_Language)
                RETURN d.source_id AS source_id,
                       d.source_title AS source_title,
                       collect(pl.name) AS prog_langs
            """)
            documents = await result.data()
            log.info("seed_concepts.documents_found", count=len(documents))

            for concept_name, config in CONCEPTS.items():
                # Upsert concept node
                await session.run(
                    "MERGE (c:Concept {name: $name}) ON CREATE SET c.seeded = true",
                    name=concept_name,
                )
                created += 1

                keywords = [k.lower() for k in config.get("keywords", [])]
                languages = config.get("languages", [])

                for doc in documents:
                    source_id = doc.get("source_id")
                    source_title = (doc.get("source_title") or "").lower()
                    prog_langs = doc.get("prog_langs") or []

                    should_link = False

                    # Match by title keyword
                    for kw in keywords:
                        if kw in source_title:
                            should_link = True
                            break

                    # Match by programming language
                    if not should_link and languages:
                        for lang in languages:
                            if lang in prog_langs:
                                should_link = True
                                break

                    if should_link and source_id:
                        await session.run("""
                            MATCH (d:Document {source_id: $source_id})
                            MATCH (c:Concept {name: $name})
                            MERGE (d)-[:HAS_CONCEPT]->(c)
                        """, source_id=source_id, name=concept_name)
                        linked += 1

            log.info("seed_concepts.complete",
                     concepts_created=created,
                     relationships_created=linked)
            print(f"\n✓ Created {created} concepts")
            print(f"✓ Created {linked} document-concept relationships")
            print("✓ Neo4j concept graph is now populated\n")

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(seed_concepts())