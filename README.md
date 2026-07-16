# Mnemonic — AI Developer Learning Platform

---

## What is Mnemonic?

Mnemonic is a personal study platform that generates coding challenges grounded in academic literature you choose, evaluates your solutions with a locally running LLM, and uses the **FSRS spaced repetition algorithm** to schedule when you study each topic next — so you spend time on what you're actually forgetting, not what you already know.

Everything runs on your own hardware. No API keys. No subscriptions. No data leaves your machine.

---

## How it works

```
make up
    ↓
Celery detects due topics → pre-generates grounded challenges
    ↓
You open http://localhost:8000/ui
    ↓
Pick a topic → read the challenge → write code in your IDE
    ↓
Paste your solution → Submit & Evaluate
    ↓
Piston executes your code in a sandbox
    ↓
LLM grades it 1–10 with feedback cited from your knowledge base
    ↓
FSRS schedules your next review based on how well you did
    ↓
Progress appears on your dashboard
```

---

## Features

- **Knowledge base** built from your own PDFs (papers, textbooks) — the system classifies, chunks, embeds, and stores them automatically
- **5-agent LangGraph pipeline** — Router → Query Expander → Retrieval → Synthesizer → Reviewer — with citation verification at every step
- **GraphRAG retrieval** — Neo4j concept pre-filter + Qdrant vector search for precise, diverse citations
- **Sandboxed code execution** via Piston (Python, C++, C#)
- **FSRS v4 spaced repetition** — 1–10 grade scale mapped to scheduling intervals
- **Web UI** with Dashboard, Challenge, Chat, Topics, and Study Log views
- **Automatic topic suggestions** from submitted code
- **Study log** synced to a secondary machine over rsync
- **Fully offline** — Qwen2.5-Coder-7B via Ollama, BGE-M3 embeddings via a secondary GPU node

---

## Architecture

```
Primary machine (RTX 4060 / 8GB VRAM)
├── FastAPI app          ← API + Web UI
├── Celery worker        ← Background challenge generation
├── Ollama               ← Qwen2.5-Coder-7B (inference)
├── Piston               ← Sandboxed code execution
├── Qdrant               ← Vector store (4,700+ chunks)
├── Neo4j                ← Concept graph (200+ concepts)
├── PostgreSQL           ← Users, topics, challenges, FSRS state
└── Redis                ← Celery broker

Secondary machine (GTX 1650 / 4GB VRAM) — optional
└── FastAPI node         ← BGE-M3 embeddings + DeBERTa NLI
```

---

## Hardware Requirements

**Primary machine (required):**
- NVIDIA GPU with **8GB+ VRAM** (tested: RTX 4060 8GB)
- 16GB+ RAM recommended
- Docker + NVIDIA Container Toolkit

**Secondary machine (optional but recommended):**
- NVIDIA GPU with **4GB+ VRAM** (tested: GTX 1650 4GB)
- Used for BGE-M3 document embedding and DeBERTa NLI citation verification
- Without it: ingestion and retrieval still work but run on CPU (slower)

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/MIGUELalpy/Mnemonic
cd mnemonic
cp .env.example .env
# Edit .env with your settings

# 2. Build and launch
make build
make up
make init

# 3. Pull the LLM model (once)
docker compose exec ollama ollama pull qwen2.5-coder:7b

# 4. Install Piston runtimes (once)
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language": "python", "version": "3.10.0"}'

# 5. Open the UI
open http://localhost:8000/ui
```

See [INSTALL.md](INSTALL.md) for the complete setup guide including the secondary node, adding books, and concept seeding.

---

## Adding Your Own Books and Papers

1. Drop PDFs into `data/intake/inbox/`
2. Run the classifier:
   ```bash
   curl -X POST http://localhost:8000/admin/classify/run \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```
3. Review proposals in `data/intake/pending_review.json`
4. Confirm and ingest:
   ```bash
   curl -X POST http://localhost:8000/admin/classify/confirm \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"run_ingestion": true}'
   ```

The system supports theory papers and applied textbooks. Foundational books like Bishop PRML, Goodfellow Deep Learning, Cormen Algorithms, and Géron Hands-On ML work especially well.

---

## Study Workflow

1. **Add a topic** in the Topics view (e.g. "Attention Mechanism", Python)
2. **Restart** — Celery auto-generates a challenge for every due topic on startup
3. **Go to Challenge** — read the generated problem grounded in your knowledge base
4. **Write your solution** in VS Code or any IDE
5. **Paste and submit** — Piston runs your code, the LLM grades it 1–10
6. **Review feedback and citations** — every critique is backed by a page reference from your books
7. **Check your dashboard** — see your grade trends and upcoming reviews

---

## API Reference

The full API is available at `http://localhost:8000/docs` when running in development mode (`ENVIRONMENT=development` in `.env`).

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/ui` | Web interface |
| `POST` | `/chat` | Query the knowledge base |
| `POST` | `/challenge/generate` | Generate a new challenge |
| `GET` | `/challenge/pending` | List pending challenges |
| `POST` | `/submit/{id}` | Submit and grade a solution |
| `GET` | `/topics` | List study topics |
| `POST` | `/topics` | Add a topic |
| `POST` | `/admin/classify/run` | Classify inbox PDFs |
| `POST` | `/admin/classify/confirm` | Move and ingest classified PDFs |
| `GET` | `/admin/db` | Database statistics |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI + Uvicorn |
| Agent pipeline | LangGraph |
| LLM inference | Ollama (Qwen2.5-Coder-7B) |
| Embeddings | BGE-M3 via HuggingFace |
| NLI verification | DeBERTa-v3-large-MNLI |
| Vector store | Qdrant |
| Graph database | Neo4j |
| Relational DB | PostgreSQL + SQLModel |
| Task queue | Celery + Redis |
| Code execution | Piston |
| Spaced repetition | FSRS v4 |
| Frontend | Vanilla JS SPA |

---

## Project Structure

```
mnemonic/
├── app/
│   ├── agents/          # LangGraph pipeline (5 agents)
│   ├── db/              # Neo4j, Qdrant, PostgreSQL clients
│   ├── ingestion/       # PDF classifier, chunker, embedder
│   ├── llm/             # Ollama client
│   ├── routers/         # FastAPI endpoints
│   ├── services/        # FSRS, Piston
│   ├── static/          # Web UI (HTML, CSS, JS)
│   └── worker/          # Celery tasks
├── scripts/
│   ├── init_db.py       # Database bootstrap
│   ├── seed_concepts.py # Neo4j concept graph population
│   ├── backup_db.sh     # PostgreSQL backup to remote
│   └── sync_logs.sh     # Rsync study logs to remote
├── data/
│   └── intake/
│       └── inbox/       # Drop PDFs here for ingestion
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
└── .env.example
```

---

## Secondary Node Setup

The secondary node runs BGE-M3 (embedding) and DeBERTa (NLI) on a separate GPU to keep the primary machine's VRAM free for inference.

See [INSTALL.md#secondary-node](INSTALL.md#secondary-node) for the full setup.

---

## License

MIT License — use it, modify it, share it.

---

## Contributing

Issues and pull requests welcome. This is a personal project built for self-study — if you find it useful and improve it, sharing back is appreciated.
