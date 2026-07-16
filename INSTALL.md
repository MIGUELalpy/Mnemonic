# Mnemonic — Installation Guide

Complete setup instructions for a two-machine self-hosted deployment.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Primary Machine Setup](#primary-machine-setup)
- [Configuration](#configuration)
- [First Launch](#first-launch)
- [Secondary Node](#secondary-node)
- [Adding Knowledge Base Content](#adding-knowledge-base-content)
- [Concept Seeding](#concept-seeding)
- [Backup and Sync Setup](#backup-and-sync-setup)
- [Makefile Reference](#makefile-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Primary machine

**Operating system:** Linux (tested on Arch Linux and Ubuntu 24). Windows with WSL2 may work but is untested.

**Required software:**

```bash
# Docker Engine (not Docker Desktop)
# Follow: https://docs.docker.com/engine/install/

# NVIDIA Container Toolkit (for GPU passthrough to Docker)
# Arch Linux:
sudo pacman -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Ubuntu:
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify GPU is accessible:
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Hardware:**
- NVIDIA GPU with 8GB+ VRAM (RTX 3070, 3080, 4060, 4070 or better recommended)
- 16GB+ system RAM
- 50GB+ free disk space (Docker images + knowledge base)

### Secondary machine (optional)

- Any Linux machine with an NVIDIA GPU (4GB+ VRAM)
- Python 3.10+
- CUDA 12.x drivers
- Must be on the same local network as the primary machine

---

## Primary Machine Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mnemonic.git
cd mnemonic
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in all values. The required fields are:

```env
# Authentication — generate a random token for your API
MNEMONIC_API_TOKEN=your_random_token_here

# PostgreSQL
POSTGRES_USER=mnemonic
POSTGRES_PASSWORD=choose_a_strong_password
POSTGRES_DB=mnemonic

# Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=choose_a_strong_password

# Your display name
MNEMONIC_USER_NAME=YourName
MNEMONIC_USER_LANG=EN

# Secondary node (set after secondary node is running, or leave default)
SECONDARY_NODE_URL=http://192.168.1.X:8001
SECONDARY_NODE_TIMEOUT=120.0

# Ollama
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5-coder:7b

# Piston
PISTON_URL=http://piston:2000

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

To generate a secure token:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Build the Docker image

```bash
make build
```

This takes 5–10 minutes on first run (downloading base images and installing Python dependencies).

---

## First Launch

### 1. Start all services

```bash
make up
```

### 2. Initialize databases

```bash
make init
```

This creates all PostgreSQL tables, Neo4j constraints, and the Qdrant collection. Safe to run multiple times — it skips anything already created.

### 3. Pull the LLM model

This downloads Qwen2.5-Coder-7B (~4.7GB) into a named Docker volume. Only needed once — survives container restarts and rebuilds.

```bash
docker compose exec ollama ollama pull qwen2.5-coder:7b
```

### 4. Install Piston language runtimes

Piston runs your code in a sandboxed environment. Install the three supported runtimes:

```bash
# Python
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language": "python", "version": "3.10.0"}'

# C++
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language": "c++", "version": "10.2.0"}'

# C# (via Mono)
curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language": "mono", "version": "6.12.0"}'
```

Wait 30–60 seconds between each install. Verify:
```bash
curl http://localhost:2000/api/v2/runtimes
```

### 5. Verify everything is running

```bash
make health
```

Expected output:
```json
{
  "status": "ok",
  "services": {
    "postgres": "ok",
    "neo4j": "ok",
    "qdrant": "ok"
  }
}
```

### 6. Open the UI

```
http://localhost:8000/ui
```

Paste your `MNEMONIC_API_TOKEN` into the API Token field in the sidebar.

---

## Secondary Node

The secondary node runs BGE-M3 (1024-dimension embeddings) and DeBERTa-v3-large-MNLI (NLI citation verification) on a separate GPU machine. This keeps the primary machine's VRAM free for Ollama inference.

**Without the secondary node:** Document ingestion will not work. The chat and challenge pipeline will still run but without vector search (no retrieved context).

### Setup on the secondary machine

```bash
# Clone the secondary node files
git clone https://github.com/YOUR_USERNAME/mnemonic.git
cd mnemonic/secondary-node

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the service
./start.sh
```

The start script checks for GPU availability and loads both models. Startup takes 30–60 seconds. When ready you'll see:

```
[OK] bgem3.ready        elapsed_s=4.8
[OK] deberta.ready      elapsed_s=2.7
[OK] node.ready         message=Secondary node is ready to serve requests.
```

Note the IP address shown and set `SECONDARY_NODE_URL` in your primary machine's `.env` accordingly, then restart:

```bash
docker compose restart app worker
```

### Secondary node endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Returns `{"status": "ok"}` when ready |
| `POST /embed` | Embeds text using BGE-M3, returns 1024-dim vector |
| `POST /classify` | NLI entailment check using DeBERTa |

### When to run the secondary node

- **During ingestion** — always required for embedding new PDFs
- **During study sessions** — needed for vector retrieval in chat and challenges
- **Can be off** — the system degrades gracefully, just without vector-grounded retrieval

---

## Adding Knowledge Base Content

The knowledge base is what makes Mnemonic grounded and useful. The more quality content you add, the better the challenges and citations become.

### Recommended books to add (Just an example of some of the books I tested)

**Theory (language-agnostic):**
- Bishop — Pattern Recognition and Machine Learning (free at bishopbook.com)
- Goodfellow, Bengio, Courville — Deep Learning (free at deeplearningbook.org)
- Hastie, Tibshirani, Friedman — Elements of Statistical Learning (free at web.stanford.edu/~hastie/ElemStatLearn)
- Deisenroth, Faisal, Ong — Mathematics for Machine Learning (free at mml-book.github.io)

**Applied Python:**
- Géron — Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- Ramalho — Fluent Python
- Jurafsky & Martin — Speech and Language Processing (free at web.stanford.edu/~jurafsky/slp3)

**Applied C++:**
- Stroustrup — A Tour of C++
- Meyers — Effective Modern C++

**Applied C#:**
- Skeet — C# in Depth

**Algorithms:**
- Cormen, Leiserson, Rivest, Stein — Introduction to Algorithms

### Ingestion process

```bash
# 1. Drop PDFs into the inbox
cp your_book.pdf data/intake/inbox/

# 2. Start the secondary node (required for embedding)
# On secondary machine: source venv/bin/activate && ./start.sh

# 3. Run the AI classifier
curl -X POST http://localhost:8000/admin/classify/run \
  -H "Authorization: Bearer YOUR_TOKEN"

# 4. Review the proposals
# Open data/intake/pending_review.json and check the classifications
# Edit any incorrect tier/domain/language assignments directly in the JSON

# 5. Confirm and ingest (this can run for hours with large textbooks)
curl -X POST http://localhost:8000/admin/classify/confirm \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"run_ingestion": true}'

# 6. Monitor progress
docker logs mnemonic_app -f
```

**Ingestion speed:** Approximately 1–2 seconds per chunk on a GTX 1650. A 700-page textbook (~2000 chunks) takes 30–60 minutes. Start ingestion before sleeping and check in the morning.

**Safe to interrupt:** Re-running the ingest command skips already-completed documents and resumes from where it left off.

### Checking ingestion progress

```bash
curl http://localhost:8000/admin/db \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Returns document count, vector count, and concept relationships.

---

## Concept Seeding

After ingestion, populate the Neo4j concept graph to enable precise retrieval. This runs once and creates 200+ concept nodes linked to relevant documents.

```bash
docker compose exec app python -m scripts.seed_concepts
```

Expected output:
```
✓ Created 216 concepts
✓ Created 567 document-concept relationships
✓ Neo4j concept graph is now populated
```

**Re-run after every rebuild** or after adding significant new content.

---

## Adding Topics and Getting Started

```bash
# Add your first study topic via the UI
# Go to http://localhost:8000/ui → Topics → Add Topic

# Or via API:
curl -X POST http://localhost:8000/topics \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Attention Mechanism", "programming_language": "Python"}'

# Restart to trigger automatic challenge generation for due topics
make down && make up
```

---

## Backup and Sync Setup

### PostgreSQL backup

Set up weekly automatic backups to a remote machine:

```bash

cp scripts/sync_logs.sh.example scripts/sync_logs.sh
cp scripts/backup_db.sh.example scripts/backup_db.sh

# Make the backup script executable
chmod +x scripts/backup_db.sh

# Test it manually first
bash scripts/backup_db.sh

# Add to cron for weekly automatic backups (Sundays at 3AM)
crontab -e
# Add this line:
0 3 * * 0 bash /home/YOUR_USER/path/to/mnemonic/scripts/backup_db.sh
```

Edit `scripts/backup_db.sh` to set your remote machine's IP and path.

### SSH key setup (recommended)

For passwordless backup and log sync:

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519 -C "mnemonic-backup"

# Copy to remote machine
ssh-copy-id user@REMOTE_IP

# Test passwordless login
ssh user@REMOTE_IP "echo ok"
```

### Study log sync

```bash
chmod +x scripts/sync_logs.sh

# Sync manually after a study session
bash scripts/sync_logs.sh

# Or click the "⇅ Sync to Debian" button in the Log view
# and copy the displayed command
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make build` | Build Docker images (use after code changes) |
| `make up` | Start all services |
| `make down` | Stop all services |
| `make init` | Initialize/reset databases |
| `make health` | Check service health |
| `make logs` | Follow app logs |

### Standard rebuild cycle

```bash
make down
docker system prune -a    # clears build cache (~30GB reclaimed)
make build
make up
make init
docker compose exec ollama ollama pull qwen2.5-coder:7b  # if model was lost
docker compose exec app python -m scripts.seed_concepts
```

> **Note:** `docker system prune -a` does NOT delete named volumes. Your databases, the Ollama model, and Piston runtimes survive a prune. Never add `--volumes` unless you intend to wipe everything.

---

## Troubleshooting

### App fails to start

```bash
docker logs mnemonic_app --tail 30
```

Common causes:
- Missing `.env` values → check all required fields are set
- Port conflict → another service on 8000, check with `ss -tlnp | grep 8000`
- Import error → run `make build` to ensure latest code is in the image

### Piston returns 400 errors

The memory limit fields may not be supported by your Piston version. In `app/services/piston.py`, ensure the payload does not include `compile_memory_limit` or `run_memory_limit`.

Also ensure runtimes are installed:
```bash
curl http://localhost:2000/api/v2/runtimes
```

### Secondary node refuses connection

```bash
# On primary machine
curl http://SECONDARY_IP:8001/health

# Check secondary node logs for errors
# Common cause: CUDA version mismatch
# Fix: ensure PyTorch CUDA version matches your driver
```

### Neo4j concept filter returns count=0

The concept seeding script needs to be re-run after every rebuild:
```bash
docker compose exec app python -m scripts.seed_concepts
```

Also ensure the Cypher query in `app/db/neo4j/operations.py` uses:
```cypher
EXISTS { (d)-[:HAS_CONCEPT]->(c:Concept) WHERE c.name IN $concept_names }
```

### Ingestion stalls or all chunks fail

The secondary node is likely throttling due to GPU heat. Signs: latency jumping from ~1s to ~20s per chunk after 30 minutes. Solution: stop ingestion, let the GPU cool for 20 minutes, restart the secondary node, and resume:
```bash
curl -X POST http://localhost:8000/admin/ingest/dir \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"force_reingest": false}'
```

### LLM responses are slow

Normal: first query after startup takes 30–60s while the model loads into VRAM. Subsequent queries are 10–20s. If consistently slow, check GPU utilization:
```bash
watch -n 2 nvidia-smi
```

If VRAM is full, another process may be using the GPU. Qwen2.5-Coder-7B needs ~5GB free VRAM.

### UI shows white page or missing styles

Static files may not be mounted correctly. Verify:
```bash
docker compose exec app ls app/static/
# Should show: app.js  index.html  style.css
```

If missing, the files weren't present during `make build`. Add them and rebuild.

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `MNEMONIC_API_TOKEN` | Yes | Bearer token for all API calls |
| `POSTGRES_USER` | Yes | PostgreSQL username |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `POSTGRES_DB` | Yes | PostgreSQL database name |
| `NEO4J_USER` | Yes | Neo4j username (default: neo4j) |
| `NEO4J_PASSWORD` | Yes | Neo4j password (min 8 chars) |
| `MNEMONIC_USER_NAME` | Yes | Your display name |
| `MNEMONIC_USER_LANG` | No | UI language: EN, PT, DE (default: EN) |
| `SECONDARY_NODE_URL` | No | Secondary GPU node URL |
| `SECONDARY_NODE_TIMEOUT` | No | Timeout in seconds (default: 120.0) |
| `OLLAMA_URL` | No | Ollama API URL (default: http://ollama:11434) |
| `OLLAMA_MODEL` | No | Model name (default: qwen2.5-coder:7b) |
| `PISTON_URL` | No | Piston URL (default: http://piston:2000) |
| `ENVIRONMENT` | No | production or development (default: production) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
