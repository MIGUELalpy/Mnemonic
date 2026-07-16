# Mnemonic — Makefile
#
# Convenience wrapper around docker compose commands.
# All targets assume you are in the project root directory.
#
# Quick start:
#   make setup    — copy .env.example → .env (then edit .env)
#   make build    — build the Docker image
#   make up       — start all services in the background
#   make init     — bootstrap all databases (run once after first `make up`)
#   make logs     — tail live logs from the API
#   make down     — stop all services

.PHONY: help setup build up down restart init logs logs-worker \
        shell shell-worker ps health db-shell neo4j-shell \
        backup clean nuke

# Default target — print help
help:
	@echo ""
	@echo "  Mnemonic — Docker management commands"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup         Copy .env.example to .env (edit before first run)"
	@echo "    make build         Build the application Docker image"
	@echo ""
	@echo "  LIFECYCLE"
	@echo "    make up            Start all services in the background"
	@echo "    make down          Stop all services (data volumes preserved)"
	@echo "    make restart       Restart the app and worker services"
	@echo "    make init          Bootstrap databases (run once after first 'make up')"
	@echo ""
	@echo "  OBSERVABILITY"
	@echo "    make logs          Tail API logs"
	@echo "    make logs-worker   Tail Celery worker logs"
	@echo "    make ps            Show running container status"
	@echo "    make health        Call the /health endpoint"
	@echo ""
	@echo "  DEBUG"
	@echo "    make shell         Open a bash shell inside the app container"
	@echo "    make shell-worker  Open a bash shell inside the worker container"
	@echo "    make db-shell      Open a psql prompt in the PostgreSQL container"
	@echo "    make neo4j-shell   Open a cypher-shell prompt in the Neo4j container"
	@echo ""
	@echo "  MAINTENANCE"
	@echo "    make backup        Dump PostgreSQL database to ./backups/"
	@echo "    make clean         Remove stopped containers and dangling images"
	@echo "    make nuke          DANGER: stop everything and delete all volumes"
	@echo ""


#  Setup 

setup:
	@if [ -f .env ]; then \
		echo "  .env already exists — skipping. Edit it directly if needed."; \
	else \
		cp .env.example .env; \
		echo "  .env created. Fill in all CHANGE_ME values before running 'make up'."; \
	fi

build:
	docker compose build --no-cache

build-fast:
	docker compose build


#  Lifecycle 

up:
	docker compose up -d
	@echo ""
	@echo "  Services started. Run 'make init' if this is the first launch."
	@echo "  API available at http://localhost:8000"
	@echo "  Run 'make health' to verify all services are ready."

down:
	docker compose down

restart:
	docker compose restart app worker

# Bootstrap all databases — idempotent, safe to run multiple times.
init:
	docker compose run --rm init
	@echo ""
	@echo "  Database bootstrap complete."


#  Observability 

logs:
	docker compose logs -f app

logs-worker:
	docker compose logs -f worker

logs-all:
	docker compose logs -f

ps:
	docker compose ps

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool


#  Debug shells 

shell:
	docker compose exec app bash

shell-worker:
	docker compose exec worker bash

db-shell:
	@source .env && docker compose exec postgres \
		psql -U $$POSTGRES_USER -d $$POSTGRES_DB

neo4j-shell:
	@source .env && docker compose exec neo4j \
		cypher-shell -u $$NEO4J_USER -p $$NEO4J_PASSWORD


#  Maintenance 

backup:
	@mkdir -p backups
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	source .env && docker compose exec -T postgres \
		pg_dump -U $$POSTGRES_USER $$POSTGRES_DB \
		> backups/mnemonic_$$TIMESTAMP.sql; \
	echo "  Backup saved to backups/mnemonic_$$TIMESTAMP.sql"

clean:
	docker compose down --remove-orphans
	docker image prune -f

# DANGER: deletes all volumes (PostgreSQL data, Neo4j graph, Qdrant vectors, Redis queue). Use only to completely reset the environment.
nuke:
	@echo "WARNING: This will delete ALL Mnemonic data including your knowledge base."
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	docker compose down -v --remove-orphans
	@echo "  All volumes deleted."
