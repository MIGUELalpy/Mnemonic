"""
Async HTTP client for the secondary node embedding service.
Retries on transient failures. Processes chunks in batches concurrently.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.ingestion.chunker import Chunk

log = structlog.get_logger(__name__)

class EmbeddingError(Exception):
    pass

class SecondaryNodeClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SecondaryNodeClient":
        self._client = httpx.AsyncClient(
            base_url=settings.SECONDARY_NODE_URL,
            timeout=settings.SECONDARY_NODE_TIMEOUT,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def health_check(self) -> dict:
        if not self._client:
            raise EmbeddingError("Client not initialized.")
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        if not self._client:
            raise EmbeddingError("Client not initialized.")
        response = await self._client.post("/embed", json={"text": text, "normalize": True})
        response.raise_for_status()
        return response.json()["vector"]

    async def embed_batch(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        results: list[tuple[Chunk, list[float]]] = []
        t_start = time.time()

        for i, chunk in enumerate(chunks):
            vector = await self._embed_chunk_safe(chunk)
            if vector is not None:
                results.append((chunk, vector))
            
            if (i + 1) % 10 == 0:
                log.info("embedder.progress",
                         processed=i+1, total=len(chunks),
                         embedded=len(results),
                         elapsed_s=round(time.time() - t_start, 1))

        log.info("embedder.all_complete", total_chunks=len(chunks),
                 embedded=len(results),
                 total_elapsed_s=round(time.time() - t_start, 1))
        return results

    async def _embed_chunk_safe(self, chunk: Chunk) -> list[float] | None:
        try:
            return await self.embed_text(chunk.text)
        except Exception as e:
            log.error("embedder.chunk_failed", chunk_index=chunk.chunk_index, error=str(e))
            return None

async def check_secondary_node_health() -> bool:
    try:
        async with SecondaryNodeClient() as client:
            health = await client.health_check()
            is_ok = health.get("status") == "ok"
            log.info("secondary_node.health", status=health.get("status"), models=health.get("models_loaded"), device=health.get("device"))
            return is_ok
    except Exception as e:
        log.error("secondary_node.unreachable", error=str(e))
        return False