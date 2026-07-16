"""
Async wrapper around the Ollama API.

All agents call this module instead of hitting Ollama directly.
Handles retries, JSON mode, and response parsing in one place.

Two main functions:
  chat() — sends a prompt, returns the full text response
  chat_json() — sends a prompt, returns a parsed dict (JSON mode)

Both functions accept a list of messages in OpenAI format:
  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

log = structlog.get_logger(__name__)

#  Core chat function 

@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def chat(
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> str:
    """
    Sends a chat prompt to Ollama and returns the response as a string.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": str}
        temperature: Sampling temperature. Use 0.1 for deterministic agent tasks, 0.7 for more creative synthesis.
        max_tokens: Maximum tokens to generate.
        timeout: HTTP timeout in seconds. Increase for long generations.
    Returns:
        The model's response as a plain string.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]

        log.debug(
            "llm.chat_complete",
            model=settings.OLLAMA_MODEL,
            prompt_tokens=_count_prompt_tokens(messages),
            response_len=len(content),
        )
        return content

@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def chat_json(
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Sends a chat prompt to Ollama in JSON mode and returns a parsed dict.

    Ollama's JSON mode (format="json") forces the model to output valid JSON.
    Use this for all structured agent outputs (routing decisions, citations, grades).

    Args:
        messages: Same format as chat(). The system prompt MUST instruct the
            model to respond only with JSON — Ollama enforces the format
            but not the schema.
        temperature: Keep at 0.1 for structured outputs.
        max_tokens: JSON outputs are typically short; 1024 is usually enough.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed dict from the model's JSON response.

    Raises:
        ValueError if the response cannot be parsed as JSON even after cleanup.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "format": "json",           # forces valid JSON output
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        raw = data["message"]["content"]

    return _parse_json_response(raw)

#  Health check 

async def check_ollama_health() -> bool:
    """
    Returns True if Ollama is reachable and the configured model is available.
    Called at startup and by the health endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # List available models
            response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            model_available = any(
                settings.OLLAMA_MODEL in m for m in models
            )
            if not model_available:
                log.warning(
                    "ollama.model_not_found",
                    model=settings.OLLAMA_MODEL,
                    available=models,
                    hint="Run: docker compose exec ollama ollama pull qwen2.5-coder:7b",
                )
            else:
                log.info("ollama.ready", model=settings.OLLAMA_MODEL)
            return model_available
    except Exception as e:
        log.error("ollama.unreachable", error=str(e))
        return False

#  Helpers 

def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    Parses a JSON response from the model, stripping markdown fences if present.
    Falls back to returning {"raw": raw} if parsing fails completely.
    """
    # Strip markdown code fences (```json ... ```)
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract a JSON object from within a larger string
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    log.warning("llm.json_parse_failed", raw_preview=raw[:200])
    raise ValueError(f"Could not parse JSON from model response: {raw[:300]}")

def _count_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate for logging — 1 token ≈ 4 characters."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4

#  Convenience builders 

def system_user(system: str, user: str) -> list[dict[str, str]]:
    """
    Shorthand for the most common message pattern: one system prompt + one user message.

    Usage:
        messages = system_user(
            system="You are a ...",
            user="Classify this: ..."
        )
        result = await chat_json(messages)
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]