"""
HTTP client for the Piston code execution API (self-hosted).

Piston runs sandboxed code execution for Python, C++, and C#.
It is added to docker-compose.yml as the `piston` service.

After first launch, install runtimes once:
  docker compose exec piston piston install python
  docker compose exec piston piston install c++
  docker compose exec piston piston install mono   # for C#

Piston API reference: https://github.com/engineer-man/piston
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

from app.config import settings

log = structlog.get_logger(__name__)

# Language → (piston_name, version_prefix)
_LANGUAGE_MAP = {
    "Python": ("python", "3"),
    "C++":    ("c++", "10"),
    "C#":     ("csharp", "6"),
}
# Execution limits
_TIMEOUT_SECONDS = 10.0
_MEMORY_LIMIT_MB = 128


@dataclass
class PistonResult:
    """Result of a Piston code execution."""
    stdout: str
    stderr: str
    compile_stderr: str
    exit_code: int
    compile_success: bool
    execution_ms: int
    timed_out: bool

    @property
    def success(self) -> bool:
        return self.compile_success and self.exit_code == 0

async def execute(
    code: str,
    programming_language: str,
    stdin: str = "",
) -> PistonResult:
    """
    Executes code in the specified language via Piston.

    Args:
        code:                 The source code to execute.
        programming_language: "Python" | "C++" | "C#"
        stdin:                Optional standard input to pass to the program.

    Returns:
        PistonResult with stdout, stderr, exit code, and timing.
    """
    if programming_language not in _LANGUAGE_MAP:
        return PistonResult(
            stdout="", stderr=f"Unsupported language: {programming_language}",
            compile_stderr="", exit_code=1, compile_success=False,
            execution_ms=0, timed_out=False,
        )

    piston_lang, version_prefix = _LANGUAGE_MAP[programming_language]

    # First, get available runtimes to find exact version
    try:
        version = await _get_runtime_version(piston_lang, version_prefix)
    except Exception as e:
        log.error("piston.runtime_lookup_failed", lang=piston_lang, error=str(e))
        return PistonResult(
            stdout="", stderr=f"Runtime not available: {piston_lang}",
            compile_stderr="", exit_code=1, compile_success=False,
            execution_ms=0, timed_out=False,
        )

    payload = {
    "language": piston_lang,
    "version": version,
    "files": [{"name": _get_filename(programming_language), "content": code}],
    "stdin": stdin,
    "args": [],
}

    import time
    t_start = time.time()

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS + 5) as client:
            response = await client.post(
                f"{settings.PISTON_URL}/api/v2/execute",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException:
        return PistonResult(
            stdout="", stderr="Execution timed out.",
            compile_stderr="", exit_code=1, compile_success=False,
            execution_ms=int(_TIMEOUT_SECONDS * 1000), timed_out=True,
        )
    except Exception as e:
        log.error("piston.execution_failed", error=str(e))
        return PistonResult(
            stdout="", stderr=f"Execution service error: {str(e)}",
            compile_stderr="", exit_code=1, compile_success=False,
            execution_ms=0, timed_out=False,
        )

    execution_ms = int((time.time() - t_start) * 1000)

    compile_data = data.get("compile", {}) or {}
    run_data = data.get("run", {}) or {}

    compile_stderr = compile_data.get("stderr", "") or ""
    compile_exit = compile_data.get("code", 0) or 0
    compile_success = compile_exit == 0 and not compile_stderr

    # For interpreted languages (Python), there's no compile step
    if not compile_data:
        compile_success = True

    stdout = run_data.get("stdout", "") or ""
    stderr = run_data.get("stderr", "") or ""
    exit_code = run_data.get("code", 0) or 0

    log.info(
        "piston.executed",
        language=programming_language,
        exit_code=exit_code,
        compile_success=compile_success,
        execution_ms=execution_ms,
        stdout_len=len(stdout),
    )

    return PistonResult(
        stdout=stdout,
        stderr=stderr,
        compile_stderr=compile_stderr,
        exit_code=exit_code,
        compile_success=compile_success,
        execution_ms=execution_ms,
        timed_out=False,
    )

async def check_piston_health() -> bool:
    """Returns True if Piston is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.PISTON_URL}/api/v2/runtimes")
            return response.status_code == 200
    except Exception:
        return False

async def _get_runtime_version(lang: str, version_prefix: str) -> str:
    """Fetches installed runtimes and returns the best matching version."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{settings.PISTON_URL}/api/v2/runtimes")
        response.raise_for_status()
        runtimes = response.json()

    matches = [
        r["version"] for r in runtimes
        if r["language"] == lang and r["version"].startswith(version_prefix)
    ]

    if not matches:
        # Try any version of this language
        matches = [r["version"] for r in runtimes if r["language"] == lang]

    if not matches:
        raise RuntimeError(f"No runtime found for {lang}. "
        f"Run: docker compose exec piston piston install {lang}")

    return sorted(matches)[-1]  # use latest

def _get_filename(programming_language: str) -> str:
    return {"Python": "solution.py", "C++": "solution.cpp", "C#": "solution.cs"}.get(
        programming_language, "solution.txt"
    )