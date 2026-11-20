"""
athena.web
==========
Browser test harness for the Athena validator (FastAPI). Optional: install the
`web` extra (`pip install "kv-ai-athena-py[web]"` / `uv sync --extra web`).

The core library never imports this package, so `import athena` keeps working
without FastAPI installed.
"""

from __future__ import annotations


def require_web_extra() -> None:
    """Raise a clear error when the optional web dependencies are missing."""
    try:
        import dotenv  # noqa: F401
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via CLI test
        raise ImportError(
            "The Athena web UI needs the 'web' extra: "
            "pip install 'kv-ai-athena-py[web]'  (or: uv sync --extra web)"
        ) from exc
