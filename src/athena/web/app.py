"""
athena.web.app
==============
FastAPI app: a test harness where you paste an LLM query and its response and
get Athena's epistemic assessment of that response.

Run (after installing the `web` extra):

    export GEMINI_API_KEY=...            # or put it in .env
    athena serve                         # == python -m athena.web
                                         # == uvicorn athena.web.app:app

then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import ValidationError
from pydantic_ai.models import Model

from athena import ActionClass, AthenaError, ConsequenceClass
from athena.core.risk.risk_level import derive_risk_level
from athena.web.llm import ModelNotConfigured, build_gemini_model
from athena.web.schemas import ValidateRequest
from athena.web.service import run_validation
from athena.web.settings import Settings

STATIC = Path(__file__).parent / "static"

ModelFactory = Callable[[str | None], Model]


def create_app(
    settings: Settings | None = None,
    model_factory: ModelFactory | None = None,
) -> FastAPI:
    """
    `model_factory(model_name_or_None) -> Model` is injectable so tests (and
    offline demos) can run the real pipeline against a scripted model.
    """
    cfg = settings or Settings.from_env()
    factory = model_factory or (lambda name: build_gemini_model(cfg, name))

    app = FastAPI(title="Athena epistemic test harness", version="0.1.1")
    app.state.settings = cfg
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model": cfg.model,
            "key_configured": cfg.key_configured or model_factory is not None,
            "max_chars": cfg.max_chars,
        }

    @app.get("/api/risk")
    async def risk(action_class: ActionClass, consequence_class: ConsequenceClass) -> dict:
        return {"risk_level": str(derive_risk_level(action_class, consequence_class))}

    @app.post("/api/validate")
    async def validate(req: ValidateRequest) -> dict:
        for name, text in (("query", req.query), ("response", req.response)):
            if len(text) > cfg.max_chars:
                raise HTTPException(413, f"'{name}' exceeds {cfg.max_chars} characters.")
        try:
            model = factory(req.model)
        except ModelNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            return await run_validation(req, model, cfg.timeout_s)
        except TimeoutError as exc:
            raise HTTPException(504, f"Validation exceeded {cfg.timeout_s:.0f}s.") from exc
        except ValidationError as exc:
            raise HTTPException(422, f"Invalid input for Athena: {exc.errors()[0]['msg']}") from exc
        except AthenaError as exc:
            logger.warning("Athena pipeline failure: {}", exc)
            raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # provider/network faults
            logger.exception("Unexpected failure during validation")
            raise HTTPException(502, f"Model call failed: {type(exc).__name__}: {exc}") from exc

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        return JSONResponse({"error": f"{where}: {first['msg']}"}, status_code=422)

    return app


app = create_app()
