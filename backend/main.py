"""
main.py
=======
FastAPI application entry point for PharmaGuard.

Run locally:
  cd PharmaGuard
  uvicorn backend.main:app --reload --port 8000

The lifespan handler initialises expensive singletons (ML model, ChromaDB)
once at startup so they are never re-created per request.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv  # type: ignore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env from project root (one level above this file's package)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.api.health import router as health_router
    from backend.api.predict import router as predict_router
except ModuleNotFoundError:
    from api.health import router as health_router
    from api.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise singletons before first request."""
    logger.info("PharmaGuard backend starting up…")

    # 1. Load ML model (once — stays in memory)
    from ml_models.predictor import load_model
    load_model()

    # 2. Initialise ChromaDB client + seed knowledge base if empty
    try:
        from rag_pipeline.knowledge_base import seed_knowledge_base
        seed_knowledge_base()
    except Exception as exc:
        logger.warning("RAG initialisation skipped: %s", exc)

    logger.info("All components initialised. Ready.")
    yield

    logger.info("PharmaGuard backend shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "PharmaGuard API",
        description = (
            "Pharmacogenomic risk prediction — VCF parsing, deep-learning "
            "variant annotation, CPIC risk classification, and RAG-grounded "
            "LLM clinical explanations."
        ),
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        lifespan    = lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.getenv("FRONTEND_URL", "https://pharmaguard.vercel.app"),
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(predict_router)

    return app


app = create_app()
