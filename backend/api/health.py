"""
api/health.py
=============
GET /api/health — liveness and readiness probe for the PharmaGuard backend.
"""

from fastapi import APIRouter

from ml_models.predictor import is_loaded, is_demo_mode

router = APIRouter()


@router.get("/api/health")
async def health_check():
    """Return service status and component readiness flags."""
    try:
        from rag_pipeline.chroma_client import get_collection
        chroma_count = get_collection().count()
        chroma_ready = chroma_count > 0
    except Exception:
        chroma_ready = False
        chroma_count = 0

    return {
        "status":              "ok",
        "model_loaded":        is_loaded(),
        "model_demo_mode":     is_demo_mode() if is_loaded() else True,
        "chroma_ready":        chroma_ready,
        "chroma_doc_count":    chroma_count,
        "api_version":         "1.0.0",
    }
