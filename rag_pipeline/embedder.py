"""
embedder.py
===========
Convert text strings into dense embedding vectors.

Primary:  NVIDIA nv-embedqa-e5-v5 via the NVIDIA API (openai-compatible).
Fallback: sentence-transformers all-MiniLM-L6-v2 (local, offline).

The embedder auto-selects the best available backend at startup and
remains consistent within a single deployment.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_backend: str = "unresolved"   # "nvidia" | "sentence_transformers"
_st_model = None               # sentence-transformers model (if loaded)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _resolve_backend() -> str:
    global _backend
    if _backend != "unresolved":
        return _backend

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        _backend = "nvidia"
    else:
        _backend = "sentence_transformers"

    logger.info("Embedder backend: %s", _backend)
    return _backend


# ---------------------------------------------------------------------------
# NVIDIA embedding (calls /v1/embeddings via openai SDK)
# ---------------------------------------------------------------------------

def _embed_nvidia(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        base_url = os.environ["NVIDIA_BASE_URL"],
        api_key  = os.environ["NVIDIA_API_KEY"],
    )
    model = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

    embeddings: List[List[float]] = []
    # Process in batches of 10 (NVIDIA API limit)
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            embeddings.append(item.embedding)

    return embeddings


# ---------------------------------------------------------------------------
# Sentence-Transformers fallback
# ---------------------------------------------------------------------------

def _embed_st(texts: List[str]) -> List[List[float]]:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("sentence-transformers model loaded: all-MiniLM-L6-v2")

    vecs = _st_model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of embedding vectors, one per input text."""
    backend = _resolve_backend()
    try:
        if backend == "nvidia":
            return _embed_nvidia(texts)
    except Exception as exc:
        logger.warning("NVIDIA embedder failed (%s) — falling back to sentence-transformers.", exc)

    return _embed_st(texts)


def embed_query(text: str) -> List[float]:
    """Convenience wrapper — embed a single query string."""
    return embed_texts([text])[0]
