"""
embedder.py
===========
Convert text strings into dense embedding vectors.

Primary:  sentence-transformers all-MiniLM-L6-v2 (local, offline).
Fallback: deterministic hash embeddings (no external dependencies).
"""

from __future__ import annotations

import logging
import os
import hashlib
from threading import Lock
from typing import List

logger = logging.getLogger(__name__)

_st_model = None               # sentence-transformers model (if loaded)
_model_lock = Lock()


def _clean_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Sentence-Transformers fallback
# ---------------------------------------------------------------------------

def _embed_st(texts: List[str]) -> List[List[float]]:
    global _st_model
    if _st_model is None:
        with _model_lock:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer  # type: ignore
                model_name = _clean_env("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
                _st_model = SentenceTransformer(model_name)
                logger.info("Embedder backend: sentence_transformers (%s)", model_name)

    vecs = _st_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vecs]


def _embed_hash(texts: List[str], dim: int = 256) -> List[List[float]]:
    """Deterministic lightweight fallback embedder (no external deps)."""
    vectors: List[List[float]] = []
    for text in texts:
        values = [0.0] * dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            values[idx] += sign

        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        vectors.append(values)

    logger.warning("Embedding fallback active: deterministic hash vectors in use.")
    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of embedding vectors, one per input text."""
    if not texts:
        return []

    try:
        return _embed_st(texts)
    except Exception as exc:
        logger.warning("sentence-transformers embedder failed (%s) — falling back to hash embeddings.", exc)
        return _embed_hash(texts)


def embed_query(text: str) -> List[float]:
    """Convenience wrapper — embed a single query string."""
    return embed_texts([text])[0]
