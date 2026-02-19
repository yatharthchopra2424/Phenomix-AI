"""
chroma_client.py
================
Initialise a persistent ChromaDB client and expose the pharma_guidelines
collection used by the RAG pipeline.

The client is created once (module-level singleton) and reused across
all request handler calls via get_collection().
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb import Collection

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "pharma_guidelines"
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Collection] = None


def _get_persist_dir() -> str:
    """Resolve the ChromaDB persistence directory from the environment."""
    path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_client() -> chromadb.PersistentClient:
    """Return (or lazily create) the singleton ChromaDB persistent client."""
    global _client
    if _client is None:
        persist_dir = _get_persist_dir()
        _client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialised at: %s", persist_dir)
    return _client


def get_collection() -> Collection:
    """
    Return (or create) the pharma_guidelines collection.

    If the collection does not yet exist, it is created with cosine distance
    metric — optimal for text embedding similarity search.
    """
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name     = _COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents).",
            _COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


def collection_is_empty() -> bool:
    """Return True if the pharma_guidelines collection has no documents."""
    try:
        return get_collection().count() == 0
    except Exception:
        return True


def reset_collection() -> None:
    """Drop and recreate the pharma_guidelines collection (use with caution)."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    get_collection()
    logger.info("Collection '%s' reset.", _COLLECTION_NAME)
