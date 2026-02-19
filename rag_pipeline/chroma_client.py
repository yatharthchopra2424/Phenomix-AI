"""
chroma_client.py
================
Initialise a persistent vector-store client and expose the pharma_guidelines
collection used by the RAG pipeline.

Primary backend: ChromaDB PersistentClient.
Fallback backend: local JSON-persisted cosine-similarity store (Python only),
used automatically when ChromaDB is unavailable/incompatible (e.g. Python 3.14).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "pharma_guidelines"
_client: Optional[Any] = None
_collection: Optional[Any] = None
_backend: str = "unresolved"  # chroma | local


def _get_persist_dir() -> str:
    """Resolve the vector-store persistence directory from the environment."""
    path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _get_local_store_file() -> str:
    return str(Path(_get_persist_dir()) / "fallback_vector_store.json")


def _cosine_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0

    similarity = dot / (na * nb)
    similarity = max(min(similarity, 1.0), -1.0)
    return 1.0 - similarity


def _metadata_matches(metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True

    if "$or" in where:
        clauses = where.get("$or", [])
        return any(_metadata_matches(metadata, clause) for clause in clauses)

    for key, condition in where.items():
        if isinstance(condition, dict) and "$eq" in condition:
            if str(metadata.get(key)) != str(condition["$eq"]):
                return False
        else:
            if str(metadata.get(key)) != str(condition):
                return False
    return True


class _LocalCollection:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._rows: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._file_path):
            self._rows = []
            return
        try:
            with open(self._file_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            self._rows = rows if isinstance(rows, list) else []
        except Exception:
            self._rows = []

    def _save(self) -> None:
        payload = {"rows": self._rows}
        with open(self._file_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    def count(self) -> int:
        return len(self._rows)

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        existing_by_id = {row.get("id"): i for i, row in enumerate(self._rows)}

        for idx, row_id in enumerate(ids):
            row = {
                "id": row_id,
                "embedding": embeddings[idx],
                "document": documents[idx],
                "metadata": metadatas[idx],
            }

            if row_id in existing_by_id:
                self._rows[existing_by_id[row_id]] = row
            else:
                self._rows.append(row)

        self._save()

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[List[Any]]]:
        if not query_embeddings:
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

        query_vec = query_embeddings[0]
        candidates = [
            row for row in self._rows
            if _metadata_matches(row.get("metadata", {}), where)
        ]

        ranked = sorted(
            candidates,
            key=lambda row: _cosine_distance(query_vec, row.get("embedding", [])),
        )

        top = ranked[: max(0, n_results)]
        distances = [_cosine_distance(query_vec, row.get("embedding", [])) for row in top]
        docs = [row.get("document", "") for row in top]
        metas = [row.get("metadata", {}) for row in top]

        return {
            "documents": [docs],
            "distances": [distances],
            "metadatas": [metas],
        }


class _LocalClient:
    def __init__(self, persist_dir: str):
        self._persist_dir = persist_dir

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> _LocalCollection:
        _ = metadata
        file_path = str(Path(self._persist_dir) / f"{name}.fallback.json")
        return _LocalCollection(file_path)

    def delete_collection(self, name: str) -> None:
        file_path = str(Path(self._persist_dir) / f"{name}.fallback.json")
        if os.path.exists(file_path):
            os.remove(file_path)


def get_client() -> Any:
    """Return (or lazily create) the singleton vector-store client."""
    global _client
    global _backend
    if _client is None:
        persist_dir = _get_persist_dir()
        try:
            import chromadb  # type: ignore

            _client = chromadb.PersistentClient(path=persist_dir)
            _backend = "chroma"
            logger.info("Vector store backend: chroma (%s)", persist_dir)
        except Exception as exc:
            _client = _LocalClient(persist_dir)
            _backend = "local"
            logger.warning("Vector store backend fallback: local (%s)", exc)
    return _client


def get_collection() -> Any:
    """
    Return (or create) the pharma_guidelines collection.

    If the collection does not yet exist, it is created with cosine distance
    metric for Chroma; local backend uses built-in cosine ranking.
    """
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name     = _COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector collection '%s' ready (%d documents) [backend=%s].",
            _COLLECTION_NAME,
            _collection.count(),
            _backend,
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
