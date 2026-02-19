"""
retriever.py
============
Semantic retrieval from the ChromaDB pharma_guidelines collection.

Strategy (Hybrid Search):
  1. Dense vector search — semantic similarity via embedding cosine distance.
     Broad recall; recognises that "myopathy" ≅ "muscle damage" ≅ "rhabdomyolysis".

  2. Metadata pre-filtering — filter on gene + drug before executing ANN search.
     Ensures that a Codeine query never retrieves DPYD / Fluorouracil chunks.

  3. BM25 keyword re-ranking — after retrieving top-K dense results, re-rank
     by keyword overlap (simple term-frequency proxy) to guarantee that precise
     identifiers like rs3892097, CYP2D6*4, SLCO1B1*5 are surface-ranked.

Returns: ordered list of raw guideline text strings ready for LLM injection.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from rag_pipeline.chroma_client import get_collection
from rag_pipeline.embedder import embed_query

logger = logging.getLogger(__name__)

_TOP_K_DENSE   = 5   # dense retrieval candidates
_TOP_K_RETURN  = 3   # final chunks returned to LLM


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_context(
    gene: str,
    drug: str,
    phenotype: str,
    diplotype: str,
    rsids: Optional[List[str]] = None,
) -> List[str]:
    """
    Retrieve the most relevant CPIC guideline chunks for a patient profile.

    Parameters
    ----------
    gene       : e.g. "CYP2D6"
    drug       : e.g. "CODEINE"
    phenotype  : e.g. "Poor Metabolizer"
    diplotype  : e.g. "*4/*4"
    rsids      : list of detected variant rsIDs (used for keyword boosting)

    Returns
    -------
    List of raw text strings (up to _TOP_K_RETURN), best first.
    """
    query = _build_query(gene, drug, phenotype, diplotype, rsids or [])
    query_vec = embed_query(query)

    collection = get_collection()
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — returning empty context.")
        return []

    # ── Dense retrieval with metadata filter ──────────────────────────────
    where_filter = _build_filter(gene, drug)
    primary_filter = _build_primary_filter(gene, drug)

    results = {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    # 1) Prefer authoritative guideline sources first
    try:
        results = collection.query(
            query_embeddings  = [query_vec],
            n_results         = min(_TOP_K_DENSE, collection.count()),
            where             = primary_filter,
            include           = ["documents", "distances", "metadatas"],
        )
    except Exception as exc:
        logger.warning("Primary filtered query failed (%s) — retrying with broad filter.", exc)

    docs_primary: List[str] = results.get("documents", [[]])[0] if results.get("documents") else []

    # 2) Fall back to broader gene/drug scope when needed
    if not docs_primary:
        results = collection.query(
            query_embeddings  = [query_vec],
            n_results         = min(_TOP_K_DENSE, collection.count()),
            where             = where_filter,
            include           = ["documents", "distances", "metadatas"],
        )

    docs:      List[str]             = results.get("documents", [[]])[0] if results.get("documents") else []
    distances: List[float]           = results.get("distances", [[]])[0] if results.get("distances") else []
    metadatas: List[Dict[str, str]]  = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    if not docs:
        return []

    # ── BM25-style keyword re-ranking ─────────────────────────────────────
    keywords = _extract_keywords(gene, drug, diplotype, rsids or [])
    scored   = _keyword_rerank(docs, distances, metadatas, keywords)

    top_chunks = [doc for doc, _ in scored[:_TOP_K_RETURN]]

    logger.debug(
        "RAG retrieved %d chunks for (%s, %s, %s).",
        len(top_chunks), gene, drug, phenotype,
    )

    return top_chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_query(
    gene: str,
    drug: str,
    phenotype: str,
    diplotype: str,
    rsids: List[str],
) -> str:
    rsid_str = ", ".join(rsids) if rsids else "no rsIDs"
    return (
        f"Clinical pharmacogenomics recommendation for {gene} {drug}. "
        f"Patient phenotype: {phenotype}. Diplotype: {diplotype}. "
        f"Detected variants: {rsid_str}. "
        f"CPIC guideline dosing recommendation and biological mechanism."
    )


def _build_filter(gene: str, drug: str) -> Dict:
    """
    ChromaDB $or filter: match either the gene OR the drug metadata field.
    Broad enough to retrieve cross-gene chunks but avoids completely unrelated entries.
    """
    return {
        "$or": [
            {"gene":  {"$eq": gene}},
            {"drug":  {"$eq": drug.upper()}},
        ]
    }


def _build_primary_filter(gene: str, drug: str) -> Dict:
    """
    Prefer authoritative guideline sources (CPIC/PharmGKB) for first-pass retrieval.
    """
    drug_upper = drug.upper()
    return {
        "$or": [
            {"gene": {"$eq": gene}, "source_type": {"$eq": "CPIC_Guideline"}},
            {"drug": {"$eq": drug_upper}, "source_type": {"$eq": "CPIC_Guideline"}},
            {"gene": {"$eq": gene}, "source_type": {"$eq": "PharmGKB"}},
            {"drug": {"$eq": drug_upper}, "source_type": {"$eq": "PharmGKB"}},
        ]
    }


def _extract_keywords(
    gene: str,
    drug: str,
    diplotype: str,
    rsids: List[str],
) -> List[str]:
    """Build a list of high-value keywords for BM25 term-frequency re-ranking."""
    kw = [gene, drug.upper(), drug.lower(), diplotype]
    kw += rsids
    # Extract individual star alleles from diplotype (e.g. "*4/*4" → ["*4"])
    stars = re.findall(r"\*[\dA-Za-z]+", diplotype)
    kw   += stars
    return [k for k in kw if k]


def _keyword_rerank(
    docs: List[str],
    distances: List[float],
    metadatas: List[Dict[str, str]],
    keywords: List[str],
) -> List[tuple]:
    """
    Combine cosine distance with a simple keyword-hit boost.

    Score = (1 - cosine_distance) + 0.1 * keyword_hits
    Higher is better.
    """
    kw_lower = [k.lower() for k in keywords]
    scored = []
    for idx, (doc, dist) in enumerate(zip(docs, distances)):
        doc_lower = doc.lower()
        hits  = sum(1 for kw in kw_lower if kw in doc_lower)
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        source_type = str(metadata.get("source_type", "")).upper()

        source_boost = 0.0
        if source_type == "CPIC_GUIDELINE":
            source_boost = 0.60
        elif source_type == "PHARMGKB":
            source_boost = 0.40
        elif source_type.startswith("UPLOADED_"):
            source_boost = -0.80

        score = (1.0 - dist) + (0.08 * hits) + source_boost
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
