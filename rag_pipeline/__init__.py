"""
rag_pipeline — Retrieval-Augmented Generation pipeline.

Components:
  chroma_client   — persistent ChromaDB connection
  knowledge_base  — seeds CPIC guideline chunks on first run
  embedder        — text → vector (local sentence-transformers + hash fallback)
  retriever       — semantic search over the guideline collection
  llm_engine      — NVIDIA GLM5 streaming explanation generator
"""
