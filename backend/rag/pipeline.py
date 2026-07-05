"""
RAG pipeline — document ingestion, embedding, and retrieval.
"""
from __future__ import annotations

from pathlib import Path


CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def ingest_documents():
    """Load and chunk documents from the corpus directory."""
    raise NotImplementedError("Ingestion pipeline coming in next iteration.")


def build_index():
    """Embed chunks and persist to ChromaDB."""
    raise NotImplementedError("Index build coming in next iteration.")


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Retrieve top-k passages relevant to the infrastructure description query."""
    raise NotImplementedError("Retrieval coming in next iteration.")
