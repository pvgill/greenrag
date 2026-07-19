"""
RAG pipeline — document ingestion, embedding, and retrieval.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "greenrag_corpus"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest_documents() -> list[dict]:
    """Load .txt files from the corpus directory and split into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    corpus_files = sorted(CORPUS_DIR.glob("*.txt"))
    if not corpus_files:
        logger.warning("No .txt files found in corpus directory: %s", CORPUS_DIR)
        return chunks

    for path in corpus_files:
        text = path.read_text(encoding="utf-8")
        parts = splitter.split_text(text)
        for i, part in enumerate(parts):
            chunk_id = hashlib.md5(f"{path.name}:{i}".encode()).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "text": part,
                    "source": path.name,
                    "chunk_index": i,
                }
            )
        logger.info("Loaded %d chunks from %s", len(parts), path.name)

    logger.info("Ingested %d total chunks from %d files", len(chunks), len(corpus_files))
    return chunks


def build_index() -> int:
    """Embed all corpus chunks and upsert into ChromaDB. Returns number of chunks indexed."""
    chunks = ingest_documents()
    if not chunks:
        return 0

    collection = _get_collection()
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["source"], "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
        logger.info("Upserted batch %d–%d into ChromaDB", start, start + len(batch) - 1)

    logger.info("Index build complete: %d chunks persisted to %s", len(chunks), CHROMA_DIR)
    return len(chunks)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return top-k passages from the index most relevant to the query."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    passages: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append(
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "score": round(1.0 - float(dist), 4),
            }
        )
    return passages
