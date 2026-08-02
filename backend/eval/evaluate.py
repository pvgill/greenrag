"""
Retrieval evaluation — Precision@k comparing dense retrieval vs. BM25 baseline.

Usage (from backend/):
    python -m eval.evaluate

Requires the ChromaDB index to be built first:
    python -c "from rag.pipeline import build_index; build_index()"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi  # noqa: E402
from rag.pipeline import retrieve, ingest_documents  # noqa: E402

TEST_QUERIES: list[dict] = [
    {
        "id": "q1",
        "query": "How can I reduce energy consumption during LLM inference?",
        "relevant_sources": {"inference_energy.txt", "cloud_infrastructure_efficiency.txt"},
    },
    {
        "id": "q2",
        "query": "Which cloud region has the lowest carbon intensity for AI workloads?",
        "relevant_sources": {"cloud_infrastructure_efficiency.txt", "carbon_aware_scheduling.txt"},
    },
    {
        "id": "q3",
        "query": "How do I measure the carbon footprint of a machine learning model?",
        "relevant_sources": {"carbon_footprint_ml.txt", "inference_energy.txt"},
    },
    {
        "id": "q4",
        "query": "What GPU hardware should I choose to minimize energy per inference token?",
        "relevant_sources": {
            "hardware_efficiency_benchmarks.txt",
            "inference_energy.txt",
            "cloud_infrastructure_efficiency.txt",
        },
    },
    {
        "id": "q5",
        "query": "How does retrieval-augmented generation support sustainability assessment?",
        "relevant_sources": {"rag_sustainability_assessment.txt"},
    },
    {
        "id": "q6",
        "query": "How does INT8 quantization reduce energy consumption during inference?",
        "relevant_sources": {"model_quantization.txt", "inference_energy.txt"},
    },
    {
        "id": "q7",
        "query": "What is the energy efficiency difference between A100 and H100 GPUs?",
        "relevant_sources": {"hardware_efficiency_benchmarks.txt"},
    },
    {
        "id": "q8",
        "query": "How can I use carbon intensity data to schedule ML workloads more sustainably?",
        "relevant_sources": {"carbon_aware_scheduling.txt", "carbon_footprint_ml.txt"},
    },
    {
        "id": "q9",
        "query": "What software tools are available to monitor GPU power consumption?",
        "relevant_sources": {"inference_energy.txt", "carbon_footprint_ml.txt"},
    },
    {
        "id": "q10",
        "query": "How does Power Usage Effectiveness affect my cloud AI deployment carbon footprint?",
        "relevant_sources": {
            "inference_energy.txt",
            "cloud_infrastructure_efficiency.txt",
            "carbon_aware_scheduling.txt",
        },
    },
]

K_VALUES = [1, 3, 5]


def precision_at_k(retrieved: list[dict], relevant: set[str], k: int) -> float:
    hits = sum(1 for p in retrieved[:k] if p["source"] in relevant)
    return hits / k


# ---------------------------------------------------------------------------
# BM25 baseline
# ---------------------------------------------------------------------------

def _build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, chunks: list[dict], bm25: BM25Okapi, k: int) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "source": chunks[idx]["source"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": round(float(scores[idx]), 4),
        }
        for idx in ranked
    ]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(k_values: list[int] = K_VALUES) -> dict:
    # Dense retrieval (GreenRAG)
    dense_per_query: list[dict] = []
    for item in TEST_QUERIES:
        passages = retrieve(item["query"], k=max(k_values))
        row: dict = {"id": item["id"], "query": item["query"], "retrieved": []}
        for p in passages:
            row["retrieved"].append(
                {"source": p["source"], "score": p["score"], "relevant": p["source"] in item["relevant_sources"]}
            )
        for k in k_values:
            row[f"P@{k}"] = precision_at_k(passages, item["relevant_sources"], k)
        dense_per_query.append(row)

    # BM25 baseline
    chunks = ingest_documents()
    bm25 = _build_bm25_index(chunks)
    bm25_per_query: list[dict] = []
    for item in TEST_QUERIES:
        passages = bm25_retrieve(item["query"], chunks, bm25, k=max(k_values))
        row = {"id": item["id"], "query": item["query"], "retrieved": []}
        for p in passages:
            row["retrieved"].append(
                {"source": p["source"], "score": p["score"], "relevant": p["source"] in item["relevant_sources"]}
            )
        for k in k_values:
            row[f"P@{k}"] = precision_at_k(passages, item["relevant_sources"], k)
        bm25_per_query.append(row)

    # Aggregate
    results: dict = {"n_queries": len(TEST_QUERIES), "k_values": k_values}
    for k in k_values:
        results[f"dense_P@{k}"] = round(sum(r[f"P@{k}"] for r in dense_per_query) / len(dense_per_query), 4)
        results[f"bm25_P@{k}"] = round(sum(r[f"P@{k}"] for r in bm25_per_query) / len(bm25_per_query), 4)

    results["dense_per_query"] = dense_per_query
    results["bm25_per_query"] = bm25_per_query
    return results


if __name__ == "__main__":
    print("Running retrieval evaluation (dense vs BM25)...")
    output = run_evaluation()
    print(f"\nResults over {output['n_queries']} test queries:")
    print(f"{'Metric':<10} {'Dense':>8} {'BM25':>8}")
    print("-" * 28)
    for k in K_VALUES:
        print(f"P@{k:<8} {output[f'dense_P@{k}']:>8.4f} {output[f'bm25_P@{k}']:>8.4f}")

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to {out_path}")
