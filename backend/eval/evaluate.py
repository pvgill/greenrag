"""
Retrieval evaluation — Precision@k on a set of manually judged test queries.

Usage (from backend/):
    python -m eval.evaluate

Requires the ChromaDB index to already be built:
    python -c "from rag.pipeline import build_index; build_index()"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from backend/ without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import retrieve  # noqa: E402

# ---------------------------------------------------------------------------
# Test queries with manually assigned relevance judgments.
# A source file is "relevant" for a query if it directly addresses the topic.
# ---------------------------------------------------------------------------
TEST_QUERIES: list[dict] = [
    {
        "id": "q1",
        "query": "How can I reduce energy consumption during LLM inference?",
        "relevant_sources": {
            "inference_energy.txt",
            "cloud_infrastructure_efficiency.txt",
        },
    },
    {
        "id": "q2",
        "query": "Which cloud region has the lowest carbon intensity for AI workloads?",
        "relevant_sources": {
            "cloud_infrastructure_efficiency.txt",
        },
    },
    {
        "id": "q3",
        "query": "How do I measure the carbon footprint of a machine learning model?",
        "relevant_sources": {
            "carbon_footprint_ml.txt",
            "inference_energy.txt",
        },
    },
    {
        "id": "q4",
        "query": "What GPU hardware should I choose to minimize energy per inference token?",
        "relevant_sources": {
            "inference_energy.txt",
            "cloud_infrastructure_efficiency.txt",
            "carbon_footprint_ml.txt",
        },
    },
    {
        "id": "q5",
        "query": "How does retrieval-augmented generation support sustainability assessment?",
        "relevant_sources": {
            "rag_sustainability_assessment.txt",
        },
    },
]

K_VALUES = [1, 3, 5]


def precision_at_k(retrieved: list[dict], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for p in top_k if p["source"] in relevant)
    return hits / k


def run_evaluation(k_values: list[int] = K_VALUES) -> dict:
    results = {}
    per_query: list[dict] = []

    for item in TEST_QUERIES:
        passages = retrieve(item["query"], k=max(k_values))
        row: dict = {"id": item["id"], "query": item["query"], "retrieved": []}
        for p in passages:
            row["retrieved"].append(
                {
                    "source": p["source"],
                    "score": p["score"],
                    "relevant": p["source"] in item["relevant_sources"],
                }
            )
        for k in k_values:
            row[f"P@{k}"] = precision_at_k(passages, item["relevant_sources"], k)
        per_query.append(row)

    for k in k_values:
        avg = sum(r[f"P@{k}"] for r in per_query) / len(per_query)
        results[f"P@{k}"] = round(avg, 4)

    results["per_query"] = per_query
    results["n_queries"] = len(TEST_QUERIES)
    results["k_values"] = k_values
    return results


if __name__ == "__main__":
    print("Running retrieval evaluation…")
    output = run_evaluation()

    print(f"\nResults over {output['n_queries']} test queries:")
    for k in K_VALUES:
        print(f"  P@{k} = {output[f'P@{k}']:.4f}")

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to {out_path}")
