"""
Recommendation generation layer.

Synthesizes sustainability recommendations from retrieved passages.
Uses OpenAI if OPENAI_API_KEY is set; falls back to an extractive
rule-based summary when no key is available.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are GreenRAG, a sustainability advisor for cloud AI infrastructure.
Given a user's infrastructure description and a set of relevant research passages,
produce 3 to 5 concrete, actionable sustainability recommendations.

Format each recommendation as:
- [RECOMMENDATION]: <one clear action>
  [RATIONALE]: <why it reduces energy or carbon, grounded in the provided passages>
  [SOURCE]: <source filename>

Only cite sources from the provided passages. Do not invent facts.
"""


def _build_context(passages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[{i}] Source: {p['source']} (score: {p['score']})\n{p['text']}")
    return "\n\n".join(parts)


def _openai_generate(description: str, passages: list[dict]) -> list[str]:
    from openai import OpenAI

    client = OpenAI()
    context = _build_context(passages)
    user_msg = (
        f"Infrastructure description:\n{description}\n\n"
        f"Relevant research passages:\n{context}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    text = response.choices[0].message.content or ""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _extractive_generate(description: str, passages: list[dict]) -> list[str]:
    """
    Rule-based fallback: extract the first sentence of each top passage
    and format it as an actionable recommendation bullet.
    """
    recommendations = []
    seen_sources: set[str] = set()

    for p in passages:
        if p["source"] in seen_sources:
            continue
        seen_sources.add(p["source"])

        sentences = [s.strip() for s in p["text"].replace("\n", " ").split(". ") if len(s.strip()) > 40]
        if not sentences:
            continue

        rec = f"- {sentences[0]}. (Source: {p['source']})"
        recommendations.append(rec)

        if len(recommendations) >= 4:
            break

    return recommendations


def generate_recommendations(description: str, passages: list[dict]) -> list[str]:
    """
    Return a list of recommendation strings for the given infrastructure
    description, grounded in the supplied retrieved passages.
    """
    if not passages:
        return ["No relevant passages retrieved; cannot generate recommendations."]

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            logger.info("Generating recommendations via OpenAI")
            return _openai_generate(description, passages)
        except Exception as exc:
            logger.warning("OpenAI generation failed (%s); falling back to extractive", exc)

    logger.info("Generating recommendations via extractive fallback")
    return _extractive_generate(description, passages)
