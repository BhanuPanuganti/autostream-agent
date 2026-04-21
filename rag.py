"""
rag.py — Local knowledge base loader and retrieval for AutoStream agent.

Uses a simple TF-IDF style keyword matching over the JSON knowledge base.
No external vector DB required — fully local and dependency-light.
"""

import json
import os
import re
from typing import List, Tuple

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")


def _load_kb() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _flatten_kb(kb: dict) -> List[Tuple[str, str]]:
    """
    Convert the structured KB into a list of (topic, text) pairs
    that can be searched against a user query.
    """
    chunks: List[Tuple[str, str]] = []

    # Company overview
    co = kb["company"]
    chunks.append(("company_overview", f"{co['name']}: {co['tagline']} {co['description']}"))

    # Plans
    for plan in kb["plans"]:
        feature_str = "; ".join(plan["features"])
        text = (
            f"{plan['name']}: ${plan['price_monthly']}/month. "
            f"Features: {feature_str}."
        )
        chunks.append((plan["name"].lower().replace(" ", "_"), text))

    # Policies
    for policy in kb["policies"]:
        chunks.append(
            (f"policy_{policy['topic'].lower().replace(' ', '_')}",
             f"{policy['topic']}: {policy['detail']}")
        )

    # FAQs
    for faq in kb["faqs"]:
        chunks.append((f"faq", f"Q: {faq['question']} A: {faq['answer']}"))

    return chunks


def _score_chunk(query: str, text: str) -> int:
    """Simple keyword overlap score."""
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    return len(query_words & text_words)


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant knowledge base chunks for a given query.
    Returns a formatted context string ready to inject into the LLM prompt.
    """
    kb = _load_kb()
    chunks = _flatten_kb(kb)

    scored = [(score, topic, text)
              for topic, text in chunks
              if (score := _score_chunk(query, text)) > 0]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return "No relevant information found in the knowledge base."

    result_lines = ["=== AutoStream Knowledge Base (relevant excerpts) ==="]
    for _, topic, text in top:
        result_lines.append(f"• {text}")
    return "\n".join(result_lines)


def get_full_kb_summary() -> str:
    """Return a full summary of the KB for system prompt injection."""
    kb = _load_kb()
    chunks = _flatten_kb(kb)
    lines = ["=== AutoStream Complete Knowledge Base ==="]
    for _, text in chunks:
        lines.append(f"• {text}")
    return "\n".join(lines)
