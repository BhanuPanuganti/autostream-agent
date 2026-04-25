"""
rag.py — Local knowledge base loader and retrieval for AutoStream agent.

Uses sentence-transformer embeddings + FAISS for semantic search.
Falls back to keyword scoring if the vector store is unavailable.
"""

import json
import os
import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")


def _load_kb() -> dict:
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(
            f"Knowledge base not found at {KB_PATH}. "
            "Ensure autostream_kb.json is in the knowledge_base/ directory."
        )
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_kb(kb: dict) -> List[Tuple[str, str]]:
    """Convert structured KB into (topic, text) pairs for indexing."""
    chunks: List[Tuple[str, str]] = []

    co = kb.get("company", {})
    if co:
        chunks.append((
            "company_overview",
            f"{co.get('name', '')}: {co.get('tagline', '')} {co.get('description', '')}",
        ))

    for plan in kb.get("plans", []):
        feature_str = "; ".join(plan.get("features", []))
        chunks.append((
            plan["name"].lower().replace(" ", "_"),
            f"{plan['name']}: ${plan['price_monthly']}/month. Features: {feature_str}.",
        ))

    for policy in kb.get("policies", []):
        chunks.append((
            f"policy_{policy['topic'].lower().replace(' ', '_')}",
            f"{policy['topic']}: {policy['detail']}",
        ))

    for faq in kb.get("faqs", []):
        chunks.append(("faq", f"Q: {faq['question']} A: {faq['answer']}"))

    return chunks


# ---------------------------------------------------------------------------
# Keyword fallback scorer (used when vector store unavailable)
# ---------------------------------------------------------------------------

def _score_chunk(query: str, text: str) -> int:
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    return len(query_words & text_words)


def _keyword_retrieve(query: str, chunks: List[Tuple[str, str]], top_k: int = 3) -> List[str]:
    scored = [(text, _score_chunk(query, text)) for _, text in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, score in scored[:top_k] if score > 0]


# ---------------------------------------------------------------------------
# Vector store (lazy init)
# ---------------------------------------------------------------------------

_store = None
_chunks: List[Tuple[str, str]] = []
_initialized = False
_init_failed = False


def _init_vector_store() -> None:
    global _store, _chunks, _initialized, _init_failed

    if _initialized or _init_failed:
        return

    try:
        from vector_store import VectorStore
        kb = _load_kb()
        _chunks = _flatten_kb(kb)
        texts = [text for _, text in _chunks]

        _store = VectorStore()
        _store.build(texts)
        _initialized = True
        logger.info("Vector store initialized with %d chunks.", len(_chunks))
    except FileNotFoundError as e:
        logger.error("KB file missing: %s", e)
        _init_failed = True
    except ImportError as e:
        logger.warning("VectorStore unavailable (%s), falling back to keyword search.", e)
        _init_failed = True
        try:
            kb = _load_kb()
            _chunks = _flatten_kb(kb)
        except Exception:
            pass
    except Exception as e:
        logger.error("Vector store init failed: %s", e)
        _init_failed = True
        try:
            kb = _load_kb()
            _chunks = _flatten_kb(kb)
        except Exception:
            pass


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Return the top_k most relevant KB passages for a query.
    Uses semantic search if available, falls back to keyword matching.
    """
    _init_vector_store()

    if not _chunks:
        return "No relevant information found."

    try:
        if _store is not None and _initialized:
            results = _store.search(query, k=top_k)
        else:
            results = _keyword_retrieve(query, _chunks, top_k=top_k)

        if not results:
            return "No relevant information found."
        return "\n".join(f"• {r}" for r in results)
    except Exception as e:
        logger.warning("Retrieval error (%s), falling back to keyword search.", e)
        try:
            results = _keyword_retrieve(query, _chunks, top_k=top_k)
            return "\n".join(f"• {r}" for r in results) if results else "No relevant information found."
        except Exception:
            return "No relevant information found."


def get_full_kb_summary() -> str:
    """Return all KB content as a flat string — used as system prompt fallback."""
    _init_vector_store()
    if not _chunks:
        try:
            kb = _load_kb()
            chunks = _flatten_kb(kb)
        except Exception:
            return "Knowledge base unavailable."
    else:
        chunks = _chunks

    lines = ["=== AutoStream Knowledge Base ==="]
    for _, text in chunks:
        lines.append(f"• {text}")
    return "\n".join(lines)