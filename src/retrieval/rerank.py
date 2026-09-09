"""Cheap lexical + intent reranker on top of cosine neighbors."""

from __future__ import annotations

import re
from typing import Any

TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if len(t) > 2}


def overlap_score(query: str, doc: str) -> float:
    q, d = _tokens(query), _tokens(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    predicted_intent: str | None = None,
    intent_bonus: float = 0.08,
    overlap_weight: float = 0.25,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for hit in hits:
        sim = float(hit.get("similarity") or 0.0)
        overlap = overlap_score(query, hit.get("customer_message") or "")
        bonus = intent_bonus if predicted_intent and hit.get("intent") == predicted_intent else 0.0
        combined = sim + overlap_weight * overlap + bonus
        item = dict(hit)
        item["similarity"] = sim
        item["rerank_score"] = round(float(combined), 4)
        item["overlap"] = round(overlap, 4)
        scored.append(item)
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored
