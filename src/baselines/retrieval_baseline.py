"""Simple retrieval baseline: copy the nearest historical brand reply."""

from __future__ import annotations

from typing import Any

from src.agent.retriever import HistoricalRetriever


class RetrievalBaseline:
    def __init__(self, retriever: HistoricalRetriever | None = None) -> None:
        self.retriever = retriever

    def _get(self) -> HistoricalRetriever:
        if self.retriever is None:
            self.retriever = HistoricalRetriever()
        return self.retriever

    def handle(self, message: str, *, intent: str | None = None) -> dict[str, Any]:
        hits = self._get().search(message, k=1, intent=intent)
        if not hits:
            return {
                "reply": "Thanks for reaching out. We will look into this.",
                "evidence": [],
                "grounded": False,
            }
        hit = hits[0]
        return {
            "reply": hit["brand_response"],
            "evidence": hits,
            "grounded": float(hit["similarity"]) >= 0.12,
        }
