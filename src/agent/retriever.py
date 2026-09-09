"""Agent-facing wrapper around the retrieval package."""

from __future__ import annotations

from typing import Any

from src.retrieval.index import RetrievalIndex, load_index
from src.retrieval.retrieve import retrieve as retrieve_hits


class HistoricalRetriever:
    def __init__(self, index: RetrievalIndex | None = None):
        self.index = index or load_index()

    def search(self, message: str, *, k: int = 5, intent: str | None = None) -> list[dict[str, Any]]:
        return retrieve_hits(self.index, message, k=k, predicted_intent=intent)
