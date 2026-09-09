from __future__ import annotations

from src.retrieval.index import build_index
from src.retrieval.retrieve import retrieve


def test_retrieve_returns_k_hits(tiny_splits):
    index = build_index(tiny_splits["train"])
    hits = retrieve(index, "my iPhone will not connect to Wi-Fi", k=3)
    assert 1 <= len(hits) <= 3
    for hit in hits:
        assert "conversation_id" in hit
        assert "brand_response" in hit
        assert "similarity" in hit
        assert hit["similarity"] <= 1.0 + 1e-6


def test_index_only_uses_provided_records(tiny_splits):
    index = build_index(tiny_splits["train"])
    train_ids = {r["conversation_id"] for r in tiny_splits["train"]}
    assert {r["conversation_id"] for r in index.records} <= train_ids
