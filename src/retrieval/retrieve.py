"""k-NN retrieval over the train-only index."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import load_config
from src.retrieval.index import RetrievalIndex
from src.retrieval.rerank import rerank


def retrieve(
    index: RetrievalIndex,
    query: str,
    *,
    k: int | None = None,
    predicted_intent: str | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config()["retrieval"]
    k = int(k or cfg["k"])
    vec = index.encode([query])
    # cosine because both sides are L2-normalized
    sims = (vec @ index.matrix.T).toarray().ravel()
    if len(sims) == 0:
        return []
    k_eff = min(k * 4, len(sims))  # over-retrieve for rerank
    top_idx = np.argpartition(-sims, kth=k_eff - 1)[:k_eff]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    hits: list[dict[str, Any]] = []
    for i in top_idx:
        rec = index.records[int(i)]
        hits.append(
            {
                "conversation_id": rec["conversation_id"],
                "customer_message": rec["customer_message"],
                "brand_response": rec["brand_response"],
                "intent": rec.get("intent") or "",
                "similarity": float(sims[int(i)]),
            }
        )
    ranked = rerank(
        query,
        hits,
        predicted_intent=predicted_intent,
        intent_bonus=float(cfg["rerank_intent_bonus"]),
        overlap_weight=float(cfg["rerank_overlap_weight"]),
    )
    return ranked[:k]
