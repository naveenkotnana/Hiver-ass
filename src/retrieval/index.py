"""Train-only TF-IDF index over historical customer messages."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import load_config
from src.paths import ARTIFACTS


class RetrievalIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix: sparse.spmatrix, records: list[dict[str, Any]]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.records = records

    def encode(self, texts: list[str]) -> sparse.spmatrix:
        return normalize(self.vectorizer.transform(texts))


def build_index(records: list[dict[str, Any]]) -> RetrievalIndex:
    cfg = load_config()["retrieval"]
    texts = [r["customer_message"] for r in records]
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=int(cfg["min_df"]),
        max_features=int(cfg["max_features"]),
        sublinear_tf=True,
        lowercase=True,
    )
    matrix = normalize(vectorizer.fit_transform(texts))
    slim = []
    for rec in records:
        slim.append(
            {
                "conversation_id": rec["conversation_id"],
                "customer_message": rec["customer_message"],
                "brand_response": rec["brand_response"],
                "intent": rec.get("silver_intent") or rec.get("intent") or "",
                "timestamp": rec.get("timestamp") or "",
            }
        )
    return RetrievalIndex(vectorizer, matrix, slim)


def save_index(index: RetrievalIndex, path=None) -> None:
    path = path or (ARTIFACTS / "index" / "tfidf.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": index.vectorizer, "matrix": index.matrix, "records": index.records},
        path,
    )


def load_index(path=None) -> RetrievalIndex:
    path = path or (ARTIFACTS / "index" / "tfidf.joblib")
    payload = joblib.load(path)
    return RetrievalIndex(payload["vectorizer"], payload["matrix"], payload["records"])
