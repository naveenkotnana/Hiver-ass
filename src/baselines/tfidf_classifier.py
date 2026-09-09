"""Simple baseline intent classifier: unigram TF-IDF + logistic regression."""

from __future__ import annotations

from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.paths import ARTIFACTS


class TfidfBaselineClassifier:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.clf: LogisticRegression | None = None
        self.labels: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfBaselineClassifier":
        spec = load_config()["baseline_tfidf"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(spec["ngram_range"]),
            min_df=int(spec["min_df"]),
            max_features=int(spec["max_features"]),
            sublinear_tf=bool(spec.get("sublinear_tf", False)),
            lowercase=True,
        )
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(
            C=float(spec.get("C", 1.0)),
            class_weight=spec.get("class_weight") or None,
            max_iter=int(spec.get("max_iter", 200)),
        )
        self.clf.fit(X, labels)
        self.labels = list(self.clf.classes_)
        return self

    def predict_one(self, text: str) -> dict[str, Any]:
        if self.vectorizer is None or self.clf is None:
            raise RuntimeError("not fitted")
        X = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(X)[0]
        i = int(proba.argmax())
        return {
            "intent": str(self.labels[i]),
            "confidence": float(proba[i]),
            "alternatives": [],
            "calibrated": False,
        }


def save_baseline(model: TfidfBaselineClassifier) -> None:
    path = ARTIFACTS / "models" / "tfidf_baseline.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_baseline() -> TfidfBaselineClassifier:
    return joblib.load(ARTIFACTS / "models" / "tfidf_baseline.joblib")
