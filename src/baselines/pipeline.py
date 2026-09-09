"""Glue the simple baseline pieces into one handle() API."""

from __future__ import annotations

from typing import Any

from src.baselines.majority import MajorityBaseline
from src.baselines.retrieval_baseline import RetrievalBaseline
from src.baselines.rules_baseline import rules_escalate
from src.baselines.tfidf_classifier import TfidfBaselineClassifier
from src.data.clean import normalize_text


class SimpleBaselineAgent:
    def __init__(self, classifier: TfidfBaselineClassifier, retriever: RetrievalBaseline):
        self.classifier = classifier
        self.retriever = retriever

    def handle(self, message: str, **_: Any) -> dict[str, Any]:
        cleaned = normalize_text(message)
        pred = self.classifier.predict_one(cleaned)
        retrieved = self.retriever.handle(cleaned, intent=pred["intent"])
        esc = rules_escalate(pred["intent"], cleaned)
        return {
            "intent": pred["intent"],
            "intent_confidence": pred["confidence"],
            "decision": esc["decision"],
            "reason": esc["reason"],
            "reply": retrieved["reply"],
            "draft_reply": retrieved["reply"],
            "evidence": retrieved["evidence"],
            "grounded": retrieved["grounded"],
        }


def majority_agent(labels: list[str]) -> MajorityBaseline:
    return MajorityBaseline().fit(labels)
