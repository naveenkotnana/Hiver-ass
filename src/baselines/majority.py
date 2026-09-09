"""Trivial baseline: majority intent, always escalate, canned reply."""

from __future__ import annotations

from collections import Counter
from typing import Any


CANNED = "Thanks for reaching out. We will look into this."


class MajorityBaseline:
    def __init__(self) -> None:
        self.majority_intent: str = "other"

    def fit(self, labels: list[str]) -> "MajorityBaseline":
        if labels:
            self.majority_intent = Counter(labels).most_common(1)[0][0]
        return self

    def handle(self, message: str, **_: Any) -> dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 1.0,
            "decision": "ESCALATE",
            "reason": "Trivial baseline always escalates.",
            "reply": CANNED,
            "draft_reply": CANNED,
            "evidence": [],
            "grounded": False,
        }
