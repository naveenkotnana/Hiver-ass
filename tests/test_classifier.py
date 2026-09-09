from __future__ import annotations

from src.agent.intent_classifier import IntentClassifier
from src.baselines.tfidf_classifier import TfidfBaselineClassifier


def test_proposed_classifier_schema(tiny_splits):
    train = tiny_splits["train"]
    texts = [r["customer_message"] for r in train]
    labels = [r["silver_intent"] for r in train]
    clf = IntentClassifier().fit(texts, labels)
    out = clf.predict_one("how do I take a screenshot on iPhone 8")
    assert "intent" in out
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["alternatives"], list)


def test_baseline_classifier_fits(tiny_splits):
    train = tiny_splits["train"]
    clf = TfidfBaselineClassifier().fit(
        [r["customer_message"] for r in train],
        [r["silver_intent"] for r in train],
    )
    out = clf.predict_one("I was charged twice for Apple Music")
    assert out["intent"]
    assert out["calibrated"] is False
