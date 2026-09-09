from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module
from src.agent.intent_classifier import IntentClassifier
from src.agent.retriever import HistoricalRetriever
from src.agent.support_agent import SupportAgent
from src.retrieval.index import build_index


def test_health():
    client = TestClient(app_module.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict(tiny_splits, monkeypatch):
    train = tiny_splits["train"]
    clf = IntentClassifier().fit(
        [r["customer_message"] for r in train],
        [r["silver_intent"] for r in train],
    )
    index = build_index(train)
    agent = SupportAgent(classifier=clf, retriever=HistoricalRetriever(index))
    app_module.get_agent.cache_clear()
    monkeypatch.setattr(app_module, "get_agent", lambda: agent)
    client = TestClient(app_module.app)
    res = client.post("/predict", json={"message": "I was charged twice for my order"})
    assert res.status_code == 200
    body = res.json()
    assert "intent" in body
    assert "decision" in body
    assert "reason" in body
    assert "reply" in body
    assert "evidence" in body
    assert "intent_confidence" in body
