from __future__ import annotations

from src.agent.escalation import decide_escalation
from src.agent.intent_classifier import IntentClassifier
from src.agent.response_generator import generate_response
from src.agent.retriever import HistoricalRetriever
from src.agent.support_agent import SupportAgent, format_cli
from src.retrieval.index import build_index


def test_escalation_high_risk():
    out = decide_escalation(
        message="I will sue you and call my lawyer about this unauthorized charge",
        intent="billing_subscription",
        intent_confidence=0.9,
        evidence=[{"similarity": 0.8, "customer_message": "x", "brand_response": "y"}],
        grounded=True,
    )
    assert out["decision"] == "ESCALATE"
    assert "lawyer" in out["reason"].lower() or "legal" in out["reason"].lower() or "high-risk" in out["reason"].lower()


def test_response_not_grounded_asks_clarification():
    out = generate_response(
        message="asdf",
        intent="other",
        evidence=[],
        min_similarity=0.12,
    )
    assert out["grounded"] is False
    assert "help" in out["reply"].lower() or "share" in out["reply"].lower()


def test_response_schema_with_evidence():
    ev = [
        {
            "conversation_id": "1",
            "customer_message": "how do I screenshot",
            "brand_response": "On iPhone 8 press side + Home.",
            "similarity": 0.9,
            "intent": "how_to_feature",
        }
    ]
    out = generate_response(message="how do I take a screenshot", intent="how_to_feature", evidence=ev)
    assert out["grounded"] is True
    assert out["evidence"][0]["conversation_id"] == "1"
    assert "refund has been issued" not in out["reply"].lower()


def test_agent_end_to_end_schema(tiny_splits):
    train = tiny_splits["train"]
    clf = IntentClassifier().fit(
        [r["customer_message"] for r in train],
        [r["silver_intent"] for r in train],
    )
    index = build_index(train)
    agent = SupportAgent(classifier=clf, retriever=HistoricalRetriever(index))
    result = agent.handle("how do I turn off Wi-Fi Assist on iOS 11")
    assert result["intent"]
    assert result["decision"] in {"AUTO_HANDLE", "ESCALATE"}
    assert result["reason"]
    assert result["draft_reply"]
    text = format_cli(result)
    assert "Intent:" in text
    assert "Decision:" in text
