"""End-to-end support agent: classify → retrieve → draft → escalate."""

from __future__ import annotations

from typing import Any

from src.agent.escalation import decide_escalation
from src.agent.intent_classifier import IntentClassifier, load_classifier
from src.agent.response_generator import generate_response, holding_reply
from src.agent.retriever import HistoricalRetriever
from src.data.clean import normalize_text


class SupportAgent:
    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        retriever: HistoricalRetriever | None = None,
    ) -> None:
        self.classifier = classifier or load_classifier()
        self.retriever = retriever or HistoricalRetriever()

    def handle(self, message: str, *, context: str = "", k: int = 5) -> dict[str, Any]:
        cleaned = normalize_text(message)
        if context:
            cleaned_for_model = f"{normalize_text(context)} [SEP] {cleaned}"
        else:
            cleaned_for_model = cleaned

        intent_out = self.classifier.predict_one(cleaned_for_model)
        evidence = self.retriever.search(cleaned, k=k, intent=intent_out["intent"])
        gen = generate_response(
            message=cleaned,
            intent=intent_out["intent"],
            evidence=evidence,
        )
        esc = decide_escalation(
            message=cleaned,
            intent=intent_out["intent"],
            intent_confidence=float(intent_out["confidence"]),
            evidence=evidence,
            grounded=bool(gen["grounded"]),
        )
        reply = gen["reply"]
        if esc["decision"] == "ESCALATE":
            # Keep a grounded draft for the human, plus a holding line.
            reply = f"{holding_reply()} Draft for the specialist: {gen['reply']}"

        return {
            "intent": intent_out["intent"],
            "intent_confidence": round(float(intent_out["confidence"]), 4),
            "intent_calibrated": bool(intent_out.get("calibrated")),
            "alternatives": intent_out.get("alternatives") or [],
            "decision": esc["decision"],
            "reason": esc["reason"],
            "escalation_confidence": esc["confidence"],
            "reply": reply,
            "draft_reply": gen["reply"],
            "evidence": gen["evidence"],
            "grounded": gen["grounded"],
            "generation_mode": gen.get("mode"),
            "customer_message": cleaned,
        }


def format_cli(result: dict[str, Any]) -> str:
    lines = [
        "Intent:",
        str(result["intent"]),
        "",
        "Decision:",
        str(result["decision"]),
        "",
        "Reason:",
        str(result["reason"]),
        "",
        "Draft Reply:",
        str(result["draft_reply"]),
        "",
        "Evidence:",
    ]
    for i, hit in enumerate(result.get("evidence") or [], start=1):
        lines.append(
            f"{i}. sim={hit.get('similarity'):.3f} intent={hit.get('intent')} "
            f"id={hit.get('conversation_id')}\n"
            f"   customer: {hit.get('customer_message')}\n"
            f"   brand: {hit.get('brand_response')}"
        )
    if not result.get("evidence"):
        lines.append("(none)")
    return "\n".join(lines)
