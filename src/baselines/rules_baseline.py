"""Simple rule-based escalation: escalate a short list of intents, else auto."""

from __future__ import annotations

from typing import Any

RISKY = {"apple_id_icloud", "billing_subscription", "order_purchase", "other"}


def rules_escalate(intent: str, message: str = "") -> dict[str, Any]:
    text = (message or "").lower()
    if any(w in text for w in ("refund", "lawyer", "stolen", "unauthorized")):
        return {
            "decision": "ESCALATE",
            "reason": "Keyword rule: refund/legal/fraud language.",
            "confidence": 0.6,
        }
    if intent in RISKY:
        return {
            "decision": "ESCALATE",
            "reason": f"Simple baseline escalates intent '{intent}'.",
            "confidence": 0.6,
        }
    return {
        "decision": "AUTO_HANDLE",
        "reason": f"Simple baseline auto-handles intent '{intent}'.",
        "confidence": 0.6,
    }
