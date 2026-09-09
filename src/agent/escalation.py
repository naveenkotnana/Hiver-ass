"""Transparent multi-signal escalation policy.

The agent cannot perform account actions. When in doubt, escalate.
False auto-handles are treated as the costly error.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import intent_defaults, load_config

HIGH_RISK = re.compile(
    r"\b(lawyer|lawsuit|legal|attorney|sue\b|fraud|stolen|hacked|unauthorized|"
    r"child|suicide|kill myself|death|police|gdpr|ccpa|chargeback)\b",
    re.I,
)
ANGER = re.compile(
    r"\b(furious|disgusting|worst company|hate you|never buying|scam|"
    r"incompetent|horrible|third time|still not fixed|again and again)\b",
    re.I,
)
ACCOUNT_ACTION = re.compile(
    r"\b(refund|unlock my|replace my|send a new|cancel my (order|subscription)|"
    r"lost (all )?(my )?photos|charged twice|did not authorize)\b",
    re.I,
)
REPEATED = re.compile(r"\b(still happening|still not|tried that|third time|again)\b", re.I)


def decide_escalation(
    *,
    message: str,
    intent: str,
    intent_confidence: float,
    evidence: list[dict[str, Any]],
    grounded: bool,
) -> dict[str, Any]:
    cfg = load_config()["escalation"]
    reasons: list[str] = []
    score = 0.0  # higher → more likely escalate

    if intent in set(cfg.get("always_escalate_intents") or []):
        reasons.append(
            f"Intent '{intent}' requires an authenticated specialist; "
            "the agent cannot take account or billing actions."
        )
        score += 1.5

    if intent_confidence < float(cfg["min_intent_confidence"]):
        reasons.append(
            f"Intent confidence {intent_confidence:.2f} is below the "
            f"{cfg['min_intent_confidence']} threshold."
        )
        score += 1.0

    top_sim = float(evidence[0]["similarity"]) if evidence else 0.0
    if not grounded or top_sim < float(cfg["min_evidence_similarity"]):
        reasons.append(
            "Historical evidence is missing or too dissimilar to safely copy a resolution."
        )
        score += 1.2

    if HIGH_RISK.search(message or ""):
        reasons.append("Message contains legal, safety, fraud, or similarly high-risk language.")
        score += 2.0
    if ANGER.search(message or ""):
        reasons.append("Severe complaint language — a human should own tone and next steps.")
        score += 0.8
    if ACCOUNT_ACTION.search(message or ""):
        reasons.append("Customer is asking for an account-specific action (refund, unlock, replace).")
        score += 1.3
    if REPEATED.search(message or ""):
        reasons.append("Customer indicates the issue is repeated / still unresolved.")
        score += 0.4

    intents_high = set(cfg.get("high_risk_intents") or [])
    if intent in intents_high and intent not in set(cfg.get("always_escalate_intents") or []):
        reasons.append(f"Intent '{intent}' is high-risk even when generic troubleshooting exists.")
        score += 0.7

    default = intent_defaults().get(intent, "ESCALATE")
    if not reasons and default == "ESCALATE":
        reasons.append(f"Taxonomy default for '{intent}' is escalate.")
        score += 0.5

    if score >= 0.7 or reasons:
        decision = "ESCALATE"
        reason = " ".join(reasons) if reasons else "Conservative default: escalate."
        # Confidence here is policy certainty, not a calibrated probability.
        conf = min(0.95, 0.55 + 0.1 * len(reasons))
    else:
        decision = "AUTO_HANDLE"
        reason = (
            f"Intent '{intent}' is a generic how-to or troubleshooting request, "
            f"confidence is {intent_confidence:.2f}, and retrieved evidence is similar enough "
            "to draft a public-style reply without promising account actions."
        )
        conf = min(0.9, 0.5 + intent_confidence * 0.3 + min(top_sim, 0.3))

    return {
        "decision": decision,
        "reason": reason,
        "confidence": round(float(conf), 3),
        "signals": reasons,
        "policy_score": round(float(score), 3),
    }
