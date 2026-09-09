"""Developer labeling protocol encoded as deterministic rules.

These rules were written after inspecting sample conversations and published
Apple Support tweet themes. They are the golden-set policy — not model output.

This is single-annotator developer labeling, not independent dual annotation.
"""

from __future__ import annotations

import re

from src.data.clean import lowercase

ESCALATE = "ESCALATE"
AUTO = "AUTO_HANDLE"

HIGH_RISK = re.compile(
    r"\b(lawyer|lawsuit|legal|attorney|sue|fraud|stolen|hacked|unauthorized|"
    r"child|suicide|kill myself|death|police|gdpr|ccpa|subpoena)\b",
    re.I,
)
ANGER = re.compile(
    r"\b(furious|disgusting|worst|hate you|never buying|scam|incompetent|"
    r"horrible|terrible service|third time|still not fixed|again and again)\b",
    re.I,
)
ACCOUNT_ACTION = re.compile(
    r"\b(refund|unlock|replace|replacement|cancel my (order|subscription)|"
    r"chargeback|send a new|lost (all )?(my )?photos|photos are gone|photos gone|"
    r"disappeared after restore)\b",
    re.I,
)
HOW_TO = re.compile(r"\b(how do i|how to|where is|where can i|what does|does \w+ support)\b", re.I)


def expected_action(message: str, intent: str) -> tuple[str, str]:
    """Return (AUTO_HANDLE|ESCALATE, reason) using the written labeling guide."""

    text = lowercase(message)

    if HIGH_RISK.search(text):
        return ESCALATE, "Legal, safety, fraud, or similarly high-risk language."
    if intent in {"apple_id_icloud", "billing_subscription"}:
        return ESCALATE, "Account or billing requests need an authenticated specialist; the agent cannot take those actions."
    if intent == "order_purchase":
        return ESCALATE, "Order lookup and shipping exceptions are account-specific."
    if intent == "backup_storage" and ACCOUNT_ACTION.search(text):
        return ESCALATE, "Possible data loss or backup recovery needs a specialist."
    if intent == "hardware_device" and re.search(
        r"won't turn on|wont turn on|not turn on|swollen|replacement|cracked|will not charge|won't charge",
        text,
    ):
        return ESCALATE, "Likely in-person hardware service; do not auto-promise coverage."
    if ANGER.search(text) and intent != "how_to_feature":
        return ESCALATE, "Severe complaint / repeated unresolved issue — human should own the tone."
    if intent == "other":
        return ESCALATE, "Intent is unclear or off-topic; a human should decide whether to engage."
    if intent == "how_to_feature" or HOW_TO.search(text):
        return AUTO, "Information / how-to request that can be answered from public documentation patterns."
    if intent in {"ios_update_bug", "connectivity"}:
        return AUTO, "Generic troubleshooting is historically handled in-thread without account access."
    if intent == "hardware_device":
        return ESCALATE, "Hardware issues often require inspection; default to a human."
    if intent == "backup_storage":
        return AUTO, "Storage / backup explanation without confirmed data-loss language."
    return ESCALATE, "Default conservative escalation."


def review_note(message: str, intent: str) -> str:
    text = lowercase(message)
    notes: list[str] = []
    if "update" in text and "battery" in text:
        notes.append("Cause named as iOS update; labeled ios_update_bug rather than hardware_device.")
    if "update" in text and ("wifi" in text or "wi-fi" in text):
        notes.append("Connectivity symptom after update; primary cause is the OS upgrade.")
    if "billing" in text and "apple id" in text:
        notes.append("Billing blocked downloads; billing_subscription over apple_id_icloud.")
    if "uber" in text or "verizon" in text or "school" in text:
        notes.append("Third-party / off-topic; other.")
    if not notes:
        notes.append("Primary intent from construction + labeling guide; single developer annotator.")
    action, reason = expected_action(message, intent)
    notes.append(f"expected_action={action} ({reason})")
    return " ".join(notes)
