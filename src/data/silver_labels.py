"""Noisy keyword silver labels used ONLY on the train/val splits.

Golden labels are independent of this function. Some train examples are
deliberately mislabeled the way a cheap heuristic would be — that is the point.
"""

from __future__ import annotations

from collections import Counter

from src.config import intent_keywords, intent_names
from src.data.clean import lowercase

# Phrases that should override weaker keyword hits.
STRONG_PATTERNS: list[tuple[str, str]] = [
    ("apple id", "apple_id_icloud"),
    ("appleid", "apple_id_icloud"),
    ("verification code", "apple_id_icloud"),
    ("two-factor", "apple_id_icloud"),
    ("2fa", "apple_id_icloud"),
    ("refund", "billing_subscription"),
    ("charged twice", "billing_subscription"),
    ("charged me", "billing_subscription"),
    ("apple music", "billing_subscription"),
    ("how do i", "how_to_feature"),
    ("how to", "how_to_feature"),
    ("where is", "how_to_feature"),
    ("storage full", "backup_storage"),
    ("photos are gone", "backup_storage"),
    ("photos gone", "backup_storage"),
    ("backup failed", "backup_storage"),
    ("tracking number", "order_purchase"),
    ("my order", "order_purchase"),
    ("carplay", "connectivity"),
    ("airdrop", "connectivity"),
    ("wi-fi", "connectivity"),
    ("wifi", "connectivity"),
]


def silver_label(text: str) -> str:
    """Return a noisy intent for training. Never used as golden truth."""

    names = intent_names()
    body = lowercase(text)
    for phrase, intent in STRONG_PATTERNS:
        if phrase in body and intent in names:
            return intent

    scores: Counter[str] = Counter()
    for intent, keywords in intent_keywords().items():
        if intent == "other":
            continue
        for kw in keywords:
            if kw and kw in body:
                scores[intent] += 1
                # Extra weight for multi-word keywords.
                if " " in kw:
                    scores[intent] += 1

    if not scores:
        return "other"
    top = scores.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        # Ties are dumped to other rather than guessed — noisy on purpose.
        return "other"
    return top[0][0]
