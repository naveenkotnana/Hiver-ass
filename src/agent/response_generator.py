"""Grounded reply drafting.

Default path: adapt the top historical Apple Support reply.
Never invent refunds, prices, timelines, or coverage decisions.
If evidence is weak, ask a clarifying question instead of guessing policy.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import load_config
from src.data.clean import normalize_text
from src.llm import complete, llm_available

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
ORDER = re.compile(r"\bW\d{8,}\b|\b\d{10,}\b")
MONEY = re.compile(r"\$\s?\d+(?:\.\d{2})?")
PROMISE = re.compile(
    r"\b(we('ve| have)? (already )?(issued|processed|refunded)|your refund is|"
    r"we will replace|replacement is on the way|arriving tomorrow|"
    r"you will receive .* by)\b",
    re.I,
)

CLARIFY = (
    "We'd like to help. Could you share the device model and the iOS version "
    "(Settings > General > About), plus what you already tried? "
    "We do not have access to your account from this channel."
)

HOLDING = (
    "Thanks for reaching out — we want a specialist to review this with you. "
    "Please don't share passwords or full card numbers. A human will take it from here; "
    "this draft is not a confirmation of any refund, replacement, or account change."
)


def _sanitize(text: str) -> str:
    value = normalize_text(text)
    value = EMAIL.sub("[redacted-email]", value)
    value = PHONE.sub("[redacted-phone]", value)
    value = re.sub(r"@USER", "", value)
    value = re.sub(r"^@\w+\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _contains_unsupported_promise(text: str) -> bool:
    return bool(PROMISE.search(text) or MONEY.search(text))


def _adapt_historical(hit: dict[str, Any], max_chars: int) -> str:
    reply = _sanitize(hit.get("brand_response") or "")
    # Drop first-person invented commitments if a historical tweet had them.
    if _contains_unsupported_promise(reply):
        reply = PROMISE.sub("", reply)
        reply = MONEY.sub("", reply)
        reply = re.sub(r"\s+", " ", reply).strip()
    if len(reply) > max_chars:
        reply = reply[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return reply


def _llm_draft(message: str, evidence: list[dict[str, Any]], intent: str) -> str | None:
    if not llm_available():
        return None
    ev_lines = []
    for i, hit in enumerate(evidence[:3], start=1):
        ev_lines.append(
            f"{i}. customer: {hit.get('customer_message','')}\n"
            f"   agent: {hit.get('brand_response','')}"
        )
    prompt = (
        "You are drafting a public Apple Support–style Twitter reply.\n"
        "Rules:\n"
        "- Use ONLY facts supported by the historical evidence below.\n"
        "- Do NOT invent refunds, prices, ship dates, coverage, or account changes.\n"
        "- Do NOT claim you looked up the customer's account.\n"
        "- Be concise (max 2-3 sentences), professional, slightly Apple-like "
        "('We'd like to help', ask for Settings > General > About when relevant).\n"
        "- If evidence is insufficient, ask a clarifying question.\n"
        f"Intent: {intent}\n"
        f"Customer: {message}\n"
        "Historical evidence:\n"
        + "\n".join(ev_lines)
        + "\nReturn only the reply text."
    )
    return complete(
        prompt,
        system="You write grounded customer-support drafts. Never invent policy.",
        temperature=0.0,
    )


def generate_response(
    *,
    message: str,
    intent: str,
    evidence: list[dict[str, Any]],
    min_similarity: float | None = None,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    min_sim = float(min_similarity if min_similarity is not None else cfg["retrieval"]["min_similarity"])
    max_chars = int(cfg["generation"]["max_reply_chars"])
    top_sim = float(evidence[0]["similarity"]) if evidence else 0.0
    grounded = bool(evidence) and top_sim >= min_sim

    slim_evidence = [
        {
            "conversation_id": h.get("conversation_id"),
            "customer_message": h.get("customer_message"),
            "brand_response": h.get("brand_response"),
            "similarity": round(float(h.get("similarity") or 0.0), 4),
            "intent": h.get("intent") or "",
        }
        for h in evidence[:5]
    ]

    if not grounded:
        return {
            "reply": CLARIFY,
            "evidence": slim_evidence,
            "grounded": False,
            "mode": "clarify",
        }

    want_llm = cfg["generation"]["use_llm"] if use_llm is None else use_llm
    drafted = _llm_draft(message, evidence, intent) if want_llm else None
    if drafted:
        reply = _sanitize(drafted)
        if _contains_unsupported_promise(reply):
            reply = _adapt_historical(evidence[0], max_chars)
            mode = "historical_after_llm_safety"
        else:
            mode = "llm_grounded"
    else:
        reply = _adapt_historical(evidence[0], max_chars)
        mode = "historical"

    if not reply:
        reply = CLARIFY
        grounded = False
        mode = "clarify"

    return {
        "reply": reply,
        "evidence": slim_evidence,
        "grounded": grounded,
        "mode": mode,
    }


def holding_reply() -> str:
    return HOLDING
