"""Structured response judge.

Primary path: deterministic heuristic judge (reproducible, no API).
Optional path: xAI LLM judge when XAI_API_KEY is set.

Scores are 0–4. The judge sees the customer message, evidence, and reply.
Fluent but ungrounded replies are penalized.
"""

from __future__ import annotations

import re
from typing import Any

from src.evaluation.metrics import mean_std
from src.llm import complete, llm_available, parse_json_object

RUBRIC = """Score each dimension 0-4:
0 unacceptable, 1 poor, 2 acceptable, 3 good, 4 excellent.

1. relevance — does the reply address the customer's issue?
2. correctness — no false resolution claims; safe given what we know
3. groundedness — claims are supported by the retrieved evidence
4. helpfulness — customer can take a next step
5. brand_consistency — sounds like a careful public Apple Support tweet
6. unsupported_claims — 4 = no invented refunds/prices/timelines/account actions;
   0 = invents a commitment. THIS IS INVERTED SO HIGHER IS BETTER.

Penalize fluent but ungrounded answers. Do not reward length.
Return JSON: {
  "relevance": int, "correctness": int, "groundedness": int,
  "helpfulness": int, "brand_consistency": int, "unsupported_claims": int,
  "overall": int, "grounded": bool, "notes": str
}
overall should be the integer mean of the six dimensions, rounded.
"""

MONEY = re.compile(r"\$\s?\d+")
PROMISE = re.compile(
    r"\b(refund(ed| is processed| has been)|we will replace|on the way|"
    r"arriving (tomorrow|today)|I've looked up your account|I can see your order)\b",
    re.I,
)
APPLEISH = re.compile(
    r"(we'd like to help|settings >|dm |reportaproblem|iforgot|support\.apple|"
    r"genius bar|do not share (your )?password)",
    re.I,
)
NEXT_STEP = re.compile(
    r"(settings|dm|reportaproblem|iforgot|reboot|force restart|about\b|tell us|could you)",
    re.I,
)


def _clip(n: int) -> int:
    return max(0, min(4, int(n)))


def heuristic_judge(message: str, reply: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    msg = (message or "").lower()
    rep = (reply or "").lower()
    ev_text = " ".join(
        (e.get("customer_message") or "") + " " + (e.get("brand_response") or "") for e in evidence
    ).lower()

    msg_toks = {t for t in re.findall(r"[a-z0-9]+", msg) if len(t) > 3}
    rep_toks = {t for t in re.findall(r"[a-z0-9]+", rep) if len(t) > 3}
    ev_toks = {t for t in re.findall(r"[a-z0-9]+", ev_text) if len(t) > 3}
    overlap_msg = len(msg_toks & rep_toks) / max(1, len(msg_toks))
    overlap_ev = len(rep_toks & ev_toks) / max(1, len(rep_toks)) if rep_toks else 0.0

    relevance = _clip(round(1 + 6 * overlap_msg))
    if "we'd like to help" in rep or "help" in rep:
        relevance = max(relevance, 2)

    invented = bool(PROMISE.search(rep) or MONEY.search(rep))
    unsupported = 0 if invented else (4 if overlap_ev >= 0.15 or not evidence else 3)
    correctness = 1 if invented else (3 if overlap_msg >= 0.08 else 2)
    groundedness = _clip(round(8 * overlap_ev))
    if not evidence:
        groundedness = min(groundedness, 1)
    helpfulness = 3 if NEXT_STEP.search(rep) else 2
    if len(rep) < 40:
        helpfulness = min(helpfulness, 1)
    brand = 3 if APPLEISH.search(rep) else 2
    if invented:
        brand = min(brand, 1)
        correctness = min(correctness, 1)

    dims = {
        "relevance": relevance,
        "correctness": correctness,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "brand_consistency": brand,
        "unsupported_claims": unsupported,
    }
    overall = int(round(sum(dims.values()) / 6))
    grounded = groundedness >= 2 and unsupported >= 3 and not invented
    notes = "heuristic_judge"
    if invented:
        notes += "; penalized invented money/promise"
    return {**dims, "overall": overall, "grounded": grounded, "notes": notes, "judge": "heuristic"}


def llm_judge(message: str, reply: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not llm_available():
        return None
    ev = []
    for i, hit in enumerate(evidence[:3], start=1):
        ev.append(f"{i}. customer: {hit.get('customer_message')}\n   brand: {hit.get('brand_response')}")
    prompt = (
        RUBRIC
        + f"\nCustomer message:\n{message}\n\nRetrieved evidence:\n"
        + ("\n".join(ev) if ev else "(none)")
        + f"\n\nGenerated reply:\n{reply}\n"
    )
    text = complete(prompt, system="You are a strict evaluation judge for support replies.", temperature=0.0)
    payload = parse_json_object(text or "")
    if not payload:
        return None
    dims = {}
    for key in (
        "relevance",
        "correctness",
        "groundedness",
        "helpfulness",
        "brand_consistency",
        "unsupported_claims",
        "overall",
    ):
        try:
            dims[key] = _clip(int(payload.get(key, 0)))
        except (TypeError, ValueError):
            dims[key] = 0
    dims["grounded"] = bool(payload.get("grounded"))
    dims["notes"] = str(payload.get("notes") or "")
    dims["judge"] = "llm"
    return dims


def judge_response(message: str, reply: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    llm = llm_judge(message, reply, evidence)
    heur = heuristic_judge(message, reply, evidence)
    if llm is None:
        return heur
    llm["heuristic_overall"] = heur["overall"]
    return llm


def aggregate_judge_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "relevance",
        "correctness",
        "groundedness",
        "helpfulness",
        "brand_consistency",
        "unsupported_claims",
        "overall",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        out[key] = mean_std([float(r.get(key) or 0) for r in rows])
    out["grounded_rate"] = float(np_mean([1.0 if r.get("grounded") else 0.0 for r in rows]))
    out["hallucination_rate"] = float(
        np_mean([1.0 if int(r.get("unsupported_claims") or 4) <= 1 else 0.0 for r in rows])
    )
    out["n"] = len(rows)
    return out


def np_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0
