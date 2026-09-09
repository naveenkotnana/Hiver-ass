"""Reconstruct customer→brand pairs from tweet-level CSVs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config
from src.data.clean import is_usable_message, normalize_text

BRAND_HINTS = {"applesupport"}


def _is_brand(author_id: str, selected_brand: str) -> bool:
    a = str(author_id).strip().lower()
    return a == selected_brand.lower() or a in BRAND_HINTS


def _as_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def _root_id(tweet_id: str, parent_of: dict[str, str]) -> str:
    seen: set[str] = set()
    current = tweet_id
    while current and current in parent_of and current not in seen:
        seen.add(current)
        current = parent_of[current]
    return current or tweet_id


def reconstruct_pairs(
    tweets: pd.DataFrame,
    *,
    brand: str | None = None,
) -> list[dict[str, Any]]:
    """Build one record per customer message that received a brand reply.

    Conversation ID is the root tweet in the in_response_to chain so that
    multi-turn threads stay together for leakage-aware splits.
    """

    cfg = load_config()
    brand = brand or cfg["brand"]
    min_c = int(cfg["data"]["min_customer_message_chars"])
    min_b = int(cfg["data"]["min_brand_response_chars"])
    max_ctx = int(cfg["data"]["max_previous_context_chars"])

    df = tweets.copy()
    df.columns = [c.strip() for c in df.columns]
    required = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tweet CSV missing columns: {sorted(missing)}")

    df["tweet_id"] = df["tweet_id"].map(_as_id)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].map(_as_id)
    df["author_id"] = df["author_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    inbound_raw = df["inbound"]
    df["inbound_bool"] = inbound_raw.map(
        lambda v: str(v).strip().lower() in {"true", "1", "yes"} if not isinstance(v, bool) else v
    )

    by_id: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for row in df.to_dict(orient="records"):
        tid = row["tweet_id"]
        if not tid:
            continue
        by_id[tid] = row
        parent = row["in_response_to_tweet_id"]
        if parent:
            parent_of[tid] = parent
            children.setdefault(parent, []).append(tid)

    pairs: list[dict[str, Any]] = []
    seen_inbound: set[str] = set()

    for tid, row in by_id.items():
        if not row["inbound_bool"]:
            continue
        if _is_brand(row["author_id"], brand):
            continue
        replies = [
            by_id[cid]
            for cid in children.get(tid, [])
            if cid in by_id
            and not by_id[cid]["inbound_bool"]
            and _is_brand(by_id[cid]["author_id"], brand)
        ]
        if not replies:
            continue
        replies.sort(key=lambda r: str(r.get("created_at") or ""))
        brand_row = replies[0]
        customer_text = normalize_text(row["text"])
        brand_text = normalize_text(brand_row["text"])
        if not is_usable_message(customer_text, min_c):
            continue
        if len(brand_text) < min_b:
            continue

        root = _root_id(tid, parent_of)
        context_bits: list[str] = []
        cursor = row["in_response_to_tweet_id"]
        hops = 0
        while cursor and cursor in by_id and hops < 6:
            prev = by_id[cursor]
            who = "brand" if _is_brand(prev["author_id"], brand) else "customer"
            context_bits.append(f"{who}: {normalize_text(prev['text'])}")
            cursor = prev.get("in_response_to_tweet_id") or ""
            hops += 1
        context_bits.reverse()
        previous_context = " | ".join(context_bits)[:max_ctx]

        pairs.append(
            {
                "conversation_id": root,
                "customer_message": customer_text,
                "brand_response": brand_text,
                "previous_context": previous_context,
                "timestamp": str(row.get("created_at") or ""),
                "brand": brand,
                "metadata": {
                    "inbound_tweet_id": tid,
                    "response_tweet_id": brand_row["tweet_id"],
                    "customer_author_id": row["author_id"],
                    "n_brand_replies": len(replies),
                },
            }
        )
        seen_inbound.add(tid)

    pairs.sort(key=lambda p: (p["timestamp"], p["conversation_id"], p["metadata"]["inbound_tweet_id"]))
    return pairs
