"""End-to-end data preparation: source CSV → pairs → splits → golden set."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.data.conversations import reconstruct_pairs
from src.data.download import prepare_source_csv
from src.data.labeling import expected_action, review_note
from src.data.load import load_tweets, write_jsonl
from src.data.sampling import stratified_sample
from src.data.silver_labels import silver_label
from src.data.splits import assert_no_leakage, conversation_split, split_summary
from src.paths import DATA, PROCESSED, SAMPLE, ensure_dirs

logger = logging.getLogger(__name__)


GENERIC_FOLLOWUP = (
    "following up",
    "still happening",
    "that did not work",
    "i already tried that",
    "no change",
)


def _attach_gold(pairs: list[dict[str, Any]], gold_path: Path) -> None:
    """Attach construction-time gold ONLY via inbound tweet id.

    Do not fall back to conversation_id: multi-turn threads include generic
    follow-ups ("still happening") that must not inherit the issue label.
    """

    if not gold_path.exists():
        return
    gold = pd.read_csv(gold_path, dtype=str)
    by_inbound = dict(zip(gold["inbound_tweet_id"].astype(str), gold["gold_intent"]))
    for rec in pairs:
        inbound = str(rec["metadata"].get("inbound_tweet_id", ""))
        rec["metadata"]["gold_intent"] = by_inbound.get(inbound)


def _filter_brand(tweets: pd.DataFrame, brand: str) -> pd.DataFrame:
    authors = tweets["author_id"].astype(str)
    inbound = tweets["inbound"].astype(str).str.lower().isin(["true", "1", "yes"])
    # Keep inbound tweets that mention the brand OR any tweet authored by the brand
    # plus inbound tweets that the brand replied to — simplest: keep rows whose
    # author is the brand or whose text mentions it, plus their reply chains.
    mask_brand = authors.str.lower() == brand.lower()
    if mask_brand.sum() == 0:
        return tweets
    brand_ids = set(tweets.loc[mask_brand, "tweet_id"].astype(str))
    # Keep tweets the brand authored, tweets the brand replied to, and inbound
    # tweets that mention the handle.
    parent = tweets["in_response_to_tweet_id"].astype(str)
    text = tweets["text"].astype(str)
    mask = mask_brand | parent.isin(brand_ids) | text.str.contains(brand, case=False, na=False)
    # Also keep parents of brand tweets so threads reconstruct.
    keep_ids = set(tweets.loc[mask, "tweet_id"].astype(str))
    keep_ids |= set(parent[mask])
    extra = tweets["tweet_id"].astype(str).isin(keep_ids)
    filtered = tweets.loc[mask | extra].copy()
    # If filtering wiped everything (sample is already brand-only), fall back.
    return filtered if len(filtered) >= 50 else tweets


def build_golden_set(
    test_records: list[dict[str, Any]],
    *,
    size: int,
    min_per_intent: int,
    seed: int,
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for rec in test_records:
        gold = rec.get("metadata", {}).get("gold_intent")
        if not gold:
            continue
        msg = (rec.get("customer_message") or "").lower()
        if any(p in msg for p in GENERIC_FOLLOWUP) and len(msg) < 80:
            continue
        labeled.append({**rec, "true_intent": gold})
    if len(labeled) < size:
        # Fall back to remaining test records with silver labels only if needed.
        have = {id(r) for r in labeled}
        for rec in test_records:
            if id(rec) in have:
                continue
            msg = (rec.get("customer_message") or "").lower()
            if any(p in msg for p in GENERIC_FOLLOWUP) and len(msg) < 80:
                continue
            gold = rec.get("metadata", {}).get("gold_intent") or silver_label(rec["customer_message"])
            labeled.append({**rec, "true_intent": gold})
            if len(labeled) >= max(size * 2, size):
                break

    sampled = stratified_sample(
        labeled,
        label_key="true_intent",
        size=size,
        min_per_class=min_per_intent,
        seed=seed,
    )
    golden: list[dict[str, Any]] = []
    for i, rec in enumerate(sampled, start=1):
        intent = rec["true_intent"]
        action, reason = expected_action(rec["customer_message"], intent)
        example_id = f"gold_{i:03d}"
        golden.append(
            {
                "example_id": example_id,
                "conversation_id": rec["conversation_id"],
                "customer_message": rec["customer_message"],
                "conversation_context": rec.get("previous_context") or "",
                "true_intent": intent,
                "expected_action": action,
                "escalation_reason": reason if action == "ESCALATE" else "",
                "notes": review_note(rec["customer_message"], intent),
                "timestamp": rec.get("timestamp") or "",
                "brand": rec.get("brand") or "AppleSupport",
                "reference_brand_response": rec.get("brand_response") or "",
            }
        )
    return golden


def prepare_dataset(*, force_sample: bool = True, seed: int | None = None) -> dict[str, Any]:
    ensure_dirs()
    cfg = load_config()
    seed = cfg["seed"] if seed is None else seed
    per_intent = int(cfg["data"]["sample_conversations_per_intent"])
    brand = cfg["brand"]

    source = prepare_source_csv(force_sample=force_sample, seed=seed, per_intent=per_intent)
    tweets = load_tweets(source)
    tweets = _filter_brand(tweets, brand)
    pairs = reconstruct_pairs(tweets, brand=brand)
    _attach_gold(pairs, SAMPLE / "gold_intents.csv")
    if len(pairs) < 50:
        raise RuntimeError(f"Only {len(pairs)} usable pairs after reconstruction.")

    splits = conversation_split(
        pairs,
        train_ratio=float(cfg["splits"]["train_ratio"]),
        val_ratio=float(cfg["splits"]["val_ratio"]),
        test_ratio=float(cfg["splits"]["test_ratio"]),
        seed=seed,
    )
    assert_no_leakage(splits)

    for name, recs in splits.items():
        # Strip gold_intent from train/val so it cannot be accidentally trained on.
        out = []
        for rec in recs:
            cloned = json.loads(json.dumps(rec))
            if name != "test":
                cloned.setdefault("metadata", {}).pop("gold_intent", None)
            cloned["silver_intent"] = silver_label(cloned["customer_message"])
            out.append(cloned)
        write_jsonl(PROCESSED / f"{name}.jsonl", out)
        splits[name] = out

    golden = build_golden_set(
        splits["test"],
        size=int(cfg["splits"]["golden_size"]),
        min_per_intent=int(cfg["splits"]["golden_min_per_intent"]),
        seed=seed,
    )
    golden_ids = {g["conversation_id"] for g in golden}

    csv_path = DATA / "golden_set.csv"
    jsonl_path = DATA / "golden_set.jsonl"
    pd.DataFrame(golden).to_csv(csv_path, index=False)
    write_jsonl(jsonl_path, golden)

    summary = split_summary(splits)
    summary_path = PROCESSED / "split_summary.csv"
    summary.to_csv(summary_path, index=False)

    stats = {
        "source": str(source),
        "n_pairs": len(pairs),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_golden": len(golden),
        "golden_conversations_held_out": len(golden_ids),
        "split_summary": summary.to_dict(orient="records"),
    }
    (PROCESSED / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Prepared dataset: %s", stats)
    return stats
