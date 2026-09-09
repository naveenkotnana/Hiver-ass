"""Conversation-level splits. Messages never leak across train/val/test."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _conversation_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for rec in records:
        cid = str(rec["conversation_id"])
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def conversation_split(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")

    rng = np.random.RandomState(seed)
    conv_ids = _conversation_ids(records)
    rng.shuffle(conv_ids)
    n = len(conv_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(conv_ids[:n_train])
    val_ids = set(conv_ids[n_train : n_train + n_val])
    test_ids = set(conv_ids[n_train + n_val :])

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for rec in records:
        cid = str(rec["conversation_id"])
        if cid in train_ids:
            buckets["train"].append(rec)
        elif cid in val_ids:
            buckets["val"].append(rec)
        else:
            buckets["test"].append(rec)
    return buckets


def assert_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    sets = {name: {str(r["conversation_id"]) for r in recs} for name, recs in splits.items()}
    overlap_tv = sets.get("train", set()) & sets.get("val", set())
    overlap_tt = sets.get("train", set()) & sets.get("test", set())
    overlap_vt = sets.get("val", set()) & sets.get("test", set())
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Conversation leakage: train∩val={len(overlap_tv)} "
            f"train∩test={len(overlap_tt)} val∩test={len(overlap_vt)}"
        )


def split_summary(splits: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for name, recs in splits.items():
        conv = {r["conversation_id"] for r in recs}
        rows.append(
            {
                "split": name,
                "messages": len(recs),
                "conversations": len(conv),
            }
        )
    return pd.DataFrame(rows)


def group_by_conversation(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["conversation_id"])].append(rec)
    return dict(grouped)
