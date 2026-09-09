"""Intent, escalation, and retrieval metrics. No fabricated numbers — callers pass predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, lab in enumerate(labels):
        per_class[lab] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": cm,
        "n": len(y_true),
    }


def escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Positive class = ESCALATE. Also report false auto-handle rate."""

    acc = float(accuracy_score(y_true, y_pred))
    pos = "ESCALATE"
    y_t = [1 if y == pos else 0 for y in y_true]
    y_p = [1 if y == pos else 0 for y in y_pred]
    p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
    # False auto-handle: predicted AUTO_HANDLE when true is ESCALATE.
    false_auto = sum(1 for t, p_ in zip(y_true, y_pred) if t == "ESCALATE" and p_ == "AUTO_HANDLE")
    should_esc = sum(1 for t in y_true if t == "ESCALATE")
    false_auto_rate = float(false_auto / should_esc) if should_esc else 0.0
    return {
        "accuracy": acc,
        "precision_escalate": float(p),
        "recall_escalate": float(r),
        "f1_escalate": float(f1),
        "false_auto_handle_rate": false_auto_rate,
        "n_escalate": int(should_esc),
        "n": len(y_true),
    }


def retrieval_metrics(
    gold_intents: list[str],
    retrieved_intent_lists: list[list[str]],
    ks: Iterable[int] = (1, 3, 5),
) -> dict:
    """Same-intent relevance: the exact test conversation is held out of the index."""

    ks = list(ks)
    hits = {k: 0 for k in ks}
    rr: list[float] = []
    n = len(gold_intents)
    for gold, retrieved in zip(gold_intents, retrieved_intent_lists):
        rank = None
        for i, intent in enumerate(retrieved, start=1):
            if intent == gold:
                rank = i
                break
        rr.append(0.0 if rank is None else 1.0 / rank)
        for k in ks:
            if gold in retrieved[:k]:
                hits[k] += 1
    out = {f"recall@{k}": (hits[k] / n if n else 0.0) for k in ks}
    out["mrr"] = float(np.mean(rr) if rr else 0.0)
    out["n"] = n
    return out


def mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}
