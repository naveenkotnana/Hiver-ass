"""Deterministic sampling for the golden set and judge-human subset."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def stratified_sample(
    records: list[dict[str, Any]],
    *,
    label_key: str,
    size: int,
    min_per_class: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample up to ``size`` records, guaranteeing a floor per label when possible."""

    rng = np.random.RandomState(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_label[str(rec[label_key])].append(rec)

    chosen: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for _label, items in sorted(by_label.items()):
        idx = rng.permutation(len(items))
        ordered = [items[i] for i in idx]
        take = min(min_per_class, len(ordered))
        chosen.extend(ordered[:take])
        leftover.extend(ordered[take:])

    rng.shuffle(leftover)
    for rec in leftover:
        if len(chosen) >= size:
            break
        chosen.append(rec)

    if len(chosen) > size:
        idx = rng.permutation(len(chosen))[:size]
        chosen = [chosen[i] for i in sorted(idx)]
    return chosen


def sample_ids(ids: Iterable[str], size: int, seed: int = 42) -> list[str]:
    rng = np.random.RandomState(seed)
    values = list(ids)
    rng.shuffle(values)
    return values[:size]
