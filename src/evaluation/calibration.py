"""Reliability-style summaries of intent confidence vs correctness."""

from __future__ import annotations

import numpy as np


def confidence_bins(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    n_bins: int = 5,
) -> list[dict]:
    correct = np.array([t == p for t, p in zip(y_true, y_pred)], dtype=float)
    conf = np.array(confidences, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "accuracy": None, "mean_confidence": None})
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "mean_confidence": float(conf[mask].mean()),
            }
        )
    return rows


def ece(y_true: list[str], y_pred: list[str], confidences: list[float], n_bins: int = 5) -> float:
    """Expected Calibration Error (not claimed as a headline)."""

    bins = confidence_bins(y_true, y_pred, confidences, n_bins=n_bins)
    n = len(y_true) or 1
    total = 0.0
    for b in bins:
        if not b["n"]:
            continue
        total += (b["n"] / n) * abs((b["accuracy"] or 0) - (b["mean_confidence"] or 0))
    return float(total)
