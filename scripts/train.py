#!/usr/bin/env python
"""Fit proposed classifier + TF-IDF baseline on silver train labels."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.intent_classifier import IntentClassifier, save_classifier  # noqa: E402
from src.baselines.tfidf_classifier import TfidfBaselineClassifier, save_baseline  # noqa: E402
from src.data.load import load_split  # noqa: E402
from src.paths import ARTIFACTS, ensure_dirs  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_dirs()
    train = load_split("train")
    val = load_split("val")
    texts = [r["customer_message"] for r in train]
    labels = [r["silver_intent"] for r in train]
    val_texts = [r["customer_message"] for r in val]
    val_labels = [r["silver_intent"] for r in val]

    proposed = IntentClassifier().fit(texts, labels, val_texts=val_texts, val_labels=val_labels)
    save_classifier(proposed)

    baseline = TfidfBaselineClassifier().fit(texts, labels)
    save_baseline(baseline)

    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "proposed_calibrated": proposed.calibrated,
        "labels": proposed.labels,
    }
    (ARTIFACTS / "models" / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
