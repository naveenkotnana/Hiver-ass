from __future__ import annotations

import pytest

from src.data.splits import assert_no_leakage, conversation_split


def test_no_conversation_overlap(tiny_pairs):
    splits = conversation_split(tiny_pairs, seed=1)
    assert_no_leakage(splits)
    ids = {name: {r["conversation_id"] for r in recs} for name, recs in splits.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert splits["train"] and splits["test"]


def test_leakage_detector_fires():
    rec = {"conversation_id": "x", "customer_message": "hi"}
    with pytest.raises(AssertionError):
        assert_no_leakage({"train": [rec], "test": [rec]})
