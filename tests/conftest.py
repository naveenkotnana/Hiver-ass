from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sample_corpus import generate_raw_tweets
from src.data.conversations import reconstruct_pairs
from src.data.silver_labels import silver_label
from src.data.splits import conversation_split


@pytest.fixture(scope="session")
def tiny_tweets():
    tweets, gold = generate_raw_tweets(per_intent=8, seed=0, extra_ambiguous=4)
    return tweets, gold


@pytest.fixture(scope="session")
def tiny_pairs(tiny_tweets):
    tweets, gold = tiny_tweets
    pairs = reconstruct_pairs(tweets, brand="AppleSupport")
    gold_map = dict(zip(gold["inbound_tweet_id"].astype(str), gold["gold_intent"]))
    for rec in pairs:
        rec["silver_intent"] = silver_label(rec["customer_message"])
        rec["metadata"]["gold_intent"] = gold_map.get(str(rec["metadata"]["inbound_tweet_id"]))
    assert len(pairs) > 20
    return pairs


@pytest.fixture(scope="session")
def tiny_splits(tiny_pairs):
    return conversation_split(tiny_pairs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0)
