from __future__ import annotations

from src.data.clean import is_usable_message, normalize_text
from src.data.conversations import reconstruct_pairs
from src.data.silver_labels import silver_label


def test_normalize_strips_urls_and_handles():
    text = "hey @randomUser check https://t.co/abc123 &amp; #ios11 @AppleSupport"
    out = normalize_text(text)
    assert "@USER" in out
    assert "@AppleSupport" in out
    assert "URL" in out
    assert "&amp;" not in out
    assert "ios11" in out


def test_unusable_short_messages():
    assert not is_usable_message("ok")
    assert is_usable_message("my iPhone will not turn on at all")


def test_reconstruct_pairs_links_brand_reply(tiny_tweets):
    tweets, _ = tiny_tweets
    pairs = reconstruct_pairs(tweets, brand="AppleSupport")
    assert pairs
    rec = pairs[0]
    assert rec["brand"] == "AppleSupport"
    assert rec["customer_message"]
    assert rec["brand_response"]
    assert rec["conversation_id"]
    assert rec["metadata"]["inbound_tweet_id"]
    assert rec["metadata"]["response_tweet_id"]


def test_silver_label_how_to():
    assert silver_label("how do I turn off Wi-Fi Assist") == "how_to_feature"


def test_silver_label_refund():
    assert silver_label("I want a refund for Apple Music") == "billing_subscription"
