"""Load raw tweets and processed conversation JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.paths import PROCESSED, RAW, SAMPLE

TWEET_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def find_raw_tweets() -> Path | None:
    for candidate in (
        RAW / "twcs.csv",
        RAW / "twcs_sample.csv",
        SAMPLE / "tweets.csv",
    ):
        if candidate.exists():
            return candidate
    if RAW.exists():
        csvs = list(RAW.glob("*.csv"))
        if csvs:
            return csvs[0]
    return None


def load_tweets(path: Path | None = None) -> pd.DataFrame:
    path = path or find_raw_tweets()
    if path is None:
        raise FileNotFoundError(
            "No tweet CSV found. Run `python scripts/prepare_data.py` "
            "(generates the sample) or place twcs.csv under data/raw/."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Kaggle file sometimes uses slightly different header casing.
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)
    return df


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def processed_split_path(split: str) -> Path:
    return PROCESSED / f"{split}.jsonl"


def load_split(split: str) -> list[dict[str, Any]]:
    path = processed_split_path(split)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/prepare_data.py")
    return read_jsonl(path)
