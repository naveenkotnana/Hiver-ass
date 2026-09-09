"""Download the Kaggle corpus when credentials exist; otherwise build the sample."""

from __future__ import annotations

import logging
import os
import subprocess
import zipfile
from pathlib import Path

from src.config import load_config
from src.data.sample_corpus import write_sample_corpus
from src.paths import RAW, SAMPLE, ensure_dirs

logger = logging.getLogger(__name__)


def kaggle_credentials_present() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def try_download_kaggle() -> Path | None:
    """Attempt `kaggle datasets download`. Return path to twcs.csv or None."""

    if not kaggle_credentials_present():
        logger.info("No Kaggle credentials; skipping full-dataset download.")
        return None

    ensure_dirs()
    cfg = load_config()
    dataset = cfg["data"]["kaggle_dataset"]
    target = RAW / "twcs.csv"
    if target.exists():
        return target

    zip_path = RAW / "customer-support-on-twitter.zip"
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(RAW),
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Kaggle download failed (%s). Using the bundled sample.", exc)
        return None

    archives = list(RAW.glob("*.zip"))
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW)
    if target.exists():
        return target
    matches = list(RAW.rglob("twcs.csv"))
    return matches[0] if matches else None


def prepare_source_csv(*, force_sample: bool = False, seed: int = 42, per_intent: int = 240) -> Path:
    """Return a tweet CSV: full Kaggle file if available, else the sample."""

    ensure_dirs()
    if not force_sample:
        kaggle_csv = try_download_kaggle()
        if kaggle_csv is not None:
            logger.info("Using Kaggle CSV at %s", kaggle_csv)
            return kaggle_csv
        existing = RAW / "twcs.csv"
        if existing.exists():
            return existing

    tweets_path, _gold_path = write_sample_corpus(per_intent=per_intent, seed=seed)
    logger.info("Wrote sample corpus to %s", tweets_path)
    return Path(tweets_path)


def brand_counts(csv_path: Path, top_n: int = 15) -> list[tuple[str, int]]:
    import pandas as pd

    df = pd.read_csv(csv_path, usecols=["author_id", "inbound"], dtype=str)
    inbound = df["inbound"].astype(str).str.lower().isin(["false", "0", "no"])
    brands = df.loc[inbound, "author_id"].value_counts().head(top_n)
    return list(brands.items())
