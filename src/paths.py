"""Repository paths. All modules should resolve files through here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
SAMPLE = DATA / "sample"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


def ensure_dirs() -> None:
    for path in (RAW, SAMPLE, PROCESSED, ARTIFACTS, REPORTS, ARTIFACTS / "models", ARTIFACTS / "index"):
        path.mkdir(parents=True, exist_ok=True)
