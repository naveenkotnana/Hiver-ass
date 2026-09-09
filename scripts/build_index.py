#!/usr/bin/env python
"""Build the train-only retrieval index."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_split  # noqa: E402
from src.retrieval.index import build_index, save_index  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train = load_split("train")
    index = build_index(train)
    save_index(index)
    print(json.dumps({"n_indexed": len(index.records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
