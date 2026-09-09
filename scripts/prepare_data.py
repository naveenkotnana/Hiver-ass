#!/usr/bin/env python
"""Prepare sample (or Kaggle) data, splits, and the golden set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import prepare_dataset  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Try to download the full Kaggle dataset instead of the sample.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    stats = prepare_dataset(force_sample=not args.kaggle, seed=args.seed)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
