#!/usr/bin/env python
"""One-command reviewer path: data → train → index → evaluate → sample predictions."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMOS = [
    "I was charged twice for my Apple Music subscription",
    "how do I turn off Wi-Fi Assist on iOS 11?",
    "my iPhone 7 keeps restarting after I updated to iOS 11",
    "my Apple ID is locked and I never got the reset email",
    "iPhone X will not stay connected to Wi-Fi",
]


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(extra or [])]
    print("\n===", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run("prepare_data.py")
    run("train.py")
    run("build_index.py")
    run("evaluate.py")
    print("\n=== sample predictions ===\n")
    for msg in DEMOS:
        print(f"\n----- {msg}")
        subprocess.run(
            [sys.executable, "-m", "src.cli", "--message", msg],
            check=True,
            cwd=str(ROOT),
        )
    print("\nDemo complete. Headline metrics are in reports/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
