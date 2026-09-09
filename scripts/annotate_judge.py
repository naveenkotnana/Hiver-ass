#!/usr/bin/env python
"""Sample proposed-agent replies for human vs judge agreement.

Fills llm_judge_* columns from the automatic judge. Leaves human_* empty
unless --score-now is used (developer reads each reply and types a score).

    python scripts/annotate_judge.py
    python scripts/annotate_judge.py --score-now
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.judge import heuristic_judge  # noqa: E402
from src.paths import ARTIFACTS, DATA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-now", action="store_true")
    parser.add_argument("--out", type=Path, default=DATA / "judge_human_validation.csv")
    args = parser.parse_args()

    pred_path = ARTIFACTS / "predictions" / "proposed.jsonl"
    if not pred_path.exists():
        print("Run python scripts/evaluate.py first.", file=sys.stderr)
        return 2
    rows = []
    with pred_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sample = rows[: args.size]

    records = []
    for row in sample:
        ev = row.get("evidence") or []
        judge = heuristic_judge(row["customer_message"], row.get("reply") or "", ev)
        rec = {
            "example_id": row["example_id"],
            "customer_message": row["customer_message"],
            "reply": row.get("reply") or "",
            "human_score": "",
            "llm_judge_score": judge["overall"],
            "human_grounded": "",
            "llm_grounded": bool(judge["grounded"]),
            "human_notes": "",
        }
        if args.score_now:
            print("=" * 72)
            print("id:", rec["example_id"])
            print("customer:", rec["customer_message"])
            print("reply:", rec["reply"])
            print("evidence:")
            for i, hit in enumerate(ev[:2], start=1):
                print(f"  {i}. {hit.get('customer_message')}")
                print(f"     {hit.get('brand_response')}")
            print("score 0-4 (0=unusable, 2=acceptable, 4=excellent)")
            rec["human_score"] = int(input("human_score: ").strip() or "2")
            rec["human_grounded"] = input("human_grounded true/false: ").strip().lower() in {
                "true",
                "1",
                "y",
                "yes",
            }
            rec["human_notes"] = input("notes: ").strip()
        records.append(rec)

    pd.DataFrame(records).to_csv(args.out, index=False)
    print("wrote", args.out, "n=", len(records), "(human cells empty)" if not args.score_now else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
