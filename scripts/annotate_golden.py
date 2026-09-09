#!/usr/bin/env python
"""Interactive golden-set review.

The bundled golden set is already developer-labeled via the protocol in
docs/golden_set.md. Use this script when you download the real Kaggle dump
and need to label a fresh sample.

Usage:
    python scripts/annotate_golden.py
    python scripts/annotate_golden.py --unlabeled data/golden_set.unlabeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import intent_names  # noqa: E402
from src.data.labeling import expected_action  # noqa: E402
from src.paths import DATA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled", type=Path, default=DATA / "golden_set.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "golden_set.reviewed.jsonl")
    args = parser.parse_args()
    names = intent_names()
    print("Intents:", ", ".join(names))
    print("Actions: AUTO_HANDLE / ESCALATE")
    print("Press Enter to keep the suggested value.\n")

    out_rows = []
    with args.unlabeled.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            print("=" * 72)
            print("id:", row.get("example_id"))
            print("message:", row.get("customer_message"))
            print("context:", row.get("conversation_context") or "(none)")
            suggested_intent = row.get("true_intent") or ""
            intent = input(f"intent [{suggested_intent}]: ").strip() or suggested_intent
            if intent not in names:
                print("unknown intent, keeping", suggested_intent)
                intent = suggested_intent
            sug_act, sug_reason = expected_action(row.get("customer_message") or "", intent)
            action = input(f"action [{sug_act}]: ").strip() or sug_act
            reason = input(f"reason [{sug_reason}]: ").strip() or sug_reason
            notes = input("notes: ").strip() or row.get("notes") or ""
            row["true_intent"] = intent
            row["expected_action"] = action
            row["escalation_reason"] = reason if action == "ESCALATE" else ""
            row["notes"] = notes
            out_rows.append(row)

    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", args.out, "n=", len(out_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
