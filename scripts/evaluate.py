#!/usr/bin/env python
"""Evaluate baselines + proposed agent on the golden set and write reports."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_all, results_table, write_predictions  # noqa: E402
from src.evaluation.human_agreement import agreement_from_csv  # noqa: E402
from src.evaluation.report import write_reports  # noqa: E402
from src.paths import REPORTS  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = evaluate_all()
    write_predictions(bundle)
    write_reports(bundle)
    table = results_table(bundle)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    agreement = agreement_from_csv()
    (REPORTS / "human_agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    print("human_agreement:", json.dumps(agreement))
    proposed = bundle["systems"]["proposed"]
    print(
        "HEADLINE intent_macro_f1={:.3f} escalation_f1={:.3f} false_auto_handle={:.3f}".format(
            proposed["intent"]["macro_f1"],
            proposed["escalation"]["f1_escalate"],
            proposed["escalation"]["false_auto_handle_rate"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
