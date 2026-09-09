"""Write reports/ artifacts from a measured evaluation bundle."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluate import confusion_plot, load_golden, results_table
from src.paths import ARTIFACTS, REPORTS


def _load_preds(name: str) -> list[dict[str, Any]]:
    path = ARTIFACTS / "predictions" / f"{name}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" if df[c].dtype == object else "---:" for c in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(floatfmt.format(val))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failure_modes(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster actual errors into the assignment's top-5 failure modes."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        intent_wrong = row["true_intent"] != row["pred_intent"]
        esc_wrong = row["expected_action"] != row["pred_decision"]
        false_auto = row["expected_action"] == "ESCALATE" and row["pred_decision"] == "AUTO_HANDLE"
        extra_esc = row["expected_action"] == "AUTO_HANDLE" and row["pred_decision"] == "ESCALATE"
        poor_judge = int(row.get("judge_overall") or 0) <= 1
        ungrounded = not row.get("grounded")
        hallu = int(row.get("judge_unsupported") or 4) <= 1
        msg = (row.get("customer_message") or "").lower()
        if not (intent_wrong or esc_wrong or poor_judge or hallu):
            continue

        if intent_wrong and row["true_intent"] == "other":
            buckets["other_absorbed_into_how_to"].append(row)
        elif false_auto:
            buckets["incorrect_escalation_false_auto_handle"].append(row)
        elif intent_wrong and row["true_intent"] in {"ios_update_bug", "hardware_device", "connectivity"}:
            buckets["ambiguous_intent_symptom_vs_cause"].append(row)
        elif intent_wrong:
            buckets["intent_misclassification"].append(row)
        elif ungrounded:
            buckets["insufficient_historical_evidence"].append(row)
        elif hallu:
            buckets["hallucinated_or_unsafe_claim"].append(row)
        elif extra_esc:
            buckets["incorrect_escalation"].append(row)
        elif esc_wrong:
            buckets["incorrect_escalation"].append(row)
        elif poor_judge:
            buckets["poor_retrieval_or_reply"].append(row)

    # Rank by frequency, keep top 5 with at least one real example.
    ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    out = []
    for mode, items in ranked:
        if not items:
            continue
        ex = items[0]
        out.append(
            {
                "failure_mode": mode,
                "count": len(items),
                "example_id": ex["example_id"],
                "customer_message": ex["customer_message"],
                "expected_intent": ex["true_intent"],
                "actual_intent": ex["pred_intent"],
                "expected_action": ex["expected_action"],
                "actual_action": ex["pred_decision"],
                "judge_overall": ex.get("judge_overall"),
            }
        )
        if len(out) == 5:
            break
    return out


FAILURE_COPY = {
    "ambiguous_intent_symptom_vs_cause": {
        "why": "The message names a symptom (battery, Wi-Fi) that also belongs to another intent, while the cause (iOS update, hardware age) is the gold label.",
        "hypothesis": "Overlapping keyword features; silver training labels are even noisier on mixed-symptom tweets.",
        "fix": "Add a cause-vs-symptom feature (e.g. 'after updating') or a small set of contrastive examples; do not merge those intents — they need different actions.",
    },
    "intent_misclassification": {
        "why": "The classifier assigned the wrong taxonomy label, so retrieval was steered toward the wrong historical cluster.",
        "hypothesis": "Class overlap plus silver-label noise on train; minority intents have fewer distinctive n-grams.",
        "fix": "Inspect the confusion matrix cells with highest volume; add targeted features or a second-stage binary for that pair.",
    },
    "insufficient_historical_evidence": {
        "why": "Top-k cosine neighbors were below the similarity floor, so the generator refused to copy a resolution.",
        "hypothesis": "Held-out phrasing that does not n-gram-overlap the train set; TF-IDF is lexical.",
        "fix": "Optional embedding retriever when a model can be cached locally; keep the refusal (do not hallucinate a policy).",
    },
    "hallucinated_or_unsafe_claim": {
        "why": "The judge flagged invented money, refunds, or account actions.",
        "hypothesis": "Copied a historical tweet that itself contained a specific commitment, or the LLM path leaked a promise.",
        "fix": "Tighten the sanitizer; treat any '$' / 'refunded' as an automatic escalate + rewrite.",
    },
    "incorrect_escalation_false_auto_handle": {
        "why": "Policy auto-handled a case the golden protocol says needs a human.",
        "hypothesis": "Intent looked like how-to/troubleshooting and evidence was similar, while the gold label saw account-specific risk.",
        "fix": "Raise the cost of false auto-handles: broaden ACCOUNT_ACTION patterns; never auto-handle backup + 'photos gone'.",
    },
    "incorrect_escalation": {
        "why": "Escalation decision disagreed with the labeling guide.",
        "hypothesis": "Policy is conservative (extra escalations) which hurts AUTO_HANDLE recall on purpose.",
        "fix": "If extra escalations dominate, narrow always-escalate intents after measuring false auto-handle, not before.",
    },
    "other_absorbed_into_how_to": {
        "why": "Vague or off-topic tweets were classified as how-to questions, so the agent drafted a product-support reply instead of declining or escalating.",
        "hypothesis": "The `other` class has weak n-grams; `how_to_feature` is a magnet because questions and @AppleSupport mentions look like support requests. Silver training labels also dump ties into other, so the decision boundary is messy.",
        "fix": "Add a cheap first-pass: if the message has no product symptom keywords, emit `other` and escalate. Evaluate that rule on val, not on golden.",
    },
    "poor_retrieval_or_reply": {
        "why": "Judge overall score ≤ 1 even when intent may have been right.",
        "hypothesis": "Nearest neighbor matched lexical overlap but not the resolution type.",
        "fix": "Rerank with overlap on issue verbs (refund vs how-to) not just tokens.",
    },
    "repeated_unresolved_issue": {
        "why": "Follow-up language ('still', 'again') was not treated as a hard escalate.",
        "hypothesis": "The repeated-issue regex is narrow.",
        "fix": "Treat any multi-turn previous_context as a prior-failure signal.",
    },
}


def write_failure_analysis(modes: list[dict[str, Any]]) -> str:
    lines = [
        "# Failure analysis",
        "",
        "Generated from **actual** proposed-agent predictions on the golden set.",
        "Examples are copied from `artifacts/predictions/proposed.jsonl`. They are not invented.",
        "",
    ]
    if not modes:
        lines.append("No clustered failures were found in this run.")
        return "\n".join(lines)

    for i, mode in enumerate(modes, start=1):
        meta = FAILURE_COPY.get(
            mode["failure_mode"],
            {
                "why": "See the example.",
                "hypothesis": "Needs manual inspection.",
                "fix": "Add a regression test around this example_id.",
            },
        )
        lines.extend(
            [
                f"## {i}. {mode['failure_mode'].replace('_', ' ')}",
                "",
                f"- Count in cluster: **{mode['count']}**",
                f"- Example ID: `{mode['example_id']}`",
                f"- Customer message: {mode['customer_message']}",
                f"- Expected: intent `{mode['expected_intent']}`, action `{mode['expected_action']}`",
                f"- Actual: intent `{mode['actual_intent']}`, action `{mode['actual_action']}`",
                f"- Judge overall: {mode.get('judge_overall')}",
                f"- Why the system failed: {meta['why']}",
                f"- Hypothesis: {meta['hypothesis']}",
                f"- Potential fix: {meta['fix']}",
                "",
            ]
        )
    return "\n".join(lines)


def write_reports(bundle: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    table = results_table(bundle)
    table.to_csv(REPORTS / "results.csv", index=False)
    metrics_only = json.loads(json.dumps(bundle))
    (REPORTS / "results.json").write_text(json.dumps(metrics_only, indent=2), encoding="utf-8")

    confusion_plot(bundle, REPORTS / "confusion_matrix.png")

    preds = _load_preds("proposed")
    modes = failure_modes(preds)
    (REPORTS / "failure_analysis.md").write_text(write_failure_analysis(modes), encoding="utf-8")
    (REPORTS / "failure_modes.json").write_text(json.dumps(modes, indent=2), encoding="utf-8")

    golden = load_golden()
    intent_counts = Counter(g["true_intent"] for g in golden)
    action_counts = Counter(g["expected_action"] for g in golden)
    proposed = bundle["systems"]["proposed"]

    headline = proposed["intent"]["macro_f1"]
    headline_md = f"""# What is misleading about my headline number?

**Headline reported in README / REPORT:** proposed-agent intent **macro-F1 = {headline:.3f}** on the {bundle['n_golden']}-example golden set.

That number is a real measurement from `python scripts/evaluate.py`. It is also easy to over-read.

## Class imbalance

Golden intent counts:

{json.dumps(dict(intent_counts), indent=2)}

Macro-F1 treats a 16-example intent the same as a 30-example intent. Accuracy would look friendlier if the majority class is easy. We report both; neither is the whole story. Per-class precision/recall in `reports/results.json` is the honest view.

## Easy examples

The sample corpus is template-generated. Many how-to questions contain the phrase "how do I", which both the silver labeler and the classifier pick up. Those examples inflate every system, including the TF-IDF baseline. They are still valid support tickets, but they are not the tickets that decide whether you trust auto-handle.

## Golden set size

n = {bundle['n_golden']}. A 3-point macro-F1 gap can be a handful of items. We do not bootstrap confidence intervals in the demo path; do not treat the third decimal as stable.

## Sampling bias

The golden set is stratified by intent from the **test conversations of a synthetic Apple Support–style sample**, not a random draw from 2017 Twitter. Real AppleSupport volume in the Kaggle dump is dominated by iOS 11 / iPhone issues in a specific month. Temporal and product mix will shift.

## Temporal drift

The official corpus is from 2017 (iOS 11, iPhone X launch). An agent trained on it would be wrong about later products. We did not pretend otherwise.

## Conversation leakage

We split on `conversation_id` and index **train only**. Tests assert no ID overlap. If someone rebuilt the index on all pairs, retrieval metrics would be fictionally high. The current Recall@k is same-intent recall, not "did we retrieve this exact tweet" — the exact tweet is held out on purpose.

## Judge bias

The default judge is a **heuristic** (token overlap + promise regex + Apple-style cues). It is reproducible and strict about invented refunds, but it can reward replies that share words with the ticket without being useful. Copying a *wrong* historical neighbor still scores as "grounded" because the reply text *is* the evidence.

Human vs heuristic numbers, when `data/judge_human_validation.csv` has filled human cells, live in `reports/human_agreement.json` and `docs/judge_human_agreement.md`. Do not cite agreement if that file has `n=0`.

## Retrieval overlap

Because the sample uses shared templates, lexical retrieval can look strong without "understanding." That is a dataset artifact. On real tweets, expect Recall@5 to drop.

## Distribution mismatch

Train labels are **silver** (keywords). Golden labels follow the developer protocol. Some train items are intentionally mislabeled. Headline F1 is "system vs developer gold," not "system vs silver."

## Aggregate metrics hide minority-intent failures

A solid macro-F1 can still mean `other` or `order_purchase` is unusable. Escalation **false auto-handle rate** ({proposed['escalation']['false_auto_handle_rate']:.3f}) matters more for trust than intent F1. We would rather over-escalate.

## Escalation class mix

Golden expected actions: {json.dumps(dict(action_counts))}.
If most tickets are ESCALATE, a trivial "always escalate" baseline gets a high escalation F1. That is why we also report the majority baseline and false auto-handle rate.

## Bottom line

Use the headline to compare **proposed vs two baselines on the same frozen golden set**. Do not use it as a production go-live number.
"""
    (REPORTS / "headline_number.md").write_text(headline_md, encoding="utf-8")

    summary_md = [
        "# Evaluation summary",
        "",
        f"Golden-set size: **{bundle['n_golden']}**",
        "",
        _md_table(table),
        "",
        "Full JSON: `reports/results.json`. Confusion matrix: `reports/confusion_matrix.png`.",
        "Failure analysis: `reports/failure_analysis.md`.",
        "Headline caveats: `reports/headline_number.md`.",
        "",
    ]
    (REPORTS / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
