"""Run the agent and baselines on the held-out golden set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.agent.support_agent import SupportAgent
from src.baselines.pipeline import SimpleBaselineAgent, majority_agent
from src.baselines.retrieval_baseline import RetrievalBaseline
from src.baselines.tfidf_classifier import load_baseline
from src.config import intent_names, load_config
from src.evaluation.calibration import confidence_bins, ece
from src.evaluation.judge import aggregate_judge_scores, judge_response
from src.evaluation.metrics import escalation_metrics, intent_metrics, retrieval_metrics
from src.paths import ARTIFACTS, DATA, REPORTS, ensure_dirs


def load_golden() -> list[dict[str, Any]]:
    path = DATA / "golden_set.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _run_system(name: str, handler: Callable[..., dict[str, Any]], golden: list[dict[str, Any]]) -> dict[str, Any]:
    labels = intent_names()
    y_true_i, y_pred_i = [], []
    y_true_e, y_pred_e = [], []
    confs: list[float] = []
    retrieved_intents: list[list[str]] = []
    gold_intents: list[str] = []
    judge_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for ex in golden:
        result = handler(ex["customer_message"], context=ex.get("conversation_context") or "")
        y_true_i.append(ex["true_intent"])
        y_pred_i.append(result.get("intent") or "other")
        y_true_e.append(ex["expected_action"])
        y_pred_e.append(result.get("decision") or "ESCALATE")
        confs.append(float(result.get("intent_confidence") or 0.0))
        ev = result.get("evidence") or []
        retrieved_intents.append([h.get("intent") or "" for h in ev])
        gold_intents.append(ex["true_intent"])
        judge = judge_response(ex["customer_message"], result.get("draft_reply") or result.get("reply") or "", ev)
        judge_rows.append(judge)
        predictions.append(
            {
                "example_id": ex["example_id"],
                "true_intent": ex["true_intent"],
                "pred_intent": result.get("intent"),
                "intent_confidence": result.get("intent_confidence"),
                "expected_action": ex["expected_action"],
                "pred_decision": result.get("decision"),
                "reason": result.get("reason"),
                "reply": result.get("draft_reply") or result.get("reply"),
                "grounded": result.get("grounded"),
                "judge_overall": judge.get("overall"),
                "judge_grounded": judge.get("grounded"),
                "judge_unsupported": judge.get("unsupported_claims"),
                "customer_message": ex["customer_message"],
                "notes": ex.get("notes"),
                "evidence": ev,
            }
        )

    intent = intent_metrics(y_true_i, y_pred_i, labels)
    escal = escalation_metrics(y_true_e, y_pred_e)
    retr = retrieval_metrics(gold_intents, retrieved_intents, ks=(1, 3, 5))
    judge_agg = aggregate_judge_scores(judge_rows)
    return {
        "name": name,
        "intent": intent,
        "escalation": escal,
        "retrieval": retr,
        "judge": judge_agg,
        "calibration": {
            "ece": ece(y_true_i, y_pred_i, confs),
            "bins": confidence_bins(y_true_i, y_pred_i, confs),
        },
        "predictions": predictions,
    }


def evaluate_all() -> dict[str, Any]:
    ensure_dirs()
    golden = load_golden()
    agent = SupportAgent()
    majority = majority_agent([g["true_intent"] for g in golden])
    # Majority should be fit on TRAIN labels, not golden. Reload train silver if present.
    from src.data.load import load_split

    train = load_split("train")
    majority = majority_agent([r.get("silver_intent") or "other" for r in train])

    simple = SimpleBaselineAgent(load_baseline(), RetrievalBaseline())

    results = {
        "majority": _run_system("majority", majority.handle, golden),
        "tfidf_baseline": _run_system("tfidf_baseline", simple.handle, golden),
        "proposed": _run_system("proposed", agent.handle, golden),
    }
    return {"n_golden": len(golden), "systems": results}


def write_predictions(bundle: dict[str, Any]) -> None:
    pred_dir = ARTIFACTS / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in bundle["systems"].items():
        path = pred_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in payload["predictions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Drop bulky predictions from the metrics snapshot
        payload = dict(payload)
        payload.pop("predictions", None)
        bundle["systems"][name] = payload
        # Keep a compact pointer
        bundle["systems"][name]["predictions_path"] = str(path)


def confusion_plot(bundle: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    rows: list[dict[str, Any]] = []
    with (ARTIFACTS / "predictions" / "proposed.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    labels = intent_names()
    y_true = [r["true_intent"] for r in rows]
    y_pred = [r["pred_intent"] for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title("Proposed agent — golden-set confusion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    _ = bundle


def results_table(bundle: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, sys in bundle["systems"].items():
        rows.append(
            {
                "system": name,
                "intent_accuracy": sys["intent"]["accuracy"],
                "intent_macro_f1": sys["intent"]["macro_f1"],
                "escalation_f1": sys["escalation"]["f1_escalate"],
                "false_auto_handle_rate": sys["escalation"]["false_auto_handle_rate"],
                "recall@5": sys["retrieval"]["recall@5"],
                "mrr": sys["retrieval"]["mrr"],
                "judge_overall_mean": sys["judge"]["overall"]["mean"],
                "hallucination_rate": sys["judge"]["hallucination_rate"],
            }
        )
    return pd.DataFrame(rows)
