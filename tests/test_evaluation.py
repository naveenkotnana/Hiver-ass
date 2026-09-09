from __future__ import annotations

from src.evaluation.human_agreement import agreement_from_csv
from src.evaluation.judge import heuristic_judge
from src.evaluation.metrics import escalation_metrics, intent_metrics, retrieval_metrics


def test_intent_metrics_perfect():
    labels = ["a", "b"]
    m = intent_metrics(["a", "b", "a"], ["a", "b", "a"], labels)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_false_auto_handle_rate():
    y_true = ["ESCALATE", "ESCALATE", "AUTO_HANDLE"]
    y_pred = ["AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE"]
    m = escalation_metrics(y_true, y_pred)
    assert abs(m["false_auto_handle_rate"] - 0.5) < 1e-9


def test_retrieval_mrr():
    m = retrieval_metrics(["x", "y"], [["a", "x"], ["y"]], ks=(1, 2))
    assert abs(m["mrr"] - (0.5 + 1.0) / 2) < 1e-9
    assert m["recall@1"] == 0.5
    assert m["recall@2"] == 1.0


def test_heuristic_judge_penalizes_refund_promise():
    bad = heuristic_judge(
        "I was charged twice",
        "Your refund has been issued for $12.99 and will arrive tomorrow.",
        [{"customer_message": "charged twice", "brand_response": "Please DM us the receipt."}],
    )
    assert bad["unsupported_claims"] <= 1
    assert bad["overall"] <= 2


def test_agreement_empty(tmp_path):
    p = tmp_path / "j.csv"
    p.write_text(
        "example_id,human_score,llm_judge_score,human_grounded,llm_grounded\n"
        "g1,,,,\n",
        encoding="utf-8",
    )
    out = agreement_from_csv(p)
    assert out["n"] == 0
    assert out["status"] == "no_human_labels"
