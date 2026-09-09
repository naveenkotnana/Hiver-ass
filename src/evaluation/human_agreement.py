"""Agreement between the automatic judge and a human rater.

Human cells must be filled by a real annotator (the developer in this repo).
Empty human scores are skipped — we never invent them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.paths import DATA


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if len(x) < 3:
        return None
    if method == "pearson":
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])
    # Spearman via ranks
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def agreement_from_csv(path: Path | None = None) -> dict[str, Any]:
    path = path or (DATA / "judge_human_validation.csv")
    if not path.exists():
        return {
            "n": 0,
            "status": "missing_file",
            "note": f"{path} not found. Run python scripts/annotate_judge.py after evaluation.",
        }
    df = pd.read_csv(path)
    needed = {"example_id", "human_score", "llm_judge_score", "human_grounded", "llm_grounded"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    scored = df.dropna(subset=["human_score", "llm_judge_score"]).copy()
    if scored.empty:
        return {
            "n": 0,
            "status": "no_human_labels",
            "note": "Human scores are empty. Run scripts/annotate_judge.py and fill them; do not invent numbers.",
        }

    human = scored["human_score"].astype(float).to_numpy()
    judge = scored["llm_judge_score"].astype(float).to_numpy()
    exact = float(np.mean(human == judge))
    within1 = float(np.mean(np.abs(human - judge) <= 1))
    kappa = None
    try:
        kappa = float(cohen_kappa_score(human.astype(int), judge.astype(int)))
    except Exception:
        kappa = None

    hg = scored["human_grounded"].astype(str).str.lower().isin(["true", "1", "yes"])
    jg = scored["llm_grounded"].astype(str).str.lower().isin(["true", "1", "yes"])
    grounded_agree = float(np.mean(hg.to_numpy() == jg.to_numpy()))
    try:
        grounded_kappa = float(cohen_kappa_score(hg.astype(int), jg.astype(int)))
    except Exception:
        grounded_kappa = None

    return {
        "n": int(len(scored)),
        "status": "ok",
        "pearson": _corr(human, judge, "pearson"),
        "spearman": _corr(human, judge, "spearman"),
        "exact_agreement": exact,
        "within_1_agreement": within1,
        "cohen_kappa_score": kappa,
        "grounded_agreement": grounded_agree,
        "grounded_kappa": grounded_kappa,
        "human_mean": float(human.mean()),
        "judge_mean": float(judge.mean()),
    }
