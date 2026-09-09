"""TF-IDF + logistic regression intent classifier with optional calibration.

Confidence is a Platt-scaled score from CalibratedClassifierCV when a
validation split is provided. It is a ranking-quality score, not a claim
of perfectly calibrated probabilities.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from src.config import intent_examples, intent_names, load_config
from src.paths import ARTIFACTS


class IntentClassifier:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.clf: Any = None
        self.labels: list[str] = intent_names()
        self.prototypes: np.ndarray | None = None
        self.calibrated: bool = False

    def _vectorizer(self, spec: dict[str, Any]) -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=tuple(spec["ngram_range"]),
            min_df=int(spec["min_df"]),
            max_features=int(spec["max_features"]),
            sublinear_tf=bool(spec.get("sublinear_tf", True)),
            lowercase=True,
        )

    def _prototype_matrix(self, X: sparse.spmatrix, y: list[str]) -> np.ndarray:
        # Class centroids in TF-IDF space + taxonomy example encodings.
        mats = []
        for label in self.labels:
            idx = [i for i, yi in enumerate(y) if yi == label]
            if idx:
                centroid = np.asarray(X[idx].mean(axis=0)).ravel()
            else:
                centroid = np.zeros(X.shape[1])
            mats.append(centroid)
        proto = np.vstack(mats)
        # Mix in taxonomy example documents if the vectorizer already exists.
        examples = intent_examples()
        extra_docs = []
        extra_ids = []
        for i, label in enumerate(self.labels):
            extra_docs.extend(examples.get(label) or [])
            extra_ids.extend([i] * len(examples.get(label) or []))
        if extra_docs and self.vectorizer is not None:
            extra_x = self.vectorizer.transform(extra_docs)
            for i, label_i in enumerate(extra_ids):
                proto[label_i] = 0.7 * proto[label_i] + 0.3 * np.asarray(extra_x[i].mean(axis=0)).ravel()
        proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
        return proto

    def _augment(self, X: sparse.spmatrix) -> sparse.spmatrix:
        Xn = normalize(X)
        if self.prototypes is None:
            return Xn
        sims = Xn @ self.prototypes.T
        if sparse.issparse(sims):
            sims = sims.toarray()
        return sparse.hstack([Xn, sims], format="csr")

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        *,
        val_texts: list[str] | None = None,
        val_labels: list[str] | None = None,
        spec: dict[str, Any] | None = None,
    ) -> "IntentClassifier":
        cfg = spec or load_config()["classifier"]
        self.labels = sorted(set(labels), key=lambda x: intent_names().index(x) if x in intent_names() else x)
        self.vectorizer = self._vectorizer(cfg)
        X = self.vectorizer.fit_transform(texts)
        self.prototypes = self._prototype_matrix(X, labels)
        X_aug = self._augment(X)
        base = LogisticRegression(
            C=float(cfg.get("C", 2.0)),
            class_weight=cfg.get("class_weight") or None,
            max_iter=int(cfg.get("max_iter", 400)),
            solver="lbfgs",
        )
        if val_texts and val_labels and len(set(val_labels)) > 1:
            base.fit(X_aug, labels)
            method = cfg.get("calibration") or "sigmoid"
            try:
                from sklearn.frozen import FrozenEstimator

                calibrator = CalibratedClassifierCV(FrozenEstimator(base), method=method)
            except Exception:
                calibrator = CalibratedClassifierCV(base, method=method, cv="prefit")
            Xv = self._augment(self.vectorizer.transform(val_texts))
            calibrator.fit(Xv, val_labels)
            self.clf = calibrator
            self.calibrated = True
        else:
            base.fit(X_aug, labels)
            self.clf = base
            self.calibrated = False
        self.labels = list(self.clf.classes_)
        return self

    def predict_one(self, text: str) -> dict[str, Any]:
        if self.vectorizer is None or self.clf is None:
            raise RuntimeError("Classifier is not fitted")
        X = self._augment(self.vectorizer.transform([text]))
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X)[0]
        else:
            # Decision function fallback — not a probability.
            scores = self.clf.decision_function(X)
            scores = np.atleast_2d(scores)
            e = np.exp(scores - scores.max())
            proba = (e / e.sum(axis=1, keepdims=True))[0]
        order = np.argsort(-proba)
        intent = str(self.labels[int(order[0])])
        alternatives = [
            {"intent": str(self.labels[int(i)]), "score": float(proba[int(i)])}
            for i in order[1:4]
        ]
        return {
            "intent": intent,
            "confidence": float(proba[int(order[0])]),
            "alternatives": alternatives,
            "calibrated": self.calibrated,
        }

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict_one(t) for t in texts]


def save_classifier(model: IntentClassifier, path=None) -> None:
    path = path or (ARTIFACTS / "models" / "intent_classifier.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path=None) -> IntentClassifier:
    path = path or (ARTIFACTS / "models" / "intent_classifier.joblib")
    return joblib.load(path)
