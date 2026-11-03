"""The project's main model: an SVM with look-alike specialists.

Objectives 1, 2 and 3: one SVM reads every character. When its top two
guesses are a known look-alike pair, a small binary SVM for that pair makes the
final call, using only the regions of the character that SHAP says separate the
pair. Which pairs count, which regions each specialist uses and whether a
specialist is kept at all are all decided on training data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config, models
from src.utils import ModiError

C = config.LOOKALIKE
MODEL_PATH = config.PATHS.models / "main_svm_lookalike.joblib"


@dataclass
class Specialist:
    """A binary SVM for one look-alike pair, restricted to the chosen feature groups."""

    pair: tuple[str, str]
    scaler: StandardScaler
    clf: SVC
    columns: np.ndarray
    regions: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(X[:, self.columns]))


@dataclass
class LookalikeModel:
    """SVM + specialists. Has the interface the explainer and the app expect."""

    preprocessor: Any
    svm: SVC
    calibrator: Any                         # decision scores -> class probabilities
    classes: np.ndarray
    specialists: dict[tuple[str, str], Specialist] = field(default_factory=dict)
    feature_set: str = "fused"
    reducer: str = "lda"
    info: dict[str, Any] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.preprocessor.transform(X)

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.svm.decision_function(self.transform(X))

    def predict_proba(self, X: np.ndarray, reduced: bool = False) -> np.ndarray:
        """Calibrated probabilities of the SVM (columns follow ``classes``)."""
        d = self.svm.decision_function(X if reduced else self.transform(X))
        return self.calibrator.predict_proba(d)

    def top_two(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.scores(X), axis=1)[:, ::-1]
        return self.classes[order[:, 0]], self.classes[order[:, 1]]

    def decide(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Final labels, plus the SVM's own choice and which pair (if any) was consulted."""
        first, second = self.top_two(X)
        final = first.copy()
        used = np.full(len(X), None, dtype=object)
        for (a, b), sp in self.specialists.items():
            m = ((first == a) & (second == b)) | ((first == b) & (second == a))
            if m.any():
                final[m] = sp.predict(X[m])
                used[np.flatnonzero(m)] = [(a, b)] * int(m.sum())
        return {"final": final, "svm": first, "second": second, "specialist": used}


# ---------------------------------------------------------------------------
# Training pieces
# ---------------------------------------------------------------------------
def fit_svm(X: np.ndarray, y: np.ndarray, feature_set: str = "fused") -> tuple[Any, SVC]:
    """The main SVM on LDA-reduced features, with the configured settings."""
    pre = models.make_preprocessor(feature_set, "lda", len(np.unique(y)))
    return pre, models.make_base("svm").fit(pre.fit_transform(X, y), y)


def fit_calibrator(scores: np.ndarray, y: np.ndarray) -> Any:
    """Turn SVM decision scores into probabilities, using out-of-fold scores."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(scores, y)


def rank_groups(X: np.ndarray, y: np.ndarray, groups: list[dict[str, Any]], seed: int = config.SEED) -> np.ndarray:
    """Mean |SHAP| of each feature group for a binary SVM trained on one pair."""
    import shap

    np.random.seed(seed)                    # Kernel SHAP samples from numpy's global generator
    Xs = StandardScaler().fit_transform(X)
    clf = SVC(kernel="rbf", C=C.specialist_C, gamma="scale", class_weight="balanced", random_state=seed).fit(Xs, y)
    base = Xs.mean(axis=0)
    picks = np.random.default_rng(seed).choice(len(Xs), size=min(C.shap_images, len(Xs)), replace=False)

    def game(masks: np.ndarray, x: np.ndarray) -> np.ndarray:
        M = np.repeat(base[None, :], len(masks), axis=0)
        for g, grp in enumerate(groups):
            on = masks[:, g] > 0.5
            if on.any():
                M[np.ix_(on, grp["columns"])] = x[grp["columns"]]
        return clf.decision_function(M)

    totals = np.zeros(len(groups))
    for i in picks:
        x = Xs[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = shap.KernelExplainer(lambda m, x=x: game(np.atleast_2d(m), x), np.zeros((1, len(groups))))
            sv = ex.shap_values(np.ones((1, len(groups))), nsamples=C.shap_samples, silent=True)
        totals += np.abs(np.asarray(sv).ravel())
    return totals / len(picks)


def choose_columns(X: np.ndarray, y: np.ndarray, groups: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Columns of the feature groups SHAP ranks highest for this pair."""
    keep = np.argsort(rank_groups(X, y, groups))[::-1][:C.top_regions]
    return np.concatenate([groups[g]["columns"] for g in keep]), [groups[g]["name"] for g in keep]


def fit_specialist(pair: tuple[str, str], X: np.ndarray, y: np.ndarray, groups: list[dict[str, Any]]) -> Specialist:
    """Train one pair's specialist on that pair's training images."""
    m = np.isin(y, pair)
    cols, names = choose_columns(X[m], y[m], groups)
    scaler = StandardScaler().fit(X[m][:, cols])
    clf = SVC(kernel="rbf", C=C.specialist_C, gamma="scale", class_weight="balanced", random_state=config.SEED)
    clf.fit(scaler.transform(X[m][:, cols]), y[m])
    return Specialist(tuple(pair), scaler, clf, cols, names)


def save(model: LookalikeModel, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load(path: Path = MODEL_PATH) -> LookalikeModel:
    if not Path(path).exists():
        raise ModiError("The trained model was not found. Run: python scripts/run_lookalike.py")
    return joblib.load(path)
