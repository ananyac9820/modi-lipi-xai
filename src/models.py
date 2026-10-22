"""Stage 5 - Models: Random Forest, SVM (RBF), XGBoost and a stacking ensemble.

Layout of a trained model:

    fused features --> [ StandardScaler -> FamilyBalancer -> PCA/LDA ]   (preprocessor, fitted once)
                   --> StackingClassifier( RF, SVM-RBF, XGBoost  ->  logistic regression )

The three base learners inside the fitted stack are trained on the full
training set, so they double as the "each model on its own" results: no model
is trained twice. Class labels are integer-encoded once (``classes`` holds the
names) because XGBoost requires integer targets.

The SVM is used WITHOUT ``probability=True`` (which would add an internal
5-fold Platt-scaling cross-validation to every fit). The stack feeds the SVM's
decision scores to the meta-learner, whose logistic regression produces the
final calibrated probabilities.

No neural network is used anywhere in this module.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.metaestimators import available_if
from xgboost import XGBClassifier

from src import config, reduce
from src.features.fusion import FamilyBalancer
from src.utils import ModiError, get_logger

log = get_logger("models")
M = config.MODELS

BASE_MODELS: tuple[str, ...] = ("rf", "svm", "xgb")
ALL_MODELS: tuple[str, ...] = (*BASE_MODELS, "stack")
DISPLAY = {"rf": "Random Forest", "svm": "SVM (RBF)", "xgb": "XGBoost", "stack": "Stacking ensemble"}


def _has(attr: str):
    return lambda self: hasattr(self.estimator, attr)


class Timed(ClassifierMixin, BaseEstimator):
    """Transparent wrapper that records how long ``fit`` took.

    Lets the stacking ensemble report each base learner's own training time
    without training any model twice.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> "Timed":
        t = time.perf_counter()
        self.estimator_ = clone(self.estimator).fit(X, y, **kw)
        self.fit_seconds_ = time.perf_counter() - t
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    @available_if(_has("predict_proba"))
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    @available_if(_has("decision_function"))
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.decision_function(X)


def make_base(name: str) -> Any:
    """A fresh, unfitted base learner with the configured hyperparameters."""
    if name == "rf":
        return RandomForestClassifier(random_state=config.SEED, **M.rf)
    if name == "svm":
        return SVC(random_state=config.SEED, **M.svm)
    if name == "xgb":
        return XGBClassifier(random_state=config.SEED, **M.xgb)
    raise ModiError(f"Unknown model '{name}'. Choose from {BASE_MODELS}.")


def make_stack(base: tuple[str, ...] = BASE_MODELS) -> StackingClassifier:
    """Stacking ensemble over the given base learners."""
    final = make_pipeline(StandardScaler(),
                          LogisticRegression(C=M.stack_final_C, max_iter=M.stack_final_max_iter))
    cv = StratifiedKFold(n_splits=M.stack_cv, shuffle=True, random_state=config.SEED)
    return StackingClassifier(estimators=[(n, Timed(make_base(n))) for n in base], final_estimator=final,
                              cv=cv, stack_method="auto", passthrough=M.stack_passthrough, n_jobs=M.stack_n_jobs)


def make_preprocessor(feature_set: str = "fused", reducer: str | None = None, n_classes: int | None = None,
                      balance: bool | None = None) -> Pipeline:
    """Standardise -> balance descriptor families -> reduce dimensionality."""
    bal = config.FUSION.balance_families if balance is None else balance
    return Pipeline([("scale", StandardScaler()),
                     ("balance", FamilyBalancer(feature_set, enabled=bal)),
                     ("reduce", reduce.make_reducer(reducer, n_classes))])


@dataclass
class TrainedModels:
    """Everything needed to predict with any of the four models."""

    preprocessor: Pipeline
    stack: StackingClassifier
    classes: np.ndarray                     # class names; model targets are indices into this
    feature_set: str = "fused"
    reducer: str = "pca"
    info: dict[str, Any] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Raw feature matrix -> reduced representation."""
        return self.preprocessor.transform(X)

    def _estimator(self, name: str) -> Any:
        if name == "stack":
            return self.stack
        if name in self.stack.named_estimators_:
            return self.stack.named_estimators_[name]
        raise ModiError(f"Model '{name}' is not part of this bundle ({list(self.stack.named_estimators_)}).")

    def predict_idx(self, name: str, X: np.ndarray, reduced: bool = False) -> np.ndarray:
        """Predicted class indices from one model."""
        Z = X if reduced else self.transform(X)
        return np.asarray(self._estimator(name).predict(Z)).astype(int)

    def predict(self, name: str, X: np.ndarray, reduced: bool = False) -> np.ndarray:
        """Predicted class names from one model ('rf', 'svm', 'xgb' or 'stack')."""
        return self.classes[self.predict_idx(name, X, reduced)]

    def predict_proba(self, X: np.ndarray, reduced: bool = False) -> np.ndarray:
        """Class probabilities from the stacking ensemble (columns follow ``classes``)."""
        return self.stack.predict_proba(X if reduced else self.transform(X))


def train(X: np.ndarray, y: np.ndarray, feature_set: str = "fused", reducer: str | None = None,
          balance: bool | None = None, base: tuple[str, ...] = BASE_MODELS) -> TrainedModels:
    """Fit the preprocessor and the stacking ensemble (and thereby all base learners).

    Args:
        X: raw feature matrix for ``feature_set``.
        y: class names.
        reducer: "pca" | "lda" | "none" (default from config).
        balance: family-balanced fusion on/off (default from config).
        base: which base learners to include.
    """
    reducer = reduce.resolve_method(reducer)
    classes, y_idx = np.unique(np.asarray(y), return_inverse=True)
    counts = np.bincount(y_idx)
    if counts.min() < M.stack_cv:
        raise ModiError(f"Stacking needs at least {M.stack_cv} training images per class; "
                        f"the smallest class has {counts.min()}.")
    t0 = time.perf_counter()
    pre = make_preprocessor(feature_set, reducer, len(classes), balance)
    Z = pre.fit_transform(X, y_idx)
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    stack = make_stack(base).fit(Z, y_idx)
    t_stack = time.perf_counter() - t1
    red = pre.named_steps["reduce"]
    info = {
        "n_train": int(len(y_idx)), "n_classes": int(len(classes)), "n_features_in": int(X.shape[1]),
        "n_features_reduced": int(Z.shape[1]),
        "pca_variance_kept": float(red.explained_) if isinstance(red, reduce.VariancePCA) else None,
        "family_balanced": bool(pre.named_steps["balance"].enabled),
        "fit_seconds": {"preprocessor": round(t_pre, 2),
                        **{n: round(stack.named_estimators_[n].fit_seconds_, 2) for n in base},
                        "stack_total": round(t_pre + t_stack, 2)},
    }
    return TrainedModels(pre, stack, classes, feature_set, reducer, info)


def model_size_bytes(obj: Any) -> int:
    """Serialised size of a model (joblib, compressed like the saved file)."""
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.getbuffer().nbytes


def save(bundle: TrainedModels, path: Path) -> Path:
    """Save a model bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load(path: Path) -> TrainedModels:
    """Load a model bundle, with a clear message if it is missing."""
    if not Path(path).exists():
        raise ModiError(f"Model file not found: {path}. Run: python scripts/run_training.py")
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - version mismatch, truncated file...
        raise ModiError(f"Could not load {path} ({type(exc).__name__}: {exc}). "
                        "Re-train with: python scripts/run_training.py") from exc
