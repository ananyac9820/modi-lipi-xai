"""Stage 5 runner: train and evaluate the classical models.

Experiments (each one is saved as soon as it finishes, so the script resumes):
  1. MAIN       fused features + PCA, 500 training images per class -> RF, SVM, XGBoost, stacking
  2. REDUCTION  the same with LDA instead of PCA
  3. ABLATION   HOG only / Zernike only / LBP only / fused (plain) / fused (balanced), all four models
  4. CV         stratified k-fold cross-validation of the whole pipeline on the main training set
  5. CURVE      learning curve at 72 / 90 / 250 / 500 / 1000 / 2000 training images per class

Every experiment uses the same fixed test set (200 images per class).

Usage:
    python scripts/run_training.py                      # everything (resumes)
    python scripts/run_training.py --only main          # one experiment
    python scripts/run_training.py --sizes 72 90 250    # selected curve points
    python scripts/run_training.py --force              # redo finished experiments
    python scripts/run_training.py --cv-folds 0         # skip cross-validation

Outputs: models/main_stack.joblib, models/main_predictions.npz and JSON files in results/metrics/.
Figures are produced by scripts/run_evaluation.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src import config, data_loader, learning_curve, models  # noqa: E402
from src.features import fusion  # noqa: E402
from src.utils import get_logger, load_json, run_cli, save_json, set_seed, stage  # noqa: E402

log = get_logger("run_training")
MET = config.PATHS.metrics
MAIN_MODEL = config.PATHS.models / "main_stack.joblib"
MAIN_PRED = config.PATHS.models / "main_predictions.npz"
TRAIN_INFO = MET / "training_info.json"
ABLATION = MET / "ablation.json"
REDUCTION = MET / "reduction_comparison.json"
CV_JSON = MET / "cross_validation.json"
EXPERIMENTS = ("main", "reduction", "ablation", "cv", "curve")


class Data:
    """Processed index plus lazily loaded feature matrices."""

    def __init__(self) -> None:
        self.index = data_loader.load_processed_index()
        self.index["_pos"] = np.arange(len(self.index))
        self._cache: dict[str, np.ndarray] = {}

    def X(self, feature_set: str) -> np.ndarray:
        if feature_set not in self._cache:
            self._cache[feature_set] = fusion.load_feature_set(self.index, feature_set)
        return self._cache[feature_set]

    def split(self, n: int) -> tuple[Any, Any, dict[str, Any]]:
        return data_loader.stratified_split(self.index, n)


def evaluate(bundle: models.TrainedModels, X_test: np.ndarray, y_test: np.ndarray, cats: np.ndarray,
             short: dict[str, int] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Score every model in a bundle on the test set."""
    t = time.perf_counter()
    Z = bundle.transform(X_test)
    t_pre = time.perf_counter() - t
    scores: dict[str, dict[str, Any]] = {}
    preds: dict[str, np.ndarray] = {}
    for name in (*bundle.stack.named_estimators_, "stack"):
        t = time.perf_counter()
        idx = bundle.predict_idx(name, Z, reduced=True)
        secs = time.perf_counter() - t
        preds[name] = idx
        s = learning_curve.score_predictions(y_test, bundle.classes[idx], cats, short)
        s["inference_ms_per_image"] = round(1000 * (secs + t_pre) / len(y_test), 4)
        s["train_time_s"] = bundle.info["fit_seconds"].get("stack_total" if name == "stack" else name)
        scores[name] = s
    return scores, preds


def fit_and_score(data: Data, n: int, feature_set: str, reducer: str | None = None,
                  balance: bool | None = None) -> tuple[models.TrainedModels, dict, dict, Any, dict]:
    """Train on the n-per-class split and evaluate on the fixed test set."""
    train, test, rep = data.split(n)
    X = data.X(feature_set)
    bundle = models.train(X[train["_pos"].to_numpy()], train["label"].to_numpy(), feature_set, reducer, balance)
    scores, preds = evaluate(bundle, X[test["_pos"].to_numpy()], test["label"].to_numpy(),
                             test["category"].to_numpy(), rep["short_classes"])
    log.info("  %s | %s | n=%d -> %s  (dims %d, %.0fs)", feature_set, bundle.reducer, n,
             ", ".join(f"{m} {100 * scores[m]['accuracy']:.2f}%" for m in scores),
             bundle.info["n_features_reduced"], bundle.info["fit_seconds"]["stack_total"])
    return bundle, scores, preds, test, rep


# ---------------------------------------------------------------------------
def run_main(data: Data, force: bool) -> None:
    n = config.SAMPLING.main_train_per_class
    if MAIN_MODEL.exists() and MAIN_PRED.exists() and TRAIN_INFO.exists() and not force:
        log.info("Main experiment already done - skipping (use --force to retrain)")
        return
    bundle, scores, preds, test, rep = fit_and_score(data, n, "fused")
    with stage("Saving main model and predictions", log):
        models.save(bundle, MAIN_MODEL)
        X_test = data.X("fused")[test["_pos"].to_numpy()]
        np.savez_compressed(MAIN_PRED, y_true=test["label"].to_numpy().astype(str),
                            categories=test["category"].to_numpy().astype(str),
                            rel_path=test["rel_path"].to_numpy().astype(str),
                            processed_path=test["processed_path"].to_numpy().astype(str),
                            classes=bundle.classes.astype(str), proba=bundle.predict_proba(X_test).astype(np.float32),
                            **{f"pred_{m}": bundle.classes[p].astype(str) for m, p in preds.items()})
        sizes = {m: models.model_size_bytes(bundle.stack.named_estimators_[m]) for m in bundle.stack.named_estimators_}
        sizes["stack"] = MAIN_MODEL.stat().st_size
        for m in scores:
            scores[m]["model_size_bytes"] = sizes[m]
        save_json({"train_per_class": n, "feature_set": "fused", "reducer": bundle.reducer, "info": bundle.info,
                   "split": {k: rep[k] for k in ("n_train", "n_test", "complete", "short_classes")},
                   "models": scores, "model_size_bytes": sizes, "config": config.as_dict()}, TRAIN_INFO)
    _curve_point(n, scores, bundle, rep)


def run_reduction(data: Data, force: bool) -> None:
    if REDUCTION.exists() and not force:
        log.info("Reduction comparison already done - skipping")
        return
    n = config.SAMPLING.main_train_per_class
    out: dict[str, Any] = {"train_per_class": n, "methods": {}}
    for method in ("pca", "lda"):
        if method == config.REDUCE.method and TRAIN_INFO.exists():
            info = load_json(TRAIN_INFO)
            out["methods"][method] = {"models": info["models"], "n_components": info["info"]["n_features_reduced"]}
            continue
        bundle, scores, _, _, _ = fit_and_score(data, n, "fused", reducer=method)
        out["methods"][method] = {"models": scores, "n_components": bundle.info["n_features_reduced"]}
    save_json(out, REDUCTION)


def run_ablation(data: Data, force: bool) -> None:
    n = config.SAMPLING.main_train_per_class
    done = (load_json(ABLATION) if ABLATION.exists() and not force else {}).get("feature_sets", {})
    variants = [("hog", "hog", None), ("zernike", "zernike", None), ("lbp", "lbp", None),
                ("fused_plain", "fused", False), ("fused", "fused", True)]
    for key, fset, balance in variants:
        if key in done:
            log.info("Ablation '%s' already done - skipping", key)
            continue
        if key == "fused" and config.FUSION.balance_families and TRAIN_INFO.exists() and not force:
            info = load_json(TRAIN_INFO)
            done[key] = {"models": info["models"], "n_features": info["info"]["n_features_in"],
                         "n_components": info["info"]["n_features_reduced"]}
        else:
            bundle, scores, _, _, _ = fit_and_score(data, n, fset, balance=balance)
            done[key] = {"models": scores, "n_features": bundle.info["n_features_in"],
                         "n_components": bundle.info["n_features_reduced"]}
        save_json({"train_per_class": n, "reducer": config.REDUCE.method, "feature_sets": done}, ABLATION)


def run_cv(data: Data, folds: int, force: bool) -> None:
    if folds < 2:
        log.info("Cross-validation skipped (--cv-folds %d)", folds)
        return
    if CV_JSON.exists() and not force:
        log.info("Cross-validation already done - skipping")
        return
    n = config.SAMPLING.main_train_per_class
    train, _, _ = data.split(n)
    X = data.X("fused")[train["_pos"].to_numpy()]
    y, cats = train["label"].to_numpy(), train["category"].to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    per_fold: list[dict[str, float]] = []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        bundle = models.train(X[tr], y[tr], "fused")
        scores, _ = evaluate(bundle, X[va], y[va], cats[va])
        per_fold.append({m: scores[m]["accuracy"] for m in scores})
        log.info("  fold %d/%d: %s", k, folds, ", ".join(f"{m} {100 * a:.2f}%" for m, a in per_fold[-1].items()))
    summary = {m: {"mean": float(np.mean([f[m] for f in per_fold])), "std": float(np.std([f[m] for f in per_fold]))}
               for m in per_fold[0]}
    save_json({"folds": folds, "train_per_class": n, "note": "Whole pipeline (scaling, PCA, models) refitted inside "
               "each fold on the main training set only; the test set is never used.",
               "per_fold_accuracy": per_fold, "summary": summary}, CV_JSON)


def _curve_point(n: int, scores: dict[str, dict[str, Any]], bundle: models.TrainedModels, rep: dict[str, Any]) -> None:
    """Store one learning-curve point; the headline series is the stacking ensemble."""
    header = {"model": "Classical stacking ensemble (HOG + Zernike + LBP -> RF + SVM + XGBoost)",
              "label": "Classical: fused features + stacking", "family": "classical",
              "feature_set": "fused", "reducer": bundle.reducer}
    point = {"train_per_class": n, **scores["stack"], "models": scores,
             "n_train": bundle.info["n_train"], "n_components": bundle.info["n_features_reduced"],
             "fit_seconds": bundle.info["fit_seconds"],
             "model_size_bytes": scores["stack"].get("model_size_bytes") or models.model_size_bytes(bundle)}
    learning_curve.upsert_point(learning_curve.CLASSICAL_JSON, header, point)


def run_curve(data: Data, sizes: list[int], force: bool) -> None:
    done = {p["train_per_class"] for p in (learning_curve.load_curve(learning_curve.CLASSICAL_JSON) or {}).get("points", [])}
    for n in sorted(sizes):
        if n in done and not force:
            log.info("Curve point %d/class already done - skipping", n)
            continue
        with stage(f"Learning-curve point: {n} training images per class", log):
            bundle, scores, _, _, rep = fit_and_score(data, n, "fused")
            _curve_point(n, scores, bundle, rep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=EXPERIMENTS, nargs="+", help="run only these experiments")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(config.SAMPLING.curve_train_sizes))
    parser.add_argument("--cv-folds", type=int, default=config.SPLIT.cv_folds)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed()
    config.PATHS.ensure()
    todo = args.only or EXPERIMENTS
    t0 = time.perf_counter()
    with stage("Loading processed index and cached features", log):
        data = Data()
        log.info("%d images, %d classes, fused dimension %d", len(data.index), data.index["label"].nunique(),
                 data.X("fused").shape[1])
    if "main" in todo:
        with stage(f"MAIN experiment ({config.SAMPLING.main_train_per_class}/class, fused, {config.REDUCE.method})", log):
            run_main(data, args.force)
    if "reduction" in todo:
        with stage("REDUCTION comparison (PCA vs LDA)", log):
            run_reduction(data, args.force)
    if "ablation" in todo:
        with stage("ABLATION (single descriptors vs fused)", log):
            run_ablation(data, args.force)
    if "cv" in todo:
        with stage(f"CROSS-VALIDATION ({args.cv_folds}-fold)", log):
            run_cv(data, args.cv_folds, args.force)
    if "curve" in todo:
        with stage("LEARNING CURVE", log):
            run_curve(data, args.sizes, args.force)
    log.info("Total time %.1f min. Next: python scripts/run_evaluation.py", (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    run_cli(main)
