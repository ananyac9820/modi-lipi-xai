"""Stage 6-7 runner: turn saved predictions into metrics, tables and figures,
and compute SHAP explanations for the main model.

Run scripts/run_training.py first. Nothing is retrained here.

Usage:
    python scripts/run_evaluation.py              # everything
    python scripts/run_evaluation.py --skip-shap  # metrics and figures only
    python scripts/run_evaluation.py --force-shap # recompute cached SHAP values

Outputs (results/metrics/*.json|csv, results/figures/*.png, results/shap/*):
    model_comparison, per_class_metrics, confusion_matrix, most_confused_pairs,
    per_category_results, ablation figure, PCA variance + 2-D scatter,
    classical learning curves, classical-vs-CNN headline figure, SHAP importance.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import config, data_loader, evaluate, learning_curve, models, preprocess, reduce  # noqa: E402
from src.features import fusion  # noqa: E402
from src.utils import ModiError, get_logger, load_json, run_cli, set_seed, stage  # noqa: E402

log = get_logger("run_evaluation")
MET, FIG = config.PATHS.metrics, config.PATHS.figures
MAIN_MODEL = config.PATHS.models / "main_stack.joblib"
MAIN_PRED = config.PATHS.models / "main_predictions.npz"


def load_predictions() -> dict[str, np.ndarray]:
    if not MAIN_PRED.exists():
        raise ModiError(f"{MAIN_PRED.name} not found. Run: python scripts/run_training.py --only main")
    with np.load(MAIN_PRED, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--force-shap", action="store_true")
    args = parser.parse_args()

    set_seed()
    config.PATHS.ensure()
    t0 = time.perf_counter()
    n_main = config.SAMPLING.main_train_per_class
    P = load_predictions()
    y_true, cats, classes = P["y_true"], P["categories"], P["classes"]
    preds = {k[5:]: v for k, v in P.items() if k.startswith("pred_")}
    class_category = dict(zip(y_true, cats))

    with stage("Main-experiment metrics, confusion analysis and figures", log):
        examples = {}
        for lab in classes:
            first = np.flatnonzero(y_true == lab)[0]
            examples[str(lab)] = preprocess.read_image(config.PROJECT_ROOT / str(P["processed_path"][first]))
        info = load_json(MET / "training_info.json")
        rep = evaluate.write_main_report(y_true, preds, cats, classes, class_category, examples,
                                         tag=f"({n_main} training images per class)",
                                         extra={"train_per_class": n_main, "n_test": int(len(y_true)),
                                                "fit_seconds": info["info"]["fit_seconds"],
                                                "model_size_bytes": info["model_size_bytes"],
                                                "inference_ms_per_image": {m: info["models"][m]["inference_ms_per_image"]
                                                                           for m in info["models"]}})
        for m, s in rep["models"].items():
            log.info("  %-18s accuracy %.2f%% | macro-F1 %.4f | per category: %s", models.DISPLAY[m],
                     100 * s["accuracy"], s["macro_f1"],
                     ", ".join(f"{c} {100 * a:.1f}%" for c, a in s["per_category_accuracy"].items()))
        worst = rep["per_class"].head(5)
        log.info("  Hardest classes (F1): %s", ", ".join(f"{r.label} {100 * r.f1:.1f}%" for r in worst.itertuples()))
        top = rep["pairs"].head(5)
        log.info("  Most confused pairs: %s", ", ".join(f"{r.class_a}↔{r.class_b} ({r.total_confusions})"
                                                         for r in top.itertuples()))

    if (MET / "ablation.json").exists():
        with stage("Ablation figure", log):
            abl = load_json(MET / "ablation.json")["feature_sets"]
            evaluate.plot_ablation({k: v["models"] for k, v in abl.items()}, FIG / "ablation.png")
            for k, v in abl.items():
                log.info("  %-24s %s", evaluate.FEATURE_SET_LABELS.get(k, k),
                         ", ".join(f"{m} {100 * v['models'][m]['accuracy']:.2f}%" for m in v["models"]))
    else:
        log.warning("No ablation results yet (run_training.py --only ablation).")

    with stage("Dimensionality-reduction figures", log):
        bundle = models.load(MAIN_MODEL)
        red = bundle.preprocessor.named_steps["reduce"]
        if isinstance(red, reduce.VariancePCA):
            reduce.plot_explained_variance(red, title=f"PCA on fused features ({bundle.info['n_features_in']} → "
                                                      f"{red.n_components_} dimensions)")
        index = data_loader.load_processed_index()
        train, _, _ = data_loader.stratified_split(index.assign(_pos=np.arange(len(index))), n_main, quiet=True)
        sample = train.groupby("label").head(100)
        X = fusion.load_feature_set(index, "fused")[sample["_pos"].to_numpy()]
        Xs = bundle.preprocessor[:2].transform(X)          # scaled + family-balanced, before reduction
        reduce.plot_scatter_2d(Xs, sample["label"].to_numpy(), sample["category"].to_numpy())

    curve = learning_curve.load_curve(learning_curve.CLASSICAL_JSON)
    if curve and curve.get("points"):
        with stage("Learning-curve figures", log):
            evaluate.plot_classical_curves(curve, FIG / "learning_curve_classical_models.png")
            learning_curve.plot_headline()
            for p in curve["points"]:
                log.info("  %5d/class: stack %.2f%% (train %.0fs)", p["train_per_class"], 100 * p["accuracy"],
                         p.get("train_time_s") or 0)

    if not args.skip_shap:
        try:
            from src import explain
        except ImportError:
            log.warning("src/explain.py not available yet - skipping SHAP.")
        else:
            with stage("SHAP explanations (global importance)", log):
                explain.run_global(force=args.force_shap)
            if len(rep["pairs"]):
                with stage("Contrastive SHAP on the most confused pairs", log):
                    top_pairs = [(r.class_a, r.class_b) for r in rep["pairs"].head(6).itertuples()]
                    explain.contrast_pairs(top_pairs, force=args.force_shap)

    log.info("Total time %.1fs. Figures in results/figures/, metrics in results/metrics/.", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    run_cli(main)
