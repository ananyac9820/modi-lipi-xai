"""Stage 6 - Evaluation: metrics, confusion analysis, ablation and figures.

First-class outputs (none of the surveyed papers publishes them):
  * per-class precision / recall / F1 for every character
  * the full confusion matrix
  * a "most confused character pairs" table, with example images
  * per-category results (digits vs vowels vs consonants)
  * an ablation table: each descriptor alone vs fused, for all four models
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src import config, learning_curve
from src.models import ALL_MODELS, DISPLAY
from src.utils import get_logger, save_json, setup_matplotlib

log = get_logger("evaluate")
FIG, MET = config.PATHS.figures, config.PATHS.metrics
BG, INK, SAFFRON, GREEN = "#fbf7ef", "#1f3a5f", "#c8812a", "#5b8c5a"
MODEL_COLOURS = {"rf": "#5b8c5a", "svm": "#8c4a5b", "xgb": "#c8812a", "stack": "#1f3a5f"}
FEATURE_SET_LABELS = {"hog": "HOG only", "zernike": "Zernike only", "lbp": "LBP only",
                      "hog+lbp": "HOG + LBP (no Zernike)", "hog+zernike": "HOG + Zernike (no LBP)",
                      "zernike+lbp": "Zernike + LBP (no HOG)", "fused_plain": "Fused (plain concat)", "fused": "Fused (family-balanced)"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def summarize(y_true: np.ndarray, y_pred: np.ndarray, categories: np.ndarray,
              short_classes: dict[str, int] | None = None) -> dict[str, Any]:
    """Headline metrics (shared scorer, so numbers match the learning curve)."""
    return learning_curve.score_predictions(y_true, y_pred, categories, short_classes)


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray,
                    class_category: dict[str, str]) -> pd.DataFrame:
    """Precision, recall, F1 and support for every class, worst F1 first."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    df = pd.DataFrame({"label": classes, "category": [class_category.get(c, "") for c in classes],
                       "precision": p, "recall": r, "f1": f, "support": s})
    return df.sort_values("f1").reset_index(drop=True)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Confusion matrix with rows = true class, columns = predicted class."""
    return confusion_matrix(y_true, y_pred, labels=classes)


def most_confused_pairs(cm: np.ndarray, classes: np.ndarray, top: int = 30) -> pd.DataFrame:
    """Character pairs that get mixed up, ranked by total confusions.

    For an unordered pair {A, B}: ``a_as_b`` = images of A predicted as B and
    ``b_as_a`` the reverse. ``pair_error_rate`` = total confusions divided by
    the pair's total test images, i.e. how often the model mixes these two.
    """
    support = cm.sum(axis=1)
    rows = []
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            ab, ba = int(cm[i, j]), int(cm[j, i])
            if ab + ba == 0:
                continue
            rows.append({"class_a": classes[i], "class_b": classes[j], "a_as_b": ab, "b_as_a": ba,
                         "total_confusions": ab + ba,
                         "a_as_b_rate": ab / max(support[i], 1), "b_as_a_rate": ba / max(support[j], 1),
                         "pair_error_rate": (ab + ba) / max(support[i] + support[j], 1)})
    df = pd.DataFrame(rows, columns=["class_a", "class_b", "a_as_b", "b_as_a", "total_confusions",
                                     "a_as_b_rate", "b_as_a_rate", "pair_error_rate"])
    return df.sort_values(["total_confusions", "class_a"], ascending=[False, True]).head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax: Any) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_confusion_matrix(cm: np.ndarray, classes: np.ndarray, path: Path, title: str = "") -> Path:
    """Row-normalised confusion matrix heatmap (diagonal = per-class recall)."""
    plt = setup_matplotlib()
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    size = max(9, 0.24 * n)
    fig, ax = plt.subplots(figsize=(size + 1, size), dpi=130)
    fig.patch.set_facecolor(BG)
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title(title or "Confusion matrix (each row sums to 1)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="fraction of the true class")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def plot_confused_pairs(pairs: pd.DataFrame, path: Path, examples: dict[str, np.ndarray] | None = None,
                        top: int = 12) -> Path:
    """Bar chart of the most confused pairs, with example glyphs of each pair."""
    plt = setup_matplotlib()
    d = pairs.head(top)
    fig = plt.figure(figsize=(12, 0.62 * len(d) + 1.6), dpi=130)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(len(d), 4, width_ratios=[0.5, 0.5, 0.25, 5], wspace=0.08, hspace=0.25)
    ax = fig.add_subplot(gs[:, 3]); _style(ax)
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["a_as_b"], color=INK, label="A mistaken for B")
    ax.barh(y, d["b_as_a"], left=d["a_as_b"], color=SAFFRON, label="B mistaken for A")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["total_confusions"], yi, f"  {r['total_confusions']}  ({100 * r['pair_error_rate']:.1f}% of the pair)",
                va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([f"{a}  ↔  {b}" for a, b in zip(d["class_a"], d["class_b"])], fontsize=9)
    ax.set_xlabel("test images confused"); ax.set_xlim(0, d["total_confusions"].max() * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Most confused character pairs (A ↔ B)", fontsize=11)
    if examples:
        for row, (_, r) in enumerate(d.iterrows()):
            for col, key in enumerate(("class_a", "class_b")):
                a = fig.add_subplot(gs[row, col]); a.axis("off")
                if r[key] in examples:
                    a.imshow(examples[r[key]], cmap="gray_r", interpolation="nearest")
                if row == 0:
                    a.set_title("A" if col == 0 else "B", fontsize=9)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(table: dict[str, dict[str, Any]], path: Path, title: str = "") -> Path:
    """Grouped bars: accuracy and macro P / R / F1 for the four models."""
    plt = setup_matplotlib()
    metrics = [("accuracy", "accuracy"), ("macro_precision", "macro precision"),
               ("macro_recall", "macro recall"), ("macro_f1", "macro F1")]
    models = [m for m in ALL_MODELS if m in table]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=130); fig.patch.set_facecolor(BG); _style(ax)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [100 * table[m][k] for k, _ in metrics]
        bars = ax.bar(np.arange(len(metrics)) + i * w, vals, w, color=MODEL_COLOURS[m], label=DISPLAY[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7)
    lo = min(100 * table[m][k] for m in models for k, _ in metrics)
    ax.set_ylim(max(0, lo - 8), 100.5)
    ax.set_xticks(np.arange(len(metrics)) + w * (len(models) - 1) / 2); ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylabel("% on the fixed test set")
    ax.set_title(title or "Model comparison", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def plot_ablation(ablation: dict[str, dict[str, dict[str, Any]]], path: Path, metric: str = "accuracy") -> Path:
    """Grouped bars: each feature set (x) for each model (colour)."""
    plt = setup_matplotlib()
    sets = [s for s in FEATURE_SET_LABELS if s in ablation]
    models = [m for m in ALL_MODELS if all(m in ablation[s] for s in sets)]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130); fig.patch.set_facecolor(BG); _style(ax)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [100 * ablation[s][m][metric] for s in sets]
        bars = ax.bar(np.arange(len(sets)) + i * w, vals, w, color=MODEL_COLOURS[m], label=DISPLAY[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=7, rotation=90)
    lo = min(100 * ablation[s][m][metric] for s in sets for m in models)
    ax.set_ylim(max(0, lo - 10), 103)
    ax.set_xticks(np.arange(len(sets)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([FEATURE_SET_LABELS[s] for s in sets])
    ax.set_ylabel(f"test {metric.replace('_', ' ')} (%)")
    ax.set_title("Ablation: each descriptor family alone vs fused", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def plot_per_class_f1(table: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Per-class F1, sorted, coloured by category."""
    from src.data_loader import CATEGORY_COLOURS

    plt = setup_matplotlib()
    d = table.sort_values("f1")
    fig, ax = plt.subplots(figsize=(max(11, 0.24 * len(d)), 4.8), dpi=130); fig.patch.set_facecolor(BG); _style(ax)
    ax.bar(range(len(d)), 100 * d["f1"], color=[CATEGORY_COLOURS.get(c, "#888") for c in d["category"]])
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d["label"], rotation=90, fontsize=8)
    ax.set_ylim(max(0, 100 * d["f1"].min() - 10), 100.5)
    ax.set_ylabel("F1 (%)"); ax.set_title(title or "Per-class F1 (hardest characters on the left)", fontsize=11)
    cats = [c for c in CATEGORY_COLOURS if c in set(d["category"])]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOURS[c]) for c in cats], cats, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def plot_classical_curves(curve: dict[str, Any], path: Path) -> Path:
    """Accuracy vs training size for all four classical models."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130); fig.patch.set_facecolor(BG); _style(ax)
    xs = [p["train_per_class"] for p in curve["points"]]
    for m in ALL_MODELS:
        if all(m in p.get("models", {}) for p in curve["points"]):
            ax.plot(xs, [100 * p["models"][m]["accuracy"] for p in curve["points"]], marker="o", lw=2,
                    color=MODEL_COLOURS[m], label=DISPLAY[m])
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
    ax.set_xlabel("training images per class (log scale)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Classical models: accuracy vs training-set size", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# One call that writes every evaluation artefact for a set of predictions
# ---------------------------------------------------------------------------
def write_main_report(y_true: np.ndarray, preds: dict[str, np.ndarray], categories: np.ndarray,
                      classes: np.ndarray, class_category: dict[str, str],
                      examples: dict[str, np.ndarray] | None = None, tag: str = "",
                      extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute and save metrics + figures for the main experiment.

    Args:
        preds: {model name: predicted labels}; must contain "stack".
        examples: {class label: one example image} for the confused-pairs figure.
        tag: text appended to figure titles (e.g. the training size).
    """
    table = {m: summarize(y_true, p, categories) for m, p in preds.items()}
    best = max(table, key=lambda m: table[m]["accuracy"])
    save_json({"models": table, "best_model": best, "display_names": DISPLAY, **(extra or {})},
              MET / "model_comparison.json")
    plot_model_comparison(table, FIG / "model_comparison.png", f"Model comparison {tag}".strip())

    pc = per_class_table(y_true, preds["stack"], classes, class_category)
    pc.to_csv(MET / "per_class_metrics.csv", index=False)
    save_json({"model": "stack", "classes": pc.to_dict(orient="records")}, MET / "per_class_metrics.json")
    plot_per_class_f1(pc, FIG / "per_class_f1.png", f"Stacking ensemble: per-class F1 {tag}".strip())

    cm = confusion(y_true, preds["stack"], classes)
    save_json({"model": "stack", "classes": classes.tolist(), "matrix": cm.tolist(),
               "note": "rows = true class, columns = predicted class"}, MET / "confusion_matrix.json")
    plot_confusion_matrix(cm, classes, FIG / "confusion_matrix.png",
                          f"Stacking ensemble confusion matrix {tag} (rows sum to 1)".strip())

    pairs = most_confused_pairs(cm, classes)
    pairs.to_csv(MET / "most_confused_pairs.csv", index=False)
    save_json({"model": "stack", "total_errors": int(cm.sum() - np.trace(cm)), "n_test": int(cm.sum()),
               "pairs": pairs.to_dict(orient="records")}, MET / "most_confused_pairs.json")
    if len(pairs):
        plot_confused_pairs(pairs, FIG / "most_confused_pairs.png", examples)

    # Confusions inside vs across categories: are digits ever mistaken for letters?
    cat_of = np.array([class_category.get(c, "") for c in classes])
    off = cm.copy(); np.fill_diagonal(off, 0)
    same = int(off[cat_of[:, None] == cat_of[None, :]].sum())
    save_json({"errors_within_same_category": same, "errors_across_categories": int(off.sum() - same),
               "per_category": {m: table[m]["per_category_accuracy"] for m in table}},
              MET / "per_category_results.json")
    return {"models": table, "best_model": best, "per_class": pc, "confusion": cm, "pairs": pairs}
