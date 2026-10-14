"""Learning-curve bookkeeping shared by the classical pipeline and the CNN
reference baseline, plus the project's headline figure.

Both sides MUST score predictions through :func:`score_predictions` so their
numbers are computed identically, and both write the same JSON schema:

    {
      "model": str, "family": "classical" | "cnn_baseline",
      "points": [ {"train_per_class": int, "accuracy": float, "macro_f1": float,
                   "accuracy_complete_classes": float | null, "complete": bool,
                   "short_classes": {...}, "per_category_accuracy": {...},
                   "train_time_s": float, "model_size_bytes": int,
                   "inference_ms_per_image": float, ...}, ... ]
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import config
from src.utils import ModiError, get_logger, load_json, save_json, setup_matplotlib

log = get_logger("learning_curve")

CLASSICAL_JSON = config.PATHS.metrics / "learning_curve_classical.json"
CNN_JSON = config.PATHS.metrics / "learning_curve_cnn.json"
HEADLINE_FIG = config.PATHS.figures / "learning_curve_classical_vs_cnn.png"

PAPERS_EFFECTIVE_TRAIN = 72
PAPERS_NOTE = ("≈ surveyed papers' training size\n(sample size only; different dataset,\n"
               "classes and handwriting)")


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, categories: np.ndarray,
                      short_classes: dict[str, int] | None = None) -> dict[str, Any]:
    """Standard metrics for one curve point.

    Args:
        y_true, y_pred: string labels.
        categories: category of each test sample (digit/vowel/consonant).
        short_classes: classes that could not fill this curve point; if any,
            accuracy is also reported on the remaining (complete) classes.
    """
    y_true, y_pred, categories = map(np.asarray, (y_true, y_pred, categories))
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test": int(len(y_true)),
        "per_category_accuracy": {
            str(c): float(accuracy_score(y_true[categories == c], y_pred[categories == c]))
            for c in sorted(set(categories))
        },
    }
    short = set(short_classes or {})
    out["complete"] = not short
    out["short_classes"] = dict(short_classes or {})
    if short:
        keep = ~np.isin(y_true, list(short))
        out["accuracy_complete_classes"] = float(accuracy_score(y_true[keep], y_pred[keep])) if keep.any() else None
    else:
        out["accuracy_complete_classes"] = None
    return out


def load_curve(path: Path) -> dict[str, Any] | None:
    """Load a curve JSON, or None if it has not been produced yet."""
    return load_json(path) if path.exists() else None


def upsert_point(path: Path, header: dict[str, Any], point: dict[str, Any]) -> None:
    """Insert/replace one curve point (keyed by train_per_class) and save, so
    long runs can be resumed point by point."""
    data = load_curve(path) or {**header, "points": []}
    data.update(header)
    data["points"] = sorted([p for p in data["points"] if p["train_per_class"] != point["train_per_class"]]
                            + [point], key=lambda p: p["train_per_class"])
    save_json(data, path)


def find_crossover(xs: list[int], a: list[float], b: list[float]) -> tuple[int, int] | None:
    """First interval (x_i, x_{i+1}) where curve b overtakes curve a (or vice
    versa). Returns None if the sign of (b - a) never changes."""
    diff = np.asarray(b) - np.asarray(a)
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            return xs[i], xs[i]
        if np.sign(diff[i]) != np.sign(diff[i + 1]) and diff[i + 1] != 0:
            return xs[i], xs[i + 1]
    return None


def plot_headline(classical: dict[str, Any] | None = None, cnn: dict[str, Any] | None = None,
                  path: Path = HEADLINE_FIG) -> Path:
    """Headline figure: accuracy vs training images per class for the
    classical ensemble and the reference CNN (left), training time (right).

    Hollow markers = incomplete points (some classes had fewer images).
    The crossover region, if any, is shaded.
    """
    classical = classical if classical is not None else load_curve(CLASSICAL_JSON)
    cnn = cnn if cnn is not None else load_curve(CNN_JSON)
    if not any(d and d.get("points") for d in (classical, cnn)):
        raise ModiError("No learning-curve results yet. Run scripts/run_training.py (classical) and/or "
                        "scripts/run_cnn_baseline.py first.")
    plt = setup_matplotlib()
    bg, ink = "#fbf7ef", "#1f3a5f"
    styles = {"classical": {"color": ink, "marker": "o"}, "cnn_baseline": {"color": "#c8812a", "marker": "s"}}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(bg)
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for data in (classical, cnn):
        if not data or not data.get("points"):
            continue
        fam = data.get("family", "classical")
        st = styles.get(fam, {"color": "#555", "marker": "^"})
        pts = data["points"]
        xs = [p["train_per_class"] for p in pts]
        ys = [100 * p["accuracy"] for p in pts]
        curves[fam] = (xs, ys)
        label = data.get("label", data.get("model", fam))
        ax.plot(xs, ys, color=st["color"], lw=2, label=label)
        for p, x, y in zip(pts, xs, ys):
            ax.scatter(x, y, s=45, marker=st["marker"], zorder=3, color=st["color"],
                       facecolors=st["color"] if p.get("complete", True) else bg, linewidths=1.5)
        ax2.plot(xs, [p.get("train_time_s", np.nan) for p in pts], color=st["color"], marker=st["marker"], lw=2,
                 label=label)
        # Classical side: also show the best SINGLE model, so the comparison is
        # never made against a weaker-than-necessary classical result.
        if fam == "classical" and all("models" in p for p in pts):
            singles = [m for m in ("rf", "svm", "xgb") if all(m in p["models"] for p in pts)]
            if singles:
                best = max(singles, key=lambda m: np.mean([p["models"][m]["accuracy"] for p in pts]))
                names = {"rf": "Random Forest", "svm": "SVM (RBF)", "xgb": "XGBoost"}
                ax.plot(xs, [100 * p["models"][best]["accuracy"] for p in pts], color=st["color"], ls="--", lw=1.5,
                        marker=".", label=f"Classical: best single model ({names[best]})")
                ax2.plot(xs, [p["models"][best].get("train_time_s", np.nan) for p in pts], color=st["color"],
                         ls="--", lw=1.2, label=f"{names[best]} alone")

    if "classical" in curves and "cnn_baseline" in curves:
        xc, yc = curves["classical"]
        xn, yn = curves["cnn_baseline"]
        common = sorted(set(xc) & set(xn))
        if len(common) >= 2:
            a = [yc[xc.index(x)] for x in common]
            b = [yn[xn.index(x)] for x in common]
            cross = find_crossover(common, a, b)
            if cross:
                ax.axvspan(cross[0], cross[1], color="#c8812a", alpha=0.12, zorder=0)
                ax.text(np.sqrt(cross[0] * cross[1]), 0.98, "crossover\nregion", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, color="#7a4d12")
            else:
                leader = "CNN" if b[-1] > a[-1] else "classical"
                ax.text(0.98, 0.04, f"no crossover in measured range ({leader} ahead throughout)",
                        transform=ax.transAxes, ha="right", fontsize=8, color="#555")

    ax.axvline(PAPERS_EFFECTIVE_TRAIN, color="#888", ls=":", lw=1)
    ax.text(PAPERS_EFFECTIVE_TRAIN * 1.04, 0.97, PAPERS_NOTE, transform=ax.get_xaxis_transform(), fontsize=7,
            va="top", color="#555")
    for a_ in (ax, ax2):
        a_.set_facecolor(bg)
        a_.set_xscale("log")
        sizes = config.SAMPLING.curve_train_sizes
        a_.set_xticks(sizes)
        a_.set_xticklabels([str(s) for s in sizes])
        a_.minorticks_off()
        a_.set_xlabel("training images per class (log scale)")
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
    ax.set_ylabel("test accuracy (%) - same fixed test set")
    ax.set_title("Where does handcrafted feature engineering stop being competitive?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.set_yscale("log")
    ax2.set_ylabel("training wall-clock time (s, log)")
    ax2.set_title("Compute cost (CPU)", fontsize=11)
    ax2.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Hollow markers: incomplete point (some classes had fewer training images). "
             "CNN is an untuned reference baseline trained on identical splits.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    log.info("Saved headline figure %s", path)
    return path
