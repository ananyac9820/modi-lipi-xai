"""Stage 4 - Dimensionality reduction: PCA and LDA behind one interface.

``make_reducer("pca" | "lda" | "none")`` returns an sklearn transformer, so
the two methods are interchangeable in every pipeline and can be compared by
changing ``config.REDUCE.method`` (or passing the name explicitly).

* PCA is unsupervised: it keeps the directions of largest variance. The
  number of components is chosen to reach a variance target, with a hard cap.
* LDA is supervised: it finds at most (n_classes - 1) directions that best
  separate the classes. Shrinkage keeps it stable when there are more
  features than images per class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src import config
from src.utils import ModiError, get_logger, setup_matplotlib

log = get_logger("reduce")
R = config.REDUCE
VARIANCE_FIG = config.PATHS.figures / "pca_explained_variance.png"
SCATTER_FIG = config.PATHS.figures / "reduction_scatter_2d.png"


class VariancePCA(BaseEstimator, TransformerMixin):
    """PCA that keeps the fewest components reaching a variance target.

    A randomized PCA with ``max_components`` is fitted once; the leading
    components that together explain ``variance`` of the TOTAL variance are
    kept. If the cap is reached first, all ``max_components`` are kept and
    the achieved variance is reported in ``explained_``.
    """

    def __init__(self, variance: float = R.pca_variance, max_components: int = R.pca_max_components,
                 random_state: int = config.SEED) -> None:
        self.variance = variance
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Any = None) -> "VariancePCA":
        k_max = int(min(self.max_components, X.shape[1], X.shape[0] - 1))
        solver = "full" if k_max >= min(X.shape) * 0.8 else "randomized"
        self.pca_ = PCA(n_components=k_max, svd_solver=solver, random_state=self.random_state).fit(X)
        total_var = float(np.var(X, axis=0, ddof=1).sum())
        ratio = self.pca_.explained_variance_ / max(total_var, 1e-12)
        self.cumulative_ = np.cumsum(ratio)
        reached = np.flatnonzero(self.cumulative_ >= self.variance)
        self.n_components_ = int(reached[0] + 1) if reached.size else k_max
        self.explained_ = float(self.cumulative_[self.n_components_ - 1])
        self.capped_ = not reached.size
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)[:, : self.n_components_]

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        full = np.zeros((Z.shape[0], self.pca_.n_components_), dtype=Z.dtype)
        full[:, : self.n_components_] = Z
        return self.pca_.inverse_transform(full)


SELECTION_JSON = config.PATHS.metrics / "reducer_selection.json"


def resolve_method(method: str | None = None) -> str:
    """Turn "auto" into the reducer chosen by training-set cross-validation."""
    method = (method or R.method).lower()
    if method != "auto":
        return method
    if not SELECTION_JSON.exists():
        raise ModiError("REDUCE.method is 'auto' but no reducer has been selected yet. "
                        "Run: python scripts/run_training.py --only select")
    import json
    return json.loads(SELECTION_JSON.read_text(encoding="utf-8"))["selected"]


def make_reducer(method: str | None = None, n_classes: int | None = None) -> Any:
    """Return the configured reducer.

    Args:
        method: "pca", "lda", "none" or "auto"; defaults to ``config.REDUCE.method``.
        n_classes: kept for API compatibility (LDA picks its own component count).
    """
    method = resolve_method(method)
    if method == "pca":
        return VariancePCA()
    if method == "lda":
        # None lets sklearn use min(n_classes - 1, n_features), which also works
        # for the 49-feature Zernike-only ablation.
        return LinearDiscriminantAnalysis(solver=R.lda_solver, shrinkage=R.lda_shrinkage,
                                          n_components=R.lda_components)
    if method == "none":
        return "passthrough"
    raise ModiError(f"Unknown reduction method '{method}'. Use 'pca', 'lda' or 'none'.")


def n_components_of(reducer: Any) -> int | None:
    """Output dimension of a fitted reducer (None for passthrough)."""
    if isinstance(reducer, VariancePCA):
        return reducer.n_components_
    if isinstance(reducer, LinearDiscriminantAnalysis):
        return int(reducer.scalings_.shape[1] if reducer.n_components is None else reducer.n_components)
    return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_explained_variance(pca: VariancePCA, path: Path = VARIANCE_FIG, title: str = "") -> Path:
    """Cumulative explained variance with the chosen component count marked."""
    plt = setup_matplotlib()
    bg = "#fbf7ef"
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=130)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    k = np.arange(1, len(pca.cumulative_) + 1)
    ax.plot(k, 100 * pca.cumulative_, color="#1f3a5f", lw=2)
    ax.axhline(100 * pca.variance, color="#c8812a", ls="--", lw=1)
    ax.axvline(pca.n_components_, color="#c8812a", ls="--", lw=1)
    ax.scatter([pca.n_components_], [100 * pca.explained_], color="#c8812a", zorder=3)
    note = f"{pca.n_components_} components → {100 * pca.explained_:.1f}% of variance"
    if pca.capped_:
        note += f"\n(cap of {pca.max_components} reached before the {100 * pca.variance:.0f}% target)"
    ax.annotate(note, (pca.n_components_, 100 * pca.explained_), xytext=(12, -30), textcoords="offset points", fontsize=9)
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.set_title(title or "PCA explained variance", fontsize=11)
    ax.set_ylim(0, 101)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return path


def plot_scatter_2d(X: np.ndarray, labels: np.ndarray, categories: np.ndarray, path: Path = SCATTER_FIG,
                    n_highlight: int = 8, max_points: int = 6000) -> Path:
    """2-D view of the classes under PCA and LDA.

    Top row: every class, coloured by category. Bottom row: a few classes
    highlighted, to show how well individual characters separate.
    ``X`` must already be scaled (and family-balanced).
    """
    from src.data_loader import CATEGORY_COLOURS

    plt = setup_matplotlib()
    rng = np.random.default_rng(config.SEED)
    if len(X) > max_points:
        keep = rng.choice(len(X), max_points, replace=False)
        X, labels, categories = X[keep], labels[keep], categories[keep]
    Zp = PCA(n_components=2, random_state=config.SEED).fit_transform(X)
    Zl = LinearDiscriminantAnalysis(solver=R.lda_solver, shrinkage=R.lda_shrinkage, n_components=2).fit_transform(X, labels)
    bg = "#fbf7ef"
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=130)
    fig.patch.set_facecolor(bg)
    chosen = sorted(rng.permutation(np.unique(labels))[:n_highlight])
    cmap = plt.get_cmap("tab10")
    for col, (Z, name) in enumerate(((Zp, "PCA (unsupervised)"), (Zl, "LDA (supervised)"))):
        ax = axes[0, col]
        for cat in [c for c in CATEGORY_COLOURS if c in set(categories)]:
            m = categories == cat
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.45, color=CATEGORY_COLOURS[cat], label=cat, linewidths=0)
        ax.set_title(f"{name}: all classes, by category", fontsize=10)
        ax.legend(frameon=False, fontsize=8, markerscale=3)
        ax = axes[1, col]
        ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.15, color="#bbb", linewidths=0)
        for i, lab in enumerate(chosen):
            m = labels == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.8, color=cmap(i % 10), label=str(lab), linewidths=0)
        ax.set_title(f"{name}: {len(chosen)} classes highlighted", fontsize=10)
        ax.legend(frameon=False, fontsize=8, markerscale=2, ncol=2)
    for ax in axes.ravel():
        ax.set_facecolor(bg)
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Classes in 2-D reduced space (fused, family-balanced features)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return path
