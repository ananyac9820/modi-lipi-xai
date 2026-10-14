"""Stage 7 - Explainability with SHAP.

WHAT IS EXPLAINED
The final stacking ensemble's class probabilities, as a function of the fused
handcrafted features (before scaling / PCA), so explanations are in terms a
person can read, not in terms of principal components.

HOW
The 2,261 fused features are grouped into 35 meaningful units:
  * HOG      - 16 groups: stroke-direction evidence in each cell of a 4x4 image grid
  * LBP      - 16 groups: local stroke patterns in each cell of a 4x4 image grid
  * Zernike  -  3 groups: coarse / medium / fine global shape (by moment order)
Kernel SHAP is then run over the groups. A group that is "switched off" is
replaced by the average training value of its features, so every SHAP value
answers: "how much did this group's actual content move the probability of
class C, compared with an average character?"

Grouping is what makes this practical on a CPU (35 players instead of 2,261)
and is exact with respect to the grouped game: the contributions plus the
baseline add up to the model's output.

CACHING
The baseline vector is cached in results/shap/baseline.npy and the explainer is
built around the in-memory model in milliseconds; global SHAP values are saved
to results/shap/global_shap.npz.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src import config, data_loader, models
from src.features import fusion
from src.utils import ModiError, get_logger, save_json, setup_matplotlib

log = get_logger("explain")
E = config.EXPLAIN
SHAP_DIR = config.PATHS.shap
BASELINE_PATH = SHAP_DIR / "baseline.npy"
GLOBAL_NPZ = SHAP_DIR / "global_shap.npz"
GLOBAL_JSON = SHAP_DIR / "global_importance.json"
GLOBAL_FIG = config.PATHS.figures / "shap_global_importance.png"
MAIN_MODEL = config.PATHS.models / "main_stack.joblib"

ROW_NAMES = {4: ["top", "upper-middle", "lower-middle", "bottom"], 3: ["top", "middle", "bottom"], 2: ["top", "bottom"]}
COL_NAMES = {4: ["left", "centre-left", "centre-right", "right"], 3: ["left", "centre", "right"], 2: ["left", "right"]}
FAMILY_TEXT = {"hog": "stroke direction", "lbp": "local stroke pattern", "zernike": "overall shape"}


def _region(i: int, j: int, rows: int, cols: int) -> str:
    r = ROW_NAMES.get(rows, [f"row {k + 1}" for k in range(rows)])[i]
    c = COL_NAMES.get(cols, [f"column {k + 1}" for k in range(cols)])[j]
    return f"{r} {c}"


def build_groups(feature_set: str = "fused") -> list[dict[str, Any]]:
    """Partition the feature vector into readable groups.

    Returns a list of dicts: name, family, columns (index array), and for
    spatial groups ``cell`` = (row, col) on the region grid.
    """
    meta = fusion.feature_meta(feature_set)
    rows, cols = E.hog_region_grid
    buckets: dict[tuple, list[int]] = {}
    for idx, m in enumerate(meta):
        if m["family"] in ("hog", "lbp"):
            i = min(int(m["row"] * rows), rows - 1)
            j = min(int(m["col"] * cols), cols - 1)
            key = (m["family"], i, j)
        else:
            n = m["order"]
            band = 0 if n <= 4 else (1 if n <= 8 else 2)
            key = ("zernike", band, 0)
        buckets.setdefault(key, []).append(idx)
    zern = ["coarse shape (orders 0-4)", "medium detail (orders 5-8)", "fine detail (orders 9+)"]
    groups = []
    for (fam, i, j), cols_idx in sorted(buckets.items(), key=lambda kv: (config.DESCRIPTORS.index(kv[0][0]), kv[0][1], kv[0][2])):
        if fam == "zernike":
            name, cell = f"Zernike: {zern[i]}", None
        else:
            name, cell = f"{fam.upper()}: {FAMILY_TEXT[fam]}, {_region(i, j, rows, cols)}", (i, j)
        groups.append({"name": name, "family": fam, "columns": np.asarray(cols_idx), "cell": cell})
    return groups


class GroupedExplainer:
    """Kernel SHAP over feature groups for the stacking ensemble.

    Args:
        bundle: trained models (preprocessor + stack).
        baseline: raw feature vector representing an "average character"
            (the training mean).
    """

    def __init__(self, bundle: models.TrainedModels, baseline: np.ndarray) -> None:
        import shap

        self.bundle = bundle
        self.baseline = np.asarray(baseline, dtype=np.float32)
        self.groups = build_groups(bundle.feature_set)
        self.classes = bundle.classes
        self._x = self.baseline.copy()
        G = len(self.groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explainer = shap.KernelExplainer(self._predict_masked, np.zeros((1, G)))
        self.base_proba = self.bundle.predict_proba(self.baseline[None, :])[0]

    # -- the "game": which groups show their real content -------------------
    def _compose(self, masks: np.ndarray, x: np.ndarray) -> np.ndarray:
        X = np.repeat(self.baseline[None, :], len(masks), axis=0)
        for g, grp in enumerate(self.groups):
            on = masks[:, g] > 0.5
            if on.any():
                X[np.ix_(on, grp["columns"])] = x[grp["columns"]]
        return X

    def _predict_masked(self, masks: np.ndarray) -> np.ndarray:
        return self.bundle.predict_proba(self._compose(np.atleast_2d(masks), self._x))

    def shap_values(self, x: np.ndarray, nsamples: int = E.kernel_nsamples) -> np.ndarray:
        """SHAP values for one raw feature vector: array (n_groups, n_classes)."""
        self._x = np.asarray(x, dtype=np.float32)
        G = len(self.groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = self._explainer.shap_values(np.ones((1, G)), nsamples=nsamples, silent=True)
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[0] == 1:           # (1, G, C)
            return sv[0]
        if sv.ndim == 3:                                # (C, 1, G) from older shap
            return sv[:, 0, :].T
        raise ModiError(f"Unexpected SHAP output shape {sv.shape}.")

    def explain(self, x: np.ndarray, top_k: int = E.top_k_features, nsamples: int = E.kernel_nsamples) -> dict[str, Any]:
        """Local explanation for one image's feature vector.

        Returns the prediction, the runner-up, and for both classes the
        groups that pushed the probability up or down the most.
        """
        proba = self.bundle.predict_proba(np.asarray(x)[None, :])[0]
        order = np.argsort(proba)[::-1]
        pred, runner = int(order[0]), int(order[1])
        sv = self.shap_values(x, nsamples)
        names = [g["name"] for g in self.groups]

        def ranked(c: int, positive: bool) -> list[dict[str, Any]]:
            vals = sv[:, c]
            idx = np.argsort(-vals if positive else vals)
            out = [{"group": names[i], "family": self.groups[i]["family"], "cell": self.groups[i]["cell"],
                    "shap": float(vals[i])} for i in idx[:top_k] if (vals[i] > 0) == positive and vals[i] != 0]
            return out

        def maps(c: int) -> dict[str, list[list[float]]]:
            rows, cols = E.hog_region_grid
            out = {}
            for fam in ("hog", "lbp"):
                grid = np.zeros((rows, cols))
                for g, grp in enumerate(self.groups):
                    if grp["family"] == fam:
                        grid[grp["cell"]] = sv[g, c]
                out[fam] = grid.tolist()
            return out

        fam_tot = {f: float(sum(sv[g, pred] for g, grp in enumerate(self.groups) if grp["family"] == f))
                   for f in config.DESCRIPTORS}
        return {
            "predicted": str(self.classes[pred]), "predicted_proba": float(proba[pred]),
            "runner_up": str(self.classes[runner]), "runner_up_proba": float(proba[runner]),
            "top3": [{"label": str(self.classes[i]), "proba": float(proba[i])} for i in order[:3]],
            "baseline_proba_predicted": float(self.base_proba[pred]),
            "baseline_proba_runner_up": float(self.base_proba[runner]),
            "toward_predicted": ranked(pred, True),
            "against_predicted": ranked(pred, False),
            "toward_runner_up": ranked(runner, True),
            "family_contribution_predicted": fam_tot,
            "region_maps_predicted": maps(pred),
            "region_maps_runner_up": maps(runner),
            "additivity_error": float(abs(self.base_proba[pred] + sv[:, pred].sum() - proba[pred])),
            "group_names": names,
            "shap_predicted": sv[:, pred].tolist(),
            "shap_runner_up": sv[:, runner].tolist(),
        }


# ---------------------------------------------------------------------------
# Building / caching
# ---------------------------------------------------------------------------
def compute_baseline(bundle: models.TrainedModels) -> np.ndarray:
    """Mean fused feature vector of the main training set ("an average character")."""
    index = data_loader.load_processed_index()
    index["_pos"] = np.arange(len(index))
    train, _, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    X = fusion.load_feature_set(index, bundle.feature_set)
    return X[train["_pos"].to_numpy()].mean(axis=0).astype(np.float32)


def load_explainer(force: bool = False, bundle: models.TrainedModels | None = None) -> GroupedExplainer:
    """Explainer for ``bundle`` (default: the saved main model).

    Only the 2,261-value baseline vector is cached on disk
    (results/shap/baseline.npy); the explainer itself is cheap to build around a
    model that is already in memory. Caching the whole explainer would pickle
    the 270 MB model a second time.
    """
    bundle = bundle or models.load(MAIN_MODEL)
    baseline = None
    if BASELINE_PATH.exists() and not force:
        try:
            if BASELINE_PATH.stat().st_mtime >= MAIN_MODEL.stat().st_mtime:
                baseline = np.load(BASELINE_PATH)
                if baseline.shape != (sum(len(g["columns"]) for g in build_groups(bundle.feature_set)),):
                    baseline = None
        except (OSError, ValueError):
            baseline = None
    if baseline is None:
        baseline = compute_baseline(bundle)
        SHAP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = BASELINE_PATH.with_suffix(".tmp.npy")
        np.save(tmp, baseline)
        tmp.replace(BASELINE_PATH)          # atomic: concurrent readers never see a half-written file
    return GroupedExplainer(bundle, baseline)


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
def _faithfulness(exp: GroupedExplainer, X: np.ndarray, sv_pred: np.ndarray, pred_idx: np.ndarray, k: int = 3) -> dict[str, float]:
    """Deletion test: switching off the k groups SHAP ranks highest should
    lower the predicted class's probability more than switching off k random groups."""
    rng = np.random.default_rng(config.SEED)
    G = len(exp.groups)
    drops_top, drops_rand = [], []
    for x, sv, c in zip(X, sv_pred, pred_idx):
        full = exp.bundle.predict_proba(x[None, :])[0, c]
        m_top = np.ones((1, G)); m_top[0, np.argsort(-sv)[:k]] = 0
        m_rnd = np.ones((1, G)); m_rnd[0, rng.choice(G, k, replace=False)] = 0
        p = exp.bundle.predict_proba(np.vstack([exp._compose(m_top, x), exp._compose(m_rnd, x)]))[:, c]
        drops_top.append(full - p[0]); drops_rand.append(full - p[1])
    return {"groups_removed": k, "mean_probability_drop_top_shap": float(np.mean(drops_top)),
            "mean_probability_drop_random": float(np.mean(drops_rand))}


def run_global(force: bool = False) -> dict[str, Any]:
    """Global importance: mean |SHAP| of each group and family over a stratified
    sample of TEST images (for each image, the SHAP values of its predicted class)."""
    if GLOBAL_JSON.exists() and GLOBAL_NPZ.exists() and not force:
        log.info("Global SHAP already computed - using cache (use --force-shap to recompute)")
        from src.utils import load_json
        result = load_json(GLOBAL_JSON)
        plot_global(result)
        return result
    exp = load_explainer(force)
    index = data_loader.load_processed_index()
    index["_pos"] = np.arange(len(index))
    _, test, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    per_class = max(1, int(np.ceil(E.global_sample_size / test["label"].nunique())))
    sample = test.groupby("label").head(per_class)
    X = fusion.load_feature_set(index, exp.bundle.feature_set)[sample["_pos"].to_numpy()]
    proba = exp.bundle.predict_proba(X)
    pred_idx = proba.argmax(1)
    log.info("Computing grouped Kernel SHAP for %d test images (%d per class, %d model evaluations each)",
             len(X), per_class, E.kernel_nsamples)
    sv_all = np.zeros((len(X), len(exp.groups), len(exp.classes)), dtype=np.float32)
    for i, x in enumerate(X):
        sv_all[i] = exp.shap_values(x)
        if (i + 1) % 25 == 0:
            log.info("  explained %d / %d", i + 1, len(X))
    sv_pred = sv_all[np.arange(len(X)), :, pred_idx]                       # (n, G)
    add_err = np.abs(exp.base_proba[pred_idx] + sv_pred.sum(1) - proba[np.arange(len(X)), pred_idx])
    mean_abs = np.abs(sv_pred).mean(0)
    names = [g["name"] for g in exp.groups]
    fam_of = np.array([g["family"] for g in exp.groups])
    fam_imp = {f: float(mean_abs[fam_of == f].sum()) for f in config.DESCRIPTORS}
    total = sum(fam_imp.values()) or 1.0
    sl = fusion.slices(exp.bundle.feature_set)
    result = {
        "model": "stacking ensemble", "method": "Kernel SHAP over 35 feature groups; baseline = mean training feature vector",
        "n_images": int(len(X)), "images_per_class": per_class, "nsamples": E.kernel_nsamples,
        "family_importance": fam_imp,
        "family_share": {f: v / total for f, v in fam_imp.items()},
        "family_n_features": {f: int(s.stop - s.start) for f, s in sl.items()},
        "family_n_groups": {f: int((fam_of == f).sum()) for f in config.DESCRIPTORS},
        "groups": sorted(({"group": n, "family": f, "cell": g["cell"], "mean_abs_shap": float(v)}
                          for n, f, g, v in zip(names, fam_of, exp.groups, mean_abs)),
                         key=lambda d: -d["mean_abs_shap"]),
        "max_additivity_error": float(add_err.max()), "mean_additivity_error": float(add_err.mean()),
        "faithfulness_deletion_test": _faithfulness(exp, X, sv_pred, pred_idx),
        "accuracy_on_sample": float((exp.classes[pred_idx] == sample["label"].to_numpy()).mean()),
    }
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GLOBAL_NPZ, shap=sv_all, pred_idx=pred_idx, labels=sample["label"].to_numpy().astype(str),
                        group_names=np.asarray(names), classes=exp.classes.astype(str))
    save_json(result, GLOBAL_JSON)
    plot_global(result)
    log.info("  Family share of total importance: %s", ", ".join(f"{f} {100 * s:.1f}%" for f, s in result["family_share"].items()))
    ft = result["faithfulness_deletion_test"]
    log.info("  Faithfulness: removing top-%d SHAP groups drops probability by %.3f vs %.3f for random groups; "
             "max additivity error %.4f", ft["groups_removed"], ft["mean_probability_drop_top_shap"],
             ft["mean_probability_drop_random"], result["max_additivity_error"])
    return result


def plot_global(result: dict[str, Any], path: Path = GLOBAL_FIG) -> Path:
    """Family shares, region heatmaps for HOG and LBP, and the top groups."""
    plt = setup_matplotlib()
    bg, colours = "#fbf7ef", {"hog": "#1f3a5f", "zernike": "#c8812a", "lbp": "#5b8c5a"}
    fig = plt.figure(figsize=(13, 8.2), dpi=130); fig.patch.set_facecolor(bg)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.38, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(bg)
    fams = list(result["family_share"])
    share = [100 * result["family_share"][f] for f in fams]
    bars = ax.bar([f.upper() for f in fams], share, color=[colours[f] for f in fams])
    for b, f, v in zip(bars, fams, share):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%\n({result['family_n_features'][f]} features)",
                ha="center", fontsize=8)
    ax.set_ylim(0, max(share) * 1.3); ax.set_ylabel("share of total mean |SHAP| (%)")
    ax.set_title("Which descriptor family matters most?", fontsize=10)

    rows, cols = config.EXPLAIN.hog_region_grid
    for k, fam in enumerate(("hog", "lbp")):
        ax = fig.add_subplot(gs[0, k + 1])
        grid = np.zeros((rows, cols))
        for g in result["groups"]:
            if g["family"] == fam and g["cell"] is not None:
                grid[tuple(g["cell"])] = g["mean_abs_shap"]
        im = ax.imshow(grid, cmap="YlOrBr" if fam == "hog" else "YlGn")
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{fam.upper()}: importance by image region", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(bg)
    top = result["groups"][:15][::-1]
    ax.barh(range(len(top)), [g["mean_abs_shap"] for g in top], color=[colours[g["family"]] for g in top])
    ax.set_yticks(range(len(top))); ax.set_yticklabels([g["group"] for g in top], fontsize=8)
    ax.set_xlabel("mean |SHAP| on the predicted class probability")
    ax.set_title(f"Top feature groups (stacking ensemble, {result['n_images']} test images)", fontsize=10)
    for a in fig.axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Contrastive explanations for the most confused pairs (novelty)
# ---------------------------------------------------------------------------
PAIRS_JSON = SHAP_DIR / "confused_pairs_contrast.json"
PAIRS_FIG = config.PATHS.figures / "shap_confused_pairs.png"


def contrast_pairs(pairs: list[tuple[str, str]], n_per_class: int = 8, force: bool = False) -> dict[str, Any]:
    """Which feature groups separate two easily confused characters?

    For every test image of class A, the contrast of a group is
    SHAP(group -> A) - SHAP(group -> B): how much that group's actual content
    favours the TRUE class over its look-alike (and symmetrically for B).
    Averaged over images of both classes, a large positive value marks the
    part of the glyph the model relies on to tell the pair apart; groups near
    zero carry no distinguishing evidence.
    """
    from src.utils import load_json

    if PAIRS_JSON.exists() and not force:
        cached = load_json(PAIRS_JSON)
        if [tuple(p["pair"]) for p in cached["pairs"]] == [tuple(p) for p in pairs]:
            plot_pairs(cached)
            return cached
    exp = load_explainer()
    index = data_loader.load_processed_index()
    index["_pos"] = np.arange(len(index))
    _, test, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    X = fusion.load_feature_set(index, exp.bundle.feature_set)
    cls = list(exp.classes)
    names = [g["name"] for g in exp.groups]
    out = []
    for a, b in pairs:
        ia, ib = cls.index(a), cls.index(b)
        contrasts, examples = [], {}
        for lab, own, other in ((a, ia, ib), (b, ib, ia)):
            rows = test[test["label"] == lab].head(n_per_class)
            examples[lab] = str(rows["processed_path"].iloc[0])
            for pos in rows["_pos"]:
                sv = exp.shap_values(X[pos])
                contrasts.append(sv[:, own] - sv[:, other])
        c = np.mean(contrasts, axis=0)
        order = np.argsort(-c)
        rows_, cols_ = E.hog_region_grid
        maps = {}
        for fam in ("hog", "lbp"):
            grid = np.zeros((rows_, cols_))
            for g, grp in enumerate(exp.groups):
                if grp["family"] == fam:
                    grid[grp["cell"]] = c[g]
            maps[fam] = grid.tolist()
        fam_share = {f: float(sum(max(c[g], 0) for g, grp in enumerate(exp.groups) if grp["family"] == f))
                     for f in config.DESCRIPTORS}
        tot = sum(fam_share.values()) or 1.0
        out.append({"pair": [a, b], "n_images": len(contrasts), "examples": examples,
                    "top_distinguishing_groups": [{"group": names[i], "contrast": float(c[i])} for i in order[:5]],
                    "family_share_of_distinguishing_evidence": {f: v / tot for f, v in fam_share.items()},
                    "region_maps": maps})
        log.info("  %s vs %s: top separating evidence = %s", a, b, names[order[0]])
    result = {"method": "mean over test images of SHAP(true class) - SHAP(look-alike class), per feature group",
              "pairs": out}
    save_json(result, PAIRS_JSON)
    plot_pairs(result)
    return result


def plot_pairs(result: dict[str, Any], path: Path = PAIRS_FIG) -> Path:
    """One row per confused pair: example glyphs, where the separating evidence lies, and top groups."""
    from src import preprocess

    plt = setup_matplotlib()
    pairs = result["pairs"]
    fig, axes = plt.subplots(len(pairs), 5, figsize=(14, 2.5 * len(pairs)), dpi=120,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 2.6]})
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor("#fbf7ef")
    for r, p in enumerate(pairs):
        a, b = p["pair"]
        for c, lab in enumerate((a, b)):
            img = preprocess.read_image(config.PROJECT_ROOT / p["examples"][lab])
            axes[r, c].imshow(255 - img, cmap="gray"); axes[r, c].set_title(lab, fontsize=10); axes[r, c].axis("off")
        for c, fam in ((2, "hog"), (3, "lbp")):
            g = np.asarray(p["region_maps"][fam]); m = float(np.abs(g).max()) or 1.0
            axes[r, c].imshow(g, cmap="RdBu_r", vmin=-m, vmax=m)
            axes[r, c].set_title(f"{fam.upper()} regions that\nseparate the pair", fontsize=8)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        ax = axes[r, 4]; ax.axis("off")
        lines = [f"{a}  vs  {b}  ({p['n_images']} test images)"] + [
            f"  {i + 1}. {g['group']}  (+{g['contrast']:.3f})" for i, g in enumerate(p["top_distinguishing_groups"][:3])]
        sh = p["family_share_of_distinguishing_evidence"]
        lines.append("  evidence share: " + ", ".join(f"{f.upper()} {100 * v:.0f}%" for f, v in sh.items()))
        ax.text(0, 0.5, "\n".join(lines), va="center", fontsize=9, family="monospace")
    fig.suptitle("What tells look-alike characters apart? Contrastive SHAP on the most confused pairs "
                 "(red = region favours the true class)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
