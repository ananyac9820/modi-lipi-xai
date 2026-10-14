"""End-to-end environment smoke test.

Imports succeeding is not proof that a library works (numba-backed SHAP in
particular can import fine and then fail when it JIT-compiles). This script
exercises every critical library on tiny dummy data and prints PASS/FAIL per
check. Exit code is non-zero if anything fails.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

SEED = 42


def _glyph(size: int = 64) -> np.ndarray:
    """A synthetic binary 'character': a ring with a vertical stroke."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    img = ((r > size * 0.25) & (r < size * 0.35)).astype(np.uint8)
    img[size // 4 : 3 * size // 4, size // 2 - 2 : size // 2 + 2] = 1
    return img


def check_numpy_sklearn() -> str:
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(SEED)
    X, y = rng.normal(size=(200, 10)), rng.integers(0, 3, 200)
    acc = RandomForestClassifier(n_estimators=20, random_state=SEED).fit(X, y).score(X, y)
    return f"RF train acc on random data = {acc:.2f}"


def check_numba() -> str:
    import numba

    @numba.njit(cache=False)
    def _sum_sq(a: np.ndarray) -> float:
        s = 0.0
        for v in a:
            s += v * v
        return s

    val = _sum_sq(np.arange(10, dtype=np.float64))
    assert val == 285.0, val
    return f"numba {numba.__version__} JIT-compiled and ran (sum of squares = {val:.0f})"


def check_shap_tree() -> str:
    import shap
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)  # only features 0 and 1 matter
    model = RandomForestClassifier(n_estimators=50, random_state=SEED).fit(X, y)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X[:50]))
    assert np.isfinite(sv).all() and np.abs(sv).sum() > 0, "SHAP values empty or non-finite"
    # Sanity: the informative features must rank top by mean |SHAP|.
    # shap >= 0.45 returns (n_samples, n_features, n_classes).
    assert sv.shape == (50, X.shape[1], 2), f"unexpected SHAP shape {sv.shape}"
    imp = np.abs(sv).mean(axis=(0, 2))
    top2 = set(np.argsort(imp)[-2:].tolist())
    assert top2 == {0, 1}, f"unexpected top features {top2}"
    return f"shap {shap.__version__} TreeExplainer values shape {sv.shape}; top features = {sorted(top2)} (expected [0, 1])"


def check_shap_kernel() -> str:
    """KernelExplainer is what milestone 6 uses on the stacking ensemble."""
    import shap
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 5))
    y = (X[:, 2] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    explainer = shap.KernelExplainer(model.predict_proba, X[:20])
    sv = np.asarray(explainer.shap_values(X[:3], nsamples=100, silent=True))
    assert np.isfinite(sv).all()
    return f"KernelExplainer values shape {sv.shape}"


def check_xgboost() -> str:
    import xgboost as xgb

    rng = np.random.default_rng(SEED)
    X, y = rng.normal(size=(200, 6)), rng.integers(0, 3, 200)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, tree_method="hist", random_state=SEED)
    proba = model.fit(X, y).predict_proba(X[:5])
    assert proba.shape == (5, 3)
    import shap

    sv = np.asarray(shap.TreeExplainer(model).shap_values(X[:5]))
    return f"xgboost {xgb.__version__} fit/predict OK; SHAP on XGBoost shape {sv.shape}"


def check_cv2() -> str:
    import cv2

    img = (_glyph() * 255).astype(np.uint8)
    t, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _ = cv2.connectedComponents(binary)
    return f"opencv {cv2.__version__} Otsu threshold={t:.0f}, connected components={n - 1}"


def check_hog() -> str:
    from skimage.feature import hog

    f = hog(_glyph().astype(float), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    assert f.shape == (1764,) and np.isfinite(f).all()
    return f"skimage HOG -> {f.shape[0]} features"


def check_lbp() -> str:
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern((_glyph() * 255).astype(np.uint8), P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    assert hist.sum() == 64 * 64
    return f"skimage LBP uniform codes in [{lbp.min():.0f}, {lbp.max():.0f}]"


def check_zernike() -> str:
    import mahotas

    g = _glyph()
    z = mahotas.features.zernike_moments(g, radius=32, degree=12)
    z_rot = mahotas.features.zernike_moments(np.rot90(g).copy(), radius=32, degree=12)
    assert z.shape == (49,) and np.isfinite(z).all()
    drift = float(np.abs(z - z_rot).max())
    return f"mahotas {mahotas.__version__} Zernike -> {z.shape[0]} moments; max change under 90° rotation = {drift:.2e}"


def check_plotting() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly
    import seaborn  # noqa: F401

    fig, ax = plt.subplots()
    ax.plot([0, 1])
    plt.close(fig)
    return f"matplotlib {matplotlib.__version__}, plotly {plotly.__version__}, seaborn OK"


def check_torch_baseline() -> str:
    """Baseline-only dependency: one deterministic CPU training step."""
    import torch

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.ReLU(), torch.nn.Flatten(),
                                torch.nn.Linear(4 * 62 * 62, 3))
    x, y = torch.rand(8, 1, 64, 64), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    return f"torch {torch.__version__} conv forward/backward OK on CPU (BASELINE dependency only)"


def check_streamlit() -> str:
    import streamlit

    return f"streamlit {streamlit.__version__} importable"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("numpy + scikit-learn", check_numpy_sklearn),
    ("numba JIT", check_numba),
    ("SHAP TreeExplainer (RF)", check_shap_tree),
    ("SHAP KernelExplainer", check_shap_kernel),
    ("XGBoost (+ SHAP)", check_xgboost),
    ("OpenCV", check_cv2),
    ("skimage HOG", check_hog),
    ("skimage LBP", check_lbp),
    ("mahotas Zernike", check_zernike),
    ("plotting libs", check_plotting),
    ("streamlit", check_streamlit),
    ("PyTorch (CNN baseline only)", check_torch_baseline),
]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}\n")
    failures = 0
    for name, fn in CHECKS:
        start = time.perf_counter()
        try:
            detail = fn()
            print(f"  PASS  {name:<26} {detail}  ({time.perf_counter() - start:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - we report every failure
            failures += 1
            print(f"  FAIL  {name:<26} {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
    print()
    if failures:
        print(f"{failures} check(s) FAILED - fix the environment before running the pipeline.")
        return 1
    print("All checks PASSED - environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
