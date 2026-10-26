"""REFERENCE BASELINE: train the untuned CNN at every learning-curve point.

Objective 4 (synopsis): trains the comparison CNN at every training size.

Uses exactly the same seeded splits and fixed test set as the classical
models, scores with the same function, and writes
results/metrics/learning_curve_cnn.json plus the headline comparison figure
results/figures/learning_curve_classical_vs_cnn.png (the classical curve is
added automatically once milestone 5 has produced it).

This is NOT part of the Modi Lipi XAI pipeline or app.

Usage:
    python scripts/run_cnn_baseline.py                    # all curve points (resumes)

Safe to stop at any time (close the window / Ctrl+C): every epoch is
checkpointed in models/baselines/checkpoints/, and the next run continues from
the last finished epoch with bit-identical results. So the long points can be
run over several nights.
    python scripts/run_cnn_baseline.py --sizes 72 90 250  # selected points
    python scripts/run_cnn_baseline.py --estimate         # time 1 epoch, extrapolate, train nothing
    python scripts/run_cnn_baseline.py --force            # retrain points already done
    python scripts/run_cnn_baseline.py --plot-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import config, data_loader, learning_curve  # noqa: E402
from src.utils import ModiError, get_logger, run_cli, set_seed, stage  # noqa: E402

log = get_logger("run_cnn_baseline")
MODEL_DIR = config.PATHS.models / "baselines"


def _estimate(images: np.ndarray, index, classes: list[str], sizes: list[int]) -> None:
    """Time one epoch on the largest feasible point and extrapolate linearly."""
    import torch

    from src.baselines import cnn_baseline as cb

    cb.seed_everything()
    n_ref = min(500, max(sizes))
    train, _, _ = data_loader.stratified_split(index, n_ref, quiet=True)
    X = cb._to_tensor(images[train["_pos"].to_numpy()])
    y = torch.tensor([classes.index(c) for c in train["label"]])
    model = cb.PlainCNN(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=config.CNN_BASELINE.learning_rate)
    t = time.perf_counter()
    cb._run_epoch(model, X, y, opt, torch.Generator().manual_seed(0))
    per_img = (time.perf_counter() - t) / len(X)
    C = config.CNN_BASELINE
    log.info("One epoch on %d images: %.1fs (%.2f ms/image)", len(X), per_img * len(X), 1000 * per_img)
    log.info("Estimated training time per point (phase 1 up to %d epochs + refit):", C.max_epochs)
    for n in sizes:
        n_img = n * len(classes)
        lo = per_img * n_img * (C.early_stopping_patience + 5) * (2 if C.refit_on_full_train else 1)
        hi = per_img * n_img * C.max_epochs * (2 if C.refit_on_full_train else 1)
        log.info("  %5d/class (%7d images): ~%5.0f - %6.0f min", n, n_img, lo / 60, hi / 60)


def _keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs (released automatically
    when it exits). Per-process request, no system setting is changed. Closing the
    lid can still sleep the laptop, depending on its power settings."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(config.SAMPLING.curve_train_sizes))
    parser.add_argument("--force", action="store_true", help="retrain points already in the results file")
    parser.add_argument("--estimate", action="store_true", help="only time one epoch and extrapolate")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    set_seed()
    config.PATHS.ensure()
    _keep_awake()
    if args.plot_only:
        learning_curve.plot_headline()
        return 0
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ModiError("PyTorch is not installed. It is a BASELINE-only dependency: pip install -r requirements.txt") from exc
    from src.baselines import cnn_baseline as cb

    with stage("Loading processed images", log):
        index = data_loader.load_processed_index()
        images = data_loader.load_processed_images(index)
        index = index.assign(_pos=np.arange(len(index)))
        classes = sorted(index["label"].unique())
    log.info("%d images, %d classes", len(index), len(classes))

    if args.estimate:
        _estimate(images, index, classes, sorted(args.sizes))
        return 0

    header = {"model": "Untuned reference CNN", "label": "Reference CNN (untuned)", "family": "cnn_baseline",
              "protocol": cb.protocol_description(), "architecture": cb.describe_architecture(len(classes)),
              "classes": classes}
    done = {p["train_per_class"] for p in (learning_curve.load_curve(learning_curve.CNN_JSON) or {}).get("points", [])}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for n in sorted(args.sizes):
        if n in done and not args.force:
            log.info("Point %d/class already done - skipping (use --force to retrain)", n)
            continue
        with stage(f"CNN baseline @ {n} training images per class", log):
            train, test, split = data_loader.stratified_split(index, n)
            ckpt_dir = MODEL_DIR / "checkpoints" / f"cnn_{n}"
            model, info = cb.train_and_evaluate(images[train["_pos"].to_numpy()], train["label"].to_numpy(), classes,
                                                ckpt_dir=ckpt_dir)
            t = time.perf_counter()
            proba = cb.predict_proba(model, images[test["_pos"].to_numpy()])
            infer_ms = 1000 * (time.perf_counter() - t) / len(test)
            pred = np.asarray(classes)[proba.argmax(1)]
            scores = learning_curve.score_predictions(test["label"].to_numpy(), pred, test["category"].to_numpy(),
                                                      split["short_classes"])
            import torch

            torch.save(model.state_dict(), MODEL_DIR / f"cnn_{n}.pt")
            point = {"train_per_class": n, **scores, **info, "inference_ms_per_image": round(infer_ms, 4),
                     "n_test": int(len(test)), "reduced_test_classes": split["reduced_test_classes"]}
            learning_curve.upsert_point(learning_curve.CNN_JSON, header, point)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            log.info("  @%d/class: accuracy %.2f%%, macro-F1 %.4f, train %.0fs, %d params (%.1f MB)%s",
                     n, 100 * scores["accuracy"], scores["macro_f1"], info["train_time_s"], info["n_parameters"],
                     info["model_size_bytes"] / 1e6, "" if scores["complete"] else "  [INCOMPLETE POINT]")

    learning_curve.plot_headline()
    return 0


if __name__ == "__main__":
    run_cli(main)
