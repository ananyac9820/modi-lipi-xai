"""Stage 3 runner: extract HOG, Zernike and LBP features for every processed
image and cache them in data/features/.

Usage:
    python scripts/run_feature_extraction.py            # all three descriptors (cached)
    python scripts/run_feature_extraction.py --force    # re-extract
    python scripts/run_feature_extraction.py --only lbp # a single descriptor

Outputs:
    data/features/<family>_<hash>.npy
    results/metrics/feature_summary.json
    results/figures/feature_examples.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import config, data_loader  # noqa: E402
from src.features import fusion, hog_features, lbp_features, zernike_features  # noqa: E402
from src.utils import get_logger, run_cli, save_json, set_seed, setup_matplotlib, stage  # noqa: E402

log = get_logger("run_feature_extraction")
SUMMARY = config.PATHS.metrics / "feature_summary.json"
EXAMPLES_FIG = config.PATHS.figures / "feature_examples.png"


def save_examples_figure(index, images: np.ndarray, n: int = 5) -> Path:
    """One row per sample: glyph | HOG | LBP code map | Zernike magnitudes."""
    plt = setup_matplotlib()
    rng = np.random.default_rng(config.SEED)
    labels = rng.permutation(index["label"].unique())[:n]
    rows = [index.index[index["label"] == lab][0] for lab in labels]
    bg = "#fbf7ef"
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.3 * len(rows)), dpi=120,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.8]})
    fig.patch.set_facecolor(bg)
    titles = ["preprocessed glyph", "HOG: stroke direction", "LBP: local patterns", "Zernike: shape moments |Z|"]
    for r, pos in enumerate(rows):
        img = images[pos]
        axes[r, 0].imshow(img, cmap="gray", interpolation="nearest")
        axes[r, 1].imshow(hog_features.visualize(img), cmap="magma")
        axes[r, 2].imshow(lbp_features.codes(img), cmap="viridis", interpolation="nearest")
        z = zernike_features.extract(img)
        axes[r, 3].bar(range(len(z)), z, color="#1f3a5f", width=0.8)
        axes[r, 3].set_facecolor(bg)
        axes[r, 3].tick_params(labelsize=7)
        for side in ("top", "right"):
            axes[r, 3].spines[side].set_visible(False)
        for c in range(3):
            axes[r, c].axis("off")
        axes[r, 0].text(-0.08, 0.5, index.loc[pos, "label"], transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=10)
        if r == 0:
            for c in range(4):
                axes[r, c].set_title(titles[c], fontsize=9)
    axes[-1, 3].set_xlabel("moment index (low order = coarse shape → high order = fine detail)", fontsize=8)
    fig.suptitle("What each descriptor family sees", fontsize=11)
    fig.tight_layout()
    fig.savefig(EXAMPLES_FIG, facecolor=bg)
    plt.close(fig)
    return EXAMPLES_FIG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="ignore cached features")
    parser.add_argument("--only", choices=config.DESCRIPTORS, help="extract a single descriptor")
    args = parser.parse_args()

    set_seed()
    config.PATHS.ensure()
    t0 = time.perf_counter()
    with stage("Loading processed images", log):
        index = data_loader.load_processed_index()
        images = data_loader.load_processed_images(index)
    log.info("%d images, %d classes, image size %s", len(index), index["label"].nunique(), images.shape[1:])

    infos = []
    for fam in ([args.only] if args.only else config.DESCRIPTORS):
        with stage(f"Extracting {fam.upper()} features", log):
            X, info = fusion.load_or_extract(index, images, fam, force=args.force)
            info.update(mean=float(X.mean()), std=float(X.std()),
                        constant_features=int((X.std(axis=0) < 1e-8).sum()))
            infos.append(info)
            log.info("  %s: %d features%s, %d constant columns", fam, X.shape[1],
                     " (from cache)" if info["cached"] else f", {info['ms_per_image']} ms/image", info["constant_features"])

    total = sum(i["n_features"] for i in infos)
    summary = {"n_images": int(len(index)), "n_classes": int(index["label"].nunique()),
               "image_size": config.PREPROCESS.image_size, "families": infos,
               "fused_dimension": total if not args.only else None,
               "family_balanced_fusion": config.FUSION.balance_families,
               "config": {k: config.as_dict()[k] for k in ("hog", "zernike", "lbp", "fusion")}}
    save_json(summary, SUMMARY)
    with stage("Saving descriptor example figure", log):
        save_examples_figure(index, images)
    log.info("Feature dimensions: %s -> fused %d", ", ".join(f"{i['family']} {i['n_features']}" for i in infos), total)
    log.info("Total time %.1fs. Next: python scripts/run_training.py", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    run_cli(main)
