"""Stage 1 runner: scan the raw dataset, report real statistics, preprocess
every image into data/processed/, and save diagnostic figures.

Usage:
    python scripts/run_preprocessing.py                     # the pool every experiment needs
    python scripts/run_preprocessing.py --max-per-class 0   # every image on disk
    python scripts/run_preprocessing.py --max-per-class 500 # a smaller seeded subset
    python scripts/run_preprocessing.py --force             # ignore cache
    python scripts/run_preprocessing.py --summary-only      # just the dataset stats

Outputs:
    results/metrics/dataset_summary.json
    results/figures/class_distribution.png
    results/figures/preprocessing_examples.png
    data/processed/<label>/*.png         (ink = 255 on black, IMAGE_SIZE square)
    data/processed/processed_index.csv   (the exact image set used downstream)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src import config, data_loader, preprocess  # noqa: E402
from src.utils import ModiError, get_logger, load_json, run_cli, save_json, set_seed, setup_matplotlib, stage  # noqa: E402

log = get_logger("run_preprocessing")
INDEX_PATH = config.PATHS.data_processed / "processed_index.csv"
EXAMPLES_FIG = config.PATHS.figures / "preprocessing_examples.png"
CHUNK = 4000


def process_all(scan: data_loader.DatasetScan, manifest: pd.DataFrame, force: bool, n_jobs: int) -> pd.DataFrame:
    """Preprocess every manifest row (skipping cached outputs). Returns the index."""
    cache_ok = preprocess.cache_is_valid() and not force
    if not cache_ok and not force and any(config.PATHS.data_processed.glob("*/*.png")):
        log.info("Preprocessing parameters changed since the last run -> reprocessing everything.")
    dest = [preprocess.processed_path(r, l) for r, l in zip(manifest["rel_path"], manifest["label"])]
    def _fresh(d: Path, src_mtime: float) -> bool:
        try:
            return d.stat().st_mtime >= src_mtime
        except OSError:
            return False

    todo_mask = np.array([not (cache_ok and _fresh(d, m)) for d, m in zip(dest, manifest["mtime"])])
    log.info("%d images selected; %d cached, %d to process", len(manifest), int((~todo_mask).sum()), int(todo_mask.sum()))

    errors: dict[str, str] = {}
    infos: list[dict] = []
    todo = manifest[todo_mask]
    start = time.perf_counter()
    for i in range(0, len(todo), CHUNK):
        chunk = todo.iloc[i: i + CHUNK]
        results = Parallel(n_jobs=n_jobs, batch_size=64)(
            delayed(preprocess.process_one)(str(scan.root / r), str(d), r)
            for r, d in zip(chunk["rel_path"], np.asarray(dest, dtype=object)[todo_mask][i: i + CHUNK]))
        for rel, err, info in results:
            if err:
                errors[rel] = err
            else:
                infos.append(info)
        done = min(i + CHUNK, len(todo))
        rate = done / max(time.perf_counter() - start, 1e-9)
        log.info("  %d / %d processed  (%.0f img/s, ETA %.0fs)", done, len(todo), rate, (len(todo) - done) / rate)
    if len(todo):
        preprocess.mark_cache_valid()

    # class_rank is kept so downstream splits reuse this exact seeded order.
    index = manifest[["rel_path", "label", "category", "category_raw", "class_rank"]].copy()
    index["processed_path"] = [str(d.relative_to(config.PROJECT_ROOT)) for d in dest]
    index = index[~index["rel_path"].isin(errors)].reset_index(drop=True)
    # Cached files from an earlier run are fine; anything still missing failed before.
    missing = ~index["processed_path"].map(lambda p: (config.PROJECT_ROOT / p).exists())
    for rel in index.loc[missing, "rel_path"]:
        errors.setdefault(rel, "processed file missing (failed in an earlier run)")
    index = index[~missing].reset_index(drop=True)
    index.to_csv(INDEX_PATH, index=False)

    stats = {
        "selected": int(len(manifest)),
        "processed_ok": int(len(index)),
        "failed": len(errors),
        "failed_by_reason": dict(Counter(_short(e) for e in errors.values()).most_common()),
        "failed_examples": dict(list(errors.items())[:25]),
        "params_hash": preprocess.params_hash(),
        "polarity_convention": "ink = 255 (white) on background = 0 (black)",
    }
    if infos:
        stats["this_run"] = {
            "images": len(infos),
            "deskew_applied_fraction": round(float(np.mean([x["deskew_applied"] for x in infos])), 4),
            "dark_ink_on_light_fraction": round(float(np.mean([x["dark_ink_on_light"] for x in infos])), 4),
            "polarity_decided_by_minority_rule": int(sum(x["polarity_rule"] == "minority" for x in infos)),
        }
    return index, stats


def _short(err: str) -> str:
    """Collapse an error message into its reason category (text before ':')."""
    return err.split(":")[0].rstrip(".")


def save_examples_figure(scan: data_loader.DatasetScan, manifest: pd.DataFrame, n: int) -> Path:
    """Before/after grid: original | Otsu binary | cleaned + deskewed | final."""
    plt = setup_matplotlib()
    rng =np.random.default_rng(config.SEED)
    labels = rng.permutation(manifest["label"].unique())[:n]
    picks = [manifest[manifest["label"] == lab].sample(1, random_state=config.SEED).iloc[0] for lab in labels]
    cols = ["original", "binary", "deskewed", "final"]
    titles = ["original", "Otsu binary\n(ink = white)", "frame + specks\nremoved, cropped", f"final {config.PREPROCESS.image_size}×{config.PREPROCESS.image_size}"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.3 * len(cols), 2.0 * len(picks)), dpi=120)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor("#fbf7ef")
    for r, row in enumerate(picks):
        try:
            res = preprocess.preprocess_file(scan.root / row["rel_path"], keep_steps=True)
            info = res.info
            note = f"slant {info['slant_deg']}°" + (" (corrected)" if info["deskew_applied"] else f" ({info['deskew_skip_reason']})")
        except ModiError as exc:
            res, note = None, str(exc)
        for c, key in enumerate(cols):
            ax = axes[r, c]
            ax.axis("off")
            if res is not None:
                ax.imshow(res.steps[key], cmap="gray", interpolation="nearest")
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].text(-0.08, 0.5, f"{row['label']}\n{row['width']}×{row['height']}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8)
        axes[r, 2].text(0.5, -0.08, note, transform=axes[r, 2].transAxes, ha="center", va="top", fontsize=6.5)
    fig.suptitle("Preprocessing check: does stroke detail survive?", fontsize=11)
    fig.tight_layout()
    EXAMPLES_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXAMPLES_FIG, facecolor=fig.get_facecolor())
    plt.close(fig)
    return EXAMPLES_FIG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pool = config.SAMPLING.pool_per_class
    parser.add_argument("--max-per-class", type=int, default=pool,
                        help=f"first N images of each class's seeded order (default {pool} = "
                             f"{config.SAMPLING.test_per_class} test + {pool - config.SAMPLING.test_per_class} "
                             "largest training size); 0 = all")
    parser.add_argument("--force", action="store_true", help="reprocess even if cached")
    parser.add_argument("--rescan", action="store_true", help="re-read every image header")
    parser.add_argument("--summary-only", action="store_true", help="only scan + report dataset stats")
    parser.add_argument("--n-jobs", type=int, default=config.PREPROCESS.n_jobs)
    parser.add_argument("--examples", type=int, default=8, help="rows in the before/after figure")
    args = parser.parse_args()

    set_seed()
    config.PATHS.ensure()
    t0 = time.perf_counter()

    with stage("Scanning dataset", log):
        scan = data_loader.scan_dataset(rescan=args.rescan)
    with stage("Writing dataset summary + class distribution", log):
        summary = data_loader.write_summary(scan)
    if args.summary_only:
        return 0

    with stage("Selecting images (deterministic per-class cap)", log):
        manifest, cap_report = data_loader.cap_per_class(scan.manifest, args.max_per_class or None)
        log.info("Per-class counts used: %s",
                 ", ".join(f"{k}={v}" for k, v in cap_report["per_class"].items()))

    with stage("Preprocessing images", log):
        index, stats = process_all(scan, manifest, args.force, args.n_jobs)
    with stage("Checking split plan for the processed pool (short-class policy)", log):
        # This is the set training will actually use (preprocessing failures removed).
        stats["split_plan_processed_pool"] = data_loader.plan_splits(index)[1]
    stats["max_per_class"] = args.max_per_class or None
    stats["per_class_used"] = cap_report["per_class"]
    stats["classes_below_cap"] = cap_report["short_classes"]
    summary = load_json(data_loader.SUMMARY_PATH)
    summary["preprocessing"] = stats
    save_json(summary, data_loader.SUMMARY_PATH)

    with stage("Saving before/after figure", log):
        save_examples_figure(scan, manifest, args.examples)

    log.info("Processed OK: %d | failed: %d %s", stats["processed_ok"], stats["failed"],
             stats["failed_by_reason"] or "")
    if stats["processed_ok"] == 0:
        raise ModiError("Every image failed preprocessing - check the examples figure and the log.")
    log.info("Index of processed images: %s", INDEX_PATH.relative_to(config.PROJECT_ROOT))
    log.info("Total time %.1fs. Next: python scripts/run_feature_extraction.py", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    run_cli(main)
