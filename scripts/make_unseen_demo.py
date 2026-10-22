"""Build a folder of demo images the models have NEVER seen.

Only raw dataset images that are not in the processed index are used. Those
were never preprocessed, so no model (classical or CNN, any curve point) was
trained or tested on them. Output, in demo_unseen/ (gitignored, dataset images
are not redistributed):

  singles/NN_<label>.png   one image per class (57), file name = true label
  words/word_K_<labels>.png  a few multi-character lines composed from unseen images
  answers.csv              true labels, plus what the model predicts

Usage:
    python scripts/make_unseen_demo.py            # build + check with the model
    python scripts/make_unseen_demo.py --no-check
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import config, data_loader  # noqa: E402
from src.utils import get_logger, run_cli  # noqa: E402

log = get_logger("make_unseen_demo")
OUT = config.PROJECT_ROOT / "demo_unseen"
RAW = config.PATHS.data_raw
WORDS = [["k", "m", "l", "r", "s"], ["a", "aa", "i", "u", "e"], ["g", "p", "v", "h"], ["one", "two", "three", "four", "five"],
         ["ch", "ja", "ta", "d", "n"]]


def unseen_pool() -> dict[str, list[Path]]:
    index = data_loader.load_processed_index()
    used = {str(Path(p)).lower() for p in index["rel_path"]}
    pool: dict[str, list[Path]] = {}
    for label, rows in index.groupby("label"):
        folder = (RAW / rows["rel_path"].iloc[0]).parent
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        fresh = [p for p in files if str(p.relative_to(RAW)).lower() not in used]
        pool[str(label)] = fresh
    return pool


def load_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {p}")
    return img


def as_dark_on_light(img: np.ndarray) -> np.ndarray:
    return 255 - img if img.mean() < 127 else img


def clean_letter(p: Path) -> np.ndarray:
    """The letter's ink only (scan frame, grid lines and specks removed), dark on white,
    so a composed word looks like a written word rather than glued scan tiles."""
    from src import preprocess

    g = preprocess.to_gray(preprocess.read_image(p))
    binary, _ = preprocess.binarize(preprocess.denoise(g))
    binary, _ = preprocess.remove_border_artifacts(binary)
    binary, _ = preprocess.remove_grid_lines(binary)
    binary, _ = preprocess.remove_specks(binary)
    return 255 - preprocess.crop_to_ink(binary)


def compose(paths: list[Path], height: int = 90, gap: int = 34) -> np.ndarray:
    tiles = []
    for p in paths:
        g = clean_letter(p)
        w = max(1, int(g.shape[1] * height / g.shape[0]))
        tiles.append(cv2.resize(g, (w, height), interpolation=cv2.INTER_AREA))
    pad = np.full((height, gap), 255, np.uint8)
    row = np.hstack([x for t in tiles for x in (t, pad)][:-1])
    return cv2.copyMakeBorder(row, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-check", action="store_true", help="skip running the model on the demo images")
    args = ap.parse_args()

    pool = unseen_pool()
    rng = np.random.default_rng(2026)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "singles").mkdir(parents=True)
    (OUT / "words").mkdir()
    rows = []
    for k, label in enumerate(sorted(pool), start=1):
        if not pool[label]:
            log.warning("no unseen image left for %s", label)
            continue
        src = pool[label][int(rng.integers(len(pool[label])))]
        dst = OUT / "singles" / f"{k:02d}_{label}.png"
        cv2.imwrite(str(dst), as_dark_on_light(load_gray(src)))
        rows.append({"file": f"singles/{dst.name}", "truth": label, "source": str(src.relative_to(RAW))})
    for k, word in enumerate(WORDS, start=1):
        word = [w for w in word if pool.get(w)]
        picks = [pool[w][int(rng.integers(len(pool[w])))] for w in word]
        dst = OUT / "words" / f"word_{k}_{'-'.join(word)}.png"
        cv2.imwrite(str(dst), compose(picks))
        rows.append({"file": f"words/{dst.name}", "truth": " ".join(word), "source": "; ".join(str(p.relative_to(RAW)) for p in picks)})

    if not args.no_check:
        from src.predict import Recognizer

        rec = Recognizer()
        ok_single = ok_chars = n_chars = 0
        for r in rows:
            img = cv2.imread(str(OUT / r["file"]))
            single = r["file"].startswith("singles/")
            res = rec.recognize(img, single=True if single else None)
            pred = [c.label or "?" for c in res.chars]
            r["predicted"] = " ".join(pred)
            r["confidence"] = " ".join(f"{c.confidence:.2f}" for c in res.chars)
            truth = r["truth"].split()
            if single:
                ok_single += pred[:1] == truth
            else:
                n_chars += len(truth)
                ok_chars += sum(a == b for a, b in zip(pred, truth)) if len(pred) == len(truth) else 0
        n_single = sum(r["file"].startswith("singles/") for r in rows)
        log.info("Single characters: %d / %d correct (%.0f%%)", ok_single, n_single, 100 * ok_single / max(n_single, 1))
        log.info("Word lines: %d / %d characters correct (counted only when segmentation found the right number)", ok_chars, n_chars)

    with open(OUT / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if args.no_check else ["file", "truth", "predicted", "confidence", "source"])
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %d images to %s", len(rows), OUT)
    return 0


if __name__ == "__main__":
    run_cli(main)
