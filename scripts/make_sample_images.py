"""Build the app's bundled sample images (app/assets/).

Synthetic samples are composed from REAL handwritten character images taken
from the fixed TEST split, so the models never saw them during training and
the true labels are known (saved in app/assets/samples.json). One extra
sample is a crop of a real manuscript page: a deliberately hard case that
shows the limits of segmentation on connected cursive writing.

Usage:
    python scripts/make_sample_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import config, data_loader, preprocess  # noqa: E402
from src.utils import get_logger, run_cli, save_json  # noqa: E402

log = get_logger("make_samples")
ASSETS = config.PATHS.app_assets
PAPER = 236


def glyph_crop(rel_path: str) -> np.ndarray:
    """Grey crop of one raw character, trimmed to its ink with a small margin."""
    gray = preprocess.read_image(config.PATHS.data_raw / rel_path)
    binary, _ = preprocess.binarize(preprocess.denoise(gray))
    binary, _ = preprocess.remove_border_artifacts(binary)
    binary, _ = preprocess.remove_specks(binary)
    ys, xs = np.nonzero(binary)
    m = 6
    y0, y1 = max(0, ys.min() - m), min(gray.shape[0], ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(gray.shape[1], xs.max() + m + 1)
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    ink = binary[y0:y1, x0:x1] > 0
    out = np.full(crop.shape, PAPER, np.float32)        # clean paper, keep only real ink strokes
    out[ink] = np.minimum(crop[ink], 120)
    return cv2.GaussianBlur(out, (3, 3), 0.6).astype(np.uint8)


def compose(rows: list[list[np.ndarray]], gap: int = 26, line_gap: int = 40, height: int = 110,
            rotate_deg: float = 0.0, seed: int = 0) -> np.ndarray:
    """Place glyph crops left-to-right in lines on a paper background."""
    rng = np.random.default_rng(seed)
    lines = []
    for glyphs in rows:
        scaled = [cv2.resize(g, (max(1, int(g.shape[1] * height / g.shape[0])), height), interpolation=cv2.INTER_AREA)
                  for g in glyphs]
        w = sum(s.shape[1] for s in scaled) + gap * (len(scaled) + 1)
        line = np.full((height + 2 * gap, w), PAPER, np.uint8)
        x = gap
        for s in scaled:
            dy = int(rng.integers(-6, 7))
            line[gap + dy: gap + dy + height, x: x + s.shape[1]] = np.minimum(
                line[gap + dy: gap + dy + height, x: x + s.shape[1]], s)
            x += s.shape[1] + gap + int(rng.integers(-6, 7))
        lines.append(line)
    W = max(l.shape[1] for l in lines) + 2 * gap
    canvas = np.full((sum(l.shape[0] for l in lines) + line_gap * (len(lines) + 1), W), PAPER, np.uint8)
    y = line_gap
    for l in lines:
        canvas[y: y + l.shape[0], gap: gap + l.shape[1]] = l
        y += l.shape[0] + line_gap
    if rotate_deg:
        h, w = canvas.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)
        canvas = cv2.warpAffine(canvas, M, (w, h), borderValue=PAPER)
    noise = rng.normal(0, 3, canvas.shape)
    return np.clip(canvas + noise, 0, 255).astype(np.uint8)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    index = data_loader.load_processed_index()
    _, test, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    rng = np.random.default_rng(config.SEED)

    def pick(label: str, k: int = 0) -> tuple[str, np.ndarray]:
        rows = test[test["label"] == label]
        r = rows.iloc[k % len(rows)]
        return label, glyph_crop(r["rel_path"])

    specs = {
        "sample_1_consonants.png": ([["k", "m", "l", "r", "s"]], 0.0, "Five consonants on one line"),
        "sample_2_two_lines.png": ([["a", "i", "u", "e"], ["g", "p", "v", "h"]], 2.5, "Two lines, slightly tilted"),
        "sample_3_digits.png": ([["one", "nine", "four", "seven", "two", "zero"]], 0.0, "A row of Modi digits"),
    }
    meta = []
    for name, (rows, rot, desc) in specs.items():
        picked = [[pick(lab, int(rng.integers(0, 50))) for lab in row] for row in rows]
        img = compose([[g for _, g in row] for row in picked], rotate_deg=rot, seed=len(meta))
        cv2.imencode(".png", img)[1].tofile(str(ASSETS / name))
        truth = [lab for row in picked for lab, _ in row]
        meta.append({"file": name, "description": desc, "ground_truth": truth, "kind": "composed from test-split images"})
        log.info("Wrote %s (%d characters)", name, len(truth))

    docs = sorted((config.PATHS.data_raw).glob("MODI_HHdoc/**/*.jpg"))
    if docs:
        page = preprocess.read_image(docs[0])
        h, w = page.shape
        crop = page[int(0.09 * h): int(0.29 * h), int(0.08 * w): int(0.85 * w)]
        crop = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imencode(".png", crop)[1].tofile(str(ASSETS / "sample_4_real_manuscript.png"))
        meta.append({"file": "sample_4_real_manuscript.png", "kind": "real manuscript crop (MODI-HHDoc)",
                     "description": "Real manuscript lines: connected cursive writing (hard case)", "ground_truth": None})
        log.info("Wrote sample_4_real_manuscript.png from %s", docs[0].name)
    save_json({"samples": meta, "note": "Composed samples use test-split images only, never training images."},
              ASSETS / "samples.json")
    class_reference_sheet()
    return 0



def class_reference_sheet() -> Path:
    """One real example per class (training split), labelled with the folder name
    and the inferred Devanagari, so the label mapping can be checked by a person."""
    import json

    from src.utils import setup_matplotlib

    plt = setup_matplotlib()
    index = data_loader.load_processed_index()
    train, _, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    labels = json.loads((config.PROJECT_ROOT / "src" / "labels.json").read_text(encoding="utf-8"))["labels"]
    order = [c for cat in ("vowel", "consonant", "digit") for c in sorted(l for l, v in labels.items() if v["category"] == cat)]
    cols = 10
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0), dpi=120)
    fig.patch.set_facecolor("#fbf7ef")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, lab in zip(axes.ravel(), order):
        ex = train[train["label"] == lab].head(3)
        tiles = [255 - preprocess.read_image(config.PROJECT_ROOT / p) for p in ex["processed_path"]]
        ax.imshow(np.hstack(tiles), cmap="gray", vmin=0, vmax=255)
        dv = labels[lab]["devanagari"] or "?"
        ax.set_title(f"{lab}  {dv}", fontsize=10, color="#b5523b" if dv == "?" else "#1f3a5f")
    fig.suptitle("Class reference sheet: 3 training examples per class. Red '?' = dental/retroflex letter "
                 "not identifiable from the folder name. Please verify src/labels.json against this.", fontsize=11)
    fig.tight_layout()
    out = config.PATHS.figures / "class_reference_sheet.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Wrote %s", out.relative_to(config.PROJECT_ROOT))
    return out


if __name__ == "__main__":
    run_cli(main)
