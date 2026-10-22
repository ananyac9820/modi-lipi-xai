"""Evaluate segmentation on synthetic words with KNOWN answers.

Words of 3-5 characters are composed from real handwritten character images
taken from the fixed TEST split (never used in training). Characters are
pushed together so their strokes touch, and a continuous headline joins the
word, as in written Modi. Each word is recognised twice:

  * projection segmentation only (blank columns)
  * projection + recognition-guided re-segmentation (src/resegment.py)

Metrics: correct character count, character error rate (Levenshtein distance
between predicted and true label sequences / true length), and exact-word
accuracy. Also measured on SEPARATED characters as a control.

Usage:
    python scripts/eval_segmentation.py              # 120 words (default)
    python scripts/eval_segmentation.py --words 300
Output: results/metrics/segmentation_eval.json, results/figures/segmentation_examples.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import config, data_loader, preprocess, segmentation  # noqa: E402
from src.predict import Recognizer  # noqa: E402
from src.utils import get_logger, run_cli, save_json, set_seed, setup_matplotlib, stage  # noqa: E402

log = get_logger("eval_segmentation")
PAPER = 236


def glyph(rel_path: str, height: int = 110) -> np.ndarray:
    """Clean grey crop of one test character, scaled to a common height."""
    gray = preprocess.read_image(config.PATHS.data_raw / rel_path)
    b, _ = preprocess.binarize(preprocess.denoise(gray))
    b, _ = preprocess.remove_border_artifacts(b)
    b, _ = preprocess.remove_grid_lines(b)
    b, _ = preprocess.remove_specks(b)
    ys, xs = np.nonzero(b)
    b = b[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    scale = height / b.shape[0]
    b = cv2.resize(b, (max(1, int(b.shape[1] * scale)), height), interpolation=cv2.INTER_NEAREST)
    return b


def compose_word(glyphs: list[np.ndarray], overlap: int, headline: bool, rng: np.random.Generator) -> np.ndarray:
    """Ink masks side by side (negative gap = touching), optional joining headline."""
    h = max(g.shape[0] for g in glyphs)
    pad = 30
    width = sum(g.shape[1] for g in glyphs) - overlap * (len(glyphs) - 1) + 2 * pad
    ink = np.zeros((h + 2 * pad, width), np.uint8)
    x = pad
    for g in glyphs:
        ink[pad: pad + g.shape[0], x: x + g.shape[1]] = np.maximum(ink[pad: pad + g.shape[0], x: x + g.shape[1]], g)
        x += g.shape[1] - overlap
    if headline:
        cv2.line(ink, (pad - 4, pad + 2), (x + overlap + 4, pad + 2), 255, 4)
    img = np.full(ink.shape, PAPER, np.float32)
    img[ink > 0] = 60
    img = cv2.GaussianBlur(img, (3, 3), 0.7) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def levenshtein(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--words", type=int, default=120)
    parser.add_argument("--split", choices=["test", "train"], default="test",
                        help="build words from test images (reported numbers) or training images (for tuning)")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    args = parser.parse_args()
    set_seed()
    rng = np.random.default_rng(config.SEED)
    index = data_loader.load_processed_index()
    train, test, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
    pool = test if args.split == "test" else train
    letters = pool[pool["category"] != "digit"]
    rec = Recognizer()
    _ = rec.bundle

    settings = {"separated": dict(overlap=-28, headline=False), "touching": dict(overlap=6, headline=True)}
    results: dict[str, dict[str, list]] = {k: {"plain": [], "reseg": []} for k in settings}
    examples = []
    with stage(f"Evaluating segmentation on {args.words} synthetic words per setting", log):
        for w in range(args.words):
            k = int(rng.integers(3, 6))
            rows = letters.sample(k, random_state=int(rng.integers(1e9)))
            truth = rows["label"].tolist()
            glyphs = [glyph(r) for r in rows["rel_path"]]
            for name, kw in settings.items():
                img = compose_word(glyphs, rng=rng, **kw)
                for mode, flag in (("plain", False), ("reseg", True)):
                    t = time.perf_counter()
                    res = rec.recognize(img, resegment=flag)
                    pred = [c.label or "?" for c in res.chars]
                    results[name][mode].append({"truth": truth, "pred": pred, "n_true": k, "n_pred": len(pred),
                                                "edits": levenshtein(pred, truth), "seconds": time.perf_counter() - t})
                    if name == "touching" and w < 4:
                        examples.append((img, res, mode, truth))
            if (w + 1) % 20 == 0:
                log.info("  %d / %d words", w + 1, args.words)

    summary = {}
    for name, modes in results.items():
        summary[name] = {}
        for mode, rs in modes.items():
            n_chars = sum(r["n_true"] for r in rs)
            summary[name][mode] = {
                "words": len(rs),
                "correct_count_rate": float(np.mean([r["n_pred"] == r["n_true"] for r in rs])),
                "character_error_rate": float(sum(r["edits"] for r in rs) / n_chars),
                "character_accuracy_when_count_correct": float(np.mean(
                    [np.mean([p == t for p, t in zip(r["pred"], r["truth"])]) for r in rs if r["n_pred"] == r["n_true"]]
                    or [np.nan])),
                "exact_word_accuracy": float(np.mean([r["pred"] == r["truth"] for r in rs])),
                "mean_seconds_per_word": float(np.mean([r["seconds"] for r in rs])),
            }
            s = summary[name][mode]
            log.info("  %-9s %-6s count right %5.1f%% | CER %5.1f%% | word acc %5.1f%% | %.1fs/word", name, mode,
                     100 * s["correct_count_rate"], 100 * s["character_error_rate"], 100 * s["exact_word_accuracy"],
                     s["mean_seconds_per_word"])
    save_json({"method": f"synthetic words from {args.split}-split character images; letters only; 3-5 characters",
               "split": args.split, "reseg_piece_penalty": config.SEGMENT.reseg_piece_penalty,
               "settings": settings, "summary": summary,
               "note": "Real manuscripts are harder (ligatures, slant, variable spacing); this measures the "
                       "segmentation logic under controlled conditions with known answers."},
              config.PATHS.metrics / f"segmentation_eval{args.tag}.json")

    plt = setup_matplotlib()
    fig, axes = plt.subplots(len(examples) // 2, 2, figsize=(14, 2.3 * len(examples) // 2), dpi=110)
    for ax, (img, res, mode, truth) in zip(axes.ravel(), examples):
        ax.imshow(segmentation.draw_boxes(res.segmentation, [c.label or "?" for c in res.chars]))
        ax.set_title(f"{'projection only' if mode == 'plain' else 'with recognition-guided split'} | "
                     f"true: {' '.join(truth)}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.PATHS.figures / f"segmentation_examples{args.tag}.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    run_cli(main)
