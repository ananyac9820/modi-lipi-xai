"""Side-by-side sheet for verifying the 10 dental / retroflex class labels.

The dataset's romanised folder names (ta, th, tha, thh, thha, d, dh, dha, n,
nn) do not say which of the ten Devanagari-order letters they are. This
script crops the reference glyphs of those ten letters from the official
Unicode code chart (paper 30 in docs/03_reference_papers, Figure 1) and puts
them next to real handwritten examples of each ambiguous class, so a person
can match them by eye and then fill in src/labels.json.

Usage:
    python scripts/make_label_verification_sheet.py
Output: results/figures/label_verification_sheet.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import config, data_loader, preprocess  # noqa: E402
from src.utils import ModiError, get_logger, run_cli, setup_matplotlib  # noqa: E402

log = get_logger("label_sheet")
PDF = config.PROJECT_ROOT / "docs" / "03_reference_papers" / "30_2011_Pandey_Proposal_to_Encode_the_Modi_Script_in_ISO_IEC_10646.pdf"
CHART_PAGE = 19                      # 0-based page holding "Figure 1: Proposed code chart for Modi"
LETTERS = [("11618", "ट", "TTA"), ("11619", "ठ", "TTHA"), ("1161A", "ड", "DDA"), ("1161B", "ढ", "DDHA"),
           ("1161C", "ण", "NNA"), ("1161D", "त", "TA"), ("1161E", "थ", "THA"), ("1161F", "द", "DA"),
           ("11620", "ध", "DHA"), ("11621", "न", "NA")]
AMBIGUOUS = ["ta", "th", "tha", "thh", "thha", "d", "dh", "dha", "n", "nn"]


def chart_glyphs() -> dict[str, np.ndarray]:
    """Crop each letter's cell from the code chart, located by its codepoint label."""
    try:
        import pymupdf
    except ImportError as exc:
        raise ModiError("PyMuPDF is needed to read the code chart: pip install pymupdf") from exc
    if not PDF.exists():
        raise ModiError(f"Unicode proposal PDF not found: {PDF}")
    page = pymupdf.open(PDF)[CHART_PAGE]
    words = {w[4]: w for w in page.get_text("words")}
    out = {}
    for cp, _, _ in LETTERS:
        if cp not in words:
            raise ModiError(f"Codepoint label {cp} not found on the chart page.")
        x0, y0, x1, _ = words[cp][:4]
        cx = (x0 + x1) / 2
        clip = pymupdf.Rect(cx - 15, y0 - 33, cx + 15, y0 - 1)
        pix = page.get_pixmap(dpi=300, clip=clip, colorspace=pymupdf.csGRAY)
        out[cp] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    return out


def main() -> int:
    plt = setup_matplotlib()
    chart = chart_glyphs()
    index = data_loader.load_processed_index()
    examples = {lab: index[index["label"] == lab].head(6)["rel_path"].tolist() for lab in AMBIGUOUS}

    fig = plt.figure(figsize=(16, 12), dpi=110)
    fig.patch.set_facecolor("#fbf7ef")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4.2], hspace=0.12)
    top = gs[0].subgridspec(1, len(LETTERS), wspace=0.15)
    for k, (cp, dev, name) in enumerate(LETTERS):
        ax = fig.add_subplot(top[k]); ax.imshow(chart[cp], cmap="gray"); ax.axis("off")
        ax.set_title(f"{dev}  {name}\nU+{cp}", fontsize=10)
    fig.text(0.5, 0.905, "Reference glyphs from the Unicode Modi code chart (Pandey 2011, Figure 1)", ha="center",
             fontsize=11)
    bottom = gs[1].subgridspec(len(AMBIGUOUS), 7, wspace=0.05, hspace=0.25, width_ratios=[0.8] + [1] * 6)
    for r, lab in enumerate(AMBIGUOUS):
        ax = fig.add_subplot(bottom[r, 0]); ax.axis("off")
        ax.text(0.5, 0.5, f"dataset\nclass '{lab}'", ha="center", va="center", fontsize=10, fontweight="bold")
        for c, rel in enumerate(examples[lab]):
            ax = fig.add_subplot(bottom[r, c + 1]); ax.axis("off")
            ax.imshow(preprocess.read_image(config.PATHS.data_raw / rel), cmap="gray")
    fig.suptitle("Which letter is each ambiguous class?  Match each row below to a reference glyph above, "
                 "then set 'devanagari' and 'verified': true in src/labels.json", fontsize=12, y=0.99)
    out = config.PATHS.figures / "label_verification_sheet.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out.relative_to(config.PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    run_cli(main)
