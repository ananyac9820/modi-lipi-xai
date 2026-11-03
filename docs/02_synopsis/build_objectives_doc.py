"""Writes Objectives_and_Implementation.md and an HTML copy for Word, with line numbers read from the code."""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
LOOK = json.loads((ROOT / "results/metrics/lookalike.json").read_text(encoding="utf-8"))


def at(path: str, name: str) -> str:
    """`path`, `name()` (line n), with n found by searching the file."""
    for i, line in enumerate((ROOT / path).read_text(encoding="utf-8").split("\n"), 1):
        if re.match(rf"\s*(def|class) {name}\b", line):
            return f"`{path}`, `{name}()` (line {i})"
    raise SystemExit(f"{name} not found in {path}")


def line_of(path: str, text: str) -> int:
    for i, line in enumerate((ROOT / path).read_text(encoding="utf-8").split("\n"), 1):
        if text in line:
            return i
    raise SystemExit(f"{text!r} not found in {path}")


RUN = "scripts/run_lookalike.py"
acc = LOOK["accuracy"]
rows = sorted(LOOK["per_pair"], key=lambda p: p["correct_svm"] - p["correct_final"])
pair_lines = "\n".join(
    f"| {' / '.join(p['pair'])} | {p['images_consulted']} | {p['correct_svm']} | {p['correct_final']} | {p['correct_final'] - p['correct_svm']:+d} |"
    for p in rows)

MD = f"""# Where each objective is implemented

Project: Handwritten Modi Character Recognition using Handcrafted Features (Group TY-A2)

All paths are inside the project folder `modi-lipi-xai`. To train and test everything for objectives 2, 3 and 4, run `python scripts\\run_lookalike.py`.

## Objective 1: Recognise 57 Modi characters with one model, an SVM on handcrafted features

Base papers: Mandke et al. 2024, Anam & Gupta 2015, Kulkarni et al. 2014

| What | Where |
|---|---|
| Cleaning with Otsu thresholding | {at("src/preprocess.py", "binarize")} |
| HOG features (stroke direction, 1,764 values) | {at("src/features/hog_features.py", "extract")} |
| Zernike moments (overall shape, 49 values) | {at("src/features/zernike_features.py", "extract")} |
| LBP features (small local patterns, 448 values) | {at("src/features/lbp_features.py", "extract")} |
| Joining the three into 2,261 values | {at("src/features/fusion.py", "extract_one")} |
| Training the SVM on those features, after LDA | {at("src/lookalike.py", "fit_svm")} |
| Testing what each feature adds | {at("scripts/run_training.py", "run_ablation")} |
| Result files | `results/metrics/lookalike.json`, `results/metrics/ablation.json` |
| In the app | Results page, Model tab |

What we got: the SVM alone reads {100 * acc["svm_vote"]:.2f}% of the 11,400 test images correctly, against the 77% Mandke et al. report for their SVM (a different dataset, so it is only a rough comparison). Without HOG it falls to 81.4%, without LBP to 83.9%, without Zernike only to 86.0%.

## Objective 2: Find the pairs the SVM mixes up most, and use SHAP to see what separates them

Base papers: Kolte & Gupta 2025, Anam & Gupta 2015, Ekbote et al. 2022

| What | Where |
|---|---|
| The SVM's guesses on held-out training images | {at(RUN, "oof_scores")} |
| Counting errors to get the 15 most confused pairs | `{RUN}`, `main()` (line {line_of(RUN, "# 2. look-alike pairs")}) |
| SHAP ranking of the 35 regions for one pair | {at("src/lookalike.py", "rank_groups")} |
| Picking the top 10 regions | {at("src/lookalike.py", "choose_columns")} |
| Most confused pairs on the test set | {at("src/evaluate.py", "most_confused_pairs")} |
| SHAP on the main SVM, and for confused pairs | {at("src/explain.py", "run_global")}, {at("src/explain.py", "contrast_pairs")} |
| Result files | `results/metrics/confused_pairs_final.json`, `results/shap/confused_pairs_contrast.json` |
| In the app | Explain page, Look-alikes tab; Results page, Errors tab |

What we got: {len(LOOK["candidate_pairs"])} candidate pairs from training data. For i and ii, SHAP points to the top right of the character, which is where the extra hook of ii sits.

## Objective 3: Build a small specialist for each pair, and let it decide when the SVM is stuck between them

This is the new idea in our project. Base paper: Kolte & Gupta 2025.

| What | Where |
|---|---|
| Training one specialist on its 10 regions | {at("src/lookalike.py", "fit_specialist")} |
| Checking it on held-out training images | {at(RUN, "test_pair_oof")} |
| Keeping it only if it beats the SVM there | `{RUN}`, `main()` (line {line_of(RUN, "# 3. keep a specialist")}) |
| Top two guesses of the SVM | {at("src/lookalike.py", "top_two")} |
| The rule: if the top two are a kept pair, the specialist decides | {at("src/lookalike.py", "decide")} |
| Using it to read a character | {at("src/predict.py", "recognize")} |
| Saved model | `models/main_svm_lookalike.joblib` |
| In the app | Read page: tap a character, and it says when a specialist decided |

What we got: {len(LOOK["pairs_kept"])} of the {len(LOOK["candidate_pairs"])} specialists passed the training check and were kept.

## Objective 4: Check on a fixed test set whether the specialists really reduce mistakes

Base paper: Bhong & Kulkarni 2025

| What | Where |
|---|---|
| Scoring the test set | `{RUN}`, `main()` (line {line_of(RUN, "# 5. the test set")}) |
| McNemar test on paired predictions | {at(RUN, "mcnemar")} |
| Result files | `results/metrics/lookalike.json`, `results/metrics/per_class_final.json` |
| In the app | Results page, Look-alikes tab |

What we got: {100 * acc["svm_vote"]:.2f}% with the SVM alone and {100 * acc["final"]:.2f}% with the specialists. They fixed {LOOK["against_svm_vote"]["fixed"]} images and broke {LOOK["against_svm_vote"]["broken"]} (McNemar p = {LOOK["against_svm_vote"]["p_value"]:.3f}). Pair by pair, on the images each specialist decided:

| Pair | Images | SVM right | Final right | Change |
|---|---|---|---|---|
{pair_lines}

Nine pairs improve and three get worse.
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    return re.sub(r"`([^`]+)`", r'<span style="font-family:Consolas;font-size:9pt">\1</span>', s)


def to_html(md: str) -> str:
    out, table = [], []

    def flush():
        if not table:
            return
        out.append('<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;width:100%;font-size:10pt">')
        for i, row in enumerate(table):
            tag = "th" if i == 0 else "td"
            cells = "".join(f"<{tag}>{inline(c.strip())}</{tag}>" for c in row)
            out.append(f"<tr>{cells}</tr>")
        out.append("</table>")
        table.clear()

    for line in md.split("\n"):
        if line.startswith("|"):
            cells = line.strip("|").split("|")
            if not set(line) <= set("|- "):
                table.append(cells)
            continue
        flush()
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush()
    return ('<html><head><meta charset="utf-8"></head><body style="font-family:Calibri;font-size:11pt">'
            + "\n".join(out) + "</body></html>")


(OUT / "Objectives_and_Implementation.md").write_text(MD, encoding="utf-8")
(OUT / "objectives.tmp.html").write_text(to_html(MD), encoding="utf-8")
print("written")
