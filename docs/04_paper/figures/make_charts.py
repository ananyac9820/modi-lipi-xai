"""IEEE-style charts for the paper, drawn from the project's result files."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "results"
OUT = Path(__file__).parent
j = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False,
})
W = 3.4  # single IEEE column, inches
NAVY, GREY = "#1c3670", "#a8a8a8"

# ---- Fig 3: wrong answers per pair, before and after the specialist ----
pairs = j("metrics/lookalike.json")["per_pair"]
pairs.sort(key=lambda p: p["correct_final"] - p["correct_svm"])
fig, ax = plt.subplots(figsize=(W, 2.6))
for i, p in enumerate(pairs):
    before, after = p["images_consulted"] - p["correct_svm"], p["images_consulted"] - p["correct_final"]
    ax.plot([before, after], [i, i], color="#777777", lw=1, zorder=1)
    ax.scatter(before, i, s=20, color=GREY, zorder=2, label="SVM alone" if i == 0 else None)
    ax.scatter(after, i, s=20, color=NAVY, zorder=3, label="With specialist" if i == 0 else None)
    d = p["correct_final"] - p["correct_svm"]
    ax.text(max(before, after) + 2.5, i, f"{d:+d}", va="center", fontsize=6.5)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels([" / ".join(p["pair"]) for p in pairs])
ax.set_xlim(0, 112)
ax.set_xlabel("Wrong answers on the images the specialist decided")
ax.grid(axis="x", lw=0.3, color="#cccccc")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_pairs.png", bbox_inches="tight", pad_inches=0.02)

# ---- Fig 4: ablation ----
a = j("metrics/ablation.json")["feature_sets"]
order = [("zernike", "Zernike only"), ("lbp", "LBP only"), ("hog", "HOG only"), ("zernike+lbp", "Zernike + LBP"),
         ("hog+zernike", "HOG + Zernike"), ("hog+lbp", "HOG + LBP"), ("fused", "All three")]
vals = [100 * a[k]["models"]["svm"]["accuracy"] for k, _ in order]
fig, ax = plt.subplots(figsize=(W, 1.75))
cols = ["#bbbbbb"] * 3 + ["#777777"] * 3 + [NAVY]
bars = ax.barh(range(len(order)), vals, color=cols, height=0.62)
for b, v in zip(bars, vals):
    ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=6.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([n for _, n in order])
ax.set_xlim(40, 92)
ax.set_xlabel("SVM test accuracy (%), 500 training images per class")
ax.grid(axis="x", lw=0.3, color="#cccccc")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight", pad_inches=0.02)

# ---- Fig 5: SHAP ----
g = j("shap/global_importance.json")
fams = ["hog", "lbp", "zernike"]
names = {"hog": "HOG", "lbp": "LBP", "zernike": "Zernike"}
nfeat = g["family_n_features"]
tot = sum(nfeat.values())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(3)
a1.bar(x - 0.19, [100 * nfeat[f] / tot for f in fams], 0.36, color="#bbbbbb", label="Share of features")
a1.bar(x + 0.19, [100 * g["family_share"][f] for f in fams], 0.36, color=NAVY, label="Share of SHAP evidence")
for i, f in enumerate(fams):
    a1.text(i + 0.19, 100 * g["family_share"][f] + 1.5, f"{100 * g['family_share'][f]:.0f}", ha="center", fontsize=6)
a1.set_xticks(x)
a1.set_xticklabels([names[f] for f in fams])
a1.set_ylabel("%")
a1.set_ylim(0, 92)
a1.legend(frameon=False, loc="upper right", fontsize=5.8)
a1.set_title("(a) Descriptor families", fontsize=7)
grid = np.zeros((4, 4))
for grp in g["groups"]:
    if grp["family"] == "hog" and grp["cell"]:
        grid[tuple(grp["cell"])] = grp["mean_abs_shap"]
im = a2.imshow(grid, cmap="Greys", vmin=0)
for r in range(4):
    for c in range(4):
        a2.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=5.5,
                color="white" if grid[r, c] > 0.6 * grid.max() else "black")
a2.set_xticks([])
a2.set_yticks([])
for s in a2.spines.values():
    s.set_visible(True)
    s.set_linewidth(0.6)
a2.set_title("(b) HOG evidence by region", fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_shap.png", bbox_inches="tight", pad_inches=0.02)
print("hog centre share", grid[1:3, 1:3].sum() / grid.sum(), "top row share", grid[0].sum() / grid.sum())
print("charts written")
