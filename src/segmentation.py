"""Stage 2 - Segmentation of an image containing several Modi characters.

Pipeline:
    grayscale -> (downscale) -> blank check -> Otsu binarisation (ink = 255)
    -> speck removal -> page skew correction (projection-profile search)
    -> LINES: horizontal projection profile
    -> per line: find the HEADLINE (shirorekha), ignore it while looking for
       character boundaries, split on empty columns of the vertical
       projection, refine with connected components, split over-wide blobs
       at their deepest column valley, merge stray fragments (dots, matras)
    -> crops in reading order (top-to-bottom lines, left-to-right characters),
       each with its share of the headline, as dark ink on white like the
       training images.

HONEST LIMITS: Modi was written as continuous cursive. Characters inside a
word often touch below the headline; projection-based segmentation cannot
separate them reliably. Every box therefore carries flags and a confidence,
and the result carries plain-language warnings that the app shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src import config, preprocess
from src.utils import get_logger

log = get_logger("segmentation")
S = config.SEGMENT

FLAG_TEXT = {
    "wide": "unusually wide: may be two or more characters joined together",
    "narrow": "unusually narrow: may be a fragment of a character",
    "tall": "unusually tall: may include parts of another line",
    "connected": "part of a long connected stroke: cursive writing could not be split reliably",
    "resegmented": "split from a joined group by recognition-guided segmentation: the cut is the model's best guess",
}


@dataclass
class CharBox:
    """One detected character, in the coordinates of ``SegmentationResult.display``."""

    x: int
    y: int
    w: int
    h: int
    line: int
    order: int
    flags: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "line": self.line, "order": self.order,
                "flags": self.flags, "segmentation_confidence": round(self.confidence, 2)}


@dataclass
class SegmentationResult:
    """Boxes, crops and diagnostics for one input image."""

    display: np.ndarray                    # grayscale image the boxes refer to (scaled + deskewed)
    binary: np.ndarray                     # ink = 255
    boxes: list[CharBox]
    crops: list[np.ndarray]                # uint8 dark ink on white, one per box
    lines: list[tuple[int, int]]
    warnings: list[str]
    skew_deg: float = 0.0
    scale: float = 1.0
    steps: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        return bool(self.boxes) and not self.warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _runs(mask: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """Index ranges [start, end) where ``mask`` is True, merging gaps shorter than ``min_gap``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            runs.append((start, prev + 1)); start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def _clean(binary: np.ndarray) -> np.ndarray:
    """Remove specks: components far smaller than a pen dot at this resolution."""
    min_area = max(6, int(2e-5 * binary.size))
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def estimate_skew(binary: np.ndarray) -> float:
    """Skew angle (degrees) that makes text lines horizontal.

    Tries angles in [-max_skew, +max_skew] and keeps the one whose horizontal
    projection profile is sharpest (highest variance): when lines are level,
    ink concentrates in a few rows.
    """
    small = binary
    if binary.shape[1] > 600:
        f = 600 / binary.shape[1]
        small = cv2.resize(binary, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_score = 0.0, -1.0
    for a in np.arange(-S.max_skew_deg, S.max_skew_deg + 1e-9, S.skew_step_deg):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def _rotate(img: np.ndarray, angle: float, border: int) -> np.ndarray:
    """Rotate about the centre, enlarging the canvas so nothing is cut off."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=border)


def find_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """Text-line bands (row ranges) from the horizontal projection profile."""
    prof = (binary > 0).sum(axis=1)
    if prof.max() == 0:
        return []
    h = binary.shape[0]
    rows = prof > S.projection_noise_ratio * prof.max()
    bands = _runs(rows, min_gap=max(1, int(S.line_min_gap_ratio * h)))
    min_h = max(3, int(S.line_min_height_ratio * h))
    return [(a, b) for a, b in bands if b - a >= min_h]


def headline_rows(line: np.ndarray) -> tuple[int, int] | None:
    """Row range of the headline inside a line band, or None if there is none.

    The headline is the row with the most ink, provided it spans most of the
    line's ink width; neighbouring rows with comparable ink belong to it.
    """
    prof = (line > 0).sum(axis=1)
    if prof.max() == 0:
        return None
    cols = np.flatnonzero((line > 0).any(axis=0))
    ink_width = cols[-1] - cols[0] + 1
    peak = int(prof.argmax())
    if prof[peak] < S.headline_min_coverage * ink_width:
        return None
    thresh = S.headline_band_ratio * prof[peak]
    a = peak
    while a > 0 and prof[a - 1] >= thresh:
        a -= 1
    b = peak + 1
    while b < len(prof) and prof[b] >= thresh:
        b += 1
    # Headlines are thin; a thick band means this is not a headline.
    if b - a > 0.25 * line.shape[0]:
        return None
    return a, b


def _valley_split(profile: np.ndarray, c0: int, c1: int) -> int | None:
    """Column (absolute) to split a wide segment at, or None if no clear valley."""
    seg = profile[c0:c1].astype(float)
    if seg.size < 6:
        return None
    lo, hi = int(0.25 * seg.size), int(0.75 * seg.size)
    if hi <= lo:
        return None
    k = lo + int(np.argmin(seg[lo:hi]))
    return c0 + k if seg[k] <= S.valley_split_ratio * seg.max() else None


def _to_crop(mask: np.ndarray) -> np.ndarray:
    """Ink mask (255 = ink) -> dark ink on white, square, with a margin like the dataset images."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.full((32, 32), 255, np.uint8)
    m = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = m.shape
    side = int(math.ceil(max(h, w) * (1 + 2 * S.crop_margin_ratio)))
    canvas = np.full((side, side), 255, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0: y0 + h, x0: x0 + w] = np.where(m > 0, 0, 255)
    return canvas


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def segment(image: np.ndarray, keep_steps: bool = True) -> SegmentationResult:
    """Split an image with several characters into crops in reading order."""
    gray = preprocess.to_gray(image)
    scale = 1.0
    if gray.shape[1] > S.max_working_width:
        scale = S.max_working_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    steps: dict[str, np.ndarray] = {"original": gray} if keep_steps else {}
    den = preprocess.denoise(gray)
    if not preprocess.has_contrast(den):
        return SegmentationResult(gray, np.zeros_like(gray), [], [], [], ["The image looks blank: no ink was found."],
                                  scale=scale, steps=steps)
    binary, _ = preprocess.binarize(den)
    binary = _clean(binary)

    skew = estimate_skew(binary)
    if abs(skew) >= S.min_skew_to_correct_deg:
        gray = _rotate(gray, skew, border=int(np.median(gray)))
        binary = _clean(np.where(_rotate(binary, skew, border=0) > 127, 255, 0).astype(np.uint8))
    else:
        skew = 0.0
    if keep_steps:
        steps["binary"] = binary.copy()

    warnings: list[str] = []
    if abs(skew) >= S.max_skew_deg - S.skew_step_deg:
        warnings.append(f"The text is strongly tilted (about {abs(skew):.0f}°); line detection may be unreliable.")

    lines = find_lines(binary)
    # 1) Split each line on blank columns of the FULL image (headline included),
    #    so a character whose parts touch only through its own headline stays whole.
    raw: list[list[int]] = []                                 # x0, x1, y0, y1, line
    work_vis = binary.copy()
    for li, (ya, yb) in enumerate(lines):
        prof = (binary[ya:yb] > 0).sum(axis=0)
        raw.extend([c0, c1, ya, yb, li] for c0, c1 in _runs(prof > 0, min_gap=S.min_col_gap))
    if not raw:
        warnings.append("No characters were detected.")
        return SegmentationResult(gray, binary, [], [], lines, warnings, skew, scale, steps)

    def ink(seg: list[int]) -> int:
        return int((binary[seg[2]:seg[3], seg[0]:seg[1]] > 0).sum())

    # 2) Drop specks: far less ink than a typical character.
    med_ink = float(np.median([ink(r) for r in raw]))
    raw = [r for r in raw if ink(r) >= 0.04 * med_ink] or raw
    med_w = float(np.median([r[1] - r[0] for r in raw]))
    line_h = {li: yb - ya for li, (ya, yb) in enumerate(lines)}

    # 3) Over-wide segments are probably several characters joined by the
    #    headline or by cursive strokes: remove the headline INSIDE the
    #    segment and split on the columns that become empty; failing that,
    #    split at the deepest column valley.
    refined: list[list[int]] = []
    for c0, c1, ya, yb, li in raw:
        w = c1 - c0
        too_wide = (len(raw) >= S.min_chars_for_stats and w > S.wide_blob_ratio * med_w) or w > S.max_char_aspect * line_h[li]
        if not too_wide:
            refined.append([c0, c1, ya, yb, li]); continue
        band = binary[ya:yb, c0:c1].copy()
        hl = headline_rows(band)
        if hl is not None:
            band[hl[0]: hl[1]] = 0
            work_vis[ya + hl[0]: ya + hl[1], c0:c1] = 0
        parts = _runs((band > 0).sum(axis=0) > 0, min_gap=S.min_col_gap)
        parts = [(c0 + p0, c0 + p1) for p0, p1 in parts if p1 - p0 >= 0.3 * min(med_w, line_h[li])]
        if len(parts) >= 2:
            # widen the pieces so the whole segment (incl. headline) is covered
            cuts = [c0] + [(parts[k][1] + parts[k + 1][0]) // 2 for k in range(len(parts) - 1)] + [c1]
            refined.extend([cuts[k], cuts[k + 1], ya, yb, li] for k in range(len(cuts) - 1))
            continue
        cut = _valley_split((band > 0).sum(axis=0), 0, w)
        if cut is not None:
            refined.append([c0, c0 + cut, ya, yb, li]); refined.append([c0 + cut, c1, ya, yb, li])
        else:
            refined.append([c0, c1, ya, yb, li])
    if keep_steps:
        steps["headline_removed"] = work_vis

    # 4) Merge fragments. Handwritten characters are often physically broken
    #    (a faint headline section, a detached stroke or dot), so a piece much
    #    narrower or lighter than a typical character is joined to its NEARER
    #    neighbour on the same line, repeatedly, until no fragment is left.
    merged = sorted(refined, key=lambda r: (r[4], r[0]))
    while len(merged) >= 2:
        widths = np.array([r[1] - r[0] for r in merged], float)
        inks = np.array([ink(r) for r in merged], float)
        mw, mi = float(np.median(widths)), float(np.median(inks))
        best = None
        for k, r in enumerate(merged):
            if not (widths[k] < S.fragment_width_ratio * mw or inks[k] < S.fragment_ink_ratio * mi):
                continue
            gaps = [(merged[j][0] - r[1] if j > k else r[0] - merged[j][1], j)
                    for j in (k - 1, k + 1) if 0 <= j < len(merged) and merged[j][4] == r[4]]
            if not gaps:
                continue
            gap, j = min(gaps)
            new_w = max(r[1], merged[j][1]) - min(r[0], merged[j][0])
            # Never create an over-wide box (that is what step 3 splits), unless
            # the piece is trivially light (e.g. a stray headline tail).
            fits = new_w <= S.wide_blob_ratio * mw or inks[k] < 0.12 * mi
            if gap <= S.fragment_max_gap_ratio * mw and fits and (best is None or gap < best[0]):
                best = (gap, k, j)
        if best is None:
            break
        _, k, j = best
        a, b = sorted((k, j))
        merged[a] = [min(merged[a][0], merged[b][0]), max(merged[a][1], merged[b][1]),
                     merged[a][2], merged[a][3], merged[a][4]]
        del merged[b]

    boxes: list[CharBox] = []
    crops: list[np.ndarray] = []
    heights = []
    for c0, c1, ya, yb, li in merged:
        sub = binary[ya:yb, c0:c1]
        rows = np.flatnonzero(sub.any(axis=1))
        if rows.size == 0:
            continue
        y0, y1 = ya + rows[0], ya + rows[-1] + 1
        heights.append(y1 - y0)
        boxes.append(CharBox(int(c0), int(y0), int(c1 - c0), int(y1 - y0), li, 0))
        crops.append(_to_crop(binary[y0:y1, c0:c1]))

    med_w = float(np.median([b.w for b in boxes])) if boxes else 0.0
    med_h = float(np.median(heights)) if heights else 0.0
    many = len(boxes) >= S.min_chars_for_stats
    for b in boxes:
        line_w = max(1, int((binary[lines[b.line][0]: lines[b.line][1]] > 0).any(axis=0).sum()))
        if b.w > S.connected_line_ratio * line_w and b.w > 2.2 * b.h:
            b.flags.append("connected")
        if many and b.w > S.wide_blob_ratio * med_w:
            b.flags.append("wide")
        if many and b.w < 0.35 * med_w:
            b.flags.append("narrow")
        if many and b.h > S.tall_blob_ratio * med_h:
            b.flags.append("tall")
        b.confidence = max(0.05, 1.0 - 0.45 * ("connected" in b.flags) - 0.35 * ("wide" in b.flags)
                           - 0.3 * ("narrow" in b.flags) - 0.2 * ("tall" in b.flags))
    for k, b in enumerate(boxes):
        b.order = k + 1

    flagged = sum(bool(b.flags) for b in boxes)
    if any("connected" in b.flags for b in boxes):
        warnings.append("Some characters are joined by continuous cursive strokes. They could not be separated "
                        "reliably, so the affected predictions are probably wrong.")
    elif flagged:
        warnings.append(f"{flagged} of {len(boxes)} detected characters look suspicious (merged, fragmented or "
                        "oversized). Check the highlighted boxes.")
    return SegmentationResult(gray, binary, boxes, crops, lines, warnings, skew, scale, steps)


def draw_boxes(result: SegmentationResult, labels: list[str] | None = None,
               colours: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    """RGB image of ``result.display`` with numbered boxes."""
    img = cv2.cvtColor(result.display, cv2.COLOR_GRAY2RGB)
    th = max(2, int(round(img.shape[1] / 500)))
    fs = max(0.5, img.shape[1] / 1400)
    for k, b in enumerate(result.boxes):
        col = colours[k] if colours else ((200, 60, 60) if b.flags else (31, 58, 95))
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), col, th)
        tag = str(b.order) + (f" {labels[k]}" if labels else "")
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th - 1))
        ty = max(tht + 4, b.y - 4)
        cv2.rectangle(img, (b.x, ty - tht - 4), (b.x + tw + 4, ty + 2), col, -1)
        cv2.putText(img, tag, (b.x + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th - 1),
                    cv2.LINE_AA)
    return img
