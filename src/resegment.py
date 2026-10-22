"""Recognition-guided segmentation of touching / joined characters.

Projection-based segmentation (src/segmentation.py) splits characters at
blank columns, so characters that TOUCH (common in cursive Modi) end up in one
over-wide box. For such a box this module:

1. proposes candidate cut columns at the valleys of the ink profile (with the
   headline / shirorekha ignored, since it joins every character of a word);
2. scores every way of cutting the box into pieces with the trained classifier
   (each piece's top probability) plus a weak prior on character width;
3. keeps the reading with the best total score (dynamic programming), so the
   classifier itself decides where one character ends and the next begins.

``apply`` works on whole lines: pieces may also span several projection boxes,
so stray fragments (detached headline tails, strokes) are merged back as well
as touching characters being split. Every original box is always one of the
hypotheses, so correct boxes are kept unless a better reading exists. Many
Modi characters are wider than they are tall (up to ~2.5x), so the width
prior is on log(width / typical width) and is deliberately weak.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src import config, segmentation
from src.segmentation import CharBox, SegmentationResult, headline_rows

R = config.SEGMENT


@dataclass
class Split:
    """Result of re-segmenting one box."""

    cuts: list[int]                 # absolute x positions of the chosen cuts (empty = keep as one)
    pieces: list[tuple[int, int]]   # (x0, x1) of each piece
    probs: list[float]              # classifier top probability of each piece
    candidates: list[int]           # all candidate cuts that were considered


def expected_width(result: SegmentationResult, line: int) -> float:
    """Typical character width on a line: median width of unflagged boxes,
    else the height of the line's ink (Modi characters are roughly square)."""
    ok = [b.w for b in result.boxes if b.line == line and not b.flags]
    if len(ok) >= 2:
        return float(np.median(ok))
    ya, yb = result.lines[line]
    rows = np.flatnonzero((result.binary[ya:yb] > 0).any(axis=1))
    return float(max(8, (rows[-1] - rows[0] + 1) * 0.9)) if rows.size else 32.0


def candidate_cuts(result: SegmentationResult, box: CharBox, exp_w: float) -> list[int]:
    """Valley columns of the ink profile inside a box, headline ignored."""
    ya, yb = result.lines[box.line]
    region = result.binary[ya:yb, box.x: box.x + box.w].copy()
    hl = headline_rows(region)
    if hl is not None:
        region[hl[0]: hl[1]] = 0
    prof = (region > 0).sum(axis=0).astype(float)
    if prof.size < 6 or prof.max() == 0:
        return []
    smooth = np.convolve(prof, np.ones(3) / 3, mode="same")
    peak = smooth.max()
    min_gap = max(3, int(R.reseg_min_piece_ratio * exp_w))
    cands = []
    for x in range(min_gap, prof.size - min_gap):
        if smooth[x] <= R.reseg_valley_ratio * peak and smooth[x] <= smooth[x - 1] and smooth[x] <= smooth[x + 1]:
            cands.append(x)
    # keep the deepest valley within each min_gap neighbourhood
    cands.sort(key=lambda x: smooth[x])
    chosen: list[int] = []
    for x in cands:
        if all(abs(x - c) >= min_gap for c in chosen):
            chosen.append(x)
    return sorted(box.x + c for c in chosen)[: R.reseg_max_candidates]


def piece_crop(result: SegmentationResult, line: int, x0: int, x1: int) -> np.ndarray | None:
    """Crop (dark ink on white) for columns [x0, x1) of a line, trimmed to ink rows."""
    ya, yb = result.lines[line]
    sub = result.binary[ya:yb, x0:x1]
    rows = np.flatnonzero(sub.any(axis=1))
    if rows.size == 0:
        return None
    return segmentation._to_crop(sub[rows[0]: rows[-1] + 1])


def best_split(result: SegmentationResult, box: CharBox,
               classify: Callable[[list[np.ndarray]], np.ndarray]) -> Split:
    """Choose cuts for one box by dynamic programming over classifier scores.

    Args:
        classify: maps a list of crops to the top-class probability of each.
    """
    exp_w = expected_width(result, box.line)
    cands = candidate_cuts(result, box, exp_w)
    points = [box.x] + cands + [box.x + box.w]
    n = len(points)
    # every piece between two points that has a plausible width (the whole box is always allowed)
    edges = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            w = points[j] - points[i]
            whole = i == 0 and j == n - 1
            if whole or R.reseg_min_piece_ratio * exp_w <= w <= R.reseg_max_piece_ratio * exp_w:
                edges.append((i, j))
    crops, keep = [], []
    for i, j in edges:
        c = piece_crop(result, box.line, points[i], points[j])
        if c is not None:
            crops.append(c); keep.append((i, j))
    probs = dict(zip(keep, classify(crops))) if crops else {}

    def score(i: int, j: int) -> float:
        p = max(probs.get((i, j), 1e-6), 1e-6)
        ratio = max(points[j] - points[i], 1) / exp_w
        prior = -0.5 * (math.log(ratio) / R.reseg_width_sigma) ** 2
        return math.log(p) + R.reseg_prior_weight * prior

    best = [-math.inf] * n
    back = [-1] * n
    best[0] = 0.0
    for j in range(1, n):
        for i in range(j):
            if (i, j) in probs and best[i] > -math.inf:
                s = best[i] + score(i, j)
                if s > best[j]:
                    best[j], back[j] = s, i
    if back[n - 1] < 0:                                   # nothing scored: keep the box whole
        return Split([], [(box.x, box.x + box.w)], [float(probs.get((0, n - 1), 0.0))], cands)
    path, j = [], n - 1
    while j > 0:
        path.append(j); j = back[j]
    path = [0] + path[::-1]
    pieces = [(points[a], points[b]) for a, b in zip(path, path[1:])]
    return Split([points[k] for k in path[1:-1]], pieces,
                 [float(probs[(a, b)]) for a, b in zip(path, path[1:])], cands)


def _ink_extent(result: SegmentationResult, line: int, x0: int, x1: int) -> tuple[int, int, int, int] | None:
    """Tight ink box (x0, y0, x1, y1) of columns [x0, x1) in a line band."""
    ya, yb = result.lines[line]
    sub = result.binary[ya:yb, x0:x1]
    cols = np.flatnonzero(sub.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(sub.any(axis=1))
    return x0 + int(cols[0]), ya + int(rows[0]), x0 + int(cols[-1]) + 1, ya + int(rows[-1]) + 1


def resegment_line(result: SegmentationResult, line: int,
                   classify: Callable[[list[np.ndarray]], np.ndarray]) -> list[tuple[int, int, bool]]:
    """Best reading of one line as a sequence of pieces.

    Possible character edges are the gaps between the projection boxes plus
    valley cuts inside boxes that are wider than usual. A piece may span
    several original boxes (merging fragments such as detached headline tails
    or strokes) or part of one box (splitting touching characters). Each piece
    is scored by the classifier's confidence plus a weak log-width prior, and
    dynamic programming picks the best sequence.

    Returns:
        list of (x0, x1, is_original_box) for the chosen pieces.
    """
    boxes = sorted((b for b in result.boxes if b.line == line), key=lambda b: b.x)
    if not boxes:
        return []
    widths = [b.w for b in boxes]
    exp_w = float(np.median(widths)) if len(boxes) >= 3 else expected_width(result, line)
    # separators: line start, middle of every gap between boxes, line end
    points = {boxes[0].x, boxes[-1].x + boxes[-1].w}
    for a, b in zip(boxes, boxes[1:]):
        points.add((a.x + a.w + b.x) // 2 if b.x > a.x + a.w else b.x)
    for b in boxes:
        if b.w > R.reseg_split_width_ratio * exp_w or {"wide", "connected"} & set(b.flags):
            points.update(candidate_cuts(result, b, exp_w))
    pts = sorted(points)
    original = {(b.x, b.x + b.w) for b in boxes}

    def snap(x0: int, x1: int) -> tuple[int, int] | None:
        ext = _ink_extent(result, line, x0, x1)
        return (ext[0], ext[2]) if ext else None

    edges, crops = [], []
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            span = pts[j] - pts[i]
            if span > R.reseg_max_piece_ratio * exp_w and j > i + 1:
                break
            ink = snap(pts[i], pts[j])
            if ink is None:
                continue
            w = ink[1] - ink[0]
            is_orig = any(abs(ink[0] - o0) <= 2 and abs(ink[1] - o1) <= 2 for o0, o1 in original)
            if not is_orig and w < R.reseg_min_piece_ratio * exp_w:
                continue
            c = piece_crop(result, line, pts[i], pts[j])
            if c is not None:
                edges.append((i, j, w, is_orig)); crops.append(c)
    if not edges:
        return [(b.x, b.x + b.w, True) for b in boxes]
    probs = classify(crops)
    score = {}
    for (i, j, w, is_orig), p in zip(edges, probs):
        prior = -0.5 * (math.log(max(w, 1) / exp_w) / R.reseg_width_sigma) ** 2
        score[(i, j)] = (math.log(max(float(p), 1e-6)) + R.reseg_prior_weight * prior - R.reseg_piece_penalty,
                         is_orig, float(p))

    n = len(pts)
    best = [-math.inf] * n
    back = [-1] * n
    best[0] = 0.0
    for j in range(1, n):
        for i in range(j):
            if best[i] == -math.inf:
                continue
            if (i, j) in score:
                cand = best[i] + score[(i, j)][0]
            elif snap(pts[i], pts[j]) is None:
                cand = best[i]                       # an empty gap: no character, no cost
            else:
                continue
            if cand > best[j]:
                best[j], back[j] = cand, i
    if back[n - 1] < 0:
        return [(b.x, b.x + b.w, True) for b in boxes]
    path, j = [], n - 1
    while j > 0:
        path.append(j); j = back[j]
    path = [0] + path[::-1]
    out = []
    for a, b in zip(path, path[1:]):
        if (a, b) in score:
            ink = snap(pts[a], pts[b])
            out.append((ink[0], ink[1], score[(a, b)][1]))
    return out


def apply(result: SegmentationResult, classify: Callable[[list[np.ndarray]], np.ndarray]) -> int:
    """Re-read every line with recognition-guided segmentation, in place.

    Returns the number of boxes that changed (split, merged or re-cut).
    """
    old = {(b.line, b.x, b.x + b.w): (b, c) for b, c in zip(result.boxes, result.crops)}
    new_boxes: list[CharBox] = []
    new_crops: list[np.ndarray] = []
    changed = 0
    suspicious = {"wide", "connected", "narrow"}
    for line in sorted({b.line for b in result.boxes}):
        line_boxes = [(b, c) for b, c in zip(result.boxes, result.crops) if b.line == line]
        # Clean lines are left alone: on training-split words, re-reading lines that projection
        # segmentation already got right slightly increased the error rate.
        if not any(suspicious & set(b.flags) for b, _ in line_boxes):
            for b, c in line_boxes:
                new_boxes.append(b); new_crops.append(c)
            continue
        for x0, x1, is_orig in resegment_line(result, line, classify):
            ext = _ink_extent(result, line, x0, x1)
            if ext is None:
                continue
            match = next((v for (ln, o0, o1), v in old.items()
                          if ln == line and abs(o0 - x0) <= 2 and abs(o1 - x1) <= 2), None)
            if match is not None:
                new_boxes.append(match[0]); new_crops.append(match[1])
                continue
            changed += 1
            ya, yb = result.lines[line]
            sub = result.binary[ext[1]:ext[3], x0:x1]
            new_boxes.append(CharBox(int(x0), int(ext[1]), int(x1 - x0), int(ext[3] - ext[1]), line, 0,
                                     ["resegmented"], R.reseg_confidence))
            new_crops.append(segmentation._to_crop(sub))
    for k, b in enumerate(new_boxes):
        b.order = k + 1
    removed = len(result.boxes) - sum(1 for b in new_boxes if "resegmented" not in b.flags)
    result.boxes, result.crops = new_boxes, new_crops
    if changed or removed:
        remaining = sum(bool({"wide", "connected"} & set(b.flags)) for b in new_boxes)
        result.warnings = [w for w in result.warnings if "cursive" not in w and "suspicious" not in w]
        result.warnings.append(
            f"Recognition-guided segmentation re-cut {changed} character box(es) (marked as re-split): joined "
            "characters were split and stray fragments merged. These cuts are the model's best guess."
            + (f" {remaining} box(es) still look merged." if remaining else ""))
    return changed
