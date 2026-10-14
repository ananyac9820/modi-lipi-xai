"""Local Binary Patterns: local micro-structure at stroke ends and junctions.

For every pixel, LBP compares it with a ring of neighbours and records the
resulting pattern: flat region, edge, corner, line end, spot. Histograms of
these patterns are collected over a grid of image cells, at two ring sizes.
Where HOG says "a stroke runs this way here", LBP says "there is a corner / a
stroke end / a crossing here", which is what separates look-alike characters.

None of the surveyed Modi papers uses LBP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern

from src import config

L = config.LBP
NAME = "lbp"



def _pattern_kind(frac: float) -> str:
    """Approximate plain-language meaning of a rotation-invariant uniform code,
    from the fraction of the ring that is at least as bright as the centre.
    These labels are a reading aid, not an exact geometric claim."""
    if frac <= 0.06:
        return "isolated point"
    if frac <= 0.30:
        return "stroke tip"
    if frac <= 0.45:
        return "convex corner"
    if frac <= 0.56:
        return "straight edge"
    if frac <= 0.90:
        return "concave corner"
    return "flat region"


def _bins(p: int) -> int:
    """Histogram bins for one scale: P + 1 uniform patterns + 1 non-uniform."""
    if L.method != "uniform":
        raise ValueError("Only the 'uniform' LBP method is supported (bin count depends on it).")
    return p + 2


def n_features() -> int:
    """Length of the LBP vector: cells x sum of bins over scales."""
    return L.grid[0] * L.grid[1] * sum(_bins(p) for p in L.points)


def prepare(img: np.ndarray) -> np.ndarray:
    """Binary ink=255 image -> blurred uint8 image.

    LBP compares grey levels; on a purely binary image almost every
    comparison is a tie. A light blur turns stroke borders into slopes so the
    codes describe the local shape of the stroke boundary.
    """
    x = img.astype(np.float32)
    if L.smooth_sigma > 0:
        x = gaussian_filter(x, L.smooth_sigma)
    return np.clip(x, 0, 255).astype(np.uint8)


def codes(img: np.ndarray, scale: int = 0) -> np.ndarray:
    """LBP code image for one scale (used for visualisation)."""
    return local_binary_pattern(prepare(img), L.points[scale], L.radii[scale], method=L.method)


def extract(img: np.ndarray) -> np.ndarray:
    """LBP vector: per-cell, L1-normalised pattern histograms at every scale."""
    x = prepare(img)
    h, w = x.shape
    rows, cols = L.grid
    r_edges = np.linspace(0, h, rows + 1).astype(int)
    c_edges = np.linspace(0, w, cols + 1).astype(int)
    feats: list[np.ndarray] = []
    for p, rad in zip(L.points, L.radii):
        code = local_binary_pattern(x, p, rad, method=L.method).astype(np.int32)
        nb = _bins(p)
        for i in range(rows):
            for j in range(cols):
                cell = code[r_edges[i]: r_edges[i + 1], c_edges[j]: c_edges[j + 1]]
                hist = np.bincount(cell.ravel(), minlength=nb)[:nb].astype(np.float32)
                feats.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(feats)


def feature_meta() -> list[dict[str, Any]]:
    """One record per feature, in vector order (scale, cell row, cell col, bin)."""
    rows, cols = L.grid
    meta = []
    for s, (p, rad) in enumerate(zip(L.points, L.radii)):
        for i in range(rows):
            for j in range(cols):
                for b in range(_bins(p)):
                    kind = "irregular pattern (junction / crossing)" if b == p + 1 else _pattern_kind(b / p)
                    meta.append({"family": NAME, "name": f"lbp_s{s}_r{i}_c{j}_p{b}", "scale": s, "radius": rad,
                                 "row": (i + 0.5) / rows, "col": (j + 0.5) / cols, "pattern": b, "pattern_kind": kind})
    return meta
