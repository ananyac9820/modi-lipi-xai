"""Histogram of Oriented Gradients: stroke direction and edge layout.

Objective 1 (synopsis): HOG is one of the three handcrafted features used instead of raw pixels.

HOG divides the glyph into cells, records in each cell how much edge energy
points in each direction, and normalises over overlapping blocks. It is the
"where are the strokes and which way do they run" descriptor. It is NOT
rotation invariant, which is why it is fused with Zernike moments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hog

from src import config

H = config.HOG
NAME = "hog"


def _blocks(image_size: int) -> tuple[int, int]:
    """Number of block positions (rows, cols) for a square image."""
    cells_r = image_size // H.pixels_per_cell[0]
    cells_c = image_size // H.pixels_per_cell[1]
    return cells_r - H.cells_per_block[0] + 1, cells_c - H.cells_per_block[1] + 1


def n_features(image_size: int = config.PREPROCESS.image_size) -> int:
    """Length of the HOG vector."""
    br, bc = _blocks(image_size)
    return br * bc * H.cells_per_block[0] * H.cells_per_block[1] * H.orientations


def prepare(img: np.ndarray) -> np.ndarray:
    """Binary ink=255 image -> smoothed float image in [0, 1].

    A binary image has only step edges; a light blur gives the gradient
    operator a slope to measure, so orientations are estimated more stably.
    """
    x = img.astype(np.float32) / 255.0
    return gaussian_filter(x, H.smooth_sigma) if H.smooth_sigma > 0 else x


def extract(img: np.ndarray) -> np.ndarray:
    """HOG vector for one preprocessed image (uint8, ink = 255)."""
    return hog(prepare(img), orientations=H.orientations, pixels_per_cell=H.pixels_per_cell,
               cells_per_block=H.cells_per_block, block_norm=H.block_norm,
               feature_vector=True).astype(np.float32)


def visualize(img: np.ndarray) -> np.ndarray:
    """HOG glyph image (for the feature notebook and the app's pipeline view)."""
    _, vis = hog(prepare(img), orientations=H.orientations, pixels_per_cell=H.pixels_per_cell,
                 cells_per_block=H.cells_per_block, block_norm=H.block_norm, visualize=True)
    return vis


def feature_meta(image_size: int = config.PREPROCESS.image_size) -> list[dict[str, Any]]:
    """One record per feature, in vector order.

    skimage flattens as (block_row, block_col, cell_row, cell_col, orientation).
    ``row`` / ``col`` give the cell's position as a fraction of the image
    (0 = top/left, 1 = bottom/right) so features can be pooled into regions.
    """
    br, bc = _blocks(image_size)
    cr, cc = H.cells_per_block
    cells_r = image_size // H.pixels_per_cell[0]
    cells_c = image_size // H.pixels_per_cell[1]
    meta = []
    for r in range(br):
        for c in range(bc):
            for i in range(cr):
                for j in range(cc):
                    for o in range(H.orientations):
                        angle = 180.0 * o / H.orientations
                        meta.append({"family": NAME, "name": f"hog_r{r + i}_c{c + j}_b{r}{c}_o{o}",
                                     "row": (r + i + 0.5) / cells_r, "col": (c + j + 0.5) / cells_c,
                                     "orientation_deg": angle})
    return meta
