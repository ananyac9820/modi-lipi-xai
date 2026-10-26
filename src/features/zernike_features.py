"""Zernike moments: rotation-invariant global shape.

Objective 1 (synopsis): Zernike moments are one of the three handcrafted features
(chosen after Kulkarni et al., 2014).

The glyph is projected onto Zernike polynomials defined on a disc centred on
the glyph's centre of mass. The magnitudes of the projections do not change
when the glyph is rotated, so they describe overall shape (loops, symmetry,
mass distribution) independently of writing angle. Low orders capture coarse
shape, high orders capture fine detail.
"""

from __future__ import annotations

import math
from typing import Any

import mahotas
import numpy as np

from src import config

Z = config.ZERNIKE
NAME = "zernike"


def radius(image_size: int = config.PREPROCESS.image_size) -> int:
    """Disc radius. Default covers the whole square (half the diagonal)."""
    return Z.radius or int(math.ceil(image_size * math.sqrt(2) / 2))


def orders() -> list[tuple[int, int]]:
    """(order n, repetition m) pairs in mahotas' output order: n ascending,
    then m ascending over the values where (n - m) is even."""
    return [(n, l) for n in range(Z.degree + 1) for l in range(n + 1) if (n - l) % 2 == 0]


def n_features() -> int:
    """Number of Zernike magnitudes for the configured degree."""
    return len(orders())


def extract(img: np.ndarray) -> np.ndarray:
    """Zernike magnitude vector for one preprocessed image (uint8, ink = 255)."""
    binary = (img > 0).astype(np.uint8)
    return mahotas.features.zernike_moments(binary, radius(img.shape[0]), degree=Z.degree).astype(np.float32)


def feature_meta() -> list[dict[str, Any]]:
    """One record per moment, in vector order."""
    return [{"family": NAME, "name": f"zernike_n{n}_m{m}", "order": n, "repetition": m} for n, m in orders()]
