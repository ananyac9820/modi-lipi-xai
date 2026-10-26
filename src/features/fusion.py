"""Descriptor fusion, single-descriptor extraction and feature caching.

Objective 1 (synopsis): joins HOG, Zernike and LBP into one 2,261-value feature vector, and builds
the single-feature and leave-one-out sets used in the ablation.

Feature sets:
    "hog", "zernike", "lbp"  - one descriptor family alone (for the ablation)
    "fused"                  - all three concatenated in ``config.DESCRIPTORS`` order
    "hog+lbp" etc.           - any two families joined with "+" (leave-one-out ablation:
                               fused minus this pair shows what the dropped family adds)

Extraction results are cached per descriptor in ``data/features/`` and keyed
by (image list, preprocessing parameters, descriptor parameters), so a change
to one descriptor's settings only re-extracts that descriptor.

``FamilyBalancer`` implements family-balanced fusion: without it, HOG's 1,764
features outvote Zernike's 49 by sheer count in PCA and in the RBF kernel.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from src import config, preprocess
from src.features import hog_features, lbp_features, zernike_features
from src.utils import ModiError, get_logger

log = get_logger("features")

EXTRACTORS = {"hog": hog_features, "zernike": zernike_features, "lbp": lbp_features}
_CONFIGS = {"hog": config.HOG, "zernike": config.ZERNIKE, "lbp": config.LBP}


# ---------------------------------------------------------------------------
# Layout of the feature vector
# ---------------------------------------------------------------------------
def families(feature_set: str) -> tuple[str, ...]:
    """Descriptor families making up a feature set."""
    if feature_set == "fused":
        return config.DESCRIPTORS
    if feature_set in EXTRACTORS:
        return (feature_set,)
    parts = tuple(feature_set.split("+"))
    if len(parts) > 1 and all(p in EXTRACTORS for p in parts) and len(set(parts)) == len(parts):
        return tuple(f for f in config.DESCRIPTORS if f in parts)      # canonical order
    raise ModiError(f"Unknown feature set '{feature_set}'. Choose from {config.FEATURE_SETS}.")


def n_features(family: str) -> int:
    """Vector length of one descriptor family."""
    return EXTRACTORS[family].n_features()


def slices(feature_set: str = "fused") -> dict[str, slice]:
    """Column range of every family inside a feature set's vector."""
    out, start = {}, 0
    for fam in families(feature_set):
        n = n_features(fam)
        out[fam] = slice(start, start + n)
        start += n
    return out


def feature_meta(feature_set: str = "fused") -> list[dict[str, Any]]:
    """Per-feature metadata (family, name, position/order...) in vector order."""
    meta: list[dict[str, Any]] = []
    for fam in families(feature_set):
        meta.extend(EXTRACTORS[fam].feature_meta())
    return meta


def feature_names(feature_set: str = "fused") -> list[str]:
    """Feature names in vector order."""
    return [m["name"] for m in feature_meta(feature_set)]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_one(img: np.ndarray, feature_set: str = "fused") -> np.ndarray:
    """Feature vector for one preprocessed image (uint8, ink = 255)."""
    return np.concatenate([EXTRACTORS[f].extract(img) for f in families(feature_set)])


def extract_parts(img: np.ndarray) -> dict[str, np.ndarray]:
    """Each family's vector separately (used by the app's pipeline view)."""
    return {f: EXTRACTORS[f].extract(img) for f in config.DESCRIPTORS}


def _extract_chunk(family: str, images: np.ndarray) -> np.ndarray:
    fn = EXTRACTORS[family].extract
    return np.stack([fn(im) for im in images])


def extract_batch(images: np.ndarray, family: str, n_jobs: int = config.FUSION.n_jobs,
                  chunk: int = 500) -> np.ndarray:
    """Extract one descriptor family for a stack of images, in parallel."""
    if len(images) == 0:
        return np.empty((0, n_features(family)), np.float32)
    parts = Parallel(n_jobs=n_jobs)(delayed(_extract_chunk)(family, images[i: i + chunk])
                                    for i in range(0, len(images), chunk))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_key(index: pd.DataFrame, family: str) -> str:
    blob = json.dumps({"paths": hashlib.sha1("\n".join(index["processed_path"]).encode()).hexdigest(),
                       "pre": preprocess.params_hash(), "cfg": asdict(_CONFIGS[family]),
                       "size": config.PREPROCESS.image_size}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def cache_path(index: pd.DataFrame, family: str):
    """File that holds the cached matrix of one family for this image set."""
    return config.PATHS.data_features / f"{family}_{_cache_key(index, family)}.npy"


def load_or_extract(index: pd.DataFrame, images: np.ndarray | None, family: str,
                    force: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the (n_images, n_features) matrix of one family, from cache if valid.

    Args:
        index: processed index (row order defines the matrix row order).
        images: uint8 image stack aligned with ``index``; may be None when the
            cache is expected to exist.
        family: "hog" | "zernike" | "lbp".
        force: ignore the cache.

    Returns:
        (matrix, info) where info has timing and whether the cache was used.
    """
    path = cache_path(index, family)
    if path.exists() and not force:
        X = np.load(path)
        if X.shape == (len(index), n_features(family)):
            return X, {"family": family, "cached": True, "n_features": X.shape[1], "seconds": 0.0}
        log.warning("Cached %s features have the wrong shape %s; re-extracting.", family, X.shape)
    if images is None:
        raise ModiError(f"No cached '{family}' features for this image set. "
                        "Run: python scripts/run_feature_extraction.py")
    t = time.perf_counter()
    X = extract_batch(images, family)
    secs = time.perf_counter() - t
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).any(axis=1).sum())
        log.warning("%s: %d image(s) produced non-finite features; replaced with 0.", family, bad)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    config.PATHS.data_features.mkdir(parents=True, exist_ok=True)
    for old in config.PATHS.data_features.glob(f"{family}_*.npy"):
        old.unlink()                      # one cache generation per family
    np.save(path, X)
    return X, {"family": family, "cached": False, "n_features": X.shape[1], "seconds": round(secs, 2),
               "ms_per_image": round(1000 * secs / max(len(images), 1), 3)}


def load_feature_set(index: pd.DataFrame, feature_set: str = "fused",
                     images: np.ndarray | None = None) -> np.ndarray:
    """Matrix for a feature set, built from the per-family caches."""
    return np.hstack([load_or_extract(index, images, fam)[0] for fam in families(feature_set)])


# ---------------------------------------------------------------------------
# Family-balanced fusion
# ---------------------------------------------------------------------------
class FamilyBalancer(BaseEstimator, TransformerMixin):
    """Scale each descriptor family by 1 / sqrt(n_features in that family).

    Applied after per-feature standardisation, this gives every family the
    same total variance, so the fused representation is not dominated by
    whichever descriptor happens to produce the longest vector. With a single
    family (or ``enabled=False``) it is the identity.

    Args:
        feature_set: which feature set the incoming columns belong to.
        enabled: turn balancing off for plain concatenation.
    """

    def __init__(self, feature_set: str = "fused", enabled: bool = config.FUSION.balance_families) -> None:
        self.feature_set = feature_set
        self.enabled = enabled

    def fit(self, X: np.ndarray, y: Any = None) -> "FamilyBalancer":
        sl = slices(self.feature_set)
        expected = sum(s.stop - s.start for s in sl.values())
        if X.shape[1] != expected:
            raise ModiError(f"FamilyBalancer expected {expected} columns for '{self.feature_set}', got {X.shape[1]}.")
        w = np.ones(X.shape[1], dtype=np.float32)
        if self.enabled and len(sl) > 1:
            for s in sl.values():
                w[s] = 1.0 / np.sqrt(s.stop - s.start)
            w *= np.sqrt(X.shape[1] / len(sl))      # keep the overall scale comparable
        self.weights_ = w
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.weights_

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(feature_names(self.feature_set), dtype=object)
