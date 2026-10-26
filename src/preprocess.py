"""Stage 1 - Preprocessing of single-character images.

Objective 1 (synopsis): cleans each image with Otsu thresholding (idea from Anam & Gupta, 2015)
before the handcrafted features are computed.

Pipeline for one image:
    read (unicode-safe, alpha flattened onto white)
    -> grayscale
    -> median blur (salt-and-pepper noise)
    -> blank-image check (no contrast -> rejected)
    -> Otsu binarisation with automatic polarity detection
    -> scan-frame removal (lines hugging the image edge)
    -> interior scan-grid line removal (thin vertical lines spanning the cell)
    -> speck removal (tiny connected components)
    -> crop to the ink bounding box
    -> moment-based slant correction (only when the estimate is trustworthy)
    -> pad to a square with a margin, area-resize to IMAGE_SIZE, re-binarise

POLARITY CONVENTION: the output is always uint8 with ink = 255 (white) and
background = 0 (black). Polarity is detected per image: ink is the minority side
of the Otsu threshold, with the image border as a tie-break when the two sides
are nearly balanced. Every downstream module (feature
extractors, segmentation, the app) assumes this convention.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src import config
from src.utils import ModiError, get_logger

log = get_logger("preprocess")

P = config.PREPROCESS
INK = 255
PARAMS_FILE = config.PATHS.data_processed / "_preprocess_params.json"


@dataclass
class PreprocessResult:
    """Final image plus every intermediate step (for figures and the app)."""

    final: np.ndarray                                  # (S, S) uint8, ink = 255
    steps: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_image(path: str | Path) -> np.ndarray:
    """Read an image as uint8 grayscale.

    Uses ``np.fromfile`` + ``cv2.imdecode`` so Windows paths with non-ASCII
    characters (e.g. Devanagari folder names) work. Transparent pixels are
    composited onto white.

    Raises:
        ModiError: if the file is missing or cannot be decoded.
    """
    path = Path(path)
    if not path.is_file():
        raise ModiError(f"Image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise ModiError(f"Could not decode image (corrupt or unsupported format): {path.name}")
    return to_gray(img)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert any uint8/uint16 BGR(A)/gray array to uint8 grayscale."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)
        img = rgb.astype(np.uint8)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def write_image(path: Path, img: np.ndarray) -> None:
    """Write a PNG, unicode-path safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(P.output_ext, img)
    if not ok:
        raise ModiError(f"Could not encode image for {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------
def denoise(gray: np.ndarray) -> np.ndarray:
    """Median blur to suppress salt-and-pepper scan noise."""
    k = P.median_kernel
    return cv2.medianBlur(gray, k) if k and k > 1 else gray


def has_contrast(gray: np.ndarray) -> bool:
    """False for blank images (paper texture only, no glyph).

    Compares robust dark/bright extremes with the median grey level; a real
    glyph always produces a clear excursion on one side.
    """
    lo, med, hi = np.percentile(gray, [P.contrast_percentile, 50, 100 - P.contrast_percentile])
    return max(med - lo, hi - med) >= P.min_contrast


def binarize(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Otsu threshold mapped to the ink=255 convention.

    Polarity: ink covers far less area than background (1-10 % in this
    dataset), so the minority side of the threshold is ink. Only when the two
    sides are nearly balanced (40-60 %) is the image border consulted
    (border pixels are normally background). The border is NOT the primary
    rule because scans often carry a faint frame line around the edge.
    """
    t, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_frac = float((bright > 0).mean())
    if bright_frac >= 0.6:
        dark_ink, rule = True, "minority"
    elif bright_frac <= 0.4:
        dark_ink, rule = False, "minority"
    else:
        border = np.concatenate([bright[0], bright[-1], bright[:, 0], bright[:, -1]])
        dark_ink, rule = float((border > 0).mean()) >= 0.5, "border"
    binary = cv2.bitwise_not(bright) if dark_ink else bright
    return binary, {"otsu_threshold": float(t), "dark_ink_on_light": dark_ink, "polarity_rule": rule}


def remove_border_artifacts(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove scan-frame lines that hug the image edge.

    About 10 % of the dataset images carry a fragment of the scanning frame
    (solid or dashed) along an edge. Left in, it stretches the bounding box
    so the real glyph ends up small and off-centre, and adds a fake stroke.

    Two steps:
      1. blank the outermost ``border_clear_ratio`` of the image;
      2. remove components that lie entirely inside the wider
         ``border_band_ratio`` band and are elongated along that edge.
    A headline (shirorekha) is never removed: it is attached to the glyph or
    lies well inside the image, so it is not confined to the band.
    """
    h, w = binary.shape
    out = binary.copy()
    c = int(round(P.border_clear_ratio * min(h, w)))
    if c > 0 and min(h, w) > 4 * c:
        out[:c], out[-c:], out[:, :c], out[:, -c:] = 0, 0, 0, 0
    band = int(round(P.border_band_ratio * min(h, w)))
    removed = 0
    if band > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        drop = np.zeros(n, dtype=bool)
        for i in range(1, n):
            x, y, cw, ch = stats[i, :4]
            horiz = (y + ch <= band or y >= h - band) and cw >= P.border_line_aspect * ch
            vert = (x + cw <= band or x >= w - band) and ch >= P.border_line_aspect * cw
            drop[i] = horiz or vert
        removed = int(drop.sum())
        if removed:
            out[drop[labels]] = 0
    return out, removed


def remove_grid_lines(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove vertical scan-grid lines that sit inside the image.

    About 3 % of dataset images contain a thin vertical line from the scanning
    grid that is NOT within the edge band (often on the right, solid or dashed).
    A thin vertical piece that is separate from the glyph is removed only if
    its column (all thin pieces sharing roughly the same x) covers most of the
    image height AND reaches beyond the glyph body: grid lines run the full
    height of the cell, while real detached strokes (e.g. in 'ai', 'am') stay
    within the character. Horizontal lines are never touched here: isolated
    horizontal strokes are usually genuine headlines (e.g. over Modi digits).
    """
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary, 0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mt, mb = stats[main, cv2.CC_STAT_TOP], stats[main, cv2.CC_STAT_TOP] + stats[main, cv2.CC_STAT_HEIGHT]
    thin = [i for i in range(1, n) if i != main
            and stats[i, cv2.CC_STAT_WIDTH] <= P.grid_line_max_width_ratio * w
            and stats[i, cv2.CC_STAT_HEIGHT] >= 3 * stats[i, cv2.CC_STAT_WIDTH]]
    clusters: list[list[int]] = []
    for i in sorted(thin, key=lambda k: stats[k, cv2.CC_STAT_LEFT]):
        cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
        if clusters:
            j = clusters[-1][-1]
            if abs(cx - (stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH] / 2)) <= P.grid_line_cluster_tol * w:
                clusters[-1].append(i)
                continue
        clusters.append([i])
    drop = np.zeros(n, dtype=bool)
    for cl in clusters:
        coverage = sum(int(stats[i, cv2.CC_STAT_HEIGHT]) for i in cl) / h
        top = min(stats[i, cv2.CC_STAT_TOP] for i in cl)
        bottom = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in cl)
        sticks_out = (mt - top) + (bottom - mb) >= P.grid_line_overhang_ratio * h
        if coverage >= P.grid_line_min_coverage and sticks_out:
            drop[cl] = True
    if not drop.any():
        return binary, 0
    out = binary.copy()
    out[drop[labels]] = 0
    return out, int(drop.sum())


def remove_specks(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove connected components smaller than a size relative to the image.

    Small but real components (dots, matras) usually exceed the threshold;
    the ratio is deliberately conservative.
    """
    if P.morph_open_kernel and P.morph_open_kernel > 1:
        kernel = np.ones((P.morph_open_kernel, P.morph_open_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    min_area = max(P.min_component_area_floor, int(P.min_component_area_ratio * binary.size))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    removed = int((~keep[1:]).sum())
    return np.where(keep[labels], INK, 0).astype(np.uint8), removed


def crop_to_ink(binary: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of the ink pixels."""
    ys, xs = np.nonzero(binary)
    return binary[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def estimate_slant(binary: np.ndarray) -> float | None:
    """Slant angle (degrees) from second-order central moments, or None when
    the moments are degenerate (e.g. a single dot or a flat stroke)."""
    m = cv2.moments(binary, binaryImage=True)
    if m["mu02"] < 1e-2 or m["m00"] < P.min_ink_pixels:
        return None
    return math.degrees(math.atan(m["mu11"] / m["mu02"]))


def deskew(binary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Shear-correct handwriting slant, but only when it is safe.

    Skipped if: disabled, slant estimate degenerate, |slant| below
    ``min_deskew_angle`` (not worth it) or above ``max_deskew_angle``
    (estimate unreliable for this glyph), or if the result changes the ink
    area by more than ``max_deskew_ink_change`` (strokes damaged).
    """
    info: dict[str, Any] = {"deskew_applied": False, "slant_deg": None, "deskew_skip_reason": None}
    if not P.deskew:
        info["deskew_skip_reason"] = "disabled"
        return binary, info
    angle = estimate_slant(binary)
    info["slant_deg"] = None if angle is None else round(angle, 2)
    if angle is None:
        info["deskew_skip_reason"] = "degenerate moments"
        return binary, info
    if abs(angle) < P.min_deskew_angle:
        info["deskew_skip_reason"] = "slant negligible"
        return binary, info
    if abs(angle) > P.max_deskew_angle:
        info["deskew_skip_reason"] = "slant estimate too large (unreliable)"
        return binary, info

    h, w = binary.shape
    shear = math.tan(math.radians(angle))
    pad = int(math.ceil(abs(shear) * h)) + 2
    canvas = cv2.copyMakeBorder(binary, 2, 2, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ch, cw = canvas.shape
    # x' = x - shear * (y - cy)  -> removes the slant around the vertical centre
    M = np.float32([[1, -shear, shear * ch / 2], [0, 1, 0]])
    sheared = cv2.warpAffine(canvas, M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
    before, after = int((binary > 0).sum()), int((sheared > 0).sum())
    if before == 0 or abs(after - before) / before > P.max_deskew_ink_change or after == 0:
        info["deskew_skip_reason"] = "ink area changed too much"
        return binary, info
    info["deskew_applied"] = True
    return crop_to_ink(sheared), info


def normalize_size(binary: np.ndarray, size: int = P.image_size) -> np.ndarray:
    """Pad to a centred square with a margin, area-resize, re-binarise."""
    h, w = binary.shape
    side = int(math.ceil(max(h, w) * (1 + 2 * P.padding_ratio)))
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0: y0 + h, x0: x0 + w] = binary
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    resized = cv2.resize(canvas, (size, size), interpolation=interp)
    return np.where(resized >= P.resize_threshold, INK, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_array(img: np.ndarray, keep_steps: bool = False, grid_lines: bool = True) -> PreprocessResult:
    """Preprocess an in-memory image (any polarity, gray or colour).

    Args:
        grid_lines: remove interior scan-grid lines (dataset images). The app
            passes False for segmentation crops, which are tight around the
            glyph and never contain grid lines.

    Raises:
        ModiError: if no ink survives (blank image).
    """
    gray = to_gray(img)
    den = denoise(gray)
    if not has_contrast(den):
        raise ModiError("Blank image (no contrast between ink and paper).")
    binary, info = binarize(den)
    deframed, n_frame = remove_border_artifacts(binary)
    n_grid = 0
    if grid_lines:
        deframed, n_grid = remove_grid_lines(deframed)
    clean, removed = remove_specks(deframed)
    info["specks_removed"] = removed
    info["frame_fragments_removed"] = n_frame
    info["grid_line_pieces_removed"] = n_grid
    if int((clean > 0).sum()) < P.min_ink_pixels:
        raise ModiError("No ink found in the image (blank or too faint after thresholding).")
    cropped = crop_to_ink(clean)
    desk, dinfo = deskew(cropped)
    info.update(dinfo)
    final = normalize_size(desk)
    if not final.any():
        raise ModiError("Character vanished after resizing (strokes too thin for the target size).")
    steps = {"original": gray, "denoised": den, "binary": binary, "cleaned": clean,
             "cropped": cropped, "deskewed": desk, "final": final} if keep_steps else {}
    return PreprocessResult(final=final, steps=steps, info=info)


def preprocess_file(path: str | Path, keep_steps: bool = False) -> PreprocessResult:
    """Read and preprocess one image file."""
    return preprocess_array(read_image(path), keep_steps=keep_steps)


# ---------------------------------------------------------------------------
# Batch processing with caching
# ---------------------------------------------------------------------------
def params_hash() -> str:
    """Hash of the preprocessing parameters; a change invalidates the cache."""
    blob = json.dumps(asdict(P), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def processed_path(rel_path: str, label: str) -> Path:
    """Destination of a processed image: data/processed/<label>/<stem>_<hash>.png.

    A short hash of the source path keeps names unique even when two raw
    sub-folders contain files with the same name.
    """
    stem = Path(rel_path).stem
    tag = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:6]
    safe_label = label.replace("/", "__")
    return config.PATHS.data_processed / safe_label / f"{stem}_{tag}{P.output_ext}"


def process_one(src: str, dst: str, rel_path: str) -> tuple[str, str | None, dict[str, Any] | None]:
    """Batch worker: preprocess ``src`` and write ``dst``.

    Paths are passed explicitly (not recomputed from config) so worker
    processes never depend on module state of the parent process.

    Returns:
        (rel_path, error message or None, info dict or None)
    """
    try:
        res = preprocess_file(src)
        write_image(Path(dst), res.final)
        return rel_path, None, {"deskew_applied": res.info["deskew_applied"],
                                "dark_ink_on_light": res.info["dark_ink_on_light"],
                                "polarity_rule": res.info["polarity_rule"]}
    except ModiError as exc:
        return rel_path, str(exc), None
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
        return rel_path, f"{type(exc).__name__}: {exc}", None


def cache_is_valid() -> bool:
    """True if processed images were produced with the current parameters."""
    try:
        return json.loads(PARAMS_FILE.read_text())["hash"] == params_hash()
    except (OSError, ValueError, KeyError):
        return False


def mark_cache_valid() -> None:
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps({"hash": params_hash(), "params": asdict(P)}, indent=2, default=str))
