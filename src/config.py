"""Central configuration for Modi Lipi XAI.

Every tunable parameter of the pipeline lives in this file. No other module
should contain "magic numbers": if you want to change an image size, a HOG cell
size, a model hyperparameter or the random seed, change it here and re-run the
relevant scripts.

The values are grouped into small frozen dataclasses so they are read-only at
runtime and easy to print/log for reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
"""Single global random seed used by numpy, sklearn, XGBoost and SHAP."""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the project."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_features: Path = PROJECT_ROOT / "data" / "features"
    models: Path = PROJECT_ROOT / "models"
    figures: Path = PROJECT_ROOT / "results" / "figures"
    metrics: Path = PROJECT_ROOT / "results" / "metrics"
    shap: Path = PROJECT_ROOT / "results" / "shap"
    app_assets: Path = PROJECT_ROOT / "app" / "assets"

    def ensure(self) -> None:
        """Create every output directory if it does not exist yet."""
        for value in asdict(self).values():
            Path(value).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """Where the data comes from and which files count as images."""

    kaggle_slug: str = "msd6013/modi-hdc-historical-handwritten-modi-script"
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    # Classes with fewer images than this are reported and excluded from
    # training (stratified splitting needs at least a few samples per class).
    min_images_per_class: int = 5


# ---------------------------------------------------------------------------
# Stage 1 - Preprocessing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters for turning a raw character image into a clean binary glyph."""

    image_size: int = 64                 # final square size (pixels)
    median_kernel: int = 3               # median blur kernel for salt/pepper noise
    morph_open_kernel: int = 2           # opening kernel to remove specks (0 = off)
    min_component_area: int = 8          # connected components smaller than this are noise
    deskew: bool = True                  # moment-based deskew
    max_deskew_angle: float = 15.0       # degrees; larger estimated skews are ignored
    padding_ratio: float = 0.12          # margin kept around the glyph before resizing
    # Otsu threshold on the grayscale image; ink is always mapped to 1 (white
    # foreground on black background) regardless of the scan's polarity.


# ---------------------------------------------------------------------------
# Stage 2 - Segmentation (multi-character images)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters for splitting a word/line image into single characters."""

    max_working_width: int = 1600         # downscale very large uploads for speed
    line_min_gap_ratio: float = 0.01      # min empty-row run (fraction of height) separating lines
    line_min_height_ratio: float = 0.02   # lines thinner than this fraction of height are noise
    projection_noise_ratio: float = 0.02  # a row/col is "empty" if ink < ratio * max profile
    min_char_area_ratio: float = 0.0008   # components smaller than this fraction of image area are dropped
    merge_overlap_ratio: float = 0.5      # horizontally overlapping components are merged (matras, dots)
    wide_blob_ratio: float = 1.6          # width > ratio * median width -> likely merged characters
    tall_blob_ratio: float = 2.2          # height > ratio * median height -> suspicious
    min_chars_for_stats: int = 3          # need this many boxes before median-based flags are reliable
    crop_padding: int = 4                 # pixels of context added around each crop


# ---------------------------------------------------------------------------
# Stage 3 - Feature extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HOGConfig:
    """Histogram of Oriented Gradients (skimage.feature.hog)."""

    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"


@dataclass(frozen=True)
class ZernikeConfig:
    """Zernike moments (mahotas.features.zernike_moments)."""

    degree: int = 12                      # max polynomial order -> 49 moments at degree 12
    radius: int | None = None             # None = image_size // 2


@dataclass(frozen=True)
class LBPConfig:
    """Local Binary Patterns (skimage.feature.local_binary_pattern)."""

    # Two scales capture both fine stroke junctions and slightly wider texture.
    points: tuple[int, ...] = (8, 16)
    radii: tuple[float, ...] = (1.0, 2.0)
    method: str = "uniform"               # rotation-invariant uniform -> P + 2 bins
    grid: tuple[int, int] = (4, 4)        # spatial histogram grid (rows, cols)


DESCRIPTORS: tuple[str, ...] = ("hog", "zernike", "lbp")
"""Descriptor families, in the order they are concatenated by fusion."""

FEATURE_SETS: tuple[str, ...] = ("hog", "zernike", "lbp", "fused")
"""Feature sets evaluated in the ablation study."""


# ---------------------------------------------------------------------------
# Stage 4 - Dimensionality reduction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReduceConfig:
    """PCA / LDA settings. ``method`` selects which one the main models use."""

    method: str = "pca"                   # "pca" | "lda" | "none"
    pca_variance: float = 0.95            # keep components explaining this much variance
    pca_max_components: int = 300         # hard cap for speed
    lda_components: int | None = None     # None = n_classes - 1
    lda_solver: str = "eigen"
    lda_shrinkage: str | float | None = "auto"  # regularised LDA; robust when features > samples/class


# ---------------------------------------------------------------------------
# Stage 5 - Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    """Train/test split and cross-validation."""

    test_size: float = 0.2
    cv_folds: int = 5


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the three base learners and the stacking ensemble."""

    rf: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
    })
    svm: dict[str, Any] = field(default_factory=lambda: {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
    })
    xgb: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    })
    # Meta-learner on top of the three base learners' out-of-fold probabilities.
    stack_final_C: float = 1.0
    stack_final_max_iter: int = 2000
    stack_cv: int = 5
    stack_passthrough: bool = False


# ---------------------------------------------------------------------------
# Stage 7 - Explainability
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExplainConfig:
    """SHAP settings. Features are explained in grouped form (spatial regions /
    moment orders) so explanations stay readable and Kernel SHAP stays fast."""

    background_size: int = 50             # background samples summarised for the explainer
    global_sample_size: int = 150         # test samples used for global importance
    kernel_nsamples: int = 400            # model evaluations per local explanation
    hog_region_grid: tuple[int, int] = (4, 4)   # HOG blocks pooled into a 4x4 grid of regions
    top_k_features: int = 10              # features shown per local explanation


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Thresholds used by the Streamlit app."""

    high_confidence: float = 0.70         # >= green
    low_confidence: float = 0.40          # <  red, in between amber
    top_k: int = 3
    max_upload_mb: int = 10


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT: str = "%H:%M:%S"
LOG_LEVEL: str = "INFO"


# ---------------------------------------------------------------------------
# Singletons used across the code base
# ---------------------------------------------------------------------------
PATHS = Paths()
DATASET = DatasetConfig()
PREPROCESS = PreprocessConfig()
SEGMENT = SegmentationConfig()
HOG = HOGConfig()
ZERNIKE = ZernikeConfig()
LBP = LBPConfig()
REDUCE = ReduceConfig()
SPLIT = SplitConfig()
MODELS = ModelConfig()
EXPLAIN = ExplainConfig()
APP = AppConfig()


def as_dict() -> dict[str, Any]:
    """Return the whole configuration as a JSON-serialisable dict (for logging
    alongside results, so every metric file records the settings that made it)."""

    def _clean(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj.relative_to(PROJECT_ROOT)) if obj.is_relative_to(PROJECT_ROOT) else str(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    return _clean({
        "seed": SEED,
        "dataset": asdict(DATASET),
        "preprocess": asdict(PREPROCESS),
        "segmentation": asdict(SEGMENT),
        "hog": asdict(HOG),
        "zernike": asdict(ZERNIKE),
        "lbp": asdict(LBP),
        "reduce": asdict(REDUCE),
        "split": asdict(SPLIT),
        "models": asdict(MODELS),
        "explain": asdict(EXPLAIN),
        "app": asdict(APP),
    })
