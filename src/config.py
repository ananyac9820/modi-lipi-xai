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
    kaggle_url: str = "https://www.kaggle.com/datasets/msd6013/modi-hdc-historical-handwritten-modi-script"
    # The Kaggle bundle holds two datasets: MODI_HChar (isolated characters,
    # organised category/class/image) and MODI_HHDoc (full document pages).
    # Only folders under a path segment containing this hint are character data.
    character_root_hint: str = "hchar"
    # Folder names (case-insensitive substrings) that are never character classes.
    exclude_dir_keywords: tuple[str, ...] = ("hhdoc", "document", "__macosx")
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    min_image_side: int = 8              # images smaller than this (px) are skipped as junk
    # Classes with fewer images than this are reported and excluded from
    # training (stratified splitting needs at least a few samples per class).
    min_images_per_class: int = 5
    # Folder-name keywords (case-insensitive) mapping a raw category folder to a
    # canonical category. Unmatched folders keep category "other".
    category_keywords: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "digit": ("digit", "numeral", "number"),
        "vowel": ("vowel", "swar"),
        "consonant": ("consonant", "vyanjan"),
    })
    # ---- Claims we compare against. NEVER used as data, only reported. ----
    # What the surveyed benchmark papers report.
    published_claim_classes: int = 46
    published_claim_per_class: int = 90
    # What the Kaggle uploader's description states (unverified).
    uploader_claim_classes: int = 57
    uploader_claim_total: int = 575_920
    uploader_claim_by_category: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "digit": {"classes": 10, "images": 101_100},
        "vowel": {"classes": 12, "images": 121_320},
        "consonant": {"classes": 35, "images": 353_500},
    })
    # Download housekeeping
    min_free_space_gb: float = 16.0      # ~10 GB zip + ~5 GB extracted characters
    document_samples: int = 20           # full-page document images extracted for segmentation tests


# ---------------------------------------------------------------------------
# Sampling: main experiment + learning curve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingConfig:
    """How images are drawn from the (large) dataset.

    Every class gets ONE deterministic random ordering of its files, seeded by
    ``SEED`` and the class label (so a class's ordering never depends on which
    other classes exist). All subsets are prefixes of that ordering:

        [ test (test_per_class) | train ... (up to max(curve_train_sizes)) ]

    * The test set is fixed and shared by every experiment.
    * Learning-curve training sets are nested (90 ⊂ 250 ⊂ 500 ⊂ ...).
    * The main experiment is simply the ``main_train_per_class`` point.

    SHORT-CLASS POLICY (see data_loader.plan_splits):
    * Test size is decided ONCE per class from its image count N, so the test
      set is identical at every curve point:
        N >= test_per_class + min(curve_train_sizes) -> test = test_per_class
        otherwise -> test = max(1, floor(scaled_test_fraction * N))  ("scaled")
    * A class with fewer training images than a curve point n stays IN that
      point with all the training images it has (the class set never changes
      between points). The point is flagged "incomplete", the short classes
      are logged and recorded, and accuracy is also reported on the complete
      classes only, so the effect is visible.

    72 approximates the surveyed papers' effective training size (90 per
    class total minus a 20 % test split). Only the SAMPLE SIZE is comparable:
    dataset, class inventory and handwriting all differ.
    """

    test_per_class: int = 200
    scaled_test_fraction: float = 0.2
    # Classes whose image count falls in this band sit near the full/scaled
    # test-size threshold, where the rule has a cliff (271 imgs -> 217 train,
    # 272 imgs -> 72 train). Reported so we can decide whether to switch to a
    # smooth rule: min(200, max(20, int(0.2 * n))).
    boundary_band: tuple[int, int] = (150, 400)
    pool_reserve: int = 30               # spare images per class to absorb preprocessing failures
    main_train_per_class: int = 500
    curve_train_sizes: tuple[int, ...] = (72, 90, 250, 500, 1000, 2000)

    @property
    def pool_per_class(self) -> int:
        """Images per class that must be preprocessed to serve every experiment
        (plus a small reserve so a few preprocessing failures cannot leave a
        class short at the largest curve point)."""
        return (self.test_per_class + max(max(self.curve_train_sizes), self.main_train_per_class)
                + self.pool_reserve)


# ---------------------------------------------------------------------------
# Stage 1 - Preprocessing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters for turning a raw character image into a clean binary glyph."""

    # POLARITY CONVENTION (used everywhere downstream):
    #   ink / foreground = 255 (white), background = 0 (black).
    # Every input is mapped to this convention whatever the scan's polarity.
    image_size: int = 64                 # final square size (pixels)
    # Blank-image check: robust dark/bright excursion from the median grey level.
    min_contrast: float = 40.0           # grey levels; below this the image has no glyph
    contrast_percentile: float = 0.2     # percentile used for the robust extremes
    # Scan-frame removal (about 10 % of dataset images carry a frame fragment).
    border_clear_ratio: float = 0.02     # outermost fraction of the image blanked outright
    border_band_ratio: float = 0.07      # edge band in which elongated components are frame lines
    border_line_aspect: float = 3.0      # length / thickness needed to count as a line
    # Interior scan-grid lines (thin vertical lines inside the image, ~3 % of images).
    grid_line_max_width_ratio: float = 0.06   # thinner than this fraction of the width
    grid_line_cluster_tol: float = 0.03       # pieces within this x-distance form one (dashed) line
    grid_line_min_coverage: float = 0.75      # the line must cover this fraction of the image height
    grid_line_overhang_ratio: float = 0.15    # ...and extend this far beyond the glyph body
    median_kernel: int = 3               # median blur kernel for salt/pepper noise (0/1 = off)
    morph_open_kernel: int = 0           # opening erodes thin strokes, so off by default
    min_component_area_ratio: float = 0.0005  # components below this fraction of the image are specks
    min_component_area_floor: int = 4    # ...but never treat components >= this many px as specks only by ratio
    min_ink_pixels: int = 10             # fewer ink pixels than this -> image is blank/unusable
    # Moment-based slant correction. OFF by default: on an isolated character
    # the moment "slant" is confounded with the glyph's own diagonal strokes,
    # and in testing it sheared upright glyphs. Dataset crops are already
    # upright; page/line skew is handled in segmentation instead. Kept as a
    # switch so its effect can be measured on real data.
    deskew: bool = False
    min_deskew_angle: float = 2.0        # degrees; smaller slants are left alone
    max_deskew_angle: float = 20.0       # degrees; larger estimates are unreliable -> skip
    max_deskew_ink_change: float = 0.15  # reject deskew if it changes ink area by more than this
    padding_ratio: float = 0.10          # margin kept around the glyph before resizing
    resize_threshold: int = 96           # re-binarise after area-resize; < 128 protects thin strokes
    output_ext: str = ".png"             # lossless, so binary images stay binary
    n_jobs: int = -1                     # parallel workers for batch preprocessing


# ---------------------------------------------------------------------------
# Stage 2 - Segmentation (multi-character images)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters for splitting a word/line image into single characters."""

    max_working_width: int = 1600         # downscale very large uploads for speed
    single_char_max_aspect: float = 1.3   # auto mode: one line and width <= 1.3 x height -> read as one character
    line_min_gap_ratio: float = 0.01      # min empty-row run (fraction of height) separating lines
    line_min_height_ratio: float = 0.02   # lines thinner than this fraction of height are noise
    projection_noise_ratio: float = 0.02  # a row/col is "empty" if ink < ratio * max profile
    min_char_area_ratio: float = 0.0008   # components smaller than this fraction of image area are dropped
    merge_overlap_ratio: float = 0.5      # horizontally overlapping components are merged (matras, dots)
    wide_blob_ratio: float = 2.0          # width > ratio * median width -> likely merged characters
    # Single Modi characters reach ~2.5x their height (measured on the dataset), so only
    # boxes wider than this x line height are presumed to be several characters.
    max_char_aspect: float = 2.6
    tall_blob_ratio: float = 2.2          # height > ratio * median height -> suspicious
    min_chars_for_stats: int = 3          # need this many boxes before median-based flags are reliable
    crop_margin_ratio: float = 0.15       # white margin around each crop, like the dataset images
    # Page / line skew: projection-profile search over this range (degrees).
    max_skew_deg: float = 12.0
    skew_step_deg: float = 0.5
    min_skew_to_correct_deg: float = 0.7
    # Headline (shirorekha): a long horizontal stroke joining the characters of
    # a word. It is ignored while finding character boundaries, then kept in the crops.
    headline_min_coverage: float = 0.55   # headline row must span this fraction of the line's ink width
    headline_band_ratio: float = 0.6      # rows with >= this fraction of the peak count belong to the headline
    min_col_gap: int = 2                  # empty columns needed to separate two characters
    valley_split_ratio: float = 0.25      # split a wide blob at a column valley below this fraction of its peak
    connected_line_ratio: float = 0.6     # one blob wider than this fraction of the line -> cursive, unreliable
    # Fragment merging: a piece narrower OR lighter than these fractions of the
    # median character joins its nearer neighbour if the gap is small enough.
    fragment_width_ratio: float = 0.5
    fragment_ink_ratio: float = 0.3
    fragment_max_gap_ratio: float = 0.6
    # Recognition-guided re-segmentation of wide / connected boxes (src/resegment.py)
    reseg_valley_ratio: float = 0.45      # a cut candidate is a profile valley below this fraction of the peak
    reseg_min_piece_ratio: float = 0.35   # pieces narrower than this x expected width are not allowed
    reseg_max_piece_ratio: float = 2.8    # ...nor wider than this (Modi characters reach ~2.5x)
    reseg_split_width_ratio: float = 1.8  # boxes wider than this x typical width get candidate cuts (tuned on train words)
    reseg_max_candidates: int = 14
    reseg_width_sigma: float = 0.6        # width prior: Gaussian on log(width / typical width)
    reseg_prior_weight: float = 0.5
    # Per-character cost. Tuned on TRAINING-split words: any positive value made the line-level
    # search merge separate characters (error 42% -> 70-80%), so it is 0.
    reseg_piece_penalty: float = 0.0
    reseg_confidence: float = 0.6         # segmentation confidence given to re-split pieces


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
    smooth_sigma: float = 1.0             # Gaussian blur before gradients (binary edges are too abrupt)


@dataclass(frozen=True)
class ZernikeConfig:
    """Zernike moments (mahotas.features.zernike_moments)."""

    degree: int = 12                      # max polynomial order -> 49 moments at degree 12
    # None = half the image diagonal, so the unit disc covers the whole glyph
    # (image_size // 2 would discard the corners).
    radius: int | None = None


@dataclass(frozen=True)
class LBPConfig:
    """Local Binary Patterns (skimage.feature.local_binary_pattern)."""

    # Two scales capture both fine stroke junctions and slightly wider texture.
    points: tuple[int, ...] = (8, 16)
    radii: tuple[float, ...] = (1.0, 2.0)
    method: str = "uniform"               # rotation-invariant uniform -> P + 2 bins
    grid: tuple[int, int] = (4, 4)        # spatial histogram grid (rows, cols)
    smooth_sigma: float = 1.0             # LBP compares grey levels, so the binary image is blurred first


@dataclass(frozen=True)
class FusionConfig:
    """How the three descriptor families are combined."""

    # Family-balanced fusion: after standardising every feature, each family
    # is scaled by 1 / sqrt(its number of features) so HOG (1,764 features)
    # cannot drown out Zernike (49) and LBP (448) in PCA or in the RBF kernel.
    # Set False for plain concatenation (kept for the ablation study).
    balance_families: bool = True
    n_jobs: int = -1                      # parallel workers for extraction


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

    # "auto" = the reducer chosen by cross-validation on the TRAINING set
    # (scripts/run_training.py --only select -> results/metrics/reducer_selection.json).
    # The test set is never used to make this choice. Explicit "pca" | "lda" | "none" also work.
    method: str = "auto"
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
    cv_folds: int = 3                     # COMPUTE CONSTRAINT: each fold retrains the full stack


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
        # No Platt scaling (probability=True, deprecated in scikit-learn 1.9): it
        # would add an internal 5-fold cross-validation to every SVM fit. The
        # stacking meta-learner consumes the SVM's decision scores instead and
        # does the calibration. probability=False is the default, so it is not passed.
        "decision_function_shape": "ovr",
        "class_weight": "balanced",
        "cache_size": 1000,
    })
    # COMPUTE CONSTRAINT: XGBoost builds one tree per class per round (57 here)
    # and the stack trains it several times. The original 300 rounds x depth 5
    # took 3-4 min at the SMALLEST curve point and hours at the larger ones.
    # These settings were chosen for CPU time only: at 72 images/class they
    # matched the original's test accuracy (65.2 % vs 65.0 %) about 4x faster.
    # (Multi-output trees were also tried and were slower here.)
    xgb: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.25,
        "max_bin": 32,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "tree_method": "hist",
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    })
    # Meta-learner (standardise -> multinomial logistic regression) on the base
    # learners' out-of-fold outputs: RF and XGBoost probabilities, SVM decision scores.
    stack_final_C: float = 1.0
    stack_final_max_iter: int = 2000
    stack_cv: int = 3                     # COMPUTE CONSTRAINT: 3 internal folds (4 fits per base model, not 6)
    stack_passthrough: bool = False
    # 1 = base learners trained one after another. Running them in parallel
    # processes oversubscribed the CPU (RF and XGBoost are multithreaded) and stalled.
    stack_n_jobs: int = 1


# ---------------------------------------------------------------------------
# REFERENCE BASELINE ONLY - small CNN (never part of the pipeline or the app)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CNNBaselineConfig:
    """A plain, FIXED, UNTUNED CNN used only as the deep-learning reference
    curve. These values are textbook defaults chosen before seeing any
    results and are deliberately never tuned: tuning only one side of the
    comparison (in either direction) would make the comparison meaningless.

    Architecture: 3 x [conv3x3 -> BatchNorm -> ReLU -> maxpool2]
    with 32/64/128 filters, then dropout -> dense(256) -> ReLU -> dropout -> dense(C).
    Same input as the classical pipeline (preprocessed 64x64 binary images),
    no augmentation (the classical side gets none either).
    """

    conv_channels: tuple[int, ...] = (32, 64, 128)
    dense_units: int = 256
    dropout: float = 0.5
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 128
    # Safety cap only; early stopping (patience below) decides when training ends.
    # Was 40: in the first run that cap cut training short at small curve sizes while
    # validation loss was still falling (72/class: 0.86 -> 0.81 at epoch 39), which would
    # have handicapped the CNN exactly where the comparison matters. Raised to 300 so
    # the stopping rule, not the cap, ends training. Nothing else in the protocol changed.
    max_epochs: int = 300
    early_stopping_patience: int = 6      # epochs without val-loss improvement
    val_fraction: float = 0.1             # carved from the TRAINING set only, per class
    # After early stopping picks the epoch count, retrain on 100 % of the
    # training set for that many epochs, so the CNN sees exactly the same
    # images as the classical models (no 10 % handicap). Costs extra time,
    # which is reported.
    refit_on_full_train: bool = True
    num_threads: int = 0                  # 0 = all CPU cores


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
    """Thresholds used by the web app."""

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
SAMPLING = SamplingConfig()
PREPROCESS = PreprocessConfig()
SEGMENT = SegmentationConfig()
HOG = HOGConfig()
ZERNIKE = ZernikeConfig()
LBP = LBPConfig()
FUSION = FusionConfig()
REDUCE = ReduceConfig()
SPLIT = SplitConfig()
MODELS = ModelConfig()
CNN_BASELINE = CNNBaselineConfig()
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
        "sampling": {**asdict(SAMPLING), "pool_per_class": SAMPLING.pool_per_class},
        "preprocess": asdict(PREPROCESS),
        "segmentation": asdict(SEGMENT),
        "hog": asdict(HOG),
        "zernike": asdict(ZERNIKE),
        "lbp": asdict(LBP),
        "fusion": asdict(FUSION),
        "reduce": asdict(REDUCE),
        "split": asdict(SPLIT),
        "models": asdict(MODELS),
        "cnn_baseline": asdict(CNN_BASELINE),
        "explain": asdict(EXPLAIN),
        "app": asdict(APP),
    })
