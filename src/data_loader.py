"""Dataset discovery for Modi character images.

Classes are discovered from disk, never hard-coded: every *leaf* folder that
contains images is one class, and the folder name is the label. The parent
folder (e.g. ``consonants`` / ``vowels`` / ``digits`` in MODI-HChar) is kept as
the class *category*.

The scan is defensive about real-world mess:
  * nested wrapper folders (``MODI_HChar/MODI_HChar/...``)
  * mixed extensions and upper/lower-case suffixes
  * non-image, hidden and OS system files
  * empty, corrupt or unreadable images, and implausibly tiny images
  * images sitting in non-leaf folders (ambiguous label)
  * duplicate or case-only-different folder names
Every skipped file is recorded with a reason.

Results are cached in ``data/processed/manifest.csv``; a re-scan only re-reads
image headers for files whose size or modification time changed.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src import config
from src.utils import ModiError, get_logger, save_json, setup_matplotlib

log = get_logger("data_loader")

DS = config.DATASET
MANIFEST_PATH = config.PATHS.data_processed / "manifest.csv"
SUMMARY_PATH = config.PATHS.metrics / "dataset_summary.json"
DISTRIBUTION_FIG = config.PATHS.figures / "class_distribution.png"

SYSTEM_FILES = {"thumbs.db", "desktop.ini", ".ds_store", "icon\r"}
MAX_EXAMPLES_PER_REASON = 25


@dataclass
class SkippedFile:
    """A file that was ignored, and why."""

    path: str
    reason: str


@dataclass
class DatasetScan:
    """Result of scanning the raw data folder."""

    root: Path
    character_roots: list[Path]
    manifest: pd.DataFrame                  # one row per usable image
    skipped: list[SkippedFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded_classes: dict[str, int] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        """Sorted class labels that survived filtering."""
        return sorted(self.manifest["label"].unique().tolist())


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def canonical_category(raw: str) -> str:
    """Map a raw category folder name (e.g. 'Consonants', 'MODI digits') to
    'digit' / 'vowel' / 'consonant', or 'other' if it matches no keyword."""
    low = raw.lower()
    for canon, keywords in DS.category_keywords.items():
        if any(k in low for k in keywords):
            return canon
    return "other"


def _is_hidden(name: str) -> bool:
    return name.startswith(".") or name.lower() in SYSTEM_FILES


def _is_excluded_dir(name: str) -> bool:
    low = name.lower()
    return _is_hidden(name) or any(k in low for k in DS.exclude_dir_keywords)


def find_character_roots(root: Path) -> list[Path]:
    """Return the top-most folders whose name contains the character-data hint.

    Falls back to ``root`` itself when no such folder exists (e.g. a dataset
    unzipped directly as class folders).
    """
    hits: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d)]
        here = Path(dirpath)
        if here != root and DS.character_root_hint in here.name.lower():
            hits.append(here)
            dirnames[:] = []          # take the top-most match; its subtree is walked later
    return hits or [root]


def _walk_images(char_roots: Iterable[Path], root: Path) -> tuple[dict[Path, list[Path]], list[SkippedFile]]:
    """Group candidate image files by containing folder; record skipped files."""
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    skipped: list[SkippedFile] = []
    for char_root in char_roots:
        for dirpath, dirnames, filenames in os.walk(char_root):
            dirnames[:] = sorted(d for d in dirnames if not _is_excluded_dir(d))
            for name in sorted(filenames):
                p = Path(dirpath) / name
                rel = str(p.relative_to(root))
                if _is_hidden(name):
                    skipped.append(SkippedFile(rel, "hidden/system file"))
                elif p.suffix.lower() not in DS.image_extensions:
                    skipped.append(SkippedFile(rel, f"non-image file ({p.suffix.lower() or 'no extension'})"))
                else:
                    by_dir[Path(dirpath)].append(p)
    return by_dir, skipped


def _assign_labels(by_dir: dict[Path, list[Path]], char_roots: list[Path], root: Path
                   ) -> tuple[list[tuple[Path, str, str]], list[SkippedFile], list[str]]:
    """Turn image folders into (path, label, category) triples."""
    skipped: list[SkippedFile] = []
    warnings: list[str] = []
    image_dirs = set(by_dir)
    class_dirs: list[Path] = []
    for d in sorted(image_dirs):
        has_image_subdirs = any(o != d and o.is_relative_to(d) for o in image_dirs)
        if d in char_roots:
            reason = "image directly in dataset root (no class folder)"
        elif has_image_subdirs:
            reason = "image in a non-leaf folder (ambiguous label)"
        else:
            class_dirs.append(d)
            continue
        skipped.extend(SkippedFile(str(p.relative_to(root)), reason) for p in by_dir[d])

    # Detect duplicate folder names (same class split across folders, or the
    # same short name used in two categories).
    name_to_dirs: dict[str, list[Path]] = defaultdict(list)
    for d in class_dirs:
        name_to_dirs[d.name.strip()].append(d)
    lower_groups: dict[str, set[str]] = defaultdict(set)
    for name in name_to_dirs:
        lower_groups[name.lower()].add(name)
    for variants in lower_groups.values():
        if len(variants) > 1:
            warnings.append(
                f"Folder names differing only by case: {sorted(variants)}. Kept as SEPARATE classes "
                "(Modi transliterations can use case meaningfully); merge them by hand if they are the same character.")

    triples: list[tuple[Path, str, str]] = []
    for name, dirs in name_to_dirs.items():
        for d in dirs:
            category = d.parent.name.strip() if d.parent not in char_roots else ""
            label = name if len(dirs) == 1 else f"{category or d.parent.name}/{name}"
            if len(dirs) > 1:
                warnings.append(f"Folder name '{name}' appears in {len(dirs)} places; using qualified label '{label}'.")
            if name != d.name:
                warnings.append(f"Folder '{d.name}' has leading/trailing whitespace; label normalised to '{name}'.")
            triples.extend((p, label, category) for p in by_dir[d])
    return triples, skipped, warnings


def _read_header(path: Path) -> tuple[int, int, str] | str:
    """Return (width, height, mode) or an error reason string."""
    try:
        if path.stat().st_size == 0:
            return "empty file (0 bytes)"
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
        if min(w, h) < DS.min_image_side:
            return f"too small ({w}x{h} px)"
        return w, h, mode
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return f"corrupt/unreadable ({type(exc).__name__})"


def _load_cached_manifest() -> pd.DataFrame | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        return pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError):
        log.warning("Cached manifest unreadable; re-scanning from scratch.")
        return None


def scan_dataset(root: Path | None = None, rescan: bool = False, n_threads: int = 16) -> DatasetScan:
    """Discover classes and images under ``root`` (default ``data/raw``).

    Args:
        root: raw data directory.
        rescan: ignore the cached header information.
        n_threads: threads used to read image headers (I/O bound).

    Raises:
        ModiError: if the folder is missing or contains no usable images.
    """
    root = (root or config.PATHS.data_raw).resolve()
    if not root.exists() or not any(p for p in root.iterdir() if p.name != ".gitkeep"):
        raise ModiError(f"No dataset found in {root}. Run: python scripts/download_data.py")

    char_roots = find_character_roots(root)
    if char_roots == [root]:
        log.warning("No folder containing '%s' found; treating every leaf folder under %s as a class.",
                    DS.character_root_hint, root)
    log.info("Character data root(s): %s", ", ".join(str(c.relative_to(root)) or "." for c in char_roots))

    by_dir, skipped = _walk_images(char_roots, root)
    triples, skipped_lbl, warnings = _assign_labels(by_dir, char_roots, root)
    skipped += skipped_lbl
    log.info("Found %d candidate image files in %d folders", len(triples), len(by_dir))

    # Re-use cached headers for unchanged files.
    cache: dict[str, tuple[Any, ...]] = {}
    old = None if rescan else _load_cached_manifest()
    if old is not None and {"rel_path", "bytes", "mtime", "width", "height", "mode"} <= set(old.columns):
        cache = {r.rel_path: (r.bytes, r.mtime, r.width, r.height, r.mode) for r in old.itertuples()}

    def _info(item: tuple[Path, str, str]) -> dict[str, Any] | SkippedFile:
        path, label, category = item
        rel = str(path.relative_to(root))
        try:
            st = path.stat()
        except OSError as exc:
            return SkippedFile(rel, f"unreadable ({type(exc).__name__})")
        hit = cache.get(rel)
        if hit and hit[0] == st.st_size and abs(float(hit[1]) - st.st_mtime) < 1e-3:
            w, h, mode = int(hit[2]), int(hit[3]), str(hit[4])
        else:
            res = _read_header(path)
            if isinstance(res, str):
                return SkippedFile(rel, res)
            w, h, mode = res
        return {"rel_path": rel, "label": label, "category": canonical_category(category),
                "category_raw": category, "ext": path.suffix.lower(),
                "width": w, "height": h, "mode": mode, "bytes": st.st_size, "mtime": st.st_mtime}

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for i, res in enumerate(pool.map(_info, triples, chunksize=512), 1):
            if isinstance(res, SkippedFile):
                skipped.append(res)
            else:
                rows.append(res)
            if i % 50000 == 0:
                log.info("  read headers of %d / %d images", i, len(triples))

    manifest = pd.DataFrame(rows, columns=["rel_path", "label", "category", "category_raw", "ext", "width",
                                           "height", "mode", "bytes", "mtime"])
    if manifest.empty:
        raise ModiError(f"No readable images found under {root}. "
                        f"{len(skipped)} files were skipped - see the log above.")
    manifest = manifest.sort_values(["label", "rel_path"]).reset_index(drop=True)
    unmapped = sorted(set(manifest.loc[manifest["category"] == "other", "category_raw"]))
    if unmapped:
        warnings.append(f"Category folder(s) {unmapped} match no digit/vowel/consonant keyword -> category 'other'. "
                        "Add a keyword in config.DATASET.category_keywords if they are real categories.")

    # Drop classes too small to split/stratify.
    counts = manifest["label"].value_counts()
    small = counts[counts < DS.min_images_per_class]
    excluded = {str(k): int(v) for k, v in small.items()}
    if excluded:
        warnings.append(f"Excluded {len(excluded)} class(es) with < {DS.min_images_per_class} images: {excluded}")
        manifest = manifest[~manifest["label"].isin(excluded)].reset_index(drop=True)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    for w in warnings:
        log.warning(w)
    return DatasetScan(root, char_roots, manifest, skipped, warnings, excluded)


def load_manifest() -> pd.DataFrame:
    """Load the cached manifest written by :func:`scan_dataset`."""
    df = _load_cached_manifest()
    if df is None or df.empty:
        raise ModiError("No dataset manifest found. Run: python scripts/run_preprocessing.py")
    return df


# ---------------------------------------------------------------------------
# Deterministic stratified sampling
# ---------------------------------------------------------------------------
def _class_rng(label: str, seed: int) -> np.random.Generator:
    """RNG for one class, derived from the global seed and a *stable* hash of
    the label (Python's hash() is salted per process, so it is not used)."""
    digest = int(hashlib.sha1(label.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([seed, digest])


def with_class_order(manifest: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Add a ``class_rank`` column: each class's files in a seeded random order.

    Each class is shuffled independently with its own RNG, after sorting by
    path, so the order of a class depends only on (seed, label, its own files).
    Adding/removing another class, or a different filesystem walk order,
    cannot change it. Every subset in the project is a prefix of this order.

    If ``manifest`` already has ``class_rank`` (e.g. the processed index, a
    prefix of the full order), that order is KEPT and only compacted to
    0..n-1. Re-shuffling a subset would produce a different permutation and
    break the "every subset is a prefix" guarantee.
    """
    parts = []
    has_rank = "class_rank" in manifest.columns
    for label, g in manifest.groupby("label", sort=True):
        if has_rank:
            g = g.sort_values("class_rank").copy()
        else:
            g = g.sort_values("rel_path")
            perm = _class_rng(str(label), seed).permutation(len(g))
            g = g.iloc[perm].copy()
        g["class_rank"] = np.arange(len(g))
        parts.append(g)
    return pd.concat(parts).sort_values(["label", "class_rank"]).reset_index(drop=True)


def _counts_report(df: pd.DataFrame, requested: int, what: str) -> dict[str, Any]:
    """Log + return per-class counts actually used, flagging classes below request."""
    counts = df["label"].value_counts().sort_index()
    short = {str(k): int(v) for k, v in counts.items() if v < requested}
    log.info("%s: %d images, %d classes, per class min %d / max %d (requested %d)",
             what, len(df), counts.size, counts.min(), counts.max(), requested)
    if short:
        log.warning("%s: %d class(es) have FEWER than the requested %d and use all they have: %s",
                    what, len(short), requested, short)
    return {"requested_per_class": requested, "total": int(len(df)),
            "per_class": {str(k): int(v) for k, v in counts.items()}, "short_classes": short}


def cap_per_class(manifest: pd.DataFrame, n: int | None, seed: int = config.SEED
                  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The first ``n`` images of every class's seeded order (``None``/0 = all).

    Classes are capped independently: a class with fewer than ``n`` images
    keeps all of them and is reported, never padded or re-weighted.
    """
    ordered = with_class_order(manifest, seed)
    if not n:
        return ordered, _counts_report(ordered, int(ordered["label"].value_counts().max()), "No cap")
    capped = ordered[ordered["class_rank"] < n].reset_index(drop=True)
    return capped, _counts_report(capped, n, f"Cap {n}/class")


SHORT_CLASS_POLICY = (
    "Test size is fixed once per class: classes with >= test_per_class + min(curve sizes) images get "
    "test_per_class test images; smaller classes get floor(scaled_test_fraction * N) (min 1), marked "
    "'scaled'. A class with fewer training images than a curve point stays in that point with every "
    "training image it has, so the class set is identical at every point; the point is flagged "
    "incomplete and accuracy is additionally reported on complete classes only."
)


def plan_splits(manifest: pd.DataFrame,
                sizes: tuple[int, ...] = config.SAMPLING.curve_train_sizes,
                test_per_class: int = config.SAMPLING.test_per_class,
                quiet: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decide each class's test size and which curve points it can fill.

    Implements the short-class policy in ``config.SamplingConfig``.

    Returns:
        (plan, report) where ``plan`` has one row per class with columns
        label, category, n_images, test_n, test_policy, available_train,
        short_at (list of curve sizes the class cannot fill).
    """
    S = config.SAMPLING
    sizes = tuple(sorted(set(sizes) | {S.main_train_per_class}))
    # Threshold uses the configured curve, not ad-hoc extra sizes, so a class's
    # test size never depends on which split happens to be requested.
    threshold = test_per_class + min(S.curve_train_sizes)
    counts = manifest.groupby("label").size()
    cats = manifest.groupby("label")["category"].first() if "category" in manifest else None
    rows = []
    for label, n_img in counts.items():
        full = n_img >= threshold
        test_n = test_per_class if full else max(1, int(np.floor(S.scaled_test_fraction * n_img)))
        avail = int(n_img - test_n)
        rows.append({"label": str(label), "category": None if cats is None else cats[label],
                     "n_images": int(n_img), "test_n": int(test_n),
                     "test_policy": "full" if full else "scaled", "available_train": avail,
                     "short_at": [n for n in sizes if avail < n]})
    plan = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

    points = {}
    for n in sizes:
        short = plan[plan["available_train"] < n]
        points[str(n)] = {"complete": bool(short.empty),
                          "short_classes": {r.label: r.available_train for r in short.itertuples()}}
    scaled = plan[plan["test_policy"] == "scaled"]
    lo, hi = S.boundary_band
    band = plan[(plan["n_images"] >= lo) & (plan["n_images"] <= hi)]
    report = {
        "boundary_band": {"range_images": [lo, hi],
                          "n_classes": int(len(band)),
                          "classes": {r.label: r.n_images for r in band.itertuples()},
                          "note": "Classes near the full/scaled threshold. If any exist, switch to the "
                                  "smooth rule min(200, max(20, int(0.2 * n)))."},
        "policy": SHORT_CLASS_POLICY,
        "full_test_threshold_images": threshold,
        "test_per_class": test_per_class,
        "reduced_test_classes": {r.label: {"images": r.n_images, "test_n": r.test_n} for r in scaled.itertuples()},
        "points": points,
        "incomplete_points": [int(n) for n, p in points.items() if not p["complete"]],
    }
    if not quiet:
        log.info("Boundary band check: %d class(es) have %d-%d images%s", len(band), lo, hi,
                 f": {report['boundary_band']['classes']}" if len(band) else " (threshold cliff is theoretical)")
        if len(band):
            log.warning("Classes near the test-size threshold exist -> the smooth test-size rule should be adopted.")
        if not scaled.empty:
            log.warning("!" * 70)
            log.warning("SHORT CLASSES: %d class(es) have < %d images, so their test set is SCALED to %.0f%%: %s",
                        len(scaled), threshold, 100 * S.scaled_test_fraction, report["reduced_test_classes"])
            log.warning("!" * 70)
        for n, p in points.items():
            if not p["complete"]:
                log.warning("Curve point %s/class is INCOMPLETE: %d class(es) have fewer training images "
                            "and use all they have: %s", n, len(p["short_classes"]), p["short_classes"])
    return plan, report


def stratified_split(manifest: pd.DataFrame, train_per_class: int,
                     test_per_class: int = config.SAMPLING.test_per_class, seed: int = config.SEED,
                     quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fixed test set + training set of up to ``train_per_class`` images per class.

    Per class (in its seeded order): test = ranks [0, test_n); train = the
    next ``train_per_class``. ``test_n`` comes from :func:`plan_splits` and
    does not depend on ``train_per_class``, so the test set is identical for
    every training size and training sets for increasing sizes are nested.

    Returns:
        (train_df, test_df, report)
    """
    ordered = with_class_order(manifest, seed)
    plan, plan_report = plan_splits(ordered, test_per_class=test_per_class, quiet=True,
                                    sizes=tuple(sorted(set(config.SAMPLING.curve_train_sizes) | {train_per_class})))
    test_n = ordered["label"].map(plan.set_index("label")["test_n"])
    test = ordered[ordered["class_rank"] < test_n]
    train = ordered[(ordered["class_rank"] >= test_n) & (ordered["class_rank"] < test_n + train_per_class)]
    point = plan_report["points"][str(train_per_class)]
    report = {"train_per_class": train_per_class,
              "complete": point["complete"], "short_classes": point["short_classes"],
              "reduced_test_classes": plan_report["reduced_test_classes"],
              "train_counts": {str(k): int(v) for k, v in train["label"].value_counts().sort_index().items()},
              "test_counts": {str(k): int(v) for k, v in test["label"].value_counts().sort_index().items()},
              "n_train": int(len(train)), "n_test": int(len(test))}
    if not quiet:
        log.info("Split @%d/class: %d train, %d test images, %d classes%s", train_per_class, len(train),
                 len(test), plan.shape[0], "" if point["complete"] else
                 f"  [INCOMPLETE: {len(point['short_classes'])} short class(es)]")
        if not point["complete"]:
            log.warning("  short classes @%d: %s", train_per_class, point["short_classes"])
    no_train = sorted(set(test["label"]) - set(train["label"]))
    if no_train:
        log.warning("Classes with NO training images: %s", no_train)
    return train.reset_index(drop=True), test.reset_index(drop=True), report


def learning_curve_splits(manifest: pd.DataFrame,
                          sizes: tuple[int, ...] = config.SAMPLING.curve_train_sizes,
                          test_per_class: int = config.SAMPLING.test_per_class, seed: int = config.SEED
                          ) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[str, Any]]:
    """One shared test set and a nested training set per curve size.

    Returns:
        (test_df, {train_per_class: train_df}, report) - ``report`` holds the
        split plan plus per-point completeness.

    Raises:
        ModiError: if the test set were ever to differ between points
            (guards the core design invariant).
    """
    _, plan_report = plan_splits(with_class_order(manifest, seed), sizes, test_per_class)
    trains: dict[int, pd.DataFrame] = {}
    points: dict[str, Any] = {}
    test = pd.DataFrame()
    for n in sorted(sizes):
        train, test_n, rep = stratified_split(manifest, n, test_per_class, seed)
        if not test.empty and not test_n["rel_path"].equals(test["rel_path"]):
            raise ModiError("Internal error: test set differs between curve points.")
        test = test_n
        trains[n] = train
        points[str(n)] = rep
    return test, trains, {"plan": plan_report, "points": points}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _per_class_stats(counts: pd.Series) -> dict[str, Any]:
    arr = counts.to_numpy()
    return {"min": int(arr.min()), "max": int(arr.max()), "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)), "std": round(float(arr.std()), 2),
            "imbalance_ratio_max_over_min": round(float(arr.max() / arr.min()), 3)}


def category_breakdown(manifest: pd.DataFrame) -> dict[str, Any]:
    """Classes, images and images-per-class statistics for each category."""
    out: dict[str, Any] = {}
    for cat, g in manifest.groupby("category"):
        counts = g["label"].value_counts().sort_index()
        out[str(cat)] = {"classes": int(counts.size), "images": int(len(g)),
                         "raw_folder_names": sorted(set(g["category_raw"])),
                         "images_per_class": _per_class_stats(counts),
                         "labels": counts.index.tolist()}
    return out


def claims_vs_measured(manifest: pd.DataFrame, by_cat: dict[str, Any]) -> dict[str, Any]:
    """Compare what is on disk against the papers' and the uploader's claims.

    Claims are reported, never used as data. Differences are stated explicitly.
    """
    counts = manifest["label"].value_counts()
    n_classes, total = int(counts.size), int(len(manifest))
    notes: list[str] = []

    papers = {"claimed_classes": DS.published_claim_classes,
              "claimed_images_per_class": DS.published_claim_per_class,
              "measured_classes": n_classes,
              "measured_images_per_class_range": [int(counts.min()), int(counts.max())],
              "classes_match": n_classes == DS.published_claim_classes,
              "per_class_match": bool(counts.min() == counts.max() == DS.published_claim_per_class)}

    up_cat = {}
    for cat, claim in DS.uploader_claim_by_category.items():
        meas = by_cat.get(cat, {"classes": 0, "images": 0})
        up_cat[cat] = {"claimed_classes": claim["classes"], "measured_classes": meas["classes"],
                       "claimed_images": claim["images"], "measured_images": meas["images"],
                       "images_difference": meas["images"] - claim["images"]}
        if meas["classes"] != claim["classes"]:
            notes.append(f"{cat}: uploader claims {claim['classes']} classes, found {meas['classes']}.")
        if meas["images"] != claim["images"]:
            notes.append(f"{cat}: uploader claims {claim['images']:,} images, found {meas['images']:,} "
                         f"({meas['images'] - claim['images']:+,}).")
    for cat in sorted(set(by_cat) - set(DS.uploader_claim_by_category)):
        notes.append(f"Found category '{cat}' ({by_cat[cat]['classes']} classes) not in the uploader's description.")
    uploader = {"claimed_classes": DS.uploader_claim_classes, "measured_classes": n_classes,
                "claimed_total_images": DS.uploader_claim_total, "measured_total_images": total,
                "total_difference": total - DS.uploader_claim_total,
                "classes_match": n_classes == DS.uploader_claim_classes,
                "total_match": total == DS.uploader_claim_total, "by_category": up_cat}
    if n_classes != DS.uploader_claim_classes:
        notes.append(f"Uploader claims {DS.uploader_claim_classes} classes; found {n_classes}.")
    if total != DS.uploader_claim_total:
        notes.append(f"Uploader claims {DS.uploader_claim_total:,} images; found {total:,} "
                     f"({total - DS.uploader_claim_total:+,}).")

    # The 46-vs-57 puzzle: papers say 46, uploader says 57, and 12 vowels +
    # 35 consonants = 47. Report what is actually on disk.
    v, c = by_cat.get("vowel", {}).get("classes", 0), by_cat.get("consonant", {}).get("classes", 0)
    d = by_cat.get("digit", {}).get("classes", 0)
    class_count_note = (
        f"Measured: {v} vowel + {c} consonant = {v + c} letter classes, plus {d} digit classes "
        f"= {v + c + d} total (excluding any 'other'). Papers report {DS.published_claim_classes}; "
        f"the uploader's own breakdown (12 + 35 = 47 letters + 10 digits) implies 57. "
        f"Neither 46 nor 47 is guaranteed to be the same character inventory as this dataset."
    )
    notes.append(class_count_note)
    return {"surveyed_papers": papers, "kaggle_uploader": uploader, "class_count_note": class_count_note,
            "differences": notes,
            "policy": "All counts in this file are measured from disk. Claims are listed only for comparison."}


def summarize(scan: DatasetScan) -> dict[str, Any]:
    """Build the JSON-serialisable dataset summary."""
    m = scan.manifest
    counts = m["label"].value_counts().sort_index()
    cats = m.groupby("label")["category"].first()
    reasons = Counter(s.reason for s in scan.skipped)
    examples: dict[str, list[str]] = defaultdict(list)
    for s in scan.skipped:
        if len(examples[s.reason]) < MAX_EXAMPLES_PER_REASON:
            examples[s.reason].append(s.path)
    by_cat = category_breakdown(m)
    return {
        "dataset": DS.kaggle_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_root": str(scan.root),
        "character_roots": [str(c.relative_to(scan.root)) or "." for c in scan.character_roots],
        "total_images": int(len(m)),
        "num_classes": int(counts.size),
        "images_per_class": _per_class_stats(counts),
        "categories": by_cat,
        "classes": [{"label": k, "category": cats[k], "count": int(v)} for k, v in counts.items()],
        "image_width": {"min": int(m["width"].min()), "max": int(m["width"].max()), "median": float(m["width"].median())},
        "image_height": {"min": int(m["height"].min()), "max": int(m["height"].max()), "median": float(m["height"].median())},
        "colour_modes": {k: int(v) for k, v in m["mode"].value_counts().items()},
        "file_extensions": {k: int(v) for k, v in m["ext"].value_counts().items()},
        "skipped_total": len(scan.skipped),
        "skipped_by_reason": dict(reasons.most_common()),
        "skipped_examples": dict(examples),
        "excluded_classes": scan.excluded_classes,
        "warnings": scan.warnings,
        "claims_vs_measured": claims_vs_measured(m, by_cat),
        "split_plan_full_dataset": plan_splits(m)[1],
        "config": config.as_dict(),
    }


def print_summary(s: dict[str, Any]) -> None:
    """Print the headline numbers in plain text."""
    ipc = s["images_per_class"]
    lines = [
        "",
        "=" * 64,
        f" DATASET SUMMARY  ({s['dataset']})",
        "=" * 64,
        f" Total usable images : {s['total_images']:,}",
        f" Number of classes   : {s['num_classes']}",
        f" Images per class    : min {ipc['min']:,} | max {ipc['max']:,} | mean {ipc['mean']:,} | "
        f"median {ipc['median']:,.0f} (max/min = {ipc['imbalance_ratio_max_over_min']})",
        f" Image width  (px)   : {s['image_width']['min']}-{s['image_width']['max']} (median {s['image_width']['median']:.0f})",
        f" Image height (px)   : {s['image_height']['min']}-{s['image_height']['max']} (median {s['image_height']['median']:.0f})",
        f" Colour modes        : {s['colour_modes']}",
        f" File types          : {s['file_extensions']}",
        f" Skipped files       : {s['skipped_total']:,}",
    ]
    lines += [f"     - {reason}: {n:,}" for reason, n in s["skipped_by_reason"].items()]
    lines.append(" Per category        :")
    for cat, v in s["categories"].items():
        st = v["images_per_class"]
        lines.append(f"     - {cat:<10} {v['classes']:>3} classes | {v['images']:>8,} images | "
                     f"per class min {st['min']:,} / mean {st['mean']:,.0f} / max {st['max']:,}")
    cvm = s["claims_vs_measured"]
    pc = cvm["surveyed_papers"]
    up = cvm["kaggle_uploader"]
    lines += [
        f" Papers' claim       : {pc['claimed_classes']} classes x {pc['claimed_images_per_class']} -> "
        f"classes match: {pc['classes_match']}, per-class match: {pc['per_class_match']}",
        f" Uploader's claim    : {up['claimed_classes']} classes / {up['claimed_total_images']:,} images -> "
        f"classes match: {up['classes_match']}, total match: {up['total_match']} "
        f"(difference {up['total_difference']:+,})",
        " Differences         :",
        *[f"     - {d}" for d in cvm["differences"]],
        "=" * 64,
    ]
    for line in lines:
        log.info(line)


CATEGORY_COLOURS = {"digit": "#c8812a", "vowel": "#5b8c5a", "consonant": "#1f3a5f", "other": "#8c4a5b"}


def plot_class_distribution(manifest: pd.DataFrame, path: Path = DISTRIBUTION_FIG) -> Path:
    """Two panels: (left) images per class sorted by count, coloured by
    category; (right) per-category spread of images-per-class."""
    plt = setup_matplotlib()
    counts = (manifest.groupby(["label", "category"]).size().reset_index(name="n")
              .sort_values(["n", "label"], ascending=[False, True]))
    cats = [c for c in CATEGORY_COLOURS if c in set(counts["category"])]
    colour = {c: CATEGORY_COLOURS[c] for c in cats}
    bg = "#fbf7ef"

    fig_w = max(13, 0.26 * len(counts) + 4)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fig_w, 5.8), dpi=130,
                                  gridspec_kw={"width_ratios": [max(3, len(counts) / 8), 1.2]})
    fig.patch.set_facecolor(bg)
    for a in (ax, ax2):
        a.set_facecolor(bg)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    ax.bar(range(len(counts)), counts["n"], color=[colour[c] for c in counts["category"]])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts["label"], rotation=90, fontsize=8)
    ax.set_xlim(-0.6, len(counts) - 0.4)
    mean = counts["n"].mean()
    ax.axhline(mean, color="#333", ls="--", lw=1)
    ax.text(len(counts) - 0.5, mean, f" mean {mean:,.0f}", va="bottom", ha="right", fontsize=8)
    ax.set_ylabel("images per class")
    ax.set_title(f"Images per class (sorted): {len(counts)} classes, {int(counts['n'].sum()):,} images",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[c]) for c in cats]
    ax.legend(handles, cats, title="category", fontsize=8, frameon=False, loc="upper right")

    rng = np.random.default_rng(0)  # jitter only; does not touch the data
    for i, c in enumerate(cats):
        vals = counts.loc[counts["category"] == c, "n"].to_numpy()
        ax2.boxplot(vals, positions=[i], widths=0.5, showfliers=False,
                    medianprops={"color": "#222"}, boxprops={"color": colour[c]},
                    whiskerprops={"color": colour[c]}, capprops={"color": colour[c]})
        ax2.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=14, color=colour[c], alpha=0.8, zorder=3)
        ax2.text(i, 1.0, f"{len(vals)} classes\n{int(vals.sum()):,} imgs", transform=ax2.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("images per class")
    ax2.set_title("By category", fontsize=11, pad=28)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return path


def write_summary(scan: DatasetScan) -> dict[str, Any]:
    """Summarise, print, save JSON and the distribution chart."""
    summary = summarize(scan)
    print_summary(summary)
    save_json(summary, SUMMARY_PATH)
    plot_class_distribution(scan.manifest)
    log.info("Wrote %s and %s", SUMMARY_PATH.relative_to(config.PROJECT_ROOT),
             DISTRIBUTION_FIG.relative_to(config.PROJECT_ROOT))
    return summary


# ---------------------------------------------------------------------------
# Processed images (shared by feature extraction and the CNN baseline)
# ---------------------------------------------------------------------------
PROCESSED_INDEX = config.PATHS.data_processed / "processed_index.csv"


def load_processed_index() -> pd.DataFrame:
    """The exact set of preprocessed images (with ``class_rank``)."""
    if not PROCESSED_INDEX.exists():
        raise ModiError("No processed images found. Run: python scripts/run_preprocessing.py")
    df = pd.read_csv(PROCESSED_INDEX, keep_default_na=False)
    if df.empty:
        raise ModiError(f"{PROCESSED_INDEX} is empty - preprocessing produced no usable images.")
    return df


def load_processed_images(index: pd.DataFrame) -> np.ndarray:
    """Load preprocessed images for ``index`` rows as uint8 (N, S, S), ink = 255.

    The array is cached next to the images, keyed by the image list and the
    preprocessing parameters, so later runs load in seconds.
    """
    from src import preprocess  # local import: preprocess pulls in OpenCV

    key = hashlib.sha1(("\n".join(index["processed_path"]) + preprocess.params_hash()).encode()).hexdigest()[:12]
    cache = config.PATHS.data_processed / f"_images_{key}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(index):
            return arr
    size = config.PREPROCESS.image_size
    arr = np.empty((len(index), size, size), np.uint8)
    for i, rel in enumerate(index["processed_path"]):
        img = preprocess.read_image(config.PROJECT_ROOT / rel)
        if img.shape != (size, size):
            raise ModiError(f"{rel} has shape {img.shape}, expected {(size, size)}. "
                            "Re-run preprocessing with --force.")
        arr[i] = img
        if (i + 1) % 25000 == 0:
            log.info("  loaded %d / %d processed images", i + 1, len(index))
    for old in config.PATHS.data_processed.glob("_images_*.npy"):
        old.unlink()  # only one cache generation is kept
    np.save(cache, arr)
    return arr
