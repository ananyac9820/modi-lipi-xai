"""Download the MODI-HDC dataset from Kaggle into data/raw/.

The Kaggle bundle is large (~10 GB). It contains:
  * MODI_HChar - isolated handwritten characters, organised category/class/image
  * MODI_HHDoc - full handwritten document pages
By default only the character images plus a small sample of document pages
(used later to test segmentation) are extracted.

Usage:
    python scripts/download_data.py                  # download via Kaggle API
    python scripts/download_data.py --force          # re-download / re-extract
    python scripts/download_data.py --from-zip PATH  # use a zip you downloaded in the browser
    python scripts/download_data.py --all-documents  # also extract every document page

Authentication (any ONE of these; checked in this order by kaggle>=2.0):
  1. New-style API token: env var KAGGLE_API_TOKEN, or file ~/.kaggle/access_token
  2. Legacy API key: file ~/.kaggle/kaggle.json  {"username": "...", "key": "..."}
     or env vars KAGGLE_USERNAME + KAGGLE_KEY
  3. OAuth: run `kaggle auth login` once (cached in ~/.kaggle/credentials.json)
(KAGGLE_CONFIG_DIR relocates kaggle.json only.)

Exit codes: 0 success, 1 download/extract failure, 2 no Kaggle credentials,
3 not enough disk space.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.utils import get_logger, stage  # noqa: E402

log = get_logger("download")

RAW = config.PATHS.data_raw
DS = config.DATASET
ZIP_NAME = DS.kaggle_slug.split("/")[1] + ".zip"


# ---------------------------------------------------------------------------
# Credential detection (names only; secret values are never printed)
# ---------------------------------------------------------------------------
def kaggle_config_dir() -> Path:
    """Folder the kaggle package reads credentials from."""
    return Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")


def detect_credentials() -> list[str]:
    """Return human-readable descriptions of every credential source present."""
    found: list[str] = []
    cfg = kaggle_config_dir()
    # kagglesdk hard-codes ~/.kaggle for token + OAuth files; KAGGLE_CONFIG_DIR
    # only relocates kaggle.json.
    home_cfg = Path.home() / ".kaggle"
    if os.environ.get("KAGGLE_API_TOKEN"):
        found.append("env var KAGGLE_API_TOKEN")
    for name in ("access_token", "access_token.txt"):
        f = home_cfg / name
        if f.is_file() and f.read_text(encoding="utf-8", errors="ignore").strip():
            found.append(f"token file {f}")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("env vars KAGGLE_USERNAME + KAGGLE_KEY")
    kj = cfg / "kaggle.json"
    if kj.is_file():
        try:
            data = json.loads(kj.read_text(encoding="utf-8"))
            if data.get("username") and data.get("key"):
                found.append(f"legacy key file {kj}")
            else:
                log.warning("%s exists but lacks 'username'/'key' fields - ignoring it.", kj)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("%s could not be parsed (%s) - ignoring it.", kj, exc)
    if (home_cfg / "credentials.json").is_file():
        found.append(f"cached OAuth login {home_cfg / 'credentials.json'}")
    return found


def manual_instructions(reason: str) -> str:
    """Instructions printed whenever the automatic route fails."""
    cfg = kaggle_config_dir()
    return f"""
==========================================================================
 Could not download the dataset automatically: {reason}

 OPTION 1 - set up Kaggle credentials, then re-run this script
   Get a key at https://www.kaggle.com/settings/api  ("Create New Token").
   Any one of the following works:
     a) Legacy key file: save the downloaded kaggle.json as
            {cfg / 'kaggle.json'}
        (it contains {{"username": "...", "key": "..."}})
     b) Environment variables (PowerShell):
            $env:KAGGLE_USERNAME = "your_username"
            $env:KAGGLE_KEY      = "your_key"
     c) New-style token: $env:KAGGLE_API_TOKEN = "<token>"
        or save the token text to {Path.home() / '.kaggle' / 'access_token'}
     d) OAuth in the browser: run  kaggle auth login
   Never commit these files - kaggle.json is already in .gitignore.

 OPTION 2 - download manually in the browser
   1. Open {DS.kaggle_url}
   2. Click "Download" (about 10 GB, needs a free Kaggle login)
   3. Then EITHER let this script extract the useful parts:
            python scripts/download_data.py --from-zip "C:\\path\\to\\archive.zip"
      OR unzip it yourself so that this folder exists:
            {RAW / 'MODI_HChar'}  (containing consonants/, vowels/, ... class folders)
==========================================================================
"""


# ---------------------------------------------------------------------------
# Presence / extraction helpers
# ---------------------------------------------------------------------------
def characters_present() -> int:
    """Count character images already extracted (stops early once clearly present)."""
    n = 0
    for p in RAW.rglob("*"):
        if p.suffix.lower() in DS.image_extensions and DS.character_root_hint in str(p).lower():
            n += 1
            if n >= 1000:
                break
    return n


def _is_character_member(name: str) -> bool:
    return any(DS.character_root_hint in part.lower() for part in PurePosixPath(name).parts)


def _safe_target(name: str) -> Path | None:
    """Resolve a zip member path inside RAW, rejecting path-traversal entries."""
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    target = (RAW / Path(*pure.parts)).resolve()
    return target if target.is_relative_to(RAW.resolve()) else None


def extract(zip_path: Path, all_documents: bool) -> tuple[int, int]:
    """Extract character images (+ a sample of document pages) from the zip.

    Returns (n_character_files, n_document_files) written.
    """
    n_chars = n_docs = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        doc_members = sorted(
            (m for m in members if not _is_character_member(m.filename)
             and PurePosixPath(m.filename).suffix.lower() in DS.image_extensions),
            key=lambda m: m.filename,
        )
        doc_keep = set(m.filename for m in (doc_members if all_documents else doc_members[: DS.document_samples]))
        total = sum(1 for m in members if _is_character_member(m.filename)) + len(doc_keep)
        log.info("Zip has %d files; extracting %d (characters + %d document pages)",
                 len(members), total, len(doc_keep))
        for i, m in enumerate(members, 1):
            keep_char = _is_character_member(m.filename)
            if not (keep_char or m.filename in doc_keep):
                continue
            target = _safe_target(m.filename)
            if target is None:
                log.warning("Skipping unsafe zip entry: %s", m.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n_chars += keep_char
            n_docs += not keep_char
            if (n_chars + n_docs) % 25000 == 0:
                log.info("  extracted %d / %d", n_chars + n_docs, total)
    return n_chars, n_docs


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def kaggle_download() -> Path:
    """Authenticate and download the dataset zip; raise RuntimeError on failure."""
    # NOTE: `import kaggle` would authenticate at import time and print help
    # text; importing the class directly keeps control here.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:  # kaggle calls exit(1) when no credentials work
        raise RuntimeError("Kaggle rejected or could not find the credentials") from exc
    api.dataset_download_files(DS.kaggle_slug, path=str(RAW), unzip=False, quiet=False, force=False)
    zips = sorted(RAW.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError("download finished but no .zip appeared in data/raw/")
    return zips[0]


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download / re-extract even if data exists")
    parser.add_argument("--from-zip", type=Path, help="extract an already-downloaded Kaggle zip")
    parser.add_argument("--all-documents", action="store_true", help="extract every document page, not a sample")
    parser.add_argument("--delete-zip", action="store_true", help="delete the zip after successful extraction")
    parser.add_argument("--skip-space-check", action="store_true")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if not args.force and characters_present() > 0:
        log.info("Character images already present in %s - skipping download (use --force to redo).", RAW)
        log.info("Next: python scripts/run_preprocessing.py")
        return 0

    try:
        if args.from_zip:
            zip_path = args.from_zip.expanduser().resolve()
            if not zip_path.is_file():
                print(manual_instructions(f"zip not found: {zip_path}"))
                return 1
        else:
            existing = RAW / ZIP_NAME
            if existing.is_file() and zipfile.is_zipfile(existing) and not args.force:
                log.info("Found previously downloaded %s - reusing it.", existing.name)
                zip_path = existing
            else:
                creds = detect_credentials()
                if not creds:
                    print(manual_instructions("no Kaggle credentials found"))
                    return 2
                log.info("Kaggle credentials found: %s", "; ".join(creds))
                space = free_gb(RAW)
                if space < DS.min_free_space_gb and not args.skip_space_check:
                    log.error("Only %.1f GB free on the data drive; need about %.0f GB "
                              "(zip + extracted characters). Free space or pass --skip-space-check.",
                              space, DS.min_free_space_gb)
                    return 3
                with stage(f"Downloading {DS.kaggle_slug} (~10 GB, this takes a while)", log):
                    zip_path = kaggle_download()

        if not zipfile.is_zipfile(zip_path):
            print(manual_instructions(f"{zip_path.name} is not a valid zip (incomplete download?)"))
            return 1
        with stage(f"Extracting {zip_path.name}", log):
            n_chars, n_docs = extract(zip_path, args.all_documents)
        if n_chars == 0:
            print(manual_instructions(
                f"no folder containing '{DS.character_root_hint}' found in the zip - the dataset layout may have changed"))
            return 1
        log.info("Extracted %d character images and %d document pages into %s", n_chars, n_docs, RAW)
        if args.delete_zip and zip_path.parent.resolve() == RAW.resolve():
            zip_path.unlink()
            log.info("Deleted %s", zip_path.name)
        elif zip_path.parent.resolve() == RAW.resolve():
            log.info("Kept %s (%.1f GB). Re-run with --delete-zip to remove it after checking the data.",
                     zip_path.name, zip_path.stat().st_size / 1024**3)
        log.info("Next: python scripts/run_preprocessing.py")
        return 0
    except KeyboardInterrupt:
        log.warning("Interrupted. Partial files may remain; re-run with --force to start clean.")
        return 1
    except Exception as exc:  # noqa: BLE001 - always end with instructions, never a traceback
        print(manual_instructions(f"{type(exc).__name__}: {exc}"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
