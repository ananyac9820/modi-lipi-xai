"""Small shared helpers: logging, stage timing, seeding, JSON I/O and
user-friendly error handling for the command-line scripts."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from src import config


class ModiError(Exception):
    """An expected, user-facing failure (missing dataset, missing model, bad
    image...). Scripts catch it and print the message without a traceback."""


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    root = logging.getLogger()
    if not root.handlers:
        # Windows consoles default to cp1252, which cannot print Devanagari
        # class names or the stage markers; fall back to replacement chars.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (ValueError, OSError):
                    pass
        logging.basicConfig(
            level=config.LOG_LEVEL,
            format=config.LOG_FORMAT,
            datefmt=config.LOG_DATEFMT,
            stream=sys.stdout,
        )
        # Third-party libraries are chatty at INFO.
        for noisy in ("matplotlib", "PIL", "numba", "shap", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


@contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log the start and duration of a pipeline stage.

    Example:
        with stage("Feature extraction"):
            ...
    """
    log = logger or get_logger("pipeline")
    log.info("▶ %s ...", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("✔ %s done in %.1fs", name, time.perf_counter() - start)


def setup_matplotlib() -> Any:
    """Headless matplotlib with a font fallback chain that covers Devanagari
    (class labels such as 'क'). Returns ``matplotlib.pyplot``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    fallbacks = [f for f in ("Nirmala UI", "Mangal", "Noto Sans Devanagari", "Lohit Devanagari") if f in installed]
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", *fallbacks]
    return plt


def set_seed(seed: int = config.SEED) -> None:
    """Seed every random number generator we rely on."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj: Any, path: Path) -> Path:
    """Write ``obj`` as pretty JSON, converting numpy types on the way."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    """Read a JSON file, raising :class:`ModiError` with a hint if missing."""
    if not path.exists():
        raise ModiError(f"Expected file not found: {path}. Run the earlier pipeline stage first.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_cli(main: Callable[[], int | None]) -> None:
    """Run a script's ``main`` so expected failures print a clean message
    instead of a raw traceback. Exit code is 1 on failure."""
    log = get_logger("cli")
    try:
        code = main()
    except ModiError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - last-resort friendly message
        log.error("Unexpected error: %s: %s", type(exc).__name__, exc)
        log.error("Re-run with the environment variable MODI_DEBUG=1 to see the full traceback.")
        if os.environ.get("MODI_DEBUG"):
            raise
        sys.exit(1)
    sys.exit(code or 0)
