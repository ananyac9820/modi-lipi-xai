"""Cached loaders for the app. Every loader degrades gracefully: a missing
artefact returns None and the page shows a helpful note instead of crashing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from src import config
from src.utils import ModiError

MET = config.PATHS.metrics
ASSETS = config.PATHS.app_assets


@st.cache_resource(show_spinner=False)
def recognizer():
    """The trained model, loaded once per server process."""
    from src.predict import Recognizer

    rec = Recognizer()
    _ = rec.bundle            # fail early with a clear ModiError if the model is missing
    return rec


@st.cache_resource(show_spinner=False)
def explainer():
    """SHAP explainer (built once, cached on disk and in memory)."""
    return recognizer().explainer()


def model_available() -> tuple[bool, str]:
    try:
        recognizer()
        return True, ""
    except ModiError as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False)
def read_json(name: str, folder: str = "metrics") -> dict[str, Any] | None:
    base = {"metrics": MET, "shap": config.PATHS.shap, "assets": ASSETS}[folder]
    path = base / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def samples() -> list[dict[str, Any]]:
    meta = read_json("samples.json", "assets") or {"samples": []}
    return [s for s in meta["samples"] if (ASSETS / s["file"]).exists()]


@st.cache_data(show_spinner=False)
def reduced_training_sample(per_class: int = 25) -> dict[str, np.ndarray] | None:
    """A small sample of training images in the model's reduced space, for the
    pipeline view's 'where does this character land' plot."""
    try:
        from src import data_loader
        from src.features import fusion

        rec = recognizer()
        index = data_loader.load_processed_index()
        index["_pos"] = np.arange(len(index))
        train, _, _ = data_loader.stratified_split(index, config.SAMPLING.main_train_per_class, quiet=True)
        s = train.groupby("label").head(per_class)
        X = fusion.load_feature_set(index, rec.bundle.feature_set)[s["_pos"].to_numpy()]
        Z = rec.bundle.transform(X)
        return {"Z": Z[:, :2], "label": s["label"].to_numpy(), "category": s["category"].to_numpy()}
    except Exception:  # noqa: BLE001 - optional visual
        return None


def image_path(rel: str) -> Path:
    return config.PROJECT_ROOT / rel
