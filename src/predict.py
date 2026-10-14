"""Single entry point used by the web app (and usable from Python).

    from src.predict import Recognizer
    rec = Recognizer()                       # loads the trained model lazily
    result = rec.recognize(image)            # segment + preprocess + features + predict
    explanation = rec.explain(result, k)     # SHAP for character k (cached explainer)

Everything the app shows comes from here, so the app never touches the
models directly. Failures surface as ``ModiError`` with a readable message.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src import config, models, preprocess, segmentation
from src.features import fusion
from src.utils import ModiError, get_logger

log = get_logger("predict")
MAIN_MODEL = config.PATHS.models / "main_stack.joblib"
LABELS_FILE = Path(__file__).resolve().parent / "labels.json"


def load_label_info() -> dict[str, dict[str, Any]]:
    """Display information per class: Devanagari, Modi Unicode, category, and
    whether the Devanagari mapping has been verified by a human."""
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))["labels"]
    return {}


def confidence_level(p: float) -> str:
    """'high' (green), 'medium' (amber) or 'low' (red)."""
    if p >= config.APP.high_confidence:
        return "high"
    if p >= config.APP.low_confidence:
        return "medium"
    return "low"


@dataclass
class CharPrediction:
    """Everything known about one segmented character."""

    order: int
    box: dict[str, Any]
    crop: np.ndarray                          # segmentation crop (dark ink on white)
    processed: np.ndarray | None              # 64x64 model input (ink = 255)
    features: np.ndarray | None               # fused raw feature vector
    label: str | None
    confidence: float
    level: str
    top: list[dict[str, Any]]
    model_votes: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    steps: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self, label_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
        info = label_info.get(self.label or "", {})
        return {"order": self.order, "label": self.label, "devanagari": info.get("devanagari"),
                "mapping_verified": info.get("verified"), "confidence": round(self.confidence, 4),
                "confidence_level": self.level, "top3": self.top, "base_model_votes": self.model_votes,
                "segmentation_flags": self.box.get("flags", []), "error": self.error, "box": self.box}


@dataclass
class RecognitionResult:
    """Output of :meth:`Recognizer.recognize`."""

    segmentation: segmentation.SegmentationResult
    chars: list[CharPrediction]
    seconds: dict[str, float]
    single_character: bool = False

    def text(self, label_info: dict[str, dict[str, Any]] | None = None, use_devanagari: bool = False) -> str:
        """Assembled reading in reading order (lines separated by ' / ')."""
        info = label_info or {}
        out, line = [], None
        for c in self.chars:
            if line is not None and c.box.get("line") != line:
                out.append("/")
            line = c.box.get("line")
            tok = c.label or "?"
            if use_devanagari and info.get(tok, {}).get("devanagari"):
                tok = info[tok]["devanagari"]
            out.append(tok if c.level != "low" else f"{tok}?")
        return " ".join(out)


class Recognizer:
    """Loads the trained stacking ensemble once and recognises images."""

    def __init__(self, model_path: Path = MAIN_MODEL) -> None:
        self.model_path = Path(model_path)
        self._bundle: models.TrainedModels | None = None
        self._explainer = None
        self.label_info = load_label_info()

    @property
    def bundle(self) -> models.TrainedModels:
        if self._bundle is None:
            self._bundle = models.load(self.model_path)
        return self._bundle

    @property
    def classes(self) -> np.ndarray:
        return self.bundle.classes

    def recognize(self, image: np.ndarray, single: bool | None = None) -> RecognitionResult:
        """Segment an image and classify every character.

        Args:
            image: grayscale or colour array.
            single: force single-character mode (None = automatic: used when
                segmentation finds exactly one box).
        """
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ModiError("The uploaded file could not be read as an image.")
        t0 = time.perf_counter()
        seg = segmentation.segment(image)
        t_seg = time.perf_counter() - t0
        is_single = single if single is not None else len(seg.boxes) == 1
        chars: list[CharPrediction] = []
        feats, ok_idx = [], []
        t1 = time.perf_counter()
        for k, (box, crop) in enumerate(zip(seg.boxes, seg.crops)):
            try:
                res = preprocess.preprocess_array(crop, keep_steps=True)
                f = fusion.extract_one(res.final, self.bundle.feature_set)
                feats.append(f); ok_idx.append(k)
                chars.append(CharPrediction(box.order, box.as_dict(), crop, res.final, f, None, 0.0, "low", [],
                                            steps=res.steps))
            except ModiError as exc:
                chars.append(CharPrediction(box.order, box.as_dict(), crop, None, None, None, 0.0, "low", [],
                                            error=str(exc)))
        t_feat = time.perf_counter() - t1
        t2 = time.perf_counter()
        if feats:
            X = np.vstack(feats)
            Z = self.bundle.transform(X)
            proba = self.bundle.predict_proba(Z, reduced=True)
            votes = {m: self.bundle.predict("rf" if m == "rf" else m, Z, reduced=True)
                     for m in self.bundle.stack.named_estimators_}
            for row, k in enumerate(ok_idx):
                p = proba[row]
                order = np.argsort(p)[::-1][: config.APP.top_k]
                c = chars[k]
                c.label = str(self.classes[order[0]])
                c.confidence = float(p[order[0]])
                # Low segmentation confidence caps how sure we claim to be.
                seg_conf = float(c.box.get("segmentation_confidence", 1.0))
                c.level = confidence_level(c.confidence * (1.0 if seg_conf >= 0.99 else seg_conf))
                c.top = [{"label": str(self.classes[i]), "proba": float(p[i])} for i in order]
                c.model_votes = {m: str(v[row]) for m, v in votes.items()}
        t_pred = time.perf_counter() - t2
        return RecognitionResult(seg, chars, {"segmentation": round(t_seg, 3), "features": round(t_feat, 3),
                                              "prediction": round(t_pred, 3)}, is_single)

    # -- explanations --------------------------------------------------------
    def explainer(self):
        """Cached SHAP explainer (built on first use, then loaded from disk)."""
        if self._explainer is None:
            from src import explain
            self._explainer = explain.load_explainer(bundle=self.bundle)
        return self._explainer

    def explain(self, result: RecognitionResult, k: int) -> dict[str, Any]:
        """SHAP explanation for character index k of a recognition result."""
        if not 0 <= k < len(result.chars):
            raise ModiError(f"No character number {k + 1} in this result.")
        c = result.chars[k]
        if c.features is None:
            raise ModiError(f"Character {c.order} could not be processed: {c.error}")
        return self.explainer().explain(c.features)

    def to_records(self, result: RecognitionResult) -> list[dict[str, Any]]:
        """Plain dicts for CSV / JSON download."""
        return [c.summary(self.label_info) for c in result.chars]
