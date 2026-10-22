"""Modi Lipi XAI - web app server.

A small Starlette app: the five pages are static HTML in ``web/static``, and
everything they show comes from the JSON API below, which reads the results
files and wraps :class:`src.predict.Recognizer`.

Run from the project root:
    python -m web.server            # http://127.0.0.1:8000
    python -m web.server --port 9000 --host 0.0.0.0   # reachable from a phone on the same Wi-Fi
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in (None, ""):          # run as a script: python web/server.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src import config, segmentation
from src.utils import ModiError

ROOT = config.PROJECT_ROOT
STATIC = Path(__file__).resolve().parent / "static"
MET, SHAP_DIR, FIG, ASSETS = config.PATHS.metrics, config.PATHS.shap, config.PATHS.figures, config.PATHS.app_assets
MAX_UPLOAD = 12 * 1024 * 1024
LEVEL_RGB = {"high": (61, 122, 74), "medium": (196, 150, 64), "low": (178, 64, 52)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def png_data_uri(img: np.ndarray, max_side: int | None = None) -> str:
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def r(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(float(x), n)


class _Recogniser:
    """Loads the model once; keeps the last few results so a character can be
    explained after it was recognised."""

    def __init__(self, keep: int = 24) -> None:
        self._rec = None
        self._lock = threading.Lock()
        self.results: OrderedDict[str, Any] = OrderedDict()
        self.keep = keep

    @property
    def rec(self):
        with self._lock:
            if self._rec is None:
                from src.predict import Recognizer

                rec = Recognizer()
                _ = rec.bundle
                self._rec = rec
        return self._rec

    def store(self, result) -> str:
        rid = uuid.uuid4().hex[:12]
        self.results[rid] = result
        while len(self.results) > self.keep:
            self.results.popitem(last=False)
        return rid


RECOGNISER = _Recogniser()


# ---------------------------------------------------------------------------
# summaries built from the results files
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def label_info() -> dict[str, dict[str, Any]]:
    from src.predict import load_label_info

    return load_label_info()


def curves() -> dict[str, Any]:
    cl = read_json(MET / "learning_curve_classical.json") or {"points": []}
    cnn = read_json(MET / "learning_curve_cnn.json") or {"points": []}
    classical = [{"n": p["train_per_class"], **{m: r(v["accuracy"]) for m, v in p.get("models", {}).items()},
                  "train_time_s": r(p.get("train_time_s"), 1)} for p in cl["points"]]
    cnn_pts = [{"n": p["train_per_class"], "cnn": r(p["accuracy"]), "train_time_s": r(p.get("train_time_s"), 1),
                "epochs": p.get("best_epoch"), "params": p.get("n_parameters")} for p in cnn["points"]]
    planned = list(config.SAMPLING.curve_train_sizes)
    crossover = None
    by_n = {p["n"]: p for p in classical}
    prev = None
    for p in sorted(cnn_pts, key=lambda q: q["n"]):
        best = max((v for k, v in by_n.get(p["n"], {}).items() if k in ("rf", "svm", "xgb", "stack") and v), default=None)
        if best is None:
            continue
        if p["cnn"] > best and prev is not None and prev[1] <= prev[2]:
            crossover = {"between": [prev[0], p["n"]]}
        prev = (p["n"], p["cnn"], best)
    return {"classical": classical, "cnn": cnn_pts, "planned": planned, "crossover": crossover,
            "cnn_architecture": cnn.get("architecture"), "reducer": cl.get("reducer"), "feature_set": cl.get("feature_set")}


def project_status() -> list[dict[str, str]]:
    """The 'focus' list on the home page: real state of each part of the project."""
    c = curves()
    planned = c["planned"]
    n_cnn = len(c["cnn"])
    n_cl = len(c["classical"])
    seg = read_json(MET / "segmentation_eval.json")
    shap = read_json(SHAP_DIR / "global_importance.json")
    labels = label_info()
    n_verified = sum(bool(v.get("verified")) for v in labels.values())
    out = [
        {"title": "Classical learning curve", "meta": f"{n_cl}/{len(planned)} pts",
         "status": "done" if n_cl >= len(planned) else "progress", "href": "results.html#curve"},
        {"title": "Reference CNN on the same splits", "meta": f"{n_cnn}/{len(planned)} pts",
         "status": "done" if n_cnn >= len(planned) else "progress", "href": "results.html#curve"},
        {"title": "SHAP explanations, 35 feature groups", "meta": f"{shap['n_images']} imgs" if shap else "-",
         "status": "done" if shap else "todo", "href": "explain.html"},
        {"title": "Word segmentation evaluation", "meta": "test words" if seg else "-",
         "status": "done" if seg else "todo", "href": "results.html#segment"},
        {"title": "Devanagari labels checked by a reader", "meta": f"{n_verified}/{len(labels)}",
         "status": "done" if labels and n_verified == len(labels) else "review", "href": "script.html"},
    ]
    return out


def overview() -> dict[str, Any]:
    ds = read_json(MET / "dataset_summary.json") or {}
    mc = read_json(MET / "model_comparison.json") or {}
    c = curves()
    best_cl = max(c["classical"], key=lambda p: p["n"]) if c["classical"] else None
    models = mc.get("models", {})
    best = mc.get("best_model")
    return {
        "dataset": {"images": ds.get("total_images"), "classes": ds.get("num_classes"),
                    "categories": {k: {"classes": v.get("classes"), "images": v.get("images")}
                                   for k, v in (ds.get("categories") or {}).items() if isinstance(v, dict)}},
        "main": {"train_per_class": mc.get("train_per_class"), "best_model": best,
                 "accuracy": {m: r(v["accuracy"]) for m, v in models.items()},
                 "macro_f1": {m: r(v["macro_f1"]) for m, v in models.items()},
                 "per_category": {m: {k: r(x) for k, x in v.get("per_category_accuracy", {}).items()} for m, v in models.items()},
                 "fit_seconds": mc.get("fit_seconds"), "inference_ms": mc.get("inference_ms_per_image"),
                 "size_bytes": mc.get("model_size_bytes"), "n_test": mc.get("n_test")},
        "largest_point": best_cl,
        "crossover": c["crossover"],
        "status": project_status(),
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
async def api_overview(_: Request) -> JSONResponse:
    return JSONResponse(overview())


async def api_curves(_: Request) -> JSONResponse:
    return JSONResponse(curves())


async def api_results(_: Request) -> JSONResponse:
    pairs = read_json(MET / "most_confused_pairs.json") or {}
    per_class = read_json(MET / "per_class_metrics.json")
    ablation = read_json(MET / "ablation.json")
    reducer = read_json(MET / "reducer_selection.json")
    seg = read_json(MET / "segmentation_eval.json")
    cat = read_json(MET / "per_category_results.json")
    literature = read_json(ASSETS / "literature.json")
    return JSONResponse({"pairs": pairs, "per_class": per_class, "ablation": ablation, "reducer": reducer,
                         "segmentation": seg, "category_errors": cat, "literature": literature})


async def api_explain_global(_: Request) -> JSONResponse:
    g = read_json(SHAP_DIR / "global_importance.json") or {}
    pairs = read_json(SHAP_DIR / "confused_pairs_contrast.json") or {}
    for p in pairs.get("pairs", []):
        p.pop("examples", None)
    return JSONResponse({"global": g, "pairs": pairs})


async def api_script(_: Request) -> JSONResponse:
    pc = read_json(MET / "per_class_metrics.json")
    f1 = {row["label"]: {k: r(row[k]) for k in ("f1", "precision", "recall")}
          for row in (pc or {}).get("classes", [])}
    labels = [{"label": k, **v, **({"metrics": f1[k]} if k in f1 else {})} for k, v in sorted(label_info().items())]
    return JSONResponse({"labels": labels})


@lru_cache(maxsize=128)
def _glyph_png(label: str) -> bytes | None:
    folder = config.PATHS.data_processed / label
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.png"))
    if not files:
        return None
    img = cv2.imread(str(files[len(files) // 2]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    ok, buf = cv2.imencode(".png", 255 - img)
    return buf.tobytes() if ok else None


async def api_glyph(request: Request) -> Response:
    label = request.path_params["label"]
    if label not in label_info():
        return Response(status_code=404)
    data = await run_in_threadpool(_glyph_png, label)
    if data is None:
        return Response(status_code=404)
    return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


async def api_samples(_: Request) -> JSONResponse:
    meta = read_json(ASSETS / "samples.json") or {"samples": []}
    return JSONResponse([s for s in meta["samples"] if (ASSETS / s["file"]).exists()])


async def sample_file(request: Request) -> Response:
    name = Path(request.path_params["name"]).name
    path = ASSETS / name
    if path.suffix.lower() != ".png" or not path.exists():
        return Response(status_code=404)
    return FileResponse(path)


async def figure_file(request: Request) -> Response:
    name = Path(request.path_params["name"]).name
    path = FIG / name
    if path.suffix.lower() != ".png" or not path.exists():
        return Response(status_code=404)
    return FileResponse(path)


def _recognise(img: np.ndarray, mode: str) -> dict[str, Any]:
    rec = RECOGNISER.rec
    single = {"single": True, "line": False}.get(mode)
    result = rec.recognize(img, single=single)
    rid = RECOGNISER.store(result)
    info = rec.label_info
    colours = [LEVEL_RGB.get(c.level, (120, 120, 120)) for c in result.chars]
    boxed = segmentation.draw_boxes(result.segmentation, [str(c.order) for c in result.chars], colours) \
        if result.chars else cv2.cvtColor(result.segmentation.display, cv2.COLOR_GRAY2RGB)
    chars = []
    for k, c in enumerate(result.chars):
        li = info.get(c.label or "", {})
        chars.append({
            "k": k, "order": c.order, "label": c.label, "devanagari": li.get("devanagari"),
            "modi": li.get("modi_unicode"), "category": li.get("category"),
            "confidence": r(c.confidence), "level": c.level, "error": c.error,
            "top": [{"label": t["label"], "proba": r(t["proba"]), "devanagari": info.get(t["label"], {}).get("devanagari")}
                    for t in c.top],
            "votes": c.model_votes, "flags": [{"flag": f, "text": segmentation.FLAG_TEXT.get(f, f)}
                                              for f in c.box.get("flags", [])],
            "line": c.box.get("line"),
            "crop": png_data_uri(c.crop, 160),
            "glyph": png_data_uri(255 - c.processed) if c.processed is not None else None,
        })
    return {"id": rid, "reading": result.text(info), "reading_devanagari": result.text(info, use_devanagari=True),
            "boxed": png_data_uri(boxed, 1400), "chars": chars, "seconds": result.seconds,
            "warnings": result.segmentation.warnings, "single": result.single_character}


async def api_recognise(request: Request) -> JSONResponse:
    form = await request.form(max_files=1, max_part_size=MAX_UPLOAD)
    mode = str(form.get("mode") or "auto")
    upload, sample = form.get("image"), form.get("sample")
    try:
        if sample:
            path = ASSETS / Path(str(sample)).name
            if not path.exists():
                return JSONResponse({"error": "Unknown sample."}, status_code=404)
            raw = path.read_bytes()
        elif upload is not None and hasattr(upload, "read"):
            raw = await upload.read()
        else:
            return JSONResponse({"error": "No image was sent."}, status_code=400)
        if len(raw) > MAX_UPLOAD:
            return JSONResponse({"error": "That image is larger than 12 MB."}, status_code=413)
        img = Image.open(io.BytesIO(raw))
        img.load()
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:   # drawn on a transparent canvas
            bg = Image.new("RGB", img.size, "white")
            bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
            img = bg
        arr = np.array(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError):
        return JSONResponse({"error": "That file could not be opened as an image. Try a PNG or JPG."}, status_code=400)
    try:
        out = await run_in_threadpool(_recognise, arr, mode)
    except ModiError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    return JSONResponse(out)


def _explain(rid: str, k: int) -> dict[str, Any]:
    result = RECOGNISER.results.get(rid)
    if result is None:
        raise ModiError("This result has expired. Run the recognition again.")
    ex = RECOGNISER.rec.explain(result, k)
    keep = ("predicted", "predicted_proba", "runner_up", "runner_up_proba", "baseline_proba_predicted",
            "toward_predicted", "against_predicted", "toward_runner_up", "family_contribution_predicted",
            "region_maps_predicted", "region_maps_runner_up", "additivity_error")
    out = {k2: ex[k2] for k2 in keep}
    info = label_info()
    out["predicted_devanagari"] = info.get(ex["predicted"], {}).get("devanagari")
    out["runner_up_devanagari"] = info.get(ex["runner_up"], {}).get("devanagari")
    return out


async def api_explain(request: Request) -> JSONResponse:
    rid, k = request.path_params["rid"], int(request.path_params["k"])
    try:
        return JSONResponse(await run_in_threadpool(_explain, rid, k))
    except ModiError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)


async def api_health(_: Request) -> JSONResponse:
    model = config.PATHS.models / "main_stack.joblib"
    return JSONResponse({"ok": True, "model_available": model.exists()})


routes = [
    Route("/api/health", api_health),
    Route("/api/overview", api_overview),
    Route("/api/curves", api_curves),
    Route("/api/results", api_results),
    Route("/api/explain-global", api_explain_global),
    Route("/api/script", api_script),
    Route("/api/glyph/{label}", api_glyph),
    Route("/api/samples", api_samples),
    Route("/api/recognise", api_recognise, methods=["POST"]),
    Route("/api/explain/{rid}/{k:int}", api_explain),
    Route("/samples/{name}", sample_file),
    Route("/figures/{name}", figure_file),
    Mount("/", StaticFiles(directory=STATIC, html=True)),
]


class Revalidate:
    """Pages and scripts are small: make browsers re-check them on every load
    (cheap 304s via ETag) so an updated page is never served stale."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].startswith(("/api/glyph", "/figures/", "/samples/")):
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                headers.append((b"cache-control", b"no-cache"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


app = Revalidate(Starlette(routes=routes))


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Modi Lipi XAI web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
