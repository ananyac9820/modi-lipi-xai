"""End-to-end tests of the web app through its HTTP API (in-process, no server).

Run:  python -m pytest -q tests/test_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starlette.testclient import TestClient  # noqa: E402

from src import config  # noqa: E402
from web.server import app  # noqa: E402

PAGES = ["index.html", "read.html", "results.html", "explain.html", "script.html"]
HAS_MODEL = (config.PATHS.models / "main_stack.joblib").exists()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("page", PAGES)
def test_pages_are_served_with_tab_bar_script(client, page):
    r = client.get("/" + page)
    assert r.status_code == 200
    assert "js/layout.js" in r.text and 'name="viewport"' in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_root_serves_home(client):
    r = client.get("/")
    assert r.status_code == 200 and 'data-page="home"' in r.text


@pytest.mark.parametrize("path", ["/js/layout.js", "/css/paper.css", "/img/icon.svg"])
def test_static_assets(client, path):
    assert client.get(path).status_code == 200


def test_overview_matches_results_files(client):
    ov = client.get("/api/overview").json()
    assert ov["dataset"]["classes"] == 57
    assert {s["status"] for s in ov["status"]} <= {"done", "progress", "review", "todo", "blocked"}
    curves = client.get("/api/curves").json()
    assert [p["n"] for p in curves["classical"]] == sorted(p["n"] for p in curves["classical"])
    assert set(curves["planned"]) >= {p["n"] for p in curves["cnn"]}


def test_crossover_is_first_size_where_cnn_beats_every_classical_model(client):
    c = client.get("/api/curves").json()
    if not c["crossover"]:
        pytest.skip("no crossover measured yet")
    lo, hi = c["crossover"]["between"]
    cl = {p["n"]: max(p[m] for m in ("rf", "svm", "xgb", "stack")) for p in c["classical"]}
    cnn = {p["n"]: p["cnn"] for p in c["cnn"]}
    assert cnn[lo] <= cl[lo] and cnn[hi] > cl[hi]


def test_script_lists_all_classes_with_metrics(client):
    labels = client.get("/api/script").json()["labels"]
    assert len(labels) == 57
    assert all("metrics" in l for l in labels)


def test_unknown_glyph_and_path_traversal_are_rejected(client):
    assert client.get("/api/glyph/not-a-class").status_code == 404
    assert client.get("/figures/..%2F..%2FREADME.md").status_code == 404
    assert client.get("/samples/..%2Fapp.py").status_code == 404


def test_recognise_rejects_non_images(client):
    r = client.post("/api/recognise", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400 and "error" in r.json()


@pytest.mark.skipif(not HAS_MODEL, reason="trained model not available")
def test_recognise_sample_then_explain(client):
    r = client.post("/api/recognise", data={"sample": "sample_3_digits.png", "mode": "auto"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["chars"]) == 6
    assert out["boxed"].startswith("data:image/png;base64,")
    assert all(c["level"] in ("high", "medium", "low") for c in out["chars"])
    ex = client.get(f"/api/explain/{out['id']}/0").json()
    assert ex["predicted"] == out["chars"][0]["label"]
    assert len(ex["region_maps_predicted"]["hog"]) == 4
    assert ex["additivity_error"] < 1e-3
    assert client.get("/api/explain/expired/0").status_code == 404
