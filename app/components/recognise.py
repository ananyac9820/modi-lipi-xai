"""Part 1: the Recognise page."""

from __future__ import annotations

import hashlib
import io
import json

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.components import data
from app.components.theme import (FAMILY_COLOUR, GREEN, INK, LEVEL_COLOUR, MUTED, RED, SAFFRON, conf_bar, esc,
                                  level_badge, plotly_layout)
from src import config, segmentation
from src.utils import ModiError

PER_ROW = 8


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def _load_upload(file) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(file.getvalue()))
        img.load()
        return np.array(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(f"“{file.name}” could not be opened as an image. Try a PNG or JPG.")
        return None


def _set_image(img: np.ndarray, name: str, truth: list[str] | None = None) -> None:
    key = hashlib.sha1(img.tobytes()[:2_000_000] + name.encode()).hexdigest()[:16]
    if st.session_state.get("image_key") != key:
        st.session_state.update(image=img, image_name=name, image_key=key, truth=truth, sel=0, result=None,
                                explanations={})


def input_section() -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="kicker">Step 1</div><h3 style="margin-top:0">Upload a Modi image</h3>',
                    unsafe_allow_html=True)
        up = st.file_uploader("Drop an image with one or more handwritten Modi characters",
                              type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed")
        st.caption("Works best with dark ink on light paper, characters separated by small gaps. "
                   "A word or a few lines is fine; whole pages are slow and segment poorly.")
        if up is not None:
            img = _load_upload(up)
            if img is not None:
                _set_image(img, up.name)
    with right:
        st.markdown('<div class="kicker">or</div><h3 style="margin-top:0">Try a sample</h3>', unsafe_allow_html=True)
        samples = data.samples()
        if not samples:
            st.info("No bundled samples found. Run `python scripts/make_sample_images.py`.")
        cols = st.columns(2)
        for k, s in enumerate(samples):
            with cols[k % 2]:
                img = np.array(Image.open(data.ASSETS / s["file"]).convert("RGB"))
                st.image(img, width="stretch")
                if st.button(s["description"], key=f"sample_{k}", width="stretch"):
                    _set_image(img, s["file"], s.get("ground_truth"))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _run(rec) -> None:
    if st.session_state.get("result") is not None:
        return
    with st.status("Reading the manuscript…", expanded=False) as status:
        st.write("Cleaning the image and finding lines and characters…")
        try:
            result = rec.recognize(st.session_state["image"])
        except ModiError as exc:
            status.update(label="Could not read this image", state="error")
            st.error(str(exc))
            return
        st.write(f"Found {len(result.chars)} character(s); extracting HOG, Zernike and LBP features…")
        st.write("Classifying with the stacking ensemble (Random Forest + SVM + XGBoost)…")
        status.update(label=f"Done: {len(result.chars)} character(s) in "
                            f"{sum(result.seconds.values()):.1f}s", state="complete")
    st.session_state["result"] = result


def _boxed_image(result, rec) -> np.ndarray:
    cols = []
    for c in result.chars:
        hexc = LEVEL_COLOUR.get(c.level, MUTED).lstrip("#")
        cols.append(tuple(int(hexc[i: i + 2], 16) for i in (0, 2, 4)))
    labels = [c.label or "?" for c in result.chars]
    return segmentation.draw_boxes(result.segmentation, labels, cols)


def _reading(result, rec) -> None:
    roman = result.text(rec.label_info)
    deva = result.text(rec.label_info, use_devanagari=True)
    truth = st.session_state.get("truth")
    n_low = sum(c.level == "low" for c in result.chars)
    extra = ""
    if truth and len(truth) == len(result.chars):
        right = sum(t == c.label for t, c in zip(truth, result.chars))
        extra = (f'<div class="muted" style="margin-top:6px">Sample with known answer: '
                 f'<b>{right} of {len(truth)}</b> correct (true: {esc(" ".join(truth))})</div>')
    elif truth:
        extra = (f'<div class="muted" style="margin-top:6px">Known answer has {len(truth)} characters but '
                 f'{len(result.chars)} were detected, so a direct comparison is not possible.</div>')
    st.markdown(
        f'<div class="card"><div class="kicker">Assembled reading (reading order, lines separated by /)</div>'
        f'<div class="big-reading">{esc(roman)}</div><div class="deva">{esc(deva)}</div>'
        f'<div class="muted">Labels are the dataset\'s romanised class names; “?” marks low confidence. '
        f'Devanagari is shown only where the mapping is unambiguous and is not yet human-verified.'
        f'{" " + str(n_low) + " character(s) flagged low confidence." if n_low else ""}</div>{extra}</div>',
        unsafe_allow_html=True)


def _strip(result, rec) -> None:
    st.markdown('<div class="kicker">Step 2</div><h3 style="margin-top:0">Characters in reading order</h3>',
                unsafe_allow_html=True)
    chars = result.chars
    for start in range(0, len(chars), PER_ROW):
        cols = st.columns(PER_ROW)
        for col, k in zip(cols, range(start, min(start + PER_ROW, len(chars)))):
            c = chars[k]
            info = rec.label_info.get(c.label or "", {})
            sel = " sel" if st.session_state.get("sel") == k else ""
            flags = c.box.get("flags", [])
            with col:
                st.image(c.crop, width="stretch")
                st.markdown(
                    f'<div class="charcard{sel}"><div class="num">#{c.order}</div>'
                    f'<div class="lab">{esc(c.label or "—")}</div><div class="dv">{esc(info.get("devanagari") or "")}</div>'
                    f'{conf_bar(c.confidence, c.level)}<div class="muted">{100 * c.confidence:.0f}%</div>'
                    f'{level_badge(c.level)}{"<br><span class=\'badge flag\'>check box</span>" if flags else ""}</div>',
                    unsafe_allow_html=True)
                if st.button("Details", key=f"det_{k}", width="stretch"):
                    st.session_state["sel"] = k


def _heat_overlay(glyph: np.ndarray, grid: list[list[float]], size: int = 224) -> np.ndarray:
    """Blend a 4x4 SHAP grid over the glyph: orange = pushed toward the class, blue = pushed away."""
    g = np.asarray(grid, float)
    m = float(np.abs(g).max()) or 1.0
    up = cv2.resize(g / m, (size, size), interpolation=cv2.INTER_NEAREST)
    base = cv2.resize(255 - glyph, (size, size), interpolation=cv2.INTER_NEAREST).astype(float)
    rgb = np.stack([base] * 3, -1)
    pos, neg = np.clip(up, 0, 1)[..., None], np.clip(-up, 0, 1)[..., None]
    rgb = rgb * (1 - 0.55 * pos) + np.array([200, 129, 42]) * 0.55 * pos
    rgb = rgb * (1 - 0.55 * neg) + np.array([61, 90, 128]) * 0.55 * neg
    for i in range(1, len(g)):
        rgb[i * size // len(g), :, :] = 210
        rgb[:, i * size // len(g), :] = 210
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _shap_bars(items: list[dict], colour: str, title: str) -> go.Figure:
    items = items[:8][::-1]
    fig = go.Figure(go.Bar(x=[abs(i["shap"]) for i in items], y=[i["group"] for i in items], orientation="h",
                           marker_color=[FAMILY_COLOUR.get(i["family"], colour) for i in items],
                           hovertemplate="%{y}<br>moves probability by %{x:.3f}<extra></extra>"))
    plotly_layout(fig, height=60 + 34 * max(1, len(items)), title=title, legend=False)
    fig.update_xaxes(title="change in probability")
    return fig


def _detail(result, rec) -> None:
    k = st.session_state.get("sel", 0)
    if not result.chars or k >= len(result.chars):
        return
    c = result.chars[k]
    info = rec.label_info.get(c.label or "", {})
    st.markdown(f'<div class="kicker">Step 3</div><h3 style="margin-top:0">Character #{c.order}: why '
                f'“{esc(c.label)}”?</h3>', unsafe_allow_html=True)
    if c.error:
        st.warning(f"This character could not be processed: {c.error}")
        return
    flags = c.box.get("flags", [])
    if flags:
        st.warning("Segmentation note: " + "; ".join(segmentation.FLAG_TEXT[f] for f in flags if f in segmentation.FLAG_TEXT))
    a, b, cc = st.columns([1, 1, 1.6], gap="medium")
    with a:
        st.image(c.crop, caption="cropped from your image", width="stretch")
    with b:
        st.image(255 - c.processed, caption=f"model input ({config.PREPROCESS.image_size}×{config.PREPROCESS.image_size})",
                 width="stretch", clamp=True)
    with cc:
        top = c.top
        fig = go.Figure(go.Bar(x=[t["proba"] for t in top][::-1], y=[t["label"] for t in top][::-1], orientation="h",
                               marker_color=[INK if t["label"] == c.label else "#cbbd9c" for t in top][::-1],
                               text=[f"{100 * t['proba']:.1f}%" for t in top][::-1], textposition="outside",
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        plotly_layout(fig, height=210, title="Top-3 predicted classes", legend=False)
        fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        votes = ", ".join(f"{m.upper()}: {v}" for m, v in c.model_votes.items())
        st.markdown(f'<div class="muted">Individual models voted → {esc(votes)}. '
                    f'{esc(info.get("note", ""))}.</div>', unsafe_allow_html=True)

    cache = st.session_state.setdefault("explanations", {})
    if k not in cache:
        with st.spinner("Computing the SHAP explanation (35 feature groups, a few seconds)…"):
            try:
                cache[k] = rec.explain(result, k)
            except ModiError as exc:
                st.warning(f"Explanation unavailable: {exc}")
                return
            except Exception as exc:  # noqa: BLE001 - never crash the page over an explanation
                st.warning(f"Explanation failed ({type(exc).__name__}). Predictions above are unaffected.")
                return
    ex = cache[k]
    st.markdown(
        f'<div class="card"><div class="kicker">SHAP explanation</div>Compared with an average character, these '
        f'parts of the image moved the probability of <b>{esc(ex["predicted"])}</b> from '
        f'{100 * ex["baseline_proba_predicted"]:.1f}% to {100 * ex["predicted_proba"]:.1f}%. '
        f'The runner-up was <b>{esc(ex["runner_up"])}</b> ({100 * ex["runner_up_proba"]:.1f}%). '
        f'<span class="muted">Bars are coloured by descriptor: <b style="color:{INK}">HOG</b> stroke direction, '
        f'<b style="color:{SAFFRON}">Zernike</b> overall shape, <b style="color:{GREEN}">LBP</b> local stroke patterns.'
        f'</span></div>', unsafe_allow_html=True)
    l, m, r = st.columns([1.3, 1.3, 1], gap="medium")
    with l:
        st.plotly_chart(_shap_bars(ex["toward_predicted"], GREEN, f"Pushed toward “{ex['predicted']}”"),
                        width="stretch", config={"displayModeBar": False})
        if ex["against_predicted"]:
            st.plotly_chart(_shap_bars(ex["against_predicted"], RED, f"Pushed away from “{ex['predicted']}”"),
                            width="stretch", config={"displayModeBar": False})
    with m:
        st.plotly_chart(_shap_bars(ex["toward_runner_up"], SAFFRON, f"Would have pushed toward “{ex['runner_up']}”"),
                        width="stretch", config={"displayModeBar": False})
        fam = ex["family_contribution_predicted"]
        st.markdown("<div class='muted'>Net contribution by descriptor family: " +
                    ", ".join(f"{f.upper()} {v:+.3f}" for f, v in fam.items()) + "</div>", unsafe_allow_html=True)
    with r:
        st.image(_heat_overlay(c.processed, ex["region_maps_predicted"]["hog"]),
                 caption="Where HOG evidence came from (orange = for, blue = against)", width="stretch")
        st.image(_heat_overlay(c.processed, ex["region_maps_predicted"]["lbp"]),
                 caption="Where LBP evidence came from", width="stretch")


def _downloads(result, rec) -> None:
    records = rec.to_records(result)
    flat = pd.DataFrame([{"order": r["order"], "label": r["label"], "devanagari": r["devanagari"],
                          "confidence": r["confidence"], "confidence_level": r["confidence_level"],
                          "top2": r["top3"][1]["label"] if len(r["top3"]) > 1 else None,
                          "top3": r["top3"][2]["label"] if len(r["top3"]) > 2 else None,
                          "segmentation_flags": ";".join(r["segmentation_flags"]),
                          "x": r["box"]["x"], "y": r["box"]["y"], "w": r["box"]["w"], "h": r["box"]["h"],
                          "line": r["box"]["line"]} for r in records])
    payload = {"image": st.session_state.get("image_name"), "reading": result.text(rec.label_info),
               "segmentation_warnings": result.segmentation.warnings, "characters": records,
               "model": "stacking ensemble over HOG + Zernike + LBP (classical ML)"}
    a, b, _ = st.columns([1, 1, 3])
    stem = str(st.session_state.get("image_name", "result")).rsplit(".", 1)[0]
    a.download_button("⬇ Download CSV", flat.to_csv(index=False).encode("utf-8"), f"{stem}_modi.csv", "text/csv",
                      width="stretch")
    b.download_button("⬇ Download JSON", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                      f"{stem}_modi.json", "application/json", width="stretch")


def render() -> None:
    input_section()
    if st.session_state.get("image") is None:
        st.markdown('<div class="card" style="text-align:center;padding:28px"><div class="kicker">Ready</div>'
                    '<h4>Upload an image or pick a sample to begin</h4><div class="muted">Each character is found, '
                    'cleaned, described with three handcrafted descriptors and classified. Then you can ask '
                    '<i>why</i>.</div></div>', unsafe_allow_html=True)
        return
    ok, msg = data.model_available()
    if not ok:
        st.error(f"The trained model is not available yet. {msg}")
        return
    rec = data.recognizer()
    _run(rec)
    result = st.session_state.get("result")
    if result is None:
        return
    st.divider()
    for w in result.segmentation.warnings:
        st.warning(w)
    if not result.chars:
        st.info("No characters were found. Try a tighter crop, darker ink, or a higher-resolution scan.")
        return
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.image(_boxed_image(result, rec), caption="Detected characters, numbered in reading order "
                 "(green = confident, amber = unsure, red = low confidence)", width="stretch")
    with right:
        _reading(result, rec)
        _downloads(result, rec)
    _strip(result, rec)
    st.divider()
    _detail(result, rec)
