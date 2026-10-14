"""Modi Lipi XAI - Streamlit web app.

Run from the project root:
    streamlit run app/app.py

Tab 1 (Recognise): upload or pick a sample, see every character found, its
prediction, confidence and a SHAP explanation.
Tab 2 (Pipeline Dashboard): what every stage did to the current image, plus
dataset, model, ablation, learning-curve, confusion, SHAP and literature results.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
# This script is app/app.py, so with app/ on sys.path "import app" would find the
# script instead of the app package. Keep only the project root on the path.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != APP_DIR]
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Modi Lipi XAI", page_icon="📜", layout="wide", initial_sidebar_state="collapsed")

from app.components import dashboard, data, recognise  # noqa: E402
from app.components.theme import CSS, esc  # noqa: E402


def hero() -> None:
    s = data.read_json("dataset_summary.json") or {}
    mc = data.read_json("model_comparison.json") or {}
    chips = ["Classical ML, no neural network", "HOG + Zernike + LBP", "RF + SVM + XGBoost stacking", "SHAP explanations"]
    if s:
        chips.insert(0, f"{s.get('num_classes', '?')} characters · {s.get('total_images', 0):,} images")
    if mc:
        chips.append(f"test accuracy {100 * mc['models']['stack']['accuracy']:.1f}%")
    st.markdown(
        '<div class="hero"><div class="modi">&#x1160E;&#x11626;</div>'
        '<h1>Modi Lipi XAI</h1>'
        '<p><b>Modi</b> (मोडी) is the cursive script used to write Marathi for about 700 years: land records, court '
        'papers and letters across Maharashtra. Almost nobody can read it today, so millions of archived documents '
        'are effectively locked. This app recognises handwritten Modi characters with <b>handcrafted features and '
        'classical machine learning</b>, and shows <b>why</b> it made every prediction.</p>'
        '<div class="chips">' + "".join(f'<span class="chip">{esc(c)}</span>' for c in chips) + "</div></div>",
        unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    hero()
    tab1, tab2 = st.tabs(["🔍  Recognise", "🧭  Pipeline Dashboard"])
    with tab1:
        recognise.render()
    with tab2:
        dashboard.render()
    st.markdown('<div class="muted" style="text-align:center;margin-top:30px">Modi Lipi XAI · semester data-science '
                'project · results are reproducible with seed 42 · see README.md</div>', unsafe_allow_html=True)


main()
