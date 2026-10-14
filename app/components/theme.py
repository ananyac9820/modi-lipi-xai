"""Visual theme: aged-manuscript palette, custom CSS and small HTML helpers."""

from __future__ import annotations

import base64
import html
import io

import numpy as np
from PIL import Image

PAPER = "#fbf7ef"
PAPER_DEEP = "#f3ead7"
INK = "#1f3a5f"
INK_SOFT = "#3d5a80"
SAFFRON = "#c8812a"
GREEN = "#4f8a5b"
AMBER = "#d49a2a"
RED = "#b5523b"
MUTED = "#6b6a63"
LEVEL_COLOUR = {"high": GREEN, "medium": AMBER, "low": RED}
FAMILY_COLOUR = {"hog": INK, "zernike": SAFFRON, "lbp": GREEN}
CATEGORY_COLOUR = {"digit": SAFFRON, "vowel": GREEN, "consonant": INK, "other": "#8c4a5b"}
MODEL_COLOUR = {"rf": GREEN, "svm": "#8c4a5b", "xgb": SAFFRON, "stack": INK}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=Noto+Sans+Modi&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, button {{
  font-family: 'Inter', 'Noto Sans Devanagari', 'Noto Sans Modi', sans-serif;
}}
h1, h2, h3, .serif {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
.stApp {{
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(200,129,42,0.08), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(31,58,95,0.06), transparent 60%),
    {PAPER};
}}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }}

/* hero */
.hero {{
  border-radius: 22px; padding: 26px 30px; margin-bottom: 18px;
  background: linear-gradient(135deg, {INK} 0%, #2b4c78 60%, #3a5f8f 100%);
  color: #fdf8ee; position: relative; overflow: hidden;
  box-shadow: 0 10px 30px rgba(31,58,95,0.18);
}}
.hero h1 {{ color: #fdf8ee !important; margin: 0 0 6px 0; font-size: 2.1rem; }}
.hero p {{ color: #e9e1cf; margin: 0; max-width: 780px; line-height: 1.55; }}
.hero .modi {{ position: absolute; right: 26px; top: 6px; font-family: 'Noto Sans Modi', serif;
  font-size: 6.2rem; color: rgba(253,248,238,0.13); line-height: 1; pointer-events: none; }}
.chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ background: rgba(253,248,238,0.14); border: 1px solid rgba(253,248,238,0.25); color: #fdf8ee;
  border-radius: 999px; padding: 4px 12px; font-size: 0.82rem; }}

/* cards */
.card {{ background: #fffdf8; border: 1px solid #e8dcc2; border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 2px 10px rgba(60,45,20,0.05); margin-bottom: 12px; }}
.card h4 {{ margin: 0 0 6px 0; color: {INK}; font-family: 'Fraunces', serif; }}
.kicker {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; color: {SAFFRON}; font-weight: 600; }}
.muted {{ color: {MUTED}; font-size: 0.88rem; }}
.big-reading {{ font-family: 'Fraunces', serif; font-size: 1.7rem; color: {INK}; line-height: 1.3; word-break: break-word; }}
.deva {{ font-family: 'Noto Sans Devanagari', serif; font-size: 1.5rem; color: {INK_SOFT}; }}
.metric {{ text-align: center; padding: 10px 6px; }}
.metric .v {{ font-family: 'Fraunces', serif; font-size: 1.8rem; color: {INK}; }}
.metric .l {{ font-size: 0.78rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}

/* character cards */
.charcard {{ background: #fffdf8; border: 1px solid #e8dcc2; border-radius: 14px; padding: 10px; text-align: center; }}
.charcard.sel {{ border: 2px solid {SAFFRON}; box-shadow: 0 0 0 3px rgba(200,129,42,0.15); }}
.charcard .num {{ font-size: 0.72rem; color: {MUTED}; }}
.charcard .lab {{ font-family: 'Fraunces', serif; font-size: 1.15rem; color: {INK}; margin-top: 2px; }}
.charcard .dv {{ font-family: 'Noto Sans Devanagari', serif; color: {INK_SOFT}; font-size: 1.0rem; min-height: 1.3em; }}
.bar {{ height: 8px; border-radius: 6px; background: #eadfc8; overflow: hidden; margin: 6px 0 2px 0; }}
.bar > span {{ display: block; height: 100%; border-radius: 6px; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 1px 8px; font-size: 0.72rem; font-weight: 600; }}
.flag {{ background: #f7e3dc; color: {RED}; }}
.okb {{ background: #e3efe5; color: {GREEN}; }}
.amb {{ background: #f8ecd2; color: #9a6a14; }}

/* pipeline */
.flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 6px 0 12px 0; }}
.flow .st {{ border-radius: 12px; padding: 8px 12px; background: #fffdf8; border: 1px solid #e3d5b8; font-size: 0.85rem; color: {INK}; }}
.flow .st.on {{ background: {INK}; color: #fdf8ee; border-color: {INK}; box-shadow: 0 4px 12px rgba(31,58,95,0.25); }}
.flow .ar {{ color: {SAFFRON}; font-weight: 700; }}
.flow .grp {{ display: flex; flex-direction: column; gap: 4px; }}

/* streamlit widgets */
div[data-testid="stFileUploaderDropzone"] {{ background: #fffdf8; border: 2px dashed #d9c49b; border-radius: 16px; }}
.stButton > button {{ border-radius: 10px; border: 1px solid #d9c49b; background: #fffdf8; color: {INK}; }}
.stButton > button:hover {{ border-color: {SAFFRON}; color: {SAFFRON}; }}
.stButton > button[kind="primary"] {{ background: {INK}; color: #fdf8ee; border: none; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background: #f3ead7; border-radius: 12px 12px 0 0; padding: 8px 18px; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ background: {INK} !important; color: #fdf8ee !important; }}
div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
@media (max-width: 760px) {{
  .hero h1 {{ font-size: 1.5rem; }} .hero .modi {{ display: none; }} .big-reading {{ font-size: 1.3rem; }}
}}
</style>
"""


def esc(text: object) -> str:
    """HTML-escape any value."""
    return html.escape(str(text))


def conf_bar(p: float, level: str) -> str:
    """Small coloured confidence bar."""
    pct = max(0.0, min(1.0, p)) * 100
    return f'<div class="bar"><span style="width:{pct:.0f}%;background:{LEVEL_COLOUR.get(level, MUTED)}"></span></div>'


def level_badge(level: str) -> str:
    cls = {"high": "okb", "medium": "amb", "low": "flag"}.get(level, "amb")
    text = {"high": "confident", "medium": "unsure", "low": "low confidence"}.get(level, level)
    return f'<span class="badge {cls}">{esc(text)}</span>'


def metric(value: str, label: str) -> str:
    return f'<div class="card metric"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def to_data_uri(img: np.ndarray) -> str:
    """Embed a numpy image in HTML."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def plotly_layout(fig, height: int = 380, title: str | None = None, legend: bool = True):
    """Apply the manuscript look to a plotly figure."""
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(family="Fraunces, serif", size=16, color=INK)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fffdf8",
        font=dict(family="Inter, 'Noto Sans Devanagari', sans-serif", color="#1d2733", size=12),
        margin=dict(l=10, r=10, t=50 if title else 16, b=10), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor="#fffdf8", font_family="Inter"),
    )
    fig.update_xaxes(gridcolor="#efe5cf", zerolinecolor="#e3d5b8")
    fig.update_yaxes(gridcolor="#efe5cf", zerolinecolor="#e3d5b8")
    return fig
