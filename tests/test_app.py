"""Headless end-to-end test of the Streamlit app (needs a trained model).

Run:  python -m pytest -q tests/test_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

pytestmark = pytest.mark.skipif(not (config.PATHS.models / "main_stack.joblib").exists(),
                                reason="trained model not available")


def test_app_recognises_sample_and_explains():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    img = np.array(Image.open(config.PATHS.app_assets / "sample_3_digits.png").convert("RGB"))
    at.session_state["image"] = img
    at.session_state["image_name"] = "sample_3_digits.png"
    at.session_state["image_key"] = "test-digits"
    at.session_state["truth"] = ["one", "nine", "four", "seven", "two", "zero"]
    at.session_state["sel"] = 0
    at.session_state["result"] = None
    at.session_state["explanations"] = {}
    at.run()
    assert not at.exception, at.exception
    result = at.session_state["result"]
    assert result is not None and len(result.chars) == 6
    assert 0 in at.session_state["explanations"]           # SHAP for the selected character rendered
    text = " ".join(m.value for m in at.markdown)
    assert "Assembled reading" in text and "correct" in text
    # Both tabs render, including every dashboard section, without an exception.
    assert any("Dataset (measured, not claimed)" in m.value for m in at.markdown)
    assert not at.warning or all("could not be drawn" not in w.value for w in at.warning)
