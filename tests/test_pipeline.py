"""Fast tests for the guarantees the project relies on. No dataset needed.

Run:  python -m pytest -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data_loader as dl, preprocess as pp, segmentation as sg  # noqa: E402
from src.features import fusion, hog_features, zernike_features  # noqa: E402
from src.utils import ModiError  # noqa: E402

SIZES = (72, 90, 250, 500, 1000, 2000)


def fake_manifest(n_per: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame([{"rel_path": f"c/{lab}/{lab}{i:05d}.jpg", "label": lab, "category": "x"}
                         for lab, n in n_per.items() for i in range(n)])


def glyph(size: int = 120, dark_ink: bool = True) -> np.ndarray:
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, 255, 4)
    cv2.line(img, (size // 4, size // 4), (3 * size // 4, size // 4), 255, 4)
    return 255 - img if dark_ink else img


# ---------------------------------------------------------------- sampling
class TestSampling:
    m = fake_manifest({"ka": 3000, "kha": 2500, "a": 2400})

    def test_deterministic_and_row_order_independent(self):
        a = dl.with_class_order(self.m)
        b = dl.with_class_order(self.m.sample(frac=1, random_state=7))
        assert a[["rel_path", "class_rank"]].equals(b[["rel_path", "class_rank"]])

    def test_same_order_in_fresh_process(self):
        code = ("import sys;sys.path.insert(0,r'%s');import pandas as pd;from src import data_loader as dl;"
                "m=pd.DataFrame([{'rel_path':f'c/ka/ka{i:05d}.jpg','label':'ka'} for i in range(3000)]);"
                "print(','.join(dl.with_class_order(m).rel_path[:5]))" % ROOT)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        assert out == ",".join(dl.with_class_order(self.m).query("label == 'ka'").rel_path[:5])

    def test_independent_of_other_classes(self):
        full = dl.with_class_order(self.m).query("label == 'ka'").rel_path.tolist()
        part = dl.with_class_order(self.m[self.m.label != "kha"]).query("label == 'ka'").rel_path.tolist()
        assert full == part

    def test_nested_disjoint_fixed_test(self):
        test, trains, rep = dl.learning_curve_splits(self.m, SIZES)
        sets = {n: set(t.rel_path) for n, t in trains.items()}
        assert all(sets[a] <= sets[b] for a, b in zip(SIZES, SIZES[1:]))
        assert all(not (s & set(test.rel_path)) for s in sets.values())
        assert (test.label.value_counts() == config.SAMPLING.test_per_class).all()

    def test_capped_pool_gives_same_split(self):
        pool, _ = dl.cap_per_class(self.m, 2200)
        a = dl.stratified_split(pool, 500, quiet=True)
        b = dl.stratified_split(self.m, 500, quiet=True)
        assert set(a[0].rel_path) == set(b[0].rel_path) and set(a[1].rel_path) == set(b[1].rel_path)


class TestShortClassPolicy:
    m = fake_manifest({"big": 2500, "mid": 700, "edge": 272, "small": 271, "tiny": 30})

    def test_test_sizes(self):
        plan, rep = dl.plan_splits(dl.with_class_order(self.m), SIZES, quiet=True)
        p = plan.set_index("label")
        assert p.loc["edge", "test_n"] == 200 and p.loc["small", "test_n"] == 54 and p.loc["tiny", "test_n"] == 6
        assert set(rep["reduced_test_classes"]) == {"small", "tiny"}

    def test_class_set_constant_and_short_flagged(self):
        _, trains, rep = dl.learning_curve_splits(self.m, SIZES)
        assert all(set(t.label) == set(self.m.label) for t in trains.values())
        assert rep["points"]["2000"]["complete"] is False
        assert rep["points"]["2000"]["short_classes"]["mid"] == 500


# ---------------------------------------------------------- preprocessing
class TestPreprocessing:
    def test_polarity_normalised(self):
        a = pp.preprocess_array(glyph(dark_ink=True)).final
        b = pp.preprocess_array(glyph(dark_ink=False)).final
        assert a.shape == (config.PREPROCESS.image_size,) * 2
        assert set(np.unique(a)) <= {0, 255} and (a > 0).mean() < 0.5
        assert np.array_equal(a, b)

    def test_frame_line_removed(self):
        g = glyph(dark_ink=True)
        framed = g.copy(); framed[1:3, 5:-5] = 0                 # scan-frame fragment on the top edge
        assert np.array_equal(pp.preprocess_array(g).final, pp.preprocess_array(framed).final)

    def test_grid_line_removed_but_real_stroke_kept(self):
        g = glyph(dark_ink=True)                                  # glyph spans rows ~25-95 of 120
        grid = g.copy(); grid[2:118, 104:106] = 0                 # full-height thin line beside the glyph
        assert np.array_equal(pp.preprocess_array(g).final, pp.preprocess_array(grid).final)
        stroke = g.copy(); stroke[40:80, 104:106] = 0             # short detached stroke inside the glyph's height
        assert not np.array_equal(pp.preprocess_array(g).final, pp.preprocess_array(stroke).final)

    def test_blank_rejected(self):
        noise = np.clip(np.random.default_rng(0).normal(230, 3, (120, 120)), 0, 255).astype(np.uint8)
        with pytest.raises(ModiError):
            pp.preprocess_array(noise)


# ---------------------------------------------------------------- features
class TestFeatures:
    img = pp.preprocess_array(glyph()).final

    def test_dimensions_match_metadata(self):
        for fam in config.DESCRIPTORS:
            v = fusion.EXTRACTORS[fam].extract(self.img)
            assert v.shape == (fusion.n_features(fam),) == (len(fusion.EXTRACTORS[fam].feature_meta()),)
            assert np.isfinite(v).all()
        assert fusion.extract_one(self.img).shape == (2261,)

    def test_zernike_rotation_invariant_hog_not(self):
        r = np.rot90(self.img).copy()
        z = np.linalg.norm(zernike_features.extract(self.img) - zernike_features.extract(r))
        h = np.linalg.norm(hog_features.extract(self.img) - hog_features.extract(r))
        assert z < 1e-6 < h

    def test_family_balancer_equal_variance(self):
        w = fusion.FamilyBalancer().fit(np.zeros((2, 2261))).weights_
        shares = [float((w[s] ** 2).sum()) for s in fusion.slices().values()]
        assert max(shares) - min(shares) < 1e-3

    def test_shap_groups_partition_features(self):
        from src.explain import build_groups

        cols = np.concatenate([g["columns"] for g in build_groups()])
        assert len(build_groups()) == 35 and np.array_equal(np.sort(cols), np.arange(2261))


# ------------------------------------------------------------ segmentation
class TestSegmentation:
    def test_separate_characters_in_reading_order(self):
        canvas = np.full((200, 700), 235, np.uint8)
        for k, x in enumerate((30, 250, 470)):
            canvas[40:160, x:x + 120] = np.minimum(canvas[40:160, x:x + 120], glyph(120))
        r = sg.segment(canvas)
        assert len(r.boxes) == 3 and [b.order for b in r.boxes] == [1, 2, 3]
        assert [b.x for b in r.boxes] == sorted(b.x for b in r.boxes)

    def test_blank_image_warns(self):
        r = sg.segment(np.full((100, 300), 240, np.uint8))
        assert not r.boxes and r.warnings

    def test_bundled_samples_match_ground_truth(self):
        import json

        meta_path = config.PATHS.app_assets / "samples.json"
        if not meta_path.exists():
            pytest.skip("samples not generated")
        for s in json.loads(meta_path.read_text(encoding="utf-8"))["samples"]:
            if s["ground_truth"]:
                r = sg.segment(pp.read_image(config.PATHS.app_assets / s["file"]))
                assert len(r.boxes) == len(s["ground_truth"]), s["file"]


# ------------------------------------------------------- re-segmentation
class TestResegmentation:
    @staticmethod
    def ink_aspect(crop):
        ys, xs = np.nonzero(crop < 128)
        return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)

    def test_touching_pair_is_split_in_two(self):
        from src import resegment

        canvas = np.full((200, 400), 235, np.uint8)
        for x in (60, 120):                                        # ink of the two glyphs touches at x = 150
            canvas[40:160, x:x + 120] = np.minimum(canvas[40:160, x:x + 120], glyph(120))
        canvas[66:70, 90:210] = 0                                  # shared headline joins the word
        r = sg.segment(canvas)
        x0 = min(b.x for b in r.boxes); x1 = max(b.x + b.w for b in r.boxes)
        y0 = min(b.y for b in r.boxes); y1 = max(b.y + b.h for b in r.boxes)
        r.boxes = [sg.CharBox(x0, y0, x1 - x0, y1 - y0, 0, 1, ["wide"], 0.5)]   # the joined pair as one box
        r.crops = [sg._to_crop(r.binary[y0:y1, x0:x1])]

        def fake_classifier(crops):                               # confident only for character-shaped ink
            return np.array([0.9 if 0.75 <= self.ink_aspect(c) <= 1.3 else 0.05 for c in crops])

        n = resegment.apply(r, fake_classifier)
        assert n >= 1 and len(r.boxes) == 2
        assert all("resegmented" in b.flags for b in r.boxes)
        assert abs(r.boxes[0].x + r.boxes[0].w - 150) <= 12        # the cut lands where the glyphs meet

    def test_single_wide_character_not_forced_apart(self):
        from src import resegment

        r = sg.segment(np.minimum(np.full((200, 300), 235, np.uint8),
                                  np.pad(glyph(120), ((40, 40), (90, 90)), constant_values=255)))
        r.boxes[0].flags.append("wide")
        n = resegment.apply(r, lambda crops: np.array([0.95 if c.shape[1] > 100 else 0.1 for c in crops]))
        assert n == 0 and len(r.boxes) == 1
