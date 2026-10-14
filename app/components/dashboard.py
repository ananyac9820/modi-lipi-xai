"""Part 2: the Pipeline Dashboard (the whole project on one screen)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import data
from app.components.theme import (CATEGORY_COLOUR, FAMILY_COLOUR, GREEN, INK, MODEL_COLOUR, MUTED, SAFFRON, esc,
                                  metric, plotly_layout)
from src import config

STAGES = ["Upload", "Preprocess", "Segment", "HOG", "Zernike", "LBP", "Fusion", "PCA / LDA", "Ensemble",
          "Prediction", "SHAP"]
DISPLAY = {"rf": "Random Forest", "svm": "SVM (RBF)", "xgb": "XGBoost", "stack": "Stacking ensemble"}
FS_LABEL = {"hog": "HOG only", "zernike": "Zernike only", "lbp": "LBP only", "fused_plain": "Fused (plain concat)",
            "fused": "Fused (family-balanced)"}
CHART = {"displayModeBar": False}


def _missing(what: str, cmd: str) -> None:
    st.info(f"{what} not available yet. Run `{cmd}`.")


# ---------------------------------------------------------------------------
# Pipeline view
# ---------------------------------------------------------------------------
def _flow_html(active: str) -> str:
    parts = []
    for s in STAGES:
        if s == "HOG":
            parts.append('<div class="grp">')
        cls = "st on" if s == active else "st"
        parts.append(f'<div class="{cls}">{esc(s)}</div>')
        if s == "LBP":
            parts.append("</div>")
        if s not in ("HOG", "Zernike", "LBP", "SHAP"):
            parts.append('<div class="ar">→</div>')
        elif s == "LBP":
            parts.append('<div class="ar">→</div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def pipeline_view() -> None:
    st.markdown('<h3>Pipeline: follow your image through every stage</h3>', unsafe_allow_html=True)
    result = st.session_state.get("result")
    stage = st.radio("Stage", STAGES, horizontal=True, label_visibility="collapsed", key="stage_pick")
    st.markdown(_flow_html(stage), unsafe_allow_html=True)
    if result is None or not result.chars:
        st.info("Recognise an image on the first tab (or pick a sample there); this view then shows what each "
                "stage actually did to it.")
        return
    rec = data.recognizer()
    chars = [c for c in result.chars if c.processed is not None]
    if not chars:
        st.warning("No character in the current image could be processed.")
        return
    default = min(st.session_state.get("sel", 0), len(chars) - 1)
    k = st.selectbox("Character to follow", range(len(chars)), index=default,
                     format_func=lambda i: f"#{chars[i].order}: {chars[i].label}")
    c = chars[k]
    seg = result.segmentation
    from src.features import fusion, hog_features, lbp_features, zernike_features

    if stage == "Upload":
        st.image(st.session_state["image"], caption=f"Uploaded image: {st.session_state.get('image_name')}",
                 width="stretch")
    elif stage == "Preprocess":
        cols = st.columns(4)
        for col, (key, cap) in zip(cols, [("original", "grayscale crop"), ("binary", "Otsu binarisation (ink = white)"),
                                          ("cleaned", "frame lines + specks removed"), ("final", "centred, 64×64")]):
            if key in c.steps:
                col.image(c.steps[key], caption=cap, width="stretch", clamp=True)
        st.caption("Polarity is normalised so ink is always white on black. Scan-frame fragments along the edges "
                   "and isolated specks are removed before the glyph is centred and resized.")
    elif stage == "Segment":
        a, b = st.columns(2)
        a.image(seg.steps.get("binary", seg.binary), caption=f"binarised page (skew corrected by {seg.skew_deg:+.1f}°)",
                width="stretch", clamp=True)
        from src import segmentation
        b.image(segmentation.draw_boxes(seg), caption=f"{len(seg.lines)} line(s), {len(seg.boxes)} character(s)",
                width="stretch")
        st.caption("Lines come from the horizontal projection profile, characters from blank columns. Over-wide "
                   "blobs are split after removing the headline (shirorekha); fragments are merged back.")
    elif stage == "HOG":
        a, b = st.columns([1, 2])
        a.image(hog_features.visualize(c.processed), caption="HOG cells: dominant stroke directions",
                width="stretch", clamp=True)
        v = hog_features.extract(c.processed)
        fig = go.Figure(go.Scatter(y=v, mode="lines", line=dict(color=INK, width=1)))
        plotly_layout(fig, 260, f"HOG vector ({v.size} values)", legend=False)
        b.plotly_chart(fig, width="stretch", config=CHART)
    elif stage == "Zernike":
        v = zernike_features.extract(c.processed)
        names = [m["name"].replace("zernike_", "") for m in zernike_features.feature_meta()]
        fig = go.Figure(go.Bar(x=names, y=v, marker_color=SAFFRON))
        plotly_layout(fig, 300, f"Zernike moment magnitudes ({v.size}), low order = coarse shape", legend=False)
        st.plotly_chart(fig, width="stretch", config=CHART)
        st.caption("These magnitudes do not change when the character is rotated: global shape, independent of "
                   "writing angle.")
    elif stage == "LBP":
        a, b = st.columns([1, 2])
        codes = lbp_features.codes(c.processed)
        a.image((codes / max(codes.max(), 1) * 255).astype(np.uint8), caption="LBP code map (radius 1)",
                width="stretch")
        v = lbp_features.extract(c.processed)
        fig = go.Figure(go.Bar(y=v, marker_color=GREEN))
        plotly_layout(fig, 260, f"LBP histograms: 4×4 cells × 2 scales ({v.size} values)", legend=False)
        b.plotly_chart(fig, width="stretch", config=CHART)
    elif stage == "Fusion":
        bal = rec.bundle.preprocessor[:2].transform(c.features[None, :])[0]
        sl = fusion.slices(rec.bundle.feature_set)
        fig = go.Figure()
        for fam, s in sl.items():
            fig.add_trace(go.Scatter(x=np.arange(s.start, s.stop), y=bal[s], mode="lines", name=fam.upper(),
                                     line=dict(color=FAMILY_COLOUR[fam], width=1)))
        plotly_layout(fig, 300, f"Fused vector after standardising + family balancing ({c.features.size} values)")
        st.plotly_chart(fig, width="stretch", config=CHART)
        st.caption("Each family is scaled by 1/√(its size) so HOG's 1,764 values cannot outvote Zernike's 49.")
    elif stage == "PCA / LDA":
        z = rec.bundle.transform(c.features[None, :])[0]
        sample = data.reduced_training_sample()
        fig = go.Figure()
        if sample is not None:
            for cat, col in CATEGORY_COLOUR.items():
                m = sample["category"] == cat
                if m.any():
                    fig.add_trace(go.Scattergl(x=sample["Z"][m, 0], y=sample["Z"][m, 1], mode="markers", name=cat,
                                               marker=dict(size=4, color=col, opacity=0.35),
                                               text=sample["label"][m], hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[z[0]], y=[z[1]], mode="markers", name="this character",
                                 marker=dict(size=16, color="#d13b2a", symbol="star", line=dict(color="white", width=1))))
        info = rec.bundle.info
        plotly_layout(fig, 420, f"{rec.bundle.reducer.upper()}: {info['n_features_in']} → {info['n_features_reduced']} "
                                f"dimensions (first two shown)")
        st.plotly_chart(fig, width="stretch", config=CHART)
    elif stage == "Ensemble":
        Z = rec.bundle.transform(c.features[None, :])
        rows = []
        for name, est in rec.bundle.stack.named_estimators_.items():
            pred = rec.bundle.classes[int(est.predict(Z)[0])]
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(Z)[0]
                rows.append({"model": DISPLAY[name], "vote": pred, "its confidence": f"{100 * p.max():.1f}%",
                             "output passed to the meta-learner": "class probabilities"})
            else:
                rows.append({"model": DISPLAY[name], "vote": pred, "its confidence": "n/a (margin-based)",
                             "output passed to the meta-learner": "decision scores"})
        rows.append({"model": "Stacking (logistic regression)", "vote": c.label,
                     "its confidence": f"{100 * c.confidence:.1f}%", "output passed to the meta-learner": "final"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The meta-learner was trained on out-of-fold outputs of the three base models, so it learns "
                   "which model to trust for which kind of character.")
    elif stage == "Prediction":
        st.dataframe(pd.DataFrame([{"rank": i + 1, "class": t["label"], "probability": f"{100 * t['proba']:.2f}%",
                                    "Devanagari": rec.label_info.get(t["label"], {}).get("devanagari") or "—"}
                                   for i, t in enumerate(c.top)]), hide_index=True, width="stretch")
    elif stage == "SHAP":
        ex = st.session_state.get("explanations", {}).get(result.chars.index(c))
        if ex is None:
            if st.button("Compute SHAP for this character"):
                with st.spinner("Computing…"):
                    st.session_state.setdefault("explanations", {})[result.chars.index(c)] = rec.explain(
                        result, result.chars.index(c))
                st.rerun()
            st.caption("Open the character's Details on the Recognise tab, or compute it here.")
        else:
            vals = np.array(ex["shap_predicted"])
            order = np.argsort(-np.abs(vals))[:12][::-1]
            fig = go.Figure(go.Bar(x=vals[order], y=[ex["group_names"][i] for i in order], orientation="h",
                                   marker_color=[GREEN if vals[i] > 0 else "#b5523b" for i in order]))
            plotly_layout(fig, 420, f"SHAP values for “{ex['predicted']}” (green = for, red = against)", legend=False)
            st.plotly_chart(fig, width="stretch", config=CHART)


# ---------------------------------------------------------------------------
# Results sections
# ---------------------------------------------------------------------------
def dataset_section() -> None:
    s = data.read_json("dataset_summary.json")
    if not s:
        _missing("Dataset summary", "python scripts/run_preprocessing.py")
        return
    ipc = s["images_per_class"]
    cols = st.columns(5)
    pp = s.get("preprocessing", {})
    for col, (v, l) in zip(cols, [(f"{s['total_images']:,}", "images on disk"), (s["num_classes"], "classes"),
                                  (f"{ipc['min']:,}–{ipc['max']:,}", "images per class"),
                                  (f"{pp.get('processed_ok', 0):,}", "preprocessed (pool)"),
                                  (s["skipped_total"] + pp.get("failed", 0), "files skipped / rejected")]):
        col.markdown(metric(str(v), l), unsafe_allow_html=True)
    df = pd.DataFrame(s["classes"]).sort_values("count", ascending=False)
    fig = go.Figure(go.Bar(x=df["label"], y=df["count"], marker_color=[CATEGORY_COLOUR.get(c, MUTED) for c in df["category"]],
                           customdata=df["category"], hovertemplate="%{x} (%{customdata}): %{y:,} images<extra></extra>"))
    plotly_layout(fig, 320, "Class balance (sorted; colour = category)", legend=False)
    fig.update_yaxes(range=[ipc["min"] * 0.995, ipc["max"] * 1.002])
    st.plotly_chart(fig, width="stretch", config=CHART)
    cvm = s.get("claims_vs_measured", {})
    st.caption("Counts are measured from disk. y-axis is zoomed: classes are almost perfectly balanced. "
               + " ".join(cvm.get("differences", [])[:3]))


def models_section() -> None:
    mc = data.read_json("model_comparison.json")
    if not mc:
        _missing("Model results", "python scripts/run_training.py && python scripts/run_evaluation.py")
        return
    cv = (data.read_json("cross_validation.json") or {}).get("summary", {})
    rows = []
    for m, s in mc["models"].items():
        rows.append({"model": DISPLAY.get(m, m), "accuracy": s["accuracy"], "macro F1": s["macro_f1"],
                     "macro precision": s["macro_precision"], "macro recall": s["macro_recall"],
                     "CV accuracy (train set)": f"{100 * cv[m]['mean']:.2f} ± {100 * cv[m]['std']:.2f}%" if m in cv else "—",
                     "train time (s)": mc.get("fit_seconds", {}).get("stack_total" if m == "stack" else m),
                     "ms / image": mc.get("inference_ms_per_image", {}).get(m)})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"accuracy": "{:.2%}", "macro F1": "{:.4f}", "macro precision": "{:.4f}",
                                  "macro recall": "{:.4f}"}).highlight_max(subset=["accuracy", "macro F1"], color="#e3efe5"),
                 hide_index=True, width="stretch")
    st.caption(f"Fixed test set: {mc.get('n_test', '?')} images ({config.SAMPLING.test_per_class} per class). "
               f"Training: {mc.get('train_per_class')} images per class. Per-category accuracy is in the tables below.")
    cat = pd.DataFrame({DISPLAY.get(m, m): s["per_category_accuracy"] for m, s in mc["models"].items()}).T
    st.dataframe(cat.style.format("{:.2%}"), width="stretch")


def ablation_section() -> None:
    ab = data.read_json("ablation.json")
    if not ab:
        _missing("Ablation results", "python scripts/run_training.py --only ablation")
        return
    fs = [k for k in FS_LABEL if k in ab["feature_sets"]]
    fig = go.Figure()
    for m in ("rf", "svm", "xgb", "stack"):
        if all(m in ab["feature_sets"][k]["models"] for k in fs):
            fig.add_trace(go.Bar(name=DISPLAY[m], x=[FS_LABEL[k] for k in fs], marker_color=MODEL_COLOUR[m],
                                 y=[100 * ab["feature_sets"][k]["models"][m]["accuracy"] for k in fs],
                                 hovertemplate="%{x}<br>" + DISPLAY[m] + ": %{y:.2f}%<extra></extra>"))
    plotly_layout(fig, 380, "Ablation: each descriptor alone vs fused (test accuracy %)")
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, width="stretch", config=CHART)


def curve_section() -> None:
    from src import learning_curve as lc

    cl, cnn = data.read_json("learning_curve_classical.json"), data.read_json("learning_curve_cnn.json")
    if not cl and not cnn:
        _missing("Learning curve", "python scripts/run_training.py --only curve")
        return
    fig = go.Figure()
    for d, col, sym in ((cl, INK, "circle"), (cnn, SAFFRON, "square")):
        if d and d.get("points"):
            pts = d["points"]
            fig.add_trace(go.Scatter(x=[p["train_per_class"] for p in pts], y=[100 * p["accuracy"] for p in pts],
                                     mode="lines+markers", name=d.get("label", d.get("model")),
                                     line=dict(color=col, width=3), marker=dict(size=9, symbol=sym),
                                     customdata=[[p.get("train_time_s") or 0] for p in pts],
                                     hovertemplate="%{x}/class: %{y:.2f}%<br>train %{customdata[0]:.0f}s<extra></extra>"))
    if cl and cl.get("points") and all("models" in p for p in cl["points"]):
        for m in ("rf", "svm", "xgb"):
            fig.add_trace(go.Scatter(x=[p["train_per_class"] for p in cl["points"]],
                                     y=[100 * p["models"][m]["accuracy"] for p in cl["points"]], mode="lines",
                                     name=DISPLAY[m], line=dict(color=MODEL_COLOUR[m], width=1, dash="dot")))
    fig.add_vline(x=lc.PAPERS_EFFECTIVE_TRAIN, line_dash="dot", line_color=MUTED,
                  annotation_text="≈ papers' training size (sample size only)", annotation_font_size=10)
    plotly_layout(fig, 420, "Accuracy vs training images per class (same fixed test set)")
    fig.update_xaxes(type="log", tickvals=list(config.SAMPLING.curve_train_sizes), title="training images per class")
    fig.update_yaxes(title="test accuracy (%)")
    st.plotly_chart(fig, width="stretch", config=CHART)
    if not cnn:
        st.caption("The untuned reference CNN curve appears here once `python scripts/run_cnn_baseline.py` has run.")


def confusion_section() -> None:
    cm = data.read_json("confusion_matrix.json")
    pairs = data.read_json("most_confused_pairs.json")
    if not cm:
        _missing("Confusion matrix", "python scripts/run_evaluation.py")
        return
    M = np.array(cm["matrix"]); cls = cm["classes"]
    norm = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=norm, x=cls, y=cls, customdata=M, colorscale="YlGnBu", zmin=0, zmax=1,
                               hovertemplate="true %{y} → predicted %{x}<br>%{customdata} images (%{z:.1%})<extra></extra>"))
    plotly_layout(fig, 720, "Confusion matrix, stacking ensemble (hover for counts)", legend=False)
    fig.update_yaxes(autorange="reversed", title="true class"); fig.update_xaxes(title="predicted class")
    st.plotly_chart(fig, width="stretch", config=CHART)
    if pairs:
        st.markdown("<h4>Most confused character pairs</h4>", unsafe_allow_html=True)
        df = pd.DataFrame(pairs["pairs"]).head(15)
        df = df.rename(columns={"class_a": "A", "class_b": "B", "a_as_b": "A read as B", "b_as_a": "B read as A",
                                "total_confusions": "total", "pair_error_rate": "pair error rate"})
        st.dataframe(df[["A", "B", "A read as B", "B read as A", "total", "pair error rate"]]
                     .style.format({"pair error rate": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption(f"{pairs['total_errors']} errors out of {pairs['n_test']} test images. No surveyed paper "
                   "publishes this table.")
    fig_path = config.PATHS.figures / "shap_confused_pairs.png"
    if fig_path.exists():
        st.markdown("<h4>What tells look-alike characters apart? (contrastive SHAP)</h4>", unsafe_allow_html=True)
        st.image(str(fig_path), width="stretch")
        st.caption("For each confused pair: SHAP for the true class minus SHAP for its look-alike, averaged over "
                   "test images of both. Red regions carry the evidence that separates the two characters.")


def shap_section() -> None:
    g = data.read_json("global_importance.json", "shap")
    if not g:
        _missing("Global SHAP importance", "python scripts/run_evaluation.py")
        return
    a, b = st.columns([1, 2])
    fams = list(g["family_share"])
    fig = go.Figure(go.Pie(labels=[f.upper() for f in fams], values=[g["family_share"][f] for f in fams], hole=0.55,
                           marker_colors=[FAMILY_COLOUR[f] for f in fams], textinfo="label+percent"))
    plotly_layout(fig, 320, "Share of importance by descriptor", legend=False)
    a.plotly_chart(fig, width="stretch", config=CHART)
    top = g["groups"][:12][::-1]
    fig = go.Figure(go.Bar(x=[t["mean_abs_shap"] for t in top], y=[t["group"] for t in top], orientation="h",
                           marker_color=[FAMILY_COLOUR[t["family"]] for t in top]))
    plotly_layout(fig, 320, "Most influential feature groups (mean |SHAP|)", legend=False)
    b.plotly_chart(fig, width="stretch", config=CHART)
    ft = g.get("faithfulness_deletion_test", {})
    if ft:
        st.caption(f"Sanity check: hiding the {ft['groups_removed']} groups SHAP ranks highest lowers the predicted "
                   f"probability by {ft['mean_probability_drop_top_shap']:.3f} on average, vs "
                   f"{ft['mean_probability_drop_random']:.3f} for {ft['groups_removed']} random groups "
                   f"({g['n_images']} test images).")


def literature_section() -> None:
    lit = data.read_json("literature.json", "assets")
    mc = data.read_json("model_comparison.json")
    rows = []
    if mc:
        best = mc["models"]["stack"]
        rows.append({"work": "THIS PROJECT: HOG + Zernike + LBP, stacking (RF + SVM + XGBoost)",
                     "approach": "Classical ML + SHAP", "dataset": f"MODI-HChar (public), 57 classes, "
                     f"{mc.get('train_per_class')} train / {config.SAMPLING.test_per_class} test per class",
                     "reported accuracy": f"{100 * best['accuracy']:.2f}% (test)", "public data?": "yes"})
    if lit:
        rows += lit["entries"]
    if not rows:
        _missing("Literature table", "app/assets/literature.json")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(f'<div class="card"><div class="kicker">Read this before comparing</div>'
                f'{esc(lit["note"] if lit else "")}</div>', unsafe_allow_html=True)


def render() -> None:
    pipeline_view()
    st.divider()
    sections = [("Dataset (measured, not claimed)", dataset_section), ("Model comparison", models_section),
                ("Learning curve: the research question", curve_section), ("Ablation", ablation_section),
                ("Confusion analysis", confusion_section), ("Explainability: global SHAP", shap_section),
                ("Literature comparison", literature_section)]
    for title, fn in sections:
        st.markdown(f"<h3>{esc(title)}</h3>", unsafe_allow_html=True)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one broken artefact must not take down the page
            st.warning(f"This section could not be drawn ({type(exc).__name__}: {exc}).")
