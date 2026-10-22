const COLOURS = { svm: "var(--navy-800)", stack: "var(--blue-500)", rf: "var(--green)", xgb: "var(--sand)", cnn: "var(--gold)" };
const ORDER = ["svm", "stack", "rf", "xgb"];
const MB = (b) => (b == null ? "–" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " kB");

function curvePanel(c) {
  const series = ORDER.map((m) => ({ name: MODEL_NAME[m], colour: COLOURS[m], width: m === "svm" ? 2.8 : 1.8, r: m === "svm" ? 3.8 : 2.8,
    points: c.classical.map((p) => ({ x: p.n, y: p[m] })) }));
  series.push({ name: "Reference CNN (untuned)", colour: COLOURS.cnn, dash: "6 4", hollow: true, width: 2.6, points: c.cnn.map((p) => ({ x: p.n, y: p.cnn })) });
  const ys = series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => v != null);
  const lo = Math.floor(Math.min(...ys) * 20) / 20, hi = Math.ceil(Math.max(...ys) * 20) / 20;
  const cnnBy = Object.fromEntries(c.cnn.map((p) => [p.n, p]));
  const rows = c.classical.map((p) => {
    const best = ORDER.reduce((a, m) => (p[m] > p[a] ? m : a), "svm");
    const cnn = cnnBy[p.n];
    const diff = cnn ? cnn.cnn - p[best] : null;
    return `<tr><td>${p.n}</td><td>${pct(p[best])}% <span class="muted">${best.toUpperCase()}</span></td>
      <td>${cnn ? pct(cnn.cnn) + "%" : '<span class="pill pill-progress sm">training</span>'}</td>
      <td class="${diff == null ? "" : diff > 0 ? "" : "best"}">${diff == null ? "–" : (diff > 0 ? "+" : "") + (100 * diff).toFixed(1)}</td>
      <td>${fmtTime(p.train_time_s)}</td><td>${cnn ? fmtTime(cnn.train_time_s) : "–"}</td></tr>`;
  }).join("");
  const cross = c.crossover?.between;
  return `
    <article class="card">
      <h3>Accuracy as training data grows</h3>
      <p class="muted">Same seeded splits and the same fixed test set for every model.</p>
      ${lineChart({ series, xTicks: c.planned, yMin: lo, yMax: hi, xLabel: "training images per class (log scale)", band: cross, height: 270 })}
      ${legend(series)}
    </article>
    ${cross ? `<div class="note" style="margin-top:14px"><b>Crossover between ${cross[0]} and ${cross[1]} images per class.</b>
      Up to ${cross[0]}, the handcrafted pipeline beats a plain CNN; from ${cross[1]} the CNN is ahead${c.cnn.length < c.planned.length
        ? `. The larger sizes (${c.planned.filter((n) => !cnnBy[n]).join(", ")}) are still training, so whether the gap keeps growing is not known yet.` : " at every larger size measured."}</div>` : ""}
    <h2 class="section-title">Point by point</h2>
    <div class="card table-scroll"><table class="table">
      <thead><tr><th>/class</th><th>Best classical</th><th>CNN</th><th>CNN − best</th><th>Classical fit</th><th>CNN fit</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <details class="more card" style="margin-top:14px"><summary>How the comparison is kept fair</summary>
      <p style="margin-top:10px">The CNN is an untuned reference: textbook architecture (${esc(c.cnn_architecture || "")}),
      fixed before any result was seen, no augmentation. It picks its epoch count on a slice of the training set,
      then is retrained on 100% of the same training images the classical models get. Without that refit it would
      see about 10% fewer images, which would favour the classical side.</p>
      <p>The 72-per-class point only matches the sample size of earlier Modi papers; their test sets and
      preprocessing differ, so it is approximately, not directly, comparable.</p>
      <p class="muted">Classical pipeline: ${esc(c.feature_set || "")} features, ${esc((c.reducer || "").toUpperCase())} reduction.</p></details>`;
}

function modelsPanel(ov, res) {
  const m = ov.main;
  const best = m.best_model;
  const rows = ORDER.map((k) => `<li><div class="row" style="--accent:${COLOURS[k]};grid-template-columns:1fr auto">
      <span class="title">${MODEL_NAME[k]}<small>macro-F1 ${m.macro_f1[k]?.toFixed(3)} · fit ${fmtTime(k === "stack" ? m.fit_seconds?.stack_total : m.fit_seconds?.[k])} · ${MB(m.size_bytes?.[k])} · ${(m.inference_ms?.[k] ?? 0).toFixed(1)} ms/image</small></span>
      <span class="pill ${k === best ? "pill-done" : "pill-todo"}">${pct(m.accuracy[k])}%</span></div></li>`).join("");
  const r = res.reducer;
  const cats = ["consonant", "vowel", "digit"];
  return `
    <p class="section-sub" style="margin-top:0">${m.train_per_class} training images per class, ${fmtInt(m.n_test)} test images.</p>
    <ul class="rows">${rows}</ul>
    ${m.accuracy.stack < m.accuracy[best] ? `<div class="note" style="margin-top:14px">The stacking ensemble does not beat the single SVM here. It stays the app’s model as planned, and the gap is reported rather than hidden.</div>` : ""}
    <h2 class="section-title">By character type</h2>
    <div class="card">${cats.map((cat) => `<p style="margin:10px 0 6px;font-weight:600;text-transform:capitalize">${cat}s</p>` +
      bars(ORDER.map((k) => ({ name: MODEL_NAME[k], value: m.per_category[k]?.[cat], colour: COLOURS[k] })), { max: 1 })).join("")}
      ${res.category_errors ? `<p class="muted" style="margin-top:12px">${fmtInt(res.category_errors.errors_within_same_category)} of the stack’s errors stay within the same type (a consonant read as another consonant); ${fmtInt(res.category_errors.errors_across_categories)} cross types.</p>` : ""}</div>
    ${r ? `<h2 class="section-title">PCA or LDA?</h2>
    <div class="card"><p>Chosen on training data only (3-fold CV, RF and SVM), before touching the test set.</p>
      ${bars(Object.entries(r.overall_cv_accuracy).map(([k, v]) => ({ name: k.toUpperCase(), value: v, colour: k === r.selected ? "var(--navy-800)" : "var(--sand)" })), { max: 1, strongFirst: false })}
      <p class="muted" style="margin-top:8px">Selected: <b>${r.selected.toUpperCase()}</b>.</p></div>` : ""}`;
}

function featuresPanel(res) {
  const a = res.ablation;
  if (!a) return `<div class="note">Ablation results are not available yet.</div>`;
  const sets = Object.entries(a.feature_sets).map(([k, v]) => ({ k, acc: v.models?.svm?.accuracy ?? Math.max(...Object.values(v.models || {}).map((x) => x.accuracy)) }))
    .sort((x, y) => y.acc - x.acc);
  const NICE = { hog: "HOG only", lbp: "LBP only", zernike: "Zernike only", fused: "All three, family-balanced", fused_plain: "All three, as concatenated" };
  const nice = (k) => NICE[k] || k;
  const acc = Object.fromEntries(sets.map((s) => [s.k, s.acc]));
  return `
    <article class="card">
      <h3>Which descriptors earn their place?</h3>
      <p class="muted">SVM accuracy at ${a.train_per_class} images per class with each descriptor set, ${esc((a.reducer || "").toUpperCase())} reduction.</p>
      ${bars(sets.map((s, i) => ({ name: nice(s.k), value: s.acc, colour: i === 0 ? "var(--navy-800)" : "var(--blue-400)" })), { max: 1, strongFirst: true })}
    </article>
    ${acc.hog != null && acc.fused != null ? `<div class="note" style="margin-top:14px">HOG alone reaches ${pct(acc.hog)}%; fusing all three descriptors adds
      ${(100 * (Math.max(acc.fused, acc.fused_plain ?? 0) - acc.hog)).toFixed(1)} points. Zernike on its own is weak (${pct(acc.zernike)}%), which matches its small SHAP share.
      ${acc.fused_plain != null && Math.abs(acc.fused - acc.fused_plain) < 0.002 ? "Balancing the three families makes no difference after LDA, which rescales features anyway." : ""}</div>` : ""}`;
}

function errorsPanel(res) {
  const p = res.pairs;
  if (!p?.pairs) return `<div class="note">No confusion analysis yet.</div>`;
  const rows = p.pairs.slice(0, 10).map((q, i) => `<li><div class="row" style="--accent:${i < 3 ? "var(--red)" : "var(--gold)"};grid-template-columns:auto 1fr auto">
      <span style="display:flex;gap:6px">${["class_a", "class_b"].map((k) => `<img src="/api/glyph/${encodeURIComponent(q[k])}" alt="${esc(q[k])}" width="40" height="40" style="background:#fff;border-radius:6px;box-shadow:inset 0 0 0 1px var(--rule)" onerror="this.remove()">`).join("")}</span>
      <span class="title">${esc(q.class_a)} ↔ ${esc(q.class_b)}<small>${q.a_as_b} one way, ${q.b_as_a} the other</small></span>
      <span class="meta">${pct(q.pair_error_rate, 0)}%</span></div></li>`).join("");
  return `
    <p class="section-sub" style="margin-top:0">${fmtInt(p.total_errors)} errors out of ${fmtInt(p.n_test)} test images (${MODEL_NAME[p.model] || p.model}). The most confused pairs:</p>
    <ul class="rows">${rows}</ul>
    <p class="muted" style="margin-top:12px">Percentage = share of the pair’s 400 test images read as the other one. The Explain tab shows which strokes separate each pair.</p>
    <a class="figure-link" href="/figures/confusion_matrix.png"><img src="/figures/confusion_matrix.png" alt="Confusion matrix of all 57 classes" loading="lazy"></a>`;
}

function segmentPanel(res) {
  const s = res.segmentation;
  if (!s) return `<div class="note">Segmentation has not been evaluated yet.</div>`;
  const block = (name, title, sub) => {
    const v = s.summary[name];
    return `<article class="card"><h3>${title}</h3><p class="muted">${sub}</p>
      <div class="table-scroll"><table class="table"><thead><tr><th></th><th>Plain</th><th>Re-segmented</th></tr></thead><tbody>
      <tr><td>Character count right</td><td>${pct(v.plain.correct_count_rate, 0)}%</td><td class="${v.reseg.correct_count_rate > v.plain.correct_count_rate ? "best" : ""}">${pct(v.reseg.correct_count_rate, 0)}%</td></tr>
      <tr><td>Character error rate</td><td>${pct(v.plain.character_error_rate, 0)}%</td><td class="${v.reseg.character_error_rate < v.plain.character_error_rate ? "best" : ""}">${pct(v.reseg.character_error_rate, 0)}%</td></tr>
      <tr><td>Whole word right</td><td>${pct(v.plain.exact_word_accuracy, 0)}%</td><td>${pct(v.reseg.exact_word_accuracy, 0)}%</td></tr>
      <tr><td>Seconds per word</td><td>${v.plain.mean_seconds_per_word.toFixed(1)}</td><td>${v.reseg.mean_seconds_per_word.toFixed(1)}</td></tr>
      </tbody></table></div></article>`;
  };
  return `
    <p class="section-sub" style="margin-top:0">${s.summary.separated.plain.words} synthetic 3–5 letter words built from test-split images. Re-segmentation settings were tuned on training-split words only.</p>
    ${block("separated", "Letters with small gaps", "The easy case: each character is its own blob.")}
    ${block("touching", "Letters joined by a headline", "The hard case, like real Modi: one connected stroke along the top.")}
    <div class="note" style="margin-top:14px">Recognition-guided re-segmentation is what makes joined writing readable at all, but cursive Modi is still the weakest link: the classifier is far more reliable than the cuts.</div>`;
}

async function main() {
  try {
    const [c, ov, res] = await Promise.all([getJSON("/api/curves"), getJSON("/api/overview"), getJSON("/api/results")]);
    $("#panel-curve").innerHTML = curvePanel(c);
    $("#panel-models").innerHTML = modelsPanel(ov, res);
    $("#panel-features").innerHTML = featuresPanel(res);
    $("#panel-errors").innerHTML = errorsPanel(res);
    $("#panel-segment").innerHTML = segmentPanel(res);
  } catch (e) {
    $("#panel-curve").innerHTML = `<div class="note warn">Could not load results: ${esc(e.message)}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", () => { initTabs(); main(); });
