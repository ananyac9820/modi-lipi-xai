const short = (r) => r.replace(/^(HOG|LBP|Zernike): /, "$1 · ").replace("stroke direction, ", "").replace("local stroke pattern, ", "");

function lookalikePanel(res) {
  const l = res.lookalike;
  if (!l) return `<div class="note">The specialists have not been trained yet.</div>`;
  const pairs = [...l.per_pair].sort((a, b) => (b.correct_final - b.correct_svm) - (a.correct_final - a.correct_svm));
  const rows = pairs.map((p) => {
    const d = p.correct_final - p.correct_svm;
    return `<li><div class="row" style="--accent:${d > 0 ? "var(--green)" : "var(--red)"};grid-template-columns:auto 1fr auto">
      <span style="display:flex;gap:6px">${p.pair.map((c) => `<img src="/api/glyph/${encodeURIComponent(c)}" alt="${esc(c)}" width="40" height="40" style="background:#fff;border-radius:6px;box-shadow:inset 0 0 0 1px var(--rule)" onerror="this.remove()">`).join("")}</span>
      <span class="title">${esc(p.pair[0])} ↔ ${esc(p.pair[1])}<small>${p.correct_svm} → ${p.correct_final} right, of ${p.images_consulted} images it decided. Mainly looks at ${esc(short(p.regions[0]))}</small></span>
      <span class="meta">${d > 0 ? "+" : ""}${d}</span></div></li>`;
  }).join("");
  const a = l.accuracy, v = l.against_svm_vote;
  return `<div class="panel-grid even"><div>
    <article class="card">
      <h3>The one new idea</h3>
      <p>When the SVM's top two guesses are a known look-alike pair, a small specialist decides. It only looks at the regions of the character that SHAP says separate those two.</p>
      ${bars([{ name: "SVM alone", value: a.svm_vote, colour: "var(--sand)" }, { name: "With specialists", value: a.final, colour: "var(--navy-800)" }], { max: 1, fmt: (x) => pct(x, 2) + "%" })}
      <p class="muted" style="margin-top:10px">${fmtInt(l.n_test)} test images. Compared with the SVM alone, the specialists fixed ${v.fixed} images and broke ${v.broken} (McNemar p = ${v.p_value.toFixed(3)}).</p>
    </article>
    <div class="note" style="margin-top:14px">Which pairs, which regions and which specialists to keep were all decided on training images. The test set was only used at the end. Most pairs improve, but not all of them. The ones with a minus sign got worse.</div>
    </div><div>
    <h2 class="section-title" style="margin-top:0">Pair by pair</h2>
    <ul class="rows">${rows}</ul></div></div>`;
}

function modelPanel(ov, res) {
  const m = ov.main, l = res.lookalike, r = res.reducer;
  if (!l) return `<div class="note">The model has not been trained yet.</div>`;
  const cats = ["digit", "vowel", "consonant"];
  return `<div class="panel-grid even"><div>
    <article class="card">
      <h3>One model: an SVM</h3>
      <p>An RBF SVM on handcrafted features (HOG, Zernike moments and LBP), reduced with LDA. Trained on ${l.train_per_class} images per character.</p>
      ${bars([{ name: "SVM alone", value: l.accuracy.svm_vote, colour: "var(--sand)" }, { name: "With specialists", value: l.accuracy.final, colour: "var(--navy-800)" }], { max: 1 })}
      <p class="muted" style="margin-top:10px">Macro-F1 with specialists: ${l.macro_f1_final.toFixed(3)}. We picked the SVM on training data, where it beat Random Forest in cross-validation.</p>
    </article>
    ${r ? `<div class="note" style="margin-top:14px">LDA over PCA was also decided on training data only (3-fold CV: ${pct(r.overall_cv_accuracy.lda)}% against ${pct(r.overall_cv_accuracy.pca)}%).</div>` : ""}
    </div><div>
    <h2 class="section-title" style="margin-top:0">By character type</h2>
    <div class="card">${cats.map((c) => `<p style="margin:10px 0 6px;font-weight:600;text-transform:capitalize">${c}s</p>` +
      bars([{ name: "SVM alone", value: m.per_category.svm[c], colour: "var(--sand)" }, { name: "With specialists", value: l.by_category[c].final, colour: "var(--navy-800)" }], { max: 1 })).join("")}
    </div></div></div>`;
}

function errorsPanel(res) {
  const p = res.pairs;
  if (!p?.pairs) return `<div class="note">No confusion analysis yet.</div>`;
  const rows = p.pairs.slice(0, 10).map((q, i) => `<li><div class="row" style="--accent:${i < 3 ? "var(--red)" : "var(--gold)"};grid-template-columns:auto 1fr auto">
      <span style="display:flex;gap:6px">${["class_a", "class_b"].map((k) => `<img src="/api/glyph/${encodeURIComponent(q[k])}" alt="${esc(q[k])}" width="40" height="40" style="background:#fff;border-radius:6px;box-shadow:inset 0 0 0 1px var(--rule)" onerror="this.remove()">`).join("")}</span>
      <span class="title">${esc(q.class_a)} ↔ ${esc(q.class_b)}<small>${q.a_as_b} one way, ${q.b_as_a} the other</small></span>
      <span class="meta">${pct(q.pair_error_rate, 0)}%</span></div></li>`).join("");
  return `
    <p class="section-sub" style="margin-top:0">${fmtInt(p.total_errors)} errors out of ${fmtInt(p.n_test)} test images, with the specialists switched on. The pairs still mixed up most:</p>
    <div class="panel-grid"><div><ul class="rows">${rows}</ul></div><div>
    <p class="muted" style="margin-top:12px">Percentage = share of the pair’s 400 test images read as the other one. The Explain tab shows which strokes separate each pair.</p></div></div>`;
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
    <div class="card-grid">${block("separated", "Letters with small gaps", "The easy case: each character is its own blob.")}
    ${block("touching", "Letters joined by a headline", "The hard case, like real Modi: one connected stroke along the top.")}</div>
    <div class="note" style="margin-top:14px">Recognition-guided re-segmentation is what makes joined writing readable at all, but cursive Modi is still the weakest link: the classifier is far more reliable than the cuts.</div>`;
}

function papersPanel(res, ov) {
  const lit = res.literature;
  if (!lit?.entries) return `<div class="note">No literature table found.</div>`;
  const l = res.lookalike;
  const tone = (a) => (/deep|cnn|vgg|resnet|incep/i.test(a) ? "var(--gold)" : /hybrid/i.test(a) ? "var(--green)" : "var(--blue-600)");
  return `
    <div class="note warn">Not a like-for-like comparison. These papers use different datasets, class counts, splits and metrics
      (often validation or even training accuracy, several on private data). They are context, not a leaderboard.</div>
    <article class="card" style="margin-top:14px;--accent:var(--navy-800)">
      <h3>This project</h3>
      <p style="margin:0">${pct(l?.accuracy.final)}% (SVM with look-alike specialists) on a fixed, held-out test set of
        ${fmtInt(l?.n_test)} images, 57 classes of a public dataset, ${l?.train_per_class} training images per class. No tuning on the test set.</p>
    </article>
    <h2 class="section-title">Published results</h2>
    <ul class="rows two-up">${lit.entries.map((e) => `<li><div class="row" style="--accent:${tone(e.approach)};grid-template-columns:1fr auto">
      <span class="title">${esc(e.work)}<small>${esc(e.approach)} · ${esc(e.dataset)}</small></span>
      <span class="pill ${e["public data?"] === "yes" ? "pill-done" : e["public data?"] === "no" ? "pill-todo" : "pill-review"} sm" title="Public data?">${esc(e["public data?"] === "yes" ? "public" : e["public data?"] === "no" ? "private" : e["public data?"] === "partly" ? "partly" : "unclear")}</span>
      <span class="meta" style="grid-column:1 / -1;white-space:normal">${esc(e["reported accuracy"])}</span></div></li>`).join("")}</ul>`;
}

async function main() {
  try {
    const [ov, res] = await Promise.all([getJSON("/api/overview"), getJSON("/api/results")]);
    $("#panel-lookalikes").innerHTML = lookalikePanel(res);
    $("#panel-model").innerHTML = modelPanel(ov, res);
    $("#panel-errors").innerHTML = errorsPanel(res);
    $("#panel-segment").innerHTML = segmentPanel(res);
    $("#panel-papers").innerHTML = papersPanel(res, ov);
  } catch (e) {
    $("#panel-lookalikes").innerHTML = `<div class="note warn">Could not load results: ${esc(e.message)}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", () => { initTabs(); main(); });
addEventListener("layoutchange", main);
