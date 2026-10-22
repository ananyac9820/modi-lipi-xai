let LABELS = [], PAIRS = [];

function card(l) {
  const f1 = l.metrics?.f1;
  return `<button class="card glyph-card" type="button" data-label="${esc(l.label)}">
    ${f1 != null ? `<span class="f1-tag${f1 < 0.75 ? " low" : ""}" title="F1 on the test set">${f1.toFixed(2)}</span>` : ""}
    <img src="/api/glyph/${encodeURIComponent(l.label)}" alt="" loading="lazy" onerror="this.outerHTML='<div class=&quot;modi&quot;>${esc(l.modi_unicode || "")}</div>'">
    <div class="deva">${esc(l.devanagari || "–")}</div>
    <div class="rom">${esc(l.label)}</div></button>`;
}

function renderGroup(cat) {
  const items = LABELS.filter((l) => l.category === cat);
  const mean = items.reduce((a, l) => a + (l.metrics?.f1 ?? 0), 0) / (items.length || 1);
  const hardest = [...items].filter((l) => l.metrics).sort((a, b) => a.metrics.f1 - b.metrics.f1)[0];
  $("#panel-" + cat).innerHTML = `
    <p class="section-sub" style="margin-top:0">${items.length} ${cat}s · mean F1 ${mean.toFixed(2)}${hardest ? ` · hardest: <b>${esc(hardest.label)}</b> (${hardest.metrics.f1.toFixed(2)})` : ""}. Tap one for details.</p>
    <div class="glyph-grid">${items.map(card).join("")}</div>`;
  $$(".glyph-card", $("#panel-" + cat)).forEach((b) => b.addEventListener("click", () => openLabel(b.dataset.label)));
}

function openLabel(name) {
  const l = LABELS.find((x) => x.label === name);
  if (!l) return;
  const m = l.metrics || {};
  const confused = PAIRS.filter((p) => p.class_a === name || p.class_b === name).slice(0, 3);
  openSheet(`
    <div class="sheet-head">
      <div class="glyphs"><img src="/api/glyph/${encodeURIComponent(name)}" alt="Handwritten sample of ${esc(name)}"></div>
      <div><div class="muted" style="text-transform:capitalize">${esc(l.category)}</div>
        <div style="font-size:30px;font-weight:700;line-height:1.1">${esc(name)} <span class="reading-deva">${esc(l.devanagari || "")}</span></div>
        ${l.modi_unicode ? `<div style="font-family:var(--font-modi);font-size:30px;color:var(--navy-800)" title="Unicode Modi">${esc(l.modi_unicode)}</div>` : ""}</div>
      <button class="close-x" data-close aria-label="Close">×</button>
    </div>
    <h3 class="section-title" style="font-size:18px">On the test set</h3>
    ${m.f1 != null ? bars([{ name: "Precision", value: m.precision }, { name: "Recall", value: m.recall }, { name: "F1", value: m.f1, colour: "var(--navy-800)" }], { max: 1 }) : `<p class="muted">No test metrics yet.</p>`}
    <p class="muted" style="margin-top:8px">200 test images, stacking ensemble.</p>
    ${confused.length ? `<h3 class="section-title" style="font-size:18px">Most often mixed up with</h3>
      <ul class="rows">${confused.map((p) => {
        const other = p.class_a === name ? p.class_b : p.class_a;
        const n = p.class_a === name ? p.a_as_b : p.b_as_a;
        return `<li><div class="row" style="--accent:var(--gold);grid-template-columns:auto 1fr auto">
          <img src="/api/glyph/${encodeURIComponent(other)}" alt="" width="44" height="44" style="background:#fff;border-radius:6px;box-shadow:inset 0 0 0 1px var(--rule)">
          <span class="title">${esc(other)} <span class="deva" style="color:var(--blue-700)">${esc(LABELS.find((x) => x.label === other)?.devanagari || "")}</span></span>
          <span class="meta">${n} of 200</span></div></li>`;
      }).join("")}</ul>` : ""}
    <div class="note" style="margin-top:16px">${l.verified ? "Devanagari mapping checked by a reader." : "Devanagari mapping not yet checked by a Modi reader."} ${l.note ? esc(l.note[0].toUpperCase() + l.note.slice(1)) + "." : ""}</div>`);
}

async function main() {
  try {
    const [s, r] = await Promise.all([getJSON("/api/script"), getJSON("/api/results")]);
    LABELS = s.labels;
    PAIRS = r.pairs?.pairs || [];
    ["consonant", "vowel", "digit"].forEach(renderGroup);
  } catch (e) {
    $("#panel-consonant").innerHTML = `<div class="note warn">${esc(e.message)}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", () => { initTabs(); main(); });
