const LEVEL = { high: ["pill-high", "Confident", "var(--green)"], medium: ["pill-medium", "Unsure", "var(--gold)"], low: ["pill-low", "Low", "var(--red)"] };
let current = null;          // last recognition result
const explained = new Map(); // k -> explanation

const mode = () => $("input[name=mode]:checked").value;

/* ---------- running a recognition ---------- */
async function recognise(form, name) {
  form.append("mode", mode());
  const out = $("#out");
  out.innerHTML = `<div class="card" style="margin-top:18px">${BOAT_LOADER}</div>`;
  const msgs = ["Cleaning the page…", "Finding lines and characters…", "Measuring strokes: HOG, Zernike, LBP…", "Asking four models…"];
  let i = 0;
  const msgEl = $(".msg", out);
  msgEl.textContent = msgs[0];
  const timer = setInterval(() => { i = Math.min(i + 1, msgs.length - 1); msgEl.textContent = msgs[i]; }, 900);
  try {
    current = await getJSON("/api/recognise", { method: "POST", body: form });
    current.name = name;
    explained.clear();
    renderResult();
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    out.innerHTML = `<div class="note warn" style="margin-top:18px">${esc(e.message)}</div>`;
  } finally {
    clearInterval(timer);
  }
}

function renderResult() {
  const r = current, out = $("#out");
  const nLow = r.chars.filter((c) => c.level === "low").length;
  const secs = Object.values(r.seconds || {}).reduce((a, b) => a + b, 0);
  out.innerHTML = `
    <h2 class="section-title">Reading</h2>
    ${r.warnings.map((w) => `<div class="note warn" style="margin-bottom:12px">${esc(w)}</div>`).join("")}
    ${r.chars.length ? `
    <article class="card">
      <div class="reading">${esc(r.reading)}</div>
      <div class="reading-deva">${esc(r.reading_devanagari)}</div>
      <p class="muted" style="margin-top:8px">${r.chars.length} character${r.chars.length > 1 ? "s" : ""} in ${secs.toFixed(1)} s.
        “?” marks low confidence${nLow ? ` (${nLow} here)` : ""}. Devanagari is an unverified mapping of the dataset’s class names.</p>
      <div class="btn-row" style="margin-top:6px">
        <button class="btn ghost" type="button" id="dl-csv">Download CSV</button>
        <button class="btn ghost" type="button" id="dl-json">Download JSON</button>
      </div>
    </article>
    <article class="card">
      <img class="boxed" src="${r.boxed}" alt="Your image with a numbered box around every detected character">
      <div class="legend" style="margin-top:10px"><span><i style="background:var(--green)"></i>confident</span><span><i style="background:var(--gold)"></i>unsure</span><span><i style="background:var(--red)"></i>low confidence</span></div>
    </article>
    <h2 class="section-title">Characters</h2>
    <p class="section-sub">Tap a character to see why the model chose it.</p>
    <ol class="rows char-grid">${r.chars.map(charRow).join("")}</ol>` :
    `<div class="note">No characters were found. Try a tighter crop, darker ink or a sharper photo.</div>`}`;
  $$(".char-row", out).forEach((b) => b.addEventListener("click", () => openChar(+b.dataset.k)));
  $("#dl-csv")?.addEventListener("click", downloadCSV);
  $("#dl-json")?.addEventListener("click", () => save(JSON.stringify({ image: r.name, reading: r.reading, characters: r.chars.map(stripImages) }, null, 2), "json", "application/json"));
}

function charRow(c) {
  const [cls, txt, colour] = LEVEL[c.level] || LEVEL.low;
  if (c.error) {
    return `<li><button class="row char-row" data-k="${c.k}" style="--accent:var(--stone)"><img class="thumb" src="${c.crop}" alt="">
      <span class="title"><span class="num">#${c.order}</span><small>${esc(c.error)}</small></span><span class="pill pill-todo sm">Skipped</span></button></li>`;
  }
  return `<li><button class="row char-row" data-k="${c.k}" style="--accent:${colour}">
    <img class="thumb" src="${c.crop}" alt="Character ${c.order}">
    <span class="title"><span class="num">#${c.order}</span> <span>${esc(c.label)}</span> <span class="deva">${esc(c.devanagari || "")}</span>
      <span class="conf" aria-hidden="true"><span style="width:${100 * c.confidence}%;background:${colour}"></span></span>
      ${c.flags.length ? `<small>check the box: ${esc(c.flags[0].flag)}</small>` : ""}</span>
    <span class="pill ${cls} sm">${pct(c.confidence, 0)}%</span></button></li>`;
}

/* ---------- one character ---------- */
function openChar(k) {
  const c = current.chars[k];
  if (!c) return;
  const [cls, txt] = LEVEL[c.level] || LEVEL.low;
  const sheet = openSheet(`
    <div class="sheet-head">
      <div class="glyphs"><img src="${c.crop}" alt="Crop from your image">${c.glyph ? `<img src="${c.glyph}" alt="What the model saw (64 by 64)">` : ""}</div>
      <div><div class="muted">Character #${c.order}</div>
        <div style="font-size:28px;font-weight:700;line-height:1.1">${esc(c.label || "–")} <span class="reading-deva">${esc(c.devanagari || "")}</span></div>
        <span class="pill ${cls} sm" style="margin-top:4px">${txt} · ${pct(c.confidence, 0)}%</span></div>
      <button class="close-x" data-close aria-label="Close">×</button>
    </div>
    ${c.flags.map((f) => `<div class="note warn" style="margin-top:14px">Segmentation: ${esc(f.text)}</div>`).join("")}
    <h3 class="section-title" style="font-size:18px;margin-top:20px">Top guesses</h3>
    ${bars(c.top.map((t, i) => ({ name: `${t.label}  ${t.devanagari || ""}`, value: t.proba, colour: i ? "var(--sand)" : "var(--navy-800)" })), { max: 1, strongFirst: true })}
    <p class="muted" style="margin-top:10px">Each model on its own: ${Object.entries(c.votes).map(([m, v]) => `<b>${m.toUpperCase()}</b> ${esc(v)}`).join(" · ")}</p>
    <h3 class="section-title" style="font-size:18px">Why this answer</h3>
    <div id="why"><button class="btn block" type="button" id="explain-btn">Explain with SHAP</button>
      <p class="muted" style="margin-top:8px">Measures how 35 regions of the three descriptors moved the probability. Takes a few seconds.</p></div>`);
  $("#explain-btn", sheet)?.addEventListener("click", () => explainChar(k, sheet));
  if (explained.has(k)) renderWhy(explained.get(k), c, sheet);
}

async function explainChar(k, sheet) {
  const why = $("#why", sheet);
  why.innerHTML = BOAT_LOADER.replace("Reading…", "Weighing the evidence…");
  try {
    const ex = await getJSON(`/api/explain/${current.id}/${k}`);
    explained.set(k, ex);
    renderWhy(ex, current.chars[k], sheet);
  } catch (e) {
    why.innerHTML = `<div class="note warn">${esc(e.message)}</div>`;
  }
}

function heat(glyph, grid, label) {
  const flat = grid.flat();
  const m = Math.max(...flat.map(Math.abs), 1e-9);
  const cells = flat.map((v) => {
    const a = (0.62 * Math.abs(v)) / m;
    const col = v >= 0 ? `rgba(184,138,62,${a})` : `rgba(47,99,168,${a})`;
    return `<span style="background:${col}"></span>`;
  }).join("");
  return `<figure><div class="heat"><img src="${glyph}" alt=""><div class="grid">${cells}</div></div><figcaption>${label}</figcaption></figure>`;
}

function groupBars(items, colourOverride) {
  if (!items.length) return `<p class="muted">Nothing pushed this way.</p>`;
  return bars(items.slice(0, 6).map((g) => ({ name: `${FAMILY[g.family]?.name || ""} · ${g.group.replace(/^(HOG|LBP|Zernike): /, "").replace("stroke direction, ", "").replace("local stroke pattern, ", "")}`,
    title: g.group, value: Math.abs(g.shap), colour: colourOverride || FAMILY[g.family]?.c })), { fmt: (v) => "+" + pct(v, 1) });
}

function renderWhy(ex, c, sheet) {
  const fam = ex.family_contribution_predicted;
  const why = $("#why", sheet);
  why.innerHTML = `
    <p>Compared with an average character, the evidence moved <b>${esc(ex.predicted)}</b> from
      ${pct(ex.baseline_proba_predicted)}% to <b>${pct(ex.predicted_proba)}%</b>. The runner-up was
      <b>${esc(ex.runner_up)}</b> ${esc(ex.runner_up_devanagari || "")} at ${pct(ex.runner_up_proba)}%.</p>
    <div class="legend" style="margin-bottom:12px">${Object.entries(FAMILY).map(([k, f]) => `<span><i style="background:${f.c}"></i>${f.name}: ${f.what}</span>`).join("")}</div>
    ${c.glyph ? `<div class="heat-pair">${heat(c.glyph, ex.region_maps_predicted.hog, "HOG evidence by region")}${heat(c.glyph, ex.region_maps_predicted.lbp, "LBP evidence by region")}</div>
      <p class="muted" style="margin-top:6px">Gold squares pushed toward “${esc(ex.predicted)}”, blue squares pushed away.</p>` : ""}
    <div class="tabs" role="tablist" style="position:static;margin-top:16px">
      <button role="tab" data-tab="for">For “${esc(ex.predicted)}”</button>
      <button role="tab" data-tab="against">Against</button>
      <button role="tab" data-tab="runner">For “${esc(ex.runner_up)}”</button>
    </div>
    <div class="panel" id="panel-for">${groupBars(ex.toward_predicted)}</div>
    <div class="panel" id="panel-against" hidden>${groupBars(ex.against_predicted, "var(--red)")}</div>
    <div class="panel" id="panel-runner" hidden>${groupBars(ex.toward_runner_up)}</div>
    <p class="muted" style="margin-top:14px">Net effect by descriptor: ${Object.entries(fam).map(([f, v]) => `${FAMILY[f]?.name || f} ${v >= 0 ? "+" : ""}${pct(v, 1)}`).join(" · ")} points.</p>`;
  initTabsLocal(why);
}

/* tabs inside the sheet must not touch the page hash */
function initTabsLocal(root) {
  const btns = $$("button[role=tab]", root);
  const show = (id) => btns.forEach((b) => {
    b.setAttribute("aria-selected", b.dataset.tab === id);
    $("#panel-" + b.dataset.tab, root).hidden = b.dataset.tab !== id;
  });
  btns.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  show(btns[0].dataset.tab);
}

/* ---------- downloads ---------- */
const stripImages = ({ crop, glyph, ...rest }) => rest;
function save(text, ext, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = `${(current.name || "modi").replace(/\.[^.]+$/, "")}_reading.${ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function downloadCSV() {
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = [["order", "label", "devanagari", "confidence", "level", "second", "third", "line", "flags"]];
  current.chars.forEach((c) => rows.push([c.order, c.label, c.devanagari, c.confidence, c.level, c.top[1]?.label, c.top[2]?.label, c.line, c.flags.map((f) => f.flag).join(";")]));
  save("﻿" + rows.map((r) => r.map(q).join(",")).join("\n"), "csv", "text/csv");
}

/* ---------- inputs ---------- */
function setupUpload() {
  const drop = $("#drop"), file = $("#file");
  file.addEventListener("change", () => {
    if (!file.files[0]) return;
    const fd = new FormData(); fd.append("image", file.files[0]);
    recognise(fd, file.files[0].name);
    file.value = "";
  });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (!f) return;
    const fd = new FormData(); fd.append("image", f);
    recognise(fd, f.name);
  });
}

async function setupSamples() {
  const list = $("#sample-list");
  try {
    const samples = await getJSON("/api/samples");
    list.innerHTML = samples.map((s, i) => `<li><button class="row char-row" data-i="${i}" style="--accent:${["var(--blue-600)", "var(--gold)", "var(--green)", "var(--navy-800)"][i % 4]};grid-template-columns:88px 1fr auto">
      <img class="thumb" style="width:88px" src="/samples/${encodeURIComponent(s.file)}" alt="">
      <span class="title">${esc(s.description)}<small>${esc(s.kind || "")}</small></span>
      <span class="pill ${s.ground_truth ? "pill-progress" : "pill-review"} sm">${s.ground_truth ? `${s.ground_truth.length} chars` : "hard"}</span></button></li>`).join("");
    $$("button", list).forEach((b) => b.addEventListener("click", () => {
      const s = samples[+b.dataset.i];
      const fd = new FormData(); fd.append("sample", s.file);
      recognise(fd, s.file);
    }));
  } catch (e) {
    list.innerHTML = `<li class="note warn">${esc(e.message)}</li>`;
  }
}

function setupPad() {
  const pad = $("#pad"), ctx = pad.getContext("2d");
  const strokes = [];
  let drawing = null;
  const reset = () => { ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, pad.width, pad.height); };
  const redraw = () => {
    reset();
    ctx.strokeStyle = "#1b2944"; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = 16;
    for (const s of strokes) {
      ctx.beginPath();
      s.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      if (s.length === 1) ctx.lineTo(s[0].x + .1, s[0].y);
      ctx.stroke();
    }
  };
  const pos = (e) => {
    const r = pad.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * pad.width, y: ((e.clientY - r.top) / r.height) * pad.height };
  };
  pad.addEventListener("pointerdown", (e) => { pad.setPointerCapture(e.pointerId); drawing = [pos(e)]; strokes.push(drawing); redraw(); });
  pad.addEventListener("pointermove", (e) => { if (!drawing) return; drawing.push(pos(e)); redraw(); });
  ["pointerup", "pointercancel"].forEach((ev) => pad.addEventListener(ev, () => { drawing = null; }));
  $("#undo").addEventListener("click", () => { strokes.pop(); redraw(); });
  $("#clear").addEventListener("click", () => { strokes.length = 0; redraw(); });
  $("#read-drawing").addEventListener("click", () => {
    if (!strokes.length) { toast("Write a character on the pad first."); return; }
    pad.toBlob((blob) => {
      const fd = new FormData(); fd.append("image", blob, "drawing.png");
      if (mode() === "auto") $("input[name=mode][value=single]").checked = true;
      recognise(fd, "drawing.png");
    }, "image/png");
  });
  redraw();
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  setupUpload();
  setupSamples();
  setupPad();
});
