/* Shared page furniture: tab bar, paper-cut header art, in-page tabs,
   bottom sheet, small SVG charts and formatting helpers. */

const NAV = [
  { href: "index.html", id: "home", label: "Home",
    icon: '<path d="M4 11.2 12 4l8 7.2V20a1 1 0 0 1-1 1h-4.6v-6h-4.8v6H5a1 1 0 0 1-1-1z" fill="currentColor"/><path d="M2.6 12 12 3.5 21.4 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>' },
  { href: "read.html", id: "read", label: "Read",
    icon: '<path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M7.5 9.5h9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M10 9.5v5.2c0 1 .8 1.8 1.8 1.8M14.5 9.5v6.8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>' },
  { href: "results.html", id: "results", label: "Results",
    icon: '<rect x="4" y="13" width="4" height="7" rx="1" fill="currentColor"/><rect x="10" y="9" width="4" height="11" rx="1" fill="currentColor"/><rect x="16" y="4" width="4" height="16" rx="1" fill="currentColor"/><path d="M3 21.2h18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>' },
  { href: "explain.html", id: "explain", label: "Explain",
    icon: '<path d="M12 3a6.2 6.2 0 0 0-3.6 11.3c.6.5 1 1.2 1 2V17h5.2v-.7c0-.8.4-1.5 1-2A6.2 6.2 0 0 0 12 3z" fill="currentColor"/><path d="M9.6 19.2h4.8M10.4 21.4h3.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>' },
  { href: "script.html", id: "script", label: "Script",
    icon: '<path d="M5 4.5h11.5a2 2 0 0 1 2 2V20H7a2 2 0 0 1-2-2z" fill="currentColor"/><path d="M5 17.5A2.5 2.5 0 0 1 7.5 15h11" fill="none" stroke="var(--paper)" stroke-width="1.5"/><path d="M8.5 8.5h6M8.5 11.3h4" stroke="var(--paper)" stroke-width="1.5" stroke-linecap="round"/>' },
];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "–" : (100 * x).toFixed(d));
const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-IN"));
const fmtTime = (s) => (s == null ? "–" : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const MODEL_NAME = { rf: "Random Forest", svm: "SVM (RBF)", xgb: "XGBoost", stack: "Stacking ensemble", cnn: "Reference CNN" };
const FAMILY = { hog: { name: "HOG", what: "stroke direction", c: "var(--hog)" }, lbp: { name: "LBP", what: "local stroke texture", c: "var(--lbp)" }, zernike: { name: "Zernike", what: "overall shape", c: "var(--zernike)" } };

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch { /* not json */ }
  if (!res.ok) throw new Error((body && body.error) || `Request failed (${res.status})`);
  return body;
}

function toast(msg, ms = 3200) {
  const t = document.createElement("div");
  t.className = "toast"; t.role = "status"; t.textContent = msg;
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}

/* ---------------- tab bar ---------------- */
function renderTabbar() {
  const page = document.body.dataset.page;
  const nav = document.createElement("nav");
  nav.className = "tabbar"; nav.setAttribute("aria-label", "Main");
  nav.innerHTML = "<ul>" + NAV.map((n) =>
    `<li><a href="${n.href}"${n.id === page ? ' aria-current="page"' : ""}><span class="ico"><svg viewBox="0 0 24 24" aria-hidden="true">${n.icon}</svg></span>${n.label}</a></li>`).join("") + "</ul>";
  document.body.append(nav);
}

/* ---------------- paper-cut art ---------------- */
const ART_DEFS = `
  <defs>
    <filter id="ps" x="-10%" y="-20%" width="120%" height="150%"><feDropShadow dx="0" dy="2.4" stdDeviation="2.2" flood-color="#081634" flood-opacity=".5"/></filter>
    <filter id="ps-soft" x="-10%" y="-20%" width="120%" height="150%"><feDropShadow dx="0" dy="1.6" stdDeviation="1.4" flood-color="#5a4318" flood-opacity=".35"/></filter>
    <filter id="grain"><feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" seed="7"/><feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .10 0"/><feComposite in2="SourceGraphic" operator="in"/></filter>
    <linearGradient id="edge-g" gradientUnits="userSpaceOnUse" x1="-10" x2="410" y1="0" y2="0"><stop offset="0" stop-color="#000"/><stop offset=".06" stop-color="#fff"/><stop offset=".94" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>
    <mask id="edge-fade" maskUnits="userSpaceOnUse" x="-320" y="0" width="1040" height="320"><rect x="-10" y="0" width="420" height="320" fill="url(#edge-g)"/></mask>
    <linearGradient id="river" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#a9c4e6"/><stop offset="1" stop-color="#6f97c8"/></linearGradient>
  </defs>`;

/* a breaking-wave crest, drawn facing right; mirrored for the other side */
const CURL = `
  <path d="M-12 120C8 96 24 70 54 58c30-12 62-4 72 22 8 20-2 40-20 44-15 3-26-8-22-20 3-10 15-12 20-5-1-12-16-19-30-12-17 9-23 32-14 52 6 14 18 24 30 30H-12z" fill="#efe8da"/>
  <path d="M30 108c6-18 18-30 36-34 14-3 24 2 28 10-10-4-22-2-30 6-10 10-12 26-4 42H24c0-8 2-16 6-24z" fill="#4677b8"/>
  <path d="M8 100c8-14 20-26 36-32M16 112c6-10 14-18 26-24" stroke="#d8cfbd" stroke-width="2" fill="none" stroke-linecap="round"/>`;

/* full-width wave band from x=-320 to 720, used to extend the scene on wide screens */
function band(y, amp, colour, phase = 0, bottom = 300) {
  let d = `M-320 ${y}`;
  for (let x = -320; x < 720; x += 110) {
    const k = ((x / 110) | 0) + phase;
    d += `c30 ${-amp * (k % 2 ? 1 : .6)} 70 ${-amp * (k % 2 ? .4 : 1)} 110 0`;
  }
  return `<path d="${d}V${bottom}H-320z" fill="${colour}"/>`;
}

const SIDE_CLOUD = (x, y, s) => `<g transform="translate(${x} ${y}) scale(${s})"><path d="M0 24c2-14 16-20 28-14 6-14 26-16 36-4 10-10 28-8 34 6 8-2 16 2 20 10v10H0z" fill="#d6c094"/><path d="M14 30c4-8 14-10 22-6 6-8 18-8 24 0 8-4 18-2 22 6v6H14z" fill="#e8d9b7"/></g>`;

function heroViewBox() {
  const el = document.getElementById("hero-art");
  const w = el ? el.clientWidth : innerWidth;
  const h = el ? el.clientHeight : innerWidth * 236 / 400;
  return w / Math.max(h, 1) > 1.85 ? "-220 64 840 236" : "0 64 400 236";
}

/* chart canvas width per layout band; pages re-render when the band changes */
const chartWidth = () => (innerWidth >= 1024 ? 560 : innerWidth >= 720 ? 640 : 360);

function heroArt() {
  return `<svg class="hero-art" id="hero-art" viewBox="${heroViewBox()}" preserveAspectRatio="xMidYMax slice" role="img" aria-label="Cut-paper illustration of a paper boat sailing down a river between layered blue waves">
  ${ART_DEFS}
  <g filter="url(#ps-soft)">${SIDE_CLOUD(-200, 84, 1)}${SIDE_CLOUD(470, 76, 1.2)}</g>
  <g filter="url(#ps)">
    ${band(134, 14, "#6f97c8")}${band(156, 16, "#4f80bf", 1)}${band(186, 18, "#2f63a8")}
    ${band(216, 18, "#24508f", 1)}${band(248, 16, "#1c3670")}
  </g>
  <g mask="url(#edge-fade)">
  <g filter="url(#ps-soft)">
    <path d="M222 104c2-16 18-24 32-18 6-16 28-20 40-6 10-14 32-14 42 2 12-8 30-6 38 8 8-4 18-2 26 4v24H222z" fill="#d6c094"/>
    <path d="M250 112c4-12 18-16 28-10 8-12 26-12 34 0 10-8 26-6 32 6 10-6 24-4 30 4 8-2 16 0 26 6v10H250z" fill="#e8d9b7"/>
    <path d="M-6 120c4-14 18-20 30-14 6-12 24-14 32-2 10-6 24-2 26 10 6-2 14 2 16 8v12H-6z" fill="#d6c094"/>
    <path d="M-6 128c6-8 16-10 24-6 6-8 18-8 24 0 8-4 18-2 22 6v8H-6z" fill="#e8d9b7"/>
  </g>
  <g filter="url(#ps)">
    <path d="M-10 132c40-14 70-8 104 0s60 4 92-6 56-10 90-2 70 8 134 0v176H-10z" fill="#6f97c8"/>
    <path d="M-10 150c34-10 58-4 82 6 22 9 46 6 68-4 26-12 48-10 66 0 24 13 52 12 86 2 30-9 64-8 118 2v164H-10z" fill="#4f80bf"/>
  </g>
  <path d="M200 140C197 154 190 166 186 182S170 214 160 236 142 280 138 300H272C262 278 250 256 240 236S216 200 216 182 206 152 203 140z" fill="url(#river)" filter="url(#ps)"/>
  <path d="M200 146c-4 10-12 20-10 32m18-30c4 10 10 18 8 30M186 214c-6 12-12 22-14 36m52-38c4 12 12 22 16 36" stroke="#dbe7f5" stroke-width="1.4" fill="none" stroke-linecap="round" opacity=".75"/>
  <g filter="url(#ps)">
    <path d="M-10 172C26 164 60 166 90 176s56 10 82 4c12-2 14 10 8 18-10 12-30 14-50 16-40 4-90 0-140 12z" fill="#2f63a8"/>
    <path d="M410 168c-36-8-70-6-98 4s-54 10-78 4c-12-2-14 10-8 18 10 12 30 16 50 18 40 4 88 0 134 10z" fill="#2f63a8"/>
    <path d="M-10 206c40-10 80-8 114 2 26 8 46 6 64 2 8 0 8 12 2 20-12 14-34 18-58 18-42 0-82-4-122 8z" fill="#24508f"/>
    <path d="M410 200c-38-8-76-6-108 4-24 8-46 8-64 2-8 0-6 12 0 20 12 14 32 20 56 20 42 0 80-6 116 4z" fill="#24508f"/>
  </g>
  <g transform="translate(205 196)" filter="url(#ps)">
    <path d="M-1-44v42" stroke="#8a7a5c" stroke-width="1.6"/>
    <path d="M1-42 22-4H1z" fill="#fbf7ef"/>
    <path d="M-3-36-18-4h15z" fill="#e7dcc6"/>
    <path d="M-1-44l9 3-9 3z" fill="#24508f"/>
    <path d="M-26-2h50l-8 10h-36z" fill="#f4eee2"/>
    <path d="M-30 12c8-3 14 3 22 0s14-3 22 0 14 3 22 0" stroke="#dbe7f5" stroke-width="1.6" fill="none" stroke-linecap="round"/>
  </g>
  <g filter="url(#ps)">
    <g transform="translate(-4 150) scale(1.05)">${CURL}</g>
    <g transform="translate(404 128) scale(-1 1) scale(.92)">${CURL}</g>
    <path d="M-10 238c40-12 80-10 110 0 20 7 40 12 56 24 8 10 4 28-6 38H-10z" fill="#1c3670"/>
    <path d="M410 232c-40-10-80-6-108 4-20 8-40 16-52 30-6 12 0 26 10 34h150z" fill="#1c3670"/>
    <g transform="translate(332 236) scale(-.56 .56)">${CURL}</g>
  </g>
  </g>
  ${band(288, 8, "#f3ecdf", 0, 320).replace("<path", '<path filter="url(#ps)"')}
  <path d="M-10 286c40-10 70-12 104-6 40 7 66 12 106 4s70-10 104-4 60 8 106 0v30H-10z" fill="#f3ecdf" filter="url(#ps)"/>
  <rect x="-320" width="1040" height="300" fill="#fff" filter="url(#grain)" opacity=".7" style="mix-blend-mode:overlay"/>
</svg>`;
}

/* slim strip of waves for inner pages */
function stripArt() {
  return `<svg class="hero-art compact" viewBox="0 0 400 92" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
  ${ART_DEFS}
  <g filter="url(#ps-soft)">
    <path d="M262 30c2-10 14-14 22-8 6-10 22-10 28 0 8-6 22-4 26 6 8-4 18 0 22 6 6-2 14 0 20 4v10H262z" fill="#d6c094"/>
    <path d="M284 36c4-6 12-8 18-4 6-6 16-6 20 0 8-4 18-2 22 4 8-2 16 0 22 4v6h-82z" fill="#e8d9b7"/>
  </g>
  <g filter="url(#ps)">
    <path d="M-10 44c40-12 72-8 108 0s62 4 96-4 60-8 96 0 64 8 120 0v60H-10z" fill="#4f80bf"/>
    <path d="M-10 58c30-8 56-4 80 4 26 9 50 6 74-2 26-9 50-8 70 0 24 10 52 10 86 2 30-7 62-6 110 2v44H-10z" fill="#2f63a8"/>
    <g transform="translate(316 34) scale(-.42 .42)">${CURL}</g>
    <path d="M-10 70c26-6 50-6 72 0 24 7 48 8 76 0 22-6 44-6 66 0 26 7 52 8 80 2 30-6 60-6 126 2v28H-10z" fill="#1c3670"/>
  </g>
  <path d="M-10 84c40-8 70-9 104-4 40 6 66 9 106 2s70-8 104-3 60 6 106 0v20H-10z" fill="#f3ecdf" filter="url(#ps)"/>
  <rect width="400" height="92" fill="#fff" filter="url(#grain)" opacity=".7" style="mix-blend-mode:overlay"/>
</svg>`;
}

const HILL = {
  blue: ["#4677b8", "#6f97c8"], gold: ["#c9a25c", "#dcc08a"], green: ["#3d7a47", "#8db17a"], navy: ["#1c3670", "#4677b8"],
};
function hill(tone = "blue") {
  const [a, b] = HILL[tone] || HILL.blue;
  return `<svg class="hill" viewBox="0 0 120 44" preserveAspectRatio="none" aria-hidden="true">
    <path d="M8 44c10-14 22-22 38-20 12 2 18-10 32-14 14-4 28 0 42 6v28z" fill="${b}"/>
    <path d="M30 44c10-10 22-14 34-12 14 2 22-8 34-10 8-1 16 1 22 4v18z" fill="${a}"/></svg>`;
}

const BOAT_LOADER = `<div class="boat-loader" role="status"><svg viewBox="0 0 60 40" aria-hidden="true"><path d="M29 2v28" stroke="#8a7a5c" stroke-width="1.5"/><path d="M31 4 46 28H31z" fill="#fff" stroke="#d8c7a4"/><path d="M27 9 16 28h11z" fill="#e7dcc6"/><path d="M10 30h40l-6 7H16z" fill="#2f63a8"/></svg><span class="msg">Reading…</span></div>`;

/* ---------------- in-page tabs ---------------- */
function initTabs(root = document, onShow = () => {}) {
  const list = $(".tabs", root);
  if (!list) return;
  const buttons = $$("button[role=tab]", list);
  const show = (id, push = true) => {
    buttons.forEach((b) => {
      const on = b.dataset.tab === id;
      b.setAttribute("aria-selected", on);
      b.tabIndex = on ? 0 : -1;
      const panel = document.getElementById("panel-" + b.dataset.tab);
      if (panel) panel.hidden = !on;
    });
    if (push) history.replaceState(null, "", "#" + id);
    const active = buttons.find((b) => b.dataset.tab === id);
    if (active) list.scrollTo({ left: active.offsetLeft - (list.clientWidth - active.offsetWidth) / 2, behavior: push ? "smooth" : "auto" });
    onShow(id);
  };
  buttons.forEach((b, i) => {
    b.addEventListener("click", () => show(b.dataset.tab));
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      const next = buttons[(i + d + buttons.length) % buttons.length];
      next.focus(); show(next.dataset.tab);
    });
  });
  const fromHash = () => {
    const hash = location.hash.slice(1);
    return buttons.some((b) => b.dataset.tab === hash) ? hash : null;
  };
  addEventListener("hashchange", () => { const h = fromHash(); if (h) show(h, false); });
  show(fromHash() || buttons[0].dataset.tab, false);
  return show;
}

/* ---------------- bottom sheet ---------------- */
function openSheet(html) {
  closeSheet();
  const back = document.createElement("div");
  back.className = "sheet-backdrop";
  const sheet = document.createElement("section");
  sheet.className = "sheet"; sheet.setAttribute("role", "dialog"); sheet.setAttribute("aria-modal", "true");
  sheet.innerHTML = `<div class="grab"></div><div class="wrap">${html}</div>`;
  document.body.append(back, sheet);
  document.body.style.overflow = "hidden";
  requestAnimationFrame(() => { back.classList.add("open"); sheet.classList.add("open"); });
  back.addEventListener("click", closeSheet);
  sheet.addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeSheet(); });
  document.addEventListener("keydown", escClose);
  setTimeout(() => $("[data-close]", sheet)?.focus(), 50);
  return sheet;
}
function escClose(e) { if (e.key === "Escape") closeSheet(); }
function closeSheet() {
  $$(".sheet, .sheet-backdrop").forEach((n) => n.remove());
  document.body.style.overflow = "";
  document.removeEventListener("keydown", escClose);
}

/* ---------------- bars ---------------- */
function bars(items, { max, colour = "var(--blue-600)", fmt = (v) => pct(v) + "%", strongFirst = false } = {}) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return `<div class="bars">${items.map((it, k) => `
    <div class="bar${strongFirst && k === 0 ? " strong" : ""}">
      <span class="name" title="${esc(it.title || it.name)}">${esc(it.name)}</span>
      <span class="track"><span class="fill" style="width:${Math.max(1.5, (100 * Math.abs(it.value)) / m)}%;background:${it.colour || colour}"></span></span>
      <span class="num">${fmt(it.value)}</span>
    </div>`).join("")}</div>`;
}

/* ---------------- line chart (learning curves) ----------------
   series: [{name, colour, dash, points:[{x,y}]}]; x on a log scale. */
function lineChart({ series, xTicks, yMin, yMax, yFmt = (v) => Math.round(v * 100) + "%", xLabel = "", band = null, height = 250, width = chartWidth() }) {
  const W = width, H = height, L = 40, R = 12, T = 12, B = 38;
  const lx = (v) => Math.log(v);
  const x0 = lx(Math.min(...xTicks)), x1 = lx(Math.max(...xTicks));
  const X = (v) => L + ((lx(v) - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B);
  const yTicks = [];
  const step = (yMax - yMin) / 4;
  for (let k = 0; k <= 4; k++) yTicks.push(yMin + k * step);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Learning curve chart">`;
  if (band) s += `<rect x="${X(band[0])}" y="${T}" width="${X(band[1]) - X(band[0])}" height="${H - T - B}" fill="var(--gold-200)" opacity=".55" rx="4"/>
      <text x="${(X(band[0]) + X(band[1])) / 2}" y="${T + 13}" text-anchor="middle" style="fill:var(--gold-600);font-size:11px;font-weight:600">crossover</text>`;
  s += `<g class="grid">${yTicks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}</g>`;
  s += yTicks.map((v) => `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${yFmt(v)}</text>`).join("");
  s += xTicks.map((v) => `<text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`).join("");
  if (xLabel) s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>`;
  s += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--rule)" stroke-width="1.5"/>`;
  for (const se of series) {
    const pts = se.points.filter((p) => p.y != null).sort((a, b) => a.x - b.x);
    if (!pts.length) continue;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="${se.colour}" stroke-width="${se.width || 2.4}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linecap="round" stroke-linejoin="round"/>`;
    s += pts.map((p) => `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="${se.r || 3.6}" fill="${se.hollow ? "var(--card)" : se.colour}" stroke="${se.colour}" stroke-width="1.8"><title>${esc(se.name)} @ ${p.x}/class: ${yFmt(p.y)}</title></circle>`).join("");
  }
  return s + "</svg>";
}

function legend(series) {
  return `<div class="chart-legend">${series.map((s) => `<span><i class="${s.dash ? "dash" : ""}" style="background:${s.colour};color:${s.colour}"></i>${esc(s.name)}</span>`).join("")}</div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  renderTabbar();
  const art = $("[data-art]");
  if (art) art.outerHTML = art.dataset.art === "hero" ? heroArt() : stripArt();
  const hero = $("#hero-art");
  if (hero) hero.setAttribute("viewBox", heroViewBox());
  let band = chartWidth(), t;
  addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      const h = $("#hero-art");
      if (h) h.setAttribute("viewBox", heroViewBox());
      if (chartWidth() !== band) { band = chartWidth(); dispatchEvent(new Event("layoutchange")); }
    }, 120);
  });
});
