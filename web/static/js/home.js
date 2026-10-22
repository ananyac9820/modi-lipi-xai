const STATUS_PILL = {
  done: ["pill-done", "Done"], progress: ["pill-progress", "In progress"],
  review: ["pill-review", "Needs review"], todo: ["pill-todo", "Not started"], blocked: ["pill-blocked", "Blocked"],
};
const STATUS_ACCENT = { done: "var(--green)", progress: "var(--blue-500)", review: "var(--gold)", todo: "var(--stone)", blocked: "var(--navy-800)" };

function statCard({ label, value, unit = "", foot, tone, href }) {
  const tag = href ? "a" : "div";
  return `<${tag} class="card stat tone-${tone}"${href ? ` href="${href}" style="text-decoration:none;color:inherit"` : ""}>
    <span class="label">${label}</span>
    <span class="value">${value}${unit ? `<small>${unit}</small>` : ""}</span>
    <span class="foot">${foot}</span>${hill(tone)}</${tag}>`;
}

function greet() {
  const h = new Date().getHours();
  const word = h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
  $("#greeting").textContent = `${word}, reader`;
}

function setupDots(track, dots) {
  const pages = () => Math.max(1, Math.round(track.scrollWidth / track.clientWidth));
  const draw = () => {
    const n = pages();
    const cur = Math.round(track.scrollLeft / track.clientWidth);
    dots.innerHTML = n < 2 ? "" : Array.from({ length: n }, (_, i) =>
      `<button type="button" aria-label="Cards page ${i + 1}" aria-current="${i === cur}"></button>`).join("");
    $$("button", dots).forEach((b, i) => b.addEventListener("click", () => track.scrollTo({ left: i * track.clientWidth, behavior: "smooth" })));
  };
  let t;
  track.addEventListener("scroll", () => { clearTimeout(t); t = setTimeout(draw, 80); }, { passive: true });
  addEventListener("resize", draw);
  draw();
}

async function main() {
  greet();
  let ov, curves, xai;
  try {
    [ov, curves, xai] = await Promise.all([getJSON("/api/overview"), getJSON("/api/curves"), getJSON("/api/explain-global")]);
  } catch (e) {
    $("#stats").innerHTML = `<div class="card note warn" style="grid-column:1/-1">Could not load results: ${esc(e.message)}</div>`;
    return;
  }
  const ds = ov.dataset, main = ov.main, cats = ds.categories || {};
  $("#chip").innerHTML = `<span>${ds.classes} characters</span><span class="dot"></span><span>${fmtInt(ds.images)} images</span>`;

  const big = ov.largest_point;
  const bestModel = big ? ["svm", "stack", "rf", "xgb"].reduce((a, m) => (big[m] > (big[a] ?? 0) ? m : a), "svm") : null;
  const share = xai.global?.family_share || {};
  const faith = xai.global?.faithfulness_deletion_test;
  const cross = ov.crossover?.between;
  const cards = [
    { label: "Accuracy", value: pct(big?.[bestModel], 0), unit: "%", tone: "blue", href: "results.html#curve",
      foot: `${(bestModel || "").toUpperCase()} at ${fmtInt(big?.n)}/class` },
    { label: "Characters", value: ds.classes, tone: "gold", href: "script.html",
      foot: `<b>${cats.digit?.classes ?? "–"}</b> digits, ${(ds.classes ?? 0) - (cats.digit?.classes ?? 0)} letters` },
    cross
      ? { label: "CNN wins at", value: cross[1], tone: "green", href: "results.html#curve", foot: "images per class" }
      : { label: "Main model", value: pct(main.accuracy?.[main.best_model], 0), unit: "%", tone: "green", href: "results.html#models",
          foot: `${(main.best_model || "").toUpperCase()} at ${main.train_per_class}/class` },
    { label: "HOG share", value: pct(share.hog, 0), unit: "%", tone: "navy", href: "explain.html",
      foot: `of SHAP evidence` },
    { label: "Faithful", value: pct(faith?.mean_probability_drop_top_shap, 0), unit: "%", tone: "gold", href: "explain.html#faith",
      foot: `drop vs <b>${pct(faith?.mean_probability_drop_random, 0)}%</b> random` },
    { label: "Images", value: Math.round(ds.images / 1000), unit: "k", tone: "blue", href: "script.html",
      foot: `~${fmtInt(Math.round(ds.images / ds.classes / 100) * 100)} each` },
  ];
  $("#stats").innerHTML = cards.map(statCard).join("");
  setupDots($("#stats"), $("#dots"));

  $("#status").innerHTML = ov.status.map((s) => {
    const [cls, txt] = STATUS_PILL[s.status] || STATUS_PILL.todo;
    return `<li><a class="row" href="${s.href}" style="--accent:${STATUS_ACCENT[s.status]}">
      <span class="title">${esc(s.title)}</span><span class="meta">${esc(s.meta)}</span><span class="pill ${cls}">${txt}</span></a></li>`;
  }).join("");

  drawMini(curves);
  addEventListener("layoutchange", () => drawMini(curves));
}

function drawMini(curves) {
  const best = curves.classical.map((p) => ({ x: p.n, y: Math.max(p.rf ?? 0, p.svm ?? 0, p.xgb ?? 0, p.stack ?? 0) }));
  const series = [
    { name: "Best classical model", colour: "var(--navy-800)", points: best },
    { name: "Reference CNN (untuned)", colour: "var(--gold)", dash: "6 4", hollow: true, points: curves.cnn.map((p) => ({ x: p.n, y: p.cnn })) },
  ];
  const ys = [...best, ...series[1].points].map((p) => p.y);
  const lo = Math.floor(Math.min(...ys) * 20) / 20, hi = Math.ceil(Math.max(...ys) * 20) / 20;
  const mini = $("#mini-curve");
  mini.classList.remove("skeleton");
  mini.style.minHeight = "";
  mini.innerHTML = lineChart({ series, xTicks: curves.planned, yMin: lo, yMax: hi, xLabel: "training images per class (log scale)",
    band: curves.crossover?.between, height: 220 }) + legend(series) +
    (curves.cnn.length < curves.planned.length ? `<p class="muted" style="margin-top:8px">CNN points still training: ${curves.planned.filter((n) => !curves.cnn.some((p) => p.n === n)).join(", ")} per class.</p>` : "");
}

main();
