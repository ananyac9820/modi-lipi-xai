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
  let ov, xai;
  try {
    [ov, xai] = await Promise.all([getJSON("/api/overview"), getJSON("/api/explain-global")]);
  } catch (e) {
    $("#stats").innerHTML = `<div class="card note warn" style="grid-column:1/-1">Could not load results: ${esc(e.message)}</div>`;
    return;
  }
  const ds = ov.dataset, cats = ds.categories || {}, l = ov.lookalike;
  $("#chip").innerHTML = `<span>${ds.classes} characters</span><span class="dot"></span><span>${fmtInt(ds.images)} images</span>`;

  const faith = xai.global?.faithfulness_deletion_test;
  const cards = [
    { label: "Accuracy", value: pct(l?.accuracy.final, 1), unit: "%", tone: "blue", href: "results.html#lookalikes",
      foot: `was <b>${pct(l?.accuracy.svm_vote, 1)}%</b> without specialists` },
    { label: "Characters", value: ds.classes, tone: "gold", href: "script.html",
      foot: `<b>${cats.digit?.classes ?? "–"}</b> digits, ${(ds.classes ?? 0) - (cats.digit?.classes ?? 0)} letters` },
    { label: "Specialists", value: l?.pairs_kept.length ?? "–", tone: "green", href: "results.html#lookalikes",
      foot: "look-alike pairs" },
    { label: "Fixed", value: l?.against_svm_vote.fixed ?? "–", tone: "navy", href: "results.html#lookalikes",
      foot: `broke <b>${l?.against_svm_vote.broken ?? "–"}</b>` },
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

  if (l) {
    const worst = [...l.per_pair].sort((a, b) => b.images_consulted - a.images_consulted).slice(0, 4);
    $("#mini-pairs").innerHTML = `<p style="font-weight:600;margin:0 0 8px">The four busiest specialists: mistakes before and after</p>` + worst.map((p) =>
      `<p class="muted" style="margin:10px 0 4px">${esc(p.pair[0])} ↔ ${esc(p.pair[1])}</p>` +
      bars([{ name: "SVM alone", value: p.images_consulted - p.correct_svm, colour: "var(--sand)" }, { name: "With specialist", value: p.images_consulted - p.correct_final, colour: "var(--navy-800)" }],
        { max: Math.max(...worst.map((q) => q.images_consulted - q.correct_svm)), fmt: (v) => String(v) })).join("");
  }
}

main();
