const FAM_ORDER = ["hog", "lbp", "zernike"];

function shareBar(share) {
  return `<div style="display:flex;height:26px;border-radius:8px;overflow:hidden;box-shadow:0 2px 0 var(--card-edge)">
    ${FAM_ORDER.map((f) => `<span title="${FAMILY[f].name} ${pct(share[f])}%" style="width:${100 * share[f]}%;background:${FAMILY[f].c}"></span>`).join("")}</div>`;
}

function famRows(g) {
  return `<ul class="rows" style="margin-top:14px">${FAM_ORDER.map((f) => `<li><div class="row" style="--accent:${FAMILY[f].c}">
    <span class="title">${FAMILY[f].name}<small>${FAMILY[f].what} · ${fmtInt(g.family_n_features[f])} features in ${g.family_n_groups[f]} groups</small></span>
    <span class="meta">${(g.family_n_features[f] / Object.values(g.family_n_features).reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of inputs</span>
    <span class="pill sm" style="background:${FAMILY[f].c};color:#fff">${pct(g.family_share[f], 0)}%</span></div></li>`).join("")}</ul>`;
}

function overview(g) {
  const top = g.groups.slice(0, 10);
  return `<div class="panel-grid"><div>
    <article class="card">
      <h3>Where the evidence comes from</h3>
      <p class="muted">Share of total |SHAP| over ${g.n_images} test images (${g.images_per_class} per character).</p>
      ${shareBar(g.family_share)}
      <div class="legend" style="margin-top:10px">${FAM_ORDER.map((f) => `<span><i style="background:${FAMILY[f].c}"></i>${FAMILY[f].name} ${pct(g.family_share[f])}%</span>`).join("")}</div>
    </article>
    ${famRows(g)}
    <div class="note" style="margin-top:14px">Zernike moments are 2% of the inputs and carry about ${pct(g.family_share.zernike, 0)}% of the evidence. HOG is 78% of the inputs and carries ${pct(g.family_share.hog, 0)}%. LBP punches above its size.</div>
    </div><div>
    <h2 class="section-title" style="margin-top:0">The ten strongest groups</h2>
    <div class="card">${bars(top.map((x) => ({ name: x.group.replace(/^(HOG|LBP|Zernike): /, "$1 · ").replace("stroke direction, ", "").replace("local stroke pattern, ", ""), title: x.group,
      value: x.mean_abs_shap, colour: FAMILY[x.family].c })), { fmt: (v) => v.toFixed(3) })}
      <p class="muted" style="margin-top:10px">Mean absolute change in the predicted class’s probability.</p></div></div></div>`;
}

function gridMap(grid, colour, title) {
  const flat = grid.flat(), m = Math.max(...flat, 1e-9);
  return `<figure style="margin:0"><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;aspect-ratio:1;max-width:220px">
    ${flat.map((v) => `<span style="display:grid;place-items:center;border-radius:5px;font-size:11px;color:${v / m > .55 ? "#fff" : "var(--ink-soft)"};background:color-mix(in srgb, ${colour} ${Math.round(8 + 92 * v / m)}%, var(--paper-deep))">${v.toFixed(2)}</span>`).join("")}
  </div><figcaption class="muted" style="margin-top:6px">${title}</figcaption></figure>`;
}

const total = (m) => m.flat().reduce((a, b) => a + b, 0) || 1;
const centre = (m) => (m[1][1] + m[1][2] + m[2][1] + m[2][2]) / total(m);
const rowShare = (m, r) => m[r].reduce((a, b) => a + b, 0) / total(m);

function regions(g) {
  const maps = { hog: [...Array(4)].map(() => Array(4).fill(0)), lbp: [...Array(4)].map(() => Array(4).fill(0)) };
  g.groups.forEach((x) => { if (x.cell && maps[x.family]) maps[x.family][x.cell[0]][x.cell[1]] = x.mean_abs_shap; });
  const z = g.groups.filter((x) => x.family === "zernike");
  return `
    <article class="card">
      <h3>Which parts of the character matter</h3>
      <p class="muted">The 64×64 character is cut into a 4×4 grid. Darker squares move predictions more, on average.</p>
      <div class="heat-pair" style="margin-top:12px">${gridMap(maps.hog, "var(--hog)", "HOG: stroke direction")}${gridMap(maps.lbp, "var(--lbp)", "LBP: stroke texture")}</div>
    </article>
    <div class="note" style="margin-top:14px">The centre four squares carry ${pct(centre(maps.hog), 0)}% of the HOG evidence; the top row carries only ${pct(rowShare(maps.hog, 0), 0)}%.
      A plausible reason: the headline (shirorekha) runs across the top of almost every letter, so it says little about which letter it is.</div>
    <h2 class="section-title">Zernike, by level of detail</h2>
    <div class="card">${bars(z.map((x) => ({ name: x.group.replace("Zernike: ", ""), value: x.mean_abs_shap, colour: "var(--zernike)" })), { fmt: (v) => v.toFixed(3) })}
      <p class="muted" style="margin-top:10px">Zernike moments describe the whole shape, so they have no position.</p></div>`;
}

function pairs(p) {
  if (!p?.pairs?.length) return `<div class="note">No contrastive analysis yet.</div>`;
  return `<p class="section-sub" style="margin-top:0">For the pairs the model confuses most: which feature groups push toward the true character rather than its look-alike, averaged over the test images of each pair.</p>
    <div class="card-grid">${p.pairs.map((q, i) => {
      const [a, b] = q.pair;
      const fam = q.family_share_of_distinguishing_evidence || {};
      return `<article class="card">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
          ${[a, b].map((c) => `<figure style="margin:0;text-align:center"><img src="/api/glyph/${encodeURIComponent(c)}" alt="${esc(c)}" width="56" height="56" style="background:#fff;border-radius:8px;box-shadow:inset 0 0 0 1px var(--rule)" loading="lazy" onerror="this.remove()"><figcaption style="font-weight:600">${esc(c)}</figcaption></figure>`).join('<span style="color:var(--gold);font-size:22px">↔</span>')}
          <span class="muted" style="margin-left:auto;text-align:right">${q.n_images} test images</span>
        </div>
        ${shareBar({ hog: fam.hog || 0, lbp: fam.lbp || 0, zernike: fam.zernike || 0 })}
        <div style="margin-top:12px">${bars(q.top_distinguishing_groups.slice(0, 4).map((x) => ({ name: x.group.replace(/^(HOG|LBP|Zernike): /, "$1 · ").replace("stroke direction, ", "").replace("local stroke pattern, ", ""), title: x.group,
          value: x.contrast, colour: FAMILY[x.group.split(":")[0].toLowerCase()]?.c })), { fmt: (v) => v.toFixed(3) })}</div>
      </article>`;
    }).join("")}</div>`;
}

function faith(g) {
  const f = g.faithfulness_deletion_test;
  return `<div class="panel-grid"><div>
    <article class="card">
      <h3>Does SHAP point at what the model really uses?</h3>
      <p>Deletion test: replace the ${f.groups_removed} groups SHAP ranks highest with their average value and see how far the
        predicted class’s probability falls. Then do the same with ${f.groups_removed} random groups.</p>
      ${bars([{ name: `Top ${f.groups_removed} by SHAP`, value: f.mean_probability_drop_top_shap, colour: "var(--navy-800)" },
               { name: `${f.groups_removed} random groups`, value: f.mean_probability_drop_random, colour: "var(--sand)" }], { max: 1, strongFirst: true })}
      <p class="muted" style="margin-top:10px">Mean drop in probability, ${g.n_images} images. A ${(f.mean_probability_drop_top_shap / f.mean_probability_drop_random).toFixed(1)}× larger drop means the explanation is faithful to the model.</p>
    </article>
    </div><div>
    <div class="facts">
      <div class="card fact"><div class="k">Groups</div><div class="v">35</div></div>
      <div class="card fact"><div class="k">Samples per explanation</div><div class="v">${g.nsamples}</div></div>
      <div class="card fact"><div class="k">Additivity error (max)</div><div class="v">${g.max_additivity_error.toExponential(0)}</div></div>
      <div class="card fact"><div class="k">Accuracy on this sample</div><div class="v">${pct(g.accuracy_on_sample, 0)}%</div></div>
    </div>
    <details class="more card" style="margin-top:14px"><summary>Method</summary>
      <p style="margin-top:10px">${esc(g.method)}. Features are grouped (16 HOG regions, 16 LBP regions, 3 Zernike detail bands) so each explanation is about something a reader can point to, instead of 2,261 individual numbers.</p></details></div></div>`;
}

async function main() {
  try {
    const d = await getJSON("/api/explain-global");
    const g = d.global;
    if (!g?.groups) throw new Error("Global SHAP results are not available yet. Run scripts/run_evaluation.py.");
    $("#panel-overview").innerHTML = overview(g);
    $("#panel-regions").innerHTML = regions(g);
    $("#panel-pairs").innerHTML = pairs(d.pairs);
    $("#panel-faith").innerHTML = faith(g);
  } catch (e) {
    $("#panel-overview").innerHTML = `<div class="note warn">${esc(e.message)}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", () => { initTabs(); main(); });
