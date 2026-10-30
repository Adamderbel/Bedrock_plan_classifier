// PlanSense web UI: upload a file, call POST /classify, show one card per page.
// All model output is inserted as text (never innerHTML), so it cannot inject markup.

const $ = (id) => document.getElementById(id);
const form = $("form");
const fileInput = $("file");
const drop = $("drop");
const submit = $("submit");
let prices = null;
let selected = null;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.setAttribute("style", v);
    else node[k] = v;
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const labelColor = (label) => `var(--c-${label || "not_a_plan"})`;
const pretty = (label) => (label || "error").replace(/_/g, " ");

// --- health / defaults -------------------------------------------------------

async function loadHealth() {
  const status = $("status");
  try {
    const res = await fetch("/health");
    const h = await res.json();
    prices = h.prices_usd_per_1m_tokens;
    status.textContent = h.model_id.split(".").slice(-1)[0];
    status.className = "status ok";
    status.title = `${h.model_id} @ ${h.region}`;
  } catch {
    status.textContent = "API unreachable";
    status.className = "status bad";
  }
}

// --- file selection ----------------------------------------------------------

function choose(file) {
  selected = file || null;
  $("fileName").textContent = selected ? `${selected.name} (${(selected.size / 1048576).toFixed(1)} MB)` : "";
  submit.disabled = !selected;
  const preview = $("preview");
  if (selected && selected.type.startsWith("image/")) {
    preview.src = URL.createObjectURL(selected);
    preview.hidden = false;
  } else {
    preview.hidden = true;
    preview.removeAttribute("src");
  }
}

fileInput.addEventListener("change", () => choose(fileInput.files[0]));
for (const ev of ["dragenter", "dragover"]) {
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
}
for (const ev of ["dragleave", "drop"]) {
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
}
drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));

// --- classify ----------------------------------------------------------------

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selected) return;
  const body = new FormData();
  body.append("file", selected);

  $("error").hidden = true;
  $("results").replaceChildren();
  $("summary").hidden = true;
  submit.disabled = true;
  const started = performance.now();
  const timer = setInterval(() => {
    const s = Math.round((performance.now() - started) / 1000);
    $("progressText").textContent = `Classifying… ${s} s (about 10 s per page)`;
  }, 500);
  $("progress").hidden = false;

  try {
    const res = await fetch("/classify", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const retry = res.headers.get("Retry-After");
      throw new Error(`${res.status}: ${data.detail || res.statusText}${retry ? ` (retry in ${retry} s)` : ""}`);
    }
    renderSummary(data, (performance.now() - started) / 1000);
    $("results").replaceChildren(...data.map(renderPage));
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
  } finally {
    clearInterval(timer);
    $("progress").hidden = true;
    submit.disabled = !selected;
  }
});

// --- rendering ---------------------------------------------------------------

function pageCost(usage) {
  if (!prices || !usage) return null;
  return (usage.input_tokens * prices.input + usage.output_tokens * prices.output) / 1e6;
}

function renderSummary(pages, seconds) {
  const counts = {};
  let usd = 0;
  for (const p of pages) {
    counts[p.label || "error"] = (counts[p.label || "error"] || 0) + 1;
    usd += pageCost(p.usage) || 0;
  }
  const stat = (k, v) => el("div", {}, el("span", { class: "k" }, k), el("span", { class: "v" }, v));
  $("summary").replaceChildren(
    stat("Pages", pages.length),
    stat("Labels", Object.entries(counts).map(([l, n]) => `${pretty(l)} ×${n}`).join(", ")),
    stat("Time", `${seconds.toFixed(1)} s`),
    stat("Est. cost", `$${usd.toFixed(4)}`),
    stat("Needs review", pages.filter((p) => p.needs_review).length),
  );
  $("summary").hidden = false;
}

function renderPage(p) {
  const color = labelColor(p.label);
  const head = el("div", { class: "card-head" },
    el("span", { class: "page" }, `Page ${p.page}`),
    el("span", { class: "badge" }, pretty(p.label)),
    p.needs_review && el("span", { class: "flag review" }, "needs review"),
  );

  const parts = [head];

  if (p.error) {
    parts.push(el("div", { class: "page-error" }, p.error));
  } else {
    const pct = Math.round((p.confidence || 0) * 100);
    parts.push(el("div", { class: "conf" },
      el("span", {}, "Confidence"),
      el("div", { class: "bar" }, el("span", { style: `width:${pct}%` })),
      el("strong", {}, `${pct}%`),
    ));
  }

  const tokens = p.usage ? `${p.usage.input_tokens}/${p.usage.output_tokens}` : "";
  const cost = pageCost(p.usage);
  parts.push(el("dl", { class: "meta" },
    el("dt", {}, "Model"), el("dd", {}, p.model_id ?? "-"),
    el("dt", {}, "Latency"), el("dd", {}, `${(p.latency_ms / 1000).toFixed(1)} s`),
    el("dt", {}, "Tokens in/out"), el("dd", {}, tokens || "-"),
    el("dt", {}, "Est. cost"), el("dd", {}, cost == null ? "-" : `$${cost.toFixed(5)}`),
  ));

  return el("article", { class: "card", style: `--label:${color}` }, parts);
}

loadHealth();
