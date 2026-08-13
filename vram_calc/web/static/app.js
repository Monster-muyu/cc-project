"use strict";
const $ = (id) => document.getElementById(id);
const PARALLEL = [1, 2, 4, 8];
const EP_VALS = [1, 2, 4, 8, 16, 32, 64];
let MODELS = [];

async function init() {
  await loadDropdowns();
  for (const id of ["tp", "pp"]) PARALLEL.forEach((n) => $(id).add(new Option(n, n)));
  EP_VALS.forEach((n) => $("ep").add(new Option(n, n)));
  bindEvents();
  onModelChange();
  recalc();
}

async function loadDropdowns() {
  MODELS = await fetch("/api/models").then((r) => r.json());
  const GPUS = await fetch("/api/gpus").then((r) => r.json());
  const ms = $("model");
  ms.innerHTML = "";
  MODELS.forEach((m) => ms.add(new Option(m.name, m.id)));
  const gs = $("gpu");
  gs.innerHTML = "";
  GPUS.forEach((g) => gs.add(new Option(`${g.name} (${g.vram_gb}GB)`, g.id)));
}

function onModelChange() {
  const m = MODELS.find((x) => x.id === $("model").value);
  const moe = !!(m && m.is_moe);
  $("ep-label").style.display = moe ? "" : "none";
  if (!moe) $("ep").value = "1";
}

function currentInput() {
  return {
    model_id: $("model").value,
    gpu_id: $("gpu").value,
    quant: $("quant").value,
    context_len: +$("context_len").value || 4096,
    concurrency: +$("concurrency").value || 1,
    engine: $("engine").value,
    tp: +$("tp").value || 1,
    pp: +$("pp").value || 1,
    ep: +$("ep").value || 1,
    kv_quant: $("kv_quant").value,
    cpu_offload: +$("cpu_offload").value,
  };
}

let timer;
function bindEvents() {
  const debounced = () => { clearTimeout(timer); timer = setTimeout(recalc, 300); };
  ["model", "quant", "gpu", "engine", "tp", "pp", "ep", "kv_quant",
   "context_len", "concurrency", "cpu_offload"].forEach((id) =>
    $(id).addEventListener("input", () => {
      if (id === "cpu_offload")
        $("offload-val").textContent = Math.round(+$("cpu_offload").value * 100) + "%";
      if (id === "model") onModelChange();
      debounced();
    }));
  $("btn-add-model").onclick = openModelModal;
  $("btn-add-gpu").onclick = openGpuModal;
}

async function recalc() {
  const body = currentInput();
  const opt = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch("/api/calc", opt).then((r) => r.json());
  if (r.error) { $("verdict").textContent = r.error; return; }
  renderResult(r);
  const s = await fetch("/api/sweep?sweep_var=concurrency&x0=1&x1=16", opt).then((r) => r.json());
  if (!s.error) $("chart-sweep").innerHTML = sweepChart(s);
}

function renderResult(r) {
  const map = { ok: ["🟢 放得下", "#2a9d3f"], tight: ["🟡 偏紧", "#e0a800"], over: ["🔴 放不下", "#c0392b"] };
  const [txt, col] = map[r.verdict];
  const hd = r.headroom_gb;
  $("verdict").innerHTML =
    `<span class="vbig" style="color:${col}">${txt}</span> 总占用 <b>${r.total_gb} GB</b> ` +
    `/ 可用 ${r.usable_gb} GB（${hd >= 0 ? "余 " + hd : "差 " + (-hd).toFixed(2)} GB）` +
    (r.num_gpus > 1 ? ` · ${r.num_gpus} 卡并行` : "");
  $("chart-capacity").innerHTML = capacityBar(r.total_gb, r.capacity_gb, r.usable_gb, r.verdict);
  $("chart-breakdown").innerHTML = stackedBar(r.breakdown, r.total_gb);
  $("breakdown-table").innerHTML = breakdownTable(r.breakdown, r.total_gb);
}

// ---- SVG charts ----
function capacityBar(total, capacity, usable, verdict) {
  const W = 360, H = 48, pad = 4, inner = W - 2 * pad;
  const frac = Math.min(total / capacity, 1);
  const col = verdict === "over" ? "#c0392b" : verdict === "tight" ? "#e0a800" : "#2a9d3f";
  const usableX = pad + (usable / capacity) * inner;
  const over = total > capacity;
  return `<svg width="${W}" height="${H}" class="chart">
    <rect x="${pad}" y="${pad}" width="${inner}" height="${H - 2 * pad}" fill="#eee" rx="5"/>
    <rect x="${pad}" y="${pad}" width="${(frac * inner).toFixed(1)}" height="${H - 2 * pad}" fill="${col}" rx="5"/>
    <line x1="${usableX.toFixed(1)}" y1="2" x2="${usableX.toFixed(1)}" y2="${H - 2}" stroke="#555" stroke-dasharray="3,2"/>
    <text x="${pad + 4}" y="${H - 7}" font-size="11" fill="#222" font-weight="bold">${total} GB${over ? " (超容量)" : ""}</text>
    <text x="${W - pad - 4}" y="${H - 7}" font-size="11" fill="#666" text-anchor="end">容量 ${capacity} GB</text>
  </svg>`;
}

function stackedBar(bd, total) {
  const comps = [["权重", bd.weights, "#3b6fb6"], ["KV", bd.kv_cache, "#9b59b6"],
                 ["激活", bd.activation, "#1abc9c"], ["开销", bd.overhead, "#95a5a6"]];
  const W = 360, H = 38, pad = 4, inner = W - 2 * pad;
  let x = pad;
  const segs = comps.map(([n, v, c]) => {
    const w = total > 0 ? (v / total) * inner : 0;
    const s = `<rect x="${x.toFixed(1)}" y="${pad}" width="${w.toFixed(1)}" height="${H - 2 * pad}" fill="${c}"/>`;
    x += w; return s;
  }).join("");
  const legend = comps.filter(([n, v]) => v > 0)
    .map(([n, v, c]) => `<span class="leg" style="background:${c}">${n} ${v}GB</span>`).join("");
  return `<svg width="${W}" height="${H}" class="chart">${segs}` +
         `<rect x="${pad}" y="${pad}" width="${inner}" height="${H - 2 * pad}" fill="none" stroke="#ccc" rx="5"/></svg>` +
         `<div class="legend">${legend}</div>`;
}

function breakdownTable(bd, total) {
  const zh = { weights: "权重", kv_cache: "KV cache", activation: "激活", overhead: "引擎开销" };
  const rows = Object.entries(bd).map(([k, v]) => {
    const pct = total > 0 ? ((v / total) * 100).toFixed(0) : 0;
    return `<tr><td>${zh[k] || k}</td><td>${v} GB</td><td>${pct}%</td></tr>`;
  }).join("");
  return `<table><tbody>${rows}<tr class="tot"><td>合计</td><td>${total} GB</td><td>100%</td></tr></tbody></table>`;
}

function sweepChart(s) {
  const pts = s.points;
  if (!pts.length) return "<p>无数据</p>";
  const W = 360, H = 170, pl = 38, pb = 22, pr = 8, pt = 10;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.total_gb);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(...ys, s.capacity_gb) * 1.1;
  const X = (x) => pl + ((x - xmin) / (xmax - xmin)) * (W - pl - pr);
  const Y = (y) => H - pb - (y / ymax) * (H - pt - pb);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.total_gb).toFixed(1)}`).join(" ");
  const capY = Y(s.capacity_gb), useY = Y(s.usable_gb);
  let mark = "";
  if (s.max_x) {
    const p = pts.find((p) => p.x === s.max_x);
    if (p) mark = `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.total_gb).toFixed(1)}" r="3.5" fill="#c0392b"/>` +
      `<text x="${X(p.x).toFixed(1)}" y="${Y(p.total_gb).toFixed(1) - 7}" font-size="10" fill="#c0392b" text-anchor="middle">最大 ${s.max_x}</text>`;
  }
  return `<svg width="${W}" height="${H}" class="chart">
    <line x1="${pl}" y1="${capY}" x2="${W - pr}" y2="${capY}" stroke="#c0392b" stroke-dasharray="4,2"/>
    <text x="${W - pr}" y="${capY - 4}" font-size="9" fill="#c0392b" text-anchor="end">容量 ${s.capacity_gb}GB</text>
    <line x1="${pl}" y1="${useY}" x2="${W - pr}" y2="${useY}" stroke="#e0a800" stroke-dasharray="3,2"/>
    <path d="${line}" fill="none" stroke="#3b6fb6" stroke-width="2"/>
    ${mark}
    <text x="${pl}" y="${H - 6}" font-size="9" fill="#666">并发数 →</text>
  </svg>`;
}

// ---- add model ----
function openModelModal() {
  $("mf-fields").classList.add("hidden");
  $("mf-save").disabled = true;
  $("mf-error").textContent = "";
  $("mf-repo").value = "";
  $("modal-model").classList.remove("hidden");
}
$("mf-cancel").onclick = () => $("modal-model").classList.add("hidden");
$("mf-fetch").onclick = async () => {
  $("mf-error").textContent = "拉取中…";
  const repo = $("mf-repo").value.trim();
  if (!repo) return;
  const m = await fetch(`/api/models/preview?repo_id=${encodeURIComponent(repo)}`).then((r) => r.json());
  if (m.error) { $("mf-error").textContent = m.error; return; }
  $("mf-name").value = m.name;
  $("mf-params").value = m.params_b;
  $("mf-layers").value = m.layers;
  $("mf-hidden").value = m.hidden_dim;
  $("mf-attn").value = m.attn_heads;
  $("mf-kv").value = m.kv_heads;
  $("mf-headdim").value = m.head_dim;
  $("mf-experts").value = m.num_experts;
  $("mf-expertparams").value = m.expert_params_b;
  $("mf-fields").classList.remove("hidden");
  $("mf-save").disabled = false;
  $("mf-error").textContent = "已解析，请核对 KV 头数等关键字段后入库。";
};
$("mf-save").onclick = async () => {
  const spec = {
    id: $("mf-repo").value.trim(), name: $("mf-name").value,
    params_b: +$("mf-params").value, layers: +$("mf-layers").value,
    hidden_dim: +$("mf-hidden").value, attn_heads: +$("mf-attn").value,
    kv_heads: +$("mf-kv").value, head_dim: +$("mf-headdim").value,
    num_experts: +$("mf-experts").value, expert_params_b: +$("mf-expertparams").value,
  };
  const r = await fetch("/api/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }).then((r) => r.json());
  if (r.error) { $("mf-error").textContent = r.error; return; }
  $("modal-model").classList.add("hidden");
  await loadDropdowns();
  $("model").value = spec.id;
  onModelChange();
  recalc();
};

// ---- add gpu ----
function openGpuModal() {
  $("gf-error").textContent = "";
  $("gf-id").value = $("gf-name").value = $("gf-vram").value = "";
  $("modal-gpu").classList.remove("hidden");
}
$("gf-cancel").onclick = () => $("modal-gpu").classList.add("hidden");
$("gf-save").onclick = async () => {
  const spec = { id: $("gf-id").value.trim(), name: $("gf-name").value, vram_gb: +$("gf-vram").value };
  const r = await fetch("/api/gpus", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }).then((r) => r.json());
  if (r.error) { $("gf-error").textContent = r.error; return; }
  $("modal-gpu").classList.add("hidden");
  await loadDropdowns();
  $("gpu").value = spec.id;
  recalc();
};

init();
