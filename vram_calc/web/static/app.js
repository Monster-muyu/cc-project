"use strict";
const $ = (id) => document.getElementById(id);
const PARALLEL = [1, 2, 4, 8];
const EP_VALS = [1, 2, 4, 8, 16, 32, 64];
// palette (dataviz skill reference)
const C = { w: "#2a78d6", kv: "#eb6834", act: "#1baf7a", oh: "#e87ba4",
            good: "#0ca30c", warn: "#fab219", crit: "#d03b3b", line: "#2a78d6" };
const CAT_LABEL = { llm: "LLM", embedding: "向量", multimodal: "多模态", vision: "视觉" };
let ALL_MODELS = [];
let activeCat = "all";

function currentModelIsMoe() {
  const m = ALL_MODELS.find((x) => x.id === $("model").value);
  return !!(m && m.is_moe);
}

// map "我有 N 张卡" -> best default parallelism (dense:TP, MoE:EP)
function applyParallelStrategy() {
  const n = +$("gpu_count").value || 1;
  $("tp").value = 1; $("pp").value = 1; $("ep").value = 1;
  const hint = $("parallel-hint");
  if (n === 1) { hint.textContent = "单卡部署。模型放不下时调高显卡数量分摊。"; return; }
  if (currentModelIsMoe()) {
    $("ep").value = n;
    hint.textContent = `${n} 卡专家并行(EP)：每卡只装 1/${n} 的专家权重`;
  } else {
    $("tp").value = n;
    hint.textContent = `${n} 卡张量并行(TP)：每卡显存 ≈ 总量 ÷ ${n}`;
  }
}

// accept "4096" / "32k" / "200k" / "1m"
function parseContext(s) {
  s = String(s).trim().toLowerCase().replace(/[,，\s]/g, "");
  const m = s.match(/^(\d+(?:\.\d+)?)([km])?$/);
  if (!m) return NaN;
  let n = parseFloat(m[1]);
  if (m[2] === "k") n *= 1000;
  else if (m[2] === "m") n *= 1000000;
  return Math.round(n);
}
function onContextInput() {
  const n = parseContext($("context_len").value);
  $("ctx-val").textContent = isNaN(n) ? "(无效)" : `= ${n.toLocaleString()}`;
}

async function init() {
  await loadDropdowns();
  for (const id of ["tp", "pp"]) PARALLEL.forEach((n) => $(id).add(new Option(n, n)));
  EP_VALS.forEach((n) => $("ep").add(new Option(n, n)));
  [1, 2, 4, 8].forEach((n) => $("gpu_count").add(new Option(n, n)));
  bindEvents();
  onContextInput();
  onModelChange();
  recalc();
}

async function loadDropdowns() {
  ALL_MODELS = await fetch("/api/models").then((r) => r.json());
  const GPUS = await fetch("/api/gpus").then((r) => r.json());
  buildCategoryFilter();
  filterModels();
  const gs = $("gpu");
  const cur = gs.value;
  gs.innerHTML = "";
  GPUS.forEach((g) => gs.add(new Option(`${g.name} (${g.vram_gb}GB)`, g.id)));
  if (cur) gs.value = cur;
}

function buildCategoryFilter() {
  const cats = ["all", ...new Set(ALL_MODELS.map((m) => m.category))];
  const el = $("cat-filter");
  el.innerHTML = "";
  cats.forEach((cat) => {
    const b = document.createElement("button");
    b.textContent = cat === "all" ? "全部" : (CAT_LABEL[cat] || cat);
    b.dataset.cat = cat;
    if (cat === activeCat) b.classList.add("active");
    b.onclick = () => { activeCat = cat; buildCategoryFilter(); filterModels(); onModelChange(); recalc(); };
    el.appendChild(b);
  });
}

function filterModels() {
  const ms = $("model");
  const cur = ms.value;
  ms.innerHTML = "";
  ALL_MODELS
    .filter((m) => activeCat === "all" || m.category === activeCat)
    .forEach((m) => ms.add(new Option(`${m.name}${m.is_moe ? " · MoE" : ""}`, m.id)));
  if (cur && [...ms.options].some((o) => o.value === cur)) ms.value = cur;
}

function onModelChange() {
  applyParallelStrategy();
  const m = ALL_MODELS.find((x) => x.id === $("model").value);
  const q = $("quant");
  if (m && m.quant) { q.value = m.quant; q.disabled = true; }   // pre-quantized repo -> lock
  else { q.disabled = false; }
}

function currentInput() {
  return {
    model_id: $("model").value, gpu_id: $("gpu").value, quant: $("quant").value,
    context_len: parseContext($("context_len").value) || 4096, concurrency: +$("concurrency").value || 1,
    engine: $("engine").value, tp: +$("tp").value || 1, pp: +$("pp").value || 1,
    ep: +$("ep").value || 1, kv_quant: $("kv_quant").value, cpu_offload: +$("cpu_offload").value,
    safety_factor: +$("gpu_util").value,
    max_num_batched_tokens: +$("max_batch").value || 8192,
  };
}

let timer;
function bindEvents() {
  const debounced = () => { clearTimeout(timer); timer = setTimeout(recalc, 300); };
  ["quant", "gpu", "engine", "tp", "pp", "ep", "kv_quant",
   "context_len", "concurrency", "max_batch", "cpu_offload", "gpu_util", "model", "gpu_count"].forEach((id) =>
    $(id).addEventListener("input", () => {
      if (id === "cpu_offload") $("offload-val").textContent = Math.round(+$("cpu_offload").value * 100) + "%";
      if (id === "gpu_util") $("util-val").textContent = Math.round(+$("gpu_util").value * 100) + "%";
      if (id === "context_len") onContextInput();
      if (id === "model" || id === "gpu_count") onModelChange();
      debounced();
    }));
  $("btn-add-model").onclick = openModelModal;
  $("btn-bulk-model").onclick = openBulkModal;
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
  const map = { ok: ["🟢 放得下", C.good], tight: ["🟡 能跑·会限流", C.warn], over: ["🔴 OOM 放不下", C.crit] };
  const [txt, col] = map[r.verdict];
  const hd = r.headroom_gb;
  const perGpu = r.num_gpus > 1 ? `（每卡 ${r.per_gpu_gb} GB）` : "";
  const v = $("verdict");
  v.className = "verdict " + r.verdict;
  const hdTxt = r.verdict === "over"
    ? `差 ${(-hd).toFixed(2)} GB,加载即 OOM`
    : (hd >= 0 ? `余 ${hd} GB` : `KV 需求超池子 ${(-hd).toFixed(1)} GB → 会限流`);
  v.innerHTML = `<span class="vbig" style="color:${col}">${txt}</span> 总占用 <b>${r.total_gb} GB</b>${perGpu} ` +
    `/ 可用 ${r.usable_gb} GB（${hdTxt}）` +
    (r.num_gpus > 1 ? ` · 共 ${r.num_gpus} 卡` : "");
  $("chart-capacity").innerHTML = capacityBar(r.total_gb, r.capacity_gb, r.usable_gb, r.verdict);
  $("chart-breakdown").innerHTML = stackedBar(r.breakdown, r.total_gb);
  $("breakdown-table").innerHTML = breakdownTable(r.breakdown, r.total_gb);

  // vLLM dynamic KV capacity: adaptive table + recommendations
  const kb = $("kv-capacity");
  if (r.max_kv_tokens > 0) {
    const B = r.max_kv_tokens;
    const uCtx = parseContext($("context_len").value) || 0;
    const uConc = +$("concurrency").value || 1;
    const fmt = (n) => n.toLocaleString();
    const fctx = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.floor(n / 1000) + "k");
    // adaptive concurrency set, always includes the user's value
    const concs = [...new Set([1, 2, 4, 8, 16, 32, 64, 128, uConc])].sort((a, b) => a - b);
    const rows = concs.map((c) => {
      const k = Math.floor(B / c);
      const hi = c === uConc ? ' class="hi"' : "";
      return `<tr${hi}><td>${c} 路</td><td>→ ${fctx(k)} / 请求</td></tr>`;
    }).join("");
    // recommendations driven by the user's inputs
    const keepConcCtx = Math.floor(B / uConc);
    const keepCtxConc = uCtx > 0 ? Math.floor(B / uCtx) : null;
    const safeB = Math.floor(B * 0.80);                 // keep 20% headroom -> lands in 放得下
    const recC = Math.max(1, Math.floor(safeB / 32768));
    const recK = Math.floor(safeB / recC);
    const reqNow = uCtx * uConc;
    let rec = `<div class="rec">▸ 保你的并发 <b>${uConc} 路</b> → 每路最大 <b>${fctx(keepConcCtx)}</b>(临界上限)</div>`;
    if (keepCtxConc !== null)
      rec += `<div class="rec">▸ 保你的上下文 <b>${fctx(uCtx)}</b> → 最多 <b>${keepCtxConc < 1 ? "不足 1 路" : keepCtxConc + " 路"}</b>(临界上限)</div>`;
    rec += `<div class="rec">▸ 推荐(保守,留余量,照着输即可)→ <b>${recC} 路 × ${fctx(recK)}</b> ✓ 放心得下</div>`;
    kb.className = "kv-box";
    kb.innerHTML =
      `💾 vLLM 动态 KV:扣掉权重+开销后剩 <b>${r.kv_budget_gb} GB</b> ≈ <b>${fmt(B)}</b> token 位。` +
      ` <span class="hint">并发↔上下文(预算固定,此消彼长,你的并发高亮):</span>` +
      `<table class="kv-tbl"><tbody>${rows}</tbody></table>${rec}` +
      `<div class="rec hint">你当前 ${fctx(uCtx)}×${uConc} = ${fmt(reqNow)} → ${reqNow <= B ? "✓ 在预算内" : "⚠ 超出"}</div>`;
  } else { kb.className = "kv-box empty"; }
}

// ---- SVG charts ----
function capacityBar(total, capacity, usable, verdict) {
  const W = 360, H = 48, pad = 4, inner = W - 2 * pad;
  const frac = Math.min(total / capacity, 1);
  const col = verdict === "over" ? C.crit : verdict === "tight" ? C.warn : C.good;
  const usableX = pad + (usable / capacity) * inner;
  const over = total > capacity;
  return `<svg width="${W}" height="${H}" class="chart">
    <rect x="${pad}" y="${pad}" width="${inner}" height="${H - 2 * pad}" fill="#eee" rx="6"/>
    <rect x="${pad}" y="${pad}" width="${(frac * inner).toFixed(1)}" height="${H - 2 * pad}" fill="${col}" rx="6"/>
    <line x1="${usableX.toFixed(1)}" y1="2" x2="${usableX.toFixed(1)}" y2="${H - 2}" stroke="#555" stroke-dasharray="3,2"/>
    <text x="${pad + 5}" y="${H - 8}" font-size="11.5" fill="#0b0b0b" font-weight="700">${total} GB${over ? " (超容量)" : ""}</text>
    <text x="${W - pad - 4}" y="${H - 8}" font-size="11" fill="#52514e" text-anchor="end">容量 ${capacity} GB</text>
  </svg>`;
}

function stackedBar(bd, total) {
  const comps = [["权重", bd.weights, C.w], ["KV", bd.kv_cache, C.kv],
                 ["激活", bd.activation, C.act], ["开销", bd.overhead, C.oh]];
  const W = 360, H = 40, pad = 4, inner = W - 2 * pad, gap = 2;
  let x = pad;
  const segs = comps.filter(([, v]) => v > 0).map(([n, v, c]) => {
    const w = total > 0 ? (v / total) * inner - gap : 0;
    const s = `<rect x="${x.toFixed(1)}" y="${pad}" width="${Math.max(w, 0).toFixed(1)}" height="${H - 2 * pad}" fill="${c}" rx="2"/>`;
    x += w + gap; return s;
  }).join("");
  const legend = comps.filter(([, v]) => v > 0)
    .map(([n, v, c]) => `<span class="leg" style="--c:${c}">${n} ${v}GB</span>`).join("");
  return `<svg width="${W}" height="${H}" class="chart">${segs}</svg><div class="legend">${legend}</div>`;
}

function breakdownTable(bd, total) {
  const zh = { weights: "权重", kv_cache: "KV cache", activation: "激活", overhead: "引擎开销" };
  const rows = Object.entries(bd).filter(([, v]) => v > 0).map(([k, v]) => {
    const pct = total > 0 ? ((v / total) * 100).toFixed(0) : 0;
    return `<tr><td>${zh[k] || k}</td><td>${v}</td><td>${pct}%</td></tr>`;
  }).join("");
  return `<table><tbody>${rows}<tr class="tot"><td>合计(固定)</td><td>${total}</td><td>100%</td></tr></tbody></table>`;
}

function sweepChart(s) {
  const pts = s.points;
  if (!pts.length) return "<p class='hint'>无数据</p>";
  const W = 360, H = 205, pl = 46, pb = 30, pr = 10, pt = 12;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.total_gb);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(...ys, s.capacity_gb) * 1.08;
  const X = (x) => pl + ((x - xmin) / (xmax - xmin)) * (W - pl - pr);
  const Y = (y) => H - pb - (y / ymax) * (H - pt - pb);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.total_gb).toFixed(1)}`).join(" ");
  const capY = Y(s.capacity_gb), useY = Y(s.usable_gb);

  // y-axis gridlines + GB tick labels
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4, y = Y(v);
    grid += `<line x1="${pl}" y1="${y.toFixed(1)}" x2="${W - pr}" y2="${y.toFixed(1)}" stroke="#e1e0d9"/>` +
            `<text x="${pl - 6}" y="${(y + 3).toFixed(1)}" font-size="9.5" fill="#898781" text-anchor="end">${Math.round(v)}</text>`;
  }
  // x-axis ticks (concurrency values)
  const every = pts.length > 9 ? 2 : 1;
  let ticks = "";
  pts.forEach((p, i) => {
    if (i % every) return;
    ticks += `<text x="${X(p.x).toFixed(1)}" y="${H - 12}" font-size="9.5" fill="#898781" text-anchor="middle">${p.x}</text>`;
  });

  let mark = "";
  if (s.max_x) {
    const p = pts.find((p) => p.x === s.max_x);
    if (p) mark = `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.total_gb).toFixed(1)}" r="3.5" fill="${C.crit}"/>` +
      `<text x="${X(p.x).toFixed(1)}" y="${Y(p.total_gb).toFixed(1) - 8}" font-size="10.5" fill="${C.crit}" text-anchor="middle" font-weight="700">最大 ${s.max_x}</text>`;
  }
  return `<svg width="${W}" height="${H}" class="chart">
    ${grid}${ticks}
    <line x1="${pl}" y1="${capY}" x2="${W - pr}" y2="${capY}" stroke="${C.crit}" stroke-dasharray="4,2"/>
    <text x="${W - pr}" y="${capY - 4}" font-size="9.5" fill="${C.crit}" text-anchor="end">容量 ${s.capacity_gb}GB</text>
    <line x1="${pl}" y1="${useY}" x2="${W - pr}" y2="${useY}" stroke="${C.warn}" stroke-dasharray="3,2"/>
    <text x="${pl + 2}" y="${useY - 4}" font-size="9.5" fill="${C.warn}" text-anchor="start">可用 ${s.usable_gb}GB</text>
    <path d="${line}" fill="none" stroke="${C.line}" stroke-width="2"/>
    ${mark}
    <text x="${pl - 34}" y="${pt}" font-size="10" fill="#52514e">GB</text>
    <text x="${W - pr}" y="${H - 1}" font-size="10" fill="#52514e" text-anchor="end">并发数 →</text>
  </svg>
  <div class="legend">
    <span class="leg" style="--c:${C.line}">总占用(随并发↑)</span>
    <span class="leg" style="--c:${C.warn}">可用(利用率×容量)</span>
    <span class="leg" style="--c:${C.crit}">显存容量</span>
  </div>`;
}

// ---- add model (single) ----
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
  const cat = $("mf-cat").value;
  const m = await fetch(`/api/models/preview?repo_id=${encodeURIComponent(repo)}&category=${cat}&source=${$("mf-source").value}`).then((r) => r.json());
  if (m.error) { $("mf-error").textContent = m.error; return; }
  $("mf-name").value = m.name; $("mf-params").value = m.params_b;
  $("mf-layers").value = m.layers; $("mf-hidden").value = m.hidden_dim;
  $("mf-attn").value = m.attn_heads; $("mf-kv").value = m.kv_heads;
  $("mf-headdim").value = m.head_dim; $("mf-experts").value = m.num_experts;
  $("mf-expertparams").value = m.expert_params_b;
  $("mf-fields").classList.remove("hidden");
  $("mf-save").disabled = false;
  $("mf-error").textContent = "已解析，请核对 KV 头数等关键字段后入库。";
};
$("mf-save").onclick = async () => {
  const spec = {
    id: $("mf-repo").value.trim(), name: $("mf-name").value, category: $("mf-cat").value,
    params_b: +$("mf-params").value, layers: +$("mf-layers").value, hidden_dim: +$("mf-hidden").value,
    attn_heads: +$("mf-attn").value, kv_heads: +$("mf-kv").value, head_dim: +$("mf-headdim").value,
    num_experts: +$("mf-experts").value, expert_params_b: +$("mf-expertparams").value,
  };
  const r = await fetch("/api/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }).then((r) => r.json());
  if (r.error) { $("mf-error").textContent = r.error; return; }
  $("modal-model").classList.add("hidden");
  await loadDropdowns(); $("model").value = spec.id; onModelChange(); recalc();
};

// ---- bulk import ----
function openBulkModal() {
  $("bf-list").value = ""; $("bf-result").textContent = "";
  $("modal-bulk").classList.remove("hidden");
}
$("bf-cancel").onclick = () => $("modal-bulk").classList.add("hidden");
$("bf-go").onclick = async () => {
  const ids = $("bf-list").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!ids.length) { $("bf-result").textContent = "请输入至少一个 Repo ID"; return; }
  $("bf-result").className = "hint"; $("bf-result").textContent = `正在拉取 ${ids.length} 个模型…`;
  const r = await fetch("/api/models/bulk", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_ids: ids, category: $("bf-cat").value, source: $("bf-source").value }),
  }).then((r) => r.json());
  const fail = r.failed || [];
  $("bf-result").className = fail.length ? "error" : "success";
  $("bf-result").innerHTML = `✅ 成功 ${r.saved.length} 个${fail.length ? ` · ❌ 失败 ${fail.length}：` + fail.map((f) => f.id).join("、") : ""}`;
  await loadDropdowns();
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
  await loadDropdowns(); $("gpu").value = spec.id; recalc();
};

init();
