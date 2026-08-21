"use strict";
// $ lives in common.js (querySelector-based); app.js historically did getElementById
const $id = (id) => document.getElementById(id);
const PARALLEL = [1, 2, 4, 8, 16, 32, 64];
const EP_VALS = [1, 2, 4, 8, 16, 32, 64];
// palette (dataviz skill reference)
const C = { w: "#2a78d6", kv: "#eb6834", act: "#1baf7a", oh: "#e87ba4",
            good: "#0ca30c", warn: "#fab219", crit: "#d03b3b", line: "#2a78d6" };
const CAT_LABEL = { llm: "LLM", embedding: "向量", multimodal: "多模态", vision: "视觉" };
let ALL_MODELS = [];
let ALL_GPUS = [];
let activeCat = "all";

function currentModel() {
  return ALL_MODELS.find((x) => x.id === $id("model").value);
}
function currentModelIsMoe() {
  return !!currentModel()?.is_moe;
}

// sync TP/PP/EP-vs-gpu-count consistency + vLLM head-divisibility hints
function syncParallelHint() {
  const th = $id("tp-hint");
  if (!th) return;
  const n = +$id("gpu_count").value || 1;
  const t = +$id("tp").value || 1, p = +$id("pp").value || 1, e = +$id("ep").value || 1;
  const used = t * p * (currentModelIsMoe() ? e : 1);
  const msgs = [];
  if (used !== n)
    msgs.push(`当前 TP×PP${currentModelIsMoe() ? "×EP" : ""} = ${used} 卡,与显卡数量 ${n} 不一致`);
  // vLLM/SGLang require total attention heads % TP == 0 (odd GPU counts often fail)
  const eng = $id("engine").value;
  const m = currentModel();
  if ((eng === "vllm" || eng === "sglang") && m && m.attn_heads && t > 1 && m.attn_heads % t !== 0)
    msgs.push(`${eng} 要求注意力头数(${m.attn_heads})能被 TP=${t} 整除,否则无法启动——建议改用 PP 或调整为能整除的卡数`);
  th.textContent = msgs.join("; ");
  // also surface the head-divisibility blocker OUTSIDE the collapsed details
  const warn = $id("tp-warn");
  if (warn) {
    const blocked = (eng === "vllm" || eng === "sglang") && m && m.attn_heads
      && t > 1 && m.attn_heads % t !== 0;
    warn.hidden = !blocked;
    if (blocked) warn.textContent =
      `⚠️ ${eng} 无法启动:${m.name} 注意力头数 ${m.attn_heads} 不能被 TP=${t} 整除。改用 PP、换能整除的卡数,或换 llama.cpp 引擎`;
  }
}

// user manually changed one of TP/PP/EP -> rebalance the others so the
// product still equals gpu_count (e.g. 3 cards: TP=1 -> PP up to 3; TP=3 -> PP back to 1)
function rebalanceParallel(changed) {
  const n = +$id("gpu_count").value || 1;
  const moe = currentModelIsMoe();
  let t = Math.max(1, +$id("tp").value || 1);
  let p = Math.max(1, +$id("pp").value || 1);
  let e = Math.max(1, +$id("ep").value || 1);
  if (!moe) e = 1;                       // EP is meaningless for dense models
  // the dimension the user just set wins; the remaining one absorbs the residue
  const residue = Math.max(1, Math.floor(n / (t * (moe ? e : 1))));
  const residueE = Math.max(1, Math.floor(n / (t * p)));
  if (changed === "tp") { p = residue; if (moe) e = Math.max(1, Math.floor(n / (t * p))); }
  else if (changed === "pp") { t = Math.max(1, Math.floor(n / p)); if (moe) e = 1; if (moe) e = Math.max(1, Math.floor(n / (t * p))); }
  else if (changed === "ep") { if (moe) { t = Math.max(1, Math.floor(n / e)); p = Math.max(1, Math.floor(n / (t * e))); } }
  $id("tp").value = t; $id("pp").value = p; $id("ep").value = e;
  const used = t * p * (moe ? e : 1);
  $id("parallel-hint").textContent = used === n
    ? `${used} 卡组合:TP=${t} × PP=${p}${moe ? ` × EP=${e}` : ""}`
    : `TP=${t} × PP=${p}${moe ? ` × EP=${e}` : ""} = ${used} 卡,显卡数量 ${n} 无法整除配置(需 TP×PP${moe ? "×EP" : ""} = ${n})`;
  syncParallelHint();
}

// map "我有 N 张卡" -> best default parallelism (dense:TP, MoE:EP; any N incl. odd works)
function applyParallelStrategy() {
  const n = +$id("gpu_count").value || 1;
  $id("tp").value = 1; $id("pp").value = 1; $id("ep").value = 1;
  const hint = $id("parallel-hint");
  if (n === 1) { hint.textContent = "单卡部署。模型放不下时调高显卡数量分摊。"; syncParallelHint(); return; }
  if (currentModelIsMoe()) {
    $id("ep").value = n;
    hint.textContent = `${n} 卡专家并行(EP)：每卡只装 1/${n} 的专家权重`;
  } else {
    $id("tp").value = n;
    hint.textContent = `${n} 卡张量并行(TP)：每卡显存 ≈ 总量 ÷ ${n}`;
  }
  syncParallelHint();
}

// accept "4096" / "32k" / "200k" / "1m"  (parseContext lives in common.js)
function onContextInput() {
  const n = parseContext($id("context_len").value);
  // parseContext returns 0 (not NaN) for junk -- keep the "(无效)" hint
  const bad = !n || isNaN(n);
  $id("ctx-val").textContent = bad ? "(无效)" : `= ${n.toLocaleString()}`;
}

async function init() {
  await loadDropdowns();
  // tp/pp/ep are now free number inputs -- nothing to populate
  bindEvents();
  onContextInput();
  onModelChange();
  recalc();
}

async function loadDropdowns() {
  ALL_MODELS = await fetch("/api/models").then((r) => r.json());
  ALL_GPUS = await fetch("/api/gpus").then((r) => r.json());
  buildCategoryFilter();
  filterModels();
  const gs = $id("gpu");
  const cur = gs.value;
  gs.innerHTML = "";
  ALL_GPUS.forEach((g) => gs.add(new Option(`${g.name} (${g.vram_gb}GB)`, g.id)));
  if (cur) gs.value = cur;
}

// FP8 hardware check: FP8 compute units exist only from Ada(sm89)/Hopper(sm90)+.
// On older cards (3090/A100=Ampere sm80-86): FP8 KV cache -> vLLM refuses to start;
// FP8 weights -> runs but is fake-fp8 (software dequant, no speedup).
function syncGpuCapability() {
  const el = $id("fp8-warn");
  if (!el) return;
  const g = ALL_GPUS.find((x) => x.id === $id("gpu").value);
  if (!g || g.supports_fp8 === undefined) { el.hidden = true; return; }
  const q = $id("quant").value, kvq = $id("kv_quant").value;
  const eng = $id("engine").value;
  const msgs = [];
  if (!g.supports_fp8) {
    if (kvq === "fp8")
      msgs.push(`KV 量化 fp8:${g.name}(${g.architecture}) 没有 FP8 硬件单元,${eng} 启动时会直接报错(compute capability 不支持)——改回 fp16/int8`);
    if (q === "fp8")
      msgs.push(`权重量化 fp8:${g.name} 无 FP8 计算路径,即使能加载也只是软件反量化模拟(不省算力、可能更慢)——建议 int8/int4`);
  }
  el.hidden = msgs.length === 0;
  el.textContent = msgs.join("; ");
}

function buildCategoryFilter() {
  const cats = ["all", ...new Set(ALL_MODELS.map((m) => m.category))];
  const el = $id("cat-filter");
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
  const ms = $id("model");
  const cur = ms.value;
  ms.innerHTML = "";
  ALL_MODELS
    .filter((m) => activeCat === "all" || m.category === activeCat)
    .forEach((m) => ms.add(new Option(`${m.name}${m.is_moe ? " · MoE" : ""}`, m.id)));
  if (cur && [...ms.options].some((o) => o.value === cur)) ms.value = cur;
}

function onModelChange() {
  applyParallelStrategy();
  const m = ALL_MODELS.find((x) => x.id === $id("model").value);
  const q = $id("quant");
  if (m && m.quant) { q.value = m.quant; q.disabled = true; }   // pre-quantized repo -> lock
  else { q.disabled = false; }
  syncGpuCapability();
}

function currentInput() {
  return {
    model_id: $id("model").value, gpu_id: $id("gpu").value, quant: $id("quant").value,
    context_len: parseContext($id("context_len").value) || 4096, concurrency: +$id("concurrency").value || 1,
    engine: $id("engine").value, tp: +$id("tp").value || 1, pp: +$id("pp").value || 1,
    ep: +$id("ep").value || 1, kv_quant: $id("kv_quant").value, cpu_offload: +$id("cpu_offload").value,
    safety_factor: +$id("gpu_util").value,
    max_num_batched_tokens: +$id("max_batch").value || 8192,
  };
}

let timer;
function bindEvents() {
  const debounced = () => { clearTimeout(timer); timer = setTimeout(recalc, 300); };
  ["quant", "gpu", "engine", "tp", "pp", "ep", "kv_quant",
   "context_len", "concurrency", "max_batch", "cpu_offload", "gpu_util", "model", "gpu_count"].forEach((id) =>
    $id(id).addEventListener("input", () => {
      if (id === "cpu_offload") $id("offload-val").textContent = Math.round(+$id("cpu_offload").value * 100) + "%";
      if (id === "gpu_util") $id("util-val").textContent = Math.round(+$id("gpu_util").value * 100) + "%";
      if (id === "context_len") onContextInput();
      if (id === "model" || id === "gpu_count") onModelChange();
      if (id === "tp" || id === "pp" || id === "ep") rebalanceParallel(id);
      if (id === "engine") syncParallelHint();
      if (id === "gpu" || id === "quant" || id === "kv_quant" || id === "engine") syncGpuCapability();
      debounced();
    }));
  $id("btn-add-model").onclick = openModelModal;
  $id("btn-bulk-model").onclick = openBulkModal;
  $id("btn-add-gpu").onclick = openGpuModal;
}

async function recalc() {
  const body = currentInput();
  const opt = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch("/api/calc", opt).then((r) => r.json());
  if (r.error) { $id("verdict").textContent = r.error; return; }
  renderResult(r);
  window.__lastCalc = r;
  const s = await fetch("/api/sweep?sweep_var=concurrency&x0=1", opt).then((r) => r.json());
  if (!s.error) $id("chart-sweep").innerHTML = sweepChart(s, "并发数");
}

function renderResult(r) {
  const map = { ok: ["🟢 放得下", C.good], tight: ["🟡 能跑·会限流", C.warn], over: ["🔴 OOM 放不下", C.crit] };
  const [txt, col] = map[r.verdict];
  const hd = r.headroom_gb;
  const perGpu = r.num_gpus > 1 ? `（每卡 ${r.per_gpu_gb} GB）` : "";
  const v = $id("verdict");
  v.className = "verdict " + r.verdict;
  const hdTxt = r.verdict === "over"
    ? `差 ${(-hd).toFixed(2)} GB,加载即 OOM`
    : (hd >= 0 ? `余 ${hd} GB` : `KV 需求超池子 ${(-hd).toFixed(1)} GB → 会限流`);
  v.innerHTML = `<span class="vbig" style="color:${col}">${txt}</span> 总占用 <b>${r.total_gb} GB</b>${perGpu} ` +
    `/ 可用 ${r.usable_gb} GB（${hdTxt}）` +
    (r.num_gpus > 1 ? ` · 共 ${r.num_gpus} 卡` : "") +
    (r.calibrated ? ` · <span title="引擎开销已按你的真实日志标定">📐已标定</span>` : "");
  $id("chart-capacity").innerHTML = capacityBar(r.total_gb, r.capacity_gb, r.usable_gb, r.verdict);
  $id("chart-breakdown").innerHTML = stackedBar(r.breakdown, r.total_gb);
  $id("breakdown-table").innerHTML = breakdownTable(r.breakdown, r.total_gb);

  // vLLM dynamic KV capacity: adaptive table + recommendations
  const kb = $id("kv-capacity");
  if (r.max_kv_tokens > 0) {
    const B = r.max_kv_tokens;
    const uCtx = parseContext($id("context_len").value) || 0;
    const uConc = +$id("concurrency").value || 1;
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
      (r.decode_tps > 0 ? ` 预估 decode ≈ <b>${Math.round(r.decode_tps)} tok/s</b><span class="hint">(带宽近似上限)</span>。` : "") +
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

function sweepChart(s, sweepLabel = "并发数") {
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

  // y-axis gridlines only (no numeric ticks: the two labeled reference lines carry the scale)
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4, y = Y(v);
    grid += `<line x1="${pl}" y1="${y.toFixed(1)}" x2="${W - pr}" y2="${y.toFixed(1)}" stroke="#e1e0d9"/>`;
  }
  // x-axis: no per-point ticks (values can overflow any fixed range); units only
  const ticks = "";

  let mark = "";
  if (s.max_x) {
    const p = pts.find((p) => p.x === s.max_x);
    if (p) mark = `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.total_gb).toFixed(1)}" r="3.5" fill="${C.crit}"/>` +
      `<text x="${X(p.x).toFixed(1)}" y="${Y(p.total_gb).toFixed(1) - 8}" font-size="10.5" fill="${C.crit}" text-anchor="middle" font-weight="700">最大 ${s.max_x}</text>`;
  } else if (pts.length && pts[0].total_gb > s.capacity_gb) {
    // even 1 concurrent request exceeds the budget -> no mark, explicit hint
    mark = `<text x="${(pl + W - pr) / 2}" y="${pt + 2}" font-size="10.5" fill="${C.crit}" text-anchor="middle" font-weight="700">预算装不下单路请求:减上下文/加卡/提高利用率</text>`;
  } else {
    // whole range fits: real max is beyond x1, say so instead of a misleading marker
    mark = `<text x="${(pl + W - pr) / 2}" y="${pt + 2}" font-size="10.5" fill="${C.good}" text-anchor="middle" font-weight="700">全范围都放得下,最大并发 > ${pts[pts.length - 1].x}</text>`;
  }
  return `<svg width="${W}" height="${H}" class="chart">
    ${grid}${ticks}
    <line x1="${pl}" y1="${capY}" x2="${W - pr}" y2="${capY}" stroke="${C.crit}" stroke-dasharray="4,2"/>
    <text x="${W - pr}" y="${capY - 4}" font-size="9.5" fill="${C.crit}" text-anchor="end">容量 ${s.capacity_gb}GB</text>
    <line x1="${pl}" y1="${useY}" x2="${W - pr}" y2="${useY}" stroke="${C.warn}" stroke-dasharray="3,2"/>
    <text x="${pl + 2}" y="${useY - 4}" font-size="9.5" fill="${C.warn}" text-anchor="start">可用 ${s.usable_gb}GB</text>
    <path d="${line}" fill="none" stroke="${C.kv}" stroke-width="2"/>
    ${mark}
    <text x="4" y="${pt + 4}" font-size="10" fill="#52514e">GB</text>
    <text x="${W - pr}" y="${H - 1}" font-size="10" fill="#52514e" text-anchor="end">${sweepLabel} →</text>
  </svg>
  <div class="legend">
    <span class="leg" style="--c:${C.kv}">KV需求(上下文×并发)</span>
    <span class="leg" style="--c:${C.warn}">KV预算(撞线=满载)</span>
    <span class="leg" style="--c:${C.crit}">预算上限</span>
  </div>`;
}

// ---- add model (single) ----
function openModelModal() {
  $id("mf-fields").classList.add("hidden");
  $id("mf-save").disabled = true;
  $id("mf-error").textContent = "";
  $id("mf-repo").value = "";
  $id("modal-model").classList.remove("hidden");
}
$id("mf-cancel").onclick = () => $id("modal-model").classList.add("hidden");
$id("mf-fetch").onclick = async () => {
  $id("mf-error").textContent = "拉取中…";
  const repo = $id("mf-repo").value.trim();
  if (!repo) return;
  const cat = $id("mf-cat").value;
  const m = await fetch(`/api/models/preview?repo_id=${encodeURIComponent(repo)}&category=${cat}&source=${$id("mf-source").value}`).then((r) => r.json());
  if (m.error) { $id("mf-error").textContent = m.error; return; }
  $id("mf-name").value = m.name; $id("mf-params").value = m.params_b;
  $id("mf-layers").value = m.layers; $id("mf-hidden").value = m.hidden_dim;
  $id("mf-attn").value = m.attn_heads; $id("mf-kv").value = m.kv_heads;
  $id("mf-headdim").value = m.head_dim; $id("mf-experts").value = m.num_experts;
  $id("mf-kvlayers").value = m.kv_layers || 0;
  $id("mf-expertparams").value = m.expert_params_b;
  $id("mf-fields").classList.remove("hidden");
  $id("mf-save").disabled = false;
  $id("mf-error").textContent = "已解析，请核对 KV 头数等关键字段后入库。";
};
$id("mf-save").onclick = async () => {
  const spec = {
    id: $id("mf-repo").value.trim(), name: $id("mf-name").value, category: $id("mf-cat").value,
    params_b: +$id("mf-params").value, layers: +$id("mf-layers").value, hidden_dim: +$id("mf-hidden").value,
    attn_heads: +$id("mf-attn").value, kv_heads: +$id("mf-kv").value, head_dim: +$id("mf-headdim").value,
    kv_layers: +$id("mf-kvlayers").value || 0,
    num_experts: +$id("mf-experts").value, expert_params_b: +$id("mf-expertparams").value,
  };
  const r = await fetch("/api/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }).then((r) => r.json());
  if (r.error) { $id("mf-error").textContent = r.error; return; }
  $id("modal-model").classList.add("hidden");
  await loadDropdowns(); $id("model").value = spec.id; onModelChange(); recalc();
};

// ---- bulk import ----
function openBulkModal() {
  $id("bf-list").value = ""; $id("bf-result").textContent = "";
  $id("modal-bulk").classList.remove("hidden");
}
$id("bf-cancel").onclick = () => $id("modal-bulk").classList.add("hidden");
$id("bf-go").onclick = async () => {
  const ids = $id("bf-list").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!ids.length) { $id("bf-result").textContent = "请输入至少一个 Repo ID"; return; }
  $id("bf-result").className = "hint"; $id("bf-result").textContent = `正在拉取 ${ids.length} 个模型…`;
  const r = await fetch("/api/models/bulk", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_ids: ids, category: $id("bf-cat").value, source: $id("bf-source").value }),
  }).then((r) => r.json());
  const fail = r.failed || [];
  $id("bf-result").className = fail.length ? "error" : "success";
  $id("bf-result").innerHTML = `✅ 成功 ${r.saved.length} 个${fail.length ? ` · ❌ 失败 ${fail.length}：` + fail.map((f) => f.id).join("、") : ""}`;
  await loadDropdowns();
};

// ---- add gpu ----
function openGpuModal() {
  $id("gf-error").textContent = "";
  $id("gf-id").value = $id("gf-name").value = $id("gf-vram").value = "";
  $id("modal-gpu").classList.remove("hidden");
}
$id("gf-cancel").onclick = () => $id("modal-gpu").classList.add("hidden");
$id("gf-save").onclick = async () => {
  const spec = { id: $id("gf-id").value.trim(), name: $id("gf-name").value, vram_gb: +$id("gf-vram").value };
  const r = await fetch("/api/gpus", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) }).then((r) => r.json());
  if (r.error) { $id("gf-error").textContent = r.error; return; }
  $id("modal-gpu").classList.add("hidden");
  await loadDropdowns(); $id("gpu").value = spec.id; recalc();
};

// ---- calibration (real-log overhead override) ----
$id("btn-calib").onclick = () => {
  $id("cb-gpu").innerHTML = [...$id("gpu").options]
    .map(o => `<option value="${o.value}" ${o.value === $id("gpu").value ? "selected" : ""}>${o.textContent}</option>`).join("");
  $id("cb-util").value = $id("gpu_util").value;
  $id("cb-res").textContent = "";
  $id("modal-calib").classList.remove("hidden");
};
$id("cb-cancel").onclick = () => $id("modal-calib").classList.add("hidden");
$id("cb-go").onclick = async () => {
  $id("cb-res").className = "hint"; $id("cb-res").textContent = "解析中…";
  const r = await fetch("/api/calibrate", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gpu_id: $id("cb-gpu").value, util: +$id("cb-util").value,
                           log_text: $id("cb-log").value }) }).then((r) => r.json());
  if (r.error) { $id("cb-res").className = "error"; $id("cb-res").textContent = r.error; return; }
  $id("cb-res").className = "success"; $id("cb-res").textContent = r.note;
  setTimeout(() => { $id("modal-calib").classList.add("hidden"); recalc(); }, 1200);
};

init();
window.__assistant_ctx = () => ({
  kind: "calc", input: currentInput(), last_result: window.__lastCalc || null,
});
