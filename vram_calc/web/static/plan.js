// 多机规划页：服务器清单 + 规划目标 → /api/plan → 方案卡
let servers = [], models = [], gpus = [];
const selected = new Set();
let deb = null;
let lastPlan = null;
const LS_SEL = "vramcalc.planSel";           // 勾选的服务器跨页面/刷新保留

function saveSel() { localStorage.setItem(LS_SEL, JSON.stringify([...selected])); }
function restoreSel() {
  try {
    const ids = JSON.parse(localStorage.getItem(LS_SEL) || "[]");
    ids.forEach(id => { if (servers.some(s => s.id === id)) selected.add(id); });
  } catch {}
}

async function init() {
  [servers, models, gpus] = await Promise.all([
    jget("/api/servers"), jget("/api/models"), jget("/api/gpus")]);
  fillModelSel();
  fillGpuSelects();
  restoreSel();
  renderServers();
  bind();
  if (selected.size) runPlan();               // 恢复勾选后直接出方案
}

function fillModelSel(keepId) {
  const cur = keepId || $("#p-model").value;
  $("#p-model").innerHTML = models.filter(m => m.category === "llm")
    .map(m => `<option value="${m.id}">${m.name}</option>`).join("");
  if (cur && models.some(m => m.id === cur)) $("#p-model").value = cur;
}

const gpuName = id => (gpus.find(g => g.id === id) || {}).name || id;

function renderServers() {
  $("#srv-list").innerHTML = servers.map(s => `
    <div class="srv">
      <input type="checkbox" data-sel="${s.id}" ${selected.has(s.id) ? "checked" : ""}
             ${s.mixed ? "disabled title=\"混插不参与规划\"" : ""}/>
      <div><div class="nm">${s.name}</div><div class="host">${s.host || ""}</div></div>
      ${s.gpus.map(g => `<span class="gpuchip ${s.mixed ? "mixwarn" : ""}">${gpuName(g.gpu_id)} <b>× ${g.count}</b></span>`).join("")}
      <button class="del" data-edit="${s.id}" title="编辑">✏</button>
      <button class="del" data-del="${s.id}" title="删除">✕</button>
    </div>`).join("") || '<p class="hint">还没有服务器，点下方添加</p>';
  const mixed = servers.filter(s => s.mixed);
  const w = $("#srv-warn");
  w.hidden = !mixed.length;
  w.textContent = mixed.length ? `${mixed.map(s => s.name).join("、")} 机内混插 GPU，vLLM 不支持混型号 TP，不参与规划` : "";
}

function bind() {
  $("#srv-list").addEventListener("change", e => {
    const id = e.target.dataset.sel;
    if (!id) return;
    e.target.checked ? selected.add(id) : selected.delete(id);
    saveSel();
    schedulePlan();
  });
  $("#srv-list").addEventListener("click", async e => {
    if (e.target.dataset.edit) {
      openSrvModal(servers.find(s => s.id === e.target.dataset.edit));
      return;
    }
    const id = e.target.dataset.del;
    if (!id) return;
    await fetch(`/api/servers/${id}`, {method: "DELETE"});
    servers = servers.filter(s => s.id !== id);
    selected.delete(id);
    saveSel();
    renderServers();
    schedulePlan();
  });
  $("#btn-add-srv").onclick = () => openSrvModal();
  $("#sf-add-row").onclick = () => addGpuRow();
  $("#sf-rows").addEventListener("click", e => {
    if (!("rmrow" in e.target.dataset)) return;
    // 至少保留一行 GPU 配置
    if (document.querySelectorAll("#sf-rows .row").length <= 1) return;
    e.target.closest(".row").remove();
  });
  $("#sf-save").onclick = saveSrv;
  $("#btn-fetch-model").onclick = openModelFetch;
  $("#mf2-go").onclick = fetchAndSaveModel;
  ["#p-ctx", "#p-conc"].forEach(sel => $(sel).addEventListener("input", schedulePlan));
  ["#p-model", "#p-quant", "#p-kvquant"].forEach(sel => $(sel).addEventListener("change", schedulePlan));
  $("#p-util").addEventListener("input", e => {
    $("#p-util-val").textContent = Math.round(e.target.value * 100) + "%";
    schedulePlan();
  });
}

function schedulePlan() {                     // debounce 800ms 自动重算（有勾选才跑）
  clearTimeout(deb);
  if (!selected.size) { $("#plans").innerHTML = ""; $("#plan-empty").hidden = false; return; }
  deb = setTimeout(runPlan, 800);
}

async function runPlan() {
  const r = await jpost("/api/plan", {
    model_id: $("#p-model").value, server_ids: [...selected],
    context_len: parseContext($("#p-ctx").value) || 4096,
    concurrency: Math.max(1, +$("#p-conc").value || 1),
    quant: $("#p-quant").value, kv_quant: $("#p-kvquant").value,
    gpu_util: +$("#p-util").value});
  lastPlan = r;
  renderPlans(r);
}

const V_TXT = {ok: "✔ 放得下", tight: "⚠ 能跑 · 有限制", over: "✕ 放不下"};
const V_CLR = {ok: "ok", tight: "tight", over: "over"};

function renderPlans({plans, warnings}) {
  $("#plan-warnings").innerHTML = (warnings || []).map(w => `<div class="warnline">${w}</div>`).join("");
  $("#plan-empty").hidden = !!(plans && plans.length);
  $("#plans").innerHTML = (plans || []).map((p, i) => {
    const barClr = p.verdict === "ok" ? "var(--brand)" : p.verdict === "tight" ? "var(--warn)" : "var(--crit)";
    return `
    <div class="card" style="margin-bottom:18px">
      <div class="plan-head">
        <h3>方案 ${i + 1} · ${p.name}</h3>
        ${i === 0 && p.verdict === "ok" ? '<span class="plan-badge rec">推荐</span>' : ""}
        ${p.badges.map(b => `<span class="plan-badge">${b}</span>`).join("")}
        <span class="plan-verdict ${V_CLR[p.verdict]}">${V_TXT[p.verdict]}</span>
      </div>
      <div class="topo">TP=${p.tp} · PP=${p.pp} · EP=${p.ep} · DP=${p.dp}　｜　${p.hosts.map(h => h[0]).join(" + ")}</div>
      <p class="plan-why">${p.why}</p>
      ${(p.warnings || []).map(w => `<div class="warnline">${w}</div>`).join("")}
      <table class="ledger"><thead><tr><th>机器</th><th>卡</th><th class="r">权重/卡</th>
        <th class="r">开销/卡</th><th class="r">全实例KV池</th><th class="r">每卡占用/可用</th>
        <th class="r">机器总占用/总可用</th><th>占用</th></tr></thead>
        <tbody>${p.rows.map(r => `
          <tr><td>${r.server_name}</td><td>${r.gpus_used}× ${r.gpu_name}</td>
          <td class="r">${r.weights_gb.toFixed(1)}</td><td class="r">${r.overhead_gb.toFixed(1)}</td>
          <td class="r">${r.kv_budget_gb.toFixed(1)}</td><td class="r">${r.total_gb.toFixed(1)} / ${r.usable_gb.toFixed(1)}</td>
          <td class="r"><b>${(r.total_gb * r.gpus_used).toFixed(1)}</b> / ${(r.usable_gb * r.gpus_used).toFixed(1)}</td>
          <td><div class="usebar"><i style="width:${Math.min(100, r.total_gb / r.usable_gb * 100).toFixed(0)}%;background:${barClr}"></i></div></td></tr>`).join("")}
        </tbody></table>
      <p class="hint">占用/可用均为<b>每卡</b>数值，机器总可用 = 每卡 × 卡数（如 8×RTX 3090 = 8 × 21.6 = 172.8 GB）</p>
      <details class="conc"><summary>📈 并发敏感度（KV 需求 = 上下文×并发 vs 本方案池容量）</summary>
        ${p.max_kv_tokens ? concChart(p.max_kv_tokens, parseContext($("#p-ctx").value) || 4096)
                          : '<p class="hint">该方案无 KV 池数据</p>'}
      </details>
      <details class="cmd"><summary>生成启动命令（${p.commands.length} 段）</summary>
        ${p.commands.map(c => `<pre><button class="copybtn" onclick="copyCmd(this)">复制</button>${c.title}\n${c.code}</pre>`).join("")}
      </details>
    </div>`;
  }).join("");
}

function copyCmd(btn) {
  navigator.clipboard.writeText(btn.parentElement.textContent.replace(/^复制/, "").trim());
  btn.textContent = "已复制";
  setTimeout(() => btn.textContent = "复制", 1200);
}

// ---- 并发敏感度小图：KV 需求(上下文×并发) vs 本方案池容量 ----
function concChart(maxTokens, ctx) {
  const ceil = Math.floor(maxTokens / ctx);            // 当前上下文最大并发
  const x1 = Math.max(ceil * 2, Math.max(8, Math.ceil(ceil * 1.3) + 1));
  const W = 540, H = 130, padL = 46, padR = 12, padT = 10, padB = 22;
  const X = x => padL + (x - 1) / (x1 - 1) * (W - padL - padR);
  const Y = t => H - padB - Math.min(t / (ctx * x1), 1.05) * (H - padT - padB);
  let pts = "";
  for (let x = 1; x <= x1; x++) pts += `${X(x).toFixed(1)},${Y(ctx * x).toFixed(1)} `;
  const by = Math.max(Y(maxTokens), padT);
  return `
  <p class="hint" style="margin:2px 0 6px">并发承载：本方案 KV 池 <b>${maxTokens.toLocaleString()}</b> tokens
    · 当前上下文 ${ctx.toLocaleString()} → <b>最多 ${ceil} 路并发</b></p>
  <svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px">
    <line x1="${padL}" y1="${by}" x2="${W - padR}" y2="${by}" stroke="#2a78d6" stroke-width="2" stroke-dasharray="6 4"/>
    <text x="${W - padR}" y="${by - 4}" text-anchor="end" font-size="11" fill="#2a78d6">池容量 ${(maxTokens / 1000).toFixed(0)}k</text>
    <polyline points="${pts}" fill="none" stroke="#eb6834" stroke-width="2"/>
    ${ceil >= 1 && ceil <= x1 ? `<circle cx="${X(ceil)}" cy="${Y(ctx * ceil)}" r="4" fill="#eb6834"/>
      <text x="${X(ceil) - 6}" y="${Y(ctx * ceil) - 7}" text-anchor="end" font-size="11" fill="#b34a1f">≈${ceil} 路</text>` : ""}
    <text x="${padL - 6}" y="${H - padB + 4}" text-anchor="end" font-size="11" fill="#898781">1</text>
    <text x="${X(x1)}" y="${H - padB + 16}" text-anchor="middle" font-size="11" fill="#898781">${x1}</text>
    <text x="${(padL + W - padR) / 2}" y="${H - 2}" text-anchor="middle" font-size="11" fill="#898781">并发数 →</text>
  </svg>`;
}

// ---- 添加/编辑服务器弹窗 ----
function fillGpuSelects() {
  document.querySelectorAll("#sf-rows select").forEach(sel => {
    sel.innerHTML = gpus.map(g => `<option value="${g.id}">${g.name} (${g.vram_gb}G)</option>`).join("");
  });
}
function addGpuRow(gpuId, count) {
  const row = document.createElement("div");
  row.className = "row sf-row";
  row.innerHTML = `<select data-g style="flex:1"></select>
    <span class="gpucnt">卡数</span>
    <input type="number" data-c min="1" max="128" value="${count || 8}" style="flex:0 0 64px"/>
    <button type="button" class="del" data-rmrow title="删除此行">✕</button>`;
  $("#sf-rows").appendChild(row);
  fillGpuSelects();
  if (gpuId) row.querySelector("[data-g]").value = gpuId;
}
function openSrvModal(srv) {                    // srv 传入 = 编辑模式（同 id 保存即覆盖）
  $("#sf-title").textContent = srv ? "编辑服务器" : "添加服务器";
  $("#sf-rows").innerHTML = "";
  $("#sf-name").value = srv ? srv.id : "";
  $("#sf-host").value = srv ? srv.host : "";
  $("#sf-error").textContent = "";
  (srv && srv.gpus.length ? srv.gpus : [null]).forEach(g => addGpuRow(g && g.gpu_id, g && g.count));
  $("#modal-srv").classList.remove("hidden");
}
$("#sf-cancel").onclick = () => $("#modal-srv").classList.add("hidden");
async function saveSrv() {
  const name = $("#sf-name").value.trim();
  if (!name) { $("#sf-error").textContent = "名称必填"; return; }
  const gpusIn = [...document.querySelectorAll("#sf-rows .row")].map(r => ({
    gpu_id: r.querySelector("[data-g]").value, count: +r.querySelector("[data-c]").value || 1}));
  const r = await jpost("/api/servers", {
    id: name, name, host: $("#sf-host").value.trim(), gpus: gpusIn});
  if (r.ok) {
    $("#modal-srv").classList.add("hidden");
    servers = await jget("/api/servers");
    renderServers();
    schedulePlan();
  } else {
    $("#sf-error").textContent = r.error || "保存失败";
  }
}

// ---- 拉取模型入库（HF/ModelScope） ----
function openModelFetch() {
  $("#mf2-res").textContent = "";
  $("#modal-mf2").classList.remove("hidden");
}
$("#mf2-cancel").onclick = () => $("#modal-mf2").classList.add("hidden");
async function fetchAndSaveModel() {
  const repo = $("#mf2-repo").value.trim();
  const src = $("#mf2-source").value;
  if (!repo) { $("#mf2-res").textContent = "请填 Repo ID"; return; }
  $("#mf2-res").textContent = "拉取中…";
  try {
    const p = await jget(`/api/models/preview?repo_id=${encodeURIComponent(repo)}&source=${src}`);
    if (p.error) { $("#mf2-res").textContent = p.error; return; }
    await jpost("/api/models", p);
    models = await jget("/api/models");
    fillModelSel(p.id);
    $("#modal-mf2").classList.add("hidden");
  } catch (e) {
    $("#mf2-res").textContent = `拉取失败: ${e.message}`;
  }
}

init();
window.__assistant_ctx = () => ({
  kind: "plan",
  servers: servers.filter(s => selected.has(s.id)).map(s => ({id: s.id, name: s.name, gpus: s.gpus})),
  goal: {model_id: $("#p-model").value, context_len: parseContext($("#p-ctx").value) || 4096,
         concurrency: Math.max(1, +$("#p-conc").value || 1), quant: $("#p-quant").value,
         kv_quant: $("#p-kvquant").value, gpu_util: +$("#p-util").value},
  last_plans: lastPlan,
});
