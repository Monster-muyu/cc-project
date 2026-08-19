// 多机规划页：服务器清单 + 规划目标 → /api/plan → 方案卡
let servers = [], models = [], gpus = [];
const selected = new Set();
let deb = null;

async function init() {
  [servers, models, gpus] = await Promise.all([
    jget("/api/servers"), jget("/api/models"), jget("/api/gpus")]);
  $("#p-model").innerHTML = models.filter(m => m.category === "llm")
    .map(m => `<option value="${m.id}">${m.name}</option>`).join("");
  fillGpuSelects();
  renderServers();
  bind();
}

const gpuName = id => (gpus.find(g => g.id === id) || {}).name || id;

function renderServers() {
  $("#srv-list").innerHTML = servers.map(s => `
    <div class="srv">
      <input type="checkbox" data-sel="${s.id}" ${selected.has(s.id) ? "checked" : ""}
             ${s.mixed ? "disabled title=\"混插不参与规划\"" : ""}/>
      <div><div class="nm">${s.name}</div><div class="host">${s.host || ""}</div></div>
      ${s.gpus.map(g => `<span class="gpuchip ${s.mixed ? "mixwarn" : ""}">${gpuName(g.gpu_id)} <b>× ${g.count}</b></span>`).join("")}
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
    schedulePlan();
  });
  $("#srv-list").addEventListener("click", async e => {
    const id = e.target.dataset.del;
    if (!id) return;
    await fetch(`/api/servers/${id}`, {method: "DELETE"});
    servers = servers.filter(s => s.id !== id);
    selected.delete(id);
    renderServers();
    schedulePlan();
  });
  $("#btn-add-srv").onclick = openSrvModal;
  $("#sf-add-row").onclick = () => addGpuRow();
  $("#sf-save").onclick = saveSrv;
  $("#btn-plan").onclick = runPlan;
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
        <th class="r">开销/卡</th><th class="r">全实例KV池</th><th class="r">占用/可用</th><th>占用</th></tr></thead>
        <tbody>${p.rows.map(r => `
          <tr><td>${r.server_name}</td><td>${r.gpus_used}× ${r.gpu_name}</td>
          <td class="r">${r.weights_gb.toFixed(1)}</td><td class="r">${r.overhead_gb.toFixed(1)}</td>
          <td class="r">${r.kv_budget_gb.toFixed(1)}</td><td class="r">${r.total_gb.toFixed(1)} / ${r.usable_gb.toFixed(1)}</td>
          <td><div class="usebar"><i style="width:${Math.min(100, r.total_gb / r.usable_gb * 100).toFixed(0)}%;background:${barClr}"></i></div></td></tr>`).join("")}
        </tbody></table>
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

// ---- 添加服务器弹窗 ----
function fillGpuSelects() {
  document.querySelectorAll("#sf-rows select").forEach(sel => {
    sel.innerHTML = gpus.map(g => `<option value="${g.id}">${g.name} (${g.vram_gb}G)</option>`).join("");
  });
}
function addGpuRow(gpuId, count) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<select data-g></select><input type="number" data-c min="1" max="128" value="${count || 8}" style="flex:0.5"/>`;
  $("#sf-rows").appendChild(row);
  fillGpuSelects();
  if (gpuId) row.querySelector("[data-g]").value = gpuId;
}
function openSrvModal() {
  $("#sf-rows").innerHTML = "";
  $("#sf-name").value = ""; $("#sf-host").value = ""; $("#sf-error").textContent = "";
  addGpuRow();
  $("#modal-srv").classList.remove("hidden");
}
$("#sf-cancel").onclick = () => $("#modal-srv").classList.add("hidden");
async function saveSrv() {
  const gpusIn = [...document.querySelectorAll("#sf-rows .row")].map(r => ({
    gpu_id: r.querySelector("[data-g]").value, count: +r.querySelector("[data-c]").value || 1}));
  const r = await jpost("/api/servers", {
    id: $("#sf-name").value.trim(), name: $("#sf-name").value.trim(),
    host: $("#sf-host").value.trim(), gpus: gpusIn});
  if (r.ok) {
    $("#modal-srv").classList.add("hidden");
    servers = await jget("/api/servers");
    renderServers();
  } else {
    $("#sf-error").textContent = r.error || "保存失败";
  }
}

init();
