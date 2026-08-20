// AI 助手抽屉：全站自注入。设置/会话存 localStorage，Key 不进后端存储。
(() => {
const LS_CFG = "vramcalc.llm", LS_SESS = "vramcalc.sessions", LS_CUR = "vramcalc.cur";
const MAX_SESS = 20;

document.body.insertAdjacentHTML("beforeend", `
<button class="fab" id="ai-fab" title="AI 部署助手">🤖</button>
<div class="dr-dim hidden" id="ai-dim"></div>
<aside class="drawer hidden" id="ai-drawer">
  <div class="dr-head">
    <b>🤖 AI 助手</b>
    <select id="ai-sess-sel" title="切换会话"></select>
    <button id="ai-new" title="新对话">＋</button>
    <button id="ai-del" title="删除当前会话">🗑</button>
    <button id="ai-cfg" title="模型接入设置">⚙</button>
    <button id="ai-close">✕</button>
  </div>
  <div class="dr-body" id="ai-body"></div>
  <div class="dr-input">
    <input id="ai-q" type="text" placeholder="例如：换成 3 张卡能开几并发？"/>
    <button id="ai-send" class="btn primary">发送</button>
  </div>
</aside>
<div class="modal hidden" id="ai-modal">
  <div class="modal-box">
    <h3>模型接入</h3>
    <label>协议 <select id="ai-protocol">
      <option value="openai">OpenAI 兼容（DeepSeek/Qwen/GLM/vLLM/Ollama…）</option>
      <option value="anthropic">Anthropic</option>
    </select></label>
    <label>Base URL <input id="ai-baseurl" placeholder="https://api.deepseek.com/v1（本地 vLLM: http://ip:8000/v1）"/></label>
    <label>API Key <input id="ai-key" type="password" placeholder="留空则匿名（部分本地服务允许）"/></label>
    <label>模型 <input id="ai-model" placeholder="deepseek-chat / qwen2.5-72b-instruct…"/></label>
    <button id="ai-test">🔌 连接测试</button>
    <span id="ai-test-res" class="hint"></span>
    <div class="error" id="ai-cfg-err"></div>
    <div class="modal-actions">
      <button id="ai-cfg-cancel">取消</button>
      <button id="ai-cfg-save" class="primary">保存</button>
    </div>
  </div>
</div>`);

const $d = id => document.getElementById(id);
let cfg = JSON.parse(localStorage.getItem(LS_CFG) || "null") ||
          {protocol: "openai", base_url: "", api_key: "", model: ""};
let busy = false;

function esc(s) { return String(s).replace(/[&<>"']/g, c =>
  ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
function inlineMd(s) {
  return esc(s).replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
               .replace(/`([^`\n]+)`/g, "<code>$1</code>")
               .replace(/\[(计算器|官方文档|经验)\]/g, (m, k) =>
                 `<span class="src ${k === "计算器" ? "calc" : k === "官方文档" ? "doc" : "exp"}">${k}</span>`);
}
function tableHtml(rows) {
  const cells = r => r.replace(/^\||\|$/g, "").split("|").map(c => inlineMd(c.trim()));
  const isSep = i => /^[\s|:-]+$/.test(rows[i]);
  let head = "";
  if (rows.length > 1 && isSep(1)) {
    head = `<thead><tr>${cells(rows[0]).map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
    rows = rows.slice(2);
  }
  return `<table class="md-tbl">${head}<tbody>${rows
    .map(r => `<tr>${cells(r).map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function mdRender(md) {
  const out = []; let inCode = false; const code = []; let tbl = [];
  const flushT = () => { if (tbl.length) { out.push(tableHtml(tbl)); tbl = []; } };
  for (const ln of String(md).split("\n")) {
    if (ln.trim().startsWith("```")) {
      if (inCode) { out.push(`<pre>${esc(code.join("\n"))}</pre>`); code.length = 0; inCode = false; }
      else { flushT(); inCode = true; }
      continue;
    }
    if (inCode) { code.push(ln); continue; }
    if (ln.trim().startsWith("|")) { tbl.push(ln.trim()); continue; }
    flushT();
    out.push(ln.trim() ? `<p>${inlineMd(ln)}</p>` : "");
  }
  if (inCode) out.push(`<pre>${esc(code.join("\n"))}</pre>`);
  flushT();
  return out.join("\n");
}

const loadSess = () => JSON.parse(localStorage.getItem(LS_SESS) || "[]");
const saveSess = ss => localStorage.setItem(LS_SESS, JSON.stringify(ss.slice(-MAX_SESS)));
function curSess(create = true) {
  const ss = loadSess();
  let s = ss.find(x => x.id === localStorage.getItem(LS_CUR));
  if (!s && create) {
    s = {id: "s" + Date.now(), title: "新对话", createdAt: new Date().toISOString(), messages: []};
    ss.push(s); saveSess(ss); localStorage.setItem(LS_CUR, s.id);
  }
  return s;
}
function renderSessSel() {
  const cur = localStorage.getItem(LS_CUR);
  const opts = loadSess().map(s =>
    `<option value="${s.id}" ${s.id === cur ? "selected" : ""}>${esc(s.title)}</option>`).join("");
  // 无选中时加占位项：否则浏览器默认选第一项，用户再点它 onchange 不触发（假选中）
  $d("ai-sess-sel").innerHTML = loadSess().some(s => s.id === cur)
    ? opts
    : `<option value="" disabled selected>（新对话）</option>` + opts;
}
function renderAll() {
  renderSessSel();
  const s = curSess(false);
  const body = $d("ai-body");
  body.innerHTML = "";
  const ctx = window.__assistant_ctx ? window.__assistant_ctx() : null;
  if (ctx) body.insertAdjacentHTML("beforeend",
    `<div class="ctxbar">📎 已附带当前页面配置${ctx.kind === "calc" ? "（计算器）" : "（多机规划）"}，提问默认基于它</div>`);
  (s ? s.messages : []).forEach(m => {
    if (m.role === "user") body.insertAdjacentHTML("beforeend",
      `<div class="msg-user">${esc(m.content)}</div>`);
    else body.insertAdjacentHTML("beforeend", `<div class="msg-ai">${mdRender(m.content)}</div>`);
  });
  body.scrollTop = body.scrollHeight;
}

function handleEvent(d, card) {
  if (d.t === "delta") {
    card.raw += d.v;
    card.el.querySelector(".md-live").textContent = card.raw;
  } else if (d.t === "tool") {
    card.el.insertAdjacentHTML("beforeend",
      `<div class="toolbub">🔧 <span class="tname">${esc(d.name)}</span> ${esc(JSON.stringify(d.args))}</div>`);
  } else if (d.t === "tool_result") {
    card.lastResult = d.result;
    try { const r = JSON.parse(d.result);
      if (r.verdict) { const b = card.el.querySelector(".toolbub:last-of-type");
        if (b) b.insertAdjacentHTML("beforeend",
          `<div class="tres">→ ${r.verdict} · 池 ${r.kv_pool_tokens} tokens</div>`); }
    } catch {}
  } else if (d.t === "error") {
    card.raw += `\n\n**出错：** ${d.v}`;
  }
  $d("ai-body").scrollTop = $d("ai-body").scrollHeight;
}
function finalize(card, s) {
  card.el.classList.remove("streaming");
  card.el.innerHTML = mdRender(card.raw);
  if (card.lastResult) {
    try { const r = JSON.parse(card.lastResult);
      if (r.verdict) card.el.insertAdjacentHTML("afterbegin",
        `<div class="verify-strip"><span class="ok">✔ ${r.verdict === "ok" ? "放得下" : r.verdict === "tight" ? "能跑·会限流" : "OOM"}</span>` +
        ` · 占用 <b>${r.per_gpu_total_gb} GB</b>/可用 <b>${r.usable_gb} GB</b>（每卡）· KV 池 <b>${r.kv_pool_tokens}</b> tokens（估算引擎）</div>`);
    } catch {}
  }
  if (location.pathname === "/" && document.getElementById("model") && card.el.querySelector("pre"))
    card.el.insertAdjacentHTML("beforeend",
      `<div class="dr-actions"><button class="btn primary" onclick="window.__ai_apply(this)">⚡ 应用到本页</button></div>`);
}
window.__ai_apply = btn => {
  const pre = [...btn.closest(".msg-ai").querySelectorAll("pre")].pop();
  const cmd = pre ? pre.textContent : "";
  const grab = re => { const x = cmd.match(re); return x ? x[1] : null; };
  const setv = (id, v) => { const el = document.getElementById(id);
    if (el && v != null) { el.value = v; el.dispatchEvent(new Event("input", {bubbles: true})); } };
  setv("context_len", grab(/--max-model-len\s+(\d+)/));
  setv("concurrency", grab(/--max-num-seqs\s+(\d+)/));
  setv("gpu_util", grab(/--gpu-memory-utilization\s+([\d.]+)/));
  setv("max_batch", grab(/--max-num-batched-tokens\s+(\d+)/));
  const tp = grab(/--tensor-parallel-size\s+(\d+)/);
  if (tp) { setv("gpu_count", tp); setv("tp", tp); }
  const kv = grab(/--kv-cache-dtype\s+(\S+)/);
  const kvs = document.getElementById("kv_quant");
  if (kv && kvs) [...kvs.options].forEach(o => { if (o.value === kv) kvs.value = kv; });
  btn.textContent = "✔ 已应用"; setTimeout(() => btn.textContent = "⚡ 应用到本页", 1500);
};
async function send() {
  if (busy) return;
  const q = $d("ai-q").value.trim(); if (!q) return;
  if (!cfg.model) { openCfg(); return; }
  const s = curSess();
  if (!s.messages.length) { s.title = q.slice(0, 20); renderSessSel(); }
  s.messages.push({role: "user", content: q});
  $d("ai-q").value = ""; busy = true; $d("ai-send").disabled = true;
  $d("ai-body").insertAdjacentHTML("beforeend", `<div class="msg-user">${esc(q)}</div>`);
  $d("ai-body").insertAdjacentHTML("beforeend",
    `<div class="msg-ai streaming"><div class="md-live"></div></div>`);
  const card = {el: $d("ai-body").lastElementChild, raw: "", lastResult: null};
  $d("ai-body").scrollTop = $d("ai-body").scrollHeight;
  try {
    const resp = await fetch("/api/assistant/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({config: cfg, messages: s.messages,
                            page_ctx: window.__assistant_ctx ? window.__assistant_ctx() : null})});
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!line.startsWith("data:")) continue;
        const d = line.slice(5).trim();
        if (d === "[DONE]") continue;
        handleEvent(JSON.parse(d), card);
      }
    }
  } catch (e) { card.raw += `\n\n**连接失败：** ${e.message}`; }
  if (card.raw.trim()) s.messages.push({role: "assistant", content: card.raw});
  saveSess(loadSess().map(x => x.id === s.id ? s : x));
  finalize(card, s);
  busy = false; $d("ai-send").disabled = false;
}

function openCfg() {
  $d("ai-protocol").value = cfg.protocol; $d("ai-baseurl").value = cfg.base_url;
  $d("ai-key").value = cfg.api_key; $d("ai-model").value = cfg.model;
  $d("ai-modal").classList.remove("hidden");
}
async function testConn() {
  $d("ai-test-res").innerHTML = "测试中…";
  const c = {protocol: $d("ai-protocol").value, base_url: $d("ai-baseurl").value.trim(),
             api_key: $d("ai-key").value, model: $d("ai-model").value.trim()};
  const r = await jpost("/api/assistant/test", {config: c});
  $d("ai-test-res").innerHTML = r.ok
    ? `<span class="dot-ok"></span>已连接 ${esc(r.model_name)}`
    : `<span class="dot-ok dot-bad"></span>${esc(r.error)}`;
}

$d("ai-fab").onclick = () => { $d("ai-drawer").classList.remove("hidden");
  $d("ai-dim").classList.remove("hidden"); renderAll(); $d("ai-q").focus(); };
const close = () => { $d("ai-drawer").classList.add("hidden"); $d("ai-dim").classList.add("hidden"); };
$d("ai-close").onclick = close; $d("ai-dim").onclick = close;
$d("ai-send").onclick = send;
$d("ai-q").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$d("ai-cfg").onclick = openCfg;
$d("ai-test").onclick = testConn;
$d("ai-cfg-cancel").onclick = () => $d("ai-modal").classList.add("hidden");
$d("ai-cfg-save").onclick = () => {
  cfg = {protocol: $d("ai-protocol").value, base_url: $d("ai-baseurl").value.trim(),
         api_key: $d("ai-key").value, model: $d("ai-model").value.trim()};
  localStorage.setItem(LS_CFG, JSON.stringify(cfg));
  $d("ai-modal").classList.add("hidden");
};
$d("ai-new").onclick = () => { localStorage.setItem(LS_CUR, ""); renderAll(); };
$d("ai-del").onclick = () => {
  const cur = localStorage.getItem(LS_CUR);
  const rest = loadSess().filter(x => x.id !== cur);
  saveSess(rest);
  // 删除后自动切到最后一个剩余会话（而不是空选中——空选中会触发假选中问题）
  localStorage.setItem(LS_CUR, rest.length ? rest[rest.length - 1].id : "");
  renderAll();
};
$d("ai-sess-sel").onchange = e => { localStorage.setItem(LS_CUR, e.target.value); renderAll(); };
})();
