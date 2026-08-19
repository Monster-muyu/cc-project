// 两页共用工具
function parseContext(s) {           // "200k"/"1m"/"4096" → tokens
  s = String(s || "").trim().toLowerCase();
  let mul = 1;
  if (s.endsWith("k")) { mul = 1e3; s = s.slice(0, -1); }
  else if (s.endsWith("m")) { mul = 1e6; s = s.slice(0, -1); }
  const n = parseFloat(s);
  return isNaN(n) ? 0 : Math.round(n * mul);
}
async function jget(u) { return (await fetch(u)).json(); }
async function jpost(u, body) {
  const r = await fetch(u, {method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify(body)});
  return r.json();
}
const $ = sel => document.querySelector(sel);
