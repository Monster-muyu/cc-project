"""FastAPI web app: 4 routes + add-model/add-gpu, serves the single-page UI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from dataclasses import asdict

import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..core.estimator import ModelSpec, GpuSpec, EstimateInput, estimate
from ..core.engines import ENGINES
from ..core.quant import QUANT_BYTES, bytes_per_kv
from ..core.cluster import ServerSpec, GpuCount, server_is_mixed
from ..core.planner import Machine, PlanInput, plan_deployment
from ..core.commands import render_commands
from ..repos import (list_models, list_gpus, get_model, get_gpu,
                     save_model, save_gpu, fetch_model_preview, fetch_and_save_many,
                     fetch_modelscope, infer_quant_from_id,
                     list_servers, get_server, save_server, delete_server)

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates"))

app = FastAPI(title="VRAM 显存计算工具")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@app.middleware("http")
async def no_cache(request, call_next):
    """Dev server: never let the browser cache HTML pages or static assets.
    Recurring 'fix not visible' reports were stale caches every time."""
    resp = await call_next(request)
    if request.url.path.startswith("/static") or "text/html" in resp.headers.get("content-type", ""):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp

KV_QUANTS = ["fp16", "fp8", "int8"]


class CalcReq(BaseModel):
    model_id: str
    gpu_id: str
    quant: str = "fp16"
    context_len: int = 4096
    concurrency: int = 1
    engine: str = "vllm"
    tp: int = 1
    pp: int = 1
    ep: int = 1
    kv_quant: str = "fp16"
    cpu_offload: float = 0.0
    safety_factor: float = 0.9
    exl2_bpw: float = 4.0
    max_num_batched_tokens: int = 8192


def _estimate(req: CalcReq):
    m = get_model(req.model_id)
    g = get_gpu(req.gpu_id)
    if m is None:
        raise ValueError(f"未知模型: {req.model_id}")
    if g is None:
        raise ValueError(f"未知显卡: {req.gpu_id}")
    return estimate(EstimateInput(
        model=m, gpu=g, quant=req.quant, context_len=req.context_len,
        concurrency=req.concurrency, engine=req.engine, tp=req.tp, pp=req.pp,
        ep=req.ep, kv_quant=req.kv_quant, cpu_offload=req.cpu_offload,
        safety_factor=req.safety_factor, exl2_bpw=req.exl2_bpw,
        max_num_batched_tokens=req.max_num_batched_tokens))


@app.get("/vllm-manual", response_class=HTMLResponse)
async def vllm_params_page(request: Request):
    data = json.loads((BASE.parent / "data" / "vllm_params.json").read_text(encoding="utf-8"))
    return templates.TemplateResponse(request, "vllm_params.html", {"data": data})


# old path stuck in browsers' heuristic cache during the layout iterations --
# new URL ships clean; old one 302s over
@app.get("/vllm-params")
async def vllm_params_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/vllm-manual", status_code=302)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "quants": list(QUANT_BYTES.keys()) + ["exl2"],
        "engines": list(ENGINES.keys()),
        "kv_quants": KV_QUANTS,
    })


@app.get("/api/models")
def api_models():
    return [{"id": m.id, "name": m.name, "is_moe": bool(m.num_experts),
             "params_b": m.params_b, "category": m.category, "attn_heads": m.attn_heads,
             "quant": m.quant or infer_quant_from_id(m.id)} for m in list_models()]


@app.get("/api/gpus")
def api_gpus():
    return [{"id": g.id, "name": g.name, "vram_gb": g.vram_gb,
             "architecture": getattr(g, "architecture", ""),
             "supports_fp8": g.supports_fp8} for g in list_gpus()]


@app.post("/api/calc")
def api_calc(req: CalcReq):
    try:
        r = _estimate(req)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    b = r.breakdown
    n = r.num_gpus
    # present AGGREGATE across all n cards; per-GPU share reported separately
    return {
        "verdict": r.verdict,
        "total_gb": round(b.total * n, 2),
        "per_gpu_gb": round(b.total, 2),
        "capacity_gb": round(r.capacity_gb * n, 2),
        "usable_gb": round(r.usable_gb * n, 2),
        "headroom_gb": round(r.headroom_gb * n, 2),
        "breakdown": {k: round(v * n, 2) for k, v in b.as_dict().items()},
        "num_gpus": n,
        "max_kv_tokens": r.max_kv_tokens,
        "kv_budget_gb": round(r.kv_budget_gb, 2),
    }


@app.post("/api/sweep")
def api_sweep(req: CalcReq,
              sweep_var: Literal["concurrency", "context_len"] = "concurrency",
              x0: int = 1, x1: int = 16):
    try:
        base = _estimate(req)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    n = base.num_gpus
    m = get_model(req.model_id)
    # bytes per KV token (full model) -- sweep plots KV DEMAND vs KV BUDGET,
    # not resident VRAM (resident is context-independent in the pool model)
    kvb = bytes_per_kv(req.kv_quant)
    bpt = 2 * (m.kv_layers or m.layers) * m.kv_heads * m.head_dim * kvb if m else 0
    budget = base.kv_budget_gb
    # adaptive x-axis: solve budget/bpt for the REAL ceiling, then sweep a range
    # that comfortably brackets it (ceiling*1.3, capped at 4096) -- never a fixed 1..N
    if bpt > 0 and budget > 0:
        per_step = bpt * req.context_len / 1e9 if sweep_var == "concurrency" else bpt * req.concurrency / 1e9
        ceiling = int(budget / per_step) if per_step > 0 else 64
        x1 = max(8, min(4096, ceiling * 13 // 10 + 2))
    points = []
    # sample at most ~40 points so large ranges stay readable
    total = x1 - max(x0, 1) + 1
    step = max(1, total // 40)
    for x in range(max(x0, 1), x1 + 1, step):
        ctx = x if sweep_var == "context_len" else req.context_len
        conc = x if sweep_var == "concurrency" else req.concurrency
        kv_demand = bpt * ctx * conc / 1e9
        points.append({"x": x, "total_gb": round(kv_demand, 2)})
    max_x = max((p["x"] for p in points if p["total_gb"] <= budget), default=None)
    return {"points": points, "capacity_gb": round(budget, 2),
            "usable_gb": round(budget * 0.98, 2), "max_x": max_x, "kv_budget_gb": budget}


@app.get("/api/models/preview")
def api_model_preview(repo_id: str, category: str = "llm", source: str = "hf"):
    try:
        m = fetch_modelscope(repo_id, category=category) if source == "ms" \
            else fetch_model_preview(repo_id, category=category)
    except Exception as e:
        return JSONResponse({"error": f"拉取失败: {e}"}, status_code=400)
    return {"id": m.id, "name": m.name, "params_b": m.params_b, "layers": m.layers,
            "hidden_dim": m.hidden_dim, "attn_heads": m.attn_heads, "kv_heads": m.kv_heads,
            "head_dim": m.head_dim, "kv_layers": m.kv_layers, "vocab_size": m.vocab_size,
            "num_experts": m.num_experts, "expert_params_b": m.expert_params_b,
            "category": m.category, "quant": m.quant}


@app.post("/api/models")
def api_save_model(spec: dict):
    try:
        m = ModelSpec(**spec)
        save_model(m)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "id": m.id}


@app.post("/api/gpus")
def api_save_gpu(spec: dict):
    try:
        g = GpuSpec(**spec)
        save_gpu(g)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "id": g.id}


@app.post("/api/models/bulk")
def api_bulk_models(payload: dict):
    """Bulk add: {"repo_ids": [..] | "id1\nid2", "category": "llm", "source": "hf|ms"}."""
    repo_ids = payload.get("repo_ids", [])
    if isinstance(repo_ids, str):
        repo_ids = repo_ids.splitlines()
    return fetch_and_save_many(repo_ids, category=payload.get("category", "llm"),
                               source=payload.get("source", "hf"))


class PlanReq(BaseModel):
    model_id: str
    server_ids: list[str]
    context_len: int = 4096
    concurrency: int = 1
    quant: str = "fp16"
    kv_quant: str = "fp16"
    gpu_util: float = 0.9
    max_num_batched_tokens: int = 8192


@app.get("/api/servers")
def api_servers():
    return [{"id": s.id, "name": s.name, "host": s.host,
             "gpus": [{"gpu_id": g.gpu_id, "count": g.count} for g in s.gpus],
             "mixed": server_is_mixed(s)} for s in list_servers()]


@app.post("/api/servers")
def api_save_server(spec: dict):
    try:
        s = ServerSpec(id=spec["id"], name=spec.get("name", spec["id"]),
                       host=spec.get("host", ""),
                       gpus=tuple(GpuCount(**g) for g in spec.get("gpus", [])))
        save_server(s)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "id": s.id}


@app.delete("/api/servers/{sid}")
def api_del_server(sid: str):
    deleted = delete_server(sid)
    return {"ok": deleted}


@app.post("/api/plan")
def api_plan(req: PlanReq):
    m = get_model(req.model_id)
    if m is None:
        return JSONResponse({"error": f"未知模型: {req.model_id}"}, status_code=400)
    machines, warnings = [], []
    for sid in req.server_ids:
        s = get_server(sid)
        if s is None:
            return JSONResponse({"error": f"未知服务器: {sid}"}, status_code=400)
        if server_is_mixed(s):
            warnings.append(f"{s.name} 机内混插 GPU，v1 不参与规划（vLLM 不支持混型号 TP）")
            continue
        g = get_gpu(s.gpus[0].gpu_id)
        if g is None:
            warnings.append(f"{s.name} 的 GPU {s.gpus[0].gpu_id} 不在显卡库，已跳过")
            continue
        machines.append(Machine(s.id, s.name, s.host, g, s.gpus[0].count))
    plans = plan_deployment(PlanInput(
        model=m, machines=tuple(machines), context_len=req.context_len,
        concurrency=req.concurrency, quant=req.quant, kv_quant=req.kv_quant,
        gpu_util=req.gpu_util,
        max_num_batched_tokens=req.max_num_batched_tokens)) if machines else []
    out = []
    for p in plans:
        d = asdict(p)
        d["commands"] = [asdict(b) for b in render_commands(
            p, req.model_id, req.context_len, req.concurrency, req.gpu_util, req.kv_quant)]
        out.append(d)
    return {"plans": out, "warnings": warnings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
