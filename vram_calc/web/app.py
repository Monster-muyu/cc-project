"""FastAPI web app: 4 routes + add-model/add-gpu, serves the single-page UI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..core.estimator import ModelSpec, GpuSpec, EstimateInput, estimate
from ..core.engines import ENGINES
from ..core.quant import QUANT_BYTES
from ..repos import (list_models, list_gpus, get_model, get_gpu,
                     save_model, save_gpu, fetch_model_preview, fetch_and_save_many,
                     infer_quant_from_id)

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates"))

app = FastAPI(title="VRAM 显存计算工具")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

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
        safety_factor=req.safety_factor, exl2_bpw=req.exl2_bpw))


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
             "params_b": m.params_b, "category": m.category,
             "quant": m.quant or infer_quant_from_id(m.id)} for m in list_models()]


@app.get("/api/gpus")
def api_gpus():
    return [{"id": g.id, "name": g.name, "vram_gb": g.vram_gb} for g in list_gpus()]


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
    points = []
    for x in range(max(x0, 1), x1 + 1):
        try:
            r = _estimate(req.model_copy(update={sweep_var: x}))
            points.append({"x": x, "total_gb": round(r.breakdown.total * n, 2)})
        except Exception:
            break
    usable = round(base.usable_gb * n, 2)
    max_x = max((p["x"] for p in points if p["total_gb"] <= usable), default=None)
    return {"points": points, "capacity_gb": round(base.capacity_gb * n, 2),
            "usable_gb": usable, "max_x": max_x}


@app.get("/api/models/preview")
def api_model_preview(repo_id: str, category: str = "llm"):
    try:
        m = fetch_model_preview(repo_id, category=category)
    except Exception as e:
        return JSONResponse({"error": f"拉取失败: {e}"}, status_code=400)
    return {"id": m.id, "name": m.name, "params_b": m.params_b, "layers": m.layers,
            "hidden_dim": m.hidden_dim, "attn_heads": m.attn_heads, "kv_heads": m.kv_heads,
            "head_dim": m.head_dim, "vocab_size": m.vocab_size,
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
    """Bulk add: {"repo_ids": [..] | "id1\nid2", "category": "llm"}."""
    repo_ids = payload.get("repo_ids", [])
    if isinstance(repo_ids, str):
        repo_ids = repo_ids.splitlines()
    return fetch_and_save_many(repo_ids, category=payload.get("category", "llm"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
