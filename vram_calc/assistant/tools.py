"""模型可调的工具：core.estimate / planner 的薄包装（不走 HTTP）。"""
from __future__ import annotations
import json

from ..repos import get_model, get_gpu, get_server
from ..core.estimator import EstimateInput, estimate
from ..core.cluster import server_is_mixed
from ..core.planner import Machine, PlanInput, plan_deployment

CALC_VRAM = {
    "name": "calc_vram",
    "description": ("估算一套 vLLM 部署配置的显存。回答任何涉及显存/并发/上下文可行性的问题"
                    "（包括假设性变更，如'换 3 张卡'）都必须先调它拿真实数字，禁止心算。"),
    "parameters": {"type": "object", "properties": {
        "model_id": {"type": "string"}, "gpu_id": {"type": "string"},
        "gpu_count": {"type": "integer"}, "tp": {"type": "integer"},
        "pp": {"type": "integer"}, "ep": {"type": "integer"},
        "context_len": {"type": "integer"}, "concurrency": {"type": "integer"},
        "quant": {"type": "string"}, "kv_quant": {"type": "string"},
        "gpu_util": {"type": "number"},
        "max_num_batched_tokens": {"type": "integer"}},
        "required": ["model_id", "gpu_id", "gpu_count", "tp", "context_len"]},
}

PLAN_MULTI_NODE = {
    "name": "plan_multi_node",
    "description": "多机部署规划：给定已入库服务器 id 列表与目标，返回 top 方案（含切分与每机账本摘要）。",
    "parameters": {"type": "object", "properties": {
        "model_id": {"type": "string"},
        "server_ids": {"type": "array", "items": {"type": "string"}},
        "context_len": {"type": "integer"}, "concurrency": {"type": "integer"},
        "quant": {"type": "string"}, "kv_quant": {"type": "string"},
        "gpu_util": {"type": "number"}},
        "required": ["model_id", "server_ids", "context_len"]},
}


def tools_for(page_ctx: dict | None) -> list[dict]:
    ts = [CALC_VRAM]
    if page_ctx and page_ctx.get("kind") == "plan":
        ts.append(PLAN_MULTI_NODE)
    return ts


def execute_tool(name: str, args: dict) -> str:
    if name == "calc_vram":
        return _calc_vram(args)
    if name == "plan_multi_node":
        return _plan_multi_node(args)
    return f"错误：未知工具 {name}"


def _calc_vram(a: dict) -> str:
    m = get_model(a.get("model_id", ""))
    if m is None:
        return f"错误：模型 {a.get('model_id')} 不在库中，请让用户先在页面添加。"
    g = get_gpu(a.get("gpu_id", ""))
    if g is None:
        return f"错误：显卡 {a.get('gpu_id')} 不在显卡库中。"
    tp, pp, ep = int(a.get("tp", 1) or 1), int(a.get("pp", 1) or 1), int(a.get("ep", 1) or 1)
    gc = int(a.get("gpu_count", tp * pp) or tp * pp)
    if gc != tp * pp:
        return f"错误：gpu_count={gc} 与 tp*pp={tp * pp} 不一致，请修正后重调。"
    ctx = max(1, int(a.get("context_len", 4096) or 4096))
    conc = max(1, int(a.get("concurrency", 1) or 1))
    r = estimate(EstimateInput(
        model=m, gpu=g, quant=a.get("quant", "fp16"), context_len=ctx,
        concurrency=conc, engine="vllm", tp=tp, pp=pp, ep=ep,
        kv_quant=a.get("kv_quant", "fp16"),
        safety_factor=float(a.get("gpu_util", 0.9) or 0.9),
        max_num_batched_tokens=int(a.get("max_num_batched_tokens", 8192) or 8192)))
    return json.dumps({
        "verdict": r.verdict,                       # ok=放得下 tight=能跑会限流 over=OOM
        "per_gpu_total_gb": round(r.breakdown.total, 2),
        "usable_gb": r.usable_gb,
        "kv_pool_tokens": r.max_kv_tokens,
        "kv_budget_gb": r.kv_budget_gb,
        "requested_tokens": ctx * conc,
        "max_concurrency_at_ctx": int(r.max_kv_tokens / ctx),
    }, ensure_ascii=False)


def _plan_multi_node(a: dict) -> str:
    m = get_model(a.get("model_id", ""))
    if m is None:
        return f"错误：模型 {a.get('model_id')} 不在库中。"
    machines, notes = [], []
    for sid in a.get("server_ids", []):
        s = get_server(sid)
        if s is None:
            return f"错误：服务器 {sid} 不在库中。"
        if server_is_mixed(s) or not s.gpus:
            notes.append(f"{s.name} 混插/无GPU 跳过")
            continue
        g = get_gpu(s.gpus[0].gpu_id)
        if g is None:
            notes.append(f"{s.name} 的 {s.gpus[0].gpu_id} 不在显卡库")
            continue
        machines.append(Machine(s.id, s.name, s.host, g, s.gpus[0].count))
    if not machines:
        return "错误：没有可参与规划的服务器（混插/无GPU/显卡缺失）。" + "；".join(notes)
    plans = plan_deployment(PlanInput(
        model=m, machines=tuple(machines),
        context_len=max(1, int(a.get("context_len", 4096) or 4096)),
        concurrency=max(1, int(a.get("concurrency", 1) or 1)),
        quant=a.get("quant", "fp16"), kv_quant=a.get("kv_quant", "fp16"),
        gpu_util=float(a.get("gpu_util", 0.9) or 0.9)))
    return json.dumps({
        "plans": [{"name": p.name, "tp": p.tp, "pp": p.pp, "ep": p.ep, "dp": p.dp,
                   "verdict": p.verdict, "why": p.why,
                   "max_kv_tokens": p.max_kv_tokens} for p in plans],
        "notes": notes,
    }, ensure_ascii=False)
