"""Multi-node planner: enumerate legal splits, score with estimate(), top plans.

Category order encodes vLLM reality (network penalty ascending):
single-machine > DP replicas > cross-node PP > cross-node TP.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from .estimator import ModelSpec, GpuSpec, EstimateInput, estimate

_TP_STEPS = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class Machine:
    server_id: str
    name: str
    host: str
    gpu: GpuSpec
    count: int


@dataclass(frozen=True)
class Candidate:
    category: str                     # single | dp | pp | tp_cross
    tp: int
    pp: int
    machines: tuple[Machine, ...]     # 参与机器（single/dp/pp 每台用 tp 张卡）


def _tp_choices(model: ModelSpec, hi: int) -> list[int]:
    """合法 TP：2 的幂、整除注意力头数、≤ hi，大到小。"""
    return [t for t in reversed(_TP_STEPS) if t <= hi and model.attn_heads % t == 0]


def enumerate_candidates(machines: tuple[Machine, ...], model: ModelSpec) -> list[Candidate]:
    out: list[Candidate] = []
    # 1) 单机：每台机器独立成实例（TP 取 ≤卡数的最大合法值）
    for m in machines:
        tps = _tp_choices(model, m.count)
        if tps:
            out.append(Candidate("single", tps[0], 1, (m,)))
    # 同型号且能独立成实例的机器分组（DP/PP/跨机TP 都从这里取）
    by_type: dict[str, list[Machine]] = {}
    for m in machines:
        if _tp_choices(model, m.count):
            by_type.setdefault(m.gpu.id, []).append(m)
    # 2) DP：同型号组内 ≥2 台，各跑一个完整副本
    for group in by_type.values():
        if len(group) >= 2:
            tp = max(_tp_choices(model, min(m.count for m in group)))
            out.append(Candidate("dp", tp, 1, tuple(group)))
    # 3) 跨机 PP：同型号、每台出 tp 张卡拼一个实例（多 tp 变体都保留，评分时挑）
    for group in by_type.values():
        for tp in _tp_choices(model, min(m.count for m in group)):
            if len(group) >= 2 and len(group) <= model.layers:
                out.append(Candidate("pp", tp, len(group), tuple(group)))
    # 4) 跨机 TP：同型号全量拼 TP（兜底，评分时标慢）
    for group in by_type.values():
        total = sum(m.count for m in group)
        if len(group) >= 2 and total in _TP_STEPS and model.attn_heads % total == 0:
            out.append(Candidate("tp_cross", total, 1, tuple(group)))
    return out


VERDICT_RANK = {"ok": 0, "tight": 1, "over": 2}
CATEGORY_NAME = {"single": "最少机器", "dp": "吞吐优先", "pp": "跨机流水线", "tp_cross": "跨机张量并行"}


@dataclass(frozen=True)
class PlanInput:
    model: ModelSpec
    machines: tuple[Machine, ...]
    context_len: int
    concurrency: int = 1
    quant: str = "fp16"
    kv_quant: str = "fp16"
    gpu_util: float = 0.9
    max_num_batched_tokens: int = 8192
    engine: str = "vllm"            # vllm | sglang（多机支持的引擎）


@dataclass(frozen=True)
class LedgerRow:
    server_id: str; server_name: str; gpu_name: str; gpus_used: int
    weights_gb: float; overhead_gb: float; activation_gb: float
    kv_budget_gb: float; total_gb: float; usable_gb: float; verdict: str


@dataclass(frozen=True)
class Plan:
    key: str; name: str; badges: tuple[str, ...]
    tp: int; pp: int; ep: int; dp: int
    verdict: str; why: str; cross_node: bool
    rows: tuple[LedgerRow, ...]
    max_kv_tokens: int; req_tokens: int; concurrency_per_replica: int
    warnings: tuple[str, ...] = ()
    hosts: tuple[tuple[str, str], ...] = ()      # (server_name, host)，Task 4 命令生成用
    net: str = ""                                # 跨机网络要求提示（单机/DP 为空）
    decode_tps: float = 0.0                      # roofline 吞吐（含 DP 副本叠加）


def _eval_candidate(c: Candidate, inp: PlanInput) -> Plan:
    m = inp.model
    dp = len(c.machines) if c.category == "dp" else 1
    conc_rp = max(1, math.ceil(inp.concurrency / dp))
    ep = c.tp if m.num_experts else 1
    rows, worst, min_kv, warns = [], "ok", None, []
    for mach_ in c.machines:
        est = estimate(EstimateInput(
            model=m, gpu=mach_.gpu, quant=inp.quant, context_len=inp.context_len,
            concurrency=conc_rp, engine=inp.engine, tp=c.tp, pp=c.pp, ep=ep,
            kv_quant=inp.kv_quant, safety_factor=inp.gpu_util,
            max_num_batched_tokens=inp.max_num_batched_tokens))
        b = est.breakdown
        rows.append(LedgerRow(mach_.server_id, mach_.name, mach_.gpu.name,
                              c.tp if c.category != "tp_cross" else mach_.count,
                              round(b.weights, 2), round(b.overhead, 2),
                              round(b.activation, 2), round(est.kv_budget_gb, 2),
                              round(b.total, 2), round(est.usable_gb, 2), est.verdict))
        if VERDICT_RANK[est.verdict] > VERDICT_RANK[worst]:
            worst = est.verdict
        min_kv = est.max_kv_tokens if min_kv is None else min(min_kv, est.max_kv_tokens)
        if inp.kv_quant == "fp8" and not mach_.gpu.supports_fp8:
            warns.append(f"{mach_.name}（{mach_.gpu.name}）无 FP8 硬件，vLLM 将拒绝启动 FP8 KV")
    req_tokens = inp.context_len * conc_rp
    head = rows[0]
    why = (f"权重 TP={c.tp} 后每卡 {head.weights_gb:.1f}GB，加开销后全实例 KV 池共 "
           f"{head.kv_budget_gb:.1f}GB；本次需求 {req_tokens} tokens"
           f"（池容量 {min_kv}）")
    if c.category == "single":
        why = "单机放得下就不跨机——跨机 TP/PP 都有网络惩罚。" + why
    elif c.category == "dp":
        why = f"{dp} 个独立副本各承担 1/{dp} 流量，总吞吐 ≈ {dp}×，副本间无需互联。" + why
    elif c.category == "pp":
        why = "多机拼一个大实例：KV 池总量更大（单序列可更长），代价是 PP 跨机 bubble、吞吐低于 DP。" + why
    else:
        why = "兜底方案：跨机 TP 需要高带宽互联（万兆/RDMA），普通以太网会很慢。" + why
    badges = {"single": ("单机即可",), "dp": (f"吞吐 ×{dp}",),
              "pp": (f"单实例 {c.tp * c.pp} 卡",), "tp_cross": ("跨机TP·慢",)}[c.category]
    # 跨机网络要求（知识规则）：TP 每层激活都过网线，PP 每 token 过一次阶段边界
    net = {"single": "", "dp": "",
           "pp": "跨机 PP：建议 ≥25Gbps（万兆起）",
           "tp_cross": "跨机 TP：建议 ≥100Gbps（IB/RoCE）"}[c.category]
    # roofline 吞吐：单副本 bw/每卡权重 × DP 副本数
    w0, bw0 = rows[0].weights_gb or 0.0, c.machines[0].gpu.memory_bw_gbps
    tps = round(0.65 * bw0 / w0 * dp, 1) if bw0 and w0 > 0 else 0.0
    return Plan(c.category, CATEGORY_NAME[c.category], badges, c.tp, c.pp, ep, dp,
                worst, why, c.category in ("pp", "tp_cross"), tuple(rows),
                min_kv or 0, req_tokens, conc_rp, tuple(warns),
                tuple((x.name, x.host) for x in c.machines), net, tps)


def plan_deployment(inp: PlanInput) -> list[Plan]:
    cands = enumerate_candidates(inp.machines, inp.model)
    best: dict[str, Plan] = {}
    for c in cands:                       # 同类别内：verdict → KV 余量 排序取最优
        p = _eval_candidate(c, inp)
        if p.key not in best or _better(p, best[p.key]):
            best[p.key] = p
    order = [k for k in ("single", "dp", "pp", "tp_cross") if k in best]
    ranked = sorted((best[k] for k in order),
                    key=lambda p: (VERDICT_RANK[p.verdict], p.cross_node, -p.max_kv_tokens))
    feasible = [p for p in ranked if p.verdict != "over"]
    return (feasible or ranked)[:3]


def _better(a: Plan, b: Plan) -> bool:
    return (VERDICT_RANK[a.verdict], -a.max_kv_tokens) < (VERDICT_RANK[b.verdict], -b.max_kv_tokens)
