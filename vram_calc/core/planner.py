"""Multi-node planner: enumerate legal splits, score with estimate(), top plans.

Category order encodes vLLM reality (network penalty ascending):
single-machine > DP replicas > cross-node PP > cross-node TP.
"""
from __future__ import annotations
from dataclasses import dataclass

from .estimator import ModelSpec, GpuSpec

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
