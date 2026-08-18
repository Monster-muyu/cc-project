# 多机部署规划 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 服务器清单 + 模型/目标 → 自动规划跨机并行切分，输出 top 3 方案（每机账本 + 推理理由 + 可执行 vLLM 启动命令）。

**Architecture:** 后端规划器 `core/planner.py` 枚举合法切分（单机 > DP > 跨机 PP > 跨机 TP 四类，按类别取最优，cap 3），每候选举调现有 `estimate()`；`core/commands.py` 由 Plan 生成命令块；Web 层加服务器 CRUD + `/api/plan` + 新页签 `/plan`。前端按已确认样式稿渲染。

**Tech Stack:** FastAPI + Jinja2 + 原生 JS（无新依赖）。

**Spec:** `docs/superpowers/specs/2026-08-18-multi-node-planning-design.md`

## Global Constraints

- Python 解释器：`E:\ANACONDA\envs\vram-calc\python.exe`（下文 `$PY`）；测试 `$PY -m pytest tests -q`
- 不新增任何 pip 依赖
- GB = 1e9（estimator.py 现有约定）
- 引擎固定 vllm（多机 ray 路径），不做引擎选项
- commit message 用中文、格式与现有历史一致（`feat:`/`fix:`/`ui:`/`docs:` 前缀）
- MoE 的 EP：vLLM 的 `--enable-expert-parallel` 使专家分布在 TP 组内，规划器对 MoE 取 `ep = tp`（与 estimator 的 `ep_eff` 语义一致）
- 混插服务器（单机多种 GPU 型号）：可入库可显示，规划时跳过并在响应中给出警告文本

---

### Task 1: ServerSpec 数据模型 + 服务器存储

**Files:**
- Create: `vram_calc/core/cluster.py`
- Modify: `vram_calc/repos/store.py`（追加 servers EntityStore 与 CRUD）
- Test: `tests/test_cluster_store.py`

**Interfaces:**
- Produces: `GpuCount(gpu_id: str, count: int)`；`ServerSpec(id, name, host, gpus: tuple[GpuCount,...])`；store 层 `list_servers() -> list[ServerSpec]`、`get_server(sid) -> ServerSpec | None`、`save_server(s) -> Path`；`server_is_mixed(s) -> bool`（机内 >1 种型号）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cluster_store.py
import pytest
from vram_calc.core.cluster import ServerSpec, GpuCount, server_is_mixed
from vram_calc.repos import store


def test_server_is_mixed():
    s = ServerSpec(id="a", name="A", host="", gpus=((GpuCount("rtx-3090", 2), GpuCount("rtx-4090", 2))))
    assert server_is_mixed(s) is True
    t = ServerSpec(id="b", name="B", host="", gpus=((GpuCount("rtx-3090", 8),)))
    assert server_is_mixed(t) is False


def test_server_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "USER_DIR", tmp_path)
    s = ServerSpec(id="srv-a", name="server-A", host="192.168.1.11",
                   gpus=((GpuCount("rtx-3090", 8),)))
    store.save_server(s)
    got = store.get_server("srv-a")
    assert got == s
    assert store.list_servers() == [s]
```

- [ ] **Step 2: 跑测试确认失败**（ModuleNotFoundError: cluster）
- [ ] **Step 3: 实现**

```python
# vram_calc/core/cluster.py
"""Server entity: one physical machine and its GPU slots.

gpus is a LIST of (gpu_id, count) so mixed-GPU machines are representable
from day one; the v1 planner skips mixed machines (vLLM can't mix types in
a TP group) but the data model won't need migration later.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuCount:
    gpu_id: str
    count: int


@dataclass(frozen=True)
class ServerSpec:
    id: str
    name: str
    host: str = ""
    gpus: tuple[GpuCount, ...] = ()


def server_is_mixed(s: ServerSpec) -> bool:
    return len({g.gpu_id for g in s.gpus}) > 1
```

store.py 追加（仿照现有 models/gpus 三段）：

```python
from ..core.cluster import ServerSpec, GpuCount   # 顶部 import 区

servers = EntityStore("servers.json", "servers")

def list_servers() -> list[ServerSpec]:
    return [_dict_to_server(e) for e in servers.list()]

def get_server(sid: str) -> ServerSpec | None:
    e = servers.get(sid)
    return _dict_to_server(e) if e else None

def save_server(s: ServerSpec) -> Path:
    return servers.save({"id": s.id, "name": s.name, "host": s.host,
                         "gpus": [{"gpu_id": g.gpu_id, "count": g.count} for g in s.gpus]})

def _dict_to_server(e: dict) -> ServerSpec:
    return ServerSpec(id=e["id"], name=e.get("name", e["id"]),
                      host=e.get("host", ""),
                      gpus=tuple(GpuCount(**g) for g in e.get("gpus", [])))
```

并建空种子 `vram_calc/data/servers.json`，内容 `[]`。

- [ ] **Step 4: `$PY -m pytest tests/test_cluster_store.py -q` → PASS**
- [ ] **Step 5: Commit** `feat:服务器实体+存储(混插可表达,规划期跳过)`

---

### Task 2: 规划器 — 候选枚举与硬约束

**Files:**
- Create: `vram_calc/core/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `ModelSpec/GpuSpec/EstimateInput/estimate`（estimator.py），Task 1 的 ServerSpec
- Produces: `Machine(server_id, name, host, gpu: GpuSpec, count: int)`；`Candidate(category, tp, pp, machines)`；`enumerate_candidates(machines: tuple[Machine,...], model: ModelSpec) -> list[Candidate]`；category ∈ `"single" | "dp" | "pp" | "tp_cross"`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_planner.py 顶部 fixture，Task 3/4 复用
import pytest
from vram_calc.core.estimator import ModelSpec, GpuSpec
from vram_calc.core.planner import Machine, enumerate_candidates

M8B = ModelSpec(id="t/8b", name="8B", params_b=8.0, layers=32, hidden_dim=4096,
                attn_heads=32, kv_heads=8, head_dim=128)
G24 = GpuSpec(id="g24", name="24G", vram_gb=24)
G12 = GpuSpec(id="g12", name="12G", vram_gb=12)

def mach(sid, gpu, n): return Machine(sid, sid, "", gpu, n)


def test_single_machine_candidates():
    cands = enumerate_candidates((mach("a", G24, 8),), M8B)
    singles = [c for c in cands if c.category == "single"]
    assert [(c.tp, c.pp) for c in singles] == [(8, 1)]


def test_count_not_head_divisor_falls_back():
    # 12 卡、32 头：TP 必须整除头数 → 回退到 8（弃 4 卡）
    cands = enumerate_candidates((mach("a", G24, 12),), M8B)
    assert any(c.category == "single" and c.tp == 8 for c in cands)


def test_dp_groups_same_type_only():
    cands = enumerate_candidates((mach("a", G24, 8), mach("b", G24, 8), mach("c", G12, 4)), M8B)
    dps = [c for c in cands if c.category == "dp"]
    assert len(dps) == 1 and set(m.server_id for m in dps[0].machines) == {"a", "b"}


def test_pp_cross_machine():
    cands = enumerate_candidates((mach("a", G12, 4), mach("b", G12, 4)), M8B)
    pps = [c for c in cands if c.category == "pp"]
    assert any(c.tp == 4 and c.pp == 2 for c in pps)


def test_tp_cross_only_same_type():
    cands = enumerate_candidates((mach("a", G12, 2), mach("b", G12, 2)), M8B)
    tcs = [c for c in cands if c.category == "tp_cross"]
    assert any(c.tp == 4 for c in tcs)          # 2+2 组成 TP4
    cands2 = enumerate_candidates((mach("a", G12, 2), mach("b", G24, 2)), M8B)
    assert not [c for c in cands2 if c.category == "tp_cross"]
```

- [ ] **Step 2: 确认失败**（ModuleNotFoundError: planner）
- [ ] **Step 3: 实现**

```python
# vram_calc/core/planner.py — Task 2 部分
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
    # 2) DP：同型号且各自能独立成实例的机器 ≥2 台
    by_type: dict[str, list[Machine]] = {}
    for m in machines:
        if _tp_choices(model, m.count):
            by_type.setdefault(m.gpu.id, []).append(m)
    for group in by_type.values():
        if len(group) >= 2:
            tp = max(_tp_choices(model, min(m.count for m in group)))
            out.append(Candidate("dp", tp, 1, tuple(group)))
    # 3) 跨机 PP：同型号、每台出 tp 张卡拼一个实例
    for group in by_type.values():
        for tp in _tp_choices(model, min(m.count for m in group)):
            stage_ms = tuple(group)
            if len(stage_ms) >= 2 and tp * len(stage_ms) and len(stage_ms) <= model.layers:
                out.append(Candidate("pp", tp, len(stage_ms), stage_ms))
    # 4) 跨机 TP：同型号全量拼 TP（兜底，评分时标慢）
    for group in by_type.values():
        total = sum(m.count for m in group)
        if len(group) >= 2 and model.attn_heads % total == 0 and total in _TP_STEPS:
            out.append(Candidate("tp_cross", total, 1, tuple(group)))
    return out
```

注意 `by_type` 只收"能独立成实例"的机器——PP/TP 跨机同样要求 tp ≤ count，用同一过滤合理。

- [ ] **Step 4: `$PY -m pytest tests/test_planner.py -q` → PASS（此时只有枚举测试）**
- [ ] **Step 5: Commit** `feat:规划器候选枚举(单机/DP/跨机PP/跨机TP+头数整除约束)`

---

### Task 3: 规划器 — 评分、账本与 top 方案组装

**Files:**
- Modify: `vram_calc/core/planner.py`（追加）
- Test: `tests/test_planner.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `Machine/Candidate/enumerate_candidates`；`estimate()`
- Produces: `PlanInput(model, machines, context_len, concurrency, quant, kv_quant, gpu_util, max_num_batched_tokens)`；`LedgerRow(server_id, server_name, gpu_name, gpus_used, weights_gb, overhead_gb, activation_gb, kv_budget_gb, total_gb, usable_gb, verdict)`；`Plan(key, name, badges, tp, pp, ep, dp, verdict, why, cross_node, rows, max_kv_tokens, req_tokens, concurrency_per_replica, warnings)`；`plan_deployment(inp: PlanInput) -> list[Plan]`（≤3 个，按 单机→DP→跨机PP→跨机TP 类别序；全部放不下时返回放不下的最佳候选）

- [ ] **Step 1: 追加失败测试**

```python
# tests/test_planner.py 追加
from vram_calc.core.planner import PlanInput, plan_deployment

def pin(**kw):  # 8B fp16=16GB 权重；默认目标 4k×1
    base = dict(model=M8B, machines=(mach("a", G24, 8), mach("b", G24, 8)),
                context_len=4096, concurrency=1)
    base.update(kw)
    return PlanInput(**base)


def test_top3_categories_and_order():
    plans = plan_deployment(pin())
    assert [p.key for p in plans][:2] == ["single", "dp"]     # 单机第一，DP 第二
    assert plans[0].verdict == "ok"
    assert plans[0].rows[0].gpus_used == 8
    assert plans[1].dp == 2 and plans[1].concurrency_per_replica == 1
    assert plans[0].why                            # why 非空
    assert not plans[0].cross_node and plans[2].cross_node is True   # 第三是跨机 PP


def test_dp_splits_concurrency():
    plans = plan_deployment(pin(concurrency=3))
    dp = [p for p in plans if p.key == "dp"][0]
    assert dp.concurrency_per_replica == 2          # ceil(3/2)


def test_small_cards_push_to_pp():
    plans = plan_deployment(pin(machines=(mach("a", G12, 4), mach("b", G12, 4))))
    # 8B fp16 单卡 12G 放不下 → 单机类被淘汰，首个方案是跨机 PP
    assert plans[0].key == "pp" and plans[0].tp == 4 and plans[0].pp == 2


def test_nothing_fits_returns_over_plan():
    plans = plan_deployment(pin(machines=(mach("a", G12, 1),)))
    assert plans and plans[0].verdict == "over"     # 全放不下也返回（带 over）


def test_fp8_kv_on_unsupported_gpu_warns():
    plans = plan_deployment(pin(kv_quant="fp8"))
    assert plans and plans[0].warnings              # G24 未声明 supports_fp8 → 警告
```

- [ ] **Step 2: 确认新测试失败**
- [ ] **Step 3: 实现（planner.py 追加）**

```python
import math
from .estimator import EstimateInput, estimate

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


def _eval_candidate(c: Candidate, inp: PlanInput) -> Plan:
    m = inp.model
    dp = len(c.machines) if c.category == "dp" else 1
    conc_rp = max(1, math.ceil(inp.concurrency / dp))
    ep = c.tp if m.num_experts else 1
    rows, worst, min_kv, warns = [], "ok", None, []
    for mach_ in c.machines:
        est = estimate(EstimateInput(
            model=m, gpu=mach_.gpu, quant=inp.quant, context_len=inp.context_len,
            concurrency=conc_rp, engine="vllm", tp=c.tp, pp=c.pp, ep=ep,
            kv_quant=inp.kv_quant, safety_factor=inp.gpu_util,
            max_num_batched_tokens=inp.max_num_batched_tokens))
        b = est.breakdown
        rows.append(LedgerRow(mach_.server_id, mach_.name, mach_.gpu.name, c.tp,
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
    why = (f"权重 TP={c.tp} 后每卡 {head.weights_gb:.1f}GB，加开销后每卡剩 "
           f"{head.kv_budget_gb:.1f}GB KV 池；本次需求 {req_tokens} tokens"
           f"（池容量 {min_kv}）")
    if c.category == "single":
        why = "单机放得下就不跨机——跨机 TP/PP 都有网络惩罚。" + why
    elif c.category == "dp":
        why = f"{dp} 个独立副本各承担一半流量，总吞吐 ≈ {dp}×，副本间无需互联。" + why
    elif c.category == "pp":
        why = "多机拼一个大实例：KV 池总量更大（单序列可更长），代价是 PP 跨机 bubble、吞吐低于 DP。" + why
    else:
        why = "兜底方案：跨机 TP 需要高带宽互联（万兆/RDMA），普通以太网会很慢。" + why
    badges = {"single": ("单机即可",), "dp": (f"吞吐 ×{dp}",),
              "pp": (f"单实例 {c.tp * c.pp} 卡",), "tp_cross": ("跨机TP·慢",)}[c.category]
    return Plan(c.category, CATEGORY_NAME[c.category], badges, c.tp, c.pp, ep, dp,
                worst, why, c.category in ("pp", "tp_cross"), tuple(rows),
                min_kv or 0, req_tokens, conc_rp, tuple(warns))


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
```

- [ ] **Step 4: 全部测试 PASS**（`$PY -m pytest tests -q`，26+旧 全绿）
- [ ] **Step 5: Commit** `feat:规划器评分与top3方案组装(每机账本/why推理/FP8警告)`

---

### Task 4: 启动命令生成

**Files:**
- Create: `vram_calc/core/commands.py`
- Test: `tests/test_commands.py`

**Interfaces:**
- Consumes: Task 3 的 `Plan`（含 machines 信息：命令里要 host/server 名——Plan 需带 machines：在 `Plan` 追加字段 `hosts: tuple[tuple[str, str], ...]`（server_name, host），`_eval_candidate` 里从 cand.machines 填充）
- Produces: `CommandBlock(title: str, code: str)`；`render_commands(plan: Plan, model_id: str, context_len: int, concurrency: int, gpu_util: float, kv_quant: str) -> list[CommandBlock]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_commands.py
from vram_calc.core.planner import Plan, LedgerRow
from vram_calc.core.commands import render_commands

def mkplan(**kw):
    d = dict(key="single", name="最少机器", badges=("单机即可",), tp=8, pp=1, ep=1, dp=1,
             verdict="ok", why="...", cross_node=False,
             rows=(LedgerRow("a", "server-A", "3090", 8, 6.95, 1.9, 0.5, 11.2, 9.35, 20.4, "ok"),),
             max_kv_tokens=900000, req_tokens=252000, concurrency_per_replica=3,
             warnings=(), hosts=(("server-A", "192.168.1.11"),))
    d.update(kw)
    return Plan(**d)


def test_single_machine_command():
    blocks = render_commands(mkplan(), "Qwen/Qwen3.8-27B", 252000, 3, 0.85, "fp16")
    assert len(blocks) == 1
    code = blocks[0].code
    assert "--tensor-parallel-size 8" in code
    assert "--max-model-len 252000" in code
    assert "--max-num-seqs 3" in code
    assert "--gpu-memory-utilization 0.85" in code
    assert "ray" not in code


def test_dp_merges_same_command():
    p = mkplan(key="dp", dp=2, hosts=(("server-A", "10.0.0.1"), ("server-B", "10.0.0.2")))
    blocks = render_commands(p, "m", 252000, 3, 0.85, "fp16")
    assert len(blocks) == 1 and "server-A" in blocks[0].code and "server-B" in blocks[0].code


def test_cross_node_needs_ray():
    p = mkplan(key="pp", tp=4, pp=2, cross_node=True,
               hosts=(("server-A", "10.0.0.1"), ("server-B", "")))
    blocks = render_commands(p, "m", 252000, 3, 0.85, "fp8")
    texts = [b.code for b in blocks]
    assert any("ray start --head" in t for t in texts)
    assert any("ray start --address=10.0.0.1:6379" in t for t in texts)
    serve = [t for t in texts if "vllm serve" in t][0]
    assert "--pipeline-parallel-size 2" in serve
    assert "--distributed-executor-backend ray" in serve
    assert "--kv-cache-dtype fp8" in serve          # 非 fp16 才出现
    assert "server-B-IP" in "".join(texts)          # host 空用占位符


def test_moe_appends_expert_parallel():
    p = mkplan(ep=8, tp=8)
    code = render_commands(p, "moe/m", 252000, 3, 0.85, "fp16")[0].code
    assert "--enable-expert-parallel" in code
```

- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**

```python
# vram_calc/core/commands.py
"""Render launch commands for a plan. Pure string assembly, no state."""
from __future__ import annotations
from dataclasses import dataclass

from .planner import Plan


@dataclass(frozen=True)
class CommandBlock:
    title: str
    code: str


def _serve_args(p: Plan, model_id, ctx, conc, util, kv_quant, ray: bool) -> str:
    a = (f"vllm serve {model_id} \\\n  --tensor-parallel-size {p.tp}")
    if p.pp > 1:
        a += f" \\\n  --pipeline-parallel-size {p.pp}"
    if ray:
        a += " \\\n  --distributed-executor-backend ray"
    if p.ep > 1:
        a += " \\\n  --enable-expert-parallel"
    if kv_quant != "fp16":
        a += f" \\\n  --kv-cache-dtype {kv_quant}"
    a += (f" \\\n  --max-model-len {ctx} --max-num-seqs {p.concurrency_per_replica}"
          f" \\\n  --gpu-memory-utilization {util}"
          " \\\n  --host 0.0.0.0 --port 8000")
    return a


def render_commands(p: Plan, model_id: str, context_len: int, concurrency: int,
                    gpu_util: float, kv_quant: str) -> list[CommandBlock]:
    hosts = p.hosts or (("server-1", ""),)
    if not p.cross_node:
        if p.dp > 1:
            who = "、".join(h[0] for h in hosts)
            code = (f"# {who} 各起一个实例（命令相同）；负载均衡由前置网关做\n"
                    + _serve_args(p, model_id, context_len, concurrency, gpu_util, kv_quant, False))
            return [CommandBlock(f"DP ×{p.dp}：每台机器", code)]
        return [CommandBlock(hosts[0][0] + "（单机）",
                _serve_args(p, model_id, context_len, concurrency, gpu_util, kv_quant, False))]
    head, *workers = hosts
    head_addr = head[1] or f"{head[0]}-IP"
    blocks = [CommandBlock(f"{head[0]}：Ray head", f"ray start --head --port=6379")]
    if workers:
        w = "\n".join(f"# {n}\nray start --address={head_addr}:6379"
                      for n, h in workers)
        blocks.append(CommandBlock("其余机器：加入集群", w))
    serve = _serve_args(p, model_id, context_len, concurrency, gpu_util, kv_quant, True)
    serve = ("# 前提：所有机器相同 vLLM/NCCL 版本、驱动一致，网络最好万兆+\n" + serve)
    blocks.append(CommandBlock(f"{head[0]}：启动（Ray 自动分配各 stage）", serve))
    return blocks
```

（Plan 追加 `hosts` 字段时同步改 Task 3 的测试构造无影响——默认值 `()`。）

- [ ] **Step 4: `$PY -m pytest tests/test_commands.py tests/test_planner.py -q` → PASS**
- [ ] **Step 5: Commit** `feat:启动命令生成(单机/DP合并/跨机ray三段/MoE expert-parallel)`

---

### Task 5: API — 服务器 CRUD + /api/plan

**Files:**
- Modify: `vram_calc/web/app.py`（追加路由；顶部 import）
- Test: `tests/test_web_plan.py`

**Interfaces:**
- Consumes: Task 1-4 全部
- Produces: `GET/POST /api/servers`、`DELETE /api/servers/{sid}`、`POST /api/plan`；`/api/plan` 请求 `{model_id, server_ids:[...], context_len, concurrency, quant, kv_quant, gpu_util, max_num_batched_tokens?}`，响应 `{plans:[Plan-asdict+commands], warnings:[混插跳过...]}`（每个 plan 附 `commands: [{title, code}]`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_web_plan.py
from fastapi.testclient import TestClient
from vram_calc.web.app import app
from vram_calc.repos import store

cli = TestClient(app)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "USER_DIR", tmp_path)
    store.save_gpu_store = None  # noop guard: gpus come from bundled json
    store.save_server(store.ServerSpec(id="srv-a", name="A", host="10.0.0.1",
                     gpus=(store.GpuCount("rtx-3090", 8),)))


def test_server_crud(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert any(s["id"] == "srv-a" for s in cli.get("/api/servers").json())
    r = cli.post("/api/servers", json={"id": "srv-b", "name": "B", "host": "",
                                       "gpus": [{"gpu_id": "rtx-4090", "count": 4},
                                                {"gpu_id": "rtx-3090", "count": 2}]})
    assert r.status_code == 200
    assert cli.delete("/api/servers/srv-b").status_code == 200


def test_plan_endpoint(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    r = cli.post("/api/plan", json={
        "model_id": "meta-llama/Meta-Llama-3-8B", "server_ids": ["srv-a"],
        "context_len": 4096, "concurrency": 1, "quant": "fp16",
        "kv_quant": "fp16", "gpu_util": 0.9})
    assert r.status_code == 200
    body = r.json()
    assert body["plans"] and body["plans"][0]["commands"]
    assert body["plans"][0]["rows"][0]["server_id"] == "srv-a"
    assert any("vllm serve" in c["code"] for c in body["plans"][0]["commands"])


def test_plan_mixed_server_warns(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    cli.post("/api/servers", json={"id": "mix", "name": "M", "host": "",
                                   "gpus": [{"gpu_id": "rtx-3090", "count": 2},
                                            {"gpu_id": "rtx-4090", "count": 2}]})
    r = cli.post("/api/plan", json={
        "model_id": "meta-llama/Meta-Llama-3-8B", "server_ids": ["mix"],
        "context_len": 4096, "concurrency": 1})
    assert r.status_code == 200
    assert r.json()["warnings"]          # 混插被跳过 → 警告 + 无方案或空 plans
```

- [ ] **Step 2: 确认失败（404）**
- [ ] **Step 3: 实现（app.py 追加）**

```python
from dataclasses import asdict
from ..core.cluster import ServerSpec, GpuCount, server_is_mixed
from ..core.planner import Machine, PlanInput, plan_deployment
from ..core.commands import render_commands
from ..repos import (list_servers, get_server, save_server, get_model, get_gpu)


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
    p = store_servers_path(sid)   # 或 EntityStore 增加 delete；见下
    ...
```

EntityStore 无 delete——给它加一个（复用面广，一处改）：

```python
# store.py EntityStore 追加
def delete(self, entity_id: str) -> bool:
    p = self.user_dir / f"{_safe(entity_id)}.json"
    if p.exists():
        p.unlink()
        return True
    return False
```

`/api/plan` 实现：

```python
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
```

（Plan dataclass 需要 `hosts` 字段供命令块用——Task 4 已加；`asdict` 会一并序列化。）

- [ ] **Step 4: `$PY -m pytest tests/test_web_plan.py -q` → PASS；全量 tests 也跑一遍**
- [ ] **Step 5: Commit** `feat:服务器CRUD+/api/plan(混插警告,方案含命令块)`

---

### Task 6: 前端 — /plan 页面 + common.js 提取 + 页签激活

**Files:**
- Create: `vram_calc/web/templates/plan.html`、`vram_calc/web/static/plan.js`、`vram_calc/web/static/common.js`
- Modify: `vram_calc/web/app.py`（`GET /plan` 路由）、`vram_calc/web/static/style.css`（追加 .plan 系列）、`vram_calc/web/static/app.js`（parseContext 移出）、`vram_calc/web/templates/index.html`（引 common.js、页签 span→a）、`vram_calc/web/templates/vllm_params.html`（页签 span→a）

**Interfaces:**
- Consumes: Task 5 的三个 API；样式稿（`E:\Claude_Code_Download\tmp\plan_ui_temp.html` 的类名与结构）
- Produces: 页面 `/plan`

- [ ] **Step 1: common.js + app.js 瘦身**

```js
// vram_calc/web/static/common.js — 两页共用工具
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
const fmtGB = n => (n >= 100 ? n.toFixed(0) : n.toFixed(1)) + " GB";
```

app.js 删除 `function parseContext(s){...}` 原定义；index.html 在 app.js 之前加 `<script src="/static/common.js?v=1"></script>`。

- [ ] **Step 2: style.css 追加（从样式稿迁移，替换硬编码色为 CSS 变量）**

```css
/* ---- 多机规划页 ---- */
.plan-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.plan-head h3 { margin:0; font-size:16px; }
.badge { font-size:11px; font-weight:700; border-radius:8px; padding:2px 9px;
  background:var(--page); color:var(--ink-2); }
.badge.rec { background:var(--brand); color:#fff; }
.plan-verdict { margin-left:auto; font-weight:700; font-size:13px; }
.plan-verdict.ok{color:var(--good)} .plan-verdict.tight{color:var(--warn)} .plan-verdict.over{color:var(--crit)}
.topo { font-family:Consolas,monospace; font-size:13px; background:#f5f8fd;
  border:1px solid #dfe9f8; border-radius:8px; padding:6px 12px; margin:10px 0;
  color:#1d4e8f; display:inline-block; }
.plan-why { font-size:12.5px; color:var(--ink-2); line-height:1.7; margin:8px 0 12px; }
.srv { border:1px solid var(--border); border-radius:10px; padding:10px 12px;
  margin:8px 0; display:flex; align-items:center; gap:10px; }
.srv .nm { font-weight:700; }
.srv .host { font-size:11.5px; color:var(--muted); font-family:Consolas,monospace; }
.gpuchip { display:inline-flex; align-items:baseline; gap:5px; background:#eef3fd;
  color:#1d5fae; border-radius:7px; padding:3px 9px; font-size:12px; font-weight:600; }
.gpuchip b { font-size:13px; }
.gpuchip.mixwarn { background:#fdf1ef; color:#b3423a; }
.ledger td, .ledger th { padding:7px 8px; border-bottom:1px solid var(--grid); font-size:12.5px; }
.ledger .r { text-align:right; font-variant-numeric:tabular-nums; }
.cmd pre { background:#0f172a; color:#d7e4f5; font-size:12px; line-height:1.7;
  border-radius:10px; padding:12px 14px; overflow-x:auto; font-family:Consolas,monospace; }
.warnline { font-size:12px; color:#b3423a; background:#fdf1ef; border-radius:8px;
  padding:7px 10px; margin-top:8px; }
```

- [ ] **Step 3: plan.html（结构照样式稿，数据全由 plan.js 填）**——左侧两卡（服务器清单 + 规划目标）、右侧 `#plans`；页签导航「多机规划」高亮。路由：

```python
@app.get("/plan", response_class=HTMLResponse)
async def plan_page(request: Request):
    return templates.TemplateResponse(request, "plan.html", {})
```

- [ ] **Step 4: plan.js**——init：`jget('/api/servers')`+`jget('/api/models')`+`jget('/api/gpus')` 填下拉；服务器渲染（勾选框、gpuchip、混插 mixwarn+禁勾、删除按钮调 DELETE）；「＋添加服务器」弹窗（名称/host/多行 GPU 型号+卡数 → POST）；「开始规划」→ `jpost('/api/plan', {...})` → 渲染方案卡：`.plan-head`（name/badges/verdict）→ `.topo`（`TP=x · PP=x · EP=x · DP=x`）→ `.plan-why` → `.ledger` 表（每机一行：机器/卡/权重/开销/KV池/占用条）→ `details.cmd`（命令块 pre + 复制按钮 `navigator.clipboard.writeText`）；参数变化 debounce 800ms 重算（复用单机页模式）。
- [ ] **Step 5: 页签激活**——index.html 与 vllm_params.html 的 `<span class="soon">多机规划</span>` → `<a href="/plan">多机规划</a>`；plan.html 头部三页签，多机规划 `.on`。
- [ ] **Step 6: Playwright 手动走查**——起服务（`$PY -m uvicorn vram_calc.web.app:app --port 8000`）：添加两台服务器（8×3090、4×3090）→ 规划 Llama-3-8B 4k → 断言出现 ≥2 方案卡、命令块可展开复制；单机页回归（模型下拉、计算、sweep 正常，common.js 未破坏）。
- [ ] **Step 7: Commit** `feat:多机规划页(服务器清单/方案卡/命令块)+common.js提取+页签激活`

---

### Task 7: 收尾 — 文档同步 + 全量验证

**Files:**
- Modify: `README.md`（功能清单加「多机规划」一小节）、`docs/design.md`（v2.1：多机规划架构段）

- [ ] **Step 1: README/设计文档补多机规划说明（各 ≤15 行：用法 + 规则速览 + API）**
- [ ] **Step 2: 全量 `$PY -m pytest tests -q` → 全绿；`$PY scripts/calibrate.py` 仍 4/4**（确认 estimator 未被波及）
- [ ] **Step 3: Playwright 走查两页一遍（计算器回归 + 多机规划）**
- [ ] **Step 4: Commit** `docs:多机规划文档同步(README+design v2.1)` 并 `git push`

---

## Self-Review 记录

- 规格覆盖：spec §3 数据模型→Task1；§4 枚举/约束/评分→Task2/3；§5 API→Task5；§6 页面→Task6；§7 命令→Task4；§8 测试→各任务+Task7。无缺口。
- 占位符扫描：Task 6 Step 4 是叙述性步骤但列明了全部数据流与函数职责（前端无单测，验收靠 Step 6 走查）——符合项目现状（单机页同样策略）。
- 类型一致性：`Machine/Candidate/PlanInput/Plan/LedgerRow/CommandBlock` 各任务签名已互相对齐；Task 4 依赖的 `Plan.hosts` 已在 Task 4 的 Consumes 里显式声明为对 Task 3 的追加改动。
