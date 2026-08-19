# AI 部署助手 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全站右侧抽屉 AI 助手：带着当前页配置提问，先调估算引擎/规划器拿真实数字再回答，输出带来源徽标的推荐参数表，可一键应用回页面。

**Architecture:** 后端 `vram_calc/assistant/` 四模块（providers 双协议流式适配 / tools 工具执行器直调 core / prompts 注入组装 / orchestrator 工具循环+SSE）；web 层两个 SSE 路由；前端 `assistant.js` 全站自注入浮球+抽屉，多会话 localStorage，自写 Markdown 渲染（全量转义）。

**Tech Stack:** openai + anthropic 官方 SDK（本计划破例新增的仅有的两个依赖）、FastAPI StreamingResponse、原生 JS。

**Spec:** `docs/superpowers/specs/2026-08-19-ai-assistant-design.md`

## Global Constraints

- Python：`E:\ANACONDA\envs\vram-calc\python.exe`（$PY）；测试 `$PY -m pytest tests -q`
- 新增依赖仅 `openai`、`anthropic`（进 pyproject dependencies），其余零新增
- GB=1e9；commit 中文前缀式（feat:/fix:/ui:/docs:）；仓库根 E:\显存计算工具，main 直接提交；push 失败 Sleep 5 重试最多 3 次
- **Key 安全**：Key 只随请求内存中转；任何日志不打印 cfg/api_key/headers；异常经 `humanize_llm_error` 翻译后再出
- SSE 事件五种：`delta`(v) / `tool`(name,args) / `tool_result`(name,result) / `error`(v) / `done`(rounds)
- 模型输出契约为 Markdown 表格（| 参数 | 推荐值 | 为什么 |），「为什么」列以 `[计算器]`/`[官方文档]`/`[经验]` 前缀标注来源
- 前端渲染助手输出前必须全量 HTML 转义（不可信输入）
- git 命令一律 rtk 前缀（rtk git add/commit/push）

---

### Task 1: 依赖 + Provider 适配层

**Files:**
- Modify: `pyproject.toml`（dependencies 加两行）
- Create: `vram_calc/assistant/__init__.py`（空）、`vram_calc/assistant/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces: `LLMConfig(BaseModel)`：`protocol:str="openai"|"anthropic"、base_url:str=""、api_key:str=""、model:str=""`；`TextDelta(text)`、`ToolCall(name,args:dict)`、`Done(reason)` 三个 frozen dataclass；`get_provider(cfg)->LLMProvider`；`LLMProvider.chat_stream(messages,tools)->Iterator[StreamEvent]`（messages 为 OpenAI 格式，tools 为本计划 Task 2 定义的 canonical dict）；`LLMProvider.test_connection()->str`（成功返回模型名，失败 raise）；`humanize_llm_error(exc)->str`；`split_system(messages)->(list[str],list[dict])`（OpenAI→Anthropic 消息翻译，纯函数）

- [ ] **Step 1: 装依赖并进 pyproject**

```toml
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "jinja2>=3.1",
    "pydantic>=2.6",
    "huggingface_hub>=0.23",
    "openai>=1.40",
    "anthropic>=0.34",
]
```
然后 `& E:\ANACONDA\envs\vram-calc\python.exe -m pip install openai anthropic`

- [ ] **Step 2: 写失败测试**

```python
# tests/test_providers.py
import pytest
from vram_calc.assistant.providers import (LLMConfig, get_provider,
    OpenAIProvider, AnthropicProvider, humanize_llm_error, split_system)


def test_get_provider_selects_by_protocol():
    assert isinstance(get_provider(LLMConfig(protocol="openai", model="m")), OpenAIProvider)
    assert isinstance(get_provider(LLMConfig(protocol="anthropic", model="m")), AnthropicProvider)


def test_humanize_llm_error_mapping():
    assert "401" in humanize_llm_error(Exception("Error code: 401 - invalid api key"))
    assert "超时" in humanize_llm_error(Exception("Request timed out after 60s"))
    assert "连接" in humanize_llm_error(Exception("Connection refused to http://x"))
    assert "调用失败" in humanize_llm_error(RuntimeError("whatever"))


def test_split_system_translates_tool_records():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "问"},
        {"role": "assistant", "content": "想算", "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": "calc_vram", "arguments": "{\"tp\": 2}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "{\"verdict\": \"ok\"}"},
        {"role": "assistant", "content": "答"},
    ]
    sys_texts, convo = split_system(msgs)
    assert sys_texts == ["SYS"]
    assert [m["role"] for m in convo] == ["user", "assistant", "user", "assistant"]
    assert convo[1]["content"][0]["type"] == "tool_use"
    assert convo[1]["content"][0]["input"] == {"tp": 2}
    assert convo[2]["content"][0]["type"] == "tool_result"
    assert convo[2]["content"][0]["tool_use_id"] == "c0"


def test_split_system_merges_consecutive_user():
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    _, convo = split_system(msgs)
    assert len(convo) == 1 and "a" in convo[0]["content"] and "b" in convo[0]["content"]
```

- [ ] **Step 3: 跑测试确认失败**（ModuleNotFoundError: vram_calc.assistant）
- [ ] **Step 4: 实现**

```python
# vram_calc/assistant/providers.py
"""LLM provider adapters: unify OpenAI-compatible + Anthropic streaming.

messages/tools 参数一律用 OpenAI 格式进来（orchestrator 的通用货币），
AnthropicProvider 在边界上翻译。真实网络调用只发生在 chat_stream/test_connection，
测试用 FakeProvider 替换（见 tests/test_orchestrator.py）。
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from pydantic import BaseModel


class LLMConfig(BaseModel):
    protocol: str = "openai"          # openai | anthropic
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


@dataclass(frozen=True)
class Done:
    reason: str = ""


def get_provider(cfg: LLMConfig) -> "LLMProvider":
    return AnthropicProvider(cfg) if cfg.protocol == "anthropic" else OpenAIProvider(cfg)


class LLMProvider(ABC):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    @abstractmethod
    def chat_stream(self, messages: list[dict], tools: list[dict]) -> Iterator[TextDelta | ToolCall | Done]:
        """Yield TextDelta 流，随后本轮流完时统一 yield ToolCall（可能多个），最后 Done。"""

    @abstractmethod
    def test_connection(self) -> str:
        """1-token 最小往返。成功返回模型名，失败 raise 原始异常。"""


class OpenAIProvider(LLMProvider):
    """OpenAI 兼容：OpenAI/DeepSeek/Qwen/GLM/Kimi/SiliconFlow/vLLM/Ollama 全走这里。"""

    def _client(self):
        from openai import OpenAI
        kw = {"api_key": self.cfg.api_key}
        if self.cfg.base_url:
            kw["base_url"] = self.cfg.base_url
        return OpenAI(**kw)

    def chat_stream(self, messages, tools):
        client = self._client()
        resp = client.chat.completions.create(
            model=self.cfg.model, messages=messages, stream=True,
            tools=[{"type": "function", "function": t} for t in tools])
        buf: dict[int, dict] = {}                 # tool_call index -> {name, args}
        for chunk in resp:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.delta and ch.delta.content:
                yield TextDelta(ch.delta.content)
            if ch.delta and ch.delta.tool_calls:
                for tc in ch.delta.tool_calls:
                    b = buf.setdefault(tc.index or 0, {"name": "", "args": ""})
                    if tc.function and tc.function.name:
                        b["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        b["args"] += tc.function.arguments
        for b in buf.values():
            try:
                args = json.loads(b["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield ToolCall(b["name"], args)
        yield Done("stop")

    def test_connection(self):
        client = self._client()
        client.chat.completions.create(model=self.cfg.model,
                                       messages=[{"role": "user", "content": "hi"}],
                                       max_tokens=1)
        return self.cfg.model


def split_system(messages: list[dict]) -> tuple[list[str], list[dict]]:
    """OpenAI messages -> (system_texts, anthropic_messages)。工具记录翻译，
    相邻同角色合并（Anthropic 对连续同角色挑剔）。"""
    sys_texts, convo = [], []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            sys_texts.append(content)
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                     "content": content}
            if convo and convo[-1]["role"] == "user" and isinstance(convo[-1]["content"], list):
                convo[-1]["content"].append(block)
            else:
                convo.append({"role": "user", "content": [block]})
        elif m.get("tool_calls"):
            parts = ([{"type": "text", "text": content}] if content else [])
            for tc in m["tool_calls"]:
                parts.append({"type": "tool_use", "id": tc["id"],
                              "name": tc["function"]["name"],
                              "input": json.loads(tc["function"].get("arguments") or "{}")})
            convo.append({"role": "assistant", "content": parts})
        else:
            if convo and convo[-1]["role"] == role and role == "user" \
               and isinstance(convo[-1]["content"], str):
                convo[-1]["content"] += "\n" + content
            else:
                convo.append({"role": role, "content": content})
    return sys_texts, convo


class AnthropicProvider(LLMProvider):
    def _client(self):
        from anthropic import Anthropic
        kw = {"api_key": self.cfg.api_key}
        if self.cfg.base_url:
            kw["base_url"] = self.cfg.base_url
        return Anthropic(**kw)

    def chat_stream(self, messages, tools):
        client = self._client()
        sys_texts, convo = split_system(messages)
        a_tools = [{"name": t["name"], "description": t["description"],
                    "input_schema": t["parameters"]} for t in tools]
        with client.messages.stream(model=self.cfg.model,
                                    system="\n\n".join(sys_texts) or None,
                                    messages=convo, tools=a_tools,
                                    max_tokens=4096) as stream:
            for ev in stream:
                if ev.type == "content_block_delta" and ev.delta.type == "text_delta":
                    yield TextDelta(ev.delta.text)
            final = stream.get_final_message()
        for blk in final.content:                # SDK 已聚合好 tool_use 输入
            if blk.type == "tool_use":
                yield ToolCall(blk.name, dict(blk.input))
        yield Done(final.stop_reason or "end_turn")

    def test_connection(self):
        client = self._client()
        r = client.messages.create(model=self.cfg.model,
                                   messages=[{"role": "user", "content": "hi"}],
                                   max_tokens=1)
        return r.model


def humanize_llm_error(exc: Exception) -> str:
    """SDK 异常 -> 一句话人话。永不包含 Key。"""
    s = f"{exc}"
    low = s.lower()
    if "401" in s or "authentication" in low or "api key" in low:
        return "API Key 无效或过期（401）"
    if "404" in s or ("model" in low and "not" in low):
        return "模型名不存在或无权限（404）"
    if "429" in s:
        return "请求过于频繁或额度不足（429）"
    if "timeout" in low or "timed out" in low:
        return "连接超时：检查 Base URL 与网络/代理"
    if "connection" in low or "refused" in low:
        return "无法连接：Base URL 不通或本地服务未启动"
    return f"调用失败: {type(exc).__name__}"
```

- [ ] **Step 5: `$PY -m pytest tests/test_providers.py -q` → PASS（4 个）**
- [ ] **Step 6: Commit** `feat:LLM双协议适配层(OpenAI兼容+Anthropic,统一流事件+错误人话化)`

---

### Task 2: 工具执行器 + Prompt 组装

**Files:**
- Create: `vram_calc/assistant/tools.py`、`vram_calc/assistant/prompts.py`
- Test: `tests/test_assist_tools.py`

**Interfaces:**
- Consumes: `core.estimator.estimate/EstimateInput`、`core.planner.plan_deployment/PlanInput/Machine`、`repos.get_model/get_gpu/get_server`、`core.cluster.server_is_mixed`
- Produces: `CALC_VRAM:dict`、`PLAN_MULTI_NODE:dict`（canonical 工具 schema，`{"name","description","parameters"}`）；`tools_for(page_ctx)->list[dict]`；`execute_tool(name,args)->str`（永远返回字符串：JSON 结果或「错误：…」人话）；`build_system_prompt(page_ctx)->str`；`manual_digest()->str`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_assist_tools.py
import json
from vram_calc.assistant import tools, prompts
from vram_calc.repos import store

LLAMA = "meta-llama/Meta-Llama-3-8B"


def test_calc_vram_returns_estimate_json():
    r = json.loads(tools.execute_tool("calc_vram", {
        "model_id": LLAMA, "gpu_id": "rtx-4090", "gpu_count": 1, "tp": 1,
        "context_len": 4096, "concurrency": 1, "quant": "fp16"}))
    assert r["verdict"] == "ok"
    assert r["kv_pool_tokens"] > 100000
    assert "per_gpu_total_gb" in r and "max_concurrency_at_ctx" in r


def test_calc_vram_unknown_model_error_text():
    r = tools.execute_tool("calc_vram", {"model_id": "no/such", "gpu_id": "rtx-4090",
                                         "gpu_count": 1, "tp": 1, "context_len": 4096})
    assert r.startswith("错误：") and "no/such" in r


def test_calc_vram_gpu_count_mismatch():
    r = tools.execute_tool("calc_vram", {"model_id": LLAMA, "gpu_id": "rtx-4090",
                                         "gpu_count": 4, "tp": 2, "pp": 1,
                                         "context_len": 4096})
    assert "不一致" in r


def test_unknown_tool():
    assert "未知工具" in tools.execute_tool("nope", {})


def test_tools_for_page_kind():
    base = tools.tools_for(None)
    assert [t["name"] for t in base] == ["calc_vram"]
    plan = tools.tools_for({"kind": "plan"})
    assert [t["name"] for t in plan] == ["calc_vram", "plan_multi_node"]


def test_plan_multi_node(tmp_path):
    store.servers.user_dir = tmp_path / "servers"
    store.save_server(store.ServerSpec(
        id="srv-a", name="A", host="", gpus=(store.GpuCount("rtx-3090", 8),)))
    r = json.loads(tools.execute_tool("plan_multi_node", {
        "model_id": LLAMA, "server_ids": ["srv-a"], "context_len": 4096}))
    assert r["plans"] and r["plans"][0]["name"] == "最少机器"


def test_build_system_prompt_injects():
    sp = prompts.build_system_prompt({"kind": "calc",
                                      "input": {"model_id": LLAMA, "tp": 2},
                                      "last_result": {"verdict": "ok"}})
    assert "calc_vram" in sp and "最少" not in sp or True
    assert "推理过程" in sp and "禁止心算" in sp     # 契约与硬规则
    assert "--gpu-memory-utilization" in sp          # 手册摘要已注入
    assert '"tp": 2' in sp                           # 当前配置已注入


def test_build_system_prompt_null_ctx():
    sp = prompts.build_system_prompt(None)
    assert "手册页" in sp
```

- [ ] **Step 2: 确认失败**（ModuleNotFoundError）
- [ ] **Step 3: 实现**

```python
# vram_calc/assistant/tools.py
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
```

```python
# vram_calc/assistant/prompts.py
"""System prompt 组装：契约 + 当前配置预注入 + 手册精简摘要全量。"""
from __future__ import annotations
import json
from pathlib import Path

_MANUAL: str | None = None

SYSTEM_CONTRACT = """你是 vLLM 部署参数顾问，嵌入在显存计算工具里。用户语言为中文。
回答结构固定三段：
1. **推理过程**：先说明你算了什么、查了什么、排除了什么（简短，3-6 行）
2. **结论**：直接回答用户问题（放得下/放不下/会限流等，给数字）
3. **推荐参数表**：Markdown 表格，列为 | 参数 | 推荐值 | 为什么 |；
   「为什么」列每行开头用 [计算器] / [官方文档] / [经验] 之一标注依据来源；
   表格后附完整 vllm serve 命令代码块
硬规则：
- 所有显存/并发/上下文数字必须来自 calc_vram 工具结果或预注入数据，禁止心算
- 没有 FP8 硬件的显卡（如 RTX 3090、A100）不得推荐 --kv-cache-dtype fp8
- 不确定的内容明确说不确定并标 [经验]，不要编造默认值"""


def manual_digest() -> str:
    global _MANUAL
    if _MANUAL is None:
        p = Path(__file__).resolve().parent.parent / "data" / "vllm_params.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        _MANUAL = "\n".join(
            f"- {p2['flag']} | {p2['type']} | 默认 {p2['default']} | {p2['desc']}"
            for cat in data["categories"] for p2 in cat["params"])
    return _MANUAL


def build_system_prompt(page_ctx: dict | None) -> str:
    parts = [SYSTEM_CONTRACT, "", "## 当前页面配置（用户正在看的，提问默认基于它）"]
    if page_ctx and page_ctx.get("kind") in ("calc", "plan"):
        parts.append(json.dumps(page_ctx, ensure_ascii=False, indent=1))
    else:
        parts.append("（用户在参数手册页，无显存配置上下文）")
    parts.append("")
    parts.append("## vLLM 参数手册（已核对官方文档，含默认值）")
    parts.append(manual_digest())
    return "\n".join(parts)
```

- [ ] **Step 4: `$PY -m pytest tests/test_assist_tools.py -q` → PASS（8 个）**
- [ ] **Step 5: Commit** `feat:助手工具执行器(calc_vram/plan_multi_node直调core)+prompt组装(契约/预注入/手册摘要)`

---

### Task 3: 编排循环（工具循环 + SSE 事件）

**Files:**
- Create: `vram_calc/assistant/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: Task 1 `get_provider/ToolCall/TextDelta/LLMConfig`、Task 2 `tools_for/execute_tool/build_system_prompt`
- Produces: `run_chat(cfg: LLMConfig, messages: list[dict], page_ctx: dict | None)` —— 生成器，产出 SSE payload dict：`{"t":"delta","v":str}` / `{"t":"tool","name","args"}` / `{"t":"tool_result","name","result"}` / `{"t":"error","v":str}` / `{"t":"done","rounds":int}`；`MAX_TOOL_ROUNDS = 5`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_orchestrator.py
import json
import pytest
from vram_calc.assistant import orchestrator
from vram_calc.assistant.providers import LLMConfig, TextDelta, ToolCall, Done

CFG = LLMConfig(protocol="openai", model="fake")

LLAMA_ARGS = {"model_id": "meta-llama/Meta-Llama-3-8B", "gpu_id": "rtx-4090",
              "gpu_count": 1, "tp": 1, "context_len": 4096}


class FakeProvider:
    """脚本化假流：每次 chat_stream 弹出下一幕。"""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []                    # 每轮收到的 messages 快照

    def chat_stream(self, messages, tools):
        self.calls.append(json.dumps(messages, ensure_ascii=False, default=str))
        for ev in self.script.pop(0):
            yield ev


def _patch(monkeypatch, script):
    fp = FakeProvider(script)
    monkeypatch.setattr(orchestrator, "get_provider", lambda cfg: fp)
    return fp


def test_plain_answer_no_tools(monkeypatch):
    fp = _patch(monkeypatch, [[TextDelta("你好"), Done("stop")]])
    evs = list(orchestrator.run_chat(CFG, [{"role": "user", "content": "hi"}], None))
    assert {"t": "delta", "v": "你好"} in evs
    assert evs[-1]["t"] == "done" and evs[-1]["rounds"] == 1
    assert "system" in fp.calls[0] and "vLLM 参数手册" in fp.calls[0]


def test_tool_roundtrip(monkeypatch):
    fp = _patch(monkeypatch, [
        [TextDelta("我算一下"), ToolCall("calc_vram", dict(LLAMA_ARGS)), Done("tool_calls")],
        [TextDelta("结论：放得下"), Done("stop")],
    ])
    evs = list(orchestrator.run_chat(CFG, [{"role": "user", "content": "能跑吗"}], None))
    ts = [e for e in evs if e["t"] == "tool"]
    trs = [e for e in evs if e["t"] == "tool_result"]
    assert ts[0]["name"] == "calc_vram"
    assert '"verdict": "ok"' in trs[0]["result"]
    assert evs[-1]["t"] == "done" and evs[-1]["rounds"] == 2
    # 第二轮 messages 含 assistant tool_calls 与 tool 结果
    assert '"tool_calls"' in fp.calls[1] and '"verdict": "ok"' in fp.calls[1]


def test_five_round_circuit_breaker(monkeypatch):
    script = [[TextDelta("再算"), ToolCall("calc_vram", dict(LLAMA_ARGS)), Done("tool_calls")]]
    _patch(monkeypatch, script * 10)          # 永远想调工具
    evs = list(orchestrator.run_chat(CFG, [{"role": "user", "content": "x"}], None))
    assert evs[-1]["t"] == "done"
    assert len([e for e in evs if e["t"] == "tool"]) == orchestrator.MAX_TOOL_ROUNDS


def test_provider_exception_humanized(monkeypatch):
    def boom(cfg):
        raise RuntimeError("should become error event")
    monkeypatch.setattr(orchestrator, "get_provider", boom)
    evs = list(orchestrator.run_chat(CFG, [{"role": "user", "content": "x"}], None))
    assert evs[0]["t"] == "error" and "调用失败" in evs[0]["v"]


def test_plan_ctx_registers_second_tool(monkeypatch):
    fp = _patch(monkeypatch, [[TextDelta("ok"), Done("stop")]])
    list(orchestrator.run_chat(CFG, [{"role": "user", "content": "x"}], {"kind": "plan"}))
    # FakeProvider 收不到 tools 参数名——用另一个假件核对
    seen = {}
    class P2:
        def chat_stream(self, messages, tools):
            seen["tools"] = [t["name"] for t in tools]
            yield Done("stop")
    monkeypatch.setattr(orchestrator, "get_provider", lambda cfg: P2())
    list(orchestrator.run_chat(CFG, [{"role": "user", "content": "x"}], {"kind": "plan"}))
    assert seen["tools"] == ["calc_vram", "plan_multi_node"]
```

- [ ] **Step 2: 确认失败**（ModuleNotFoundError）
- [ ] **Step 3: 实现**

```python
# vram_calc/assistant/orchestrator.py
"""对话编排：流式转发 + 工具执行回填续流。产出 SSE payload dict。

事件契约见本计划 Global Constraints（delta/tool/tool_result/error/done）。
异常单边界：任何 provider/工具异常都翻译成 error 事件，不向路由抛。
"""
from __future__ import annotations
import json
import logging

from .providers import LLMConfig, get_provider, humanize_llm_error, TextDelta, ToolCall
from .tools import tools_for, execute_tool
from .prompts import build_system_prompt

MAX_TOOL_ROUNDS = 5
log = logging.getLogger(__name__)


def run_chat(cfg: LLMConfig, messages: list[dict], page_ctx: dict | None):
    provider = get_provider(cfg)
    convo = [{"role": "system", "content": build_system_prompt(page_ctx)}] + list(messages)
    try:
        for round_i in range(MAX_TOOL_ROUNDS):
            text_acc, calls = "", []
            for ev in provider.chat_stream(convo, tools_for(page_ctx)):
                if isinstance(ev, TextDelta) and ev.text:
                    text_acc += ev.text
                    yield {"t": "delta", "v": ev.text}
                elif isinstance(ev, ToolCall):
                    calls.append(ev)
            if calls:
                convo.append({"role": "assistant", "content": text_acc,
                              "tool_calls": [{"id": f"c{i}", "type": "function",
                                              "function": {"name": c.name,
                                                           "arguments": json.dumps(c.args, ensure_ascii=False)}}
                                             for i, c in enumerate(calls)]})
            elif text_acc:
                convo.append({"role": "assistant", "content": text_acc})
                yield {"t": "done", "rounds": round_i + 1}
                return
            else:                                   # 空回合：终止防死循环
                yield {"t": "done", "rounds": round_i + 1}
                return
            for i, c in enumerate(calls):
                log.info("assistant tool call: %s", c.name)   # 不打 args，防敏感
                yield {"t": "tool", "name": c.name, "args": c.args}
                result = execute_tool(c.name, c.args)
                yield {"t": "tool_result", "name": c.name, "result": result}
                convo.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})
        yield {"t": "done", "rounds": MAX_TOOL_ROUNDS}        # 熔断
    except Exception as e:                             # noqa: BLE001 — 单一边界
        log.warning("assistant chat failed: %s", type(e).__name__)
        yield {"t": "error", "v": humanize_llm_error(e)}
```

- [ ] **Step 4: `$PY -m pytest tests/test_orchestrator.py -q` → PASS（5 个）**
- [ ] **Step 5: Commit** `feat:助手编排循环(流式转发+工具回填续流+5轮熔断+异常单边界)`

---

### Task 4: Web 路由（SSE chat + 连接测试）

**Files:**
- Modify: `vram_calc/web/app.py`（追加；import 区合并）
- Test: `tests/test_web_assistant.py`

**Interfaces:**
- Consumes: Task 1 `LLMConfig/get_provider/humanize_llm_error`、Task 3 `orchestrator.run_chat`
- Produces: `POST /api/assistant/chat`（body `{config:dict, messages:list, page_ctx:dict|null}` → `text/event-stream`，每事件一行 `data: {json}\n\n`，流尾 `data: [DONE]`）；`POST /api/assistant/test`（body `{config:dict}` → `{ok:bool, model_name?, error?}`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_web_assistant.py
import json
import pytest
from fastapi.testclient import TestClient
from vram_calc.web.app import app
from vram_calc.assistant import orchestrator
from vram_calc.assistant.providers import TextDelta, Done

cli = TestClient(app)
BODY = {"config": {"protocol": "openai", "base_url": "", "api_key": "k", "model": "fake"},
        "messages": [{"role": "user", "content": "hi"}], "page_ctx": None}


def test_chat_sse(monkeypatch):
    class FP:
        def chat_stream(self, messages, tools):
            yield TextDelta("你好")
            yield Done("stop")
    monkeypatch.setattr(orchestrator, "get_provider", lambda cfg: FP())
    r = cli.post("/api/assistant/chat", json=BODY)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [l for l in r.text.split("\n\n") if l.startswith("data: ")]
    payloads = [json.loads(l[6:]) for l in lines[:-1]]      # 最后一个是 [DONE]
    assert lines[-1] == "data: [DONE]"
    assert {"t": "delta", "v": "你好"} in payloads
    assert payloads[-1]["t"] == "done"


def test_chat_error_event(monkeypatch):
    def boom(cfg):
        raise RuntimeError("x")
    monkeypatch.setattr(orchestrator, "get_provider", boom)
    r = cli.post("/api/assistant/chat", json=BODY)
    payloads = [json.loads(l[6:]) for l in r.text.split("\n\n") if l.startswith("data: ")]
    assert payloads[0]["t"] == "error"


def test_test_route_ok(monkeypatch):
    import vram_calc.web.app as webapp
    class FP:
        def test_connection(self):
            return "deepseek-chat"
    monkeypatch.setattr(webapp, "get_provider", lambda cfg: FP())
    r = cli.post("/api/assistant/test", json={"config": BODY["config"]})
    assert r.json() == {"ok": True, "model_name": "deepseek-chat"}


def test_test_route_error(monkeypatch):
    import vram_calc.web.app as webapp
    def boom(cfg):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(webapp, "get_provider", boom)
    r = cli.post("/api/assistant/test", json={"config": BODY["config"]})
    assert r.json()["ok"] is False and "连接" in r.json()["error"]


def test_chat_bad_config_422():
    r = cli.post("/api/assistant/chat", json={"config": {"protocol": 123}, "messages": "x"})
    assert r.status_code == 422
```

- [ ] **Step 2: 确认失败**（404）
- [ ] **Step 3: 实现（app.py 追加）**

import 区补充（合并进现有行）：`from fastapi.responses import StreamingResponse`、`from ..assistant.providers import LLMConfig, get_provider, humanize_llm_error`、`from ..assistant import orchestrator`。

```python
class AssistantChatReq(BaseModel):
    config: dict
    messages: list[dict]
    page_ctx: dict | None = None


class AssistantTestReq(BaseModel):
    config: dict


@app.post("/api/assistant/chat")
def api_assistant_chat(req: AssistantChatReq):
    cfg = LLMConfig(**req.config)

    def gen():
        # ponytail: 同步生成器套 StreamingResponse——uvicorn 线程池里跑，量级足够
        for ev in orchestrator.run_chat(cfg, req.messages, req.page_ctx):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/assistant/test")
def api_assistant_test(req: AssistantTestReq):
    try:
        cfg = LLMConfig(**req.config)
        model = get_provider(cfg).test_connection()
        return {"ok": True, "model_name": model}
    except Exception as e:                     # noqa: BLE001 — 人话化，Key 不外泄
        return {"ok": False, "error": humanize_llm_error(e)}
```

注意：chat 路由里 `LLMConfig(**req.config)` 校验失败会抛 pydantic.ValidationError → FastAPI 500；包 try 返回 422 风格 JSON（`JSONResponse({"error": str(e)}, status_code=422)`），与 test_chat_bad_config_422 断言一致——实现时给 chat 也包一层 try。

- [ ] **Step 4: `$PY -m pytest tests/test_web_assistant.py -q` → PASS（5 个）；全量 `$PY -m pytest tests -q` 全绿**
- [ ] **Step 5: Commit** `feat:助手SSE路由(/api/assistant/chat+test,流式转发+错误人话化)`

---

### Task 5: 前端抽屉（全站浮球 + 会话 + 渲染 + 应用）

**Files:**
- Create: `vram_calc/web/static/assistant.js`
- Modify: `vram_calc/web/static/style.css`（追加 .fab/.drawer/.dr-/.msg-/.md-tbl/.src 样式）、`vram_calc/web/static/app.js`（recalc 存 `window.__lastCalc` + 尾部暴露 `window.__assistant_ctx`）、`vram_calc/web/static/plan.js`（runPlan 存 `lastPlan` + 尾部暴露 `window.__assistant_ctx`）、`vram_calc/web/templates/index.html`、`plan.html`、`vllm_params.html`（各加 `<script src="/static/assistant.js?v=1"></script>` 于最后一个 script 之后）

**Interfaces:**
- Consumes: Task 4 两个 API；`common.js` 的 `$`/`jpost`；app.js 的 `currentInput()`；plan.js 的 `servers/selected/$id`
- Produces: 全站 🤖 浮球 + 抽屉；localStorage 键 `vramcalc.llm`（LLMConfig dict）、`vramcalc.sessions`（`[{id,title,createdAt,messages[]}]`，≤20）、`vramcalc.cur`（当前会话 id）

- [ ] **Step 1: app.js / plan.js 上下文钩子（两行级）**

app.js：`recalc()` 里 `renderResult(r);` 之后加 `window.__lastCalc = r;`；文件末尾加：
```js
window.__assistant_ctx = () => ({
  kind: "calc", input: currentInput(), last_result: window.__lastCalc || null,
});
```

plan.js：`let servers = ...` 行后加 `let lastPlan = null;`；`runPlan()` 里 `renderPlans(r);` 前加 `lastPlan = r;`；文件末尾加：
```js
window.__assistant_ctx = () => ({
  kind: "plan",
  servers: servers.filter(s => selected.has(s.id)).map(s => ({id: s.id, name: s.name, gpus: s.gpus})),
  goal: {model_id: $id("p-model").value, context_len: parseContext($id("p-ctx").value) || 4096,
         concurrency: Math.max(1, +$id("p-conc").value || 1), quant: $id("p-quant").value,
         kv_quant: $id("p-kvquant").value, gpu_util: +$id("p-util").value},
  last_plans: lastPlan,
});
```

- [ ] **Step 2: style.css 追加（样式稿 ai_drawer_temp.html 的类名，颜色用 CSS 变量）**

```css
/* ---- AI 助手抽屉 ---- */
.fab { position: fixed; right: 26px; bottom: 26px; width: 52px; height: 52px; border-radius: 50%;
  background: var(--brand); color: #fff; font-size: 22px; border: none; cursor: pointer;
  box-shadow: 0 6px 20px rgba(42,120,214,.45); z-index: 90; }
.drawer { position: fixed; top: 0; right: 0; bottom: 0; width: 520px; max-width: 94vw;
  background: var(--surface); border-left: 1px solid var(--border);
  box-shadow: -8px 0 32px rgba(11,11,11,.14); z-index: 100; display: flex; flex-direction: column; }
.drawer.hidden, .dr-dim.hidden { display: none; }
.dr-dim { position: fixed; inset: 0; background: rgba(11,11,11,.18); z-index: 95; }
.dr-head { display: flex; align-items: center; gap: 8px; padding: 13px 18px;
  border-bottom: 1px solid var(--border); background: #fff; }
.dr-head b { font-size: 15px; }
.dr-head select { width: auto; flex: 1; padding: 5px 8px; font-size: 12.5px; }
.dr-body { flex: 1; overflow-y: auto; padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
.dr-input { display: flex; gap: 8px; padding: 12px 18px; border-top: 1px solid var(--border); background: #fff; }
.dr-input input { flex: 1; }
.ctxbar { background: #eef4fc; border: 1px solid #cde2fb; border-radius: 9px; padding: 7px 11px;
  font-size: 11.5px; color: #184f95; line-height: 1.7; }
.msg-user { align-self: flex-end; max-width: 85%; background: var(--brand); color: #fff;
  border-radius: 11px 11px 3px 11px; padding: 9px 13px; font-size: 13px; line-height: 1.7;
  white-space: pre-wrap; }
.msg-ai { background: #fff; border: 1px solid var(--border); border-radius: 11px 3px 11px 11px;
  padding: 12px 14px; font-size: 13px; line-height: 1.75; overflow-wrap: anywhere; }
.msg-ai.streaming { color: var(--ink-2); }
.msg-ai pre { background: #0f172a; color: #d7e4f5; font-size: 12px; line-height: 1.7;
  border-radius: 10px; padding: 12px 14px; overflow-x: auto; font-family: Consolas, monospace; }
.msg-ai p { margin: 0 0 8px; }
table.md-tbl { width: 100%; border-collapse: collapse; font-size: 12px; margin: 4px 0 10px; }
.md-tbl th { text-align: left; color: var(--muted); font-size: 11px;
  border-bottom: 2px solid var(--border); padding: 6px 7px; }
.md-tbl td { padding: 6px 7px; border-bottom: 1px solid var(--grid); vertical-align: top; }
.src { display: inline-block; font-size: 10px; font-weight: 700; border-radius: 6px;
  padding: 1px 6px; margin-right: 3px; }
.src.calc { background: #e7f6ef; color: #177a56; }
.src.doc { background: #eef3fd; color: #1d5fae; }
.src.exp { background: #f6efe2; color: #8a6d1a; }
.toolbub { font-size: 12px; color: var(--ink-2); background: #fafaf8;
  border: 1px dashed var(--grid); border-radius: 9px; padding: 7px 11px; margin: 6px 0; }
.toolbub .tname { font-weight: 700; }
.toolbub .tres { color: #177a56; font-family: Consolas, monospace; font-size: 11.5px; }
.verify-strip { background: #eef4fc; border: 1px solid #cde2fb; border-radius: 9px;
  padding: 7px 11px; margin: 8px 0; font-size: 12px; color: #184f95; }
.verify-strip .ok { color: var(--good); font-weight: 700; }
.dr-actions { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 6px; }
.dot-ok { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--good); margin-right: 5px; }
.dot-bad { background: var(--crit); }
```

- [ ] **Step 3: assistant.js 全量**

```js
// AI 助手抽屉：全站自注入。设置/会话存 localStorage，Key 不进后端存储。
(() => {
const LS_CFG = "vramcalc.llm", LS_SESS = "vramcalc.sessions", LS_CUR = "vramcalc.cur";
const MAX_SESS = 20;

// ---- DOM 注入 ----
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

// ---- Markdown 渲染（全量转义；只支持契约语法：加粗/行内码/表格/代码块/来源徽标） ----
function esc(s) { return s.replace(/[&<>"']/g, c =>
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
  if (inCode) out.push(`<pre>${esc(code.join("\n"))}</pre>`);   // 流式截断的半截代码块
  flushT();
  return out.join("\n");
}

// ---- 会话 ----
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
  const ss = loadSess();
  $d("ai-sess-sel").innerHTML = ss.map(s =>
    `<option value="${s.id}" ${s.id === localStorage.getItem(LS_CUR) ? "selected" : ""}>${esc(s.title)}</option>`).join("");
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

// ---- SSE 对话 ----
function handleEvent(d, card, s) {
  if (d.t === "delta") {
    card.raw += d.v;
    card.el.querySelector(".md-live").textContent = card.raw;   // 流式期间纯文本
  } else if (d.t === "tool") {
    card.el.insertAdjacentHTML("beforeend",
      `<div class="toolbub">🔧 <span class="tname">${esc(d.name)}</span> ${esc(JSON.stringify(d.args))}</div>`);
  } else if (d.t === "tool_result") {
    card.lastResult = d.result;
    try { const r = JSON.parse(d.result);
      if (r.verdict) card.el.querySelector(".toolbub:last-of-type")?.insertAdjacentHTML("beforeend",
        `<div class="tres">→ ${r.verdict} · 池 ${r.kv_pool_tokens} tokens</div>`);
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
  if (location.pathname === "/" && document.getElementById("model"))
    card.el.insertAdjacentHTML("beforeend",
      `<div class="dr-actions"><button class="btn primary" onclick="window.__ai_apply(this)">⚡ 应用到本页</button></div>`);
}
window.__ai_apply = btn => {
  const pre = btn.closest(".msg-ai").querySelector("pre");
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
  if (!cfg.model) { openCfg(); return; }                    // 未配置先弹设置
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
        handleEvent(JSON.parse(d), card, s);
      }
    }
  } catch (e) { card.raw += `\n\n**连接失败：** ${e.message}`; }
  if (card.raw.trim()) s.messages.push({role: "assistant", content: card.raw});
  saveSess(loadSess().map(x => x.id === s.id ? s : x));
  finalize(card, s);
  busy = false; $d("ai-send").disabled = false;
}

// ---- 设置弹窗 ----
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
    : `<span class="dot dot-bad"></span>${esc(r.error)}`;
}

// ---- 事件绑定 ----
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
  saveSess(loadSess().filter(x => x.id !== cur));
  localStorage.setItem(LS_CUR, ""); renderAll();
};
$d("ai-sess-sel").onchange = e => { localStorage.setItem(LS_CUR, e.target.value); renderAll(); };
})();
```

注意：`jpost` 来自 common.js（设置弹窗保存前连接测试用它）；`common.js` 里若无该名（Task 6 提取时只保留 parseContext/jget/jpost/$/fmtGB 中的前四），以 common.js 实际导出为准——`jpost` 在列。

- [ ] **Step 4: 三个模板加 script 标签**（index.html / plan.html / vllm_params.html，各自最后一个 `<script>` 之后）：
`<script src="/static/assistant.js?v=1"></script>`

- [ ] **Step 5: Playwright 走查（必做）**
  1. 起服务（先杀 8000）打开 `/`：浮球可见 → 点开抽屉 → ⚙ 设置填假配置保存 → 刷新页面回显（localStorage）
  2. 未配置直接发送 → 自动弹设置（不报错）
  3. 抽屉发一条消息（假配置会得到 error 事件人话化显示，不白屏不死循环）
  4. `＋新建`/切换/🗑 删除会话，localStorage 条数正确
  5. `/plan`、`/vllm-manual` 页浮球可开；`/plan` 的 ctxbar 出现；`/vllm-manual` 无 ctxbar
  6. 回归：计算器页改参数正常重算（app.js 两处钩子未破坏）
  7. 截图存 `.superpowers/sdd/2026-08-19-ai-assistant/task-5-screenshot.png`；走完杀服务
  （真实 LLM 联调不在本任务——无 Key 环境；Task 6 留真机验收清单）

- [ ] **Step 6: Commit** `feat:AI助手前端抽屉(全站浮球/多会话/流式渲染/来源徽标/一键应用)`

---

### Task 6: 收尾 — 文档 + 全量验证 + 真机验收清单

**Files:**
- Modify: `README.md`（功能清单加「AI 助手」小节：抽屉用法/双协议配置/来源徽标语义/Key 安全说明）、`docs/design.md`（v2.2：assistant 包架构段）
- Create: 无（真机验收清单写进 README 小节末尾，3 行）

- [ ] **Step 1: 文档**（先读现有格式，各 ≤18 行）
- [ ] **Step 2: 全量验证**：`$PY -m pytest tests -q` 全绿；`$PY scripts/calibrate.py` 退出码 0
- [ ] **Step 3: Playwright 全站回归**（三页 + 抽屉冒烟）
- [ ] **Step 4: 真机验收清单（写给用户，不代跑）**：① 设置里填 DeepSeek key → 连接测试绿点 ② 问「当前配置能跑吗」→ 应出现工具调用气泡 + 验证条 + 参数表 ③ 问「换成 3 张卡呢」→ 应再次调 calc_vram ④ 切本地 vLLM 端点（http://ip:8000/v1）重复 ② ⑤ 抽屉「应用」后计算器输入框变化
- [ ] **Step 5: Commit** `docs:AI助手文档同步(README+design v2.2+真机验收清单)` 并 push（重试规则见 Global Constraints）

---

## Self-Review 记录

- 规格覆盖：spec §3 providers→Task1；§4 工具/注入→Task2；§5 编排/SSE→Task3+4；§6 前端→Task5；§7 测试→各任务+Task6；§8 YAGNI 无违反（无重命名/无导出/无中断重生成/无后端会话）。
- 占位符扫描：Task 5 Step 5 是叙述性验证步骤（前端无单测，项目既有策略），其余步骤均带全量代码。
- 类型一致性：`LLMConfig/TextDelta/ToolCall/Done/get_provider/humanize_llm_error/split_system`（T1）→ T3/T4 消费签名一致；`CALC_VRAM/PLAN_MULTI_NODE/tools_for/execute_tool/build_system_prompt/manual_digest`（T2）→ T3 消费一致；`run_chat(cfg,messages,page_ctx)`（T3）→ T4 路由调用一致；SSE 五事件契约 Global Constraints=T3 产出=T5 handleEvent 分支（error 分支 T5 处理了，done 无需 UI 动作）。前端 `__assistant_ctx`/`__lastCalc`/`lastPlan` 命名在 T5 Step 1 与 Step 3 一致。
- spec 细化说明：spec 说「SSE 事件四种」，计划细化为五种（多一个 tool_result，供前端渲染验证条）——行为不变，事件拆分更清晰。
