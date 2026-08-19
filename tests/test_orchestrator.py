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
    # tool 结果以字符串进 content，快照 json.dumps 后内部引号被转义
    assert '"tool_calls"' in fp.calls[1] and '\\"verdict\\": \\"ok\\"' in fp.calls[1]


def test_five_round_circuit_breaker(monkeypatch):
    script = [[TextDelta("再算"), ToolCall("calc_vram", dict(LLAMA_ARGS)), Done("tool_calls")]]
    _patch(monkeypatch, script * 10)
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
    seen = {}
    class P2:
        def chat_stream(self, messages, tools):
            seen["tools"] = [t["name"] for t in tools]
            yield Done("stop")
    monkeypatch.setattr(orchestrator, "get_provider", lambda cfg: P2())
    list(orchestrator.run_chat(CFG, [{"role": "user", "content": "x"}], {"kind": "plan"}))
    assert seen["tools"] == ["calc_vram", "plan_multi_node"]
