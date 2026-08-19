"""对话编排：流式转发 + 工具执行回填续流。产出 SSE payload dict。

事件契约：delta / tool / tool_result / error / done。
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
    try:
        provider = get_provider(cfg)
        convo = [{"role": "system", "content": build_system_prompt(page_ctx)}] + list(messages)
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
