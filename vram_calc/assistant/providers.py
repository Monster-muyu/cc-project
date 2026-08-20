# vram_calc/assistant/providers.py
"""LLM provider adapters: unify OpenAI-compatible + Anthropic streaming.

messages/tools 参数一律用 OpenAI 格式进来（orchestrator 的通用货币），
AnthropicProvider 在边界上翻译。真实网络调用只发生在 chat_stream/test_connection，
测试用 FakeProvider 替换。
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

    @abstractmethod
    def list_models(self) -> list[str]:
        """列出服务端可用模型 id。失败 raise 原始异常。"""


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
        kw = {"model": self.cfg.model, "messages": messages, "stream": True}
        if tools:                                  # 空 tools 数组两家 API 都 400
            kw["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = client.chat.completions.create(**kw)
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

    def list_models(self):
        client = self._client()
        return sorted(m.id for m in client.models.list())


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
        kw = {"model": self.cfg.model,
              "system": "\n\n".join(sys_texts) or None,
              "messages": convo, "max_tokens": 4096}
        if tools:                                  # 空 tools 数组两家 API 都 400
            kw["tools"] = a_tools
        with client.messages.stream(**kw) as stream:
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

    def list_models(self):
        client = self._client()
        return sorted(m.id for m in client.models.list())


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
