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
    assert convo[1]["content"][0]["type"] == "text"
    assert convo[1]["content"][0]["text"] == "想算"
    assert convo[1]["content"][1]["type"] == "tool_use"
    assert convo[1]["content"][1]["input"] == {"tp": 2}
    assert convo[2]["content"][0]["type"] == "tool_result"
    assert convo[2]["content"][0]["tool_use_id"] == "c0"


def test_split_system_merges_consecutive_user():
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    _, convo = split_system(msgs)
    assert len(convo) == 1 and "a" in convo[0]["content"] and "b" in convo[0]["content"]
