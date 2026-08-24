"""提示词注入防护：输入拦截（免 LLM 调用）+ 输出泄露哨兵。

正则只拦固定攻击话术；模糊的话题判断交给系统提示词里的规则锁。
没有 100% 的防注入——这里的目标是拦掉随意试探的绝大多数。
"""
from __future__ import annotations
import re

REFUSAL = "🚫 该请求试图绕过助手限制或获取内部指令，已拒绝。我只回答显存估算与推理部署相关问题。"
TRUNCATED = "\n\n⚠️ 检测到内部指令泄露，回答已截断。"

# 覆写类（要求无视既有规则）+ 窃取类（索要系统提示词/内部规则）
_BLOCK_PATTERNS = [
    r"忽略.{0,4}(指令|提示|设定|规则|约束)",
    r"无视.{0,4}(指令|提示|设定|规则)",
    r"ignore\s+.*?(instructions?|prompts?|rules?)",
    r"(进入|启用|打开|激活).{0,8}(开发者模式|开发模式|developer\s*mode|dan\s*mode)",
    r"(扮演|充当|假装).{0,12}(没有|无|不受|忽略).{0,4}(限制|约束|规则)",
    r"(jailbreak|越狱|绕过|解除).{0,4}(限制|防护|规则)",
    r"(你的|内部|初始)(系统)?(提示词?|指令|设定|规则|prompt)",
    r"(输出|打印|复述|显示|透露|泄露|展示|给出|告诉我).{0,8}(提示词?|初始指令|系统设定|你的?prompt)",
    r"(repeat|show|print|reveal|display|output).{0,12}(system\s*prompt|your\s+prompt|initial\s+instructions?)",
    r"(what|which)\s+is\s+your\s+(system\s+)?(prompt|instruction)",
]
_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]

# 提示词原文才有的特征串（回答里合法出现的「推理过程/结论」等结构头不在列）
LEAK_SENTINELS = (
    "你是 vLLM 部署参数顾问",
    "硬规则",
    "## 当前页面配置",
    "## vLLM 参数手册",
)


def input_blocked(messages: list[dict]) -> bool:
    """任意一条用户消息（含历史轮）命中攻击模式即拦截。"""
    for m in messages:
        if m.get("role") != "user":
            continue
        text = m.get("content")
        if isinstance(text, str) and any(r.search(text) for r in _BLOCK_RE):
            return True
    return False


def leaks(text: str) -> bool:
    """累积回答里出现提示词原文特征串 = 泄露。"""
    return any(s in text for s in LEAK_SENTINELS)
