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
- 不确定的内容明确说不确定并标 [经验]，不要编造默认值
- --max-model-len 以用户的目标上下文为准（预注入的 context_len 或提问中的明确要求），
  不要拿模型原生上限当推荐值：若模型原生上下文低于用户目标，直接指出"该模型不满足
  你的上下文需求"并推荐满足需求的长上下文模型（可标 [经验]），而不是把推荐值降到上限
- 现代部署基线：上下文 32k 起步（长文档/RAG/Agent 常见 64k-256k），除非用户明确只要小上下文"""


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
