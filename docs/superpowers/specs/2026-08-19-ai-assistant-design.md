# AI 部署助手 — 设计文档

日期：2026-08-19
状态：已与用户逐节确认（5 节全过，含两份样式稿：独立页稿与抽屉稿，最终采抽屉）

## 1. 背景与目标

显存计算工具已有计算器（单机估算）与多机规划两个页面。AI 助手作为全局侧边抽屉内嵌：
用户在任意页带着当前配置提问，助手先调本工具估算引擎拿真实数字、检索参数手册，
再组织成带来源依据的 vLLM 部署参数建议，支持一键应用回页面。

防幻觉是核心机制：显存数字必须来自估算引擎，文档依据来自 vllm_params.json，
模型自身知识只能以「经验」徽标出现。

## 2. 已确认的决策

| 决策点 | 结论 |
|---|---|
| 页面形态 | 全站浮球 🤖 + 右侧抽屉（非独立页）；520px，设置收进 ⚙ 弹窗 |
| 连接架构 | 后端代理：Key 随请求内存中转不落盘，后端组装 prompt、转发 SSE |
| 估算引擎调用 | 混合：当前配置预注入 + `calc_vram` 工具调用（假设性问题也能拿真数字） |
| 输出方式 | SSE 流式；工具调用时推状态事件；表格 done 后整渲染 |
| 协议范围 | OpenAI 兼容（DeepSeek/Qwen/GLM/vLLM/Ollama 全覆盖）+ Anthropic 原生，v1 双协议 |
| 会话 | 多会话一步到位：localStorage sessions[]（上限 20，自动标题，无重命名） |
| LLM 客户端 | 官方 SDK（openai + anthropic 两个新依赖），薄适配层统一 StreamEvent |
| 实现架构 | `vram_calc/assistant/` 包（providers/tools/prompts/orchestrator）+ 2 个 SSE 路由 + 全站 assistant.js |

## 3. Provider 适配层（assistant/providers.py）

```python
class LLMProvider(ABC):
    def chat_stream(self, messages, tools, model) -> Iterator[StreamEvent]
# StreamEvent: TextDelta(text) | ToolCall(name, args) | Done(reason)
```

- OpenAIProvider：openai SDK，base_url 参数切换厂商；AnthropicProvider：anthropic SDK，
  翻译成统一 StreamEvent
- LLMConfig = {protocol: "openai"|"anthropic", base_url, api_key, model}——前端随身带
- 连接测试：1-token 最小对话；SDK 异常翻译成人话（401/超时/URL 不通）
- 手册注入：vllm_params.json 压缩摘要（flag/默认值/一句话，~6KB）全量进 system prompt

## 4. 工具与注入（assistant/tools.py + prompts.py）

**calc_vram 工具**（模型侧 schema）：
params = model_id, gpu_id, gpu_count, tp, pp, ep, context_len, concurrency,
quant, kv_quant, gpu_util, max_num_batched_tokens。
gpu_count 仅用于校验 tp×pp 一致性（不一致返回错误文本让模型重调），
估算本身用 tp/pp/ep。
执行器直调 core.estimate（不走 HTTP），返回 verdict/每卡占用/可用/KV 池 tokens/
最大并发；校验失败返回错误文本（模型自纠重调）。
多机页上下文额外注册 plan_multi_node(server_ids, …) → 调 planner 返回方案摘要。

**prompt 组装（每轮）**：
① 角色与输出契约（推理过程 → 结论 → 参数表，依据标 [计算器]/[官方文档]/[经验]）
② 当前配置预注入 + estimate 结果 JSON（多机页换服务器清单+最近方案）
③ 手册精简摘要全量
④ 硬规则：显存数字必须来自 calc_vram；无 FP8 硬件的卡不得推荐 fp8 KV；
   不确定就说不确定
messages = 完整历史（含工具调用记录）

**上下文携带**：计算器页 = currentInput() 全字段 + 页面估算结果；
plan 页 = 勾选服务器 + 规划目标 + 最近方案 JSON；手册页 = null。

## 5. 编排循环与 SSE（assistant/orchestrator.py + web 路由）

生成器循环，最多 5 轮工具调用熔断：
TextDelta → SSE {"t":"delta"}；ToolCall → SSE {"t":"tool",name,args} → 执行 →
结果追加为 tool 消息 → 续流；Done → 需要则下一轮，否则 {"t":"done"}。

前端事件四种：delta / tool / error / done。
推荐参数表 = 模型按契约输出 Markdown 表格（参数|推荐值|为什么），前端渲染成样式
（[来源] 前缀替换为三色徽标）。不做结构化 JSON 解析（流式 + 严格解析是坑）。
显存验证条由前端从最后一次 tool 事件结果渲染，数字 100% 来自引擎。

```
POST /api/assistant/chat   # {config, messages, page_ctx} → SSE
POST /api/assistant/test   # {config} → {ok, model_name, error?}
```

后端无状态；异常日志对 Key 脱敏。

## 6. 前端抽屉（static/assistant.js + style.css，全站挂载）

- common.js 之后引入 assistant.js，自注入浮球+抽屉 DOM，各页零改动
- 抽屉头：会话下拉/＋新建/🗑删当前/⚙设置/✕收起；设置存 localStorage `vramcalc.llm`
- 上下文条：调当前页暴露的 `window.__assistant_ctx()`（app.js/plan.js 各自提供）
- Markdown 渲染自写 ~60 行（加粗/表格/代码块），全量 HTML 转义
  （助手输出是不可信输入，补上历史欠账）；表格流式期间显示纯文本，done 后整卡重渲染
- ⚡ 应用到本页：index 页按参数映射表回填输入框并 recalc；plan 页隐藏按钮
- v1 发送后不可中断

## 7. 测试

- pytest 离线：tools 执行器结果断言与错误路径；prompts 包含断言；
  orchestrator 用 FakeProvider（脚本化假流：delta→toolcall→done）断言事件序列、
  工具回填、5 轮熔断；TestClient 走 SSE 与 test 路由
- 真实 API 手动验收（DeepSeek key + 本地 vLLM 各一遍）
- Playwright：浮球全站/设置回显/多会话管理/应用回填触发重算

## 8. 不做（YAGNI）

后端会话存储、token 计数/费用/限流、多模型并行对比、流式中断重生成、
手册检索/RAG、会话重命名与导出、多工具注册扩展框架（就两个工具，直接写）
