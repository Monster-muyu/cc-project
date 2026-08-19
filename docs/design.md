# VRAM 显存计算工具 — 设计文档

> 版本: v2.2 · 状态: 已实现(与代码同步) · 初稿 2026-08-13 · 同步 2026-08-19
> v2.2 变更: 新增**AI 助手**(见 §16):`assistant/` 包四模块 + 两个 SSE 路由 + 前端抽屉,新依赖 openai/anthropic。
> v2.1 变更: 新增**多机规划**(见 §15):ServerSpec 服务器实体 + EntityStore 泛化存储、
> planner 枚举四类候选(单机/DP/跨机PP/跨机TP)并评分取 top3、commands 生成启动命令、
> `/api/plan` 与 `/plan` 页面。
> v2.0 变更: KV 改 vLLM 分页池模型(动态预算/max_kv_tokens);verdict 改三档真实语义
> (OOM=权重放不下 / 能跑·会限流=负载超池子 / 放得下);参数新增 max_num_batched_tokens
> 与显存利用率(gpu_memory_utilization);模型源新增 ModelScope;UI 新增并发↔上下文推荐、
> quant 自动识别锁定、显卡数量自由输入、上下文单位(k/m);敏感度图改"KV 需求 vs KV 预算"。

## 1. 概述与目标

一个本地运行的 Web 工具，帮助用户在部署大语言模型（LLM）前估算所需显存。覆盖**本地推理（llama.cpp）+ 服务端推理（vLLM/SGLang）**两类场景。

**核心价值**：选模型 + 选量化 + 选显卡 + 调参数 → 给出"能不能跑 / 还差多少 / 最多扛几路并发"。

## 2. 范围

### v1 纳入
- 模型库：精选基准 + 按 HF repo ID 拉取入库（参数量走 `model_info().safetensors`，架构走 `config.json`）
- 显卡库：静态基准 + 用户自定义添加
- 量化：FP16/BF16、FP8、INT8/INT4（GPTQ/AWQ）、GGUF（按 bpw 档位）、EXL2（指定位宽）
- 推理参数：上下文长度、并发请求数、KV 量化、CPU offload 比例
- 并行：张量并行（TP）、流水线并行（PP）、**专家并行（EP，简化版，仅 MoE）**、多节点、同构多卡
- 引擎开销画像：vLLM / llama.cpp / SGLang / Ollama
- 输出：结论三档（放得下/偏紧/放不下）+ 显存明细分解 + 3 张可视化图

### 明确不做（v1）
- ❌ 训练/LoRA 显存估算（后续版本）
- ❌ DP（数据并行）：每卡是完整副本，纯为吞吐，**不影响单卡显存**，故不纳入
- ❌ 异构集群（混合不同型号显卡）——vLLM/SGLang 张量并行强制同构，仅 llama.cpp 支持，建模为 bin-packing 难题，YAGNI
- ❌ 自动联网更新显卡库（爬虫解析脆、源会变，YAGNI）
- ❌ 推理速度（tokens/s）估算 —— 数据已预留（带宽/算力字段），后加粗估版成本低，v1 不做

## 3. 技术栈与环境

| 项 | 选择 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | HF 生态原生，数值/类型顺手 |
| 隔离环境 | conda 独立环境 `vram-calc` | 用户机 Anaconda 在 `E:\ANACONDA`，项目隔离 |
| Web 后端 | FastAPI + Uvicorn | 轻量，原生 async，路由清晰 |
| 模板 | Jinja2 | 服务端渲染，无前端构建链 |
| 前端 | 原生 HTML/CSS/JS + SVG | 三张图手绘 SVG，**零前端框架依赖** |
| HF 集成 | huggingface_hub | 仅"添加模型"时用 `model_info` + `hf_hub_download(config.json)` |
| 数据校验 | Pydantic | FastAPI 自带，请求/响应模型 |
| 元数据来源 | HF Hub API（`/api/models/{id}` 的 `safetensors.parameters`） | 参数量服务端算好，**零文件下载** |

## 4. 架构总览

```
┌─────────────────────────────────────────────────────┐
│  浏览器（原生 JS + SVG，三张图实时重算）              │
└────────────────────────┬────────────────────────────┘
                         │ fetch JSON
┌────────────────────────▼────────────────────────────┐
│  FastAPI（4 路由：页面 / calc / sweep / models）      │
└────────────────────────┬────────────────────────────┘
                         │ 调用
┌────────────────────────▼────────────────────────────┐
│  核心库 vram_calc/core（纯 Python，零外部依赖，可测）  │
│  estimator = weights + kv_cache + activation +        │
│              overhead，经 parallel(TP/PP/EP/offload)   │
└────────────────────────┬────────────────────────────┘
                         │ 读
┌────────────────────────▼────────────────────────────┐
│  数据层（运行时合并）                                  │
│  打包基准 data/models, data/gpus（只读）              │
│  ∪ 用户库 ~/.vram_calc/{models,gpus}（可写）         │
└─────────────────────────────────────────────────────┘
```

核心库独立、可 import、可单测；Web 层是薄壳。

## 5. 显存计算模型

**总公式**：`单卡显存 = 权重 + KV cache + 激活 + 开销`

### 5.1 权重（Weights）
```
权重显存 = 参数量 × 每参数字节数(量化格式)
```
量化 → 字节数表（`core/quant.py`）：

| 量化 | 字节/参数 | 备注 |
|---|---|---|
| FP16 / BF16 | 2.0 | 基准 |
| FP8 | 1.0 | |
| INT8（GPTQ/AWQ-8bit） | ~1.05 | 含分组缩放开销 |
| INT4（GPTQ/AWQ-4bit） | ~0.55 | 含分组开销 |
| GGUF | 按档 bpw ÷ 8：Q4_K_M≈4.5、Q5_K_M≈5.5、Q6_K≈6.6、Q8_0≈8.5 | bits-per-weight 表 |
| EXL2 | 用户指定位宽 ÷ 8 | |

### 5.2 KV cache —— vLLM 分页池模型(核心)

vLLM 启动时把 `gpu_memory_utilization` 扣掉权重+开销后的显存**全部划给一个分页 KV 池**,
池大小与 `max_model_len`、并发数**无关**:

```
KV 预算(每卡) = 显存利用率×卡容量 − 权重 − 激活 − 开销
KV 池(聚合)  = KV 预算 × (TP × PP)          # EP 卡只装专家不装 KV
每 token 字节 = 2 × 层数L × KV头数H_kv × 头维度D_h × KV元素字节   # 全模型口径
max_kv_tokens = KV 池 ÷ 每 token 字节        # 池子能装多少 token
```

- **用 KV 头数 H_kv**(GQA 关键:Llama-3-8B 32 注意力头仅 8 KV 头)
- KV 元素字节:FP16=2,FP8/INT8=1(由 `kv_quant` 决定)
- 运行时请求按需从池里拿块;池满 → 抢占/换出(变慢,**不 OOM**)
- 填 max_model_len 超池容量不会 OOM(部分 vLLM 版本会拒绝启动,见 README 注意事项)

### 5.3 激活(Transient prefill activation)
```
激活 ≈ max_num_batched_tokens × 隐藏维度 × 2字节 × 系数(12) ÷ (TP × PP)
```
- 由 vLLM 的 `max_num_batched_tokens`(chunked prefill 批量)驱动,**不是** 上下文×并发
- 激活值恒为 fp16(2 字节),与权重量化无关(计算精度仍是 fp16)
# ponytail: 系数 12 为经验值,标定见 scripts/calibrate.py

### 5.4 开销（Overhead，引擎相关）
```
开销 = 引擎固定基线 + 权重 × 引擎比例
```
`core/engines.py` 存每个引擎画像 `(基线GB, 权重比例)`：
- vLLM：基线较高（CUDA graphs、paged-attention 缓冲），比例中等
- llama.cpp：基线低，比例小
- SGLang / Ollama：各自一档
- 线性项**封顶** `cap_gb`:超大模型(如 DeepSeek-V3 661GB 权重)线性外推会失真(曾算出 41GB 开销),封顶后 vLLM ≤ 7.5GB

### 5.5 多卡分摊（TP/PP/EP，含多节点）

**Dense 模型**：
```
每卡权重 = 参数量 × 字节 ÷ (TP × PP)
每卡 KV  = (每层KV × KV头在TP上的份额) × (该PP阶段的层数 L/PP)
```

**MoE 模型**（权重拆 dense / expert 两部分）：
```
每卡权重 = dense参数 × 字节 ÷ (TP × PP)
         + 专家参数 × 字节 ÷ (EP × PP)     # 专家只由 EP 切，TP 不切专家
每卡 KV  = 同 dense（EP 不切注意力/KV）
```
- TP（张量并行）：层内切到 `tp` 张卡（节点内），切 dense 部分 + 注意力/KV
- PP（流水线并行）：层切到 `pp` 个阶段（常跨节点）
- EP（专家并行）：**仅 MoE**，专家权重 ÷ `ep` 张卡
- 多机：`总卡数 = 节点数 × 每节点卡数`，用户指定 TP/PP/EP 组合
- NCCL 通信缓冲：小的附加开销（引擎相关）
- 边界检查：EP ≤ 专家总数；整除不均按平均值近似
```
# ponytail: EP 假设专家均匀分布；shared expert（DeepSeek-V3）近似按 dense 处理
```

### 5.6 CPU offload（llama.cpp）
```
用户指定 GPU offload 比例 r（0~1，多少层在 GPU 上）
→ GPU 权重 = 总权重 × r；GPU 的 KV = 仅 offload 到 GPU 的层
其余在 CPU 内存，不计 VRAM
```

### 5.7 结论（Verdict）

已被 §5.8 取代——v2.0 按 vLLM 真实语义判定(OOM=固定部分超限;能跑·会限流=负载超池子;放得下)。

### 5.8 结论(Verdict,vLLM 真实语义)
```
over  (OOM 放不下) : 权重+激活+开销 > 显存利用率×容量 → 加载即爆,与上下文无关
ok    (放得下)     : 固定部分放得下,且 上下文×并发 ≤ max_kv_tokens
tight (能跑·会限流): 固定部分放得下,但负载超池子 → 抢占/排队,变慢不崩
```
UI 同时给出 KV 池容量、并发↔上下文对照表(用户并发高亮)、三条推荐
(保并发/保上下文/保守推荐——按 80% 预算 + 向下取整,保证照着输放得下)。

## 6. 数据来源

### 6.1 模型库
**精选基准**（`data/models/`，静态 JSON 随包分发），每条字段：
```json
{
  "id": "meta-llama/Meta-Llama-3-8B",
  "name": "Llama 3 8B",
  "params_b": 8, "layers": 32, "hidden_dim": 4096,
  "attn_heads": 32, "kv_heads": 8, "head_dim": 128,
  "vocab_size": 128256,
  "num_experts": 0, "expert_params_b": 0,
  "quantizations": ["fp16","int8","int4","gguf-q4_k_m","gguf-q5_k_m","gguf-q8_0"],
  "category": "llm"
}
```
（`num_experts` / `expert_params_b` 为 MoE 专用，dense 模型设 0。）

**按需拉取**(添加模型/批量导入弹窗,来源可选 HuggingFace / ModelScope):
- HF:`model_info().safetensors.parameters`(参数量精确)+ `config.json`(架构)
- ModelScope(魔搭,国内直连):`GET modelscope.cn/api/v1/models/{id}/repo?FilePath=config.json`
  取架构;其 API 无参数量明细(仅 StorageSize,量化后失真)→ **架构估算参数量(实测误差 ~1%)**,
  预览框可手改。模型 ID 存为 `ms/{repo_id}` 前缀以区分来源
- 预量化仓库(AWQ/GPTQ)从 config 的 `quantization_config` 自动识别量化并**锁定**下拉框;
  兜底:从仓库名推断(`infer_quant_from_id`,AWQ-INT4→int4 等)
- VL/多模态模型架构嵌套在 `text_config` 下,别名解析器自动下钻(视觉编码器忽略,轻微低估)
- MoE 专家数别名含 `n_routed_experts`(DeepSeek),专家参数按 3×hidden×intermediate×层数估算

### 6.2 config.json 字段名不统一 → 别名解析器（`core/arch_resolver.py`）
不同架构字段名不同，按优先级回退：
```
layers:   num_hidden_layers → num_layers → n_layer
kv_heads: num_key_value_heads → num_kv_heads → 回退 = attn_heads（MHA 老模型）
head_dim: head_dim → hidden_size / num_attention_heads
hidden:   hidden_size → n_embd → d_model
experts:  num_local_experts → num_experts → 0
```
**特殊架构**（带 ponytail 标注的已知简化）：
- DeepSeek MLA：KV 压缩（`kv_lora_rank`），KV cache 公式不同 → 近似
- MoE（Mixtral/DeepSeek-MoE）：参数量取**全部专家总数**（专家常驻显存），非"激活参数"

**关键原则**：参数量来自 `safetensors.parameters`（精确），不靠名字正则、不靠 `total_size÷字节`（混合/量化 dtype 会错）。解析可自动，落库须人确认。

### 6.3 显卡库
**静态基准**（`data/gpus/`，随包分发），每条：
```json
{
  "id": "nvidia-a100-80g", "name": "NVIDIA A100 80GB",
  "vram_gb": 80, "memory_bw_gbps": 2039, "architecture": "ampere",
  "vendor": "nvidia", "fp16_tflops": 312,
  "supports_fp8": false, "supports_bf16": true
}
```
用户实际只需 `vram_gb`；其余字段（带宽/算力）为预留——后续加推理速度粗估时用，当前仅做"显存够但带宽/算力或为瓶颈"的软提醒。
**自定义**：Web UI "添加显卡"表单，仅 `vram_gb` 必填 → 存 `~/.vram_calc/gpus/`，运行时与基准合并、不被升级覆盖。

## 7. Web UI 设计

单页左右分栏。

### 7.1 选择面板（左）
- 模型 ▼ → 量化 ▼（按模型支持项过滤）
- 显卡 ▼
- 推理参数:上下文长度(文本框,支持 32k/200k/1m 单位;=每请求**实际活跃上下文**,非最大窗口)、
  并发请求数、最大批处理(vLLM max_num_batched_tokens)
- 并行:推理引擎 ▼、**显卡数量(自由输入 1~128)**——自动选策略(dense→TP,MoE→EP,实时提示)、
  显卡利用率滑块(= gpu_memory_utilization)、KV 量化 ▼、CPU offload 滑块(llama.cpp 专属)、
  高级折叠内手动 TP/PP/EP(档位到 64)
- [🧮 计算] 按钮（移动端用，桌面端 debounce 300ms 自动算）

> 控件按性质分：枚举型（模型/显卡/引擎/TP/PP/EP）走下拉框 + 旁边"➕添加"入口（库里没有的走弹窗补，不在下拉里自由输入以保证数据完整）；数值型（上下文/并发）走数值输入框，支持手动输入。

### 7.2 结果面板（右）— 3 张图 + 明细
**图 1 · 容量条（结论区主视觉）**：GPU 总容量容器内填需求量，绿/黄/红，一眼看满没满。
**图 2 · 堆叠柱（明细区）**：权重/KV/激活/开销占比。
**图 3 · 敏感度曲线**:**KV 需求(上下文×并发×每token字节)vs KV 预算**——横轴扫并发,
橙线撞黄虚线(预算)处 = 最多可承载并发;调上下文/KV量化曲线实时变。带坐标轴/网格/刻度/图例。
另:KV 容量框(预算 GB + token 位 + 并发↔上下文表 + 三推荐);多卡时 headline 用合计口径
(可用=卡数×单卡,每卡分摊单独标注)。
+ 显存明细分解表（各项 GB + 百分比 + 合计 + 可用空间）。

三个图均原生 SVG + vanilla JS 客户端渲染，**零图表库依赖**。配色统一（亮/暗、色盲安全，实现时用 dataviz skill）。

### 7.3 模态弹窗
- **添加模型**：填 repo ID → 自动拉取解析 → 字段可编辑预览（尤其 KV 头数等架构字段）→ 确认入库
- **添加显卡**：名称 + 显存必填，带宽/架构选填

## 8. API 设计（4 路由）

```
GET  /                Jinja2 主页面
GET  /api/calc        单次显存分解（喂图1+图2）
GET  /api/sweep       扫变量返回点数组（图3）
POST /api/models      添加模型（拉取+解析+入库）
```
（另有 `GET /api/models`、`GET /api/gpus` 列出库，含用户自定义。）

`/api/calc` 请求：
```json
{ "model_id", "quant", "gpu_id",
  "context_len", "concurrency",
  "engine", "tp", "pp", "ep", "nodes",
  "kv_quant", "cpu_offload",
  "safety_factor", "max_num_batched_tokens" }
```
`/api/calc` 响应：
```json
{ "verdict": "ok",
  "total_gb": 19.87, "per_gpu_gb": 19.87,
  "capacity_gb": 24, "usable_gb": 21.6, "headroom_gb": 1.73,
  "breakdown": {"weights":16.06,"kv_cache":0.54,"activation":0.81,"overhead":2.46},
  "num_gpus": 1, "max_kv_tokens": 17327, "kv_budget_gb": 2.27 }
```
`total_gb/breakdown` 为**总占用**(含请求负载的 KV=上下文×并发;多卡为合计口径),池容量见 `max_kv_tokens`/`kv_budget_gb`。

`/api/sweep` 响应：
```json
{ "points": [{"x":1,"total_gb":4.19}, ...],
  "capacity_gb": 22.1, "usable_gb": 21.7, "max_x": 5, "kv_budget_gb": 22.1 }
```
`total_gb` 为 **KV 需求**(bpt×上下文×并发),容量线即 KV 预算。

普通计算零网络（模型已在本地库）；仅"添加模型"联网两次。

## 9. 项目结构
```
vram_calc/
├── data/
│   ├── models/          # 精选模型 JSON（打包分发）
│   └── gpus/            # 显卡 JSON（打包分发）
├── core/
│   ├── estimator.py     # 总估算：weights+kv+activation+overhead
│   ├── weights.py       # 权重（量化查表）
│   ├── kv_cache.py      # KV cache
│   ├── engines.py       # 引擎开销画像
│   ├── parallel.py      # TP/PP/EP/offload 分摊
│   └── arch_resolver.py # config.json 别名解析+例外清单
├── repos/
│   ├── model_repo.py    # 模型库 CRUD + HF 拉取
│   └── gpu_repo.py      # 显卡库查询
├── web/
│   ├── app.py           # FastAPI 路由
│   ├── templates/       # Jinja2 模板
│   └── static/          # CSS/JS（含 SVG 渲染）
├── tests/
│   └── test_estimator.py
├── docs/design.md       # 本文档
└── pyproject.toml
```
核心模块预计 300-400 行，纯数学零外部依赖。

## 10. 依赖
```
fastapi, uvicorn, jinja2, pydantic   # Web 层
huggingface_hub                       # 仅"添加模型"
（核心 estimator 零依赖）
```
Python 3.11+，conda 环境 `vram-calc`。

## 11. 已知简化与上限（ponytail 标注，带升级路径）
- 激活用经验公式，非实测 → 要精度接 profiling
- 引擎开销为画像系数，非实测 → 用户可调系数
- KV × 并发 = 最坏假设（每路并发占满上下文 C 的 KV）→ 线性近似，对分页注意力/前缀缓存偏保守
- EP（简化版）假设专家均匀分布，shared expert 近似按 dense 处理 → 模型结构极特殊时不准
- 异构集群不做 → 引擎支持改善后补
- 训练/LoRA 不做 → 后续版本

## 12. 环境搭建
```powershell
# 1. 建独立环境
& "E:\ANACONDA\Scripts\conda.exe" create -n vram-calc python=3.11 -y
# 2. 装依赖（项目根目录，pyproject.toml 就位后）
& "E:\ANACONDA\Scripts\conda.exe" run -n vram-calc pip install -e .
# 3. 启动（conda run 规避 PowerShell 非交互式 activate 问题）
& "E:\ANACONDA\Scripts\conda.exe" run -n vram-calc python -m vram_calc.web.app
```

## 13. 验收标准
- [ ] 核心库可独立 import，`estimator.estimate(...)` 返回结构化分解
- [ ] `test_estimator.py` 覆盖：各量化格式、KV cache（GQA/MHA）、TP/PP/**EP（含 MoE）**分摊、offload
- [ ] Web 页三张图随参数实时更新
- [ ] 添加模型：拉取 → 解析预览（可编辑）→ 入库，零权重下载
- [ ] 添加显卡：仅显存必填即可入库
- [ ] 用 Llama-3-8B + RTX 4090 的已知显存数据交叉验证估算误差 ≤ 10%
- [ ] 用 Mixtral/DeepSeek-V3 + 多卡 EP 验证 MoE 多卡显存合理

## 14. 实现期待办（设计期不瞎编，留到实现用真实数据做）
- 标定各引擎 `engines.py` 画像数值（基线 GB / 权重比例）
- 确定激活公式默认系数
- 验证 `model_info().safetensors` 的精确属性访问路径
- 人工填充精选基准：~30-50 个热门模型架构字段（含若干 MoE 的 `expert_params`）

## 15. 多机规划（v2.1）

回答"手头这几台服务器怎么部署某个模型"。四个新模块 + 一页一 API，复用 `estimate()` 做逐机评分。

**数据模型**（`core/cluster.py`）：`ServerSpec(id, name, host, gpus=[(gpu_id, count)])`——
gpus 是列表，混插机型天然可表达；`server_is_mixed()` 判定后**规划期跳过**（vLLM 不支持混型号 TP），
数据模型不留迁移坑。存储走泛化后的 `repos/store.py::EntityStore`（打包基准 ∪ 用户 `~/.vram_calc/servers/`）。

**规划器**（`core/planner.py`）：
- `enumerate_candidates` 枚举四类候选（类别序即网络惩罚升序）：
  **single** 单机独立成实例 / **dp** 同型号 ≥2 台各跑副本（吞吐 ×N，副本间零互联）/
  **pp** 跨机流水线 / **tp_cross** 跨机 TP（兜底，标慢）。TP 合法性 = 2 的幂 ∧ 整除注意力头数 ∧ ≤ 卡数。
- `_eval_candidate` 对每台机器跑一遍 `estimate()`：产 `LedgerRow` 逐机账本（权重/开销/激活/KV 池/verdict）、
  why 推理文案、FP8 硬件警告；MoE 模型 TP 即 EP 变体（`ep=tp`）。
- `plan_deployment` 同类别内取（verdict, KV 余量）最优，跨类别按（verdict, 是否跨机, KV 余量）排，
  返回 top3（全 over 则退回 ranked 前 3）。

**命令生成**（`core/commands.py`）：单机/DP 一段 `vllm serve`（DP 注明每台同命令）；
跨机方案三段——Ray head / worker join / serve（`--distributed-executor-backend ray`，
MoE 附 `--enable-expert-parallel`）。

**Web**：`POST /api/plan`（model_id + server_ids + 目标参数 → plans/warnings，逐机验混插与 GPU 在库）；
`/plan` 页 = 服务器清单（勾选参与）+ 规划目标 → 方案卡（topo/badges/账本表/命令块，debounce 800ms 自动重算）。

## 16. AI 助手（v2.2）

`vram_calc/assistant/` 包四模块，浏览器抽屉 + SSE 流式对话，回答部署问题并给出可回填的推荐参数表。

- **providers.py**：LLM 适配层。内部统一 OpenAI 消息/工具格式，`AnthropicProvider` 在边界翻译；
  `OpenAIProvider` 兼容 DeepSeek/通义/Kimi 及本地 vLLM/Ollama。`chat_stream` 产出 TextDelta/ToolCall/Done。
- **tools.py**：`calc_vram`（包 core.estimate）与 `plan_multi_node`（包 planner）的工具 schema + 执行，
  不走 HTTP；按页面上下文决定暴露哪个工具。
- **prompts.py**：system prompt = 契约（三段式回答 + [计算器]/[官方文档]/[经验] 来源标注 + 禁心算）
  + 当前页面配置预注入 + vLLM 手册精简摘要。
- **orchestrator.py**：`run_chat` 流式转发，工具调用 → 本地执行 → 结果回填续流（≤5 轮）；
  事件契约 delta/tool/tool_result/error/done，异常单边界翻译成 error。
- **路由**：`POST /api/assistant/chat`（SSE 流）与 `POST /api/assistant/test`（连接测试）。
  Key 由前端 localStorage 随请求体传入，仅在内存中转发，不落盘。
- **前端**：右下角浮球 + 抽屉（`static/assistant.js`），自动附带页面配置，推荐参数表带"应用"回填。
- **依赖**：新增 `openai>=1.40`、`anthropic>=0.34`。测试用 FakeProvider 替换，零真实网络。
