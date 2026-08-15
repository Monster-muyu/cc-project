### Task 1: 设计文档同步到 v2.0

**Files:**
- Modify: `docs/design.md`(头部 + §5.2~5.4、新增 §5.8、§6、§7、§8 整节替换)

- [ ] **Step 1: 更新文档头部**

把 `docs/design.md` 开头的引用块替换为:

```markdown
# VRAM 显存计算工具 — 设计文档

> 版本: v2.0 · 状态: 已实现(与代码同步) · 初稿 2026-08-13 · 同步 2026-08-15
> v2.0 变更: KV 改 vLLM 分页池模型(动态预算/max_kv_tokens);verdict 改三档真实语义
> (OOM=权重放不下 / 能跑·会限流=负载超池子 / 放得下);参数新增 max_num_batched_tokens
> 与显存利用率(gpu_memory_utilization);模型源新增 ModelScope;UI 新增并发↔上下文推荐、
> quant 自动识别锁定、显卡数量自由输入、上下文单位(k/m);敏感度图改"KV 需求 vs KV 预算"。
```

- [ ] **Step 2: 替换 §5.2(KV cache)为 KV 池模型**

将"### 5.2 KV cache(推理大头)"整节(含公式代码块)替换为:

```markdown
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
```

- [ ] **Step 3: 替换 §5.3 激活**

将"### 5.3 激活(Activations)"整节替换为:

```markdown
### 5.3 激活(Transient prefill activation)
```
激活 ≈ max_num_batched_tokens × 隐藏维度 × 2字节 × 系数(12) ÷ (TP × PP)
```
- 由 vLLM 的 `max_num_batched_tokens`(chunked prefill 批量)驱动,**不是** 上下文×并发
- 激活值恒为 fp16(2 字节),与权重量化无关(计算精度仍是 fp16)
# ponytail: 系数 12 为经验值,标定见 scripts/calibrate.py
```

- [ ] **Step 4: §5.4 开销小节追加封顶说明**

在 §5.4 的公式代码块后追加一行:

```markdown
- 线性项**封顶** `cap_gb`:超大模型(如 DeepSeek-V3 661GB 权重)线性外推会失真(曾算出 41GB 开销),封顶后 vLLM ≤ 7.5GB
```

- [ ] **Step 5: 新增 §5.8 verdict 真实语义(插在 §5.7 之后)**

```markdown
### 5.8 结论(Verdict,vLLM 真实语义)
```
over  (OOM 放不下) : 权重+激活+开销 > 显存利用率×容量 → 加载即爆,与上下文无关
ok    (放得下)     : 固定部分放得下,且 上下文×并发 ≤ max_kv_tokens
tight (能跑·会限流): 固定部分放得下,但负载超池子 → 抢占/排队,变慢不崩
```
UI 同时给出 KV 池容量、并发↔上下文对照表(用户并发高亮)、三条推荐
(保并发/保上下文/保守推荐——按 80% 预算 + 向下取整,保证照着输放得下)。
```

- [ ] **Step 6: 替换 §6.1 拉取流程(双源)**

将 §6.1 中"**按需拉取**"条目替换为:

```markdown
**按需拉取**(添加模型/批量导入弹窗,来源可选 HuggingFace / ModelScope):
- HF:`model_info().safetensors.parameters`(参数量精确)+ `config.json`(架构)
- ModelScope(魔搭,国内直连):`GET modelscope.cn/api/v1/models/{id}/repo?FilePath=config.json`
  取架构;其 API 无参数量明细(仅 StorageSize,量化后失真)→ **架构估算参数量(实测误差 ~1%)**,
  预览框可手改。模型 ID 存为 `ms/{repo_id}` 前缀以区分来源
- 预量化仓库(AWQ/GPTQ)从 config 的 `quantization_config` 自动识别量化并**锁定**下拉框;
  兜底:从仓库名推断(`infer_quant_from_id`,AWQ-INT4→int4 等)
- VL/多模态模型架构嵌套在 `text_config` 下,别名解析器自动下钻(视觉编码器忽略,轻微低估)
- MoE 专家数别名含 `n_routed_experts`(DeepSeek),专家参数按 3×hidden×intermediate×层数估算
```

- [ ] **Step 7: 替换 §7.1/§7.2 控件与图表描述**

将 §7.1 的"推理参数"与"并行策略"两行替换为:

```markdown
- 推理参数:上下文长度(文本框,支持 32k/200k/1m 单位;=每请求**实际活跃上下文**,非最大窗口)、
  并发请求数、最大批处理(vLLM max_num_batched_tokens)
- 并行:推理引擎 ▼、**显卡数量(自由输入 1~128)**——自动选策略(dense→TP,MoE→EP,实时提示)、
  显卡利用率滑块(= gpu_memory_utilization)、KV 量化 ▼、CPU offload 滑块(llama.cpp 专属)、
  高级折叠内手动 TP/PP/EP(档位到 64)
```

将 §7.2 的"**图 3 · 敏感度曲线**"一行替换为:

```markdown
**图 3 · 敏感度曲线**:**KV 需求(上下文×并发×每token字节)vs KV 预算**——横轴扫并发,
橙线撞黄虚线(预算)处 = 最多可承载并发;调上下文/KV量化曲线实时变。带坐标轴/网格/刻度/图例。
另:KV 容量框(预算 GB + token 位 + 并发↔上下文表 + 三推荐);多卡时 headline 用合计口径
(可用=卡数×单卡,每卡分摊单独标注)。
```

- [ ] **Step 8: 替换 §8 的 /api/calc 响应与 /api/sweep**

将 §8 中 `/api/calc` 响应代码块替换为:

```json
{ "verdict": "ok|tight|over",
  "total_gb": 21.1, "per_gpu_gb": 10.55,
  "capacity_gb": 48, "usable_gb": 40.8, "headroom_gb": 19.7,
  "breakdown": {"weights":16.13,"kv_cache":0,"activation":1.01,"overhead":3.97},
  "num_gpus": 2, "max_kv_tokens": 150275, "kv_budget_gb": 19.7 }
```
说明追加一行:`total_gb/breakdown` 为**固定占用**(KV 动态,不在内),多卡为合计口径。

将 `/api/sweep` 响应代码块替换为:

```json
{ "points": [{"x":1,"total_gb":4.19}, ...],
  "capacity_gb": 22.1, "usable_gb": 21.7, "max_x": 5, "kv_budget_gb": 22.1 }
```
说明追加一行:`total_gb` 为 **KV 需求**(bpt×上下文×并发),容量线即 KV 预算。

- [ ] **Step 9: 提交**

```bash
git add docs/design.md
git commit -m "docs:设计文档同步v2.0(KV池模型/verdict真实语义/双源/UI现状)"
```

---


