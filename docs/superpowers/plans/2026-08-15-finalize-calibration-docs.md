# 收尾:文档同步 + 引擎开销封顶 + 标定脚本 + README 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把设计文档同步到已实现的 v2.0 状态,修掉超大模型 overhead 线性失真,落地标定对账脚本和 README。

**Architecture:** 纯文档 + 一个核心小改(engines.py 开销封顶)+ 一个独立脚本(scripts/calibrate.py)。核心估算器 KV 池模型、双源拉取、推荐系统已实现并有 22 项测试,本计划不动它们的语义。

**Tech Stack:** Python 3.11 (conda env `vram-calc`,python.exe 在 `E:\ANACONDA\envs\vram-calc\python.exe`)、pytest、纯 stdlib 脚本。

**环境注意(所有命令通用):**
- 跑 Python 一律用 `"E:\ANACONDA\envs\vram-calc\python.exe"`(下文简写 `$py`)
- `pip install -e .` 必须带 `--no-build-isolation`
- 长脚本用 `Get-Content <file> | & $py -` (stdin 方式)或直接 `& $py <file>`,**不要**用 `conda run -c`(不支持多行)
- 服务器重启:`Stop-Process` 杀 8000 端口进程后 `& $py -m uvicorn vram_calc.web.app:app --host 127.0.0.1 --port 8000`
- 前端改动要 bump `index.html` 里 `?v=N` 并让用户硬刷新

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `docs/design.md` | 修改 | 设计文档同步 v2.0(§头/§5/§6/§7/§8) |
| `vram_calc/core/engines.py` | 修改 | EngineProfile 加 `cap_gb`,防超大模型线性失真 |
| `vram_calc/core/estimator.py` | 修改(1行) | overhead 取 `min(线性项, cap)` |
| `tests/test_estimator.py` | 修改 | 加封顶测试 |
| `scripts/calibrate.py` | 新建 | 估算 vs 参考值对账表(含用户实测入口) |
| `README.md` | 新建 | 环境搭建/启动/功能/GitLab |

---

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

### Task 2: 引擎开销封顶(修超大模型线性失真)

**Files:**
- Modify: `vram_calc/core/engines.py`(EngineProfile 加 cap_gb)
- Modify: `vram_calc/core/estimator.py`(overhead 一行)
- Test: `tests/test_estimator.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_estimator.py` 末尾追加:

```python
def test_overhead_capped_for_huge_models():
    # DeepSeek-V3 661GB 权重:线性 0.06 会算出 ~40GB 开销(失真),封顶后 ≤ baseline+cap
    ds = ModelSpec(id="dsv3", name="DeepSeek-V3", params_b=671, layers=61, hidden_dim=7168,
                   attn_heads=128, kv_heads=128, head_dim=128,
                   num_experts=256, expert_params_b=660.0)
    h100 = GpuSpec(id="h100", name="H100 80G", vram_gb=80)
    r = estimate(EstimateInput(model=ds, gpu=h100, quant="fp8", context_len=4096))
    assert r.breakdown.overhead <= 1.5 + 6.0 + 0.01    # vllm baseline 1.5 + cap 6.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest tests/test_estimator.py::test_overhead_capped_for_huge_models -q`
Expected: FAIL(overhead ≈ 41 > 7.51)

- [ ] **Step 3: 改 engines.py**

整文件替换为:

```python
"""Inference-engine overhead profiles: (baseline GB, weight-ratio, cap GB).

overhead = baseline_gb + min(on_gpu_weight_gb * weight_ratio, cap_gb)
  - baseline_gb : fixed runtime cost (CUDA context, graphs, paged-attn buffers)
  - weight_ratio: overhead that scales with resident weight size
  - cap_gb      : linear term breaks down on huge models (661GB weights -> 40GB
                  overhead was absurd); real engines don't scale that far.

# ponytail: coefficients are placeholders. Calibrate with scripts/calibrate.py
# and your vLLM startup log ("# CUDA blocks: N" -> pool = N * 16 tokens).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineProfile:
    baseline_gb: float
    weight_ratio: float
    cap_gb: float = 4.0


ENGINES: dict[str, EngineProfile] = {
    "vllm": EngineProfile(1.5, 0.06, 6.0),
    "sglang": EngineProfile(1.2, 0.05, 5.0),
    "llama_cpp": EngineProfile(0.3, 0.02, 1.5),
    "ollama": EngineProfile(0.8, 0.04, 3.0),
}


def get_engine(name: str) -> EngineProfile:
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; known: {sorted(ENGINES)}")
    return ENGINES[name]
```

- [ ] **Step 4: 改 estimator.py 的 overhead 行**

将:
```python
    overhead = eng.baseline_gb + (dense_w + expert_w) * gpu_frac * eng.weight_ratio
```
替换为:
```python
    overhead = eng.baseline_gb + min(
        (dense_w + expert_w) * gpu_frac * eng.weight_ratio, eng.cap_gb)
```

- [ ] **Step 5: 跑全量测试**

Run: `& "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest -q`
Expected: 23 passed

- [ ] **Step 6: 提交**

```bash
git add vram_calc/core/engines.py vram_calc/core/estimator.py tests/test_estimator.py
git commit -m "fix:引擎开销线性项封顶cap_gb(修DeepSeek-V3级超大模型41GB开销失真)"
```

---

### Task 3: 标定对账脚本 scripts/calibrate.py

**Files:**
- Create: `scripts/calibrate.py`

- [ ] **Step 1: 写脚本**

创建 `scripts/calibrate.py`,内容如下(自包含,退出码 0=全部达标):

```python
"""Cross-validate the estimator against reference points.

Usage:
  python scripts/calibrate.py                          # built-in public references
  python scripts/calibrate.py --real-kv-tokens 152000  # + your real vLLM pool
      (real value = "CUDA blocks" from vLLM startup log x 16)

Exit 0 = all cases within threshold; exit 1 = at least one out of range.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vram_calc.core import ModelSpec, GpuSpec, EstimateInput, estimate  # noqa: E402

RTX4090 = GpuSpec(id="rtx-4090", name="RTX 4090", vram_gb=24)
RTX3090 = GpuSpec(id="rtx-3090", name="RTX 3090", vram_gb=24)
H100 = GpuSpec(id="h100", name="H100 80G", vram_gb=80)

LLAMA3_8B = ModelSpec(id="meta-llama/Meta-Llama-3-8B", name="Llama 3 8B",
                      params_b=8.03, layers=32, hidden_dim=4096, attn_heads=32,
                      kv_heads=8, head_dim=128, vocab_size=128256)
MIXTRAL = ModelSpec(id="mistralai/Mixtral-8x7B-Instruct-v0.1", name="Mixtral 8x7B",
                    params_b=46.7, layers=32, hidden_dim=4096, attn_heads=32,
                    kv_heads=8, head_dim=128, num_experts=8, expert_params_b=45.0)
DSV3 = ModelSpec(id="deepseek-ai/DeepSeek-V3", name="DeepSeek-V3", params_b=671,
                 layers=61, hidden_dim=7168, attn_heads=128, kv_heads=128,
                 head_dim=128, num_experts=256, expert_params_b=660.0)
# 用户真实部署的模型(cyankiwi/Qwen3.6-27B-AWQ-INT4,HF 实测参数)
QWEN36_AWQ = ModelSpec(id="cyankiwi/Qwen3.6-27B-AWQ-INT4", name="Qwen3.6-27B-AWQ-INT4",
                       params_b=29.325, layers=64, hidden_dim=5120, attn_heads=24,
                       kv_heads=4, head_dim=256, quant="int4")


def build_cases(real_kv_tokens):
    """(标签, 预测值, 参考值, 容差%, 单位)"""
    cases = []
    # 1. 权重:纯算术,0 容差
    r = estimate(EstimateInput(model=LLAMA3_8B, gpu=RTX4090, quant="fp16", context_len=4096))
    cases.append(("Llama3-8B fp16 权重", r.breakdown.weights, 8.03 * 2, 0.1, "GB"))
    # 2. Llama3-8B fp16 @4090 固定占用 vs 社区实测 19~21(取中值 20.0)
    cases.append(("Llama3-8B fp16 @4090 固定占用", r.breakdown.total, 20.0, 10.0, "GB"))
    # 3. Mixtral fp16 权重合计 = 46.7*2(精确)
    r = estimate(EstimateInput(model=MIXTRAL, gpu=H100, quant="fp16", context_len=4096))
    cases.append(("Mixtral 8x7B fp16 权重", r.breakdown.weights, 93.4, 0.5, "GB"))
    # 4. DeepSeek-V3 fp8 开销封顶 sanity(真实 vLLM 远到不了 40GB)
    r = estimate(EstimateInput(model=DSV3, gpu=H100, quant="fp8", context_len=4096))
    cases.append(("DeepSeek-V3 fp8 开销(封顶)", r.breakdown.overhead, 7.5, 15.0, "GB"))
    # 5. 用户真实部署:2x3090 TP2 util0.85 kv fp8 的 KV 池容量 vs 启动日志实测
    if real_kv_tokens:
        r = estimate(EstimateInput(model=QWEN36_AWQ, gpu=RTX3090, quant="int4",
                                   context_len=32768, concurrency=4, engine="vllm",
                                   tp=2, kv_quant="fp8", safety_factor=0.85,
                                   max_num_batched_tokens=8192))
        cases.append(("Qwen3.6-27B-AWQ 2x3090 KV池", r.max_kv_tokens,
                      float(real_kv_tokens), 10.0, "tokens"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-kv-tokens", type=int, default=None,
                    help="vLLM 启动日志 '# CUDA blocks' x 16 的实测值")
    args = ap.parse_args()

    cases = build_cases(args.real_kv_tokens)
    print(f"{'用例':38} {'预测':>12} {'参考':>12} {'误差':>8}  判定")
    print("-" * 78)
    failures = 0
    for tag, pred, ref, tol, unit in cases:
        err = abs(pred - ref) / ref * 100
        ok = err <= tol
        failures += 0 if ok else 1
        print(f"{tag:38} {pred:>12,.1f} {ref:>12,.1f} {err:>7.1f}%  {'✅' if ok else '❌'} ({unit}, 容差{tol}%)")
    print("-" * 78)
    print("全部达标 ✅" if failures == 0 else f"{failures} 项超容差 ❌")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑脚本验证**

Run: `& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py`
Expected: 4 行用例,权重/开销类全 ✅,退出码 0。
(Llama3-8B 固定占用若 ❌ 超 10%,记录误差值——这正是待用真实数据调 ACTIVATION_COEFF/引擎系数的输入,不要瞎调)

- [ ] **Step 3: 带实测参数复验(拿到用户 CUDA blocks 后)**

Run: `& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py --real-kv-tokens 152000`
(152000 为示例;用真实值。Expected: 第 5 行 ✅ 误差 ≤10%)

- [ ] **Step 4: 提交**

```bash
git add scripts/calibrate.py
git commit -m "feat:标定对账脚本(公开参考+用户实测KV池入口,超容差退出码1)"
```

---

### Task 4: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 README**

```markdown
# VRAM 显存计算工具

部署 LLM 前估算显存:选模型 + 量化 + 显卡 + 并行/并发参数 → 能不能跑、最多扛几路并发、KV 池容量多少。

仓库: http://192.168.243.204/root/vram-calc · 设计文档: `docs/design.md` · 测试: 23 项

## 环境搭建(Windows + Anaconda)

```powershell
& "E:\ANACONDA\Scripts\conda.exe" create -n vram-calc python=3.11 -y
& "E:\ANACONDA\envs\vram-calc\python.exe" -m pip install -e . --no-build-isolation
& "E:\ANACONDA\envs\vram-calc\python.exe" -m pip install pytest
```

> `--no-build-isolation` 必带:构建隔离的临时 venv 在本机拉不到 setuptools。

## 启动

```powershell
& "E:\ANACONDA\envs\vram-calc\python.exe" -m uvicorn vram_calc.web.app:app --host 127.0.0.1 --port 8000
```
浏览器开 http://127.0.0.1:8000/ (改前端后 Ctrl+Shift+R 硬刷新)。

## 测试与标定

```powershell
& "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest -q
& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py                 # 公开参考对账
& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py --real-kv-tokens 152000  # + 实测
```

## 功能速览
- **计算模型**:vLLM 分页 KV 池(显存利用率扣权重→池容量→max_kv_tokens);
  verdict 三档 = OOM 放不下 / 能跑·会限流 / 放得下
- **模型库**:精选 23+;添加/批量导入支持 **HuggingFace(参数量精确)与 ModelScope(国内直连)**
- **量化**:FP16/FP8/INT4/INT8/GGUF/EXL2;AWQ/GPTQ 预量化仓库自动识别并锁定
- **并行**:显卡数量自由输入(dense 自动 TP,MoE 自动 EP),高级手动 TP/PP/EP 到 64
- **推荐**:并发↔上下文对照表 + 保并发/保上下文/保守推荐(照着输保证放得下)

## 已知简化(升级路径见代码内 ponytail 注释)
- 激活系数、引擎开销画像为经验值 → `scripts/calibrate.py` 对账后调整
- DeepSeek MLA 的 KV 为近似;VL 模型忽略视觉编码器(轻微低估)
- 异构多卡、训练/LoRA、tokens/s 不做(v1 范围外)
- vLLM 部分版本在 max_model_len > 单序列池容量时会拒绝启动(工具显示"能跑·会限流"指运行时不 OOM)
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs:README(环境搭建/启动/测试标定/功能/已知简化)"
```

---

### Task 5: 收尾验证与推送

- [ ] **Step 1: 全量测试**

Run: `& "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest -q`
Expected: 23 passed

- [ ] **Step 2: 标定脚本**

Run: `& "E:\ANACONDA\envs\vram-calc\python.exe" scripts\calibrate.py`
Expected: 退出码 0,权重/开销用例 ✅

- [ ] **Step 3: 服务冒烟**

重启 uvicorn 后 `Invoke-WebRequest http://127.0.0.1:8000/api/gpus -UseBasicParsing` → HTTP 200

- [ ] **Step 4: 推送 GitLab**

```bash
git push origin main
```

---

## Self-Review 记录

- 覆盖:设计文档同步✅ 开销封顶✅ 标定✅ README✅(用户反馈的 5 条已在此前迭代闭环,不在本计划)
- 无占位符;`--real-kv-tokens` 是可选 CLI 参数而非 TODO
- 类型一致:EngineProfile(baseline_gb, weight_ratio, cap_gb) 三字段在 engines.py/estimator.py/测试中一致;
  calibrate.py 只用 core 公开五元组,无新类型
- 注意:Task 2 改核心后需重启 uvicorn 才反映到 Web;Task 1 纯文档无需重启
