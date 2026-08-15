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


