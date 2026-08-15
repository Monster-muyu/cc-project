# Task 2 Report: 引擎开销封顶(修超大模型线性失真)

## What was implemented

Added `cap_gb` field to `EngineProfile` and applied `min(linear_term, cap_gb)` in the estimator overhead calculation. This prevents the linear overhead formula from producing absurd values for huge models (e.g. DeepSeek-V3: 41.76GB overhead -> capped to 7.5GB).

## Files changed

- `vram_calc/core/engines.py` — Added `cap_gb` field (default 4.0) to `EngineProfile`; each engine gets a specific cap (vllm=6.0, sglang=5.0, llama_cpp=1.5, ollama=3.0)
- `vram_calc/core/estimator.py` — One-line change: wrapped linear overhead term in `min(..., eng.cap_gb)`
- `tests/test_estimator.py` — Added `test_overhead_capped_for_huge_models` for DeepSeek-V3 scenario

## TDD Evidence

### RED

```
$ & "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest tests/test_estimator.py::test_overhead_capped_for_huge_models -q
F
FAILED tests/test_estimator.py::test_overhead_capped_for_huge_models
  AssertionError: assert 41.76 <= 7.51
```

Overhead was 41.76GB (linear 0.06 * 661GB weights) — clearly absurd.

### GREEN

```
$ & "E:\ANACONDA\envs\vram-calc\python.exe" -m pytest -q
.......................
23 passed in 0.15s
```

## Self-review notes

- Diff is minimal: 3 files, +23/-9 lines. Exactly as prescribed in the brief.
- All 22 existing tests still pass (no regression).
- The `min()` clamp is the simplest possible fix — ponytail-approved.
- `cap_gb` defaults to 4.0 so any future engine entry is safe even without specifying it.
