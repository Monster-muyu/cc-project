"""Inference-engine overhead profiles: (baseline GB, weight-ratio).

overhead = baseline_gb + on_gpu_weight_gb * weight_ratio
  - baseline_gb : fixed runtime cost (CUDA context, graphs, paged-attn buffers)
  - weight_ratio: overhead that scales with resident weight size

# ponytail: these coefficients are PLACEHOLDERS. Calibrate against real
# nvidia-smi readings in step 5. Upgrade path: per-engine empirical fit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineProfile:
    baseline_gb: float
    weight_ratio: float


ENGINES: dict[str, EngineProfile] = {
    "vllm": EngineProfile(1.5, 0.06),
    "sglang": EngineProfile(1.2, 0.05),
    "llama_cpp": EngineProfile(0.3, 0.02),
    "ollama": EngineProfile(0.8, 0.04),
}


def get_engine(name: str) -> EngineProfile:
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; known: {sorted(ENGINES)}")
    return ENGINES[name]
