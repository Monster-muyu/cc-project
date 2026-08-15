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
