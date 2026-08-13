"""Core VRAM estimator.

total (per GPU) = weights + kv_cache + activation + overhead

  - weights : params * bytes/param, dense split by TP, experts by EP, both /PP
  - kv_cache: 2 * L * C * kv_heads * head_dim * kv_bytes * concurrency (/TP /PP)
  - activation: heuristic, scales with tokens in flight
  - overhead : engine baseline + ratio of resident weight

CPU offload keeps only (1 - cpu_offload) of layers on the GPU.
node count does NOT affect per-GPU VRAM (only total fleet size / throughput),
so it is intentionally absent from the calc.
"""

from __future__ import annotations

from dataclasses import dataclass

from .quant import bytes_per_param, bytes_per_kv
from .engines import get_engine

GB = 1e9  # decimal GB; VRAM is quoted in 10^9 bytes (24 GB card == 24e9)

# ponytail: activation is heuristic. Inference activations are small vs training;
# coeff ~ a handful of activation tensors per token. FlashAttention keeps it low.
ACTIVATION_COEFF = 12.0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    params_b: float              # total params (billions); incl. experts for MoE
    layers: int
    hidden_dim: int
    attn_heads: int
    kv_heads: int                # == attn_heads for plain MHA (non-GQA)
    head_dim: int
    vocab_size: int = 0
    num_experts: int = 0         # MoE only; 0 == dense
    expert_params_b: float = 0.0 # MoE: params living in experts (billions)
    quantizations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GpuSpec:
    id: str
    name: str
    vram_gb: float
    memory_bw_gbps: float = 0.0
    fp16_tflops: float = 0.0
    supports_fp8: bool = False
    supports_bf16: bool = True


@dataclass(frozen=True)
class EstimateInput:
    model: ModelSpec
    gpu: GpuSpec
    quant: str
    context_len: int
    concurrency: int = 1
    engine: str = "vllm"
    tp: int = 1
    pp: int = 1
    ep: int = 1
    kv_quant: str = "fp16"
    cpu_offload: float = 0.0     # fraction of layers offloaded to CPU (0..1)
    safety_factor: float = 0.9
    exl2_bpw: float = 4.0


@dataclass(frozen=True)
class Breakdown:
    weights: float
    kv_cache: float
    activation: float
    overhead: float

    @property
    def total(self) -> float:
        return self.weights + self.kv_cache + self.activation + self.overhead

    def as_dict(self) -> dict[str, float]:
        return {"weights": self.weights, "kv_cache": self.kv_cache,
                "activation": self.activation, "overhead": self.overhead}


@dataclass(frozen=True)
class Estimate:
    breakdown: Breakdown
    capacity_gb: float
    usable_gb: float           # capacity * safety_factor
    headroom_gb: float         # usable - total (negative == over capacity)
    verdict: str               # "ok" | "tight" | "over"
    num_gpus: int              # GPUs holding one model replica (tp*pp*ep_eff)


def _verdict(headroom: float, usable: float) -> str:
    if headroom < 0:
        return "over"
    if headroom < 0.10 * usable:   # < 10% headroom -> tight
        return "tight"
    return "ok"


def estimate(inp: EstimateInput) -> Estimate:
    m = inp.model
    bpp = bytes_per_param(inp.quant, inp.exl2_bpw)
    kvb = bytes_per_kv(inp.kv_quant)
    gpu_frac = 1.0 - inp.cpu_offload        # fraction resident on GPU

    # --- weights (GB) ---
    dense_params_b = m.params_b - m.expert_params_b
    ep_eff = inp.ep if m.num_experts else 1  # EP only meaningful for MoE
    dense_w = dense_params_b * bpp / (inp.tp * inp.pp)
    expert_w = m.expert_params_b * bpp / (ep_eff * inp.pp)
    weights = (dense_w + expert_w) * gpu_frac

    # --- KV cache (GB) ---
    # per-layer KV split across TP (KV heads / TP), stage holds L/PP layers
    kv_heads_per_gpu = m.kv_heads / inp.tp
    layers_per_gpu = m.layers / inp.pp
    # ponytail: assume tp divides kv_heads; fractional = average per GPU
    kv = (2 * layers_per_gpu * inp.context_len * kv_heads_per_gpu
          * m.head_dim * kvb * inp.concurrency) / GB * gpu_frac

    # --- activation (GB) ---
    activation = (inp.concurrency * inp.context_len * m.hidden_dim
                  * bpp * ACTIVATION_COEFF) / GB * gpu_frac

    # --- overhead ---
    eng = get_engine(inp.engine)
    overhead = eng.baseline_gb + (dense_w + expert_w) * gpu_frac * eng.weight_ratio

    bd = Breakdown(weights=weights, kv_cache=kv,
                   activation=activation, overhead=overhead)

    capacity = inp.gpu.vram_gb
    usable = capacity * inp.safety_factor
    headroom = usable - bd.total
    num_gpus = inp.tp * inp.pp * ep_eff
    return Estimate(breakdown=bd, capacity_gb=capacity, usable_gb=usable,
                    headroom_gb=headroom, verdict=_verdict(headroom, usable),
                    num_gpus=num_gpus)


if __name__ == "__main__":
    # quick eyeball check: Llama-3-8B FP16 on a single RTX 4090
    llama3_8b = ModelSpec(
        id="meta-llama/Meta-Llama-3-8B", name="Llama 3 8B",
        params_b=8.03, layers=32, hidden_dim=4096, attn_heads=32,
        kv_heads=8, head_dim=128, vocab_size=128256,
        quantizations=("fp16", "int8", "int4"))
    rtx4090 = GpuSpec(id="rtx-4090", name="RTX 4090", vram_gb=24)
    r = estimate(EstimateInput(model=llama3_8b, gpu=rtx4090, quant="fp16",
                               context_len=4096, concurrency=1, engine="vllm"))
    print(f"{llama3_8b.name} fp16 @ 4090, ctx=4096")
    print(f"  weights   {r.breakdown.weights:6.2f} GB")
    print(f"  kv_cache  {r.breakdown.kv_cache:6.2f} GB")
    print(f"  activation{r.breakdown.activation:6.2f} GB")
    print(f"  overhead  {r.breakdown.overhead:6.2f} GB")
    print(f"  TOTAL     {r.breakdown.total:6.2f} GB  (usable {r.usable_gb:.1f})")
    print(f"  verdict: {r.verdict}  headroom {r.headroom_gb:+.2f} GB")
