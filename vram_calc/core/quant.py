"""Quantization -> bytes-per-parameter lookup.

GGUF / EXL2 use "bits per weight (bpw)"; bytes = bpw / 8.
INT4/INT8 carry a few % group-scale overhead.
"""

# bytes per model parameter by quantization format
QUANT_BYTES: dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.05,   # GPTQ/AWQ 8-bit + group-scale overhead
    "int4": 0.55,   # GPTQ/AWQ 4-bit + group overhead
    # GGUF: bpw values from the llama.cpp quant docs
    "gguf-q2_k": 2.6 / 8,
    "gguf-q3_k_m": 3.9 / 8,
    "gguf-q4_k_m": 4.5 / 8,
    "gguf-q5_k_m": 5.5 / 8,
    "gguf-q6_k": 6.6 / 8,
    "gguf-q8_0": 8.5 / 8,
}

# bytes per KV-cache element by KV precision
KV_BYTES: dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}

DEFAULT_EXL2_BPW = 4.0


def bytes_per_param(quant: str, exl2_bpw: float = DEFAULT_EXL2_BPW) -> float:
    """Bytes per model parameter for a quantization format."""
    if quant.startswith("exl2"):
        return exl2_bpw / 8.0
    if quant not in QUANT_BYTES:
        raise ValueError(f"unknown quant {quant!r}; known: {sorted(QUANT_BYTES)}")
    return QUANT_BYTES[quant]


def bytes_per_kv(kv_quant: str) -> float:
    if kv_quant not in KV_BYTES:
        raise ValueError(f"unknown kv_quant {kv_quant!r}; known: {sorted(KV_BYTES)}")
    return KV_BYTES[kv_quant]
