"""Parse a real vLLM startup log into observed numbers, for engine-overhead calibration.

The estimator's overhead term (baseline + ratio) is a generic heuristic; a real
startup log tells us the TRUE weights / KV pool / occupancy on the user's own
hardware, so the gap becomes a measured override stored per GPU.

Log variants supported (V0 and V1 engines):
  Loading model weights took 8.0963 GiB
  # CUDA blocks: 947[, # CPU blocks: 2048]
  GPU KV cache size: 14,752.00 MiB          (per-GPU, V1)
  Maximum KV cache size can hold up to 151472 tokens   (V0)
  Maximum concurrency for 204800 tokens per request: 1.00x
"""
from __future__ import annotations

import re

RX = {
    "weights_gib": re.compile(r"Loading model weights took ([\d.]+)\s*GiB"),
    "cuda_blocks": re.compile(r"# CUDA blocks: (\d+)"),
    "kv_mib": re.compile(r"GPU KV cache size: ([\d,.]+)\s*MiB"),
    "kv_tokens": re.compile(r"can hold up to ([\d,]+) tokens"),
    "max_ctx": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request"),
}
BLOCK_SIZE = 16      # vLLM default; # CUDA blocks × 16 = pool tokens


def parse_vllm_log(text: str) -> dict:
    """Extract observed numbers; missing lines -> None. Last match wins."""
    out: dict = {"weights_gib": None, "kv_pool_tokens": None,
                 "kv_pool_gib_per_gpu": None, "max_ctx_tokens": None}
    for key, rx in RX.items():
        ms = rx.findall(text)
        if not ms:
            continue
        last = ms[-1].replace(",", "")
        if key == "weights_gib":
            out["weights_gib"] = float(last)
        elif key == "cuda_blocks":
            blocks = int(last)
            if out["kv_pool_tokens"] is None:
                out["kv_pool_tokens"] = blocks * BLOCK_SIZE
        elif key == "kv_mib":
            out["kv_pool_gib_per_gpu"] = round(float(last) / 1024, 2)
        elif key == "kv_tokens":
            out["kv_pool_tokens"] = int(last)
        elif key == "max_ctx":
            out["max_ctx_tokens"] = int(last)
    return out


def observed_overhead(parsed: dict, vram_gb: float, util: float,
                      activation_gb: float = 0.0) -> float | None:
    """真实引擎开销(含残余激活) = 利用率×显存 − 真实权重 − 真实KV池(−激活估计).

    KV 池需要 GiB 数字：优先用 'GPU KV cache size' 行；只有 tokens 时无法
    无模型信息换算，返回 None 由调用方（有模型上下文）处理。
    """
    kv = parsed.get("kv_pool_gib_per_gpu")
    w = parsed.get("weights_gib")
    if kv is None or w is None:
        return None
    return round(util * vram_gb - w - kv - activation_gb, 2)
