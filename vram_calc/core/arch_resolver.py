"""Resolve an HF config.json dict into our architecture fields.

HF model configs use inconsistent key names across architectures; this maps
the common aliases in priority order. Parameter COUNT is NOT resolved here --
it comes from safetensors metadata (see repos.store.fetch_model_preview).
"""

from __future__ import annotations


def resolve_arch(config: dict) -> dict:
    # multimodal/VL models (Qwen-VL, LLaVA, ...) nest the LLM config under text_config.
    # ponytail: vision encoder weight is ignored -- small VRAM under-estimate for VL models.
    target = config
    if not (config.get("num_hidden_layers") or config.get("num_layers") or config.get("n_layer")):
        for nested_key in ("text_config", "language_config", "language_model_config"):
            nested = config.get(nested_key)
            if isinstance(nested, dict):
                target = nested
                break

    def first(*keys, default=None):
        for k in keys:
            v = target.get(k)
            if v is not None:
                return v
        return default

    layers = first("num_hidden_layers", "num_layers", "n_layer")
    hidden = first("hidden_size", "n_embd", "d_model")
    attn = first("num_attention_heads", "num_heads", "n_head")
    kv = first("num_key_value_heads", "num_kv_heads") or attn   # MHA fallback
    vocab = first("vocab_size", default=0) or 0
    head_dim = first("head_dim")
    if not head_dim and hidden and attn:
        head_dim = hidden // attn

    # GDN / linear-attention hybrids (Qwen3.8, Qwen3-Next, ...) list per-layer
    # attention types; only the full-attention layers hold a growing KV cache.
    layer_types = target.get("layer_types")
    kv_layers = 0
    if isinstance(layer_types, list) and layer_types:
        full = [t for t in layer_types if "attention" in t and "linear" not in t]
        if 0 < len(full) < len(layer_types):   # hybrid confirmed; pure attn -> 0 (== all)
            kv_layers = len(full)

    num_experts = first("num_local_experts", "num_experts", "n_routed_experts", default=0) or 0
    intermediate = first("moe_intermediate_size", "intermediate_size", default=0) or 0

    expert_params_b = 0.0
    if num_experts and intermediate and hidden and layers:
        # ponytail: rough estimate -- 3 projections (gate/up/down) per expert FFN
        # across all layers. For EP sizing only; total params still from safetensors.
        expert_params_b = num_experts * 3 * hidden * intermediate * layers / 1e9

    return {
        "layers": layers, "hidden_dim": hidden, "attn_heads": attn,
        "kv_heads": kv, "head_dim": head_dim, "vocab_size": vocab,
        "num_experts": num_experts, "expert_params_b": round(expert_params_b, 2),
        "kv_layers": kv_layers,
    }


def detect_quant(config: dict) -> str:
    """Detect fixed quantization from a pre-quantized repo's config.

    Returns "" for base (unquantized) models -> free choice of quant.
    AWQ/GPTQ carry a quantization_config block; GGUF repos expose it in filenames
    (handled at fetch time, not here).
    """
    qc = config.get("quantization_config") or {}
    method = qc.get("quant_method", "") if isinstance(qc, dict) else ""
    bits = qc.get("bits") if isinstance(qc, dict) else None
    if method in ("awq", "gptq", "bitsandbytes", "eetq"):
        if bits == 8:
            return "int8"
        return "int4"          # 4-bit is the overwhelmingly common case
    return ""
