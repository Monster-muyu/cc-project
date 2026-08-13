"""Resolve an HF config.json dict into our architecture fields.

HF model configs use inconsistent key names across architectures; this maps
the common aliases in priority order. Parameter COUNT is NOT resolved here --
it comes from safetensors metadata (see repos.store.fetch_model_preview).
"""

from __future__ import annotations


def resolve_arch(config: dict) -> dict:
    def first(*keys, default=None):
        for k in keys:
            v = config.get(k)
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
    num_experts = first("num_local_experts", "num_experts", default=0) or 0
    intermediate = first("intermediate_size", default=0) or 0

    expert_params_b = 0.0
    if num_experts and intermediate and hidden and layers:
        # ponytail: rough estimate -- 3 projections (gate/up/down) per expert FFN
        # across all layers. For EP sizing only; total params still from safetensors.
        expert_params_b = num_experts * 3 * hidden * intermediate * layers / 1e9

    return {
        "layers": layers, "hidden_dim": hidden, "attn_heads": attn,
        "kv_heads": kv, "head_dim": head_dim, "vocab_size": vocab,
        "num_experts": num_experts, "expert_params_b": round(expert_params_b, 2),
    }
