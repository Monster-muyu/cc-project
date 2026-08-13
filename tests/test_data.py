"""Tests for arch resolver + entity stores (offline). Run: pytest."""

import pytest

from vram_calc.core.arch_resolver import resolve_arch
from vram_calc.repos.store import (
    EntityStore, list_models, get_model, list_gpus, get_gpu,
)


# ---- arch resolver alias mapping ----
def test_resolve_arch_standard_keys():
    cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
           "num_key_value_heads": 8, "vocab_size": 128256}
    a = resolve_arch(cfg)
    assert a["layers"] == 32
    assert a["hidden_dim"] == 4096
    assert a["kv_heads"] == 8                 # GQA
    assert a["head_dim"] == 128              # 4096 / 32


def test_resolve_arch_gpt_keys_and_mha_fallback():
    cfg = {"n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257}
    a = resolve_arch(cfg)
    assert a["layers"] == 12
    assert a["hidden_dim"] == 768
    assert a["kv_heads"] == 12               # MHA: no kv field -> fallback to attn heads


def test_resolve_arch_explicit_head_dim():
    cfg = {"num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 4,
           "head_dim": 96}
    assert resolve_arch(cfg)["head_dim"] == 96


def test_resolve_arch_nested_text_config():
    # VL/multimodal: LLM arch nested under text_config (top level lacks it)
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"],
           "text_config": {"num_hidden_layers": 28, "hidden_size": 3584,
                           "num_attention_heads": 28, "num_key_value_heads": 4,
                           "head_dim": 128, "vocab_size": 152064},
           "vision_config": {"some": "stuff"}}
    a = resolve_arch(cfg)
    assert a["layers"] == 28
    assert a["hidden_dim"] == 3584
    assert a["kv_heads"] == 4
    assert a["head_dim"] == 128


def test_resolve_arch_moe_expert_estimate():
    # Mixtral-ish: 8 experts across 32 layers
    cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
           "num_key_value_heads": 8, "vocab_size": 32000,
           "num_local_experts": 8, "intermediate_size": 14336}
    a = resolve_arch(cfg)
    assert a["num_experts"] == 8
    assert a["expert_params_b"] > 40          # ~45B experts


# ---- entity store merge + override ----
def test_store_bundled_loads():
    store = EntityStore("models.json", "models")
    bundled = store._load_bundled()
    ids = list(bundled.keys())
    assert "meta-llama/Meta-Llama-3-8B" in ids
    assert "deepseek-ai/DeepSeek-V3" in ids


def test_store_merge_and_user_override(tmp_path):
    store = EntityStore("models.json", "models")
    store.user_dir = tmp_path / "models"      # inject temp user dir

    # user adds a custom model
    store.save({"id": "custom/x", "name": "X", "params_b": 1, "layers": 2,
                "hidden_dim": 8, "attn_heads": 2, "kv_heads": 2, "head_dim": 4})
    ids = [e["id"] for e in store.list()]
    assert "custom/x" in ids
    assert "meta-llama/Meta-Llama-3-8B" in ids   # bundled still present

    # user overrides a bundled id -> user wins
    store.save({"id": "meta-llama/Meta-Llama-3-8B", "name": "OVERRIDE",
                "params_b": 1, "layers": 2, "hidden_dim": 8,
                "attn_heads": 2, "kv_heads": 2, "head_dim": 4})
    assert store.get("meta-llama/Meta-Llama-3-8B")["name"] == "OVERRIDE"


def test_list_and_get_models():
    ms = list_models()
    assert len(ms) >= 8
    m = get_model("meta-llama/Meta-Llama-3-8B")
    assert m is not None and m.layers == 32 and m.kv_heads == 8
    assert get_model("nope/none") is None


def test_list_and_get_gpus():
    gs = list_gpus()
    assert len(gs) >= 10
    g = get_gpu("rtx-4090")
    assert g is not None and g.vram_gb == 24
    assert get_gpu("nope") is None
