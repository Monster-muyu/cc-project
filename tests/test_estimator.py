"""Tests for the core VRAM estimator. Run: pytest."""

import pytest

from vram_calc.core import (
    ModelSpec, GpuSpec, EstimateInput, estimate, bytes_per_param,
)

RTX4090 = GpuSpec(id="rtx-4090", name="RTX 4090", vram_gb=24)

LLAMA3_8B = ModelSpec(
    id="meta-llama/Meta-Llama-3-8B", name="Llama 3 8B",
    params_b=8.03, layers=32, hidden_dim=4096, attn_heads=32,
    kv_heads=8, head_dim=128, vocab_size=128256,
)

# Mixtral 8x7B-like MoE: ~46.7B total, experts dominate (~39B)
MIXTRAL = ModelSpec(
    id="mistralai/Mixtral-8x7B-v0.1", name="Mixtral 8x7B",
    params_b=46.7, layers=32, hidden_dim=4096, attn_heads=32,
    kv_heads=8, head_dim=128, num_experts=8, expert_params_b=39.0,
)


def _inp(model, gpu=RTX4090, **kw):
    base = dict(quant="fp16", context_len=4096, concurrency=1)
    base.update(kw)
    return EstimateInput(model=model, gpu=gpu, **base)


# ---- quantization lookup ----
def test_quant_table():
    assert bytes_per_param("fp16") == 2.0
    assert bytes_per_param("int4") == 0.55
    assert bytes_per_param("gguf-q4_k_m") == pytest.approx(4.5 / 8)
    assert bytes_per_param("exl2", exl2_bpw=4.5) == pytest.approx(4.5 / 8)


def test_unknown_quant_raises():
    with pytest.raises(ValueError):
        bytes_per_param("nonsense")


# ---- weights: pure arithmetic ----
def test_weights_fp16_arithmetic():
    # 8.03B params * 2 bytes = 16.06 GB (deterministic, "ground truth")
    r = estimate(_inp(LLAMA3_8B))
    assert r.breakdown.weights == pytest.approx(8.03 * 2, rel=1e-6)


def test_weights_scale_with_quant():
    fp16 = estimate(_inp(LLAMA3_8B, quant="fp16")).breakdown.weights
    int4 = estimate(_inp(LLAMA3_8B, quant="int4")).breakdown.weights
    assert int4 == pytest.approx(fp16 * (0.55 / 2.0))


# ---- KV cache ----
def test_kv_cache_value():
    # KV scales with context: 2*32*4096*8*128*2 bytes ~= 0.537 GB at ctx=4096
    r = estimate(_inp(LLAMA3_8B))
    assert r.breakdown.kv_cache == pytest.approx(0.537, rel=0.01)
    # and it grows with context (the whole point of a VRAM calculator)
    r32 = estimate(_inp(LLAMA3_8B, context_len=32768))
    assert r32.breakdown.kv_cache == pytest.approx(r.breakdown.kv_cache * 8, rel=1e-6)


def test_gqa_vs_mha_kv_capacity():
    # GQA kv_heads=8 vs MHA kv_heads=32: MHA uses 4x bytes/token -> ~1/4 the capacity
    mha = ModelSpec(id="m", name="m", params_b=8.0, layers=32, hidden_dim=4096,
                    attn_heads=32, kv_heads=32, head_dim=128)
    gqa = estimate(_inp(LLAMA3_8B)).max_kv_tokens
    mha_t = estimate(_inp(mha)).max_kv_tokens
    assert mha_t < gqa                              # MHA (more kv heads) -> less KV capacity
    assert 3.8 < gqa / mha_t < 4.2                 # 4x kv heads -> ~1/4 capacity


# ---- tensor / pipeline parallel ----
def test_tp_halves_weights_and_kv():
    e1 = estimate(_inp(LLAMA3_8B, tp=1))
    e2 = estimate(_inp(LLAMA3_8B, tp=2))
    assert e2.breakdown.weights == pytest.approx(e1.breakdown.weights / 2)
    assert e2.breakdown.kv_cache == pytest.approx(e1.breakdown.kv_cache / 2)


def test_pp_divides_weights_and_kv():
    e1 = estimate(_inp(LLAMA3_8B, pp=1))
    e2 = estimate(_inp(LLAMA3_8B, pp=2))
    assert e2.breakdown.weights == pytest.approx(e1.breakdown.weights / 2)
    assert e2.breakdown.kv_cache == pytest.approx(e1.breakdown.kv_cache / 2)


# ---- expert parallel (MoE) ----
def test_ep_only_splits_experts():
    dense_w = (MIXTRAL.params_b - MIXTRAL.expert_params_b) * 2  # 15.4 GB
    expert_total = MIXTRAL.expert_params_b * 2                  # 78 GB
    for ep in (1, 2, 4, 8):
        r = estimate(_inp(MIXTRAL, ep=ep))
        expected = dense_w + expert_total / ep
        assert r.breakdown.weights == pytest.approx(expected, rel=1e-6)


def test_ep_ignored_for_dense():
    e1 = estimate(_inp(LLAMA3_8B, ep=1))
    e4 = estimate(_inp(LLAMA3_8B, ep=4))   # dense: EP must not change anything
    assert e4.breakdown.weights == pytest.approx(e1.breakdown.weights)
    assert e4.num_gpus == e1.num_gpus       # ep_eff stays 1 for dense


# ---- CPU offload ----
def test_cpu_offload_scales_resident():
    full = estimate(_inp(LLAMA3_8B, cpu_offload=0.0))
    half = estimate(_inp(LLAMA3_8B, cpu_offload=0.5))
    assert half.breakdown.weights == pytest.approx(full.breakdown.weights * 0.5)
    assert half.breakdown.kv_cache == pytest.approx(full.breakdown.kv_cache * 0.5)


# ---- verdict / fit ----
def test_llama3_8b_fits_4090():
    r = estimate(_inp(LLAMA3_8B, context_len=4096))
    assert r.breakdown.total < RTX4090.vram_gb
    assert r.headroom_gb > 0
    assert r.verdict != "over"


def test_oversize_is_over():
    big = ModelSpec(id="big", name="big", params_b=40, layers=80, hidden_dim=8192,
                    attn_heads=64, kv_heads=8, head_dim=128)
    r = estimate(_inp(big, context_len=4096))
    assert r.verdict == "over"
    assert r.headroom_gb < 0
