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
def test_gdn_hybrid_kv_layers():
    # Qwen3.8-27B-style GDN hybrid: 64 layers but only 16 hold KV ->
    # bytes/token and requested-load KV both 4x smaller than pure attention.
    gdn = ModelSpec(id="gdn", name="gdn", params_b=27.78, layers=64, hidden_dim=5120,
                    attn_heads=40, kv_heads=4, head_dim=128, kv_layers=16)
    pure = ModelSpec(id="pure", name="pure", params_b=27.78, layers=64, hidden_dim=5120,
                     attn_heads=40, kv_heads=4, head_dim=128)
    r_gdn = estimate(_inp(gdn, context_len=200_000, concurrency=2))
    r_pure = estimate(_inp(pure, context_len=200_000, concurrency=2))
    assert r_gdn.breakdown.kv_cache == pytest.approx(r_pure.breakdown.kv_cache / 4)
    assert r_gdn.max_kv_tokens == pytest.approx(r_pure.max_kv_tokens * 4)
    # user's manual math: 16 layers * 2 * 4 heads * 128 dim * 2B = 32,768 B/token;
    # 200k ctx * 2 conc = 400k tokens -> 400_000 * 32768 / 1e9 = 13.1 GB
    assert r_gdn.breakdown.kv_cache == pytest.approx(13.1, rel=0.02)


def test_kv_layers_zero_means_all():
    # kv_layers=0 or == layers must behave identically to omitting the field
    m0 = ModelSpec(id="m0", name="m0", params_b=1, layers=8, hidden_dim=256,
                   attn_heads=4, kv_heads=2, head_dim=64, kv_layers=0)
    m8 = ModelSpec(id="m8", name="m8", params_b=1, layers=8, hidden_dim=256,
                   attn_heads=4, kv_heads=2, head_dim=64, kv_layers=8)
    assert estimate(_inp(m0)).max_kv_tokens == estimate(_inp(m8)).max_kv_tokens


def test_kv_pool_dynamic():
    # pool capacity is computed and context-INDEPENDENT (vLLM fills util at startup)
    r = estimate(_inp(LLAMA3_8B))
    assert r.max_kv_tokens > 10000
    r200 = estimate(_inp(LLAMA3_8B, context_len=200000))
    assert r200.max_kv_tokens == r.max_kv_tokens
    # but the displayed KV (requested load) grows with context -> total responds
    assert r200.breakdown.kv_cache > r.breakdown.kv_cache
    assert r200.breakdown.total > r.breakdown.total
    # a 200k load over a ~24k pool is "tight" (preemption), NOT "over" (OOM)
    assert r200.verdict == "tight"


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


def test_overhead_capped_for_huge_models():
    # DeepSeek-V3 661GB 权重:线性 0.06 会算出 ~40GB 开销(失真),封顶后 ≤ baseline+cap
    ds = ModelSpec(id="dsv3", name="DeepSeek-V3", params_b=671, layers=61, hidden_dim=7168,
                   attn_heads=128, kv_heads=128, head_dim=128,
                   num_experts=256, expert_params_b=660.0)
    h100 = GpuSpec(id="h100", name="H100 80G", vram_gb=80)
    r = estimate(EstimateInput(model=ds, gpu=h100, quant="fp8", context_len=4096))
    assert r.breakdown.overhead <= 1.5 + 6.0 + 0.01    # vllm baseline 1.5 + cap 6.0
