"""Cross-validate the estimator against reference points.

Usage:
  python scripts/calibrate.py                          # built-in public references
  python scripts/calibrate.py --real-kv-tokens 152000  # + your real vLLM pool
      (real value = "CUDA blocks" from vLLM startup log x 16)

Exit 0 = all cases within threshold; exit 1 = at least one out of range.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vram_calc.core import ModelSpec, GpuSpec, EstimateInput, estimate  # noqa: E402

RTX4090 = GpuSpec(id="rtx-4090", name="RTX 4090", vram_gb=24)
RTX3090 = GpuSpec(id="rtx-3090", name="RTX 3090", vram_gb=24)
H100 = GpuSpec(id="h100", name="H100 80G", vram_gb=80)

LLAMA3_8B = ModelSpec(id="meta-llama/Meta-Llama-3-8B", name="Llama 3 8B",
                      params_b=8.03, layers=32, hidden_dim=4096, attn_heads=32,
                      kv_heads=8, head_dim=128, vocab_size=128256)
MIXTRAL = ModelSpec(id="mistralai/Mixtral-8x7B-Instruct-v0.1", name="Mixtral 8x7B",
                    params_b=46.7, layers=32, hidden_dim=4096, attn_heads=32,
                    kv_heads=8, head_dim=128, num_experts=8, expert_params_b=45.0)
DSV3 = ModelSpec(id="deepseek-ai/DeepSeek-V3", name="DeepSeek-V3", params_b=671,
                 layers=61, hidden_dim=7168, attn_heads=128, kv_heads=128,
                 head_dim=128, num_experts=256, expert_params_b=660.0)
# 用户真实部署的模型(cyankiwi/Qwen3.6-27B-AWQ-INT4,HF 实测参数)
QWEN36_AWQ = ModelSpec(id="cyankiwi/Qwen3.6-27B-AWQ-INT4", name="Qwen3.6-27B-AWQ-INT4",
                       params_b=29.325, layers=64, hidden_dim=5120, attn_heads=24,
                       kv_heads=4, head_dim=256, quant="int4")


def build_cases(real_kv_tokens):
    """(标签, 预测值, 参考值, 容差%, 单位)"""
    cases = []
    # 1. 权重:纯算术,0 容差
    r = estimate(EstimateInput(model=LLAMA3_8B, gpu=RTX4090, quant="fp16", context_len=4096))
    cases.append(("Llama3-8B fp16 权重", r.breakdown.weights, 8.03 * 2, 0.1, "GB"))
    # 2. Llama3-8B fp16 @4090 固定占用 vs 社区实测 19~21(取中值 20.0)
    cases.append(("Llama3-8B fp16 @4090 固定占用", r.breakdown.total, 20.0, 10.0, "GB"))
    # 3. Mixtral fp16 权重合计 = 46.7*2(精确)
    r = estimate(EstimateInput(model=MIXTRAL, gpu=H100, quant="fp16", context_len=4096))
    cases.append(("Mixtral 8x7B fp16 权重", r.breakdown.weights, 93.4, 0.5, "GB"))
    # 4. DeepSeek-V3 fp8 开销封顶 sanity(真实 vLLM 远到不了 40GB)
    r = estimate(EstimateInput(model=DSV3, gpu=H100, quant="fp8", context_len=4096))
    cases.append(("DeepSeek-V3 fp8 开销(封顶)", r.breakdown.overhead, 7.5, 15.0, "GB"))
    # 5. 用户真实部署:2x3090 TP2 util0.85 kv fp8 的 KV 池容量 vs 启动日志实测
    if real_kv_tokens:
        r = estimate(EstimateInput(model=QWEN36_AWQ, gpu=RTX3090, quant="int4",
                                   context_len=32768, concurrency=4, engine="vllm",
                                   tp=2, kv_quant="fp8", safety_factor=0.85,
                                   max_num_batched_tokens=8192))
        cases.append(("Qwen3.6-27B-AWQ 2x3090 KV池", r.max_kv_tokens,
                      float(real_kv_tokens), 10.0, "tokens"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-kv-tokens", type=int, default=None,
                    help="vLLM 启动日志 '# CUDA blocks' x 16 的实测值")
    args = ap.parse_args()

    cases = build_cases(args.real_kv_tokens)
    print(f"{'用例':38} {'预测':>12} {'参考':>12} {'误差':>8}  判定")
    print("-" * 78)
    failures = 0
    for tag, pred, ref, tol, unit in cases:
        err = abs(pred - ref) / ref * 100
        ok = err <= tol
        failures += 0 if ok else 1
        print(f"{tag:38} {pred:>12,.1f} {ref:>12,.1f} {err:>7.1f}%  {'✅' if ok else '❌'} ({unit}, 容差{tol}%)")
    print("-" * 78)
    print("全部达标 ✅" if failures == 0 else f"{failures} 项超容差 ❌")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
