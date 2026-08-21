from vram_calc.core.calibration import parse_vllm_log, observed_overhead

V1_SAMPLE = """INFO 08-20 12:00:00 [model_runner.py:1118] Loading model weights took 8.0963 GiB
INFO 08-20 12:00:01 [gpu_worker.py:59] # CUDA blocks: 947
INFO 08-20 12:00:01 [gpu_model_runner.py:1191] Maximum concurrency for 204800 tokens per request: 1.00x
INFO 08-20 12:00:01 [gpu_model_runner.py:1192] GPU KV cache size: 14,752.00 MiB"""

V0_SAMPLE = """INFO ... Loading model weights took 8.10 GiB
INFO ... # CUDA blocks: 947, # CPU blocks: 2048
INFO ... GPU KV cache size: 15152.00 MiB
INFO ... Maximum KV cache size can hold up to 151472 tokens"""


def test_parse_v1_log():
    r = parse_vllm_log(V1_SAMPLE)
    assert r["weights_gib"] == 8.0963
    assert r["kv_pool_gib_per_gpu"] == round(14752.00 / 1024, 2)
    assert r["max_ctx_tokens"] == 204800


def test_parse_v0_log():
    r = parse_vllm_log(V0_SAMPLE)
    assert r["weights_gib"] == 8.10
    assert r["kv_pool_tokens"] == 151472        # 'can hold up to' 行优先
    assert r["kv_pool_gib_per_gpu"] == round(15152.00 / 1024, 2)


def test_blocks_only_gives_tokens():
    r = parse_vllm_log("# CUDA blocks: 100")
    assert r["kv_pool_tokens"] == 100 * 16


def test_empty_log_all_none():
    r = parse_vllm_log("nothing useful here")
    assert r["weights_gib"] is None and r["kv_pool_tokens"] is None


def test_observed_overhead_math():
    parsed = {"weights_gib": 8.1, "kv_pool_gib_per_gpu": 14.4}
    # 3090 24G, util 0.9: 21.6 - 8.1 - 14.4 = -0.9 → 开销被低估/利用率为填的
    assert observed_overhead(parsed, 24, 0.9) == -0.9
    assert observed_overhead(parsed, 24, 0.95, activation_gb=0.3) == \
        round(0.95 * 24 - 8.1 - 14.4 - 0.3, 2)


def test_observed_overhead_needs_gib():
    assert observed_overhead({"weights_gib": 8.0}, 24, 0.9) is None

def test_decode_tps_roofline():
    from vram_calc.core.estimator import ModelSpec, GpuSpec, EstimateInput, estimate
    m = ModelSpec(id="t/8b", name="8B", params_b=8.0, layers=32, hidden_dim=4096,
                  attn_heads=32, kv_heads=8, head_dim=128)
    g = GpuSpec(id="g", name="g", vram_gb=24, memory_bw_gbps=1000)
    r = estimate(EstimateInput(model=m, gpu=g, quant="fp16", context_len=4096))
    assert abs(r.decode_tps - 0.65 * 1000 / 16.06) < 2      # 16GB 权重 fp16

    from vram_calc.core.planner import Machine, PlanInput, plan_deployment
    # 8G 卡放不下单机 → 逼出跨机方案（否则四类全 ok 时 tp_cross 被 [:3] 截断）
    g8 = GpuSpec(id="g8", name="8G", vram_gb=8, memory_bw_gbps=300)
    plans = plan_deployment(PlanInput(
        model=m, machines=(Machine("a", "A", "", g8, 2), Machine("b", "B", "", g8, 2)),
        context_len=4096))
    pp = [p for p in plans if p.key == "pp"][0]
    assert "25Gbps" in pp.net                                   # 跨机 PP 网络要求
    tc = [p for p in plans if p.key == "tp_cross"]
    assert tc and "100Gbps" in tc[0].net
    dp = [p for p in plans if p.key == "dp"]
    assert (not dp) or dp[0].decode_tps >= pp.decode_tps       # DP 若在则吞吐叠加


def test_calibrate_api_roundtrip(tmp_path):
    import vram_calc.repos.store as store
    from vram_calc.web.app import app
    from fastapi.testclient import TestClient
    store.USER_DIR = tmp_path
    cli = TestClient(app)
    log = ("INFO ... Loading model weights took 4.60 GiB\n"
           "INFO ... GPU KV cache size: 13,824.00 MiB\n"
           "INFO ... # CUDA blocks: 947")
    r = cli.post("/api/calibrate", json={"log_text": log, "gpu_id": "rtx-3090", "util": 0.92})
    assert r.status_code == 200 and r.json()["ok"]
    assert store.load_calibration()["vllm:rtx-3090"]["overhead_gb"] == \
        round(0.92 * 24 - 4.60 - round(13824.00 / 1024, 2), 2)
    # 数字对不上（开销为负）→ 400 拒绝
    bad = cli.post("/api/calibrate", json={"log_text": log, "gpu_id": "rtx-3090", "util": 0.7})
    assert bad.status_code == 400
    # 标定后 /api/calc 带标定标记
    r2 = cli.post("/api/calc", json={"model_id": "meta-llama/Meta-Llama-3-8B",
                                     "gpu_id": "rtx-3090", "quant": "fp16",
                                     "context_len": 4096})
    assert r2.json()["calibrated"] is True
