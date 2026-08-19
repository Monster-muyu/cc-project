import json
from vram_calc.assistant import tools, prompts
from vram_calc.repos import store

LLAMA = "meta-llama/Meta-Llama-3-8B"


def test_calc_vram_returns_estimate_json():
    r = json.loads(tools.execute_tool("calc_vram", {
        "model_id": LLAMA, "gpu_id": "rtx-4090", "gpu_count": 1, "tp": 1,
        "context_len": 4096, "concurrency": 1, "quant": "fp16"}))
    assert r["verdict"] == "ok"
    assert r["kv_pool_tokens"] > 10000
    assert "per_gpu_total_gb" in r and "max_concurrency_at_ctx" in r


def test_calc_vram_unknown_model_error_text():
    r = tools.execute_tool("calc_vram", {"model_id": "no/such", "gpu_id": "rtx-4090",
                                         "gpu_count": 1, "tp": 1, "context_len": 4096})
    assert r.startswith("错误：") and "no/such" in r


def test_calc_vram_gpu_count_mismatch():
    r = tools.execute_tool("calc_vram", {"model_id": LLAMA, "gpu_id": "rtx-4090",
                                         "gpu_count": 4, "tp": 2, "pp": 1,
                                         "context_len": 4096})
    assert "不一致" in r


def test_unknown_tool():
    assert "未知工具" in tools.execute_tool("nope", {})


def test_tools_for_page_kind():
    assert [t["name"] for t in tools.tools_for(None)] == ["calc_vram"]
    assert [t["name"] for t in tools.tools_for({"kind": "plan"})] == ["calc_vram", "plan_multi_node"]


def test_plan_multi_node(tmp_path):
    store.servers.user_dir = tmp_path / "servers"
    store.save_server(store.ServerSpec(
        id="srv-a", name="A", host="", gpus=(store.GpuCount("rtx-3090", 8),)))
    r = json.loads(tools.execute_tool("plan_multi_node", {
        "model_id": LLAMA, "server_ids": ["srv-a"], "context_len": 4096}))
    assert r["plans"] and r["plans"][0]["name"] == "最少机器"


def test_build_system_prompt_injects():
    sp = prompts.build_system_prompt({"kind": "calc",
                                      "input": {"model_id": LLAMA, "tp": 2},
                                      "last_result": {"verdict": "ok"}})
    assert "推理过程" in sp and "禁止心算" in sp     # 契约与硬规则
    assert "--gpu-memory-utilization" in sp          # 手册摘要已注入
    assert '"tp": 2' in sp                           # 当前配置已注入


def test_build_system_prompt_null_ctx():
    sp = prompts.build_system_prompt(None)
    assert "手册页" in sp
