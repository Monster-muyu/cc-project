from vram_calc.core.planner import Plan, LedgerRow
from vram_calc.core.commands import render_commands

def mkplan(**kw):
    d = dict(key="single", name="最少机器", badges=("单机即可",), tp=8, pp=1, ep=1, dp=1,
             verdict="ok", why="...", cross_node=False,
             rows=(LedgerRow("a", "server-A", "3090", 8, 6.95, 1.9, 0.5, 11.2, 9.35, 20.4, "ok"),),
             max_kv_tokens=900000, req_tokens=252000, concurrency_per_replica=3,
             warnings=(), hosts=(("server-A", "192.168.1.11"),))
    d.update(kw)
    return Plan(**d)


def test_single_machine_command():
    blocks = render_commands(mkplan(), "Qwen/Qwen3.8-27B", 252000, 3, 0.85, "fp16")
    assert len(blocks) == 1
    code = blocks[0].code
    assert "--tensor-parallel-size 8" in code
    assert "--max-model-len 252000" in code
    assert "--max-num-seqs 3" in code
    assert "--gpu-memory-utilization 0.85" in code
    assert "ray" not in code


def test_dp_merges_same_command():
    p = mkplan(key="dp", dp=2, hosts=(("server-A", "10.0.0.1"), ("server-B", "10.0.0.2")))
    blocks = render_commands(p, "m", 252000, 3, 0.85, "fp16")
    assert len(blocks) == 1 and "server-A" in blocks[0].code and "server-B" in blocks[0].code


def test_cross_node_needs_ray():
    p = mkplan(key="pp", tp=4, pp=2, cross_node=True,
               hosts=(("server-A", "10.0.0.1"), ("server-B", "")))
    blocks = render_commands(p, "m", 252000, 3, 0.85, "fp8")
    texts = [b.code for b in blocks]
    assert any("ray start --head" in t for t in texts)
    assert any("ray start --address=10.0.0.1:6379" in t for t in texts)
    serve = [t for t in texts if "vllm serve" in t][0]
    assert "--pipeline-parallel-size 2" in serve
    assert "--distributed-executor-backend ray" in serve
    assert "--kv-cache-dtype fp8" in serve          # 非 fp16 才出现
    assert "server-B-IP" in "".join(texts)          # host 空用占位符


def test_moe_appends_expert_parallel():
    p = mkplan(ep=8, tp=8)
    code = render_commands(p, "moe/m", 252000, 3, 0.85, "fp16")[0].code
    assert "--enable-expert-parallel" in code
