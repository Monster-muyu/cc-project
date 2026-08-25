# vram_calc/core/commands.py
"""Render launch commands for a plan. Pure string assembly, no state."""
from __future__ import annotations
from dataclasses import dataclass

from .planner import Plan


@dataclass(frozen=True)
class CommandBlock:
    title: str
    code: str


def _vllm_serve(p: Plan, model_id: str, ctx: int, util: float,
                kv_quant: str, ray: bool) -> str:
    a = f"vllm serve {model_id} \\\n  --tensor-parallel-size {p.tp}"
    if p.pp > 1:
        a += f" \\\n  --pipeline-parallel-size {p.pp}"
    if ray:
        a += " \\\n  --distributed-executor-backend ray"
    if p.ep > 1:
        a += " \\\n  --enable-expert-parallel"
    if kv_quant != "fp16":
        a += f" \\\n  --kv-cache-dtype {kv_quant}"
    a += (f" \\\n  --max-model-len {ctx} --max-num-seqs {p.concurrency_per_replica}"
          f" \\\n  --gpu-memory-utilization {util}"
          " \\\n  --host 0.0.0.0 --port 8000")
    return a


def _sglang_serve(p: Plan, model_id: str, ctx: int, util: float,
                  kv_quant: str, node: tuple[str, str] | None) -> str:
    """SGLang: 单机直接 launch_server；多机带 dist-init-addr/nnodes/node-rank（不走 Ray）."""
    a = f"python -m sglang.launch_server --model-path {model_id} \\\n  --tp {p.tp}"
    if p.dp > 1:
        a += f" \\\n  --dp {p.dp}"
    if p.ep > 1:
        a += f" \\\n  --ep-size {p.ep}"
    a += f" \\\n  --context-length {ctx} --max-running-requests {p.concurrency_per_replica}" \
         f" \\\n  --mem-fraction-static {util}"
    if kv_quant == "fp8":
        a += " \\\n  --kv-cache-dtype fp8_e5m2"
    if node is not None:
        name, host = node
        a += f" \\\n  --dist-init-addr {host or name + '-IP'}:5000"
    a += " \\\n  --host 0.0.0.0 --port 30000"
    return a


def _serve_args(p: Plan, model_id: str, ctx: int, util: float,
                kv_quant: str, ray: bool, engine: str = "vllm") -> str:
    if engine == "sglang":
        return _sglang_serve(p, model_id, ctx, util, kv_quant, None)
    return _vllm_serve(p, model_id, ctx, util, kv_quant, ray)


def render_single_commands(engine: str, model_id: str, ctx: int, concurrency: int,
                           util: float, kv_quant: str, tp: int = 1, pp: int = 1,
                           ep: int = 1) -> list[CommandBlock]:
    """单机页：按当前配置直接出一条启动命令（多机方案走 render_commands）。"""
    if engine in ("vllm", "sglang"):
        shim = Plan(key="single", name="单机", badges=(), tp=tp, pp=pp, ep=ep, dp=1,
                    verdict="ok", why="", cross_node=False, rows=(),
                    max_kv_tokens=0, req_tokens=0, concurrency_per_replica=concurrency)
        return [CommandBlock("单机启动", _serve_args(shim, model_id, ctx, util,
                                                    kv_quant, False, engine))]
    if engine == "llama_cpp":
        a = (f"# -m 换成你的 .gguf 文件路径\nllama-server -m {model_id} \\\n"
             f"  -ngl 99 -c {ctx} -np {concurrency} -fa")
        if kv_quant != "fp16":
            a += f" \\\n  --cache-type-k {kv_quant} --cache-type-v {kv_quant}"
        if pp > 1:                                    # 按层切分多卡
            a += f" \\\n  --split-mode layer --tensor-split \"{'1,' * (pp - 1)}1\""
        a += " \\\n  --host 0.0.0.0 --port 8080"
        return [CommandBlock("单机启动", a)]
    # ollama：环境变量控制并发/KV精度，ctx 走 Modelfile；多卡自动按层切分
    env = f"OLLAMA_NUM_PARALLEL={concurrency}"
    if kv_quant != "fp16":
        env += f" OLLAMA_KV_CACHE_TYPE={kv_quant}"
    code = (f"env {env} ollama serve\n"
            f"# 上下文：Modelfile 写 PARAMETER num_ctx {ctx} 后 ollama create <名> -f Modelfile")
    return [CommandBlock("单机启动", code)]


def render_commands(p: Plan, model_id: str, context_len: int, concurrency: int,
                    gpu_util: float, kv_quant: str, engine: str = "vllm") -> list[CommandBlock]:
    hosts = p.hosts or (("server-1", ""),)
    if not p.cross_node:
        if p.dp > 1:
            who = "、".join(h[0] for h in hosts)
            code = (f"# {who} 各起一个实例（命令相同）；负载均衡由前置网关做\n"
                    + _serve_args(p, model_id, context_len, gpu_util, kv_quant, False, engine))
            return [CommandBlock(f"DP ×{p.dp}：每台机器", code)]
        return [CommandBlock(hosts[0][0] + "（单机）",
                _serve_args(p, model_id, context_len, gpu_util, kv_quant, False, engine))]
    head, *workers = hosts
    head_addr = head[1] or f"{head[0]}-IP"
    if engine == "sglang":
        # SGLang 多机：每台机器各自 launch_server，靠 dist-init-addr/nnodes/node-rank 组网
        nn = len(hosts)
        blocks = []
        for i, (name, host) in enumerate(hosts):
            code = (f"# 网络要求：{p.net}\n" if p.net else "") + \
                _sglang_serve(p, model_id, context_len, gpu_util, kv_quant, (name, host)) + \
                f" \\\n  --nnodes {nn} --node-rank {i}"
            blocks.append(CommandBlock(f"{name}（node-rank {i}）", code))
        return blocks
    blocks = [CommandBlock(f"{head[0]}：Ray head", f"ray start --head --port=6379")]
    if workers:
        blocks.append(CommandBlock("其余机器：加入集群",
            "\n".join(f"# {n}（{h or n + '-IP'}）\nray start --address={head_addr}:6379"
                      for n, h in workers)))
    serve = ("# 前提：所有机器相同 vLLM/NCCL 版本、驱动一致\n"
             + (f"# 网络要求：{p.net}\n" if p.net else "")
             + _vllm_serve(p, model_id, context_len, gpu_util, kv_quant, True))
    blocks.append(CommandBlock(f"{head[0]}：启动（Ray 自动分配各 stage）", serve))
    return blocks
