import pytest
from vram_calc.core.estimator import ModelSpec, GpuSpec
from vram_calc.core.planner import Machine, enumerate_candidates
from vram_calc.core.planner import PlanInput, plan_deployment

M8B = ModelSpec(id="t/8b", name="8B", params_b=8.0, layers=32, hidden_dim=4096,
                attn_heads=32, kv_heads=8, head_dim=128)
G24 = GpuSpec(id="g24", name="24G", vram_gb=24)
G12 = GpuSpec(id="g12", name="12G", vram_gb=12)

def mach(sid, gpu, n): return Machine(sid, sid, "", gpu, n)


def test_single_machine_candidates():
    cands = enumerate_candidates((mach("a", G24, 8),), M8B)
    singles = [c for c in cands if c.category == "single"]
    assert [(c.tp, c.pp) for c in singles] == [(8, 1)]


def test_count_not_head_divisor_falls_back():
    # 12 卡、32 头：TP 必须整除头数 → 回退到 8（弃 4 卡）
    cands = enumerate_candidates((mach("a", G24, 12),), M8B)
    assert any(c.category == "single" and c.tp == 8 for c in cands)


def test_dp_groups_same_type_only():
    cands = enumerate_candidates((mach("a", G24, 8), mach("b", G24, 8), mach("c", G12, 4)), M8B)
    dps = [c for c in cands if c.category == "dp"]
    assert len(dps) == 1 and set(m.server_id for m in dps[0].machines) == {"a", "b"}


def test_pp_cross_machine():
    cands = enumerate_candidates((mach("a", G12, 4), mach("b", G12, 4)), M8B)
    pps = [c for c in cands if c.category == "pp"]
    assert any(c.tp == 4 and c.pp == 2 for c in pps)


def test_tp_cross_only_same_type():
    cands = enumerate_candidates((mach("a", G12, 2), mach("b", G12, 2)), M8B)
    tcs = [c for c in cands if c.category == "tp_cross"]
    assert any(c.tp == 4 for c in tcs)          # 2+2 组成 TP4
    cands2 = enumerate_candidates((mach("a", G12, 2), mach("b", G24, 2)), M8B)
    assert not [c for c in cands2 if c.category == "tp_cross"]


def pin(**kw):  # 8B fp16=16GB 权重；默认目标 4k×1
    base = dict(model=M8B, machines=(mach("a", G24, 8), mach("b", G24, 8)),
                context_len=4096, concurrency=1)
    base.update(kw)
    return PlanInput(**base)


def test_top3_categories_and_order():
    plans = plan_deployment(pin())
    assert [p.key for p in plans][:2] == ["single", "dp"]     # 单机第一，DP 第二
    assert plans[0].verdict == "ok"
    assert plans[0].rows[0].gpus_used == 8
    assert plans[1].dp == 2 and plans[1].concurrency_per_replica == 1
    assert plans[0].why                            # why 非空
    assert not plans[0].cross_node and plans[2].cross_node is True   # 第三是跨机 PP


def test_dp_splits_concurrency():
    plans = plan_deployment(pin(concurrency=3))
    dp = [p for p in plans if p.key == "dp"][0]
    assert dp.concurrency_per_replica == 2          # ceil(3/2)


def test_small_cards_push_to_pp():
    plans = plan_deployment(pin(machines=(mach("a", G12, 1), mach("b", G12, 1))))
    # 8B fp16 单卡 12G 放不下(16GB权重) → 单机/DP over，首个方案是跨机 PP
    assert plans[0].key == "pp" and plans[0].tp == 1 and plans[0].pp == 2


def test_nothing_fits_returns_over_plan():
    plans = plan_deployment(pin(machines=(mach("a", G12, 1),)))
    assert plans and plans[0].verdict == "over"     # 全放不下也返回（带 over）


def test_fp8_kv_on_unsupported_gpu_warns():
    plans = plan_deployment(pin(kv_quant="fp8"))
    assert plans and plans[0].warnings              # G24 未声明 supports_fp8 → 警告
