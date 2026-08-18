import pytest
from vram_calc.core.estimator import ModelSpec, GpuSpec
from vram_calc.core.planner import Machine, enumerate_candidates

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
