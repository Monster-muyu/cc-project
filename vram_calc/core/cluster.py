# vram_calc/core/cluster.py
"""Server entity: one physical machine and its GPU slots.

gpus is a LIST of (gpu_id, count) so mixed-GPU machines are representable
from day one; the v1 planner skips mixed machines (vLLM can't mix types in
a TP group) but the data model won't need migration later.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuCount:
    gpu_id: str
    count: int


@dataclass(frozen=True)
class ServerSpec:
    id: str
    name: str
    host: str = ""
    gpus: tuple[GpuCount, ...] = ()


def server_is_mixed(s: ServerSpec) -> bool:
    return len({g.gpu_id for g in s.gpus}) > 1
