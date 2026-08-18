import pytest
from vram_calc.core.cluster import ServerSpec, GpuCount, server_is_mixed
from vram_calc.repos import store


def test_server_is_mixed():
    s = ServerSpec(id="a", name="A", host="", gpus=((GpuCount("rtx-3090", 2), GpuCount("rtx-4090", 2))))
    assert server_is_mixed(s) is True
    t = ServerSpec(id="b", name="B", host="", gpus=((GpuCount("rtx-3090", 8),)))
    assert server_is_mixed(t) is False


def test_server_store_roundtrip(tmp_path):
    store.servers.user_dir = tmp_path / "servers"    # 现有模式（tests/test_data.py:85）
    s = ServerSpec(id="srv-a", name="server-A", host="192.168.1.11",
                   gpus=((GpuCount("rtx-3090", 8),)))
    store.save_server(s)
    got = store.get_server("srv-a")
    assert got == s
    assert any(x.id == "srv-a" for x in store.list_servers())
