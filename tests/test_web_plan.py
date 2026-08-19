from fastapi.testclient import TestClient

from vram_calc.web.app import app
from vram_calc.repos import store

cli = TestClient(app)


def _setup(tmp_path):
    store.servers.user_dir = tmp_path / "servers"
    store.save_server(store.ServerSpec(id="srv-a", name="A", host="10.0.0.1",
                      gpus=(store.GpuCount("rtx-3090", 8),)))


def test_server_crud(tmp_path):
    _setup(tmp_path)
    assert any(s["id"] == "srv-a" for s in cli.get("/api/servers").json())
    r = cli.post("/api/servers", json={"id": "srv-b", "name": "B", "host": "",
                                       "gpus": [{"gpu_id": "rtx-4090", "count": 4},
                                                {"gpu_id": "rtx-3090", "count": 2}]})
    assert r.status_code == 200
    got = [s for s in cli.get("/api/servers").json() if s["id"] == "srv-b"][0]
    assert got["mixed"] is True                       # 混插标记
    assert cli.delete("/api/servers/srv-b").status_code == 200
    assert not [s for s in cli.get("/api/servers").json() if s["id"] == "srv-b"]


def test_plan_endpoint(tmp_path):
    _setup(tmp_path)
    r = cli.post("/api/plan", json={
        "model_id": "meta-llama/Meta-Llama-3-8B", "server_ids": ["srv-a"],
        "context_len": 4096, "concurrency": 1, "quant": "fp16",
        "kv_quant": "fp16", "gpu_util": 0.9})
    assert r.status_code == 200
    body = r.json()
    assert body["plans"] and body["plans"][0]["commands"]
    assert body["plans"][0]["rows"][0]["server_id"] == "srv-a"
    assert any("vllm serve" in c["code"] for c in body["plans"][0]["commands"])


def test_plan_mixed_server_warns(tmp_path):
    _setup(tmp_path)
    cli.post("/api/servers", json={"id": "mix", "name": "M", "host": "",
                                   "gpus": [{"gpu_id": "rtx-3090", "count": 2},
                                            {"gpu_id": "rtx-4090", "count": 2}]})
    r = cli.post("/api/plan", json={
        "model_id": "meta-llama/Meta-Llama-3-8B", "server_ids": ["mix"],
        "context_len": 4096, "concurrency": 1})
    assert r.status_code == 200
    assert r.json()["warnings"] and r.json()["plans"] == []


def test_plan_unknown_ids_400(tmp_path):
    _setup(tmp_path)
    assert cli.post("/api/plan", json={"model_id": "no/such", "server_ids": ["srv-a"],
                                       "context_len": 4096}).status_code == 400
    assert cli.post("/api/plan", json={"model_id": "meta-llama/Meta-Llama-3-8B",
                                       "server_ids": ["ghost"], "context_len": 4096}).status_code == 400
