import json
import pytest
from fastapi.testclient import TestClient
from vram_calc.web.app import app
from vram_calc.assistant import orchestrator
from vram_calc.assistant.providers import TextDelta, Done

cli = TestClient(app)
BODY = {"config": {"protocol": "openai", "base_url": "", "api_key": "k", "model": "fake"},
        "messages": [{"role": "user", "content": "hi"}], "page_ctx": None}


def test_chat_sse(monkeypatch):
    class FP:
        def chat_stream(self, messages, tools):
            yield TextDelta("你好")
            yield Done("stop")
    monkeypatch.setattr(orchestrator, "get_provider", lambda cfg: FP())
    r = cli.post("/api/assistant/chat", json=BODY)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [l for l in r.text.split("\n\n") if l.startswith("data: ")]
    payloads = [json.loads(l[6:]) for l in lines[:-1]]      # 最后一个是 [DONE]
    assert lines[-1] == "data: [DONE]"
    assert {"t": "delta", "v": "你好"} in payloads
    assert payloads[-1]["t"] == "done"


def test_chat_error_event(monkeypatch):
    def boom(cfg):
        raise RuntimeError("x")
    monkeypatch.setattr(orchestrator, "get_provider", boom)
    r = cli.post("/api/assistant/chat", json=BODY)
    payloads = [json.loads(l[6:]) for l in r.text.split("\n\n") if l.startswith("data: ")]
    assert payloads[0]["t"] == "error"


def test_test_route_ok(monkeypatch):
    import vram_calc.web.app as webapp
    class FP:
        def test_connection(self):
            return "deepseek-chat"
    monkeypatch.setattr(webapp, "get_provider", lambda cfg: FP())
    r = cli.post("/api/assistant/test", json={"config": BODY["config"]})
    assert r.json() == {"ok": True, "model_name": "deepseek-chat"}


def test_test_route_error(monkeypatch):
    import vram_calc.web.app as webapp
    def boom(cfg):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(webapp, "get_provider", boom)
    r = cli.post("/api/assistant/test", json={"config": BODY["config"]})
    assert r.json()["ok"] is False and "连接" in r.json()["error"]


def test_chat_bad_config_422():
    r = cli.post("/api/assistant/chat", json={"config": {"protocol": 123}, "messages": "x"})
    assert r.status_code == 422
